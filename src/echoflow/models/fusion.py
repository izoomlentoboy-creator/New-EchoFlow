"""EchoFlowNet - hierarchical, multi-task hybrid network.

Why this structure (the "new structure" the task asks for)
----------------------------------------------------------
Voice screening is a two-question problem: *is the voice pathological?* and
*which pathology?*  EchoFlowNet mirrors that clinical reasoning with a shared
encoder feeding three heads:

  * Head A  - healthy vs pathological           (binary screening, tuned for
                                                  high sensitivity)
  * Head B  - pathology subtype (4-way)         (only meaningful when path.)
  * Head C  - dysphonia severity (regression)   (AVQI-like 0..1 index)

The shared encoder fuses two complementary views with **gated fusion**:
  * a SE-CNN over the log-mel spectrogram with **attention pooling** over time
    (focuses on rough / broken segments rather than averaging them away);
  * an MLP over interpretable clinical biomarkers (jitter/shimmer/CPP/GNE/...).

Each branch emits a sigmoid gate that re-weights the other, so an unreliable
branch is suppressed per-sample.  The three losses are balanced automatically
with Kendall's homoscedastic uncertainty weighting (learnable log-variances).

The final 5-class distribution is assembled hierarchically:
    P(healthy)        = P(bin = healthy)
    P(pathology = k)  = P(bin = pathological) * P(subtype = k)
which keeps the screening and subtyping decisions coherent.

All ops convert cleanly to Core ML; biomarker standardization is baked into
buffers so the deployed model consumes raw features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# subtype head covers the 4 non-healthy classes (indices 1..4 of CLASSES)
N_SUBTYPES = 4


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention for the CNN branch."""
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Linear(ch, max(4, ch // r))
        self.fc2 = nn.Linear(max(4, ch // r), ch)

    def forward(self, x):                              # (B, C, H, W)
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s[:, :, None, None]


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, se=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.se = SEBlock(cout) if se else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.se(x)
        return self.pool(x)


class CNNBranch(nn.Module):
    """SE-CNN over the log-mel spectrogram with temporal attention pooling."""
    def __init__(self, emb_dim=128):
        super().__init__()
        self.b1 = ConvBlock(1, 16)
        self.b2 = ConvBlock(16, 32)
        self.b3 = ConvBlock(32, 64)
        self.b4 = ConvBlock(64, 96)
        self.att = nn.Linear(96, 1)            # attention score per time step
        self.proj = nn.Linear(96, emb_dim)

    def forward(self, x):                      # (B,1,M,T)
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        x = x.mean(dim=2)                      # collapse frequency -> (B,C,T')
        x = x.transpose(1, 2)                  # (B, T', C)
        w = torch.softmax(self.att(x), dim=1)  # (B, T', 1) attention weights
        x = (x * w).sum(dim=1)                 # weighted temporal pool -> (B,C)
        return F.relu(self.proj(x))


class BioBranch(nn.Module):
    """MLP over standardized clinical biomarkers -> embedding."""
    def __init__(self, n_in, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, emb_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    """Each branch emits a sigmoid gate that re-weights the other branch."""
    def __init__(self, dim):
        super().__init__()
        self.gate_a = nn.Linear(dim, dim)
        self.gate_b = nn.Linear(dim, dim)

    def forward(self, a, b):
        ga = torch.sigmoid(self.gate_a(b))     # bio gates cnn
        gb = torch.sigmoid(self.gate_b(a))     # cnn gates bio
        return torch.cat([a * ga, b * gb], dim=1)


class EchoFlowNet(nn.Module):
    def __init__(self, n_acoustic: int, n_classes: int = 5, emb_dim: int = 128):
        super().__init__()
        self.n_classes = n_classes
        self.cnn = CNNBranch(emb_dim)
        self.bio = BioBranch(n_acoustic, emb_dim)
        self.fusion = GatedFusion(emb_dim)
        self.trunk = nn.Sequential(
            nn.Linear(2 * emb_dim, 128), nn.ReLU(), nn.Dropout(0.4),
        )
        self.head_bin = nn.Linear(128, 2)            # healthy vs pathological
        self.head_sub = nn.Linear(128, N_SUBTYPES)   # which pathology
        self.head_sev = nn.Linear(128, 1)            # severity (pre-sigmoid)

        # Kendall homoscedastic uncertainty weights (learnable log-variances)
        self.log_var = nn.Parameter(torch.zeros(3))

        # baked-in biomarker standardization (set after fitting on train set)
        self.register_buffer("bio_mean", torch.zeros(n_acoustic))
        self.register_buffer("bio_std", torch.ones(n_acoustic))

    def set_bio_stats(self, mean, std):
        with torch.no_grad():
            self.bio_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
            std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)
            self.bio_std.copy_(std)

    def embed(self, mel, bio):
        bio = (bio - self.bio_mean) / self.bio_std
        ca = self.cnn(mel)
        ba = self.bio(bio)
        return self.trunk(self.fusion(ca, ba))

    def forward(self, mel, bio):
        """Returns raw head outputs: (logits_bin, logits_sub, severity)."""
        z = self.embed(mel, bio)
        return self.head_bin(z), self.head_sub(z), torch.sigmoid(self.head_sev(z))

    def class_probs(self, mel, bio):
        """Assemble the coherent 5-class probability vector + severity.

        P(healthy)       = softmax(bin)[healthy]
        P(pathology = k) = softmax(bin)[path] * softmax(sub)[k]
        """
        logb, logs, sev = self.forward(mel, bio)
        pb = torch.softmax(logb, dim=1)          # (B,2): [healthy, path]
        ps = torch.softmax(logs, dim=1)          # (B,4)
        healthy = pb[:, :1]                      # (B,1)
        path = pb[:, 1:2] * ps                   # (B,4)
        probs = torch.cat([healthy, path], dim=1)  # (B,5)
        return probs, sev
