"""In-graph log-mel front-end (raw waveform -> log-mel spectrogram).

This is the cornerstone of the "fully independent" design.  The STFT is
implemented as a fixed-weight ``Conv1d`` (a real/imag DFT basis windowed by a
Hann window), followed by a fixed mel filterbank matmul, a log, and
per-utterance mean/variance normalization.

Crucially the SAME module is used to (a) precompute spectrograms for training
and (b) sit at the input of the exported Core ML graph.  Because training and
inference share the *identical* operation, there is zero feature drift — and
the CNN branch needs no librosa (or any library) on device.  Every op used
(conv1d, elementwise, matmul, mean/std, log) converts cleanly to Core ML.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..config import AudioConfig


class MelFrontend(nn.Module):
    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.cfg = cfg
        n_fft = cfg.n_fft
        self.n_fft = n_fft
        self.hop = cfg.hop_length
        self.pad = n_fft // 2
        self.preemph = cfg.preemphasis

        # --- windowed DFT basis as conv weights: (n_bins, 1, n_fft) ---
        n_bins = n_fft // 2 + 1
        window = np.hanning(cfg.win_length)
        # center the analysis window inside the n_fft kernel
        wpad = np.zeros(n_fft, dtype=np.float64)
        off = (n_fft - cfg.win_length) // 2
        wpad[off:off + cfg.win_length] = window
        k = np.arange(n_fft)
        freqs = np.arange(n_bins)
        basis = 2 * np.pi * np.outer(freqs, k) / n_fft     # (n_bins, n_fft)
        real = (np.cos(basis) * wpad)[:, None, :]
        imag = (-np.sin(basis) * wpad)[:, None, :]
        self.register_buffer("dft_real",
                             torch.tensor(real, dtype=torch.float32))
        self.register_buffer("dft_imag",
                             torch.tensor(imag, dtype=torch.float32))

        # --- mel filterbank (built once with librosa, used as constant) ---
        import librosa
        melfb = librosa.filters.mel(sr=cfg.sample_rate, n_fft=n_fft,
                                    n_mels=cfg.n_mels, fmin=cfg.fmin,
                                    fmax=cfg.fmax).astype(np.float32)
        self.register_buffer("melfb", torch.tensor(melfb))   # (n_mels, n_bins)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) float32 -> (B, 1, n_mels, n_frames) normalized log-mel."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        # pre-emphasis (boosts higher formants where pathology cues live)
        if self.preemph:
            first = wav[:, :1]
            rest = wav[:, 1:] - self.preemph * wav[:, :-1]
            wav = torch.cat([first, rest], dim=1)
        x = wav.unsqueeze(1)                                 # (B,1,T)
        x = nn.functional.pad(x, (self.pad, self.pad), mode="reflect")
        real = nn.functional.conv1d(x, self.dft_real, stride=self.hop)
        imag = nn.functional.conv1d(x, self.dft_imag, stride=self.hop)
        power = real ** 2 + imag ** 2                        # (B, n_bins, F)
        mel = torch.matmul(self.melfb, power)                # (B, n_mels, F)
        logmel = torch.log(mel + 1e-6)
        # per-utterance CMVN over (mel, time)
        mean = logmel.mean(dim=(1, 2), keepdim=True)
        std = logmel.std(dim=(1, 2), keepdim=True) + 1e-6
        logmel = (logmel - mean) / std
        return logmel.unsqueeze(1)                           # (B,1,n_mels,F)

    @torch.no_grad()
    def numpy_mel(self, wav: np.ndarray) -> np.ndarray:
        """Convenience: single waveform -> (n_mels, n_frames) numpy."""
        t = torch.from_numpy(np.asarray(wav, dtype=np.float32))
        return self.forward(t)[0, 0].cpu().numpy()           # (n_mels, F)
