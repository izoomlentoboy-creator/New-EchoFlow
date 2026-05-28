"""Physiologically-grounded synthetic voice generator.

No public clinical corpus is available in this environment, so we generate
sustained-vowel recordings with a **source-filter model** whose parameters are
drawn from clinically-motivated distributions for each pathology class.  This
is a well-established technique (parametric voice synthesis is routinely used
to augment dysphonia datasets) and lets the whole pipeline be trained and
validated end-to-end *today*.  When you obtain a real corpus (see
``data/download.py``) the very same training code consumes it instead.

Model
-----
* Glottal source: a pulse train at the F0 contour, each pulse shaped by a
  Rosenberg-like glottal flow derivative, modulated by:
    - jitter   : cycle-to-cycle period perturbation (frequency instability)
    - shimmer  : cycle-to-cycle amplitude perturbation
    - tremor   : slow (3-8 Hz) F0/amplitude modulation
    - subharmonics / diplophonia : alternating-cycle amplitude (period
      doubling) seen in severe roughness
* Aspiration noise: additive band-shaped noise = breathiness.
* Vocal-tract filter: a cascade of formant resonators for a sustained /a/.

Each class perturbs these knobs in a characteristic way (see ``CLASS_PARAMS``).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..config import CLASSES


@dataclass
class VoiceParams:
    f0: float                 # base fundamental frequency (Hz)
    jitter: float             # fractional period perturbation (0..)
    shimmer: float            # fractional amplitude perturbation
    hnr_noise: float          # aspiration noise level (0=clean .. higher=breathy)
    tremor_rate: float        # Hz of low-frequency modulation
    tremor_depth: float       # fractional modulation depth
    subharmonic: float        # 0..1 strength of period-doubling
    f0_drift: float           # slow random walk of F0 (instability)
    voice_breaks: float       # probability mass of silent (aphonic) gaps
    formants: tuple           # (f1, f2, f3, f4) Hz for the vowel


# Clinically-motivated parameter ranges (mean, std) per class.  The signatures
# are deliberately contrasted along the axes clinicians use to tell disorders
# apart: roughness (jitter/shimmer/subharmonics) vs breathiness (aspiration
# noise) vs instability (tremor/breaks) vs pitch (F0).
CLASS_PARAMS = {
    "healthy": dict(           # clean, periodic, low perturbation
        f0=(140, 40), jitter=(0.004, 0.0015), shimmer=(0.025, 0.01),
        hnr_noise=(0.02, 0.01), tremor_rate=(0, 0), tremor_depth=(0.005, 0.003),
        subharmonic=(0.0, 0.0), f0_drift=(0.002, 0.001), voice_breaks=(0.0, 0.0),
    ),
    "hyperfunctional": dict(   # nodules / polyps -> ROUGH: high jitter/shimmer
        f0=(160, 45), jitter=(0.022, 0.006), shimmer=(0.11, 0.03),  # & strong
        hnr_noise=(0.05, 0.02), tremor_rate=(0, 0), tremor_depth=(0.01, 0.005),  # subharmonics,
        subharmonic=(0.5, 0.12), f0_drift=(0.005, 0.002), voice_breaks=(0.0, 0.0),  # little breath
    ),
    "paralysis": dict(         # glottic insufficiency -> BREATHY: noise-dominant,
        f0=(135, 45), jitter=(0.010, 0.004), shimmer=(0.055, 0.02),  # little roughness
        hnr_noise=(0.45, 0.10), tremor_rate=(0, 0), tremor_depth=(0.01, 0.005),
        subharmonic=(0.05, 0.04), f0_drift=(0.006, 0.003), voice_breaks=(0.02, 0.015),
    ),
    "neurological": dict(      # spasmodic / parkinsonian -> TREMOR + voice breaks
        f0=(150, 45), jitter=(0.012, 0.005), shimmer=(0.06, 0.02),
        hnr_noise=(0.12, 0.05), tremor_rate=(5.0, 1.2), tremor_depth=(0.09, 0.02),
        subharmonic=(0.12, 0.08), f0_drift=(0.018, 0.006), voice_breaks=(0.12, 0.04),
    ),
    "inflammatory": dict(      # laryngitis / edema -> LOW PITCH, moderate mixed
        f0=(100, 25), jitter=(0.009, 0.004), shimmer=(0.05, 0.02),
        hnr_noise=(0.16, 0.05), tremor_rate=(0, 0), tremor_depth=(0.008, 0.004),
        subharmonic=(0.18, 0.08), f0_drift=(0.006, 0.003), voice_breaks=(0.01, 0.01),
    ),
}

# A few vowel formant templates so the model does not key on a single timbre.
VOWEL_FORMANTS = [
    (730, 1090, 2440, 3400),   # /a/
    (530, 1840, 2480, 3500),   # /e/
    (270, 2290, 3010, 3700),   # /i/
    (570, 840, 2410, 3300),    # /o/
]


def _sample(rng, mean_std):
    m, s = mean_std
    if s == 0:
        return m
    return float(rng.normal(m, s))


class SyntheticVoiceGenerator:
    def __init__(self, sample_rate: int, seed: int = 0):
        self.sr = sample_rate
        self.rng = np.random.default_rng(seed)

    # -- parameter sampling -------------------------------------------------
    def sample_params(self, cls: str) -> VoiceParams:
        p = CLASS_PARAMS[cls]
        rng = self.rng
        formants = VOWEL_FORMANTS[rng.integers(len(VOWEL_FORMANTS))]
        # small inter-speaker formant variability
        formants = tuple(f * rng.uniform(0.92, 1.08) for f in formants)
        return VoiceParams(
            f0=max(70, _sample(rng, p["f0"])),
            jitter=max(0.0, _sample(rng, p["jitter"])),
            shimmer=max(0.0, _sample(rng, p["shimmer"])),
            hnr_noise=max(0.0, _sample(rng, p["hnr_noise"])),
            tremor_rate=max(0.0, _sample(rng, p["tremor_rate"])),
            tremor_depth=max(0.0, _sample(rng, p["tremor_depth"])),
            subharmonic=float(np.clip(_sample(rng, p["subharmonic"]), 0, 1)),
            f0_drift=max(0.0, _sample(rng, p["f0_drift"])),
            voice_breaks=float(np.clip(_sample(rng, p["voice_breaks"]), 0, 0.5)),
            formants=formants,
        )

    # -- waveform synthesis -------------------------------------------------
    def render(self, vp: VoiceParams, seconds: float) -> np.ndarray:
        sr = self.sr
        n = int(seconds * sr)
        rng = self.rng

        # ---- F0 contour: base + slow random-walk drift + optional tremor ----
        t = np.arange(n) / sr
        drift = np.cumsum(rng.standard_normal(n)) * vp.f0_drift
        drift = drift - drift.mean()
        f0_contour = vp.f0 * (1.0 + drift)
        if vp.tremor_rate > 0:
            trem = vp.tremor_depth * np.sin(2 * np.pi * vp.tremor_rate * t
                                            + rng.uniform(0, 2 * np.pi))
            f0_contour = f0_contour * (1.0 + trem)
        f0_contour = np.clip(f0_contour, 60, 500)

        # ---- glottal pulse train with jitter / shimmer / subharmonics ----
        source = np.zeros(n, dtype=np.float64)
        pos = 0.0
        cycle = 0
        while pos < n:
            f0_here = f0_contour[min(int(pos), n - 1)]
            period = sr / f0_here
            # jitter: perturb period
            period *= (1.0 + vp.jitter * rng.standard_normal())
            period = max(period, sr / 500)
            # shimmer: perturb amplitude
            amp = 1.0 + vp.shimmer * rng.standard_normal()
            # subharmonic: alternate-cycle amplitude reduction (period doubling)
            if vp.subharmonic > 0 and (cycle % 2 == 1):
                amp *= (1.0 - vp.subharmonic)
            idx = int(pos)
            if idx < n:
                self._add_glottal_pulse(source, idx, period, amp)
            pos += period
            cycle += 1

        # ---- aspiration / turbulent noise (breathiness) ----
        if vp.hnr_noise > 0:
            noise = rng.standard_normal(n)
            noise = self._highpass(noise, 1500, sr)   # turbulent noise is HF
            source = source + vp.hnr_noise * noise * np.std(source)

        # ---- vocal-tract filter: cascade of formant resonators ----
        voice = source
        for f, bw in zip(vp.formants, (90, 110, 140, 200)):
            voice = self._formant(voice, f, bw, sr)

        # ---- voice breaks (intermittent aphonia) ----
        if vp.voice_breaks > 0:
            voice = self._apply_breaks(voice, vp.voice_breaks, sr)

        voice = voice / (np.max(np.abs(voice)) + 1e-9)
        return voice.astype(np.float32)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _add_glottal_pulse(buf, idx, period, amp):
        # Rosenberg-like glottal flow derivative over one open phase
        L = int(period)
        if L < 4 or idx + L > len(buf):
            L = min(L, len(buf) - idx)
        if L < 4:
            return
        oq = 0.6                              # open quotient
        op = int(L * oq)
        x = np.linspace(0, np.pi, max(op, 2))
        pulse = np.sin(x) ** 2
        pulse = np.gradient(pulse)           # flow derivative -> richer spectrum
        buf[idx:idx + len(pulse)] += amp * pulse

    @staticmethod
    def _formant(x, freq, bw, sr):
        # second-order resonator (Klatt-style), applied with lfilter for speed
        from scipy.signal import lfilter
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * freq / sr
        a1 = -2 * r * np.cos(theta)
        a2 = r * r
        b0 = 1 - a1 - a2                     # ~unity gain at the resonance
        return lfilter([b0], [1.0, a1, a2], x)

    @staticmethod
    def _highpass(x, cutoff, sr):
        from scipy.signal import butter, lfilter
        b, a = butter(2, cutoff / (sr / 2), btype="high")
        return lfilter(b, a, x)

    def _apply_breaks(self, x, prob, sr):
        # randomly silence short windows to emulate voice arrests
        out = x.copy()
        win = int(0.08 * sr)
        i = 0
        while i < len(out):
            if self.rng.random() < prob:
                out[i:i + win] *= self.rng.uniform(0.0, 0.2)
            i += win
        return out


def severity_of(vp: VoiceParams) -> float:
    """Ground-truth dysphonia severity in [0, 1] derived from the source
    parameters - an AVQI-like aggregate used to supervise the severity head.
    Healthy voices score near 0; severe dysphonia approaches 1.
    """
    s = (0.25 * (vp.jitter / 0.02)
         + 0.25 * (vp.shimmer / 0.10)
         + 0.25 * (vp.hnr_noise / 0.30)
         + 0.15 * (vp.voice_breaks / 0.08)
         + 0.10 * vp.subharmonic
         + 0.10 * (vp.tremor_depth / 0.06))
    return float(np.clip(s, 0.0, 1.0))


def synthesize(cls: str, sample_rate: int, seconds: float,
               seed: int = 0) -> np.ndarray:
    """Convenience one-shot: render one waveform for ``cls``."""
    g = SyntheticVoiceGenerator(sample_rate, seed=seed)
    return g.render(g.sample_params(cls), seconds)
