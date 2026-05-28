"""Lightweight waveform augmentation used during training only.

Augmentation makes the model robust to the acoustic conditions of consumer
phones (room noise, mic colouration, level changes) without altering the
clinical perturbation cues that distinguish the pathology classes.
"""
from __future__ import annotations

import numpy as np


class Augmenter:
    def __init__(self, sr: int, seed: int = 0):
        self.sr = sr
        self.rng = np.random.default_rng(seed)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        x = wav.copy()
        # additive Gaussian / pink-ish noise (room + mic self-noise)
        if self.rng.random() < 0.7:
            snr_db = self.rng.uniform(15, 40)
            x = self._add_noise(x, snr_db)
        # random gain (recording level)
        if self.rng.random() < 0.5:
            x = x * self.rng.uniform(0.6, 1.4)
        # mild time shift
        if self.rng.random() < 0.5:
            shift = int(self.rng.uniform(-0.05, 0.05) * self.sr)
            x = np.roll(x, shift)
        # simple first-order mic colouration
        if self.rng.random() < 0.3:
            a = self.rng.uniform(-0.3, 0.3)
            x = np.append(x[0], x[1:] - a * x[:-1])
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    def _add_noise(self, x: np.ndarray, snr_db: float) -> np.ndarray:
        sig_p = np.mean(x ** 2) + 1e-10
        noise = self.rng.standard_normal(len(x)).astype(np.float32)
        noise_p = np.mean(noise ** 2) + 1e-10
        target_noise_p = sig_p / (10 ** (snr_db / 10))
        noise *= np.sqrt(target_noise_p / noise_p)
        return x + noise


class ChannelSimulator:
    """Domain randomization of the *recording channel* (sim->real bridge).

    Synthetic clean voice is passed through a randomly-sampled acquisition
    chain - room reverberation, microphone colouration, additive environmental
    noise (white / mains hum), band-limiting (telephone/codec) and occasional
    soft clipping.  Applying this to every synthetic sample forces the model to
    learn pathology cues that are *invariant* to how a phone records them,
    which is the key to closing the synthetic->real gap.

    Effects are deliberately moderate so the clinical perturbation cues
    (jitter/shimmer/CPP) survive, exactly as they do on real phone recordings.
    """
    def __init__(self, sr: int, seed: int = 0):
        self.sr = sr
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        r = self.rng
        y = x.astype(np.float32).copy()
        # 1) room reverberation (short exponential-decay impulse response)
        if r.random() < 0.6:
            y = self._reverb(y)
        # 2) microphone colouration (random low-order IIR tilt)
        if r.random() < 0.6:
            y = self._mic_color(y)
        # 3) band-limiting (telephone / lossy codec)
        if r.random() < 0.35:
            y = self._bandlimit(y)
        # 4) additive environmental noise + occasional mains hum
        if r.random() < 0.8:
            y = self._add_noise(y, r.uniform(12, 38))
        if r.random() < 0.2:
            y = self._mains_hum(y)
        # 5) level + occasional soft clipping
        y = y * r.uniform(0.5, 1.3)
        if r.random() < 0.15:
            thr = r.uniform(0.6, 0.95)
            y = np.tanh(y / thr) * thr
        peak = np.max(np.abs(y)) + 1e-9
        if peak > 1.0:
            y = y / peak
        return y.astype(np.float32)

    def _reverb(self, x):
        dur = self.rng.uniform(0.03, 0.18)
        n = int(dur * self.sr)
        t = np.arange(n)
        decay = np.exp(-t / (self.rng.uniform(0.01, 0.06) * self.sr))
        ir = (self.rng.standard_normal(n) * decay).astype(np.float32)
        ir[0] = 1.0
        from scipy.signal import fftconvolve
        y = fftconvolve(x, ir)[: len(x)]
        return y / (np.max(np.abs(y)) + 1e-9)

    def _mic_color(self, x):
        from scipy.signal import lfilter
        a = self.rng.uniform(-0.4, 0.4)
        return lfilter([1.0, a], [1.0], x).astype(np.float32)

    def _bandlimit(self, x):
        from scipy.signal import butter, lfilter
        lo = self.rng.uniform(150, 400) / (self.sr / 2)
        hi = self.rng.uniform(3000, 3800) / (self.sr / 2)
        b, a = butter(4, [lo, min(hi, 0.99)], btype="band")
        return lfilter(b, a, x).astype(np.float32)

    def _add_noise(self, x, snr_db):
        sig_p = np.mean(x ** 2) + 1e-10
        noise = self.rng.standard_normal(len(x)).astype(np.float32)
        noise *= np.sqrt((sig_p / (10 ** (snr_db / 10))) /
                         (np.mean(noise ** 2) + 1e-10))
        return x + noise

    def _mains_hum(self, x):
        f = self.rng.choice([50.0, 60.0])
        t = np.arange(len(x)) / self.sr
        amp = self.rng.uniform(0.005, 0.03) * (np.max(np.abs(x)) + 1e-6)
        hum = amp * np.sin(2 * np.pi * f * t)
        hum += 0.5 * amp * np.sin(2 * np.pi * 2 * f * t)
        return (x + hum).astype(np.float32)
