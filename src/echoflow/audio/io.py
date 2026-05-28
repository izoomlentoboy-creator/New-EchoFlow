"""Audio loading and resampling utilities."""
from __future__ import annotations

import numpy as np


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load an audio file as a mono float32 waveform at ``target_sr``.

    Uses soundfile when possible and falls back to librosa for formats that
    soundfile cannot decode (e.g. mp3/m4a).  Returns a 1-D float32 array in
    the range [-1, 1].
    """
    try:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:                       # stereo -> mono
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = _resample(wav, sr, target_sr)
        return wav.astype(np.float32)
    except Exception:
        import librosa
        wav, _ = librosa.load(path, sr=target_sr, mono=True)
        return wav.astype(np.float32)


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        return resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    except Exception:
        import librosa
        return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
