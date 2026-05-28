"""Waveform preprocessing: silence trimming, normalization, fixed length.

These operations MUST be mirrored exactly on the iOS side (see
ios/EchoFlow/AudioRecorder.swift) so that inference matches training.
"""
from __future__ import annotations

import numpy as np


def rms_normalize(wav: np.ndarray, target_dbfs: float = -20.0,
                  eps: float = 1e-8) -> np.ndarray:
    """Normalize loudness to a target RMS level in dBFS.

    Loudness normalization is essential for pathology detection because the
    clinically relevant cues are *spectral / perturbation* features, not the
    absolute recording gain (which varies wildly between phones and rooms).
    """
    rms = np.sqrt(np.mean(wav ** 2) + eps)
    target_rms = 10 ** (target_dbfs / 20.0)
    gain = target_rms / (rms + eps)
    out = wav * gain
    peak = np.max(np.abs(out))
    if peak > 1.0:                      # prevent clipping
        out = out / peak
    return out.astype(np.float32)


def trim_silence(wav: np.ndarray, sr: int, top_db: float = 30.0,
                 frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    """Energy-based voice activity detection that keeps only voiced regions.

    A simple, dependency-light VAD that is trivial to re-implement in Swift.
    """
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(wav) < frame:
        return wav
    # frame energy in dB relative to the loudest frame
    n_frames = 1 + (len(wav) - frame) // hop
    energies = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = wav[i * hop: i * hop + frame]
        energies[i] = np.sqrt(np.mean(seg ** 2) + 1e-10)
    ref = energies.max()
    if ref <= 0:
        return wav
    db = 20 * np.log10(energies / ref + 1e-10)
    voiced = db > -top_db
    if not voiced.any():
        return wav
    first = np.argmax(voiced)
    last = len(voiced) - np.argmax(voiced[::-1])
    start = first * hop
    end = min(len(wav), last * hop + frame)
    return wav[start:end]


def fix_length(wav: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop or symmetrically zero-pad to ``target_len`` samples."""
    n = len(wav)
    if n == target_len:
        return wav
    if n > target_len:
        start = (n - target_len) // 2
        return wav[start:start + target_len]
    pad = target_len - n
    left = pad // 2
    right = pad - left
    return np.pad(wav, (left, right), mode="constant")


def preprocess_waveform(wav: np.ndarray, sr: int, target_len: int,
                        do_trim: bool = True) -> np.ndarray:
    """Full preprocessing chain used at train and inference time."""
    if do_trim:
        wav = trim_silence(wav, sr)
    wav = rms_normalize(wav)
    wav = fix_length(wav, target_len)
    return wav.astype(np.float32)
