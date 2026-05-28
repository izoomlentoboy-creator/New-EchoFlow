"""Unified feature extraction tying the audio front-end to both model branches.

The log-mel spectrogram is produced by the torch ``MelFrontend`` (the very same
module embedded in the exported Core ML graph), so training-time and on-device
spectrograms are identical by construction.  The clinical biomarkers are
produced by ``acoustic_biomarkers`` and returned *unnormalized* (the model
standardizes them internally).
"""
from __future__ import annotations

from typing import Tuple
import numpy as np

from ..config import Config
from ..audio.preprocess import preprocess_waveform
from .acoustic import acoustic_biomarkers


class FeatureExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        from ..models.frontend import MelFrontend
        self.frontend = MelFrontend(cfg.audio).eval()

    def from_waveform(self, wav: np.ndarray, do_trim: bool = True
                      ) -> Tuple[np.ndarray, np.ndarray]:
        a = self.cfg.audio
        wav = preprocess_waveform(wav, a.sample_rate, a.clip_samples,
                                  do_trim=do_trim)
        mel = self.frontend.numpy_mel(wav)             # (n_mels, T)
        bio = acoustic_biomarkers(wav, a.sample_rate)  # (39,)
        return mel, bio

    def from_file(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        from ..audio.io import load_audio
        wav = load_audio(path, self.cfg.audio.sample_rate)
        return self.from_waveform(wav)
