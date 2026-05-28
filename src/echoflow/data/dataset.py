"""Datasets: synthetic generation (+domain randomization) and real-corpus
loading, plus the multi-task torch ``Dataset``."""
from __future__ import annotations

from typing import Tuple
import os
import numpy as np

import torch
from torch.utils.data import Dataset

from ..config import Config, CLASSES
from ..features.extractor import FeatureExtractor
from ..audio.augment import ChannelSimulator
from ..audio.io import load_audio
from .synthetic import SyntheticVoiceGenerator, severity_of


class Corpus:
    """Container for extracted features + multi-task targets."""
    def __init__(self, mels, bios, labels, severities):
        self.mels = mels            # (N,1,n_mels,T)
        self.bios = bios            # (N, n_acoustic)
        self.labels = labels        # (N,) 5-class
        self.severities = severities  # (N,) in [0,1] or -1 if unknown


def build_synthetic_dataset(cfg: Config, verbose: bool = True) -> Corpus:
    """Generate the full synthetic corpus with recording-channel
    randomization and extract features once."""
    a = cfg.audio
    extractor = FeatureExtractor(cfg)
    chan = ChannelSimulator(a.sample_rate, seed=cfg.train.seed + 999)
    n_per = cfg.train.samples_per_class
    mels, bios, labels, sevs = [], [], [], []
    for ci, cls in enumerate(CLASSES):
        gen = SyntheticVoiceGenerator(a.sample_rate, seed=cfg.train.seed + ci)
        for _ in range(n_per):
            vp = gen.sample_params(cls)
            wav = gen.render(vp, a.clip_seconds)
            wav = chan(wav)                          # domain randomization
            mel, bio = extractor.from_waveform(wav, do_trim=False)
            mels.append(mel[None].astype(np.float32))
            bios.append(bio)
            labels.append(ci)
            sevs.append(severity_of(vp))
        if verbose:
            print(f"  [{cls:>15}] generated {n_per} samples")
    return Corpus(np.stack(mels), np.stack(bios),
                  np.asarray(labels, dtype=np.int64),
                  np.asarray(sevs, dtype=np.float32))


def load_real_dataset(cfg: Config, root: str, verbose: bool = True) -> Corpus:
    """Load a real corpus laid out as ``root/<class_name>/*.wav``.

    Class folder names must match ``echoflow.config.CLASSES``.  Severity is
    unknown for real data (set to -1 and masked out of the severity loss).
    """
    a = cfg.audio
    extractor = FeatureExtractor(cfg)
    mels, bios, labels, sevs = [], [], [], []
    for ci, cls in enumerate(CLASSES):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        files = [f for f in os.listdir(cdir)
                 if f.lower().endswith((".wav", ".flac", ".ogg", ".mp3", ".m4a"))]
        for f in files:
            wav = load_audio(os.path.join(cdir, f), a.sample_rate)
            mel, bio = extractor.from_waveform(wav, do_trim=True)
            mels.append(mel[None].astype(np.float32))
            bios.append(bio)
            labels.append(ci)
            sevs.append(-1.0)
        if verbose:
            print(f"  [{cls:>15}] loaded {len(files)} files")
    if not mels:
        raise RuntimeError(f"No audio found under {root} with class subfolders "
                           f"{CLASSES}")
    return Corpus(np.stack(mels), np.stack(bios),
                  np.asarray(labels, dtype=np.int64),
                  np.asarray(sevs, dtype=np.float32))


class VoiceDataset(Dataset):
    """Multi-task tensor dataset with optional spectrogram augmentation.

    Yields (mel, bio, label5, label_bin, label_sub, severity).
      * label5     : 0..4 full class
      * label_bin  : 0 healthy / 1 pathological
      * label_sub  : 0..3 pathology subtype, or -1 (ignored) for healthy
      * severity   : float in [0,1], or -1 (ignored/unknown)
    Biomarkers are returned raw; the model standardizes them internally.
    """

    def __init__(self, corpus: Corpus, train: bool = False, seed: int = 0):
        self.c = corpus
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.c.labels)

    def _spec_augment(self, mel: np.ndarray) -> np.ndarray:
        mel = mel.copy()
        _, F, T = mel.shape
        if self.rng.random() < 0.5:                 # frequency mask
            f = self.rng.integers(0, max(1, F // 6))
            f0 = self.rng.integers(0, max(1, F - f))
            mel[:, f0:f0 + f, :] = 0.0
        if self.rng.random() < 0.5:                 # time mask
            tt = self.rng.integers(0, max(1, T // 6))
            t0 = self.rng.integers(0, max(1, T - tt))
            mel[:, :, t0:t0 + tt] = 0.0
        return mel

    def __getitem__(self, i):
        mel = self.c.mels[i]
        if self.train:
            mel = self._spec_augment(mel)
        label = int(self.c.labels[i])
        label_bin = 0 if label == 0 else 1
        label_sub = -1 if label == 0 else label - 1
        sev = float(self.c.severities[i])
        return (torch.from_numpy(mel.astype(np.float32)),
                torch.from_numpy(self.c.bios[i].astype(np.float32)),
                label, label_bin, label_sub, sev)
