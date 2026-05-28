"""Central configuration for the EchoFlow voice-pathology pipeline.

All hyper-parameters, the class taxonomy and the audio front-end settings live
here so that training, evaluation, prediction and the Core ML export stay in
perfect agreement.  Keeping a single source of truth is critical: the iOS app
must compute *exactly* the same features as training time, otherwise the model
silently degrades.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List
import json


# ---------------------------------------------------------------------------
# Clinical taxonomy
# ---------------------------------------------------------------------------
# The order of this list defines the integer label of every class and the
# order of the softmax output vector.  Do NOT reorder without retraining and
# re-exporting the Core ML model.
CLASSES: List[str] = [
    "healthy",          # 0 - normal voice
    "hyperfunctional",  # 1 - nodules / polyps (rough, high jitter & shimmer)
    "paralysis",        # 2 - glottic insufficiency / RLN palsy (breathy, noisy)
    "neurological",     # 3 - spasmodic / parkinsonian (tremor, voice breaks)
    "inflammatory",     # 4 - laryngitis / edema (low pitch, moderate roughness)
]

CLASS_DESCRIPTIONS = {
    "healthy": "Здоровый голос без признаков патологии.",
    "hyperfunctional": "Гиперфункциональная дисфония (узелки/полипы): шероховатость, высокий джиттер и шиммер.",
    "paralysis": "Глоточная недостаточность / парез гортани: придыхательность, высокий уровень шума.",
    "neurological": "Неврологическая дисфония (спастическая/паркинсоническая): тремор, нестабильность F0, срывы голоса.",
    "inflammatory": "Воспалительная дисфония (ларингит/отёк): сниженный тон, умеренная шероховатость.",
}

NUM_CLASSES = len(CLASSES)


@dataclass
class AudioConfig:
    """Front-end audio settings shared by Python and the iOS client."""
    sample_rate: int = 16000          # Hz - clinical voice analysis standard
    clip_seconds: float = 3.0         # fixed analysis window
    n_fft: int = 512
    hop_length: int = 160             # 10 ms hop at 16 kHz
    win_length: int = 400             # 25 ms window
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 8000.0
    preemphasis: float = 0.97

    @property
    def clip_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)

    @property
    def n_frames(self) -> int:
        return self.clip_samples // self.hop_length + 1


@dataclass
class FeatureConfig:
    """Handcrafted clinical-biomarker branch dimensionality."""
    # see features/acoustic.py for the exact ordered list (ACOUSTIC_NAMES)
    n_acoustic: int = 39


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 1337
    # synthetic dataset size (per class) used when no real corpus is present
    samples_per_class: int = 600
    label_smoothing: float = 0.05
    early_stop_patience: int = 6


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    classes: List[str] = field(default_factory=lambda: list(CLASSES))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


DEFAULT_CONFIG = Config()
