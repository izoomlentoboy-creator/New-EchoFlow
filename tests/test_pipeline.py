"""End-to-end sanity tests for the EchoFlow pipeline (CPU, fast)."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from echoflow.config import Config, CLASSES, NUM_CLASSES
from echoflow.data.synthetic import SyntheticVoiceGenerator, severity_of
from echoflow.features.extractor import FeatureExtractor
from echoflow.features.acoustic import acoustic_biomarkers, ACOUSTIC_NAMES, N_ACOUSTIC
from echoflow.models.fusion import EchoFlowNet
from echoflow.models.frontend import MelFrontend

CFG = Config()


def test_acoustic_vector_shape_and_names():
    assert len(ACOUSTIC_NAMES) == N_ACOUSTIC == CFG.features.n_acoustic
    assert len(set(ACOUSTIC_NAMES)) == len(ACOUSTIC_NAMES)  # unique names


def test_feature_extraction_shapes():
    g = SyntheticVoiceGenerator(CFG.audio.sample_rate, seed=0)
    ex = FeatureExtractor(CFG)
    wav = g.render(g.sample_params("healthy"), CFG.audio.clip_seconds)
    mel, bio = ex.from_waveform(wav, do_trim=False)
    assert mel.shape == (CFG.audio.n_mels, CFG.audio.n_frames)
    assert bio.shape == (N_ACOUSTIC,)
    assert np.isfinite(mel).all() and np.isfinite(bio).all()


def test_severity_monotonic_healthy_low():
    g = SyntheticVoiceGenerator(CFG.audio.sample_rate, seed=1)
    sev_healthy = np.mean([severity_of(g.sample_params("healthy")) for _ in range(50)])
    sev_path = np.mean([severity_of(g.sample_params("hyperfunctional")) for _ in range(50)])
    assert 0.0 <= sev_healthy < sev_path <= 1.0


def test_biomarkers_separate_healthy_from_pathology():
    """CPP and HNR should be higher for healthy; jitter/shimmer lower."""
    g = SyntheticVoiceGenerator(CFG.audio.sample_rate, seed=2)
    sr = CFG.audio.sample_rate
    idx = {n: i for i, n in enumerate(ACOUSTIC_NAMES)}

    def avg(cls, n=12):
        vals = [acoustic_biomarkers(g.render(g.sample_params(cls),
                CFG.audio.clip_seconds), sr) for _ in range(n)]
        return np.mean(vals, axis=0)

    h, p = avg("healthy"), avg("hyperfunctional")
    assert h[idx["cpp_mean"]] > p[idx["cpp_mean"]]
    assert h[idx["hnr_mean"]] > p[idx["hnr_mean"]]
    assert h[idx["jitter_local"]] < p[idx["jitter_local"]]


def test_model_forward_and_class_probs():
    model = EchoFlowNet(N_ACOUSTIC, NUM_CLASSES).eval()
    mel = torch.randn(4, 1, CFG.audio.n_mels, CFG.audio.n_frames)
    bio = torch.randn(4, N_ACOUSTIC)
    logb, logs, sev = model(mel, bio)
    assert logb.shape == (4, 2) and logs.shape == (4, 4) and sev.shape == (4, 1)
    probs, sev2 = model.class_probs(mel, bio)
    assert probs.shape == (4, NUM_CLASSES)
    assert torch.allclose(probs.sum(1), torch.ones(4), atol=1e-5)
    assert (sev2 >= 0).all() and (sev2 <= 1).all()


def test_melfrontend_matches_extractor():
    """The frontend used by the extractor must define the spectrogram."""
    fe = MelFrontend(CFG.audio).eval()
    g = SyntheticVoiceGenerator(CFG.audio.sample_rate, seed=3)
    from echoflow.audio.preprocess import preprocess_waveform
    wav = g.render(g.sample_params("paralysis"), CFG.audio.clip_seconds)
    wav = preprocess_waveform(wav, CFG.audio.sample_rate, CFG.audio.clip_samples,
                              do_trim=False)
    mel = fe.numpy_mel(wav)
    assert mel.shape == (CFG.audio.n_mels, CFG.audio.n_frames)
    assert abs(float(mel.mean())) < 1e-3   # per-utterance CMVN -> ~zero mean


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
