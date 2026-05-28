"""Export a trained EchoFlowNet to a Core ML .mlpackage for iOS.

    python -m echoflow.export_coreml --ckpt artifacts/echoflow.pt \
        --out artifacts/EchoFlow.mlpackage

The exported model is **fully self-contained**:
  * inputs  - ``waveform`` (1, clip_samples) raw audio + ``bio`` (1, n_acoustic)
  * the log-mel spectrogram is computed *inside the graph* (MelFrontend), so
    iOS only has to feed the raw waveform and the clinical biomarkers
  * biomarker standardization and the calibration temperature are baked in
  * outputs  - ``classLabel`` + ``classProbs`` (5-class) and ``severity`` (0..1)

Run on macOS to also compile to .mlmodelc; conversion itself works anywhere
coremltools is installed.
"""
from __future__ import annotations

import argparse
import torch
import torch.nn as nn

from .config import Config, CLASSES
from .models.fusion import EchoFlowNet
from .models.frontend import MelFrontend


class DeployModel(nn.Module):
    """waveform + bio -> (5-class probabilities, severity), calibrated."""
    def __init__(self, cfg: Config, model: EchoFlowNet, temperature: float):
        super().__init__()
        self.frontend = MelFrontend(cfg.audio)
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, waveform, bio):
        mel = self.frontend(waveform)                  # (B,1,n_mels,T)
        logb, logs, sev = self.model(mel, bio)
        pb = torch.softmax(logb / self.temperature, dim=1)
        ps = torch.softmax(logs / self.temperature, dim=1)
        probs = torch.cat([pb[:, :1], pb[:, 1:2] * ps], dim=1)
        return probs, sev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/echoflow.pt")
    ap.add_argument("--out", default="artifacts/EchoFlow.mlpackage")
    args = ap.parse_args()

    import coremltools as ct

    cfg = Config()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = EchoFlowNet(cfg.features.n_acoustic, len(CLASSES))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    temperature = float(ckpt.get("temperature", 1.0))

    deploy = DeployModel(cfg, model, temperature).eval()
    a = cfg.audio
    ex_wav = torch.randn(1, a.clip_samples)
    ex_bio = torch.randn(1, cfg.features.n_acoustic)
    traced = torch.jit.trace(deploy, (ex_wav, ex_bio))

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="waveform", shape=ex_wav.shape),
                    ct.TensorType(name="bio", shape=ex_bio.shape)],
            outputs=[ct.TensorType(name="probs"), ct.TensorType(name="severity")],
            classifier_config=ct.ClassifierConfig(
                class_labels=list(CLASSES),
                predicted_probabilities_output="probs"),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15,
        )
    except Exception as e:               # fallback: plain regressor-style outputs
        print(f"[warn] classifier_config path failed ({e}); exporting plain outputs")
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="waveform", shape=ex_wav.shape),
                    ct.TensorType(name="bio", shape=ex_bio.shape)],
            outputs=[ct.TensorType(name="probs"), ct.TensorType(name="severity")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15,
        )

    mlmodel.author = "EchoFlow"
    mlmodel.short_description = ("On-device voice pathology screening: healthy "
                                 "vs dysphonia subtype + severity.")
    mlmodel.input_description["waveform"] = (
        f"Mono {a.sample_rate} Hz waveform, {a.clip_seconds}s "
        f"({a.clip_samples} samples), RMS-normalized.")
    mlmodel.input_description["bio"] = (
        f"{cfg.features.n_acoustic} raw clinical acoustic biomarkers "
        "(see ACOUSTIC_NAMES / BiomarkerExtractor.swift).")
    mlmodel.save(args.out)
    print(f"Saved Core ML model -> {args.out}")
    print(f"Class order: {CLASSES}")


if __name__ == "__main__":
    main()
