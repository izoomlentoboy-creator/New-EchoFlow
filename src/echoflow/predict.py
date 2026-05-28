"""Inference on a single audio file with a trained checkpoint.

    python -m echoflow.predict path/to/voice.wav
    python -m echoflow.predict path/to/voice.wav --ckpt artifacts/echoflow.pt

Outputs a calibrated, hierarchical verdict (healthy vs pathological, the most
likely pathology subtype, a severity index) and *abstains* ("inconclusive")
when the recording is not a usable voice or the model is not confident enough.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from .config import Config, CLASSES, CLASS_DESCRIPTIONS
from .features.extractor import FeatureExtractor
from .features.acoustic import ACOUSTIC_NAMES
from .models.fusion import EchoFlowNet

# abstain below these thresholds (medical safety: prefer "see a specialist")
MIN_CONFIDENCE = 0.45
MIN_VOICED_FRACTION = 0.30


def load_model(ckpt_path: str, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config()
    model = EchoFlowNet(cfg.features.n_acoustic, len(CLASSES))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    temperature = float(ckpt.get("temperature", 1.0))
    return model, cfg, temperature


def predict_arrays(model, mel, bio, temperature=1.0, device="cpu"):
    mel_t = torch.from_numpy(mel[None, None].astype(np.float32)).to(device)
    bio_t = torch.from_numpy(bio[None].astype(np.float32)).to(device)
    with torch.no_grad():
        logb, logs, sev = model(mel_t, bio_t)
        pb = torch.softmax(logb / temperature, 1)
        ps = torch.softmax(logs / temperature, 1)
        probs = torch.cat([pb[:, :1], pb[:, 1:2] * ps], 1)[0].cpu().numpy()
        severity = float(sev[0, 0])
    return probs, severity


def predict_file(model, cfg, path, temperature=1.0, device="cpu"):
    ex = FeatureExtractor(cfg)
    mel, bio = ex.from_file(path)
    voiced_fraction = float(bio[ACOUSTIC_NAMES.index("voiced_fraction")])
    probs, severity = predict_arrays(model, mel, bio, temperature, device)
    return probs, severity, voiced_fraction


def format_report(probs, severity, voiced_fraction):
    order = np.argsort(probs)[::-1]
    top = order[0]
    conf = float(probs[top])
    lines = []
    if voiced_fraction < MIN_VOICED_FRACTION:
        lines.append("ВЕРДИКТ: неинформативно — в записи мало голоса. "
                     "Запишите устойчивый гласный /а/ 3–5 секунд.")
    elif conf < MIN_CONFIDENCE:
        lines.append(f"ВЕРДИКТ: неинформативно (уверенность {conf*100:.0f}%). "
                     "Рекомендуется консультация специалиста.")
    elif CLASSES[top] == "healthy":
        lines.append(f"ВЕРДИКТ: голос здоров ({conf*100:.1f}%).")
    else:
        path_prob = float(probs[1:].sum())
        lines.append(f"ВЕРДИКТ: вероятна патология ({path_prob*100:.1f}%).")
        lines.append(f"Наиболее вероятный тип: {CLASSES[top]} ({conf*100:.1f}%) — "
                     f"{CLASS_DESCRIPTIONS[CLASSES[top]]}")
        sev_band = ("лёгкая" if severity < 0.33 else
                    "умеренная" if severity < 0.66 else "выраженная")
        lines.append(f"Индекс тяжести: {severity:.2f} ({sev_band}).")
    lines.append("\nВероятности по классам:")
    for i in order:
        bar = "#" * int(probs[i] * 30)
        lines.append(f"  {CLASSES[i]:>15}: {probs[i]*100:5.1f}% {bar}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--ckpt", default="artifacts/echoflow.pt")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    model, cfg, temperature = load_model(args.ckpt)
    probs, severity, vf = predict_file(model, cfg, args.audio, temperature)

    if args.json:
        print(json.dumps({
            "file": args.audio,
            "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
            "severity": severity, "voiced_fraction": vf,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"\nФайл: {args.audio}")
        print(format_report(probs, severity, vf))


if __name__ == "__main__":
    main()
