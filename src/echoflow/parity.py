"""Generate reference (waveform -> biomarkers) pairs for the Swift parity test.

    python -m echoflow.parity --out artifacts/parity_refs.json

The iOS ``BiomarkerParityTests`` loads this file, runs the Swift
``BiomarkerExtractor`` on each waveform and asserts the 39-dim vector matches
the Python reference within tolerance. This is the contract that guarantees
on-device features equal training-time features.
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from .config import Config
from .data.synthetic import SyntheticVoiceGenerator
from .features.acoustic import acoustic_biomarkers, ACOUSTIC_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/parity_refs.json")
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    cfg = Config()
    sr = cfg.audio.sample_rate
    g = SyntheticVoiceGenerator(sr, seed=4242)
    classes = cfg.classes
    cases = []
    for i in range(args.n):
        cls = classes[i % len(classes)]
        wav = g.render(g.sample_params(cls), args.seconds).astype(np.float32)
        bio = acoustic_biomarkers(wav, sr)
        cases.append({
            "class": cls,
            "waveform": [round(float(v), 7) for v in wav],
            "bio": {n: round(float(b), 6) for n, b in zip(ACOUSTIC_NAMES, bio)},
        })
    payload = {
        "sample_rate": sr,
        "names": ACOUSTIC_NAMES,
        "n_fft": 1024, "hop": 256,
        "tolerance_rel": 0.05,
        "cases": cases,
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {len(cases)} parity cases -> {args.out}")


if __name__ == "__main__":
    main()
