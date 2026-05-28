#!/usr/bin/env bash
# End-to-end demo: train (small) -> predict a synthetic sample -> export Core ML.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "==> [1/4] Train a quick model on synthetic data"
python -m echoflow.train --samples-per-class 200 --epochs 15 --out artifacts

echo "==> [2/4] Render a synthetic test recording"
python - <<'PY'
import soundfile as sf
from echoflow.config import Config
from echoflow.data.synthetic import synthesize
cfg = Config()
wav = synthesize("neurological", cfg.audio.sample_rate, 3.0, seed=777)
sf.write("artifacts/demo_voice.wav", wav, cfg.audio.sample_rate)
print("wrote artifacts/demo_voice.wav")
PY

echo "==> [3/4] Predict"
python -m echoflow.predict artifacts/demo_voice.wav

echo "==> [4/4] Export Core ML + parity refs"
python -m echoflow.export_coreml --out artifacts/EchoFlow.mlpackage || \
    echo "(install coremltools to export Core ML)"
python -m echoflow.parity --out artifacts/parity_refs.json

echo "==> Done. See artifacts/."
