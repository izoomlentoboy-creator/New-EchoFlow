# CLAUDE.md — working notes for this repo

EchoFlow: on-device voice-pathology screening (healthy vs dysphonia subtype +
severity), built to ship as a Core ML model in an iOS app.

## Run things
```bash
export PYTHONPATH=src
python -m echoflow.train --samples-per-class 500 --epochs 30   # synthetic
python -m echoflow.train --data-root data/real                 # real corpus
python -m echoflow.predict <audio.wav>
python -m echoflow.export_coreml --out artifacts/EchoFlow.mlpackage
python -m echoflow.parity --out artifacts/parity_refs.json
# tests (no pytest dependency required):
PYTHONPATH=src python -c "import tests.test_pipeline as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]"
```

## Key invariants (don't break these)
- `echoflow.config` is the single source of truth. `CLASSES` order defines all
  label indices and the Core ML output order — never reorder without retraining
  AND re-exporting.
- `FeatureConfig.n_acoustic` must equal `len(ACOUSTIC_NAMES)` in
  `features/acoustic.py` (currently 39). The test suite checks this.
- The log-mel spectrogram is defined by `models/frontend.MelFrontend` and used
  BOTH to build training features AND inside the export graph — so the CNN
  branch never drifts. Don't compute mels with librosa for training.
- Biomarker standardization + calibration temperature are baked into the model
  / export. The iOS side feeds RAW waveform + RAW biomarkers.
- `ios/EchoFlow/BiomarkerExtractor.swift` must mirror `features/acoustic.py`;
  `ios/EchoFlowTests/BiomarkerParityTests.swift` enforces it against
  `artifacts/parity_refs.json`.

## Environment notes
- No GPU; CPU training is fine (small model). Dataset hosts (PhysioNet etc.)
  are network-blocked here, hence the synthetic generator default.
- Honest framing: synthetic-trained metrics ≠ real-patient accuracy. Always say
  so. Not a medical device.

## Where to improve next
- Plug in a real corpus (`data/download.py` has a diagnosis→class map).
- Bigger CNN / longer training for subtype separation (hyperfunctional vs
  paralysis vs inflammatory overlap most).
- Add k-fold CV and per-speaker splitting once real data exists (avoid leakage).
