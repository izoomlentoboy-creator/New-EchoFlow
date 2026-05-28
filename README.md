# EchoFlow — Voice Pathology Detection AI

EchoFlow analyses a short voice recording (a sustained vowel `/a/`) and answers
two clinical questions:

1. **Is the voice healthy or pathological?** (binary screening, tuned for high
   sensitivity — it tries hard not to miss disease)
2. **Which pathology category?** — `hyperfunctional` (nodules/polyps),
   `paralysis` (glottic insufficiency), `neurological` (spasmodic/parkinsonian),
   `inflammatory` (laryngitis/edema) — plus a **severity index**.

It is built from scratch to be **fully on-device** on iOS via Core ML: no
server, no network, no third-party runtime at inference.

> ### ⚠️ Honest disclaimer — read before trusting any number
> The model ships trained on a **physiologically-grounded synthetic generator**
> because no real clinical corpus is available in this build (dataset hosts are
> network-blocked). **A synthetic-only model is NOT clinically valid on real
> patients.** EchoFlow is engineered to *transfer* to real data (domain
> randomization + physically-invariant biomarkers), and the same command
> retrains it on a real corpus when you have one. This is a research /
> educational tool and **not a medical device**. Real clinical use requires
> real labelled data and clinical validation.

---

## Why this design (the short version)

- **Hierarchical, multi-task model** mirrors clinical reasoning: a shared
  encoder feeds a *screening* head (healthy vs pathological), a *subtype* head,
  and a *severity* regressor. The final 5-class probability is assembled
  hierarchically so screening and subtyping stay coherent.
- **Hybrid encoder**: an SE-CNN with temporal **attention pooling** over the
  log-mel spectrogram, fused (via learned **gated fusion**) with an MLP over
  **interpretable clinical biomarkers** — F0 stats, jitter, shimmer, HNR,
  **CPP/CPPS** (the #1 dysphonia marker), **GNE** (breathiness), spectral tilt,
  MFCCs. The biomarkers make decisions explainable and transfer better sim→real.
- **Fully independent inference**: the log-mel spectrogram is computed *inside
  the Core ML graph* (a fixed-weight Conv1d STFT + mel filterbank), so the CNN
  branch cannot drift from training. Biomarkers are computed by a native Swift
  module guarded by a **parity test**. Normalization + calibration temperature
  are baked into the model.
- **Safety**: probabilities are **temperature-calibrated**, and the model
  **abstains** ("inconclusive") on non-voice input or low confidence.

See [`DESIGN.md`](DESIGN.md) for the full rationale.

## Project layout

```
src/echoflow/
  config.py          taxonomy + audio/feature/train settings (one source of truth)
  audio/             io, preprocess (VAD/normalize), augment + channel simulator
  features/          acoustic biomarkers, in-graph mel, unified extractor
  data/              synthetic generator, datasets, real-corpus helper
  models/            mel frontend, hybrid hierarchical net (EchoFlowNet)
  train.py predict.py export_coreml.py parity.py
ios/EchoFlow/        Swift: AudioRecorder, BiomarkerExtractor, DSPHelpers, VoiceAnalyzer
ios/EchoFlowTests/   BiomarkerParityTests
tests/               python unit tests
artifacts/           trained checkpoint, Core ML model, metrics, parity refs
```

## Quickstart

```bash
pip install -r requirements.txt

# 1) Train (synthetic data with domain randomization, no dataset needed)
python -m echoflow.train --samples-per-class 500 --epochs 30

# 2) Predict on a recording
python -m echoflow.predict path/to/voice.wav

# 3) Export the on-device Core ML model
python -m echoflow.export_coreml --ckpt artifacts/echoflow.pt \
    --out artifacts/EchoFlow.mlpackage

# 4) Generate Swift parity references
python -m echoflow.parity --out artifacts/parity_refs.json
```

## Training on a REAL corpus (recommended for any real use)

Two freely-available corpora work well (download where you have internet):

- **VOICED** (PhysioNet): https://physionet.org/content/voiced/1.0.0/
- **Saarbrücken Voice Database (SVD)**: http://stimmdb.coli.uni-saarland.de/

Arrange audio into class subfolders (see `echoflow/data/download.py` for a
diagnosis→class mapping helper):

```
data/real/{healthy,hyperfunctional,paralysis,neurological,inflammatory}/*.wav
```

then retrain with the *same* command:

```bash
python -m echoflow.train --data-root data/real --epochs 40
```

Everything downstream (calibration, export, iOS) is unchanged.

## iOS integration

1. Add `EchoFlow.mlpackage` (Xcode compiles it to `.mlmodelc`) and the four
   Swift files in `ios/EchoFlow/` to your app target.
2. Record and analyse:

```swift
let recorder = AudioRecorder()
try recorder.start()
// ... ask the user to sustain "ahh" for ~3-5 seconds ...
let waveform = recorder.stopAndProcess()        // 16 kHz, 3 s, normalized

let analyzer = try VoiceAnalyzer()               // loads EchoFlow.mlmodelc
let result = try analyzer.analyze(waveform: waveform)
print(result.summary)                            // verdict (RU), with abstention
print(result.probabilities)                      // per-class probabilities
```

3. **Before clinical-style use, run `BiomarkerParityTests`** (bundle
   `parity_refs.json`) so the Swift biomarkers match Python within tolerance.
   The CNN branch is exact by construction.

## Metrics

Test-set metrics for the shipped (synthetic) model are written to
`artifacts/metrics.json` after training: accuracy, macro-F1, screening
sensitivity/specificity, severity MAE, calibration temperature, and per-class
precision/recall/F1. **These reflect synthetic data** — expect different
numbers on real recordings.

## License & intended use

Research and educational use. **Not a medical device.** Do not use for
diagnosis without a qualified clinician and proper validation.
