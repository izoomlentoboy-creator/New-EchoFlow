# EchoFlow — Design Document

Voice-pathology detection AI, built from scratch to be embedded **fully
on-device** in an iOS app. This document is the single source of truth for the
architecture, training logic, and the honest limitations.

## 0. The honest constraint (read this first)

There is **no real clinical corpus** available in the build environment
(dataset hosts are blocked by network policy). The pipeline therefore trains on
a **physiologically-grounded synthetic generator** by default.

A model trained only on synthetic audio is **not clinically valid** on real
patients. We mitigate the sim→real gap by design (domain randomization +
physically-invariant biomarkers), and the *same* training command consumes a
real corpus when you provide one:

```bash
python -m echoflow.train --data-root data/real
```

Clinical-grade accuracy requires real labelled data + clinical validation.

## 1. "Fully independent" = 100% on-device

- **Model input is the raw waveform.** The log-mel spectrogram is computed
  *inside the Core ML graph* (STFT as a fixed-weight Conv1d + mel filterbank),
  so the CNN branch is self-contained and cannot drift from training.
- **Clinical biomarkers** (jitter/shimmer/CPP/…) need control flow, so they are
  computed by a native Swift module that **mirrors the Python implementation**,
  guarded by a **parity test** (reference vectors shipped in `artifacts/`).
- All normalization constants are **baked into the graph** (buffers).
- No network, no server, no third-party runtime needed at inference.

## 2. Model — hierarchical multi-task (matches clinical reasoning)

```
waveform ─► MelFrontend (in-graph STFT+mel+log) ─► SE-CNN ─► attention pool ─┐
                                                                             ├─ Gated Fusion ─► shared embedding ─┬─► Head A: healthy vs pathological (binary)
clinical biomarkers ─────────────────► BioMLP ───────────────────────────────┘                                   ├─► Head B: pathology subtype (4-way)
                                                                                                                  └─► Head C: severity index (regression)
```

- **Attention pooling** over time focuses on informative frames (voice breaks,
  rough segments) instead of averaging them away.
- **Gated fusion**: each branch emits a sigmoid gate re-weighting the other, so
  an unreliable branch can be suppressed per-sample.
- **Calibration** via temperature scaling on the validation set.
- **Abstention**: low max-probability or non-voice input ⇒ "inconclusive".
- **Multi-task uncertainty weighting** (Kendall homoscedastic) balances the
  three losses automatically.

## 3. Clinical biomarkers (the interpretable branch)

Evidence-based, partly invariant to the sim→real gap:

| Feature | Why |
|---|---|
| F0 mean/std/min/max/range | pitch & instability |
| Jitter (local, ddp) | frequency perturbation (mild–moderate periodic dysphonia) |
| Shimmer (local, dda) | amplitude perturbation |
| HNR mean/std | overall noise |
| **CPP / CPPS** | **#1 dysphonia marker; robust on aperiodic voice** |
| **GNE** | breathiness / glottal noise |
| **LTAS tilt / spectral slope** | hyperfunction vs breathiness |
| voiced fraction, ZCR | voicing stability |
| spectral centroid/bandwidth/rolloff/flatness | timbre |
| MFCC means | spectral envelope |

The clinically-validated **AVQI-style** combination (CPPS + HNR + shimmer +
LTAS slope) drives the severity head.

## 4. Training logic — beating sim→real

- **Domain randomization** in synthesis: microphone IRs, codec artifacts,
  reverberation, noise types (white/hum/babble), clipping, level, age/sex F0 &
  formant ranges, multiple vowels. Forces invariant pathology features.
- **Augmentation**: waveform aug + SpecAugment + mixup.
- **Optimizer**: AdamW + cosine LR (+warmup), label smoothing, grad clip.
- **Selection**: stratified split / CV, early stopping on **macro-F1**.
- **Post-hoc**: temperature calibration + screening-threshold tuning to a target
  sensitivity (don't miss disease).
- **Metrics**: confusion matrix, per-class P/R/F1, macro-F1, AUROC, ECE.

## 5. Taxonomy (output classes)

0 `healthy` · 1 `hyperfunctional` (nodules/polyps) · 2 `paralysis` (glottic
insufficiency) · 3 `neurological` (spasmodic/parkinsonian) · 4 `inflammatory`
(laryngitis/edema). Head A collapses 1–4 into "pathological".

## 6. Repository layout

```
src/echoflow/
  config.py            taxonomy + audio/feature/train settings (single source of truth)
  audio/               io, preprocess (VAD/normalize), augment
  features/            acoustic biomarkers, spectral mel, unified extractor
  data/                synthetic generator, datasets, real-corpus download helper
  models/              mel frontend, CNN/bio branches, fusion, hierarchical net
  train.py evaluate.py predict.py export_coreml.py calibrate.py
ios/EchoFlow/          Swift: AudioRecorder, BiomarkerExtractor, VoiceAnalyzer
tests/                 unit + parity tests
artifacts/             trained checkpoint, Core ML model, metrics, parity refs
```
