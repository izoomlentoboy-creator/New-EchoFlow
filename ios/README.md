# EchoFlow — iOS integration

Fully on-device voice-pathology screening with Core ML. No network at runtime.

## Files

| File | Role |
|---|---|
| `EchoFlow/AudioRecorder.swift` | Mic capture → 16 kHz mono → trim/normalize/fix-length (mirrors `preprocess.py`) |
| `EchoFlow/DSPHelpers.swift` | FFT, mel filterbank (Slaney), DCT-II, autocorrelation, Hilbert — dependency-light |
| `EchoFlow/BiomarkerExtractor.swift` | Native 39-dim clinical biomarkers (port of `acoustic.py`) |
| `EchoFlow/VoiceAnalyzer.swift` | Core ML glue → calibrated hierarchical verdict + abstention |
| `EchoFlowTests/BiomarkerParityTests.swift` | Asserts Swift biomarkers ≈ Python reference |

## Setup

1. Build the model and references in Python:
   ```bash
   python -m echoflow.export_coreml --out artifacts/EchoFlow.mlpackage
   python -m echoflow.parity --out artifacts/parity_refs.json
   ```
2. Drag `EchoFlow.mlpackage` into your Xcode project (it compiles to
   `EchoFlow.mlmodelc`). Add the four Swift files to your app target.
3. Add `parity_refs.json` to your **test** target's resources and run
   `BiomarkerParityTests` once on a device/simulator. It must pass before you
   rely on the biomarker branch. (The CNN branch is exact by construction since
   the spectrogram is computed inside the Core ML graph.)
4. Request microphone permission: add `NSMicrophoneUsageDescription` to
   `Info.plist`.

## Model I/O contract

- **Inputs**
  - `waveform`: `MLMultiArray` shape `[1, 48000]`, Float32 — 3 s @ 16 kHz,
    produced by `AudioRecorder.stopAndProcess()`.
  - `bio`: `MLMultiArray` shape `[1, 39]`, Float32 — `BiomarkerExtractor.compute(...)`.
- **Outputs**
  - `classLabel` / `classProbs` — top class + per-class probability dict
    (order: `healthy, hyperfunctional, paralysis, neurological, inflammatory`).
  - `severity` — `[1,1]` Float32 in `[0,1]`.

`VoiceAnalyzer` already wraps all of this and returns a `VoiceResult` with a
ready-to-show Russian `summary`, including the **abstention** behaviour
(non-voice or low-confidence → "неинформативно").

## Notes

- Ask the user to sustain the vowel **/a/** ("аааа") for 3–5 seconds in a quiet
  room for best results.
- The model is small (a few MB) and runs in well under a second on the Neural
  Engine / CPU.
- **Not a medical device.** Show appropriate disclaimers in the UI.
