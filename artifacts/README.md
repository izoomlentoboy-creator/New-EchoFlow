# artifacts/

Outputs produced by the pipeline.

| File | Produced by | Description |
|---|---|---|
| `echoflow.pt` | `echoflow.train` | PyTorch checkpoint: model weights, biomarker mean/std, calibration temperature, config, class names |
| `metrics.json` | `echoflow.train` | Test metrics (accuracy, macro-F1, screening sensitivity/specificity, severity MAE, per-class P/R/F1) |
| `config.json` | `echoflow.train` | Exact config used for the run |
| `EchoFlow.mlpackage` | `echoflow.export_coreml` | On-device Core ML model (waveform + bio → class probs + severity) |
| `parity_refs.json` | `echoflow.parity` | Reference biomarker vectors for the Swift parity test |

> Metrics in `metrics.json` reflect the **synthetic** training data unless you
> trained with `--data-root`. They are not real-patient accuracy. See the root
> `README.md` disclaimer.
