# Preprocessing v2.1 — completed

Final output: `outputs/forecast_v2_1_20260916/`.
Processing and independent persisted-artifact verification completed successfully.
21 regression tests and the six-dataset integration passed. No forecasting models
were trained. Existing forecasting experiment PID 9834 remains running separately.

Reproduce/resume from `/home/work/AIDC`:

```bash
PYTHONPATH=trace/preprocessing OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python -m powerprep all --config trace/preprocessing/configs/forecast_v2.yaml --output trace/preprocessing/outputs/forecast_v2_1_20260916
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python trace/preprocessing/tests/verify_full_v2.py
```

Evidence: `full_verification.json`, `integration_verification.json` in the separate
`forecast_v2_complete_smoke_20260916` fixture, and the main `handoff_ko.md`.
Actual data/window counts and supported configurations are in `dataset_summary.csv`,
`resolution_manifest.csv`, `forecast_support.csv`, and `window_counts.csv`.
The main execution log is saved as `execution.log`.

Prior outputs remain preserved: `forecast_v2_20260916` is an incomplete superseded
attempt; `forecast_v2_complete_20260916` completed before partial-clock recovery.
Only `forecast_v2_1_20260916` is the final corrected preprocessing result.
NLR inference split/window settings remain explicitly unsupported under conservative
shared-acquisition grouping. Raw sensor coverage and physical sensor cadence remain
unknown; supplied NLR interpolation is not asserted causal. Optional event detection
was verified on small samples, not run on the full dataset.
