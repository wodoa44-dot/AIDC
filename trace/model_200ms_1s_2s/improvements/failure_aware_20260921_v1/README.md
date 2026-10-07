# Failure-aware retraining experiment

This versioned experiment compares absolute-power, residual-power, and bounded transition-weighted residual Quantile LightGBM models. It freezes a 12-template grid per horizon and selects both the minimum mean-CRPS candidate and a candidate using the specified 5%-feasible reliability rank. All assessments are retrospective.

Run from `/home/work/AIDC`:

```bash
pytest -q trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/tests
python trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/experiment.py --smoke
python trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/experiment.py
```

The commands are resumable. A configuration hash rejects incompatible checkpoints. Stages can be resumed explicitly with `--stage search`, `--stage evaluate`, or `--stage finalize`; normally use the full command so all remaining compatible work is discovered.

Standalone export checks:

```bash
python trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/replay.py --package trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/models/power-forecast-failure-aware-20260921-v1/history_meta_full15_deployment.joblib --output /tmp/failure_aware_replay.parquet
python trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/benchmark.py --package trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/models/power-forecast-failure-aware-20260921-v1/history_meta_full15_deployment.joblib --output /tmp/failure_aware_benchmark.json
```

Primary outputs are under `artifacts/full`; smoke outputs are isolated under `artifacts/smoke`. New bundles live under `models/power-forecast-failure-aware-20260921-v1`. The original deployment package is never modified.
