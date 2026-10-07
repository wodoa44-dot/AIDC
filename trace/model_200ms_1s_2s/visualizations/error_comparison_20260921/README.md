# Error visualization results

Generated from the frozen experiment outputs; no model was retrained.

## Main matched model comparison

- Grouped-outer 1 s: CRPS reduction 11.3%, MAE reduction 10.3%, >500 W rate reduction 15.5%.
- Grouped-outer 2 s: CRPS reduction 9.2%, MAE reduction 10.9%, >500 W rate reduction 13.4%.
- The matched comparison is old_frozen vs reliability_selection, history_meta/full15, seed 42, run-balanced.
- Reused holdout is shown separately because it had already been inspected and is not fresh confirmation.

## Predictability experiment

- Best mean history gain vs current-only: resolution 200 ms / horizon 2000 ms (54.2% CRPSS).
- Lowest mean history gain vs current-only: resolution 50 ms / horizon 50 ms (-8.9% CRPSS).
- Values are unweighted means across four cohorts; the min-max band in the CRPS plot exposes cohort spread.
- Predictability and failure-aware results are not pooled because their grids, targets, and evaluation populations differ.

## Files

- `model_error_overview.png`: line charts of CRPS, MAE, p95 error, and >500 W rate across 1 s and 2 s horizons.
- `model_transition_errors.png`: line charts of falling/rising/stable endpoint MAE.
- `model_per_run_paired.png`: per-run original vs failure-aware MAE.
- `model_error_quantiles.png`: absolute-error 50th, 90th, 95th, and 99th percentiles.
- `model_tolerance_curves.png`: fraction of targets above each absolute-error threshold.
- `model_run_deltas.png`: paired per-run MAE lines; red lines show worse runs.
- `model_transition_change_lines.png`: MAE reduction by transition and horizon; below zero means worse.
- `predictability_crps.png`: CRPS by resolution, horizon, and predictor.
- `predictability_history_gain_heatmap.png`: history gain over current-only.
- `predictability_transition_crps.png`: history-model CRPS by transition stratum.
- CSV files contain the exact plotted aggregate values and reductions.
