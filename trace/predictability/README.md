# Power predictability experiment

Actual continuous power forecasting with numerical CRPS, chronological within-run evaluation, and causal completed-bin features. This is an initial predictability study, not TRACE reproduction.

## Complete run

From `/home/work/AIDC`:

```bash
python trace/predictability/run.py --config trace/predictability/configs/default.yaml
```

An existing output directory is never overwritten. For another run, add `--experiment-id your_unique_id`. All outputs are under `trace/predictability/outputs/<experiment_id>/`.

```bash
python -m pytest -q trace/predictability/tests
python trace/predictability/run.py --config trace/predictability/configs/default.yaml --smoke
```

Dependencies use the existing environment: numpy, pandas, pyarrow, LightGBM, scikit-learn, matplotlib, PyYAML, joblib, tabulate, pytest. No upgrade, external API, GPU, or network required. Package versions and seed are saved for each execution.

## Design

- Read existing `preprocessing/outputs/manifest.csv` and actual `prepared/*.parquet`. Verify SHA256 against source and exact time/power equivalence using existing `powerprep.adapters.load_native`. Never use event rows as a time series.
- Primary input: highres_node, 8 GPU sum, CPU excluded. Separate B200/H100 and image/LLM cohorts; first three canonical source paths per cohort, fixed before scores. Other canonical runs and duplicate aliases have explicit exclusions.
- NLR training and online-rate are excluded: `create_dataframe_multinode` uses two-sided interpolation, including the training 200 ms grid. No causal reconstruction or supplementary interpolated-grid claim is made.
- Raw recording cadence ~20 ms is irregular and sensor update resolution unverified. Select 50/200/1000 ms completed bins, no upsampling or filling. `(end-r,end]` recorded sample arithmetic mean, timestamp=end, minimum 80% nominal sample-count coverage. Omit incomplete segment-edge bins. Split on invalid values, timestamp non-increase, >5 median-dt gaps and missing bins.
- Select compatible horizons among 50/200/1000/2000 ms: nine resolution/horizon pairs. The unsupported 20 ms grid has material holes due to jitter; 100/500 ms omitted for manageable CPU budget. Save `selected_grid.csv` before fitting. Do not round horizons.
- Run-relative chronological 60/20/20 split; both issuance and target within split; trailing input may use previously observed earlier splits. Never cross segments. Same target identities across three conditions, all history candidates use maximum-history warm-up. Test targets on common 200 ms multiples where possible (1000 ms cadence at 1000 ms resolution). Coarser resolutions change the target.
- Unconditional training empirical quantiles; current-only and history quantile LightGBM with equal per-fit tree budget. Trailing history 200/1000/5000 ms selected only on validation; exclude incompatible windows. History duration describes time support of bins, including current. Save choice before test, without train+validation refit. Training/validation deterministic sampling limits and exact indices recorded.
- Fixed quantile grid, monotone rearrangement; pre-correction crossing recorded. Numerical CRPS: trapezoidal 2*pinball integral, constant outer quantiles through q=0/1. Same approximation for all forecasts; no scaling. Zero-reference CRPSS returns NaN.
- Training 80th percentile absolute horizon change freezes rising/falling threshold. Retrospective target change strata never enter features.
- Paired 10 s temporal block bootstrap within fixed known runs, 500 replicates. Segment boundaries preserved. Four chronological test blocks/run. Intervals are exploratory and cannot establish unseen-run generalization.
- Representative examples: 200 ms bins, first compatible horizon >=1000 ms, lexicographically first run ID, first eligible 20 s of test. All plotted forecasts aligned to target timestamp.

## Output layout

`excluded_sampling_provenance.json`, `data_audit.csv`, `input_manifest.csv`, `inventory_all_files.csv`, `excluded_inputs.csv`, `excluded_segments.csv`, `prior_quality_all_runs.csv`, `bin_audit.csv`, `split_summary.csv`, `selected_grid.csv`, `effective_config.json`, `environment.json`, `selected_validation_configs.csv`, `validation_metrics.csv`, `metrics.csv`, `test_block_metrics.csv`, `transition_metrics.csv`, `confidence_intervals.csv`, `comparison_summary.csv`, `timing.csv`, `execution.log`, `status.json`, `report_ko.md`.

- `continuous/`: coverage-qualified causal bins with run/segment/scope/provenance.
- `predictions/`: Zstandard Parquet, per-target observed power, issue/target times, quantiles, CRPS, cohort, split and strata for every condition.
- `models/`: selected model ensembles, empirical quantiles, validation choice, train/validation/test sample indices. Indices refer to deterministic concatenation from `src.data.build`.
- `plots/`: CRPS and CRPSS versus horizon, resolution/horizon history-gain heatmaps, observed/forecast interval plots, transition scores; scopes/cohorts remain separate.

`src/core.py`: causal math; `src/data.py`: audit/adapters; `src/experiment.py`: fit/score/bootstrap; `src/reporting.py`: plots and Korean interpretation. Tests check CRPS known cases, causality, split and segment boundaries, target matching and physical durations.

## Completed execution

Main experiment `main_20260914`: 36 cells completed in 72.2 s; final artifact checks 9 passed. Successful real-data smoke: `smoke_final_20260914` (4 cells). A failed initial smoke was retained in `smoke_20260914/status.json`; its one-bin feature issue was corrected before the main run. The first detached shell launch was cleaned up by the execution host before Python started; the monitored nohup session completed with PID 2267. See `logs/main_20260914.log` and `outputs/main_20260914/status.json`.
