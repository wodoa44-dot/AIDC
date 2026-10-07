# Grid-stability forecast pilot

Current progress and remaining dependencies are in [STATUS.md](STATUS.md).

All work is offline and device controls are dry-run. Run from the repository root:

```bash
PYTHONPATH=trace/grid_stability_forecast python -m unittest discover -s trace/grid_stability_forecast/tests -v
python trace/grid_stability_forecast/audit_current_model.py
python trace/grid_stability_forecast/audit_data.py
python trace/grid_stability_forecast/build_targets.py
python trace/grid_stability_forecast/train.py
python trace/grid_stability_forecast/evaluate.py
python trace/grid_stability_forecast/control_replay.py
python trace/grid_stability_forecast/learning_curve.py
python trace/grid_stability_forecast/negative_control.py
python trace/grid_stability_forecast/ablate_static.py
python trace/grid_stability_forecast/recount_events.py
python trace/grid_stability_forecast/compare_events_same_source.py
python trace/grid_stability_forecast/unseen_condition.py
python trace/grid_stability_forecast/summarize_unseen.py
python trace/grid_stability_forecast/evaluate_unseen.py
python trace/grid_stability_forecast/make_unseen_figure.py
python trace/grid_stability_forecast/nested_select.py
python trace/grid_stability_forecast/verify_nested.py
python trace/grid_stability_forecast/summarize_nested.py
python trace/grid_stability_forecast/train_events.py
python trace/grid_stability_forecast/make_tables.py
python trace/grid_stability_forecast/make_figures.py
python trace/grid_stability_forecast/verify.py
python trace/grid_stability_forecast/run_grid.py
python trace/grid_stability_forecast/make_grid_tables.py
python trace/grid_stability_forecast/make_grid_figures.py
python trace/grid_stability_forecast/summarize_nested_cv.py
python trace/grid_stability_forecast/evaluate_onset.py
python trace/grid_stability_forecast/extract_online_runtime.py
python trace/grid_stability_forecast/online_runtime_pilot.py
python trace/grid_stability_forecast/summarize_online_runtime.py
python trace/grid_stability_forecast/run_long_horizon.py
python trace/grid_stability_forecast/summarize_long_horizon.py
python trace/grid_stability_forecast/audit_highres_sessions.py
python trace/grid_stability_forecast/assess_day_sensitivity.py
python trace/grid_stability_forecast/run_nlr_training_pilot.py
python trace/grid_stability_forecast/summarize_nlr_training.py
```

The +1 s pilot is under `results/actual_20260926/`; its eight leave-one-static-feature-out fits are under `results/ablation_static_20260926/`; the NLR training offline recovery remeasurement is under results/event_v2_nlr_training_20260926/; the fixed-setting 10-horizon sweep is under `results/grid_fixed_20260926/`. The runner resumes only verified horizon cells with the same code hash. The forecast experiments use existing `forecast_v2_complete_20260916` bins and do not copy original data. The grouping and source hashes are recorded. `--compact` in `train.py` changes the score grid, so it must be reported as a separate experiment directory.

The program still needs grouped tuning for the remaining 0.2–1.8 s horizons, independent-session validation, Phase 0 device identification, and validated PCC replay. NLR online-rate runtime logs now support a timestamped, within-acquisition dynamic-metadata pilot; the existing highres event probabilities have an exploratory one-to-one 200 ms step-onset assessment. Empty results are never filled with synthetic performance.

The retrospective leave-one-GPU/workload-combination-out assessment with three sampling seeds is in results/unseen_condition_20260926/. The grouped inner search is in results/nested_select_20260926/ and reuses the earlier outer holdout; it is exploratory.

Outer 5-fold / inner 3-fold staged search completed in results/nested_cv5_20260926/ on 2026-09-27. All five fold prediction hashes and fold verification files are recorded in progress.json and fold*/results/verification.json. `summarize_nested_cv.py` regenerates transition, calibration, selected-parameter and Korean report files from the verified predictions. RUNNING.md is the historical launch and resume record; no runner remains active.

The fixed-setting 5/10/30 s sweep completed in results/long_fixed_20260927/. The highres source-timestamp audit found four collection days perfectly aligned with four GPU/workload cohorts; see results/actual_20260926/highres_session_report_ko.md. NLR online-rate B1–B4 is a separate one-acquisition exploration in results/online_runtime_pilot_20260927/.

The NLR training +1 s CPU+GPU forecast pilot is separate from highres GPU-only results in results/nlr_training_pilot_20260927/. Its eight held-out jobs come from one source project; the report treats metadata gain as descriptive.


The reusable grouped inner horizon runner has completed +2/+5/+10/+30 s on the existing highres holdout. `compare_nested_horizons.py` joins these with the +1 s nested result and creates run-balanced CRPS, interval coverage, and paired-gain figures. `diagnose_calibration.py` checks quantile reliability and outcome-defined transition coverage without fitting to test predictions. `tune_onset_threshold.py` selects alarm thresholds on four held-out development runs and applies them to the existing test probabilities. All figures and detailed reports stay under `results/`; the code-only Git publication excludes those derived data files.

```bash
python trace/grid_stability_forecast/compare_nested_horizons.py
python trace/grid_stability_forecast/diagnose_calibration.py --out trace/grid_stability_forecast/results/nested_h30000_20260927
python trace/grid_stability_forecast/tune_onset_threshold.py
```
