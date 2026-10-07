# Failure-aware deployment anomaly audit

## Scope

This is a retrospective audit of frozen seed-42 reliability-selection predictions (126,091 targets). No model was retrained, selected, tuned, or calibrated. Grouped outer (12 runs) and previously inspected reused holdout (4 runs) are reported separately. Diagnostic thresholds are not safety limits.

## Pooled tail and sustained-error results

- grouped_outer, 1000 ms: MAE 416.6 W; p95/p99/max absolute error 1852.3/3234.8/5123.1 W; signed bias +23.3 W; >500 W 23.3%; 2850 episodes, median/p90/max 0.8/1.4/4.2 s.
- grouped_outer, 2000 ms: MAE 589.0 W; p95/p99/max absolute error 2358.0/3461.6/5106.2 W; signed bias +7.2 W; >500 W 38.1%; 2955 episodes, median/p90/max 0.8/2.2/28.4 s.
- reused_holdout, 1000 ms: MAE 278.1 W; p95/p99/max absolute error 1395.3/3017.0/4964.4 W; signed bias +91.8 W; >500 W 14.7%; 657 episodes, median/p90/max 0.8/1.2/2.2 s.
- reused_holdout, 2000 ms: MAE 477.8 W; p95/p99/max absolute error 2604.3/3665.9/4637.9 W; signed bias -19.7 W; >500 W 20.9%; 631 episodes, median/p90/max 0.4/2.4/15.6 s.

## Interpretation

- Stable endpoints are much easier than rising/falling endpoints, but large transition errors remain. Negative error denotes underprediction; rising periods tend toward underprediction and falling periods toward overprediction.
- The 2 s reused-holdout interval calibration is the clearest limitation: coverage is materially below nominal, especially for the reliability-selected 300-tree model.
- Maxima are observed retrospective cases, not upper bounds. Episode duration is persistence of forecast error across issuance times, not physical control-failure duration.
- Run and cohort results are heterogeneous; four reused-holdout runs are too few for a broad generalization.

## Artifacts

`worst_cases.csv` contains deterministic extreme cases; `outlier_targets_over_500w.parquet` contains every >500 W target; `failure_episodes.parquet` contains all episodes for the fixed 50–1000 W grid.
