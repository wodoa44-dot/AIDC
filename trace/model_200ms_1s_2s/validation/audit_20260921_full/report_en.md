# Frozen lightweight AIDC forecasting retrospective validation

## Scope and verdict

This audit reloaded frozen fold-specific and final bundles. It did not retrain, tune, recalibrate, or change postprocessing. Grouped outer is retrospective and the four-run reused holdout was previously inspected; neither is fresh prospective confirmation.

All 16 reconciliation checks passed: **True**. Integrity passed: **True**. Canonical model-condition rows: 629,374.

## Average predictive value

- 1000 ms full-15 grouped outer CRPS: history-only 367.52 W, metadata 349.52 W, gain 4.90%.
- 2000 ms full-15 grouped outer CRPS: history-only 521.75 W, metadata 473.06 W, gain 9.33%.

Persistence is a point baseline only; it is not assigned an interval or probabilistic score.

## Per-run and cohort consistency

- 1000 ms: improved 9/12 runs, tied 0, worsened 3; median gain 6.51%, range -23.69% to 23.51%.
- 2000 ms: improved 9/12 runs, tied 0, worsened 3; median gain 6.87%, range -6.23% to 30.53%.

Each grouped-outer cohort has three independent runs and each reused-holdout cohort one. Cohort patterns are descriptive and metadata effects remain confounded.

## Rare and sustained errors

- grouped_outer, 1000 ms: MAE/RMSE 486.7/837.2 W; p95/p99/max absolute error 1958.5/3273.7/4784.8 W; under/over rates 51.3%/48.7%.
- grouped_outer, 2000 ms: MAE/RMSE 688.8/1026.5 W; p95/p99/max absolute error 2433.1/3143.6/5140.8 W; under/over rates 48.2%/51.8%.
- reused_holdout, 1000 ms: MAE/RMSE 305.5/618.7 W; p95/p99/max absolute error 1312.6/2889.4/4912.3 W; under/over rates 53.1%/46.9%.
- reused_holdout, 2000 ms: MAE/RMSE 607.7/967.1 W; p95/p99/max absolute error 2644.0/3182.2/4788.3 W; under/over rates 44.1%/55.9%.
- At diagnostic 500 W, grouped-outer 1000 ms exceeds at 30.2% of valid issue times; longest episode 33.4 s.
- At diagnostic 500 W, grouped-outer 2000 ms exceeds at 44.5% of valid issue times; longest episode 28.2 s.

Episodes measure persistent forecast error across issuance times, not physical control failure. Maxima are observations, not bounds.

## Endpoint changes and persistence

- 1000 ms rising: MAE 606.5 W; improvement versus persistence nan%.
- 1000 ms falling: MAE 919.5 W; improvement versus persistence nan%.
- 1000 ms stable: MAE 218.5 W; improvement versus persistence nan%.
- 2000 ms rising: MAE 666.2 W; improvement versus persistence nan%.
- 2000 ms falling: MAE 1038.6 W; improvement versus persistence nan%.
- 2000 ms stable: MAE 450.2 W; improvement versus persistence nan%.

These are endpoint-change diagnostics, not ramp-onset, lead-time, peak, or trajectory tests.

## Uncertainty reliability

- grouped_outer, 1000 ms: q10-q90 coverage 79.7%, lower/upper violations 11.8%/8.5%, width 1800.3 W, interval score 2509.0 W.
- grouped_outer, 2000 ms: q10-q90 coverage 67.5%, lower/upper violations 19.6%/12.9%, width 2229.4 W, interval score 3141.1 W.
- reused_holdout, 1000 ms: q10-q90 coverage 76.7%, lower/upper violations 9.7%/13.6%, width 1425.8 W, interval score 1790.9 W.
- reused_holdout, 2000 ms: q10-q90 coverage 63.6%, lower/upper violations 24.6%/11.8%, width 1768.4 W, interval score 2962.5 W.

Coverage varies by run, cohort, and direction. No calibration was fitted.

## Compact export effects

- 1000 ms history_meta: compact sorting q50 mean/max change versus raw 1.4/520.1 W; compact-versus-full max 1102.2 W; compact/full MAE 305.5/300.1 W.
- 1000 ms history_only: compact sorting q50 mean/max change versus raw 1.0/348.2 W; compact-versus-full max 1651.2 W; compact/full MAE 336.5/335.7 W.
- 2000 ms history_meta: compact sorting q50 mean/max change versus raw 45.7/1808.4 W; compact-versus-full max 1205.7 W; compact/full MAE 607.7/587.7 W.
- 2000 ms history_only: compact sorting q50 mean/max change versus raw 31.7/1672.3 W; compact-versus-full max 982.2 W; compact/full MAE 634.6/627.1 W.

The reported >1 kW change is reproduced, but paired evidence does not establish export sorting as the main overall weakness.

## Coverage, provenance, and limits

The inventory contains 1,418 rows. Predictions cover every eligible cached feature row (stride 1, no evaluation cap); no interpolation was used. Gaps and segment boundaries break episodes.
Each outer bundle matched its fold test membership with zero train/test index overlap. Final models were evaluated only on the four excluded reused-holdout runs.

- 1000 ms metadata-minus-history CRPS -18.0 W, paired-run bootstrap 95% interval [-40.8, 2.8] (12 runs).
- 2000 ms metadata-minus-history CRPS -48.7 W, paired-run bootstrap 95% interval [-82.1, -16.3] (12 runs).

Bootstrap uses 2,000 paired-run replicates after averaging seeds within run. It excludes retraining and selection uncertainty.
New untouched runs, operational tolerances, serving timing, controller behavior, hardware safeguards, and safety/control readiness remain unverified. The tolerance grid is not a safety threshold.

## Reproducibility

Commands and SHA-256 hashes are recorded in execution.log and source_artifact_hashes.csv.
