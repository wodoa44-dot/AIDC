# Failure diagnosis frozen before retraining

This diagnosis uses the frozen retrospective audit at `validation/audit_20260921_full_v3`. The grouped-outer and four-run reused-holdout results were already inspected, so every comparison in this experiment is retrospective. The five saved diagnostic selections were reconstructed as continuous ±10 s windows from `per_target_evaluation.parquet`; selected timestamps are illustrations, not tuning observations.

## Verified evidence

- Grouped-outer metadata gains in full-15 CRPS are 4.90% at 1 s and 9.33% at 2 s.
- Compact-model absolute-error exceedance above the diagnostic 500 W threshold is 30.2% at 1 s and 44.5% at 2 s. The longest episodes are 33.4 s and 28.2 s.
- Several LLM-training runs are worse than persistence. In particular, stable endpoints have strongly negative persistence-relative MAE improvement (-754.2% at 1 s and -1515.5% at 2 s).
- The selected 2 s worst case has current power 5327.6 W, observed endpoint 6805.0 W, and compact q50 1664.1 W (absolute error 5140.8 W).
- The selected long-error episode starts with current/endpoint power 987.4/986.8 W and compact q50 1523.8 W. Across its reconstructed window the full-15 q50 stays roughly 0.5--1.7 kW too high while actual power remains near 1 kW.
- Compact/full quantile rearrangement can materially change q50. The audit reports compact-versus-full changes above 1 kW and, at one selected stable 2 s point, raw q50 is 1801.2 W, full-grid q50 is 3846.5 W, and compact q50 is 4131.9 W for a 2060.0 W endpoint. This is important but paired aggregate evidence does not identify export postprocessing as the main failure source.

## Evidence-supported classifications

| Pattern | Classification | Evidence |
|---|---|---|
| Persistent level bias | Supported | A stable H100 LLM window remains near 987 W while q50 remains persistently above it; long 500 W episodes occur in both horizons. |
| Incorrect transition direction | Supported | Transition confusion and signed errors show systematic underprediction on rises and overprediction on falls; the largest 2 s error occurs during a rising endpoint after recent falling power. |
| Missed endpoint decrease | Supported | The largest reused-holdout overprediction occurs on a fall from 6700.5 W current to 1732.6 W endpoint while compact q50 remains 6644.9 W. |
| Excessive predicted change | Supported | Stable endpoints show very poor persistence-relative MAE, including the near-987 W episode predicted near 1524 W. |
| Possible timing/phase mismatch | Insufficient evidence | Some reconstructed windows have higher cross-correlation at small offsets, but the preferred offset is inconsistent and combines model lag, periodic structure, and endpoint definition. No causal phase mechanism is inferred. |
| Quantile postprocessing effects | Supported as a contributor, not established as primary | Full-grid and compact rearrangement can change q50 substantially, while full-15 aggregate MAE is only moderately better. |

## Hypotheses tested

The experiment separately tests target parameterization (absolute versus residual), bounded transition weighting, and the aggregation/model-selection rule. The 100 W transition definition, 500 W diagnostic threshold, 5% CRPS feasibility tolerance, candidate grid, and weights are frozen before scoring and are not tuned against the historical examples above. Residual prediction is not presumed to guarantee stability. Calibration is measured on the ordinary unweighted held-out distribution and is not altered.
