# Audit of completed expanded metadata ablation

## Scope
Read-only audit of `expanded_main_cpu_bounded_20260915` and `confirmation_v2_20260915`; no model fitting or forecasting was performed. The matching reported numerical lineage identifies these, rather than smoke, initial-12-run, or abandoned output directories.

## Headline findings
1. The grid has **19**, not 20, compatible cells (6+5+3+3+2). This is a reporting/counting error, not a missing cell.
2. The YAML says four estimators, but the actual candidate dictionaries hard-code **12 boosting rounds**; results must not be described as four-tree models.
3. Metadata was constructed and passed in reconstructed code evidence as one-hot GPU/workload/AI-model plus numeric batch, with training-fold vocabularies and explicit UNKNOWN columns. GPU count is constant and excluded. No persisted matrix/encoder proves the runtime values beyond this reconstruction.
4. No models were saved. Thus metadata split use, actual tree depths/leaves, and whether equal predictions arose from no metadata splits cannot be established retrospectively.
5. Direct saved-prediction joins show whether equality is exact at sample/quantile level; see `prediction_equality.csv`. Equal-score cells should not be interpreted as a rounding phenomenon when their equality fraction is 1.0.
6. Outer test groups are separated and reserved confirmation runs are absent from main outer predictions. But the inner validation is chronological rows within non-test runs, not grouped-inner validation, so grouped-outer results are exploratory rather than fully nested grouped evidence.

## Trustworthiness
The published CRPS can be recomputed from saved post-rearrangement predictions and run-balanced seed-averaging in `recomputed_metrics.csv`. The negative conclusion is trustworthy only as a result of this constrained implementation: pooled 250-row fits, 12 rounds, incomplete model provenance, and non-grouped inner validation. It does not establish that metadata is absent, unused by trees, or unhelpful at larger training capacity. A future development-only learning-curve study should vary pooled/per-run training support and boosting rounds, save models/encoders/indices, and keep these confirmation runs out of selection.
