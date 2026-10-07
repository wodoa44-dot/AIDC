# AIDC residual shrinkage diagnosis

Saved development OOF predictions only. No fit, inference, gate, new sampling, or 2025-09-22 evaluation. All prior experiments remain untouched.

Run commands from `/home/work/AIDC` with the existing system Python (`/usr/bin/python`):

```bash
python experiments/prediction_models/aidc_residual_shrinkage_diagnosis_20261001/pipeline.py --smoke --output-dir experiments/prediction_models/aidc_residual_shrinkage_diagnosis_20261001/outputs/smoke_v3
python experiments/prediction_models/aidc_residual_shrinkage_diagnosis_20261001/manage.py launch --output-dir experiments/prediction_models/aidc_residual_shrinkage_diagnosis_20261001/outputs/full_v2
python experiments/prediction_models/aidc_residual_shrinkage_diagnosis_20261001/manage.py status --output-dir experiments/prediction_models/aidc_residual_shrinkage_diagnosis_20261001/outputs/full_v2
```

To resume the same input/code/config identity, use `manage.py launch ... --resume`. Duplicate jobs are prevented by launch and pipeline locks. Logs and lifecycle files from prior invocations are archived. A changed identity requires a new output directory; existing artifacts are not overwritten to bypass the identity check.

Regenerate figures/report from saved aggregate CSV files, without prediction loading or calculation:

```bash
python experiments/prediction_models/aidc_residual_shrinkage_diagnosis_20261001/manage.py launch --output-dir experiments/prediction_models/aidc_residual_shrinkage_diagnosis_20261001/outputs/full_v2 --report-only
```

This mode verifies both input/code/config hashes and aggregate CSV hashes first. `timing.json` retains the full analysis timing; `report_only_timing.json` records regeneration separately. `completion_status.json` describes the latest invocation.

`smoke_v1` preserves an initial failure from requiring bit-exact `(target-anchor)==1` subtraction. IEEE floating-point subtraction can differ by roughly 1e-14 seconds. The corrected check requires bit-exact `target == anchor+1` plus duration error <=1e-12 seconds. `smoke_v2` and `full_v1` were successful before final report formatting and timing refinements. `smoke_v3` and `full_v2` contain the final code results. These are repeated arithmetic on identical saved predictions, with no model fitting or inference.

Main outputs include canonical `analysis_anchors.parquet`, SHA-256 input manifest, provenance, all requested metrics and decomposition CSVs, verification JSON, PNG/SVG figures and plotting CSVs, Korean Markdown/HTML report, timings, and lifecycle records. Empty-region metrics and p99 with fewer than 100 anchors are missing values in CSV and N/A in reports.
