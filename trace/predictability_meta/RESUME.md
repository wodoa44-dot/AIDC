# Corrected capacity experiment — Stage C continuation

The supported continuation entry point is `corrected_experiment.py`. The historical
`corrected_capacity.py` driver and prior smoke output directories are preserved.
Run commands from `/home/work/AIDC`.

## Implemented

- Fresh Stage-C outer-training refits for legacy-budget, improved-controlled, and
  independently tuned pairs; seeds 17, 42, 2026, with 15 quantiles.
- Persisted selection provenance, encoders including imputation values, exact
  indices/weights, model bundles, requested/resolved/fitted parameters and targets.
- Configuration/input/fold-aware task identities with atomic completion records
  and per-artifact SHA-256 verification. A/B tasks also resume independently.
- Development-only final selection and refits, followed by explicitly labelled
  reused-holdout assessment. Run-balanced paired reporting, coverage/width,
  fixed 100 W transition metrics, metadata split usage, plots and English report.

## Development integration commands and progress

```bash
python trace/predictability_meta/corrected_experiment.py --experiment-id stage_c_dev_resume_20260915 --development-integration --stop-after 15
python trace/predictability_meta/tests/verify_stage_c_integration.py
```

The first command was executed and intentionally exited with code 75 after 15
committed tasks (4 A, 8 B, 3 C). The verification command resumed it: 15 valid
tasks skipped and 15 remaining tasks completed. The independent uninterrupted comparison also passed: maximum prediction
difference was exactly 0.0; all saved completed-task timestamps and hashes were
unchanged. Incomplete-task recomputation and incompatible-config rejection passed. Tolerance: absolute and relative 1e-10. All integration
inputs exclude the four reused-holdout runs. The integration retains all three
final seeds and all 15 quantiles; only its cell/search/learning-curve grid is reduced.

Integration artifacts: `outputs/stage_c_dev_resume_20260915/` and
`outputs/stage_c_dev_uninterrupted_20260915/`. The verification result is
`outputs/stage_c_dev_resume_20260915/integration_verification.json`.

## Production command (running)

```bash
python trace/predictability_meta/corrected_experiment.py --experiment-id corrected_full_20260915
```

Started at 2026-09-16 01:07:18 UTC in persistent command session 65234.
Log: `logs/corrected_full_20260915.log`. Status: `outputs/corrected_full_20260915/status.json`.
Feature preparation completed for all 19 valid cells. Full-budget Stage A is running;
no production assessment results exist yet. Persistent monitoring session: 35529.
The monitor updates the live progress below every 30 seconds and verifies all
12,960 expected A/B/final-selection/C tasks and their artifacts at completion.

Rerun this exact command to resume. Do not add `--development-integration`.
Production reads the fixed 32-run audited inventory, all 19 valid cells,
per-run budgets 250/1000/5000/all, rounds 12/50/100/300, the 24-template
Stage-B plan, and grouped inner/outer folds. Templates whose history cannot be
represented at a cell resolution are explicitly ineligible. Training uses two
LightGBM threads and one fitting task at a time. No hidden budget overrides.

Models and per-target predictions reside under `outputs/<id>/tasks/<identity>/`;
`complete.json` is written only after all task artifacts exist. Invalid or incomplete
artifacts are recomputed; incompatible configurations are rejected. Selection
records live in `stage_b_selections.json` and `final_development_selections.json`.
The holdout freeze manifest is `holdout_selection_freeze.json`.

Legacy sampling is 250 evenly spaced pooled row positions with uniform weights
and 12 rounds, using the history-only-selected history. This is a refitted legacy
budget reference, not an exact replay of the old temporal training restriction.
Bootstrap intervals describe paired independent-run fixed-prediction uncertainty,
not retraining uncertainty. Development integration results are not main results.

## Live production progress

```json
{
  "experiment_id": "corrected_full_20260915",
  "pid": 1236,
  "status": "completed",
  "committed_tasks": {
    "B": 7668,
    "B_final": 2556,
    "A": 1368,
    "C": 1368
  },
  "updated_utc": "2026-09-20T01:36:58Z",
  "last_commit_utc": "2026-09-20T01:36:32Z"
}
```

Production completed and final integrity checks passed. See `outputs/corrected_full_20260915/production_verification.json` and `report_en.md`.
