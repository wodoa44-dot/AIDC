# Appendix tables and figures

## Rebuild

```bash
python reports/appendix_visuals/build_visuals.py
```

This command only reads completed artifacts and rewrites this report directory. It does not train or tune a model.

## Selected experiments

- `trace/predictability/outputs/main_20260914`: completed 36-cell within-run chronological experiment.
- `trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/artifacts/full`: completion checks and replay passed; grouped outer and separately labelled reused holdout.
- Smoke, failed, running, and partial outputs are excluded.

CRPS/MAE/RMSE main tables use run-macro aggregation: metric per run, followed by an equal-weight mean over runs. Heatmap improvement is `100 × (current-only CRPS − history CRPS) / current-only CRPS`. Failure-aware comparisons use seed 42, history+metadata, full 15 quantiles, and matched test targets.

## Deliberate omissions and limitations

- No retrieval panel: the selected experiments contain no retrieval implementation or retrieved keys/values. The RAFT layout is only adapted as a visual organization principle.
- No event count: saved transition tables contain forecast-window counts, not independently identified physical-event counts.
- No training-time comparison for deployment packages: comparable training duration was not saved. Process max RSS is reported only as a high-water mark.
- Figure A3 shows only stored 1 s and 2 s endpoints; it does not invent a future trajectory.
- Reused holdout was previously inspected and is not fresh prospective evidence.
- No local RAFT PDF was found; the supplied table/figure description was used. LaTeX was emitted but not compiled because `pdflatex` is unavailable in the environment.
