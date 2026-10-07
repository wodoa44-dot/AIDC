# Forecasting-ready preprocessing (v2)

The implemented v2 pipeline is documented in [FORECAST_V2.md](FORECAST_V2.md).
Use `configs/forecast_v2.yaml` and a separate versioned output directory. It adds
native-order quality segmentation, evidence-aware metadata, named resolutions,
strict split/window indices, training-only normalization and independent stages.
No forecasting models are trained. The legacy workflow below documents v1 replay
and its old limitations; existing experiment inputs remain unchanged.

---

# Power trace preprocessing and transition analysis

This package inventories, deduplicates, validates, prepares, and analyzes the selected AI power traces. It addresses whether passive traces contain long low-power intervals followed by rapid recovery. It does not forecast load or implement filler control.

## Reproduction

Run from this directory with the existing repository Python environment:

```bash
cd /home/work/AIDC/trace/preprocessing
python -m powerprep inventory
python -m powerprep all
pytest -q
```

A representative smoke run is:

```bash
python -m powerprep all --output outputs/smoke --limit 1
```

Optional NLR inference datasets use the same adapter:

```bash
python -m powerprep all --datasets nlr_offline nlr_online_finite --output outputs/optional
```

Stages `preprocess`, `detect`, and `report` execute the integrated incremental pipeline so their requested artifact and dependencies stay consistent. Prepared files are reused only when both source hash and configuration fingerprint match. Use `--no-resume` to rebuild. Inputs remain read-only; all artifacts are written below the selected output directory.

## Data interpretation

High-resolution Node files expose `gpu0_power_W` through `gpu7_power_W`. `power_w` is their strict row-wise sum: all eight values must be valid. CPU values are excluded, so its scope is **one-node, eight-GPU sum**. Per-GPU values and GPU dispersion remain in prepared files.

NLR aggregation source establishes that aggregate `power[W]` includes NVML GPU power and RAPL CPU/package/core power. Training additionally sums all participating nodes, so training has **whole-job CPU+GPU** scope; inference is **single-node CPU+GPU**. These scopes must not be treated as equivalent to the High-resolution GPU-only measurements.

NLR online-rate source logs are near 200 ms. Its postprocessor creates a 1 ms common index and applies first-order polynomial interpolation. The 1 ms points are therefore an interpolation grid, not direct measurements or verified sensor updates. The pipeline retains provenance and prepares 0.1 s time-aware means with a configurable bin coverage requirement. Interpretations below the source cadence are flagged unsupported. No coarser series is upsampled.

`delta_event_s` is the window used to measure recovery slope. `low_duration_s` is time from confirmed low-state entry to recovery onset. `recovery_rise_time_10_90_s` is the elapsed time between first crossings of 10% and 90% of the recovered excursion. These quantities are distinct.

## Detector semantics

For each uninterrupted segment, the detector computes a trailing robust median baseline over `baseline_window_s`. The baseline is eligible when its median absolute deviation is below the configured fraction. A low state requires both the absolute and exploratory baseline-relative drop thresholds for `low_persistence_s`. Recovery begins when power reaches `recovery_fraction` of the pre-event baseline and remains there for `recovery_persistence_s`.

Timestamps have these meanings:

- `drop_onset_s`: last sample immediately before confirmed low-state entry.
- `low_entry_s`: first sample of the persistent low state.
- `recovery_onset_s`: first sample of the persistent recovered state.
- `recovery_confirmation_s`: last sample required to confirm recovery persistence.

The detector emits at most one event for an occupied low interval within one configuration. A trace that ends or reaches a gap while low is right-censored; its end is not substituted as recovery. A low state at trace start has insufficient baseline and is not forced into an event. Files are never joined. Missing power, non-positive timestamp steps, resets, and large gaps split segments. The default large-gap threshold is five times each run's median positive interval.

Thresholds of 10%, 20%, and 30% are exploratory baseline-relative sensitivity settings. They are neither rated-power thresholds nor grid-code limits. Absolute watts remain primary, no per-run maximum normalization is performed, and no rated fractions are calculated without documented denominators.

## Outputs

- `manifest.csv`: source hashes, canonical aliases, metadata status, scope, and evidence.
- `duplicates.csv`: all exact-content duplicate aliases.
- `run_quality.csv`: timing, gaps, missing data, repeated values, and effective-resolution flags.
- `prepared/*.parquet`: validated analysis series plus hash/config sidecars.
- `events.parquet` and `events.csv`: detector results; CSV is capped at 100,000 rows.
- `run_summary.csv` and `threshold_sensitivity.csv`: observation and configuration summaries.
- `plots/`: full-run examples and aggregate distributions.
- `effective_config.json` and `report.md`: machine-readable settings and observed conclusions.

Rows from different detector configurations are sensitivity analyses of the same physical traces and must not be summed as independent events. Passive traces do not measure actuator delay `d_act`, interference `eta`, or filler calibration. Causes such as checkpointing, evaluation, or failure are not inferred without supporting logs.
