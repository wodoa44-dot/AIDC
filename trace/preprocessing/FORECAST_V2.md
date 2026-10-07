# Forecasting data preparation v2

This extends the existing `python -m powerprep` CLI. Select
`configs/forecast_v2.yaml` explicitly. Historical v1 outputs and forecasting
experiments retain their old inputs; v2 refuses to write into the legacy output
root or adopt an unversioned existing output directory.

## Reproduce

From `/home/work/AIDC`:

```bash
PYTHONPATH=trace/preprocessing OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python -m powerprep all --config trace/preprocessing/configs/forecast_v2.yaml --output trace/preprocessing/outputs/forecast_v2_1_20260916
```

The exact same command resumes. Each native/variant file has a completion sidecar
covering source/config/schema/adapter/transformation hashes and artifact checksums.
Missing/corrupted or transformation-incompatible caches are rebuilt. Changing the
configuration requires a new output directory. Source changes detected after an
inventory require refreshing inventory before standardization. Atomic renames keep
incomplete tables out of completed artifact paths. Split/window and report stages
are deterministic and rebuild their small aggregate outputs.

Independent stages, in dependency order, use the same options:

```bash
python -m powerprep inventory   --config configs/forecast_v2.yaml --output outputs/NEW_VERSION
python -m powerprep preprocess --config configs/forecast_v2.yaml --output outputs/NEW_VERSION
python -m powerprep resolutions --config configs/forecast_v2.yaml --output outputs/NEW_VERSION
python -m powerprep windows     --config configs/forecast_v2.yaml --output outputs/NEW_VERSION
python -m powerprep report      --config configs/forecast_v2.yaml --output outputs/NEW_VERSION
# Optional, after resolutions; never implied by preprocess:
python -m powerprep detect      --config configs/forecast_v2.yaml --output outputs/NEW_VERSION
```

Run that stage-by-stage block from `trace/preprocessing`. A missing or incompatible
prerequisite is reported, not silently executed. `all` calls the stages in order;
event detection is disabled in the supplied full-data configuration.

## Contract and changes

- Native validation precedes binning and preserves order, resets, invalid rows,
  native GPU channels, original timestamps and row lineage.
- Persistent segments break at invalid observations, negative power, duplicate
  timestamps, reversals and configured large gaps. Zero power remains valid.
- Native `standardized/` includes invalid observations; native `prepared/` is the
  valid analysis view. This deliberately differs from the legacy online-rate
  prepared representation: select a named resolution variant explicitly.
- Coarser bins use left-held time integration within a segment's observed support.
  Full-bin availability and label are its right edge. No extrapolated endpoint,
  upsampling, midpoint availability or bridging of invalid gaps is allowed.
- Durations sum observed segment endpoint differences. Total span, valid duration,
  and union of supports eligible for each window configuration are distinct.
- `run_metadata.parquet` has one canonical row per run; `source_metadata.parquet`
  preserves alias declarations and every original metadata-table field in JSON.
  `metadata_evidence.jsonl` distinguishes verified, declared, inferred and unknown.
  Filename model-size tokens without established units do not become parameter counts.
- Source configurations recover the exact offline NLR model path and declared
  GPUs per node when available. Unsupported batch semantics, precision,
  accumulation, clock/cap settings and physical sensor cadence remain unknown.
- Exact hashes confirm aliases. Similar configurations do not confirm duplicates.
  Shared NLR inference acquisition groups are conservative leakage groups. Their
  within-run/global-run separation is marked unsupported when safe isolation
  cannot be established. This does not silently substitute a different protocol.
- Chronological boundaries are assigned in original row order before windows, and
  reused across resolutions. Context and target must stay within one split and
  segment. Durations must be exact resolution multiples; no silent rounding.
- Optional z-score parameters use unique training bins once per run, scoped by
  dataset/protocol/resolution; physical power stays in watts. Inverse rules persist.
- No model, retrieval index, future event feature or control-feasibility claim is made.

NLR's upstream polynomial interpolation can use future observations; online-rate
also inherits backward-fill cleaning. Aggregate-grid coverage is not raw sensor
coverage. These limitations are carried in run and resolution metadata. The
aggregate RAPL package/core accounting is inherited, not independently calibrated.

## Optional events

The v2 optional detector starts 10% crossing selection at the final recovery
episode, uses directed crossings and leaves ambiguous/censored/same-sample rise
times null. It explicitly names the post-threshold slope. Centered smoothing and
event labels are offline only. V1 definitions and old artifacts are unchanged.
Authoritative events.csv is complete, with a matching Parquet table. Distributions
are separated by dataset and detector configuration; zero-event datasets appear
in the report. No hardcoded metadata-match or event-count conclusions are used.

## Validation

```bash
PYTHONPATH=trace/preprocessing OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q trace/preprocessing/tests
PYTHONPATH=trace/preprocessing OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m powerprep all --config trace/preprocessing/configs/forecast_v2.yaml --output trace/preprocessing/outputs/forecast_v2_complete_smoke_20260916 --limit 1
PYTHONPATH=trace/preprocessing OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python trace/preprocessing/tests/verify_v2_integration.py
```

The integration verifier checks separate stages, preserved cache hashes/mtimes,
actual persisted window bounds and availability, and complete event exports.
Regression tests include complete fractional bins whose coverage differs from one
only by floating-point rounding (1 ns absolute / 1e-10 relative tolerance).

Full-run live status: `outputs/forecast_v2_1_20260916/progress.json`.
Full run logs: `/tmp/powerprep_v2_1.log` (copied into the output directory on completion).
Actual results: `dataset_summary.csv`, `resolution_manifest.csv`,
`forecast_support.csv`, `window_counts.csv`, `preprocessing_report.md`.
Schema: `data_dictionary.md` and `quality_flags.json`.

## Executed verification and current run

21 regression tests passed. The six-dataset integration completed with 5,090
windows; 23 native/resolution caches retained identical content and modification
times across separate-stage resume. Every persisted integration window was checked
against its source resolution file. Complete event export and optional detection
were verified on these small samples. No models were trained.

The final scope also includes case-insensitive discovery of 53 single-machine
`.CSV` files, which v1 did not ingest. Its GPU board-power target stays separate
from node GPU sums and NLR CPU+GPU aggregates. The source CPU package and GPU rail
channels are retained without summing overlapping components into the target.
Source text notes are retained, including contradictory model-size declarations.

`forecast_v2_20260916` and earlier smoke outputs are superseded development
artifacts, not final full-data results. They are preserved. The full final run is
`forecast_v2_1_20260916`, completed in persistent command session 68501 at low CPU priority.
The final verification passed at 2026-09-16T11:37:38Z.
The pre-existing corrected forecasting experiment (PID 9834) was not stopped,
restarted, or modified by this work.

The first completed full output `forecast_v2_complete_20260916` remains preserved.
The current `forecast_v2_1_20260916` additionally recovers within-epoch relative
time in three single-machine files whose timestamps omit hours. No missing absolute
hour is fabricated and reversals still split epochs. See timestamp_hour_known in
native rows and hour_unknown_valid_samples in quality/run metadata.

Final persisted-artifact verification and Korean handoff:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python trace/preprocessing/tests/verify_full_v2.py
```

This waits for completion and writes full_verification.json and handoff_ko.md.

## Measurement scope

Here "native" means the as-distributed source table before this package's own
transformations. For NLR it is already the supplied `01_aggregated_datasets`
representation, not independent raw sensor updates. `00_raw_datasets` logs and
acquisition settings are provenance sources; this pipeline does not rerun upstream
interpolation or claim recovered raw coverage. `03_whole-facility_profiles`
contains downstream synthetic facility profiles according to the local README and
is intentionally excluded from the six measured-data target groups. It is not
silently counted as another set of independent measured runs.

Short `mm:ss` clocks carry unknown absolute hours. Native quality tables separately
count valid observations with unknown hours. Coarse rounded duplicate timestamps
remain auditable exclusions and split segments; they are not converted into guessed
higher-frequency timestamps. This can reduce supported context lengths even when
individual power observations are finite.

## Final verified result

All six datasets completed. Source files: 2,562; exact aliases: 23; canonical runs: 2,539.
Native observations: 39,695,243; valid prepared observations: 39,693,185; exclusions: 2,058.
Window indices: 361,293. All 10,115 cache records and all 220 files contributing
windows passed final checks, including training-only normalization reconstruction.
See `outputs/forecast_v2_1_20260916/full_verification.json` and `handoff_ko.md`.
