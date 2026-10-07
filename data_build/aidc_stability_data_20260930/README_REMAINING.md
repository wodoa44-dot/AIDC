# Follow-up execution

The original Stage4 outputs and existing data builds remain unchanged. The
follow-up artifacts are in `outputs/remaining_20260929/`. The Korean report is
`outputs/remaining_20260929/report_ko.md`.

## Network replay

ANDES 1.9.2 and its missing dependencies were installed inside `.grid_deps/`.
Generated numerical functions are in `.grid_pycode/`. The system Python's
scientific packages are reused; actual versions are recorded in the network
output's `environment.json`. No working virtual environment is required.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/run_network_replay.py --build-id NEW_BUILD_ID
```

This runs the official Kundur case with its original line trip disabled and
constant P/Q load conversion. Only the active-power load at bus 7 varies.
Each of the 32 archived runs contributes its first eligible common 10-second
window, for C0-C5. The original assumed actuator outputs are replayed; no
physical GPU, storage or UPS commands are sent. The 1,000-copy scaling is a
scenario assumption. Voltage is a simulated bus voltage and frequency is
generator center-of-inertia frequency, not a facility meter signal.

The runner refuses to overwrite an existing run. `--max-runs 1` is a smoke
run. The generated report explicitly separates benchmark evidence from
unavailable facility measurements and service-performance effects.

## Existing result audit

```bash
python scripts/audit_remaining.py --build-id NEW_AUDIT_ID
```

The audit reads CSV headers and Parquet footers across both prompt projects.
It reconciles historical top-level status with completed follow-up artifacts.
It does not refit completed forecasting experiments or certify every sample.

## Command-aligned observations

```bash
PYTHONPATH=src python scripts/analyze_phase0.py --source TRIAL_DIR --output NEW_ANALYSIS_DIR
```

Supply `*_commands.parquet` and matching `*_nvml.parquet` files using the
existing collector schema. The new analysis checks successful commands,
trial/GPU identity, baseline and plateau coverage, sample gaps, evidence
types and a sustained 90-percent response. It reports an observed response,
not a causal cap/filler effect. Matched no-treatment trials, workload progress
and service metrics are still required. The old `identify` CLI is unchanged;
this follow-up analyzer is an explicit separate entry point.

## Facility numeric channels

```bash
PYTHONPATH=src python scripts/import_facility.py --source METER_CSV --output NEW_IMPORT_DIR
```

Input is a long-format CSV or Parquet with `timestamp_utc`, `available_at_utc`,
`meter_id`, `measurement_boundary`, `quantity`, `value`, `unit`, `evidence_type`,
`clock_offset_s`, `clock_uncertainty_s` and `missing_reason`.

| Quantity | Unit | Boundary |
|---|---|---|
| pcc_active_power_w | W | PCC |
| pcc_reactive_power_var | var | PCC |
| voltage_v | V | PCC |
| frequency_hz | Hz | PCC |
| ess_soc | 1 | ESS |

Timestamps need explicit time zones. Missing values require a reason.
Simulated/synthetic records, duplicate keys and implicit unit conversions
are rejected. Declared measurement provenance is preserved, not independently
authenticated. Clock offsets are retained, not silently applied. UPS mode
needs a separate categorical source. No facility data file is fabricated.

## Verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests
```

Use a new build ID for future data builds: added source modules change the
code fingerprint, so existing cached builds must not be silently resumed.
