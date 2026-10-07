# AIDC grid/metadata offline experiment

All new code, configuration, cache, models and outputs live in this project. Sources under `../../trace` are read only. Python: `/usr/bin/python`.

```bash
cd /home/work/AIDC/experiments/aidc_grid_metadata_20260929
PYTHONPATH=src /usr/bin/python -m aidc_exp.cli all --config configs/smoke.yaml --id smoke_check
bash scripts/launch_full.sh
bash scripts/status.sh
bash scripts/resume.sh
```

Primary E0 target is eight GPU power channels summed per highres node, CPU excluded. Bins are right edged 200 ms sample means, with sample count coverage; sensor averaging and collection latency are unknown. Mean is arithmetic, not time weighted. The 32 unique SHA256 sources are included. SHA256 aliases are excluded. Different files are treated as separate run groups, although independence of collection sessions has not been established.

NLR aggregate interpolation prevents a causal evaluation from aggregate files. No validated grid dynamics case, GPU cap performance model or measured actuator model has been verified. E1 control is an assumption based power balance proxy; E2/E3 are blocked. Do not interpret E1 as measured grid stability.

The full experiment is computationally large on one visible CPU. Task parquet files permit resuming completed fits when code, config and manifest fingerprints agree. Inner screening uses three quantiles and a separate full grid score. Controlled metadata comparisons hold P2 capacity fixed. The independent tuning track, event classifier, event matching and dynamic grid response remain pending implementation and are never labelled completed.

## Current full run (2026-09-28 UTC)

`full_20260928T180455Z` is launched with nohup (wrapper PID 2231, worker PID 2240). Audit and preprocessing finished: 32 unique highres runs, 144,000 completed-bin positions, 60,264 issuance positions with uninterrupted 30 s history before horizon filtering. Training is in progress. Root `/home/work/AIDC` is not a Git repository, so Git commit and dirty diff hashes are unavailable; the run stores a code hash and manifest hash instead.

An official dynamic benchmark candidate is [ANDES Kundur two-area example](https://docs.andes.app/en/stable/tutorials/02-first-simulation.html). ANDES is [GPL v3](https://github.com/CURENT/andes). Installation in this environment failed due DNS failures while fetching dependencies, and the case license and load-injection interface have not been verified. No E2 result is produced.

The initial smoke report at `outputs/smoke_corrected_20260928` contains only 3 runs at 1 s and 3 quantiles. It is a pipeline check, not the full study. The experimental E1 control has assumed, not identified, delay, capacity, efficiency, and energy bounds. A controlled GPU response model is unavailable, so C1 is equal to C0. The full output status lists event classification, metadata ablation and independent tuning as not implemented.


## Continuation (2026-09-29 00:13 UTC)

The original worker stopped without an exit-code file after writing 210 of 750 planned prediction parts. The same experiment was resumed from matching fingerprints under a detached wrapper (PID 3616; worker PID 3646). Check `bash scripts/status.sh full_20260928T180455Z` for the current process and part count. If it stops, run `bash scripts/resume_detached.sh full_20260928T180455Z`; the script refuses a duplicate live process, uses the project lock, and appends to the execution log. A `running` stage in status.json alone is not proof that the worker is alive.

The full E0 comparison and E1 rollout are incomplete until their stages finish. The current 210 parts cover fold 0 completely and fold 1 horizons 0.2–0.8 s; they are not a full-run result. The 9 existing core tests passed on continuation. The controlled track and assumed E1 proxy are the only implemented branches of the broader plan; independent tuning, event classification, metadata ablation, and E2/E3 remain unavailable as recorded in status.json.


## Feasible follow-up tracks (2026-09-29 UTC)

The source code under `src/` remains frozen for the checkpointed full run. New, independently rerunnable studies live under `scripts/` and write to separate output directories. Saved metrics, figure data, PNG and PDF files are under `outputs/full_20260928T180455Z/feasible_followup_v1/`.

- `run_feasible_events_v2.py`: all five outer folds, 1/2 s up/down path-excursion binary forecasts, P2/P4. Training-only thresholds and inner false-alarm-budget selection. `match_feasible_events.py` evaluates one-to-one crossing/alert episodes. The earlier `feasible_events_v1` is explicitly superseded because its event label used adjacent-bin ramp rather than path excursion.
- `run_feasible_ablation.py`: full15 P4 metadata feature deletion and five run-level train-metadata shuffles at the P2-selected capacity. The fold-0 P4 reference was verified bit-for-bit against the original stored P4 predictions. `continue_feasible_ablation.sh` resumes other folds when selections appear.
- `reconstruct_nlr_causal_pilot.py`: five one-node NLR raw CPU/GPU logs, last sample at or before each 200 ms right edge, age <=0.4 s, no interpolation. Timestamp as availability is an explicit unverified assumption. `run_nlr_pilot_predict.py` provides a separate compact3 P0/P2 study, not pooled with highres.
- `run_feasible_control_sensitivity.py`: assumed E1 capacity/delay sensitivity on paired fold-0 P2/P4 predictions. `run_frequency_exploratory.py` applies published single-area inertia/governor parameters to an assumed 1000-copy, 100 MVA scenario. It is frequency-only exploratory and not an E2 network or E3 facility validation.
- `run_independent_primary.py`: 24-candidate independent P2/P3/P4 tuning at primary 1/2 s, with per-candidate checkpoints. `evaluate_extended_primary.py` streams tail, interval and calibration metrics from available controlled-track prediction parts; its scheduler follows new files.

Background processes are tracked by each track's `pid.txt`, `execution.log`, `status.json` and `exit_code.txt` where applicable. `watch_full.sh` can restart the unchanged full worker after an interrupted exit, limited to three attempts. Full E0, ablation and independent tuning remain in progress until their status files say complete. Rebuild the follow-up verification and figures with:

```bash
PYTHONPATH=src /usr/bin/python scripts/summarize_feasible.py
/usr/bin/python scripts/report_feasible.py  # regenerate figures and figure-data CSVs
PYTHONPATH=src:scripts /usr/bin/python -m pytest -q tests/test_core.py tests/test_feasible.py
```
