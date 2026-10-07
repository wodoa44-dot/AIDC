# AIDC stability data build

This independent project builds D1/D2 from the raw highres eight-GPU node traces. It keeps prior `trace/` and `experiments/` read only. The original Korean execution prompt is [PROMPT_DATA_BUILD_20260930.md](PROMPT_DATA_BUILD_20260930.md).

```bash
cd /home/work/AIDC/data_build/aidc_stability_data_20260930
PYTHONPATH=src python -m aidc_data audit --build-id demo
PYTHONPATH=src python -m aidc_data build-existing --build-id demo --smoke-runs 2
PYTHONPATH=src python -m aidc_data validate --build-id demo
PYTHONPATH=src python -m aidc_data prepare-phase0 --build-id demo
PYTHONPATH=src python -m aidc_data simulate-grid --build-id demo
PYTHONPATH=src python -m aidc_data report --build-id demo
scripts/launch_existing.sh full_$(date -u +%Y%m%dT%H%M%SZ)
scripts/status.sh BUILD_ID
scripts/resume.sh BUILD_ID
```

The default `all`/`build-existing` path does not control a GPU. `collect --live --config PATH` requires an authorized GPU UUID, workload and runtime configuration. The current live collector is passive NVML polling; active cap trials need an explicit harness and paired performance logs. Published NLR aggregates are inventory only because they contain future interpolation. D4 is frequency only under scenario assumptions. Outputs include `build_status.json`, `validation.json`, `dataset_manifest.json`, `data_dictionary.md`, and Korean dataset card/report.

Set `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`; all scripts do so. Cache resume requires matching source manifest, config and code hashes. Source row SHA hashes are taken from the upstream manifest; not every large raw file is rehashed during a build. Config snapshot is the YAML in this project and its SHA is in build status. The external source files stay in place.

## Additional raw NLR track and review artifacts

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python scripts/build_nlr_raw.py --build-id main_20260929
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python scripts/finalize_nlr_raw.py --build-id main_20260929
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python scripts/export_resolutions.py main_20260929
python scripts/make_review_plots.py main_20260929
python scripts/example_stage4_loader.py --build-id main_20260929 --fold 0
python scripts/annotate_grid_provenance.py main_20260929
python scripts/finalize_status.py main_20260929
```

The NLR raw track parses 302 native 4-GPU NVML node log files directly. It is stored under `outputs/main_20260929/nlr_raw/` with per-file partitions and its own manifest, quality audit and grouped split. NLR source time zones and independent sensor refresh remain unverified. Run the finalizer after the full NLR build completes. Additional 50/100/500/1000 ms bins are derived from highres raw timestamps and do not establish independent sensor bandwidth.
