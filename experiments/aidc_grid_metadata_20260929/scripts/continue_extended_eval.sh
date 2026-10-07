#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
out="$project_dir/outputs/full_20260928T180455Z/extended_primary_eval_v1"
exec 8>"$out/scheduler.lock"
flock -n 8 || exit 0
export PYTHONPATH="$project_dir/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$project_dir"
while true; do
  /usr/bin/python scripts/evaluate_extended_primary.py
  count="$(find "$out/part_metrics" -maxdepth 1 -name '*.parquet' | wc -l)"
  echo "extended_eval=$count/150 at $(date -u +%FT%TZ)"
  if (( count >= 150 )); then exit 0; fi
  sleep 120
done
