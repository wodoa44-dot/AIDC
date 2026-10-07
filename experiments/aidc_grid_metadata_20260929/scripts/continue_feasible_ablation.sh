#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
experiment_id="${1:-$(cat "$project_dir/latest_full_id.txt")}" 
out="$project_dir/outputs/$experiment_id/feasible_ablation_v1"
mkdir -p "$out"
exec 8>"$out/scheduler.lock"
flock -n 8 || { echo "Ablation scheduler already running"; exit 0; }
export PYTHONPATH="$project_dir/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$project_dir"
while true; do
  count="$(find "$out/prediction_parts" -maxdepth 1 -name '*.parquet' 2>/dev/null | wc -l)"
  if (( count >= 130 )); then echo "Ablation complete: $count/130"; exit 0; fi
  flock "$out/ablation.lock" /usr/bin/python scripts/run_feasible_ablation.py --id "$experiment_id"
  count="$(find "$out/prediction_parts" -maxdepth 1 -name '*.parquet' 2>/dev/null | wc -l)"
  echo "Ablation checkpoint: $count/130 at $(date -u +%FT%TZ)"
  if (( count >= 130 )); then exit 0; fi
  sleep 120
done
