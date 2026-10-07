#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
experiment_id="${1:-$(cat "$project_dir/latest_full_id.txt")}" 
export PYTHONPATH="$project_dir/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec /usr/bin/python -m aidc_exp.cli resume --config "$project_dir/configs/full.yaml" --id "$experiment_id"
