#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# exec preserves the nohup PID and forwards TERM directly to the sequential runner.
exec /usr/bin/python -u "$root/pipeline.py" "$@"
