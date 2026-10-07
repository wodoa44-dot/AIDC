#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
build_id="${1:-full_$(date -u +%Y%m%dT%H%M%SZ)}"
out="outputs/$build_id"
mkdir -p "$out"
if [[ -f "$out/pid" ]] && kill -0 "$(cat "$out/pid")" 2>/dev/null; then
  echo "already running PID $(cat "$out/pid")"; exit 1
fi
printf '%s\n' "$(date -u +%FT%TZ)" > "$out/launch_utc"
TZ=Asia/Seoul date +%FT%T%z > "$out/launch_kst"
nohup bash -c 'cd "$1"; export PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4; python -m aidc_data build-existing --build-id "$2" > "outputs/$2/build.log" 2>&1; code=$?; printf "%s\n" "$code" > "outputs/$2/exit_code"; exit "$code"' _ "$PWD" "$build_id" >/dev/null 2>&1 &
echo $! > "$out/pid"
echo "$build_id PID $(cat "$out/pid") log $out/build.log"
