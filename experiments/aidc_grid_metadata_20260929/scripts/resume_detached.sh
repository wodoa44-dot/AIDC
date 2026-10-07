#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
experiment_id="${1:-$(cat "$project_dir/latest_full_id.txt")}" 
output_dir="$project_dir/outputs/$experiment_id"
[[ -f "$output_dir/fingerprint.json" ]] || { echo "Unknown experiment: $experiment_id" >&2; exit 1; }
if [[ -f "$output_dir/pid.txt" ]]; then
  old_pid="$(cat "$output_dir/pid.txt")"
  if kill -0 "$old_pid" 2>/dev/null && [[ "$(tr '\0' ' ' < "/proc/$old_pid/cmdline" 2>/dev/null || true)" == *"$experiment_id"* ]]; then
    echo "Already running: pid=$old_pid"; exit 0
  fi
fi
cd "$project_dir"
setsid nohup bash -c '
  exec 9>"$1/full.lock"
  flock -n 9 || exit 75
  out="$1/outputs/$2"
  printf "%s\n" "$$" > "$out/pid.txt"
  rm -f "$out/exit_code.txt"
  export PYTHONPATH="$1/src"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  set +e
  /usr/bin/python -m aidc_exp.cli resume --config "$1/configs/full.yaml" --id "$2" >> "$out/execution.log" 2>&1
  rc=$?
  printf "%s\n" "$rc" > "$out/exit_code.txt"
  exit "$rc"
' _ "$project_dir" "$experiment_id" </dev/null >/dev/null 2>&1 &
launcher_pid=$!
for attempt in {1..30}; do
  if [[ -f "$output_dir/pid.txt" ]] && [[ "$(cat "$output_dir/pid.txt")" == "$launcher_pid" ]]; then break; fi
  sleep 0.1
done
if ! kill -0 "$launcher_pid" 2>/dev/null; then
  echo "Resume failed to start; inspect $output_dir/execution.log" >&2; exit 1
fi
echo "experiment_id=$experiment_id pid=$launcher_pid log=$output_dir/execution.log"
