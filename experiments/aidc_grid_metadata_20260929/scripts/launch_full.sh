#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
experiment_id="full_$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="$project_dir/outputs/$experiment_id"
mkdir -p "$output_dir"
exec 9>"$project_dir/full.lock"
flock -n 9 || { echo "A full experiment is already running" >&2; exit 1; }
printf '%s\n' "$experiment_id" > "$project_dir/latest_full_id.txt"
date -u +%FT%TZ > "$output_dir/start_utc.txt"
TZ=Asia/Seoul date +%FT%T%z > "$output_dir/start_kst.txt"
printf '%s\n' '/usr/bin/python' > "$output_dir/python.txt"
export PYTHONPATH="$project_dir/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$project_dir"
setsid nohup bash -c 'echo $$ > "$1/pid.txt"; /usr/bin/python -m aidc_exp.cli all --config "$2" --id "$3" > "$1/execution.log" 2>&1; printf "%s\n" "$?" > "$1/exit_code.txt"' _ "$output_dir" "$project_dir/configs/full.yaml" "$experiment_id" </dev/null >/dev/null 2>&1 &
pid=$!
echo "$pid" > "$output_dir/launcher_pid.txt"
for attempt in {1..20}; do
  if [[ -f "$output_dir/pid.txt" ]]; then break; fi
  sleep 0.1
done
if [[ ! -f "$output_dir/pid.txt" ]]; then echo "child failed to start" >&2; exit 1; fi
echo "$experiment_id $pid $output_dir/execution.log"
