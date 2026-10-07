#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
experiment_id="${1:-$(cat "$project_dir/latest_full_id.txt")}" 
output_dir="$project_dir/outputs/$experiment_id"
echo "experiment_id=$experiment_id"
if [[ -f "$output_dir/exit_code.txt" ]]; then echo "exit_code=$(cat "$output_dir/exit_code.txt")"; fi
if [[ -f "$output_dir/pid.txt" ]]; then
  pid="$(cat "$output_dir/pid.txt")"
  if kill -0 "$pid" 2>/dev/null && [[ "$(tr "\0" " " < "/proc/$pid/cmdline" 2>/dev/null || true)" == *"$experiment_id"* ]]; then
    echo "process=alive_matching_identity pid=$pid"
  else
    echo "process=not_alive_or_mismatch pid=$pid"
    if [[ ! -f "$output_dir/exit_code.txt" ]]; then echo "execution=interrupted_without_exit_code"; fi
  fi
fi
if [[ -f "$output_dir/heartbeat_utc.txt" ]]; then echo "heartbeat=$(cat "$output_dir/heartbeat_utc.txt")"; fi
if [[ -d "$output_dir/prediction_parts" ]]; then
  count="$(find "$output_dir/prediction_parts" -maxdepth 1 -name '*.parquet' | wc -l)"
  echo "prediction_parts=$count"
fi
if [[ -f "$output_dir/status.json" ]]; then cat "$output_dir/status.json"; fi
if [[ -f "$output_dir/execution.log" ]]; then tail -n 12 "$output_dir/execution.log"; fi
