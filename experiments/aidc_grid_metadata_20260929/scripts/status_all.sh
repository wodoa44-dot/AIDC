#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
experiment_id="${1:-$(cat "$project_dir/latest_full_id.txt")}" 
main="$project_dir/outputs/$experiment_id"
echo "experiment_id=$experiment_id"
for entry in "full:$main:750:prediction_parts" "events:$main/feasible_events_v2:40:prediction_parts" "ablation:$main/feasible_ablation_v1:130:prediction_parts" "independent:$main/independent_primary_v1:90:prediction_parts" "extended_eval:$main/extended_primary_eval_v1:150:part_metrics" "nlr:$project_dir/outputs/nlr_raw_causal_pilot_v1/forecast_compact3_v1:10:prediction_parts"; do
  IFS=: read -r name dir total child <<< "$entry"
  count=0
  if [[ -d "$dir/$child" ]]; then count="$(find "$dir/$child" -maxdepth 1 -name '*.parquet' | wc -l)"; fi
  process=none
  if [[ -f "$dir/pid.txt" ]]; then
    pid="$(cat "$dir/pid.txt")"
    if kill -0 "$pid" 2>/dev/null && [[ "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)" == *"$experiment_id"* || "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)" == *"run_nlr_pilot_predict.py"* ]]; then process="alive($pid)"; else process="stopped($pid)"; fi
  fi
  code=pending
  if [[ -f "$dir/exit_code.txt" ]]; then code="$(cat "$dir/exit_code.txt")"; fi
  extra=""
  if [[ "$name" == "independent" && -d "$dir/screens" ]]; then
    shopt -s nullglob
    screen_files=("$dir/screens"/*.csv)
    shopt -u nullglob
    if (( ${#screen_files[@]} )); then
      screened="$(awk 'FNR>1 {n++} END {print n+0}' "${screen_files[@]}")"
      extra=" screened=$screened/720"
    fi
  fi
  if [[ -f "$dir/scheduler_pid.txt" ]]; then
    scheduler_pid="$(cat "$dir/scheduler_pid.txt")"
    if kill -0 "$scheduler_pid" 2>/dev/null; then extra="$extra scheduler=alive($scheduler_pid)"; fi
  fi
  echo "$name=$count/$total process=$process exit=$code$extra"
done
