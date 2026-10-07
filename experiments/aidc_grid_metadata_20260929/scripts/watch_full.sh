#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
experiment_id="${1:-$(cat "$project_dir/latest_full_id.txt")}" 
out="$project_dir/outputs/$experiment_id"
exec 8>"$out/watch.lock"
flock -n 8 || { echo "watcher already running"; exit 0; }
restarts=0
while true; do
  if [[ -f "$out/exit_code.txt" ]]; then
    echo "worker exit_code=$(cat "$out/exit_code.txt") at $(date -u +%FT%TZ)"; exit 0
  fi
  alive=0
  if [[ -f "$out/pid.txt" ]]; then
    pid="$(cat "$out/pid.txt")"
    if kill -0 "$pid" 2>/dev/null && [[ "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)" == *"$experiment_id"* ]]; then alive=1; fi
  fi
  if (( alive == 0 )); then
    if (( restarts >= 3 )); then echo "restart limit reached at $(date -u +%FT%TZ)"; exit 1; fi
    echo "resuming interrupted worker at $(date -u +%FT%TZ)"
    bash "$project_dir/scripts/resume_detached.sh" "$experiment_id"
    restarts=$((restarts+1))
  fi
  sleep 60
done
