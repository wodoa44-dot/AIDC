#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
build_id="${1:?build id required}"; out="outputs/$build_id"
if [[ -f "$out/build_status.json" ]]; then cat "$out/build_status.json"; fi
if [[ -f "$out/pid" ]]; then
  pid="$(cat "$out/pid")"
  if kill -0 "$pid" 2>/dev/null; then echo "process running pid=$pid"; else echo "process stopped pid=$pid"; fi
fi
if [[ -f "$out/exit_code" ]]; then echo "exit_code=$(cat "$out/exit_code")"; fi
