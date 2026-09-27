#!/usr/bin/env python
"""Read completed-bin observations as JSON lines and emit forecast JSON lines."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.online import ForecastService

ap = argparse.ArgumentParser()
ap.add_argument("--package", required=True)
args = ap.parse_args(); service = ForecastService(args.package)
for line in sys.stdin:
    row = json.loads(line)
    print(json.dumps(service.update(row["timestamp_ms"], row["power_w"], row.get("metadata", {}))))
