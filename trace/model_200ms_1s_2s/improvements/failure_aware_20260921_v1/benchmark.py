#!/usr/bin/env python
"""Warm benchmark of both horizons (30 quantile models) together."""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import joblib
import numpy as np

from runtime import predict_item
import sys
HERE = Path(__file__).resolve().parent; BASE = HERE.parents[1]; sys.path.insert(0, str(BASE))
from src.core import power_features


def percentiles(values): return {f"p{x}": float(np.percentile(values, x)) for x in (50, 95, 99)}


def run_benchmark(package_path: Path, output_path: Path, trials: int = 1000) -> dict:
    start = time.perf_counter(); package = joblib.load(package_path); load_ms = (time.perf_counter()-start)*1000
    metadata = {"gpu": "__UNSEEN_GPU__", "workload": "llm_training",
                "batch": 16, "ai_model": "__UNSEEN_MODEL__"}
    values = np.linspace(1800, 2200, 25); feature_times, prediction_times, totals = [], [], []
    for _ in range(20):
        for item in package["horizons"].values():
            pf = power_features(values, 200, item["history_ms"]); predict_item(item, pf, metadata)
    for _ in range(trials):
        begin = time.perf_counter_ns()
        features = [power_features(values, 200, item["history_ms"]) for item in package["horizons"].values()]
        middle = time.perf_counter_ns()
        for feature, item in zip(features, package["horizons"].values()): predict_item(item, feature, metadata)
        end = time.perf_counter_ns()
        feature_times.append((middle-begin)/1000); prediction_times.append((end-middle)/1000); totals.append((end-begin)/1000)
    result = {"package": str(package_path), "package_bytes": package_path.stat().st_size,
        "load_ms": load_ms, "trials": trials, "quantile_models_both_horizons": 30,
        "feature_construction_us": percentiles(feature_times),
        "prediction_only_30_models_us": percentiles(prediction_times),
        "end_to_end_30_models_us": percentiles(totals), "bin_collection_delay_ms": 200,
        "model_threads": 1, "process_workers": 1,
        "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_method": "process high-water mark from resource.getrusage; not incremental model RSS",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "logical_cpu_count": os.cpu_count()},
        "note": "Local warmed benchmark; comparison with the 200 ms cadence is not a real-time guarantee."}
    output_path.write_text(json.dumps(result, indent=2)); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--trials", type=int, default=1000)
    args=parser.parse_args(); print(json.dumps(run_benchmark(Path(args.package), Path(args.output), args.trials), indent=2))


if __name__ == "__main__": main()
