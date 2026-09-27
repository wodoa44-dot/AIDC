#!/usr/bin/env python
"""Continuous four-run replay and batch/export equivalence verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
CORRECTED = ROOT / "trace/predictability_meta/outputs/corrected_full_20260915"
sys.path.insert(0, str(BASE))
from src.core import rearrange
from runtime import ForecastService, predict_item


def run_replay(package_path: Path, output_path: Path, max_rows_per_run: int | None = None) -> dict:
    package = joblib.load(package_path)
    held = pd.read_csv(CORRECTED / "reused_holdout_runs.csv")
    cells, powers, lookups = {}, {}, {}
    for horizon in (1000, 2000):
        cells[horizon] = pd.read_parquet(CORRECTED / f"features/meta_200_{horizon}.parquet")
        z = np.load(CORRECTED / f"features/power_200_{horizon}.npz")
        powers[horizon] = {k: z[k] for k in z.files}
        lookups[horizon] = {(row.run_id, round(row.issue_s, 7)): int(index)
                            for index, row in cells[horizon][["run_id", "issue_s"]].iterrows()}
    rows, max_difference, compared = [], 0.0, 0
    for run_id in held.run_id:
        service = ForecastService(package_path)
        stream = cells[1000][cells[1000].run_id.eq(run_id)].sort_values(
            ["segment_id", "subsegment_id", "issue_s"], kind="mergesort")
        if max_rows_per_run is not None:
            stream = stream.iloc[:max_rows_per_run]
        previous_group = None
        for row in stream.itertuples():
            group = (row.segment_id, row.subsegment_id)
            if group != previous_group:
                service.reset(); previous_group = group
            metadata = {"gpu": row.gpu, "workload": row.workload,
                        "batch": row.batch, "ai_model": row.ai_model}
            response = service.update(round(row.issue_s * 1000), row.current, metadata)
            if response["status"] != "ready":
                continue
            for forecast in response["forecasts"]:
                horizon = forecast["horizon_ms"]; meta = cells[horizon]
                index = lookups[horizon].get((run_id, round(row.issue_s, 7)))
                if index is None:
                    continue
                item = package["horizons"][str(horizon)]
                raw, direct, _ = predict_item(item, powers[horizon][str(item["history_ms"])][index], metadata)
                quantiles = list(map(float, item["quantiles"])); requested = (.1, .5, .9)
                online = np.array(list(forecast["quantiles"].values()))
                expected = direct[0, [quantiles.index(q) for q in requested]]
                difference = float(np.max(np.abs(online - expected)))
                max_difference = max(max_difference, difference); compared += 1
                rows.append({"run_id": run_id, "issue_s": row.issue_s, "horizon_ms": horizon,
                    "q10": online[0], "q50": online[1], "q90": online[2],
                    "raw_q50": raw[0, quantiles.index(.5)], "batch_replay_max_abs_diff_w": difference,
                    "full_quantile_count_computed": forecast["full_quantile_count_computed"]})
    if not rows:
        raise RuntimeError("replay produced no comparable predictions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    result = {"passed": max_difference <= 1e-10, "runs": int(pd.DataFrame(rows).run_id.nunique()),
              "n_compared": compared, "max_abs_difference_w": max_difference,
              "full_quantiles_computed_per_horizon": 15, "output": str(output_path)}
    if not result["passed"]:
        raise AssertionError(result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--max-rows-per-run", type=int)
    args = parser.parse_args()
    print(json.dumps(run_replay(Path(args.package), Path(args.output), args.max_rows_per_run), indent=2))


if __name__ == "__main__": main()
