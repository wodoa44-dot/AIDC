"""Detailed calibration and transition metrics for unseen-cohort predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def run(out: Path):
    out = Path(out)
    columns = ["held_out_cohort","run_id","seed","condition","y","q_0.1","q_0.9",
               "absolute_error_w","crps_w","transition"]
    frame = pd.read_parquet(out/"per_target_predictions.parquet",columns=columns)
    frame["covered_80"] = (frame.y >= frame["q_0.1"]) & (frame.y <= frame["q_0.9"])
    frame["width_80_w"] = frame["q_0.9"] - frame["q_0.1"]
    keys = ["held_out_cohort","run_id","seed","condition"]
    calibration = frame.groupby(keys).agg(n_targets=("y","size"),
        coverage_80=("covered_80","mean"), width_80_w=("width_80_w","mean")).reset_index()
    calibration.to_csv(out/"calibration_metrics.csv",index=False)
    transition = frame.groupby(keys+["transition"]).agg(n_targets=("y","size"),
        crps_w=("crps_w","mean"), mae_w=("absolute_error_w","mean"),
        p95_w=("absolute_error_w",lambda x:x.quantile(.95)),
        p99_w=("absolute_error_w",lambda x:x.quantile(.99))).reset_index()
    transition.to_csv(out/"transition_metrics.csv",index=False)
    tail = frame.groupby(keys).agg(error_p95_w=("absolute_error_w",lambda x:x.quantile(.95)),
        error_p99_w=("absolute_error_w",lambda x:x.quantile(.99)),
        max_error_w=("absolute_error_w","max"),
        worst_1pct_mean_w=("absolute_error_w",
            lambda x:x.nlargest(max(1,int(np.ceil(len(x)*.01)))).mean())).reset_index()
    tail.to_csv(out/"tail_metrics.csv",index=False)
    summary = calibration.groupby("condition").agg(coverage_80=("coverage_80","mean"),
        width_80_w=("width_80_w","mean")).reset_index()
    summary.to_csv(out/"calibration_summary.csv",index=False)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,
        default=HERE/"results/unseen_condition_20260926")
    print(run(parser.parse_args().out).to_string(index=False))
