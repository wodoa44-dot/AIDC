from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import QUANTILES, crps_quantile, write_json

HERE = Path(__file__).resolve().parent


def verify(out):
    out = Path(out)
    target = pd.read_parquet(out / "targets.parquet")
    predictions = pd.read_parquet(out / "per_target_predictions.parquet")
    split = pd.read_csv(out / "split_manifest.csv")
    per = pd.read_csv(out / "per_run_metrics.csv")
    aggregate = pd.read_csv(out / "aggregate_metrics.csv")
    qcols = [f"q_{q:g}" for q in QUANTILES]
    grouped = predictions.groupby(["run_id", "issue_s", "horizon_ms"])
    checks = {
        "target_horizon_s": bool(np.allclose(target.target_end_s-target.issue_s, target.horizon_ms/1000)),
        "target_start_before_end": bool((target.target_start_s < target.target_end_s).all()),
        "no_train_test_source_overlap": not bool(set(split.loc[split.split == "train", "group_id"]) &
                                                 set(split.loc[split.split == "test", "group_id"])),
        "same_test_rows_all_conditions": bool(grouped.condition.nunique().eq(3).all()),
        "same_test_truth_all_conditions": bool(grouped.y.nunique().eq(1).all()),
        "no_quantile_crossing_after_correction": bool((np.diff(predictions[qcols].to_numpy(), axis=1) >= -1e-8).all()),
        "test_groups_disjoint_from_train": not bool(set(predictions.group_id) &
            set(split.loc[split.split == "train", "group_id"])),
    }
    sample = predictions.iloc[::max(1, len(predictions)//1000)]
    checks["saved_quantile_score_matches"] = bool(np.allclose(
        sample.crps_w, crps_quantile(sample.y.to_numpy(), sample[qcols].to_numpy()), atol=1e-8))
    for row in aggregate.itertuples():
        selected = per[(per.condition == row.condition) & (per.horizon_ms == row.horizon_ms)]
        checks[f"aggregate_{row.condition}"] = bool(np.isclose(selected.crps_w.mean(), row.crps_w))
    checks["all_passed"] = all(checks.values())
    write_json(out / "verification.json", checks)
    if not checks["all_passed"]:
        raise AssertionError(checks)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/actual_20260926"))
    print(verify(parser.parse_args().out))
