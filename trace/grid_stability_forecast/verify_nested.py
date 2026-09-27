"""Validate staged grouped search and outer stored predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import QUANTILES, crps_quantile, write_json

HERE = Path(__file__).resolve().parent


def verify(out: Path, source: Path):
    out, source = Path(out), Path(source)
    candidate = json.loads((out / "candidate_manifest.json").read_text())
    selection = json.loads((out / "selection.json").read_text())
    inner = pd.read_csv(out / "inner_folds.csv")
    outer = pd.read_csv(source / "split_manifest.csv").set_index("group_id").split
    screen = pd.read_csv(out / "screening_metrics.csv")
    full = pd.read_csv(out / "full15_inner_metrics.csv")
    predictions = pd.read_parquet(out / "per_target_predictions.parquet")
    per = pd.read_csv(out / "per_run_metrics.csv")
    aggregate = pd.read_csv(out / "aggregate_metrics.csv")
    checks = {}
    checks["24_unique_common_candidates"] = bool(len(candidate["candidates"]) == 24 and
        len({c["candidate_id"] for c in candidate["candidates"]}) == 24)
    checks["three_disjoint_inner_folds"] = bool(inner.run_id.nunique() == len(inner) and
        set(inner.inner_fold) == {0,1,2} and inner.group_id.map(outer).eq("train").all())
    checks["screen_all_candidates_both_conditions"] = bool(
        screen.groupby("condition").candidate_id.nunique().to_dict() ==
        {"history_only":24,"history_static_s2":24})
    checks["full15_shortlist_six_each"] = bool(
        full.groupby("condition").candidate_id.nunique().to_dict() ==
        {"history_only":6,"history_static_s2":6})
    checks["selected_inner_minimum"] = bool(all(
        np.isclose(r["inner_run_balanced_crps_w"],
                   full.loc[(full.condition == c)&(full.candidate_id == r["candidate_id"]),
                            "inner_run_balanced_crps_w"].iloc[0]) and
        np.isclose(r["inner_run_balanced_crps_w"],
                   full.loc[full.condition == c,"inner_run_balanced_crps_w"].min())
        for c,r in selection.items()))
    expected_conditions = {"history_only_tuned","history_static_s2_controlled",
                           "history_static_s2_tuned"}
    checks["three_paired_outer_conditions"] = bool(
        set(predictions.condition) == expected_conditions and
        predictions.groupby(["run_id","issue_s","seed"]).condition.nunique().eq(3).all())
    checks["three_outer_seeds"] = bool(set(predictions.seed) == {17,42,2026})
    hp = pd.read_csv(out / "selected_hyperparameters.csv")
    fixed = hp[hp.feature_condition == "history_only_tuned"].set_index(["seed","quantile"])
    controlled = hp[hp.feature_condition == "history_static_s2_controlled"].set_index(["seed","quantile"])
    capacity = ["candidate_id","selected_trees","fitted_trees","n_estimators",
                "learning_rate","num_leaves","min_child_samples","reg_lambda",
                "reg_alpha","subsample","subsample_freq","colsample_bytree"]
    checks["controlled_capacity_identical"] = bool(
        fixed.index.equals(controlled.index) and
        fixed[capacity].equals(controlled[capacity]))
    sample = pd.read_parquet(out / "outer_sample_manifest.parquet")
    checks["outer_samples_train_only_and_equal_budget"] = bool(
        sample.groupby("seed").size().eq(22000).all() and
        sample.run_id.nunique() == 22 and
        not set(sample.run_id) & set(predictions.run_id) and
        sample.groupby(["seed","run_id"]).size().eq(1000).all())
    checks["ten_outer_test_runs"] = bool(predictions.run_id.nunique() == 10 and
        not set(predictions.run_id) & set(inner.run_id))
    qcols = [f"q_{q:g}" for q in QUANTILES]
    q = predictions[qcols].to_numpy()
    checks["quantiles_ordered"] = bool(np.all(np.diff(q,axis=1)>=-1e-8))
    checks["score_recomputed"] = bool(np.allclose(
        crps_quantile(predictions.y.to_numpy(),q),
        predictions.crps_w.to_numpy(),atol=1e-7))
    rebuilt = predictions.groupby(["condition","seed","run_id"]).crps_w.mean()
    recorded = per.set_index(["condition","seed","run_id"]).crps_w
    checks["per_run_reconciles"] = bool(np.allclose(rebuilt.loc[recorded.index],recorded))
    checks["aggregate_reconciles"] = bool(all(np.isclose(
        per.loc[per.condition == row.condition,"crps_w"].mean(), row.crps_w)
        for row in aggregate.itertuples()))
    checks["all_passed"] = all(checks.values())
    write_json(out / "verification.json", checks)
    if not checks["all_passed"]:
        raise AssertionError(checks)
    return checks


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=HERE/"results/nested_select_20260926")
    parser.add_argument("--source",type=Path,default=HERE/"results/actual_20260926")
    args=parser.parse_args()
    print(json.dumps(verify(args.out,args.source),indent=2))
