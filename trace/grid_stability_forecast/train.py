from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from core import QUANTILES, crps_quantile, history_features, static_features, write_json

HERE = Path(__file__).resolve().parent


def split_groups(frame):
    groups = frame[["group_id", "gpu_model", "workload_type"]].drop_duplicates()
    if groups.group_id.duplicated().any():
        raise ValueError("source group spans more than one condition")
    groups = groups.sort_values("group_id")
    groups["split"] = "train"
    for _, cohort in groups.groupby(["gpu_model", "workload_type"]):
        indices = cohort.index.to_numpy()
        groups.loc[indices[::4], "split"] = "test"
    assert not set(groups.loc[groups.split == "train", "group_id"]) & set(groups.loc[groups.split == "test", "group_id"])
    return groups


def sample_indices(frame, eligible, cap=1500):
    parts = []
    for _, group in frame.iloc[eligible].groupby("run_id", sort=True):
        idx = group.index.to_numpy()
        parts.append(idx[np.linspace(0, len(idx)-1, min(cap, len(idx)), dtype=int)])
    return np.sort(np.concatenate(parts))


def run(out, compact=False, horizon_ms=1000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "targets.parquet"
    frame = pd.read_parquet(data_path)
    frame = frame.loc[frame.horizon_ms == horizon_ms].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no +{horizon_ms} ms targets")
    split = split_groups(frame)
    split.to_csv(out / "split_manifest.csv", index=False)
    by_group = split.set_index("group_id").split
    labels = frame.group_id.map(by_group)
    train_all = np.flatnonzero(labels == "train")
    test = np.flatnonzero(labels == "test")
    train = sample_indices(frame, train_all)
    assert len(set(frame.iloc[train].run_id) & set(frame.iloc[test].run_id)) == 0
    sample = frame.iloc[train].groupby("run_id").size()
    weights = frame.iloc[train].run_id.map(1 / sample).to_numpy() * len(sample)
    power = np.vstack([history_features(x) for x in frame.history])
    s1_train, s1_test = static_features(frame, train, test,
        columns=("gpu_model", "workload_type", "model_name", "batch_size"))
    static_train, static_test = static_features(frame, train, test)
    qgrid = np.array([.1, .5, .9]) if compact else QUANTILES
    configs = {"history_only": (power[train], power[test]),
               "history_static_s1": (np.column_stack([power[train], s1_train]),
                                     np.column_stack([power[test], s1_test])),
               "history_static_s2": (np.column_stack([power[train], static_train]),
                                     np.column_stack([power[test], static_test]))}
    predictions = []
    parameter_rows = []
    started = time.time()
    for condition, (xtrain, xtest) in configs.items():
        raw = []
        for q in qgrid:
            params = dict(objective="quantile", alpha=float(q), n_estimators=100, num_leaves=15,
                          max_depth=-1, learning_rate=.07, min_child_samples=30, reg_alpha=0., reg_lambda=0.,
                          colsample_bytree=1., subsample=1., subsample_freq=0, random_state=42,
                          n_jobs=1, verbosity=-1)
            model = lgb.LGBMRegressor(**params)
            model.fit(xtrain, frame.y.to_numpy()[train], sample_weight=weights)
            raw.append(model.predict(xtest))
            parameter_rows.append(dict(outer_fold=1, target="power_w", horizon_ms=horizon_ms,
                feature_condition=condition, quantile=q, seed=42, tree_count=model.booster_.num_trees(),
                inner_score=np.nan, parameter_source="fixed historical lightweight setting; pilot, no inner selection",
                **{k: v for k, v in params.items() if k != "alpha"}))
        pred = np.column_stack(raw)
        crossing = np.mean(np.any(np.diff(pred, axis=1) < 0, axis=1))
        pred = np.sort(pred, axis=1)
        values = frame.iloc[test].drop(columns=["history"]).copy()
        values["condition"] = condition
        values["fold"] = 1
        values["seed"] = 42
        values["assessment"] = "new_grouped_pilot"
        values["crps_w"] = crps_quantile(values.y.to_numpy(), pred, qgrid)
        values["crossing_fraction_before_correction"] = crossing
        for j, q in enumerate(qgrid):
            values[f"q_{q:g}"] = pred[:, j]
        values["q50"] = pred[:, np.where(qgrid == .5)[0][0]]
        values["absolute_error_w"] = np.abs(values.y - values.q50)
        values["transition"] = np.where(values.delta_w >= 100, "rise", np.where(values.delta_w <= -100, "drop", "steady"))
        predictions.append(values)
    output = pd.concat(predictions, ignore_index=True)
    output.to_parquet(out / "per_target_predictions.parquet", index=False)
    pd.DataFrame(parameter_rows).to_csv(out / "selected_hyperparameters.csv", index=False)
    config = {"assessment": "new_grouped_pilot", "horizons_ms": [horizon_ms], "history_ms": 5000,
              "quantiles": qgrid.tolist(), "quantile_score": "2*trapezoid(pinball,[0,q,1]); endpoint held",
              "max_rows_per_run": 1500, "run_weight": "equal", "seed": 42,
              "selection": "fixed historical lightweight settings; no new hyperparameter search",
              "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "train_rows": len(train), "test_rows": len(test), "train_runs": int(frame.iloc[train].run_id.nunique()),
              "test_runs": int(frame.iloc[test].run_id.nunique()), "elapsed_s": time.time()-started,
              "dynamic_metadata": "unavailable in this pilot", "actuation": "dry-run only"}
    write_json(out / "resolved_config.json", config)
    write_json(out / "candidate_manifest.json", {"candidates": ["fixed_h5000_t100_l15"],
        "status": "pilot; requested 24-candidate grouped search not executed"})
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/actual_20260926"))
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--horizon-ms", type=int, default=1000)
    args = parser.parse_args()
    run(args.out, args.compact, args.horizon_ms)
