from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core import QUANTILES, crps_quantile, history_features, static_features
from train import sample_indices

HERE = Path(__file__).resolve().parent


def run(out):
    out = Path(out)
    frame = pd.read_parquet(out / "targets.parquet")
    frame = frame[frame.horizon_ms == 1000].reset_index(drop=True)
    split = pd.read_csv(out / "split_manifest.csv")
    split_map = split.set_index("group_id").split
    test = np.flatnonzero(frame.group_id.map(split_map) == "test")
    training = split[split.split == "train"]
    cohorts = [g.group_id.tolist() for _, g in training.groupby(["gpu_model", "workload_type"], sort=True)]
    power = np.vstack([history_features(x) for x in frame.history])
    y = frame.y.to_numpy(float)
    results = []
    for requested in (4, 8, 16, len(training)):
        chosen = []
        if requested == len(training):
            chosen = training.group_id.tolist()
        else:
            per = requested // len(cohorts)
            chosen = [group for cohort in cohorts for group in cohort[:per]]
        eligible = np.flatnonzero(frame.group_id.isin(chosen))
        ids = sample_indices(frame, eligible, cap=750)
        counts = frame.iloc[ids].groupby("run_id").size()
        weights = frame.iloc[ids].run_id.map(1 / counts).to_numpy() * len(counts)
        s_train, s_test = static_features(frame, ids, test)
        matrices = {"history_only": (power[ids], power[test]),
                    "history_static_s2": (np.column_stack([power[ids], s_train]),
                                          np.column_stack([power[test], s_test]))}
        for condition, (xtrain, xtest) in matrices.items():
            raw = []
            for q in QUANTILES:
                model = lgb.LGBMRegressor(objective="quantile", alpha=float(q), n_estimators=100,
                    num_leaves=15, learning_rate=.07, min_child_samples=30, random_state=42,
                    n_jobs=1, verbosity=-1)
                model.fit(xtrain, y[ids], sample_weight=weights)
                raw.append(model.predict(xtest))
            pred = np.sort(np.column_stack(raw), axis=1)
            scored = frame.iloc[test][["run_id", "delta_w"]].copy()
            scored["crps_w"] = crps_quantile(y[test], pred)
            scored["absolute_error_w"] = np.abs(y[test] - pred[:, 7])
            scored["transition"] = np.abs(scored.delta_w) >= 100
            run_crps = scored.groupby("run_id").crps_w.mean()
            transition = scored[scored.transition].groupby("run_id").absolute_error_w.quantile(.95)
            results.append(dict(requested_train_runs=requested, actual_train_runs=len(chosen),
                train_rows=len(ids), test_runs=scored.run_id.nunique(), condition=condition,
                crps_w=run_crps.mean(), transition_p95_w=transition.mean(),
                score_grid="full15", seed=42, rows_per_train_run=750,
                assessment="fixed-setting independent-run pilot"))
        pd.DataFrame(results).to_csv(out / "learning_curve.csv", index=False)
        print("completed", requested, "runs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/actual_20260926"))
    run(parser.parse_args().out)
