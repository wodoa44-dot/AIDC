"""Fixed-setting, leave-one-static-feature-out assessment for the +1 s pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core import QUANTILES, STATIC, crps_quantile, history_features, static_features, write_json
from train import sample_indices

HERE = Path(__file__).resolve().parent


def run(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "resolved_config.json").read_text())
    target_path = source / "targets.parquet"
    digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    if digest != config["source_sha256"] or config["horizons_ms"] != [1000]:
        raise ValueError("source data or horizon differs from the saved pilot")
    frame = pd.read_parquet(target_path)
    frame = frame.loc[frame.horizon_ms == 1000].reset_index(drop=True)
    split = pd.read_csv(source / "split_manifest.csv").set_index("group_id").split
    if frame.group_id.map(split).isna().any():
        raise ValueError("unmapped source group")
    train_all = np.flatnonzero(frame.group_id.map(split) == "train")
    test = np.flatnonzero(frame.group_id.map(split) == "test")
    train = sample_indices(frame, train_all, cap=config["max_rows_per_run"])
    counts = frame.iloc[train].groupby("run_id").size()
    weights = frame.iloc[train].run_id.map(1 / counts).to_numpy() * len(counts)
    power = np.vstack([history_features(x) for x in frame.history])
    truth = frame.y.to_numpy()
    baseline = pd.read_csv(source / "per_run_metrics.csv")
    full = baseline.loc[(baseline.condition == "history_static_s2") & (baseline.horizon_ms == 1000)].set_index("run_id")
    if set(full.index) != set(frame.iloc[test].run_id.unique()):
        raise ValueError("saved full S2 score does not cover the test runs")
    predictions = []
    per_run = []
    parameter_rows = []
    for excluded in STATIC:
        fields = tuple(c for c in STATIC if c != excluded)
        meta_train, meta_test = static_features(frame, train, test, columns=fields)
        xtrain = np.column_stack([power[train], meta_train])
        xtest = np.column_stack([power[test], meta_test])
        raw = []
        for q in QUANTILES:
            params = dict(objective="quantile", alpha=float(q), n_estimators=100, num_leaves=15,
                          max_depth=-1, learning_rate=.07, min_child_samples=30, reg_alpha=0.,
                          reg_lambda=0., colsample_bytree=1., subsample=1., subsample_freq=0,
                          random_state=42, n_jobs=1, verbosity=-1)
            model = lgb.LGBMRegressor(**params)
            model.fit(xtrain, truth[train], sample_weight=weights)
            raw.append(model.predict(xtest))
            parameter_rows.append(dict(excluded_feature=excluded, quantile=q,
                                       fitted_trees=model.booster_.num_trees(), **params))
        raw = np.column_stack(raw)
        ordered = np.sort(raw, axis=1)
        scored = frame.iloc[test][["run_id", "issue_s", "target_start_s", "target_end_s", "y"]].copy()
        scored["excluded_feature"] = excluded
        scored["crps_w"] = crps_quantile(truth[test], ordered)
        scored["absolute_error_w"] = np.abs(truth[test] - ordered[:, np.where(QUANTILES == .5)[0][0]])
        scored["crossing_before_correction"] = np.any(np.diff(raw, axis=1) < 0, axis=1)
        for j, q in enumerate(QUANTILES):
            scored[f"q_{q:g}"] = ordered[:, j]
        predictions.append(scored)
        for run_id, group in scored.groupby("run_id"):
            per_run.append(dict(excluded_feature=excluded, run_id=run_id, n_targets=len(group),
                                crps_w=group.crps_w.mean(), mae_w=group.absolute_error_w.mean()))
        print(f"excluded {excluded}: {len(test)} matched test rows", flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    per_run = pd.DataFrame(per_run)
    summary = []
    rng = np.random.default_rng(2026)
    for excluded, group in per_run.groupby("excluded_feature", sort=False):
        paired = group.set_index("run_id").loc[full.index]
        s2 = full.crps_w.to_numpy()
        removed = paired.crps_w.to_numpy()
        draw = rng.integers(0, len(s2), size=(2000, len(s2)))
        effect = 100 * (removed[draw].mean(axis=1) / s2[draw].mean(axis=1) - 1)
        summary.append(dict(feature=excluded, n_test_runs=len(s2), n_test_targets=int(paired.n_targets.sum()),
                            full_s2_crps_w=s2.mean(), removed_crps_w=removed.mean(),
                            contribution_pct=100 * (removed.mean() / s2.mean() - 1),
                            ci_low_pct=np.quantile(effect, .025), ci_high_pct=np.quantile(effect, .975),
                            interpretation="positive means S2 scored better than leave-one-out; paired stored-fit run bootstrap"))
    predictions.to_parquet(destination / "per_target_predictions.parquet", index=False)
    per_run.to_csv(destination / "per_run_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(destination / "metadata_ablation.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(destination / "fitted_parameters.csv", index=False)
    write_json(destination / "manifest.json", dict(source=str(source), source_sha256=digest,
        split_sha256=hashlib.sha256((source / "split_manifest.csv").read_bytes()).hexdigest(),
        comparison="S2 minus one feature; fixed 100-tree models; no tuning or independent seed",
        train_rows=len(train), test_rows=len(test), train_runs=len(counts), test_runs=len(full),
        quantiles=QUANTILES.tolist(), seed=42, bootstrap_seed=2026, bootstrap_replicates=2000,
        features=list(STATIC), physical_interpretation="associational predictive contribution only"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=HERE / "results/actual_20260926")
    parser.add_argument("--out", type=Path, default=HERE / "results/ablation_static_20260926")
    args = parser.parse_args()
    run(args.source, args.out)
