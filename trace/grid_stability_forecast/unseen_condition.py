"""Leave-one GPU/workload combination out; fixed-setting exploratory assessment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core import QUANTILES, crps_quantile, history_features, static_features, write_json

HERE = Path(__file__).resolve().parent
SEEDS = (17, 42, 2026)
CONDITIONS = ("history_only", "history_static_s1", "history_static_s2")


def sampled_rows(frame, indices, seed, cap=1500):
    rng = np.random.default_rng(seed)
    picked = []
    for _, group in frame.iloc[indices].groupby("run_id", sort=True):
        eligible = group.index.to_numpy()
        chosen = rng.choice(eligible, size=min(cap, len(eligible)), replace=False)
        picked.append(np.sort(chosen))
    return np.sort(np.concatenate(picked))


def run(source: Path, out: Path):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((source / "resolved_config.json").read_text())
    target_path = source / "targets.parquet"
    digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    if digest != cfg["source_sha256"]:
        raise ValueError("pilot target hash differs")
    frame = pd.read_parquet(target_path)
    frame = frame.loc[frame.horizon_ms == 1000].reset_index(drop=True)
    groups = frame[["group_id", "run_id", "gpu_model", "workload_type"]].drop_duplicates()
    if groups.run_id.duplicated().any() or groups.group_id.duplicated().any():
        raise ValueError("group is not a unique run; inspect source/session mapping")
    groups["cohort"] = groups.gpu_model.astype(str) + "__" + groups.workload_type.astype(str)
    cohorts = sorted(groups.cohort.unique())
    if len(cohorts) != 4 or min(groups.groupby("cohort").size()) < 5:
        raise ValueError("expected four cohorts with at least five runs")
    groups.to_csv(out / "split_manifest.csv", index=False)
    power = np.vstack([history_features(x) for x in frame.history])
    truth = frame.y.to_numpy()
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    code_hash = hashlib.sha256(Path(__file__).read_bytes() + (HERE / "core.py").read_bytes()).hexdigest()
    for fold, cohort in enumerate(cohorts, 1):
        test_groups = set(groups.loc[groups.cohort == cohort, "group_id"])
        test = np.flatnonzero(frame.group_id.isin(test_groups))
        train_all = np.flatnonzero(~frame.group_id.isin(test_groups))
        for seed in SEEDS:
            key = f"fold{fold}_seed{seed}"
            cell = out / key
            pred_path = cell / "predictions.parquet"
            if progress.get(key, {}).get("status") == "verified" and progress[key].get("code_sha256") == code_hash:
                saved = pd.read_parquet(pred_path, columns=["run_id", "issue_s", "condition"])
                if len(saved) == len(test) * len(CONDITIONS) and saved.condition.nunique() == len(CONDITIONS):
                    print("reused", key, flush=True)
                    continue
            cell.mkdir(exist_ok=True)
            progress[key] = dict(status="training", code_sha256=code_hash, source_sha256=digest,
                                 held_out_cohort=cohort, seed=seed)
            write_json(progress_path, progress)
            train = sampled_rows(frame, train_all, seed, cap=cfg["max_rows_per_run"])
            counts = frame.iloc[train].groupby("run_id").size()
            weights = frame.iloc[train].run_id.map(1 / counts).to_numpy() * len(counts)
            s1tr, s1te = static_features(frame, train, test,
                columns=("gpu_model", "workload_type", "model_name", "batch_size"))
            s2tr, s2te = static_features(frame, train, test)
            features = {
                "history_only": (power[train], power[test]),
                "history_static_s1": (np.column_stack([power[train], s1tr]),
                                      np.column_stack([power[test], s1te])),
                "history_static_s2": (np.column_stack([power[train], s2tr]),
                                      np.column_stack([power[test], s2te])),
            }
            rows = []
            parameters = []
            for condition, (xtrain, xtest) in features.items():
                predicted = []
                for q in QUANTILES:
                    params = dict(objective="quantile", alpha=float(q), n_estimators=100,
                                  num_leaves=15, max_depth=-1, learning_rate=.07,
                                  min_child_samples=30, reg_alpha=0., reg_lambda=0.,
                                  colsample_bytree=1., subsample=1., subsample_freq=0,
                                  random_state=seed, n_jobs=1, verbosity=-1)
                    model = lgb.LGBMRegressor(**params)
                    model.fit(xtrain, truth[train], sample_weight=weights)
                    predicted.append(model.predict(xtest))
                    parameters.append(dict(condition=condition, quantile=q, seed=seed,
                                           fitted_trees=model.booster_.num_trees(), **params))
                raw = np.column_stack(predicted)
                ordered = np.sort(raw, axis=1)
                scored = frame.iloc[test][["run_id", "group_id", "issue_s",
                    "target_start_s", "target_end_s", "y", "delta_w"]].copy()
                scored["held_out_cohort"] = cohort
                scored["fold"] = fold
                scored["seed"] = seed
                scored["condition"] = condition
                scored["crps_w"] = crps_quantile(truth[test], ordered)
                scored["q50"] = ordered[:, np.where(QUANTILES == .5)[0][0]]
                scored["absolute_error_w"] = np.abs(scored.y - scored.q50)
                scored["crossing_before_correction"] = np.any(np.diff(raw, axis=1) < 0, axis=1)
                scored["transition"] = np.where(scored.delta_w >= 100, "rise",
                                         np.where(scored.delta_w <= -100, "drop", "steady"))
                for j, q in enumerate(QUANTILES):
                    scored[f"q_{q:g}"] = ordered[:, j]
                rows.append(scored)
            predictions = pd.concat(rows, ignore_index=True)
            predictions.to_parquet(pred_path, index=False)
            pd.DataFrame(parameters).to_csv(cell / "fitted_parameters.csv", index=False)
            write_json(cell / "manifest.json", dict(held_out_cohort=cohort, fold=fold, seed=seed,
                train_run_ids=sorted(frame.iloc[train].run_id.unique().tolist()),
                test_run_ids=sorted(frame.iloc[test].run_id.unique().tolist()),
                train_row_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                train_rows=len(train), test_rows=len(test), source_sha256=digest,
                score="15-quantile CRPS approximation with held endpoint tails",
                selection="fixed historical setting, no inner tuning"))
            if not predictions.groupby(["run_id", "issue_s"]).condition.nunique().eq(3).all():
                raise AssertionError("paired test rows mismatch")
            progress[key]["status"] = "verified"
            progress[key]["train_rows"] = len(train)
            progress[key]["test_rows"] = len(test)
            write_json(progress_path, progress)
            print("verified", key, cohort, len(test), flush=True)
    all_predictions = pd.concat(
        [pd.read_parquet(out / f"fold{f}_seed{s}" / "predictions.parquet")
         for f in range(1, len(cohorts)+1) for s in SEEDS], ignore_index=True)
    all_predictions.to_parquet(out / "per_target_predictions.parquet", index=False)
    per_run = all_predictions.groupby(["held_out_cohort", "fold", "seed", "condition", "run_id"]).agg(
        n_targets=("y", "size"), crps_w=("crps_w", "mean"),
        mae_w=("absolute_error_w", "mean"),
        p95_w=("absolute_error_w", lambda x: x.quantile(.95)),
        p99_w=("absolute_error_w", lambda x: x.quantile(.99))).reset_index()
    per_run.to_csv(out / "per_run_metrics.csv", index=False)
    per_run.groupby(["held_out_cohort", "condition"]).agg(
        n_runs=("run_id", "nunique"), crps_w=("crps_w", "mean"),
        mae_w=("mae_w", "mean"), p95_w=("p95_w", "mean"),
        p99_w=("p99_w", "mean")).reset_index().to_csv(out / "cohort_metrics.csv", index=False)
    run_seed = per_run.pivot(index=["run_id", "held_out_cohort", "seed"],
                             columns="condition", values="crps_w").reset_index()
    run_mean = run_seed.groupby(["run_id", "held_out_cohort"])[list(CONDITIONS)].mean().reset_index()
    rng = np.random.default_rng(2026)
    effects = []
    for condition in CONDITIONS[1:]:
        base = run_mean.history_only.to_numpy()
        meta = run_mean[condition].to_numpy()
        draw = rng.integers(0, len(base), size=(2000, len(base)))
        boot = 100 * (1 - meta[draw].mean(axis=1) / base[draw].mean(axis=1))
        effects.append(dict(condition=condition, independent_runs=len(base),
            run_balanced_history_crps_w=base.mean(), run_balanced_meta_crps_w=meta.mean(),
            gain_pct=100*(1-meta.mean()/base.mean()), ci_low_pct=np.quantile(boot,.025),
            ci_high_pct=np.quantile(boot,.975), bootstrap_replicates=2000,
            assessment="retrospective leave-one-GPU/workload-combination-out, fixed setting"))
    pd.DataFrame(effects).to_csv(out / "paired_gain_ci.csv", index=False)
    write_json(out / "resolved_config.json", dict(assessment="retrospective unseen GPU/workload combination",
        seeds=SEEDS, held_out_cohorts=cohorts, source_sha256=digest,
        same_rows_weights_capacity_within_each_fold_seed=True,
        quantiles=QUANTILES.tolist(), fixed_historical_model=True,
        sessions_independent_verified=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=HERE / "results/actual_20260926")
    parser.add_argument("--out", type=Path, default=HERE / "results/unseen_condition_20260926")
    args = parser.parse_args()
    run(args.source, args.out)
