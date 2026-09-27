"""Budgeted grouped inner search on the reused +1 s outer split.

The 24 common candidates receive a q50 screening fit on three inner folds.
The top six per feature condition receive full 15-quantile inner evaluation.
This is an exploratory staged search; outer test was used in prior pilot reports.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core import QUANTILES, crps_quantile, history_features, static_features, write_json
from unseen_condition import sampled_rows

HERE = Path(__file__).resolve().parent
FEATURES = ("history_only", "history_static_s2")
SELECTION_SEED = 2026
CAP = 1000
SCREEN_TREES = 200
FULL_TREES = 1000
SHORTLIST = 6


def candidates():
    space = list(itertools.product((.03, .05), (7, 15, 31), (50, 200, 500), (0., 1., 10.)))
    rng = np.random.default_rng(2026)
    chosen = rng.choice(len(space), size=24, replace=False)
    return [dict(candidate_id=f"c{i:02d}", learning_rate=space[j][0],
                 num_leaves=space[j][1], min_child_samples=space[j][2],
                 reg_lambda=space[j][3]) for i, j in enumerate(chosen)]


def inner_folds(frame, outer_train):
    groups = frame.iloc[outer_train][["run_id", "group_id", "gpu_model", "workload_type"]].drop_duplicates()
    if groups.run_id.duplicated().any():
        raise ValueError("inner source group spans multiple runs")
    groups["cohort"] = groups.gpu_model.astype(str) + "__" + groups.workload_type.astype(str)
    groups["inner_fold"] = -1
    for _, cohort in groups.groupby("cohort"):
        ordered = cohort.sort_values("group_id")
        groups.loc[ordered.index, "inner_fold"] = np.arange(len(ordered)) % 3
    if set(groups.inner_fold) != {0, 1, 2}:
        raise ValueError("three nonempty inner folds required")
    return groups


def params(candidate, quantile, trees, seed):
    return dict(objective="quantile", alpha=float(quantile), n_estimators=trees,
        num_leaves=int(candidate["num_leaves"]), max_depth=-1,
        learning_rate=float(candidate["learning_rate"]),
        min_child_samples=int(candidate["min_child_samples"]),
        reg_alpha=0., reg_lambda=float(candidate["reg_lambda"]),
        colsample_bytree=1., subsample=1., subsample_freq=0,
        random_state=seed, n_jobs=1, verbosity=-1)


def fit_fold(frame, power, sample, val, condition, candidate, quantiles, trees, early_stop):
    counts = frame.iloc[sample].groupby("run_id").size()
    weights = frame.iloc[sample].run_id.map(1 / counts).to_numpy() * len(counts)
    if condition == "history_only":
        xtr, xval = power[sample], power[val]
    else:
        tr, va = static_features(frame, sample, val)
        xtr, xval = np.column_stack([power[sample], tr]), np.column_stack([power[val], va])
    truth = frame.y.to_numpy()
    predictions, fitted = [], []
    for q in quantiles:
        model = lgb.LGBMRegressor(**params(candidate, q, trees, SELECTION_SEED))
        kw = dict(eval_set=[(xval, truth[val])], eval_metric="quantile",
                  early_stopping_rounds=50, verbose=False) if early_stop else {}
        model.fit(xtr, truth[sample], sample_weight=weights, **kw)
        predictions.append(model.predict(xval))
        fitted.append(int(model.best_iteration_ or model.booster_.num_trees()))
    return np.column_stack(predictions), fitted


def run(source: Path, out: Path):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target_path = source / "targets.parquet"
    config = json.loads((source / "resolved_config.json").read_text())
    digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    if digest != config["source_sha256"]:
        raise ValueError("target source has changed")
    frame = pd.read_parquet(target_path)
    frame = frame.loc[frame.horizon_ms == 1000].reset_index(drop=True)
    outer = pd.read_csv(source / "split_manifest.csv").set_index("group_id").split
    if frame.group_id.map(outer).isna().any():
        raise ValueError("outer split missing source group")
    outer_train = np.flatnonzero(frame.group_id.map(outer) == "train")
    outer_test = np.flatnonzero(frame.group_id.map(outer) == "test")
    folds = inner_folds(frame, outer_train)
    folds.to_csv(out / "inner_folds.csv", index=False)
    by_run = folds.set_index("run_id").inner_fold
    fold_assignment = frame.run_id.map(by_run)
    power = np.vstack([history_features(x) for x in frame.history])
    inventory = candidates()
    manifest = dict(candidates=inventory, generation_seed=2026, screening_quantile=.5,
        screening_trees=SCREEN_TREES, shortlist_per_condition=SHORTLIST,
        full_quantiles=QUANTILES.tolist(), full_tree_cap=FULL_TREES,
        early_stopping_patience=50, inner_folds=3, sample_cap_per_training_run=CAP,
        assessment="reused outer holdout; exploratory, not fresh confirmation",
        source_sha256=digest)
    write_json(out / "candidate_manifest.json", manifest)
    code_hash = hashlib.sha256(Path(__file__).read_bytes() + (HERE / "core.py").read_bytes()).hexdigest()
    screen_rows, full_rows = [], []
    for condition in FEATURES:
        for candidate in inventory:
            cid = candidate["candidate_id"]
            cell = out / "screen" / condition / f"{cid}.json"
            if cell.exists():
                record = json.loads(cell.read_text())
                if record.get("code_sha256") == code_hash and record.get("source_sha256") == digest:
                    screen_rows.append(record)
                    continue
            cell.parent.mkdir(parents=True, exist_ok=True)
            fold_scores = []
            for fold in range(3):
                train_pool = np.flatnonzero((fold_assignment != fold) & frame.group_id.map(outer).eq("train"))
                sample = sampled_rows(frame, train_pool, SELECTION_SEED, CAP)
                val = np.flatnonzero(fold_assignment == fold)
                pred, _ = fit_fold(frame, power, sample, val, condition, candidate,
                                   [.5], SCREEN_TREES, False)
                y = frame.y.to_numpy()[val]
                pinball = np.maximum(.5*(y-pred[:, 0]), -.5*(y-pred[:, 0]))
                scored = pd.DataFrame(dict(run_id=frame.iloc[val].run_id.to_numpy(), loss=pinball))
                fold_scores.extend(scored.groupby("run_id").loss.mean().tolist())
            record = dict(condition=condition, candidate_id=cid, screen_loss=float(np.mean(fold_scores)),
                          code_sha256=code_hash, source_sha256=digest)
            write_json(cell, record)
            screen_rows.append(record)
            print("screen", condition, cid, round(record["screen_loss"], 4), flush=True)
        screen = pd.DataFrame([r for r in screen_rows if r["condition"] == condition])
        shortlist = screen.nsmallest(SHORTLIST, "screen_loss").candidate_id.tolist()
        for cid in shortlist:
            candidate = next(c for c in inventory if c["candidate_id"] == cid)
            cell = out / "full15" / condition / f"{cid}.json"
            if cell.exists():
                record = json.loads(cell.read_text())
                if record.get("code_sha256") == code_hash and record.get("source_sha256") == digest:
                    full_rows.append(record)
                    continue
            cell.parent.mkdir(parents=True, exist_ok=True)
            run_scores, tree_counts = [], []
            for fold in range(3):
                train_pool = np.flatnonzero((fold_assignment != fold) & frame.group_id.map(outer).eq("train"))
                sample = sampled_rows(frame, train_pool, SELECTION_SEED, CAP)
                val = np.flatnonzero(fold_assignment == fold)
                pred, fitted = fit_fold(frame, power, sample, val, condition, candidate,
                                         QUANTILES, FULL_TREES, True)
                pred = np.sort(pred, axis=1)
                scores = crps_quantile(frame.y.to_numpy()[val], pred)
                scored = pd.DataFrame(dict(run_id=frame.iloc[val].run_id.to_numpy(), crps_w=scores))
                run_scores.extend(scored.groupby("run_id").crps_w.mean().tolist())
                tree_counts.append(fitted)
            record = dict(condition=condition, candidate_id=cid,
                inner_run_balanced_crps_w=float(np.mean(run_scores)),
                tree_count_per_quantile=np.median(tree_counts, axis=0).astype(int).tolist(),
                code_sha256=code_hash, source_sha256=digest)
            write_json(cell, record)
            full_rows.append(record)
            print("full15", condition, cid, round(record["inner_run_balanced_crps_w"], 4), flush=True)
    pd.DataFrame(screen_rows).to_csv(out / "screening_metrics.csv", index=False)
    pd.DataFrame(full_rows).drop(columns=["tree_count_per_quantile"]).to_csv(
        out / "full15_inner_metrics.csv", index=False)
    selected = {}
    for condition in FEATURES:
        eligible = [r for r in full_rows if r["condition"] == condition]
        selected[condition] = min(eligible, key=lambda x: x["inner_run_balanced_crps_w"])
    write_json(out / "selection.json", selected)
    print("selection", json.dumps({c:r["candidate_id"] for c,r in selected.items()}), flush=True)
    evaluate_outer(frame, power, outer_train, outer_test, inventory, selected, out, digest)


def evaluate_outer(frame, power, train_pool, test, inventory, selected, out, digest):
    specs = {
        "history_only_tuned": ("history_only", selected["history_only"]),
        "history_static_s2_controlled": ("history_static_s2", selected["history_only"]),
        "history_static_s2_tuned": ("history_static_s2", selected["history_static_s2"]),
    }
    all_rows, params_rows = [], []
    truth = frame.y.to_numpy()
    for seed in (17, 42, 2026):
        train = sampled_rows(frame, train_pool, seed, CAP)
        counts = frame.iloc[train].groupby("run_id").size()
        weights = frame.iloc[train].run_id.map(1 / counts).to_numpy() * len(counts)
        metadata_train, metadata_test = static_features(frame, train, test)
        for condition, (feature_type, selected_record) in specs.items():
            candidate = next(c for c in inventory if c["candidate_id"] == selected_record["candidate_id"])
            if feature_type == "history_only":
                xtr, xte = power[train], power[test]
            else:
                xtr = np.column_stack([power[train], metadata_train])
                xte = np.column_stack([power[test], metadata_test])
            raw = []
            for j, q in enumerate(QUANTILES):
                trees = selected_record["tree_count_per_quantile"][j]
                model = lgb.LGBMRegressor(**params(candidate, q, trees, seed))
                model.fit(xtr, truth[train], sample_weight=weights)
                raw.append(model.predict(xte))
                params_rows.append(dict(feature_condition=condition, quantile=q, seed=seed,
                    candidate_id=candidate["candidate_id"], selected_trees=trees,
                    fitted_trees=model.booster_.num_trees(), inner_score=selected_record["inner_run_balanced_crps_w"],
                    **params(candidate, q, trees, seed)))
            raw = np.column_stack(raw)
            corrected = np.sort(raw, axis=1)
            scored = frame.iloc[test][["run_id", "group_id", "issue_s", "target_start_s",
                "target_end_s", "y", "delta_w"]].copy()
            scored["condition"] = condition
            scored["seed"] = seed
            scored["crps_w"] = crps_quantile(truth[test], corrected)
            scored["q50"] = corrected[:, np.where(QUANTILES == .5)[0][0]]
            scored["absolute_error_w"] = np.abs(scored.y - scored.q50)
            scored["crossing_before_correction"] = np.any(np.diff(raw, axis=1) < 0, axis=1)
            for j, q in enumerate(QUANTILES):
                scored[f"q_{q:g}"] = corrected[:, j]
            all_rows.append(scored)
            print("outer", seed, condition, flush=True)
    predictions = pd.concat(all_rows, ignore_index=True)
    predictions.to_parquet(out / "per_target_predictions.parquet", index=False)
    pd.DataFrame(params_rows).to_csv(out / "selected_hyperparameters.csv", index=False)
    per_run = predictions.groupby(["seed", "condition", "run_id"]).agg(
        n_targets=("y", "size"), crps_w=("crps_w", "mean"),
        mae_w=("absolute_error_w", "mean"),
        p95_w=("absolute_error_w", lambda x:x.quantile(.95)),
        p99_w=("absolute_error_w", lambda x:x.quantile(.99))).reset_index()
    per_run.to_csv(out / "per_run_metrics.csv", index=False)
    per_run.groupby("condition").agg(n_runs=("run_id","nunique"), crps_w=("crps_w","mean"),
        mae_w=("mae_w","mean"), p95_w=("p95_w","mean"), p99_w=("p99_w","mean")).reset_index().to_csv(
            out / "aggregate_metrics.csv", index=False)
    pivot = per_run.pivot(index=["run_id","seed"],columns="condition",values="crps_w").reset_index()
    pivot = pivot.groupby("run_id")[list(specs)].mean()
    base = pivot.history_only_tuned.to_numpy()
    rng = np.random.default_rng(2026)
    effects = []
    for condition in list(specs)[1:]:
        meta = pivot[condition].to_numpy()
        draw = rng.integers(0,len(base),size=(2000,len(base)))
        boot = 100*(1-meta[draw].mean(axis=1)/base[draw].mean(axis=1))
        effects.append(dict(condition=condition,n_test_runs=len(base),gain_pct=100*(1-meta.mean()/base.mean()),
            ci_low_pct=np.quantile(boot,.025),ci_high_pct=np.quantile(boot,.975),
            bootstrap_replicates=2000,assessment="reused outer holdout; three sampling seeds"))
    pd.DataFrame(effects).to_csv(out/"paired_gain_ci.csv",index=False)
    write_json(out/"resolved_config.json",dict(source_sha256=digest,assessment="exploratory grouped inner selected reused outer holdout",
        screening="24 candidates q50 200 trees, top six full15 3-fold inner",
        outer_seeds=[17,42,2026],training_cap_per_run=CAP,capacity_controlled=True,
        sessions_independent_verified=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=HERE/"results/actual_20260926")
    parser.add_argument("--out",type=Path,default=HERE/"results/nested_select_20260926")
    args=parser.parse_args()
    run(args.source,args.out)
