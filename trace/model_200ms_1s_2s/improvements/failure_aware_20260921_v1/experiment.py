#!/usr/bin/env python
"""Resumable failure-aware retraining, evaluation, export, and reporting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[1]
CORRECTED = ROOT / "trace/predictability_meta/outputs/corrected_full_20260915"
AUDIT = BASE / "validation/audit_20260921_full_v3"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "validation"))
from src.core import (RESEARCH_QUANTILES, choose_per_run, crps, fit_encoder,
                      fit_models, power_feature_names, raw_predict, rearrange,
                      serialized_size, transform)
from metrics import failure_episodes, interval_metrics, transition_labels


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default))
    os.replace(tmp, path)


def json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default).encode()).hexdigest()


def validate_or_create_signature(path: Path, value: dict) -> str:
    signature = digest(value)
    if path.exists() and json.loads(path.read_text())["sha256"] != signature:
        raise ValueError("incompatible checkpoint configuration")
    atomic_json(path, {"sha256": signature})
    return signature


def reconstruct_power_predictions(raw: np.ndarray, current: np.ndarray, family: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if family == "absolute":
        return raw
    return raw + np.asarray(current, dtype=float).reshape(-1, 1)


def load_cell(horizon: int):
    meta = pd.read_parquet(CORRECTED / f"features/meta_200_{horizon}.parquet")
    powers = np.load(CORRECTED / f"features/power_200_{horizon}.npz")
    return meta, {k: powers[k] for k in powers.files}


def candidate_grid(cfg: dict) -> list[dict]:
    rows = []
    for family in cfg["families"]:
        for history in cfg["histories_ms"]:
            for trees in cfg["n_estimators"]:
                rows.append({
                    "candidate_id": f"{family}__h{history}_t{trees}_l{cfg['num_leaves']}",
                    "family": family,
                    "history_ms": history,
                    "n_estimators": trees,
                    "num_leaves": cfg["num_leaves"],
                    "learning_rate": cfg["learning_rate"],
                    "min_child_samples": cfg["min_child_samples"],
                    "reg_alpha": cfg["reg_alpha"],
                    "reg_lambda": cfg["reg_lambda"],
                    "feature_fraction": cfg["feature_fraction"],
                    "bagging_fraction": cfg["bagging_fraction"],
                    "bagging_freq": cfg["bagging_freq"],
                    "n_jobs": cfg["model_threads"],
                })
    assert len(rows) == 12
    return rows


def make_final_folds(dev: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    for cohort, group in dev.sort_values("run_id").groupby("cohort", sort=True):
        for j, row in enumerate(group.itertuples()):
            rows.append({"run_id": row.run_id, "cohort": cohort, "fold": j % n + 1})
    return pd.DataFrame(rows)


def split_runs(protocol: str, folds: pd.DataFrame, final_folds: pd.DataFrame,
               outer: int, inner: int) -> tuple[list[str], list[str]]:
    if protocol == "outer":
        f = folds[(folds.outer_fold == outer) & (folds.inner_fold == inner)]
        return (f.loc[f.role == "inner_train", "run_id"].tolist(),
                f.loc[f.role == "inner_val", "run_id"].tolist())
    return (final_folds.loc[final_folds.fold != inner, "run_id"].tolist(),
            final_folds.loc[final_folds.fold == inner, "run_id"].tolist())


def outer_runs(folds: pd.DataFrame, outer: int) -> tuple[list[str], list[str]]:
    f = folds[(folds.outer_fold == outer) & folds.inner_fold.isna()]
    return (f.loc[f.role == "outer_train", "run_id"].tolist(),
            f.loc[f.role == "outer_test", "run_id"].tolist())


def transition_weight_table(meta: pd.DataFrame, ids: np.ndarray, cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Inverse-frequency strata within run, capped, then normalized to equal run totals."""
    selected = meta.iloc[ids].copy()
    selected["sample_index"] = ids
    selected["stratum"] = transition_labels(selected.y, selected.current, cfg["transition_threshold_w"])
    selected["unnormalized_weight"] = 0.0
    selected["capped_weight"] = 0.0
    selected["final_weight"] = 0.0
    summaries = []
    for run_id, positions in selected.groupby("run_id", sort=True).groups.items():
        loc = selected.index.get_indexer(positions)
        strata = selected.loc[positions, "stratum"]
        counts = strata.value_counts()
        raw = strata.map(lambda s: 1.0 / counts[s]).to_numpy(float)
        cap = float(raw.min() * cfg["weight_ratio_cap"])
        capped = np.minimum(raw, cap)
        final = capped / capped.sum()
        selected.iloc[loc, selected.columns.get_loc("unnormalized_weight")] = raw
        selected.iloc[loc, selected.columns.get_loc("capped_weight")] = capped
        selected.iloc[loc, selected.columns.get_loc("final_weight")] = final
        for stratum in ("stable", "rising", "falling"):
            present = stratum in counts
            mask = strata.to_numpy() == stratum
            summaries.append({
                "run_id": run_id, "stratum": stratum,
                "count": int(counts.get(stratum, 0)), "present": bool(present),
                "unnormalized_weight_per_row": float(raw[mask][0]) if present else np.nan,
                "capped_weight_per_row": float(capped[mask][0]) if present else np.nan,
                "final_weight_per_row": float(final[mask][0]) if present else np.nan,
                "final_total_weight": float(final[mask].sum()) if present else 0.0,
                "run_total_weight": float(final.sum()), "cap": cap,
            })
    totals = selected.groupby("run_id").final_weight.sum().to_numpy()
    np.testing.assert_allclose(totals, np.ones_like(totals), rtol=0, atol=1e-12)
    return selected.final_weight.to_numpy(float), pd.DataFrame(summaries)


def ordinary_equal_run_weights(meta: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    counts = meta.iloc[ids].groupby("run_id").size()
    weights = meta.iloc[ids].run_id.map(1.0 / counts).to_numpy(float)
    totals = pd.DataFrame({"run_id": meta.iloc[ids].run_id.to_numpy(), "w": weights}).groupby("run_id").w.sum()
    np.testing.assert_allclose(totals.to_numpy(), np.ones(len(totals)), rtol=0, atol=1e-12)
    return weights


def save_sample(out: Path, label: str, horizon: int, history: int, family: str,
                meta: pd.DataFrame, eligible: np.ndarray, cfg: dict) -> Path:
    path = out / "training_samples" / f"{label}__h{horizon}__hist{history}__{family}.npz"
    summary_path = path.with_suffix(".weights.csv")
    if path.exists() and summary_path.exists():
        return path
    ids = choose_per_run(meta, eligible, cfg["max_rows_per_run"])
    strata = transition_labels(meta.iloc[ids].y, meta.iloc[ids].current, cfg["transition_threshold_w"])
    if family == "residual_weighted":
        weights, summary = transition_weight_table(meta, ids, cfg)
    else:
        weights = ordinary_equal_run_weights(meta, ids)
        summary = (pd.DataFrame({"run_id": meta.iloc[ids].run_id.to_numpy(), "stratum": strata,
                                 "final_weight": weights})
                   .groupby(["run_id", "stratum"], as_index=False)
                   .agg(count=("final_weight", "size"), final_total_weight=("final_weight", "sum")))
        summary["present"] = True
        summary["unnormalized_weight_per_row"] = summary.final_total_weight / summary["count"]
        summary["capped_weight_per_row"] = summary.unnormalized_weight_per_row
        summary["final_weight_per_row"] = summary.unnormalized_weight_per_row
        summary["run_total_weight"] = 1.0
        summary["cap"] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indices=ids, eligible=eligible, weights=weights,
                        run_id=meta.iloc[ids].run_id.to_numpy(),
                        target_index=ids, stratum=strata)
    summary.to_csv(summary_path, index=False)
    return path


def longest_exceedance_seconds(group: pd.DataFrame, threshold: float, resolution_s: float = .2) -> float:
    longest = 0
    for _, segment in group.groupby(["segment_id", "subsegment_id"], sort=False):
        g = segment.sort_values("issue_s")
        times = g.issue_s.to_numpy(float)
        hits = np.abs(g.error_w.to_numpy(float)) > threshold
        streak = 0
        previous = None
        for timestamp, hit in zip(times, hits):
            consecutive = previous is not None and np.isclose(timestamp - previous, resolution_s, atol=1e-7)
            streak = streak + 1 if hit and consecutive else (1 if hit else 0)
            longest = max(longest, streak)
            previous = timestamp
    return float(longest * resolution_s)


def selection_metrics(frame: pd.DataFrame, threshold: float) -> tuple[dict, list[dict]]:
    per_run = []
    for run_id, group in frame.groupby("run_id", sort=True):
        absolute = np.abs(group.error_w.to_numpy(float))
        cutoff = np.quantile(absolute, .95)
        per_run.append({
            "run_id": run_id,
            "crps_w": float(group.crps_w.mean()),
            "mae_w": float(absolute.mean()),
            "worst_5pct_mean_abs_error_w": float(absolute[absolute >= cutoff].mean()),
            "exceedance_500_fraction": float(np.mean(absolute > threshold)),
            "longest_500_episode_s": longest_exceedance_seconds(group, threshold),
            "n_targets": int(len(group)),
        })
    p = pd.DataFrame(per_run)
    metrics = {
        "run_balanced_crps_w": float(p.crps_w.mean()),
        "run_balanced_mae_w": float(p.mae_w.mean()),
        "mean_run_worst_5pct_mae_w": float(p.worst_5pct_mean_abs_error_w.mean()),
        "mean_run_exceedance_500_fraction": float(p.exceedance_500_fraction.mean()),
        "mean_run_longest_500_episode_s": float(p.longest_500_episode_s.mean()),
        "n_validation_runs": int(len(p)),
        "n_validation_targets": int(p.n_targets.sum()),
    }
    return metrics, per_run


def make_prediction_frame(meta: pd.DataFrame, ids: np.ndarray, raw_power: np.ndarray,
                          ordered: np.ndarray) -> pd.DataFrame:
    q = list(map(float, RESEARCH_QUANTILES))
    cols = ["run_id", "cohort", "segment_id", "subsegment_id", "issue_s", "target_s",
            "target_bin", "y", "current"]
    frame = meta.iloc[ids][cols].copy()
    frame["target_index"] = ids
    for j, quantile in enumerate(q):
        frame[f"raw_q{quantile:g}"] = raw_power[:, j]
        frame[f"q{quantile:g}"] = ordered[:, j]
    frame["raw_q50"] = raw_power[:, q.index(.5)]
    frame["q50"] = ordered[:, q.index(.5)]
    frame["error_w"] = frame.q50 - frame.y
    frame["crps_w"] = crps(frame.y.to_numpy(), ordered, RESEARCH_QUANTILES)
    frame["transition_stratum"] = transition_labels(frame.y, frame.current)
    return frame


def fit_candidate(spec: dict) -> dict:
    meta, powers = load_cell(int(spec["horizon_ms"]))
    p = spec["parameters"]
    train = np.asarray(spec["train_ids"], dtype=int)
    val = np.asarray(spec["val_ids"], dtype=int)
    saved = np.load(spec["sample_path"])
    ids, weights = saved["indices"], saved["weights"]
    enc = fit_encoder(meta, train, "history_only")
    x = transform(powers[str(p["history_ms"])], meta, enc)
    y = meta.y.to_numpy(float)
    fit_y = y - meta.current.to_numpy(float) if p["family"] != "absolute" else y
    start = time.perf_counter()
    models = fit_models(x, fit_y, ids, weights, RESEARCH_QUANTILES, p, spec["seed"])
    fit_s = time.perf_counter() - start
    raw = raw_predict(models, x[val])
    raw_power = reconstruct_power_predictions(raw, meta.iloc[val].current.to_numpy(float), p["family"])
    ordered, crossing = rearrange(raw_power)
    frame = make_prediction_frame(meta, val, raw_power, ordered)
    metrics, per_run = selection_metrics(frame, spec["failure_threshold_w"])
    result = {k: spec[k] for k in ("protocol", "horizon_ms", "fold", "inner_fold", "seed")}
    result.update(p)
    result.update(metrics)
    result.update({
        "fit_s": fit_s, "crossing_rate": crossing,
        "model_bytes_full15": serialized_size(models, enc, p, RESEARCH_QUANTILES),
        "n_features": int(x.shape[1]), "n_train_sampled": int(len(ids)),
        "tree_counts": [int(m.booster_.num_trees()) for m in models],
        "early_termination": any(m.booster_.num_trees() != p["n_estimators"] for m in models),
        "per_run": per_run,
    })
    return result


def run_tasks(tasks: list[dict], worker, workers: int, label: str) -> None:
    if not tasks:
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            atomic_json(Path(task["done"]), result)
            print(label, result.get("horizon_ms"), result.get("candidate_id", ""),
                  result.get("fold", ""), result.get("inner_fold", ""), flush=True)


def search_candidates(out: Path, cfg: dict, protocol: str, folds: pd.DataFrame,
                      final_folds: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = []
    outer_values = [1] if (smoke or protocol == "final") else list(range(1, cfg["outer_folds"] + 1))
    inner_values = [1] if smoke else list(range(1, cfg["inner_folds"] + 1))
    cap = min(500, cfg["max_rows_per_run"]) if smoke else cfg["max_rows_per_run"]
    local_cfg = cfg | {"max_rows_per_run": cap}
    for horizon in cfg["horizons_ms"]:
        meta, _ = load_cell(horizon)
        for outer in outer_values:
            for inner in inner_values:
                train_runs, val_runs = split_runs(protocol, folds, final_folds, outer, inner)
                assert set(train_runs).isdisjoint(val_runs)
                train = np.flatnonzero(meta.run_id.isin(train_runs))
                val = np.flatnonzero(meta.run_id.isin(val_runs))
                split_label = f"{protocol}_outer{outer}_inner{inner}"
                for p in candidate_grid(cfg):
                    sample = save_sample(out, split_label, horizon, p["history_ms"], p["family"],
                                         meta, train, local_cfg)
                    identity = {"protocol": protocol, "horizon_ms": horizon, "fold": outer,
                                "inner_fold": inner, "seed": cfg["selection_seed"], "parameters": p,
                                "train_ids": train.tolist(), "val_ids": val.tolist(),
                                "sample_path": str(sample), "failure_threshold_w": cfg["failure_threshold_w"],
                                "smoke": smoke}
                    key = digest(identity)
                    done = out / "checkpoints" / f"candidate_{key}.json"
                    if not done.exists():
                        tasks.append(identity | {"done": str(done)})
    run_tasks(tasks, fit_candidate, cfg["workers"], f"candidate-{protocol}")
    records = [json.loads(p.read_text()) for p in (out / "checkpoints").glob("candidate_*.json")]
    records = [r for r in records if r["protocol"] == protocol]
    fold_rows = [{k: v for k, v in r.items() if k != "per_run"} for r in records]
    fold_frame = pd.DataFrame(fold_rows).sort_values(["horizon_ms", "fold", "inner_fold", "candidate_id"])
    fold_frame.to_csv(out / f"{protocol}_candidate_fold_metrics.csv", index=False)
    aggregate_rows = []
    group_keys = ["horizon_ms"] + (["fold"] if protocol == "outer" else [])
    for keys, group in pd.DataFrame(records).groupby(group_keys):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        for candidate_id, candidates in group.groupby("candidate_id"):
            runs = pd.DataFrame([x for row in candidates.per_run for x in row])
            first = candidates.iloc[0]
            aggregate_rows.append(dict(zip(group_keys, key_values)) | {
                "candidate_id": candidate_id, "family": first.family,
                "history_ms": int(first.history_ms), "n_estimators": int(first.n_estimators),
                "num_leaves": int(first.num_leaves),
                "run_balanced_crps_w": runs.crps_w.mean(),
                "run_balanced_mae_w": runs.mae_w.mean(),
                "mean_run_worst_5pct_mae_w": runs.worst_5pct_mean_abs_error_w.mean(),
                "mean_run_exceedance_500_fraction": runs.exceedance_500_fraction.mean(),
                "mean_run_longest_500_episode_s": runs.longest_500_episode_s.mean(),
                "model_bytes_full15": candidates.model_bytes_full15.mean(),
                "fit_s": candidates.fit_s.sum(), "n_validation_runs": runs.run_id.nunique(),
                "n_validation_targets": runs.n_targets.sum(),
            })
    aggregate = pd.DataFrame(aggregate_rows)
    return fold_frame, select_candidates(aggregate, cfg, protocol, out)


def select_candidates(table: pd.DataFrame, cfg: dict, protocol: str, out: Path) -> pd.DataFrame:
    keys = ["horizon_ms"] + (["fold"] if protocol == "outer" else [])
    rows, selections = [], []
    for values, group in table.groupby(keys, sort=True):
        g = group.copy()
        best = float(g.run_balanced_crps_w.min())
        g["best_crps_w"] = best
        g["crps_ratio_to_best"] = g.run_balanced_crps_w / best
        g["reliability_feasible"] = g.run_balanced_crps_w <= best * (1 + cfg["crps_feasibility_tolerance"])
        g["tail_rank"] = np.nan; g["exceedance_rank"] = np.nan; g["episode_rank"] = np.nan
        feasible = g.reliability_feasible
        for metric, rank_name in [
            ("mean_run_worst_5pct_mae_w", "tail_rank"),
            ("mean_run_exceedance_500_fraction", "exceedance_rank"),
            ("mean_run_longest_500_episode_s", "episode_rank")]:
            g.loc[feasible, rank_name] = g.loc[feasible, metric].rank(method="average", ascending=True)
        g["average_reliability_rank"] = g[["tail_rank", "exceedance_rank", "episode_rank"]].mean(axis=1)
        mean_pick = g.sort_values(["run_balanced_crps_w", "model_bytes_full15", "candidate_id"]).iloc[0]
        reliable_pick = g[g.reliability_feasible].sort_values(
            ["average_reliability_rank", "run_balanced_crps_w", "model_bytes_full15", "candidate_id"]).iloc[0]
        key_values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(keys, key_values))
        for procedure, pick in (("mean_crps_selection", mean_pick), ("reliability_selection", reliable_pick)):
            selections.append(identity | {"procedure": procedure, "candidate_id": pick.candidate_id,
                "family": pick.family, "history_ms": int(pick.history_ms),
                "n_estimators": int(pick.n_estimators), "num_leaves": int(pick.num_leaves),
                "validation_crps_w": float(pick.run_balanced_crps_w),
                "average_reliability_rank": float(pick.average_reliability_rank) if pd.notna(pick.average_reliability_rank) else None})
        rows.append(g)
    scored = pd.concat(rows, ignore_index=True)
    selected = pd.DataFrame(selections)
    scored.to_csv(out / f"{protocol}_candidate_metrics.csv", index=False)
    selected.to_csv(out / f"{protocol}_selections.csv", index=False)
    atomic_json(out / f"{protocol}_selections.json", selections)
    return selected


def parameters_from_selection(row: pd.Series, cfg: dict) -> dict:
    match = [p for p in candidate_grid(cfg) if p["candidate_id"] == row.candidate_id]
    if len(match) != 1:
        raise RuntimeError(f"cannot resolve {row.candidate_id}")
    return match[0]


def fit_evaluation(spec: dict) -> dict:
    horizon = int(spec["horizon_ms"])
    meta, powers = load_cell(horizon)
    p = spec["parameters"]
    train = np.asarray(spec["train_ids"], dtype=int)
    test = np.asarray(spec["test_ids"], dtype=int)
    saved = np.load(spec["sample_path"])
    ids, weights = saved["indices"], saved["weights"]
    enc = fit_encoder(meta, train, spec["condition"])
    x = transform(powers[str(p["history_ms"])], meta, enc)
    y = meta.y.to_numpy(float)
    fit_y = y - meta.current.to_numpy(float) if p["family"] != "absolute" else y
    start = time.perf_counter()
    models = fit_models(x, fit_y, ids, weights, RESEARCH_QUANTILES, p, spec["seed"])
    fit_s = time.perf_counter() - start
    raw = raw_predict(models, x[test])
    raw_power = reconstruct_power_predictions(raw, meta.iloc[test].current.to_numpy(float), p["family"])
    ordered, crossing = rearrange(raw_power)
    frame = make_prediction_frame(meta, test, raw_power, ordered)
    for key in ("assessment", "procedure", "horizon_ms", "fold", "seed", "condition"):
        frame[key] = spec[key]
    frame["candidate_id"] = p["candidate_id"]
    frame["family"] = p["family"]
    directory = Path(spec["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    prediction_path = directory / "predictions.parquet"
    frame.to_parquet(prediction_path, index=False)
    bundle = {"models": models, "encoder": enc, "parameters": p,
              "quantiles": RESEARCH_QUANTILES.tolist(), "history_ms": p["history_ms"],
              "resolution_ms": 200, "family": p["family"], "condition": spec["condition"]}
    bundle_path = directory / "full15_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    loaded = joblib.load(bundle_path)
    check_raw = raw_predict(loaded["models"], x[test])
    if p["family"] != "absolute":
        check_raw += meta.iloc[test].current.to_numpy(float)[:, None]
    check, _ = rearrange(check_raw)
    max_reload = float(np.max(np.abs(check - ordered))) if len(test) else 0.0
    if max_reload > 1e-10:
        raise AssertionError(max_reload)
    return {k: spec[k] for k in ("assessment", "procedure", "horizon_ms", "fold", "seed", "condition") } | {
        "candidate_id": p["candidate_id"], "family": p["family"], "history_ms": p["history_ms"],
        "n_estimators": p["n_estimators"], "n_train_sampled": int(len(ids)), "n_test": int(len(test)),
        "fit_s": fit_s, "crossing_rate": crossing, "model_bytes_full15": bundle_path.stat().st_size,
        "prediction_path": str(prediction_path), "bundle_path": str(bundle_path),
        "reload_max_abs_difference_w": max_reload,
    }


def evaluate_outer(out: Path, cfg: dict, folds: pd.DataFrame, selections: pd.DataFrame,
                   smoke: bool) -> pd.DataFrame:
    tasks = []
    use = selections[selections.fold.eq(1)] if smoke else selections
    cap = min(500, cfg["max_rows_per_run"]) if smoke else cfg["max_rows_per_run"]
    local_cfg = cfg | {"max_rows_per_run": cap}
    seeds = [cfg["deployment_seed"]] if smoke else cfg["evaluation_seeds"]
    for row in use.itertuples():
        horizon, fold = int(row.horizon_ms), int(row.fold)
        meta, _ = load_cell(horizon)
        train_runs, test_runs = outer_runs(folds, fold)
        train = np.flatnonzero(meta.run_id.isin(train_runs)); test = np.flatnonzero(meta.run_id.isin(test_runs))
        p = parameters_from_selection(pd.Series(row._asdict()), cfg)
        for condition in ("history_only", "history_meta"):
            for seed in seeds:
                label = f"outerfit_fold{fold}_{row.procedure}"
                sample = save_sample(out, label, horizon, p["history_ms"], p["family"], meta, train, local_cfg)
                identity = {"assessment": "grouped_outer", "procedure": row.procedure,
                            "horizon_ms": horizon, "fold": fold, "seed": seed, "condition": condition,
                            "parameters": p, "train_ids": train.tolist(), "test_ids": test.tolist(),
                            "sample_path": str(sample), "smoke": smoke}
                key = digest(identity); directory = out / "evaluation" / key
                done = out / "checkpoints" / f"evaluation_{key}.json"
                if not done.exists(): tasks.append(identity | {"directory": str(directory), "done": str(done)})
    run_tasks(tasks, fit_evaluation, cfg["workers"], "outer-evaluation")
    records = [json.loads(p.read_text()) for p in (out / "checkpoints").glob("evaluation_*.json")]
    records = [r for r in records if r["assessment"] == "grouped_outer"]
    pd.DataFrame(records).to_csv(out / "outer_evaluation_tasks.csv", index=False)
    paths = sorted({r["prediction_path"] for r in records})
    predictions = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    predictions.to_parquet(out / "outer_predictions.parquet", index=False)
    return predictions


def fit_final(out: Path, cfg: dict, dev: pd.DataFrame, hold: pd.DataFrame,
              selections: pd.DataFrame, smoke: bool) -> tuple[pd.DataFrame, Path]:
    tasks = []
    cap = min(500, cfg["max_rows_per_run"]) if smoke else cfg["max_rows_per_run"]
    local_cfg = cfg | {"max_rows_per_run": cap}
    seeds = [cfg["deployment_seed"]] if smoke else cfg["evaluation_seeds"]
    for row in selections.itertuples():
        horizon = int(row.horizon_ms); meta, _ = load_cell(horizon)
        train = np.flatnonzero(meta.run_id.isin(dev.run_id)); test = np.flatnonzero(meta.run_id.isin(hold.run_id))
        p = parameters_from_selection(pd.Series(row._asdict()), cfg)
        for condition in ("history_only", "history_meta"):
            for seed in seeds:
                label = f"finalfit_{row.procedure}"
                sample = save_sample(out, label, horizon, p["history_ms"], p["family"], meta, train, local_cfg)
                identity = {"assessment": "reused_holdout", "procedure": row.procedure,
                            "horizon_ms": horizon, "fold": 0, "seed": seed, "condition": condition,
                            "parameters": p, "train_ids": train.tolist(), "test_ids": test.tolist(),
                            "sample_path": str(sample), "smoke": smoke}
                key = digest(identity); directory = out / "final_evaluation" / key
                done = out / "checkpoints" / f"evaluation_{key}.json"
                if not done.exists(): tasks.append(identity | {"directory": str(directory), "done": str(done)})
    run_tasks(tasks, fit_evaluation, cfg["workers"], "final-evaluation")
    records = [json.loads(p.read_text()) for p in (out / "checkpoints").glob("evaluation_*.json")]
    records = [r for r in records if r["assessment"] == "reused_holdout"]
    pd.DataFrame(records).to_csv(out / "final_evaluation_tasks.csv", index=False)
    predictions = pd.concat([pd.read_parquet(p) for p in sorted({r["prediction_path"] for r in records})], ignore_index=True)
    predictions.to_parquet(out / "reused_holdout_predictions.parquet", index=False)
    model_root = HERE / "models" / (cfg["model_version"] + ("-smoke" if smoke else ""))
    model_root.mkdir(parents=True, exist_ok=True)
    deploy = [r for r in records if r["procedure"] == "reliability_selection" and r["seed"] == cfg["deployment_seed"]]
    manifest_rows = []
    for condition in ("history_only", "history_meta"):
        horizons = {}
        for record in deploy:
            if record["condition"] != condition:
                continue
            source = Path(record["bundle_path"]); item = joblib.load(source)
            horizons[str(record["horizon_ms"])] = item
            target = model_root / f"full15_h{record['horizon_ms']}_{condition}.joblib"
            shutil.copy2(source, target)
            manifest_rows.append({"condition": condition, "horizon_ms": record["horizon_ms"],
                                  "candidate_id": record["candidate_id"], "source": str(source),
                                  "export": str(target), "bytes": target.stat().st_size})
        package = {"model_version": cfg["model_version"], "condition": condition,
                   "resolution_ms": cfg["resolution_ms"], "quantiles": cfg["quantiles"],
                   "ordering_correction": "full-15 row-wise ascending sort before optional response subsetting",
                   "power_scope": "GPU sum (8 GPUs); CPU/facility excluded", "horizons": horizons}
        joblib.dump(package, model_root / f"{condition}_full15_deployment.joblib", compress=3)
    pd.DataFrame(manifest_rows).to_csv(out / "deployment_manifest.csv", index=False)
    return predictions, model_root


def validate_scope(out: Path, cfg: dict, folds: pd.DataFrame, dev: pd.DataFrame, hold: pd.DataFrame) -> None:
    checks = {"development_runs": int(dev.run_id.nunique()), "reused_holdout_runs": int(hold.run_id.nunique()),
              "development_holdout_disjoint": set(dev.run_id).isdisjoint(hold.run_id),
              "horizons": {}, "passed": True}
    checks["passed"] &= checks["development_runs"] == 28 and checks["reused_holdout_runs"] == 4
    for horizon in cfg["horizons_ms"]:
        meta, powers = load_cell(horizon)
        info = {"rows": len(meta), "runs": int(meta.run_id.nunique()),
                "has_5000ms_features": "5000" in powers,
                "max_horizon_error_ms": float(np.max(np.abs((meta.target_s-meta.issue_s)*1000-horizon))),
                "current_feature_max_abs_diff_w": float(np.max(np.abs(powers["5000"][:, 0]-meta.current))),
                "segment_columns_present": all(x in meta for x in ("segment_id", "subsegment_id"))}
        info["passed"] = info["runs"] == 32 and info["has_5000ms_features"] and info["max_horizon_error_ms"] < 1e-6
        checks["horizons"][str(horizon)] = info; checks["passed"] &= info["passed"]
    for outer in range(1, cfg["outer_folds"]+1):
        train, test = outer_runs(folds, outer); checks["passed"] &= set(train).isdisjoint(test)
    atomic_json(out / "data_integrity_checks.json", checks)
    if not checks["passed"]: raise AssertionError(checks)


def copy_manifests(out: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("checkpoints", "training_samples", "evaluation", "final_evaluation", "plots"):
        (out / sub).mkdir(exist_ok=True)
    shutil.copy2(CORRECTED / "input_manifest.csv", out / "input_manifest.csv")
    shutil.copy2(CORRECTED / "splits/folds.csv", out / "outer_inner_split_manifest.csv")
    shutil.copy2(CORRECTED / "development_runs.csv", out / "development_runs.csv")
    shutil.copy2(CORRECTED / "reused_holdout_runs.csv", out / "reused_holdout_runs.csv")
    shutil.copy2(HERE / "failure_diagnosis.md", out / "failure_diagnosis.md")
    folds = pd.read_csv(CORRECTED / "splits/folds.csv")
    dev = pd.read_csv(CORRECTED / "development_runs.csv")
    hold = pd.read_csv(CORRECTED / "reused_holdout_runs.csv")
    final_folds = make_final_folds(dev, cfg["inner_folds"])
    final_folds.to_csv(out / "development_inner_split_manifest.csv", index=False)
    schemas = []
    for history in cfg["histories_ms"]:
        schemas.extend({"history_ms": history, "feature": x, "availability": "completed issue-time bins only"}
                       for x in power_feature_names(cfg["resolution_ms"], history))
    pd.DataFrame(schemas).to_csv(out / "feature_schemas.csv", index=False)
    return folds, dev, hold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=("all", "search", "evaluate", "finalize"), default="all")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    mode = "smoke" if args.smoke else "full"
    out = HERE / "artifacts" / mode
    out.mkdir(parents=True, exist_ok=True)
    signature_path = out / "configuration_signature.json"
    validate_or_create_signature(signature_path, cfg | {"mode": mode})
    atomic_json(out / "frozen_experiment_spec.json", cfg | {"mode": mode,
        "selection_rule": "CRPS within 5% of best, then equal-weight mean ranks of run tail, exceedance fraction, and longest episode",
        "comparison_status": "retrospective; reused holdout previously inspected"})
    atomic_json(out / "environment.json", {"python": sys.version, "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "workers": cfg["workers"], "model_threads": cfg["model_threads"],
        "versions": {name: __import__(name).__version__ for name in ("numpy", "pandas", "lightgbm", "joblib")}})
    status = {"status": "running", "mode": mode, "stage": args.stage,
              "started_utc": time.strftime("%FT%TZ", time.gmtime()), "pid": os.getpid()}
    atomic_json(out / "status.json", status)
    log = (out / "execution.log").open("a", buffering=1); original = sys.stdout
    class Tee:
        def write(self, text): original.write(text); log.write(text)
        def flush(self): original.flush(); log.flush()
    sys.stdout = Tee()
    try:
        folds, dev, hold = copy_manifests(out, cfg)
        final_folds = pd.read_csv(out / "development_inner_split_manifest.csv")
        validate_scope(out, cfg, folds, dev, hold)
        if args.stage in ("all", "search"):
            _, outer_sel = search_candidates(out, cfg, "outer", folds, final_folds, args.smoke)
            _, final_sel = search_candidates(out, cfg, "final", folds, final_folds, args.smoke)
        else:
            outer_sel = pd.read_csv(out / "outer_selections.csv")
            final_sel = pd.read_csv(out / "final_selections.csv")
        if args.stage in ("all", "evaluate"):
            evaluate_outer(out, cfg, folds, outer_sel, args.smoke)
        if args.stage in ("all", "finalize"):
            hold_pred, model_root = fit_final(out, cfg, dev, hold, final_sel, args.smoke)
            from reporting import build_all_reports
            build_all_reports(out, cfg, model_root, smoke=args.smoke)
        status.update(status="completed", finished_utc=time.strftime("%FT%TZ", time.gmtime()),
                      maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(out / "status.json", status)
        print("EXPERIMENT_COMPLETE", out, flush=True)
    except Exception as exc:
        status.update(status="failed", error=repr(exc), finished_utc=time.strftime("%FT%TZ", time.gmtime()))
        atomic_json(out / "status.json", status)
        raise
    finally:
        sys.stdout = original; log.close()


if __name__ == "__main__":
    main()
