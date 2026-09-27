#!/usr/bin/env python
"""Resumable full lightweight experiment, final fit, export, and reporting."""
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
ROOT = HERE.parents[1]
CORRECTED = ROOT / "trace/predictability_meta/outputs/corrected_full_20260915"
sys.path.insert(0, str(HERE))
from src.core import (RESEARCH_QUANTILES, choose_per_run, crps, equal_run_weights,
    fit_encoder, fit_models, latency_us, power_feature_names, raw_predict, rearrange,
    score_frame, serialized_size, summarize, transform)


def atomic_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str)); os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def candidates(cfg):
    rows = []
    for history in cfg["histories_ms"]:
        for trees in cfg["n_estimators"]:
            for leaves in cfg["num_leaves"]:
                rows.append({"candidate_id": f"h{history}_t{trees}_l{leaves}", "history_ms": history,
                    "n_estimators": trees, "num_leaves": leaves, "learning_rate": cfg["learning_rate"],
                    "min_child_samples": cfg["min_child_samples"], "reg_alpha": cfg["reg_alpha"],
                    "reg_lambda": cfg["reg_lambda"], "n_jobs": cfg["model_threads"]})
    return rows


def load_cell(horizon):
    meta = pd.read_parquet(CORRECTED / f"features/meta_200_{horizon}.parquet")
    powers = np.load(CORRECTED / f"features/power_200_{horizon}.npz")
    return meta, {k: powers[k] for k in powers.files}


def groups_from_corrected(folds, outer, inner=None):
    if inner is None:
        f = folds[(folds.outer_fold == outer) & folds.inner_fold.isna()]
        return f.loc[f.role == "outer_train", "run_id"].tolist(), f.loc[f.role == "outer_test", "run_id"].tolist()
    f = folds[(folds.outer_fold == outer) & (folds.inner_fold == inner)]
    return f.loc[f.role == "inner_train", "run_id"].tolist(), f.loc[f.role == "inner_val", "run_id"].tolist()


def make_final_folds(dev, n=3):
    rows = []
    for cohort, group in dev.sort_values("run_id").groupby("cohort", sort=True):
        for j, row in enumerate(group.itertuples()):
            rows.append({"run_id": row.run_id, "cohort": cohort, "fold": j % n + 1})
    return pd.DataFrame(rows)


def save_sample(out, label, horizon, history, meta, train_ids, cap):
    path = out / "training_samples" / f"{label}_h{horizon}_hist{history}.npz"
    if path.exists():
        z = np.load(path); return z["indices"], z["weights"]
    ids = choose_per_run(meta, train_ids, cap); weights = equal_run_weights(meta, ids)
    np.savez_compressed(path, indices=ids, weights=weights,
                        run_id=meta.iloc[ids].run_id.to_numpy(), target_index=ids)
    return ids, weights


def candidate_task(spec):
    horizon = spec["horizon_ms"]; meta, powers = load_cell(horizon)
    train = np.asarray(spec["train_ids"], dtype=int); val = np.asarray(spec["val_ids"], dtype=int)
    sampled = np.load(spec["sample_path"]); ids, weights = sampled["indices"], sampled["weights"]
    p = spec["parameters"]; enc = fit_encoder(meta, train, "history_only")
    x = transform(powers[str(p["history_ms"])], meta, enc); y = meta.y.to_numpy()
    start = time.perf_counter(); models = fit_models(x, y, ids, weights, RESEARCH_QUANTILES, p, spec["seed"])
    fit_s = time.perf_counter() - start
    raw = raw_predict(models, x[val]); pred, crossing = rearrange(raw)
    frame = score_frame(meta, val, pred, RESEARCH_QUANTILES); summary, per_run = summarize(frame)
    keep = [3, 7, 11]; deploy = [models[i] for i in keep]
    result = {k: spec[k] for k in ("protocol", "horizon_ms", "fold", "inner_fold", "seed")}
    result.update(p); result.update(summary); result.update({"candidate_id": p["candidate_id"],
        "fit_s": fit_s, "crossing_rate": crossing, "model_bytes_3q": serialized_size(deploy, enc, p, [.1,.5,.9]),
        "n_features": int(x.shape[1]), "n_train_sampled": int(len(ids)),
        "tree_counts": [int(m.booster_.num_trees()) for m in models],
        "early_termination": any(m.booster_.num_trees() != p["n_estimators"] for m in models)})
    result.update(latency_us(deploy, x[val], trials=10)); return result


def run_candidate_search(out, cfg, protocol, folds, final_folds, smoke=False):
    tasks, cand = [], candidates(cfg)
    if smoke: cand = cand[:1]
    for horizon in cfg["horizons_ms"]:
        meta, _ = load_cell(horizon)
        outer_values = [1] if (smoke or protocol == "final") else range(1, cfg["outer_folds"] + 1)
        for outer in outer_values:
            inner_values = [1] if smoke else range(1, cfg["inner_folds"] + 1)
            for inner in inner_values:
                if protocol == "outer":
                    train_runs, val_runs = groups_from_corrected(folds, outer, inner)
                    label = f"outer{outer}_inner{inner}"
                else:
                    val_runs = final_folds.loc[final_folds.fold == inner, "run_id"].tolist()
                    train_runs = final_folds.loc[final_folds.fold != inner, "run_id"].tolist()
                    label = f"development_fold{inner}"
                train_ids = np.flatnonzero(meta.run_id.isin(train_runs)); val_ids = np.flatnonzero(meta.run_id.isin(val_runs))
                assert set(train_runs).isdisjoint(val_runs)
                for p in cand:
                    _, _ = save_sample(out, label, horizon, p["history_ms"], meta, train_ids, cfg["max_rows_per_run"])
                    sample_path = out / "training_samples" / f"{label}_h{horizon}_hist{p['history_ms']}.npz"
                    for seed in ([cfg["deployment_seed"]] if smoke else cfg["seeds"]):
                        key = digest({"protocol": protocol, "h": horizon, "o": outer, "i": inner, "p": p, "seed": seed})
                        done = out / "checkpoints" / f"candidate_{key}.json"
                        if done.exists(): continue
                        tasks.append({"protocol": protocol, "horizon_ms": horizon, "fold": outer,
                            "inner_fold": inner, "seed": seed, "parameters": p, "train_ids": train_ids.tolist(),
                            "val_ids": val_ids.tolist(), "sample_path": str(sample_path), "done": str(done)})
    if tasks:
        with ProcessPoolExecutor(max_workers=min(cfg["workers"], len(tasks))) as pool:
            mapping = {pool.submit(candidate_task, t): t for t in tasks}
            for future in as_completed(mapping):
                task = mapping[future]; result = future.result(); atomic_json(task["done"], result)
                print("candidate", protocol, result["horizon_ms"], result["candidate_id"], result["fold"], result["inner_fold"], result["seed"], flush=True)
    records = [json.loads(p.read_text()) for p in (out / "checkpoints").glob("candidate_*.json")]
    frame = pd.DataFrame([r for r in records if r["protocol"] == protocol])
    frame.to_csv(out / f"{protocol}_candidate_scores.csv", index=False)
    return frame


def select_candidates(scores, cfg, protocol):
    selected = []
    keys = ["horizon_ms"] + (["fold"] if protocol == "outer" else [])
    for values, group in scores.groupby(keys):
        means = group.groupby("candidate_id", as_index=False).agg(
            validation_crps_w=("run_balanced_crps_w", "mean"), model_bytes_3q=("model_bytes_3q", "mean"),
            latency_us_p50=("latency_us_p50", "mean"), n_features=("n_features", "first"))
        best = means.validation_crps_w.min(); eligible = means[means.validation_crps_w <= best * (1 + cfg["selection_tolerance"])]
        choice = eligible.sort_values(["model_bytes_3q", "latency_us_p50", "n_features", "candidate_id"]).iloc[0]
        original = group[group.candidate_id == choice.candidate_id].iloc[0]
        key_values = values if isinstance(values, tuple) else (values,)
        record = dict(zip(keys, key_values)); record.update({"candidate_id": choice.candidate_id,
            "best_crps_w": float(best), "selected_crps_w": float(choice.validation_crps_w),
            "within_one_percent": True, "parameters": {k: original[k].item() if hasattr(original[k], "item") else original[k]
                for k in ("candidate_id", "history_ms", "n_estimators", "num_leaves", "learning_rate",
                          "min_child_samples", "reg_alpha", "reg_lambda", "n_jobs")}})
        selected.append(record)
    return selected


def evaluation_task(spec):
    horizon = spec["horizon_ms"]; meta, powers = load_cell(horizon); p = spec["parameters"]
    train = np.asarray(spec["train_ids"], dtype=int); test = np.asarray(spec["test_ids"], dtype=int)
    ids = choose_per_run(meta, train, spec["cap"]); weights = equal_run_weights(meta, ids)
    enc = fit_encoder(meta, train, spec["condition"]); x = transform(powers[str(p["history_ms"])], meta, enc)
    models = fit_models(x, meta.y.to_numpy(), ids, weights, RESEARCH_QUANTILES, p, spec["seed"])
    raw = raw_predict(models, x[test]); pred, crossing = rearrange(raw)
    frame = score_frame(meta, test, pred, RESEARCH_QUANTILES)
    for key in ("assessment", "horizon_ms", "fold", "seed", "condition"): frame[key] = spec[key]
    directory = Path(spec["directory"]); directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "predictions.parquet", index=False)
    bundle = {"models": models, "encoder": enc, "parameters": p, "quantiles": RESEARCH_QUANTILES.tolist(),
              "history_ms": p["history_ms"], "resolution_ms": 200}
    joblib.dump(bundle, directory / "full_evaluation_bundle.joblib", compress=3)
    np.savez_compressed(directory / "indices_weights.npz", train_sample=ids, eligible_train=train, test=test, weights=weights)
    names = power_feature_names(200, p["history_ms"]) + enc["names"]
    diagnostics = []
    for model, q in zip(models, RESEARCH_QUANTILES):
        for name, splits, gain in zip(names, model.booster_.feature_importance("split"), model.booster_.feature_importance("gain")):
            diagnostics.append({"feature": name, "quantile": q, "splits": int(splits), "gain": float(gain)})
    pd.DataFrame(diagnostics).to_csv(directory / "tree_diagnostics.csv", index=False)
    loaded = joblib.load(directory / "full_evaluation_bundle.joblib")
    check, _ = rearrange(raw_predict(loaded["models"], x[test])); np.testing.assert_allclose(check, pred, rtol=1e-10, atol=1e-10)
    summary, _ = summarize(frame); return spec | summary | {"crossing_rate": crossing,
        "tree_counts": [int(m.booster_.num_trees()) for m in models], "n_features": x.shape[1],
        "bundle_path": str(directory / "full_evaluation_bundle.joblib"), "prediction_path": str(directory / "predictions.parquet")}


def run_outer_evaluation(out, cfg, folds, selections, smoke=False):
    tasks = []
    for selected in selections:
        horizon, fold, p = int(selected["horizon_ms"]), int(selected["fold"]), selected["parameters"]
        meta, _ = load_cell(horizon); train_runs, test_runs = groups_from_corrected(folds, fold)
        train, test = np.flatnonzero(meta.run_id.isin(train_runs)), np.flatnonzero(meta.run_id.isin(test_runs))
        for condition in ("history_only", "history_meta"):
            for seed in ([cfg["deployment_seed"]] if smoke else cfg["seeds"]):
                ident = {"assessment": "grouped_outer", "horizon_ms": horizon, "fold": fold, "seed": seed,
                         "condition": condition, "parameters": p, "cap": cfg["max_rows_per_run"],
                         "train_ids": train.tolist(), "test_ids": test.tolist()}
                key = digest(ident); directory = out / "evaluation" / key; done = out / "checkpoints" / f"evaluation_{key}.json"
                if not done.exists(): tasks.append(ident | {"directory": str(directory), "done": str(done)})
    if tasks:
        with ProcessPoolExecutor(max_workers=min(cfg["workers"], len(tasks))) as pool:
            mapping = {pool.submit(evaluation_task, t): t for t in tasks}
            for future in as_completed(mapping):
                task = mapping[future]; result = future.result(); atomic_json(task["done"], result)
                print("evaluation", result["horizon_ms"], result["fold"], result["condition"], result["seed"], flush=True)
    records = [json.loads(p.read_text()) for p in (out / "checkpoints").glob("evaluation_*.json")]
    pd.DataFrame(records).to_csv(out / "outer_evaluation_tasks.csv", index=False)
    predictions = pd.concat([pd.read_parquet(r["prediction_path"]) for r in records], ignore_index=True)
    predictions.to_parquet(out / "outer_predictions.parquet", index=False)
    return predictions


def paired_summary(predictions, reference):
    rows, per_runs = [], []
    group_keys = ["horizon_ms", "fold", "seed", "condition"]
    stats = predictions.groupby(group_keys + ["run_id"], as_index=False).agg(crps_w=("crps_w", "mean"), mae_w=("mae_w", "mean"),
        coverage_80=("covered_80", "mean"), width_80_w=("width_80_w", "mean"), coverage_90=("covered_90", "mean"),
        width_90_w=("width_90_w", "mean"), n_targets=("y", "size"), cohort=("cohort", "first"))
    per_runs.append(stats)
    for horizon, h in stats.groupby("horizon_ms"):
        pair = h.pivot_table(index=["fold", "seed", "run_id", "cohort"], columns="condition", values="crps_w").reset_index()
        pair["delta_crps_w"] = pair.history_meta - pair.history_only
        pair["meta_gain"] = 1 - pair.history_meta / pair.history_only
        hist = h[h.condition == "history_only"]; meta = h[h.condition == "history_meta"]
        ref = reference[(reference.assessment == "grouped_outer") & (reference.resolution_ms == 200) &
                        (reference.horizon_ms == horizon) & (reference.pair == "improved_controlled")].iloc[0]
        rows.append({"assessment": "grouped_outer", "horizon_ms": horizon, "n_independent_runs": pair.run_id.nunique(),
            "history_only_crps_w": hist.crps_w.mean(), "history_meta_crps_w": meta.crps_w.mean(),
            "absolute_delta_crps_w": pair.delta_crps_w.mean(), "meta_gain": 1 - meta.crps_w.mean() / hist.crps_w.mean(),
            "history_reference_crps_w": ref.history_only, "meta_reference_crps_w": ref.history_all_meta,
            "history_change_vs_reference": 1-hist.crps_w.mean()/ref.history_only,
            "meta_change_vs_reference": 1-meta.crps_w.mean()/ref.history_all_meta,
            "median_mae_history_w": hist.mae_w.mean(), "median_mae_meta_w": meta.mae_w.mean(),
            "coverage_80_history": hist.coverage_80.mean(), "coverage_80_meta": meta.coverage_80.mean(),
            "width_80_history_w": hist.width_80_w.mean(), "width_80_meta_w": meta.width_80_w.mean(),
            "coverage_90_history": hist.coverage_90.mean(), "coverage_90_meta": meta.coverage_90.mean(),
            "width_90_history_w": hist.width_90_w.mean(), "width_90_meta_w": meta.width_90_w.mean()})
    return pd.DataFrame(rows), pd.concat(per_runs, ignore_index=True)


def fit_final(out, cfg, dev, holdout, selections):
    model_root = HERE / "models" / cfg["model_version"]; model_root.mkdir(parents=True, exist_ok=True)
    package_items = {"history_only": {}, "history_meta": {}}
    all_holdout = []
    for selected in selections:
        horizon, p = int(selected["horizon_ms"]), selected["parameters"]; meta, powers = load_cell(horizon)
        train = np.flatnonzero(meta.run_id.isin(dev.run_id)); test = np.flatnonzero(meta.run_id.isin(holdout.run_id))
        ids = choose_per_run(meta, train, cfg["max_rows_per_run"]); weights = equal_run_weights(meta, ids)
        for condition in ("history_only", "history_meta"):
            enc = fit_encoder(meta, train, condition); x = transform(powers[str(p["history_ms"])], meta, enc)
            models = fit_models(x, meta.y.to_numpy(), ids, weights, RESEARCH_QUANTILES, p, cfg["deployment_seed"])
            full = {"models": models, "encoder": enc, "parameters": p, "quantiles": RESEARCH_QUANTILES.tolist(),
                    "history_ms": p["history_ms"], "resolution_ms": 200}
            full_path = model_root / f"full_h{horizon}_{condition}.joblib"; joblib.dump(full, full_path, compress=3)
            keep = [3, 7, 11]; item = {"models": [models[i] for i in keep], "encoder": enc, "parameters": p,
                "quantiles": [.1,.5,.9], "history_ms": p["history_ms"]}; package_items[condition][str(horizon)] = item
            raw15 = raw_predict(models, x[test]); pred15, cross15 = rearrange(raw15)
            raw3 = raw15[:, keep]; pred3, cross3 = rearrange(raw3)
            f = score_frame(meta, test, pred15, RESEARCH_QUANTILES)
            f["assessment"] = "reused_holdout"; f["horizon_ms"] = horizon; f["condition"] = condition
            f["q10_export"] = pred3[:,0]; f["q50_export"] = pred3[:,1]; f["q90_export"] = pred3[:,2]
            f["export_q50_abs_change_w"] = np.abs(pred3[:,1]-pred15[:,7])
            f["export_crossing_before"] = cross3; f["full_crossing_before"] = cross15
            all_holdout.append(f)
            d = model_root / f"h{horizon}_{condition}"; d.mkdir(exist_ok=True)
            np.savez_compressed(d / "training_identities_weights.npz", indices=ids, weights=weights,
                                run_id=meta.iloc[ids].run_id.to_numpy(), target_index=ids)
            atomic_json(d / "feature_schema.json", {"power_features": power_feature_names(200, p["history_ms"]),
                "metadata_features": enc["names"], "encoder": enc, "target": f"P_200ms(t + {horizon} ms)",
                "units": "W", "power_scope": "sum of eight GPUs in one node; CPU excluded"})
    for condition, horizons in package_items.items():
        package = {"model_version": cfg["model_version"], "condition": condition, "resolution_ms": 200,
            "power_scope": "GPU sum (8 GPUs); CPU excluded", "ordering_correction": "row-wise ascending sort",
            "horizons": horizons}
        joblib.dump(package, model_root / f"{condition}_deployment.joblib", compress=3)
    hold = pd.concat(all_holdout, ignore_index=True); hold.to_parquet(out / "reused_holdout_predictions.parquet", index=False)
    return model_root, hold


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=str(HERE / "configs/lite_20260920.json")); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(); cfg = json.loads(Path(args.config).read_text())
    experiment = cfg["experiment_id"] + ("_smoke" if args.smoke else "")
    out = HERE / "outputs" / experiment; out.mkdir(parents=True, exist_ok=True)
    for d in ("checkpoints", "training_samples", "evaluation", "plots"): (out / d).mkdir(exist_ok=True)
    signature = digest(cfg | {"smoke": args.smoke}); signature_path = out / "configuration_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text())["sha256"] != signature: raise ValueError("incompatible checkpoint configuration")
    atomic_json(signature_path, {"sha256": signature}); atomic_json(out / "effective_config.json", cfg | {"smoke": args.smoke})
    status = {"status": "running", "started_utc": time.strftime("%FT%TZ", time.gmtime()), "pid": os.getpid()}; atomic_json(out / "status.json", status)
    log = (out / "execution.log").open("a", buffering=1); old_stdout = sys.stdout
    class Tee:
        def write(self, x): old_stdout.write(x); log.write(x)
        def flush(self): old_stdout.flush(); log.flush()
    sys.stdout = Tee()
    try:
        shutil.copy2(CORRECTED / "input_manifest.csv", out / "source_manifest.csv")
        shutil.copy2(CORRECTED / "splits/folds.csv", out / "corrected_folds.csv")
        inv = pd.read_csv(CORRECTED / "input_manifest.csv"); dev = pd.read_csv(CORRECTED / "development_runs.csv"); hold = pd.read_csv(CORRECTED / "reused_holdout_runs.csv")
        dev = dev.merge(inv[["run_id", "eligibility", "previously_evaluated"]], on="run_id", how="left")
        hold = hold.merge(inv[["run_id", "eligibility", "previously_evaluated"]], on="run_id", how="left")
        inv[["run_id", "cohort", "eligibility", "reserved_confirmation", "previously_evaluated"]].to_csv(out / "run_inventory.csv", index=False)
        dev.to_csv(out / "development_runs.csv", index=False); hold.to_csv(out / "reused_holdout_runs.csv", index=False)
        folds = pd.read_csv(CORRECTED / "splits/folds.csv"); final_folds = make_final_folds(dev, cfg["inner_folds"]); final_folds.to_csv(out / "development_folds.csv", index=False)
        schemas = []
        for history in cfg["histories_ms"]: schemas += [{"history_ms": history, "feature": x} for x in power_feature_names(200, history)]
        pd.DataFrame(schemas).to_csv(out / "power_feature_schemas.csv", index=False)
        atomic_json(out / "environment.json", {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "model_threads": cfg["model_threads"], "workers": cfg["workers"], "maxrss_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss"})
        outer_scores = run_candidate_search(out, cfg, "outer", folds, final_folds, args.smoke)
        outer_selections = select_candidates(outer_scores, cfg, "outer"); atomic_json(out / "outer_selections.json", outer_selections)
        outer_pred = run_outer_evaluation(out, cfg, folds, outer_selections, args.smoke)
        reference = pd.read_csv(CORRECTED / "comparison_summary.csv")
        summary, per_run = paired_summary(outer_pred, reference); summary.to_csv(out / "comparison_summary.csv", index=False); per_run.to_csv(out / "per_run_metrics.csv", index=False)
        if args.smoke:
            status.update(status="completed", finished_utc=time.strftime("%FT%TZ", time.gmtime()), note="end-to-end smoke through grouped outer evaluation")
            atomic_json(out / "status.json", status); return
        final_scores = run_candidate_search(out, cfg, "final", folds, final_folds, False)
        final_selections = select_candidates(final_scores, cfg, "final"); atomic_json(out / "final_selections.json", final_selections)
        model_root, hold_pred = fit_final(out, cfg, dev, hold, final_selections)
        status.update(status="trained", model_root=str(model_root), maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(out / "status.json", status)
        print("TRAINING_COMPLETE", model_root, flush=True)
    except Exception as exc:
        status.update(status="failed", error=repr(exc), finished_utc=time.strftime("%FT%TZ", time.gmtime())); atomic_json(out / "status.json", status); raise
    finally:
        sys.stdout = old_stdout; log.close()


if __name__ == "__main__": main()
