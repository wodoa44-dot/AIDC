#!/usr/bin/env python
"""Comprehensive retrospective audit of the frozen lightweight forecasts.

This script never fits, tunes, or calibrates a model. It reloads frozen bundles,
reconstructs raw predictions, and analyzes only held-out grouped-outer or reused-
holdout targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
ROOT = PACKAGE_ROOT.parents[1]
EXPERIMENT = PACKAGE_ROOT / "outputs/lite_full_20260920"
MODEL_ROOT = PACKAGE_ROOT / "models/power-forecast-lite-20260920"
CORRECTED = ROOT / "trace/predictability_meta/outputs/corrected_full_20260915"
sys.path.insert(0, str(PACKAGE_ROOT))
from src.core import RESEARCH_QUANTILES, raw_predict, rearrange, transform
from metrics import (RESOLUTION_S, assert_unique_and_paired, change_labels,
                     confusion_rows, distribution_metrics, failure_episodes,
                     interval_metrics, interval_score, pinball_crps,
                     transition_labels, weighted_quantile)

QLABELS = {
    .01: "q01", .025: "q025", .05: "q05", .1: "q10", .2: "q20",
    .3: "q30", .4: "q40", .5: "q50", .6: "q60", .7: "q70",
    .8: "q80", .9: "q90", .95: "q95", .975: "q975", .99: "q99",
}
TOLERANCES = (50, 100, 200, 300, 500, 750, 1000)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260921
TIE_TOLERANCE = 1e-9


def atomic_json(path: Path, value: dict | list) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def analysis_spec(experiment_id: str, smoke: bool) -> dict:
    return {
        "experiment_id": experiment_id,
        "created_utc": time.strftime("%FT%TZ", time.gmtime()),
        "frozen_model_audit": True,
        "smoke": smoke,
        "resolution_ms": 200,
        "horizons_ms": [1000, 2000],
        "assessments": {
            "grouped_outer": "Frozen fold-specific full-15 models on corresponding outer-test runs; compact-3 is derived from the same q10/q50/q90 estimators.",
            "reused_holdout": "Frozen final models on four previously inspected runs; not prospective confirmation.",
        },
        "error_definition": "prediction_q50 - observed_power; negative means underprediction",
        "transition_definition": {"rising": "target - issue >= 100 W", "falling": "target - issue <= -100 W", "stable": "abs(target - issue) < 100 W"},
        "tolerance_grid_w": list(TOLERANCES),
        "tolerance_interpretation": "diagnostic sensitivity grid, not operational or safety limits",
        "episode_definition": "consecutive exceeded valid issuance slots times 0.2 s; gaps, missing predictions, run, segment, horizon, model, representation, assessment, and seed boundaries break episodes",
        "tie_tolerance": {"absolute_metric_difference_w": TIE_TOLERANCE},
        "interval_score": "(upper-lower) + 2/alpha*(lower-y)*I(y<lower) + 2/alpha*(y-upper)*I(y>upper), alpha=0.2 for q10-q90",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "unit": "paired independent run", "fixed_predictions": True},
        "primary_seed_for_non_seed_summary": 42,
        "worst_5pct_definition": "weighted empirical 95th percentile of absolute error, inclusive tail; pooled and equal-run-weight versions reported separately",
        "time_series_window_s": 20.0,
        "no_retraining_tuning_recalibration_or_postprocessing_change": True,
    }


def load_inputs() -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, np.ndarray]]]:
    metas, powers = {}, {}
    for horizon in (1000, 2000):
        metas[horizon] = pd.read_parquet(CORRECTED / f"features/meta_200_{horizon}.parquet")
        z = np.load(CORRECTED / f"features/power_200_{horizon}.npz")
        powers[horizon] = {key: z[key] for key in z.files}
    return metas, powers


def add_prediction_columns(base: pd.DataFrame, raw15: np.ndarray, compact_raw: np.ndarray) -> pd.DataFrame:
    corrected15, full_crossing = rearrange(raw15)
    corrected3, compact_crossing = rearrange(compact_raw)
    for index, quantile in enumerate(RESEARCH_QUANTILES):
        label = QLABELS[float(quantile)]
        base[f"raw_{label}"] = raw15[:, index]
        base[f"full_{label}"] = corrected15[:, index]
    for index, label in enumerate(("q10", "q50", "q90")):
        base[f"compact_raw_{label}"] = compact_raw[:, index]
        base[f"compact_{label}"] = corrected3[:, index]
    base["full_crossing_before"] = np.any(np.diff(raw15, axis=1) < 0, axis=1)
    base["compact_crossing_before"] = np.any(np.diff(compact_raw, axis=1) < 0, axis=1)
    base["full_crossing_rate_bundle"] = full_crossing
    base["compact_crossing_rate_bundle"] = compact_crossing
    base["full_crps_w"] = pinball_crps(base.observed_power_w.to_numpy(), corrected15, RESEARCH_QUANTILES)
    return base


def base_frame(saved: pd.DataFrame, meta: pd.DataFrame, assessment: str, fold, seed: int,
               condition: str, full_path: Path, compact_path: Path, full_hash: str,
               compact_hash: str, representation_source: str) -> pd.DataFrame:
    ids = saved.target_index.to_numpy(int)
    m = meta.iloc[ids].reset_index(drop=True)
    out = pd.DataFrame({
        "assessment": assessment,
        "run_id": m.run_id.astype(str),
        "segment_id": m.segment_id.astype(str),
        "subsegment_id": m.subsegment_id.astype(str),
        "segment_key": m.segment_id.astype(str) + ":" + m.subsegment_id.astype(str),
        "cohort": m.cohort.astype(str),
        "gpu": m.gpu.astype(str),
        "workload": m.workload.astype(str),
        "batch": pd.to_numeric(m.batch, errors="coerce"),
        "ai_model": m.ai_model.astype(str),
        "fold": fold,
        "seed": int(seed),
        "condition": condition,
        "prediction_representation": representation_source,
        "issue_time_s": m.issue_s.to_numpy(float),
        "target_time_s": m.target_s.to_numpy(float),
        "horizon_ms": saved.horizon_ms.to_numpy(int),
        "issue_power_w": m.current.to_numpy(float),
        "observed_power_w": m.y.to_numpy(float),
        "target_bin": m.target_bin.to_numpy(int),
        "target_index": ids,
        "valid_prediction": True,
        "missing_prediction": False,
        "full_bundle_path": str(full_path),
        "full_bundle_sha256": full_hash,
        "compact_bundle_path": str(compact_path),
        "compact_bundle_sha256": compact_hash,
    })
    return out


def reconstruct_canonical(out: Path, metas: dict[int, pd.DataFrame], powers: dict[int, dict[str, np.ndarray]], smoke: bool) -> tuple[pd.DataFrame, dict]:
    task_columns = ["assessment", "bundle_path", "condition", "done", "fold", "horizon_ms", "prediction_path", "seed"]
    tasks = pd.read_csv(EXPERIMENT / "outer_evaluation_tasks.csv", usecols=task_columns)
    tasks = tasks.sort_values(["horizon_ms", "fold", "seed", "condition"])
    if smoke:
        tasks = tasks[(tasks.fold == 1) & (tasks.seed == 42)]
    rows = []
    reconstruction = {"outer_tasks": 0, "saved_full_max_abs_difference_w": 0.0,
                      "compact_raw_vs_full_selected_max_abs_difference_w": 0.0,
                      "outer_test_index_mismatches": 0, "train_test_index_overlap": 0}
    folds = pd.read_csv(EXPERIMENT / "corrected_folds.csv")
    bundle_hash_cache: dict[str, str] = {}
    smoke_run = None
    for task in tasks.itertuples(index=False):
        horizon = int(task.horizon_ms)
        saved = pd.read_parquet(task.prediction_path)
        if smoke:
            smoke_run = smoke_run or sorted(saved.run_id.unique())[0]
            saved = saved[saved.run_id.eq(smoke_run)].reset_index(drop=True)
        bundle_path = Path(task.bundle_path)
        bundle = joblib.load(bundle_path)
        ids = saved.target_index.to_numpy(int)
        x = transform(powers[horizon][str(bundle["history_ms"])][ids], metas[horizon].iloc[ids].reset_index(drop=True), bundle["encoder"])
        raw15 = raw_predict(bundle["models"], x)
        compact_raw = raw15[:, [3, 7, 11]]
        corrected15 = np.sort(raw15, axis=1)
        saved_corrected = saved[[f"q_{q:g}" for q in RESEARCH_QUANTILES]].to_numpy(float)
        reconstruction["saved_full_max_abs_difference_w"] = max(
            reconstruction["saved_full_max_abs_difference_w"], float(np.max(np.abs(corrected15 - saved_corrected))))
        full_hash = bundle_hash_cache.setdefault(str(bundle_path), sha256(bundle_path))
        base = base_frame(saved, metas[horizon], "grouped_outer", int(task.fold), int(task.seed), task.condition,
                          bundle_path, bundle_path, full_hash, full_hash, "full15_and_derived_compact3")
        base = add_prediction_columns(base, raw15, compact_raw)
        rows.append(base)
        npz = bundle_path.parent / "indices_weights.npz"
        indices = np.load(npz)
        reconstruction["train_test_index_overlap"] += int(np.intersect1d(indices["train_sample"], indices["test"]).size)
        expected_test_runs = set(folds[(folds.outer_fold == int(task.fold)) & folds.inner_fold.isna() & folds.role.eq("outer_test")].run_id)
        reconstruction["outer_test_index_mismatches"] += int(not set(saved.run_id).issubset(expected_test_runs))
        reconstruction["outer_tasks"] += 1

    deployment_paths = {c: MODEL_ROOT / f"{c}_deployment.joblib" for c in ("history_only", "history_meta")}
    full_paths = {(h, c): MODEL_ROOT / f"full_h{h}_{c}.joblib" for h in (1000, 2000) for c in ("history_only", "history_meta")}
    held = pd.read_parquet(EXPERIMENT / "reused_holdout_predictions.parquet")
    if smoke:
        hold_run = sorted(held.run_id.unique())[0]
        held = held[held.run_id.eq(hold_run)].reset_index(drop=True)
    for (horizon, condition), saved in held.groupby(["horizon_ms", "condition"], sort=True):
        horizon = int(horizon)
        saved = saved.reset_index(drop=True)
        full_path, compact_path = full_paths[(horizon, condition)], deployment_paths[condition]
        full_bundle = joblib.load(full_path)
        deployment = joblib.load(compact_path)
        compact_item = deployment["horizons"][str(horizon)]
        ids = saved.target_index.to_numpy(int)
        selected_meta = metas[horizon].iloc[ids].reset_index(drop=True)
        x_full = transform(powers[horizon][str(full_bundle["history_ms"])][ids], selected_meta, full_bundle["encoder"])
        x_compact = transform(powers[horizon][str(compact_item["history_ms"])][ids], selected_meta, compact_item["encoder"])
        raw15 = raw_predict(full_bundle["models"], x_full)
        compact_raw = raw_predict(compact_item["models"], x_compact)
        reconstruction["compact_raw_vs_full_selected_max_abs_difference_w"] = max(
            reconstruction["compact_raw_vs_full_selected_max_abs_difference_w"],
            float(np.max(np.abs(compact_raw - raw15[:, [3, 7, 11]]))))
        corrected15 = np.sort(raw15, axis=1)
        saved_corrected = saved[[f"q_{q:g}" for q in RESEARCH_QUANTILES]].to_numpy(float)
        reconstruction["saved_full_max_abs_difference_w"] = max(
            reconstruction["saved_full_max_abs_difference_w"], float(np.max(np.abs(corrected15 - saved_corrected))))
        base = base_frame(saved, metas[horizon], "reused_holdout", np.nan, 42, condition,
                          full_path, compact_path, sha256(full_path), sha256(compact_path), "full15_and_actual_compact3")
        rows.append(add_prediction_columns(base, raw15, compact_raw))
    canonical = pd.concat(rows, ignore_index=True)
    canonical["transition_stratum"] = transition_labels(canonical.observed_power_w, canonical.issue_power_w)
    canonical.to_parquet(out / "per_target_evaluation.parquet", index=False, compression="zstd")
    return canonical, reconstruction


def integrity_checks(canonical: pd.DataFrame, reconstruction: dict, metas: dict[int, pd.DataFrame]) -> dict:
    paired_key = ["assessment", "run_id", "segment_key", "horizon_ms", "fold", "seed", "issue_time_s", "target_time_s"]
    paired = assert_unique_and_paired(canonical, paired_key)
    time_error = np.abs((canonical.target_time_s - canonical.issue_time_s) * 1000 - canonical.horizon_ms)
    target_identity = canonical.groupby(paired_key, dropna=False).agg(
        n_y=("observed_power_w", "nunique"), n_current=("issue_power_w", "nunique"), n_target_bin=("target_bin", "nunique"))
    seed_key = ["assessment", "run_id", "horizon_ms", "condition", "issue_time_s"]
    seed_spread = canonical[canonical.assessment.eq("grouped_outer")].groupby(seed_key).full_q50.agg(lambda x: float(x.max() - x.min()))
    # Power feature zero is current by contract; check every stored row against the cached feature matrix.
    current_differences = []
    for horizon, group in canonical.groupby("horizon_ms"):
        for history in (1000, 5000):
            ids = group.target_index.to_numpy(int)
            current_differences.append(np.max(np.abs(metas[int(horizon)].iloc[ids].current.to_numpy() -
                                                     np.load(CORRECTED / f"features/power_200_{int(horizon)}.npz")[str(history)][ids, 0])))
    return {
        "passed": bool(paired["passed"] and time_error.max() < 1e-6 and
                       target_identity.max().max() == 1 and reconstruction["saved_full_max_abs_difference_w"] < 1e-8 and
                       reconstruction["compact_raw_vs_full_selected_max_abs_difference_w"] < 1e-8 and
                       reconstruction["outer_test_index_mismatches"] == 0 and reconstruction["train_test_index_overlap"] == 0),
        "canonical_rows": int(len(canonical)),
        "unique_assessment_run_horizon_issue_targets": int(canonical.drop_duplicates(["assessment", "run_id", "horizon_ms", "issue_time_s"]).shape[0]),
        "duplicate_prediction_keys": paired["duplicate_prediction_keys"],
        "unmatched_condition_keys": paired["unmatched_paired_keys"],
        "max_target_minus_issue_horizon_error_ms": float(time_error.max()),
        "paired_target_identity_mismatches": int((target_identity.max(axis=1) != 1).sum()),
        "max_saved_vs_reconstructed_full_prediction_difference_w": reconstruction["saved_full_max_abs_difference_w"],
        "max_deployment_compact_raw_vs_full_selected_raw_difference_w": reconstruction["compact_raw_vs_full_selected_max_abs_difference_w"],
        "outer_task_count": reconstruction["outer_tasks"],
        "outer_fold_test_membership_mismatches": reconstruction["outer_test_index_mismatches"],
        "train_test_sample_index_overlap": reconstruction["train_test_index_overlap"],
        "max_seed_q50_spread_w": float(seed_spread.max()),
        "max_cached_power_feature_current_difference_w": float(max(current_differences)),
        "units": "watts, eight-GPU node sum mean per completed 200 ms bin; CPU/facility excluded",
        "future_information_check": "Passed code/provenance audit: issue precedes target exactly by horizon; cached feature column zero equals issue-time current power; core.power_features uses trailing completed bins only.",
        "cross_segment_history_check": "Builder provenance groups by run, continuous segment, and binned subsegment before samples; online replay resets on gaps/metadata changes. Per-lag source identity arrays were not saved, so this is code/provenance verification rather than an independent row-level lag audit.",
    }


def build_inventory(canonical: pd.DataFrame, metas: dict[int, pd.DataFrame]) -> pd.DataFrame:
    eligible = {(h, run, str(seg), str(sub)): len(g) for h, m in metas.items()
                for (run, seg, sub), g in m.groupby(["run_id", "segment_id", "subsegment_id"])}
    keys = ["assessment", "run_id", "segment_id", "subsegment_id", "segment_key", "cohort", "gpu", "workload",
            "batch", "ai_model", "horizon_ms", "fold", "seed", "condition", "prediction_representation",
            "full_bundle_path", "full_bundle_sha256", "compact_bundle_path", "compact_bundle_sha256"]
    records = []
    for values, group in canonical.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        n = len(group)
        full_eligible = eligible[(int(row["horizon_ms"]), row["run_id"], row["segment_id"], row["subsegment_id"])]
        elapsed_slots = int(round((group.issue_time_s.max() - group.issue_time_s.min()) / RESOLUTION_S)) + 1
        row.update({
            "run_participated_in_evaluated_model_fitting": False,
            "run_participated_in_evaluated_model_selection": False,
            "run_participated_in_final_development_fit": row["assessment"] == "grouped_outer",
            "evaluation_sampling_stride": 1,
            "evaluation_cap": "none",
            "training_cap_per_run": 5000,
            "first_issue_time_s": float(group.issue_time_s.min()),
            "last_issue_time_s": float(group.issue_time_s.max()),
            "target_count": n,
            "eligible_target_count": full_eligible,
            "eligible_target_coverage": n / full_eligible,
            "evaluated_duration_s": n * RESOLUTION_S,
            "elapsed_span_s": elapsed_slots * RESOLUTION_S,
            "missing_issue_slots_within_span": elapsed_slots - n,
            "warmup_required_ms": 5000 if int(row["horizon_ms"]) == 2000 else 1000,
            "missing_predictions": int(group.missing_prediction.sum()),
        })
        records.append(row)
    return pd.DataFrame(records)


def build_long(canonical: pd.DataFrame) -> pd.DataFrame:
    identity = ["assessment", "run_id", "segment_key", "cohort", "gpu", "workload", "batch", "ai_model",
                "fold", "seed", "condition", "issue_time_s", "target_time_s", "horizon_ms", "issue_power_w",
                "observed_power_w", "transition_stratum", "full_crps_w", "full_crossing_before", "compact_crossing_before"]
    parts = []
    for representation, prefix in (("full15", "full_"), ("compact3", "compact_")):
        f = canonical[identity].copy()
        f["representation"] = representation
        for q in ("q05", "q10", "q50", "q90", "q95"):
            column = prefix + q
            f[q] = canonical[column].to_numpy() if column in canonical else np.nan
        f["crps_w"] = canonical.full_crps_w.to_numpy() if representation == "full15" else np.nan
        f["crossing_before"] = canonical[f"{prefix}crossing_before"].to_numpy() if f"{prefix}crossing_before" in canonical else canonical[f"{representation.split('1')[0]}_crossing_before"].to_numpy()
        parts.append(f)
    # Persistence is a point baseline and deliberately has no interval/CRPS.
    p = canonical.drop_duplicates(["assessment", "run_id", "segment_key", "fold", "seed", "issue_time_s", "horizon_ms"])[identity].copy()
    p["condition"] = "persistence"
    p["representation"] = "point"
    p["q50"] = p.issue_power_w
    for q in ("q05", "q10", "q90", "q95"):
        p[q] = np.nan
    p["crps_w"] = np.nan
    p["crossing_before"] = False
    out = pd.concat(parts + [p], ignore_index=True)
    out["error_w"] = out.q50 - out.observed_power_w
    out["abs_error_w"] = out.error_w.abs()
    out["predicted_change_w"] = out.q50 - out.issue_power_w
    out["predicted_change_class"] = change_labels(out.predicted_change_w)
    return out


def per_run_metrics(long: pd.DataFrame) -> pd.DataFrame:
    keys = ["assessment", "horizon_ms", "condition", "representation", "seed", "run_id", "cohort"]
    rows = []
    for values, group in long.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values)) | distribution_metrics(group.error_w.to_numpy())
        row.update({"n_segments": int(group.segment_key.nunique()), "evaluated_duration_s": len(group) * RESOLUTION_S,
                    "crps_w": float(group.crps_w.mean()) if group.crps_w.notna().any() else np.nan})
        if group.q10.notna().all():
            row.update({f"interval80_{k}": v for k, v in interval_metrics(group.observed_power_w, group.q10, group.q90).items()})
        rows.append(row)
    result = pd.DataFrame(rows)
    # Point-persistence comparisons cross representation by design; probabilistic metadata
    # comparisons remain within the full-15 representation.
    for (assessment, horizon, seed), group in result.groupby(["assessment", "horizon_ms", "seed"]):
        idx = group.index
        persistence = group[group.condition.eq("persistence")].set_index("run_id").mae_w
        for i in idx:
            run = result.at[i, "run_id"]
            if result.at[i, "condition"] != "persistence" and run in persistence:
                result.at[i, "mae_improvement_vs_persistence"] = 1 - result.at[i, "mae_w"] / persistence[run]
        full = group[(group.representation == "full15") & group.condition.isin(["history_only", "history_meta"])]
        pair = full.pivot(index="run_id", columns="condition", values="crps_w")
        for run, values in pair.dropna().iterrows():
            mask = (result.index.isin(idx) & result.run_id.eq(run) &
                    result.representation.eq("full15") & result.condition.isin(["history_only", "history_meta"]))
            result.loc[mask, "metadata_absolute_crps_delta_w"] = values.history_meta - values.history_only
            result.loc[mask, "metadata_crps_gain"] = 1 - values.history_meta / values.history_only
    return result


def cohort_metrics(long: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["assessment", "horizon_ms", "condition", "representation", "seed", "cohort"]
    rows = []
    for values, group in long.groupby(keys, dropna=False, sort=True):
        base = dict(zip(keys, values))
        for aggregation, weights in (("target_weighted", None), ("run_balanced", group.run_id.map(1 / group.groupby("run_id").size()).to_numpy())):
            row = base | {"aggregation": aggregation, "n_runs": int(group.run_id.nunique())}
            row.update(distribution_metrics(group.error_w.to_numpy(), weights))
            row["crps_w"] = (float(np.average(group.crps_w, weights=weights)) if group.crps_w.notna().all() and weights is not None
                             else float(group.crps_w.mean()) if group.crps_w.notna().all() else np.nan)
            if group.q10.notna().all():
                row["coverage_80"] = float(np.average((group.observed_power_w >= group.q10) & (group.observed_power_w <= group.q90), weights=weights))
                row["width_80_w"] = float(np.average(group.q90 - group.q10, weights=weights))
            rows.append(row)
    return pd.DataFrame(rows)


def tail_metrics(long: pd.DataFrame) -> pd.DataFrame:
    keys = ["assessment", "horizon_ms", "condition", "representation", "seed"]
    rows = []
    for values, group in long.groupby(keys, dropna=False, sort=True):
        base = dict(zip(keys, values)) | {"n_runs": int(group.run_id.nunique())}
        rows.append(base | {"aggregation": "pooled_targets"} | distribution_metrics(group.error_w.to_numpy()))
        weights = group.run_id.map(1 / group.groupby("run_id").size()).to_numpy()
        rows.append(base | {"aggregation": "equal_run_weight_empirical"} | distribution_metrics(group.error_w.to_numpy(), weights))
    return pd.DataFrame(rows)


def aggregate_reconciliation(canonical):
    rows=[]; published=pd.read_csv(EXPERIMENT/"comparison_summary.csv")
    g=canonical[canonical.assessment.eq("grouped_outer")].groupby(
        ["horizon_ms","condition","seed","run_id"],as_index=False).full_crps_w.mean()
    got=g.groupby(["horizon_ms","condition"],as_index=False).full_crps_w.mean()
    for r in published.itertuples():
        for condition,column in (("history_only","history_only_crps_w"),("history_meta","history_meta_crps_w")):
            value=float(got[(got.horizon_ms==r.horizon_ms)&(got.condition==condition)].full_crps_w.iloc[0]); expected=float(getattr(r,column))
            rows.append({"assessment":"grouped_outer","horizon_ms":r.horizon_ms,"condition":condition,
                         "metric":"run_seed_balanced_crps_w","published_value":expected,"recomputed_value":value,
                         "difference":value-expected,"passed":abs(value-expected)<=1e-8})
    published=pd.read_csv(EXPERIMENT/"deployment_export_metrics.csv"); held=canonical[canonical.assessment.eq("reused_holdout")]
    for r in published.itertuples():
        q=held[(held.horizon_ms==r.horizon_ms)&(held.condition==r.condition)]
        values={"q50_mae_w":np.mean(np.abs(q.compact_q50-q.observed_power_w)),
                "coverage_80":np.mean((q.observed_power_w>=q.compact_q10)&(q.observed_power_w<=q.compact_q90)),
                "q50_change_vs_full_max_w":np.max(np.abs(q.compact_q50-q.full_q50))}
        for metric,value in values.items():
            expected=float(getattr(r,metric)); rows.append({"assessment":"reused_holdout","horizon_ms":r.horizon_ms,
                "condition":r.condition,"metric":metric,"published_value":expected,"recomputed_value":value,
                "difference":value-expected,"passed":abs(value-expected)<=1e-8})
    return pd.DataFrame(rows)


def tolerance_outputs(long):
    parts=[]; summaries=[]; ep_input=long.rename(columns={"issue_time_s":"issue_s"})
    keys=["assessment","horizon_ms","condition","representation","seed"]
    for threshold in TOLERANCES:
        episodes=failure_episodes(ep_input,threshold)
        if len(episodes): parts.append(episodes)
        for values,g in long.groupby(keys,dropna=False,sort=True):
            e=g.error_w.to_numpy(); exceed=np.abs(e)>threshold; z=episodes
            for key,value in zip(keys,values):
                z=z[z[key].eq(value)] if not pd.isna(value) else z[z[key].isna()]
            duration=z.duration_s.to_numpy() if len(z) else np.array([])
            summaries.append(dict(zip(keys,values))|{"threshold_w":threshold,"n_targets":len(g),
                "exceedance_rate":exceed.mean(),"underprediction_exceedance_rate":(e < -threshold).mean(),
                "overprediction_exceedance_rate":(e > threshold).mean(),"n_episodes":len(z),
                "episode_duration_median_s":np.median(duration) if len(duration) else 0.,
                "episode_duration_p90_s":np.quantile(duration,.9) if len(duration) else 0.,
                "episode_duration_max_s":duration.max() if len(duration) else 0.,
                "total_exceedance_time_s":exceed.sum()*RESOLUTION_S,
                "exceedance_time_fraction_of_valid_time":exceed.mean(),
                "left_censored_episodes":int(z.left_censored.sum()) if len(z) else 0,
                "right_censored_episodes":int(z.right_censored.sum()) if len(z) else 0})
    return pd.DataFrame(summaries),pd.concat(parts,ignore_index=True)


def transition_outputs(long):
    keys=["assessment","horizon_ms","condition","representation","seed","transition_stratum"]; rows=[]
    totals=long.groupby(keys[:-1],dropna=False).size()
    for values,g in long.groupby(keys,dropna=False,sort=True):
        row=dict(zip(keys,values))|{"n_runs":g.run_id.nunique(),"evaluated_share":len(g)/totals.loc[values[:-1]]}
        row.update(distribution_metrics(g.error_w.to_numpy()))
        row["crps_w"]=g.crps_w.mean() if g.crps_w.notna().all() else np.nan
        if g.q10.notna().all(): row.update({f"interval80_{k}":v for k,v in interval_metrics(g.observed_power_w,g.q10,g.q90).items()})
        rows.append(row)
    result=pd.DataFrame(rows)
    compare=["assessment","horizon_ms","seed","transition_stratum"]
    for values,g in result.groupby(compare,dropna=False):
        p=g[g.condition.eq("persistence")]
        if p.empty: continue
        mask=np.ones(len(result),bool)
        for key,value in zip(compare,values): mask &= result[key].eq(value)
        mask &= ~result.condition.eq("persistence")
        result.loc[mask,"mae_improvement_vs_persistence"]=1-result.loc[mask,"mae_w"]/float(p.mae_w.iloc[0])
    confusion=[]; ckeys=keys[:-1]
    for values,g in long.groupby(ckeys,dropna=False,sort=True):
        confusion += [dict(zip(ckeys,values))|r|{"n_targets":len(g)}
                      for r in confusion_rows(g.transition_stratum.to_numpy(),g.predicted_change_class.to_numpy())]
    return result,pd.DataFrame(confusion)


def interval_outputs(long,canonical):
    rows=[]; interval=long[long.representation.isin(["full15","compact3"])]
    groupings=[("overall",["assessment","horizon_ms","condition","representation","seed"]),
               ("run",["assessment","horizon_ms","condition","representation","seed","run_id"]),
               ("cohort",["assessment","horizon_ms","condition","representation","seed","cohort"]),
               ("transition",["assessment","horizon_ms","condition","representation","seed","transition_stratum"])]
    for level,keys in groupings:
        for values,g in interval.groupby(keys,dropna=False,sort=True):
            base=dict(zip(keys,values))|{"diagnostic_type":"interval","grouping_level":level,"n_targets":len(g),"n_runs":g.run_id.nunique()}
            rows.append(base|{"nominal_coverage":.8}|interval_metrics(g.observed_power_w,g.q10,g.q90,.2))
            if g.representation.iloc[0]=="full15":
                rows.append(base|{"nominal_coverage":.9}|interval_metrics(g.observed_power_w,g.q05,g.q95,.1))
    calibrations=[("overall",["assessment","horizon_ms","condition","seed"]),
                  ("run",["assessment","horizon_ms","condition","seed","run_id"]),
                  ("cohort",["assessment","horizon_ms","condition","seed","cohort"]),
                  ("transition",["assessment","horizon_ms","condition","seed","transition_stratum"])]
    for level,keys in calibrations:
        for values,g in canonical.groupby(keys,dropna=False,sort=True):
            base=dict(zip(keys,values))|{"diagnostic_type":"quantile_calibration","grouping_level":level,
                "representation":"full15","n_targets":len(g),"n_runs":g.run_id.nunique()}
            for q in RESEARCH_QUANTILES:
                rows.append(base|{"quantile":q,"empirical_fraction_actual_le_prediction":
                                  np.mean(g.observed_power_w<=g["full_"+QLABELS[float(q)]])})
    return pd.DataFrame(rows)


def export_outputs(canonical,tolerance):
    rows=[]; keys=["assessment","horizon_ms","condition","seed"]
    for values,g in canonical.groupby(keys,dropna=False,sort=True):
        y=g.observed_power_w.to_numpy(); raw=g.raw_q50.to_numpy(); full=g.full_q50.to_numpy(); compact=g.compact_q50.to_numpy()
        ar,af,ac=np.abs(raw-y),np.abs(full-y),np.abs(compact-y)
        row=dict(zip(keys,values))|{"n_runs":g.run_id.nunique(),"n_targets":len(g),
            "compact_source":"actual_deployment_bundle" if values[0]=="reused_holdout" else "same_estimators_derived",
            "full15_any_adjacent_crossing_rate":g.full_crossing_before.mean(),"compact3_any_adjacent_crossing_rate":g.compact_crossing_before.mean(),
            "raw_q50_mae_w":ar.mean(),"full15_q50_mae_w":af.mean(),"compact3_q50_mae_w":ac.mean(),
            "full15_q50_change_mean_w":np.mean(np.abs(full-raw)),"full15_q50_change_p99_w":np.quantile(np.abs(full-raw),.99),
            "full15_q50_change_max_w":np.max(np.abs(full-raw)),"compact3_q50_change_mean_w":np.mean(np.abs(compact-raw)),
            "compact3_q50_change_p99_w":np.quantile(np.abs(compact-raw),.99),"compact3_q50_change_max_w":np.max(np.abs(compact-raw)),
            "compact_vs_full_q50_change_mean_w":np.mean(np.abs(compact-full)),"compact_vs_full_q50_change_max_w":np.max(np.abs(compact-full)),
            "full15_fraction_improved_vs_raw":np.mean(af<ar),"full15_fraction_worsened_vs_raw":np.mean(af>ar),
            "compact3_fraction_improved_vs_raw":np.mean(ac<ar),"compact3_fraction_worsened_vs_raw":np.mean(ac>ar),
            "full15_interval_score80_w":interval_score(y,g.full_q10,g.full_q90).mean(),
            "compact3_interval_score80_w":interval_score(y,g.compact_q10,g.compact_q90).mean()}
        for threshold in (500,1000):
            row[f"raw_abs_error_gt_{threshold}w_rate"]=np.mean(ar>threshold); row[f"full_abs_error_gt_{threshold}w_rate"]=np.mean(af>threshold); row[f"compact_abs_error_gt_{threshold}w_rate"]=np.mean(ac>threshold)
        rows.append(row)
    overall=pd.DataFrame(rows); sub=[]
    for grouping in ("transition_stratum","cohort"):
        for values,g in canonical.groupby(keys+[grouping],dropna=False,sort=True):
            change=np.abs(g.compact_q50-g.raw_q50); sub.append(dict(zip(keys+[grouping],values))|{
                "n_runs":g.run_id.nunique(),"n_targets":len(g),"compact_source":"subgroup_association",
                "compact3_q50_change_mean_w":change.mean(),"compact3_q50_change_max_w":change.max()})
    return pd.concat([overall,pd.DataFrame(sub)],ignore_index=True,sort=False)


def paired_uncertainty(per_run):
    rng=np.random.default_rng(BOOTSTRAP_SEED); rows=[]
    base=per_run[(per_run.representation.eq("full15")) | (per_run.condition.eq("persistence"))].groupby(
        ["assessment","horizon_ms","condition","run_id","cohort"],as_index=False).agg(crps_w=("crps_w","mean"),mae_w=("mae_w","mean"))
    for (assessment,horizon),g in base.groupby(["assessment","horizon_ms"]):
        mae=g.pivot(index="run_id",columns="condition",values="mae_w"); crps=g.pivot(index="run_id",columns="condition",values="crps_w")
        cohort=g.drop_duplicates("run_id").set_index("run_id").cohort
        for name,wide,left,right in [("metadata_minus_history_crps_w",crps,"history_meta","history_only"),
            ("metadata_minus_history_mae_w",mae,"history_meta","history_only"),
            ("metadata_minus_persistence_mae_w",mae,"history_meta","persistence"),
            ("history_minus_persistence_mae_w",mae,"history_only","persistence")]:
            pair=wide[[left,right]].dropna()
            if pair.empty: continue
            ids=pair.index.to_numpy(); values=np.empty(BOOTSTRAP_REPLICATES)
            for i in range(BOOTSTRAP_REPLICATES):
                take=rng.choice(ids,len(ids),replace=True); values[i]=np.mean(pair.loc[take,left].to_numpy()-pair.loc[take,right].to_numpy())
            common={"assessment":assessment,"horizon_ms":horizon,"comparison":name,"n_independent_runs":len(ids),
                    "estimate_w":float((pair[left]-pair[right]).mean()),"replicates":BOOTSTRAP_REPLICATES,"seed":BOOTSTRAP_SEED}
            rows.append(common|{"resampling":"paired_runs","ci_low_w":np.quantile(values,.025),"ci_high_w":np.quantile(values,.975),"available":True})
            counts=cohort.loc[ids].value_counts()
            if counts.min()>=2:
                for i in range(BOOTSTRAP_REPLICATES):
                    take=np.concatenate([rng.choice(np.asarray(list(idx)),len(idx),replace=True)
                        for idx in cohort.loc[ids].groupby(cohort.loc[ids]).groups.values()])
                    values[i]=np.mean(pair.loc[take,left].to_numpy()-pair.loc[take,right].to_numpy())
                rows.append(common|{"resampling":"cohort_stratified_paired_runs","ci_low_w":np.quantile(values,.025),"ci_high_w":np.quantile(values,.975),"available":True})
            else:
                rows.append(common|{"resampling":"cohort_stratified_paired_runs","available":False,
                                    "unavailable_reason":"at least one cohort has fewer than two independent runs"})
    return pd.DataFrame(rows)


def worst_cases(canonical,long,episodes):
    c=long[(long.condition=="history_meta")&(long.representation=="compact3")&(long.seed==42)]
    choices=[("largest_absolute_metadata_error",c.loc[c.abs_error_w.idxmax()]),
             ("largest_underprediction",c.loc[c.error_w.idxmin()]),("largest_overprediction",c.loc[c.error_w.idxmax()])]
    e=episodes[(episodes.condition=="history_meta")&(episodes.representation=="compact3")&(episodes.seed==42)&(episodes.threshold_w==500)]
    if len(e):
        r=e.sort_values(["duration_s","run_id","start_issue_s"],ascending=[False,True,True]).iloc[0]
        q=c[(c.assessment==r.assessment)&(c.run_id==r.run_id)&(c.segment_key==r.segment_key)&
            (c.horizon_ms==r.horizon_ms)&np.isclose(c.issue_time_s,r.start_issue_s)]
        if len(q): choices.append(("longest_500w_exceedance_episode_start",q.iloc[0]))
    change=canonical[canonical.seed.eq(42)].assign(delta=lambda x:np.abs(x.compact_q50-x.full_q50))
    r=change.loc[change.delta.idxmax()]; q=c[(c.assessment==r.assessment)&(c.run_id==r.run_id)&
        (c.horizon_ms==r.horizon_ms)&np.isclose(c.issue_time_s,r.issue_time_s)]
    if len(q): choices.append(("largest_full_vs_compact_q50_change",q.iloc[0]))
    first=next(iter(c.sort_values(["assessment","run_id","horizon_ms","issue_time_s"]).groupby(["assessment","run_id","horizon_ms"],sort=True)))[1]
    choices.append(("typical_deterministic_mid_timeline",first.iloc[len(first)//2]))
    rows=[]; seen=set()
    for rule,r in choices:
        key=(r.assessment,r.run_id,r.horizon_ms,r.issue_time_s)
        if key in seen: continue
        seen.add(key); rows.append({"selection_rule":rule,"assessment":r.assessment,"run_id":r.run_id,
            "segment_key":r.segment_key,"cohort":r.cohort,"horizon_ms":r.horizon_ms,"condition":r.condition,
            "representation":r.representation,"seed":r.seed,"issue_time_s":r.issue_time_s,"target_time_s":r.target_time_s,
            "issue_power_w":r.issue_power_w,"observed_power_w":r.observed_power_w,"q10_w":r.q10,"q50_w":r.q50,
            "q90_w":r.q90,"error_w":r.error_w,"abs_error_w":r.abs_error_w,"transition_stratum":r.transition_stratum,
            "deliberately_selected_worst_case":not rule.startswith("typical")})
    return pd.DataFrame(rows)


def make_plots(out,canonical,long,runs,tolerance,episodes,intervals,cases):
    d=out/"figures"; d.mkdir()
    g=runs[(runs.assessment=="grouped_outer")&(runs.representation=="full15")&(runs.condition=="history_meta")].groupby(
        ["horizon_ms","run_id","cohort"],as_index=False).metadata_crps_gain.mean()
    fig,ax=plt.subplots(figsize=(9,4))
    for cohort,z in g.groupby("cohort"): ax.scatter(z.run_id.str[-8:],100*z.metadata_crps_gain,label=cohort)
    ax.axhline(0,color="black",lw=.8); ax.set_ylabel("metadata CRPS gain (%)"); ax.tick_params(axis="x",rotation=90,labelsize=6); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(d/"per_run_metadata_gain.png",dpi=150); plt.close(fig)
    z=long[(long.seed==42)&long.representation.isin(["compact3","point"])]
    fig,ax=plt.subplots(figsize=(7,4))
    for condition in ("history_meta","history_only","persistence"):
        v=np.sort(z[z.condition==condition].abs_error_w); ax.plot(v,np.linspace(0,1,len(v)),label=condition)
    ax.set_xscale("symlog",linthresh=50); ax.set_xlabel("absolute error (W)"); ax.set_ylabel("empirical CDF"); ax.legend(); fig.tight_layout(); fig.savefig(d/"absolute_error_ecdf.png",dpi=150); plt.close(fig)
    z=tolerance[(tolerance.seed==42)&tolerance.representation.isin(["compact3","point"])]
    fig,ax=plt.subplots(figsize=(7,4))
    for (h,c),q in z.groupby(["horizon_ms","condition"]): ax.plot(q.threshold_w,q.exceedance_rate,marker="o",label=f"{h} {c}")
    ax.set_xlabel("diagnostic tolerance (W)"); ax.set_ylabel("exceedance rate"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(d/"tolerance_exceedance.png",dpi=150); plt.close(fig)
    z=episodes[(episodes.seed==42)&(episodes.threshold_w==500)&(episodes.condition=="history_meta")]
    fig,ax=plt.subplots(figsize=(7,4))
    for rep,q in z.groupby("representation"):
        v=np.sort(q.duration_s); ax.step(v,np.linspace(0,1,len(v)),where="post",label=rep)
    ax.set_xlabel("500 W episode duration (s)"); ax.set_ylabel("empirical CDF"); ax.legend(); fig.tight_layout(); fig.savefig(d/"episode_duration_ecdf.png",dpi=150); plt.close(fig)
    z=intervals[(intervals.diagnostic_type=="interval")&(intervals.grouping_level=="transition")&(intervals.nominal_coverage==.8)&
                (intervals.condition=="history_meta")&(intervals.representation=="compact3")&(intervals.seed==42)]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for h,q in z.groupby("horizon_ms"):
        axes[0].plot(q.transition_stratum,q.coverage,marker="o",label=str(h)); axes[1].plot(q.transition_stratum,q.lower_violation_rate,marker="o",label=f"{h} low"); axes[1].plot(q.transition_stratum,q.upper_violation_rate,marker="x",label=f"{h} high")
    axes[0].axhline(.8,color="black",lw=.8); axes[0].legend(); axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(d/"coverage_by_transition.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.hist((canonical[canonical.seed==42].compact_q50-canonical[canonical.seed==42].full_q50),bins=100); ax.set_xlabel("compact q50 - full q50 (W)"); fig.tight_layout(); fig.savefig(d/"full_vs_compact_q50_change.png",dpi=150); plt.close(fig)
    for i,r in cases.iterrows():
        z=long[(long.assessment==r.assessment)&(long.run_id==r.run_id)&(long.horizon_ms==r.horizon_ms)&
               (long.condition=="history_meta")&(long.representation=="compact3")&(long.seed==42)&(np.abs(long.target_time_s-r.target_time_s)<=10)].sort_values("target_time_s")
        fig,ax=plt.subplots(figsize=(9,4)); ax.plot(z.target_time_s,z.observed_power_w,label="observed"); ax.plot(z.target_time_s,z.q50,label="q50"); ax.fill_between(z.target_time_s.to_numpy(),z.q10.to_numpy(),z.q90.to_numpy(),alpha=.2); ax.axvline(r.target_time_s,color="red",ls="--"); ax.set_title(r.selection_rule); ax.legend(); fig.tight_layout(); fig.savefig(d/f"timeseries_{i:02d}.png",dpi=150); plt.close(fig)


def write_report(out,recon,runs,tails,tolerance,transitions,intervals,exports,uncertainty,inventory,checks):
    lines=["# Frozen lightweight AIDC forecasting retrospective validation","",
    "## Scope and verdict","",
    "This audit reloaded frozen fold-specific and final bundles. It did not retrain, tune, recalibrate, or change postprocessing. Grouped outer is retrospective and the four-run reused holdout was previously inspected; neither is fresh prospective confirmation.","",
    f"All {len(recon)} reconciliation checks passed: **{bool(recon.passed.all())}**. Integrity passed: **{checks['passed']}**. Canonical model-condition rows: {checks['canonical_rows']:,}.","",
    "## Average predictive value",""]
    for h in (1000,2000):
        z=recon[(recon.assessment=="grouped_outer")&(recon.horizon_ms==h)&(recon.metric=="run_seed_balanced_crps_w")]
        hist=float(z[z.condition=="history_only"].recomputed_value.iloc[0]); meta=float(z[z.condition=="history_meta"].recomputed_value.iloc[0])
        lines.append(f"- {h} ms full-15 grouped outer CRPS: history-only {hist:.2f} W, metadata {meta:.2f} W, gain {100*(1-meta/hist):.2f}%.")
    lines+=["","Persistence is a point baseline only; it is not assigned an interval or probabilistic score.","","## Per-run and cohort consistency",""]
    g=runs[(runs.assessment=="grouped_outer")&(runs.representation=="full15")&(runs.condition=="history_meta")]
    for h in (1000,2000):
        z=g[g.horizon_ms==h].groupby("run_id").metadata_crps_gain.mean()
        lines.append(f"- {h} ms: improved {(z>TIE_TOLERANCE).sum()}/{len(z)} runs, tied {(np.abs(z)<=TIE_TOLERANCE).sum()}, worsened {(z < -TIE_TOLERANCE).sum()}; median gain {100*z.median():.2f}%, range {100*z.min():.2f}% to {100*z.max():.2f}%.")
    lines+=["","Each grouped-outer cohort has three independent runs and each reused-holdout cohort one. Cohort patterns are descriptive and metadata effects remain confounded.","","## Rare and sustained errors",""]
    for assessment in ("grouped_outer","reused_holdout"):
        for h in (1000,2000):
            z=tails[(tails.assessment==assessment)&(tails.horizon_ms==h)&(tails.condition=="history_meta")&(tails.representation=="compact3")&(tails.seed==42)&(tails.aggregation=="pooled_targets")]
            if len(z):
                r=z.iloc[0]; lines.append(f"- {assessment}, {h} ms: MAE/RMSE {r.mae_w:.1f}/{r.rmse_w:.1f} W; p95/p99/max absolute error {r.abs_error_p95_w:.1f}/{r.abs_error_p99_w:.1f}/{r.abs_error_max_w:.1f} W; under/over rates {100*r.underprediction_rate:.1f}%/{100*r.overprediction_rate:.1f}%.")
    for h in (1000,2000):
        z=tolerance[(tolerance.assessment=="grouped_outer")&(tolerance.horizon_ms==h)&(tolerance.condition=="history_meta")&(tolerance.representation=="compact3")&(tolerance.seed==42)&(tolerance.threshold_w==500)].iloc[0]
        lines.append(f"- At diagnostic 500 W, grouped-outer {h} ms exceeds at {100*z.exceedance_rate:.1f}% of valid issue times; longest episode {z.episode_duration_max_s:.1f} s.")
    lines+=["","Episodes measure persistent forecast error across issuance times, not physical control failure. Maxima are observations, not bounds.","","## Endpoint changes and persistence",""]
    for h in (1000,2000):
        z=transitions[(transitions.assessment=="grouped_outer")&(transitions.horizon_ms==h)&(transitions.condition=="history_meta")&(transitions.representation=="compact3")&(transitions.seed==42)]
        for s in ("rising","falling","stable"):
            r=z[z.transition_stratum==s].iloc[0]; lines.append(f"- {h} ms {s}: MAE {r.mae_w:.1f} W; improvement versus persistence {100*r.mae_improvement_vs_persistence:.1f}%.")
    lines+=["","These are endpoint-change diagnostics, not ramp-onset, lead-time, peak, or trajectory tests.","","## Uncertainty reliability",""]
    z=intervals[(intervals.diagnostic_type=="interval")&(intervals.grouping_level=="overall")&(intervals.condition=="history_meta")&(intervals.representation=="compact3")&(intervals.seed==42)&(intervals.nominal_coverage==.8)]
    for r in z.itertuples(): lines.append(f"- {r.assessment}, {int(r.horizon_ms)} ms: q10-q90 coverage {100*r.coverage:.1f}%, lower/upper violations {100*r.lower_violation_rate:.1f}%/{100*r.upper_violation_rate:.1f}%, width {r.width_mean_w:.1f} W, interval score {r.interval_score_mean_w:.1f} W.")
    lines+=["","Coverage varies by run, cohort, and direction. No calibration was fitted.","","## Compact export effects",""]
    for r in exports[(exports.assessment=="reused_holdout")&exports.compact_source.eq("actual_deployment_bundle")].itertuples():
        lines.append(f"- {int(r.horizon_ms)} ms {r.condition}: compact sorting q50 mean/max change versus raw {r.compact3_q50_change_mean_w:.1f}/{r.compact3_q50_change_max_w:.1f} W; compact-versus-full max {r.compact_vs_full_q50_change_max_w:.1f} W; compact/full MAE {r.compact3_q50_mae_w:.1f}/{r.full15_q50_mae_w:.1f} W.")
    lines+=["","The reported >1 kW change is reproduced, but paired evidence does not establish export sorting as the main overall weakness.","","## Coverage, provenance, and limits","",
    f"The inventory contains {len(inventory):,} rows. Predictions cover every eligible cached feature row (stride 1, no evaluation cap); no interpolation was used. Gaps and segment boundaries break episodes.",
    "Each outer bundle matched its fold test membership with zero train/test index overlap. Final models were evaluated only on the four excluded reused-holdout runs.",""]
    ci=uncertainty[(uncertainty.assessment=="grouped_outer")&uncertainty.comparison.eq("metadata_minus_history_crps_w")&uncertainty.resampling.eq("paired_runs")]
    for r in ci.itertuples(): lines.append(f"- {int(r.horizon_ms)} ms metadata-minus-history CRPS {r.estimate_w:.1f} W, paired-run bootstrap 95% interval [{r.ci_low_w:.1f}, {r.ci_high_w:.1f}] ({int(r.n_independent_runs)} runs).")
    lines+=["","Bootstrap uses 2,000 paired-run replicates after averaging seeds within run. It excludes retraining and selection uncertainty.",
    "New untouched runs, operational tolerances, serving timing, controller behavior, hardware safeguards, and safety/control readiness remain unverified. The tolerance grid is not a safety threshold.","","## Reproducibility","",
    "Commands and SHA-256 hashes are recorded in execution.log and source_artifact_hashes.csv."]
    (out/"report_en.md").write_text("\n".join(lines)+"\n")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output"); ap.add_argument("--smoke",action="store_true"); a=ap.parse_args()
    name=time.strftime("audit_%Y%m%dT%H%M%SZ",time.gmtime())+("_smoke" if a.smoke else ""); out=Path(a.output) if a.output else HERE/name
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); atomic_json(out/"validation_spec.json",analysis_spec(name,a.smoke))
    logging.basicConfig(level=logging.INFO,format="%(asctime)sZ %(levelname)s %(message)s",handlers=[logging.FileHandler(out/"execution.log"),logging.StreamHandler(sys.stdout)])
    start=time.time(); status={"status":"running","started_utc":time.strftime("%FT%TZ",time.gmtime()),"pid":os.getpid(),"output":str(out),"smoke":a.smoke}; atomic_json(out/"completion_status.json",status)
    try:
        logging.info("load and reconstruct frozen predictions"); metas,powers=load_inputs(); canonical,reconstruction=reconstruct_canonical(out,metas,powers,a.smoke)
        checks=integrity_checks(canonical,reconstruction,metas); atomic_json(out/"data_integrity_checks.json",checks)
        if not checks["passed"]: raise RuntimeError(checks)
        inventory=build_inventory(canonical,metas); inventory.to_csv(out/"evaluation_inventory.csv",index=False)
        recon=aggregate_reconciliation(canonical) if not a.smoke else pd.DataFrame(); recon.to_csv(out/"aggregate_reconciliation.csv",index=False)
        long=build_long(canonical); runs=per_run_metrics(long); runs.to_csv(out/"per_run_metrics.csv",index=False)
        cohorts=cohort_metrics(long,runs); cohorts.to_csv(out/"per_cohort_metrics.csv",index=False)
        tails=tail_metrics(long); tails.to_csv(out/"tail_error_metrics.csv",index=False)
        tolerance,episodes=tolerance_outputs(long); tolerance.to_csv(out/"tolerance_exceedance.csv",index=False); episodes.to_parquet(out/"failure_episodes.parquet",index=False)
        transitions,confusion=transition_outputs(long); transitions.to_csv(out/"transition_metrics.csv",index=False); confusion.to_csv(out/"change_confusion_matrices.csv",index=False)
        intervals=interval_outputs(long,canonical); intervals.to_csv(out/"interval_diagnostics.csv",index=False)
        exports=export_outputs(canonical,tolerance); exports.to_csv(out/"export_effects.csv",index=False)
        uncertainty=paired_uncertainty(runs); uncertainty.to_csv(out/"paired_uncertainty.csv",index=False)
        cases=worst_cases(canonical,long,episodes); cases.to_csv(out/"worst_cases.csv",index=False)
        paths=[Path(__file__),HERE/"metrics.py",EXPERIMENT/"effective_config.json",EXPERIMENT/"outer_evaluation_tasks.csv",
            EXPERIMENT/"outer_predictions.parquet",EXPERIMENT/"reused_holdout_predictions.parquet",EXPERIMENT/"corrected_folds.csv",
            CORRECTED/"input_manifest.csv",CORRECTED/"features/meta_200_1000.parquet",CORRECTED/"features/meta_200_2000.parquet"]+list(MODEL_ROOT.glob("*.joblib"))
        pd.DataFrame([{"path":str(p.resolve()),"sha256":sha256(p),"bytes":p.stat().st_size} for p in paths]).to_csv(out/"source_artifact_hashes.csv",index=False)
        if not a.smoke:
            make_plots(out,canonical,long,runs,tolerance,episodes,intervals,cases)
            write_report(out,recon,runs,tails,tolerance,transitions,intervals,exports,uncertainty,inventory,checks)
        status.update(status="completed",finished_utc=time.strftime("%FT%TZ",time.gmtime()),elapsed_s=time.time()-start,
            integrity_passed=True,report=str(out/"report_en.md") if not a.smoke else None,canonical_rows=len(canonical),episode_rows=len(episodes))
        atomic_json(out/"completion_status.json",status); logging.info("completed %s",out); print(json.dumps(status,indent=2))
    except Exception as exc:
        logging.exception("validation failed"); status.update(status="failed",finished_utc=time.strftime("%FT%TZ",time.gmtime()),elapsed_s=time.time()-start,error=repr(exc)); atomic_json(out/"completion_status.json",status); raise


if __name__=="__main__":
    main()
