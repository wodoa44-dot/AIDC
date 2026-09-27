"""Shared offline/online feature, encoding, fitting, and scoring contracts."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


RESEARCH_QUANTILES = np.array([.01, .025, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99])
DEPLOY_QUANTILES = np.array([.1, .5, .9])
META_COLUMNS = ("gpu", "workload", "batch", "ai_model")


def power_feature_names(resolution_ms: int, history_ms: int) -> list[str]:
    k = history_ms // resolution_ms
    if k == 1:
        return ["current"]
    names = ["current"]
    names += [f"lag_{lag * resolution_ms}ms" for lag in sorted({max(1, k // 4), max(1, k // 2), k - 1}) if 0 < lag < k]
    for w in sorted({max(2, k // 4), max(2, k // 2), k}):
        if w <= k:
            names += [f"mean_{w * resolution_ms}ms", f"std_{w * resolution_ms}ms",
                      f"min_{w * resolution_ms}ms", f"max_{w * resolution_ms}ms",
                      f"slope_{w * resolution_ms}ms"]
    return names


def power_features(values: Iterable[float], resolution_ms: int, history_ms: int) -> np.ndarray:
    """Causal trailing features, byte-for-byte aligned with predictability/core.py."""
    p = np.asarray(list(values), dtype=np.float64)
    k = history_ms // resolution_ms
    if len(p) < k:
        raise ValueError(f"need {k} completed bins, got {len(p)}")
    p = p[-k:]
    if k == 1:
        return p[-1:].astype(np.float32)
    out = [p[-1]]
    for lag in sorted({max(1, k // 4), max(1, k // 2), k - 1}):
        if 0 < lag < k:
            out.append(p[-1-lag])
    for w in sorted({max(2, k // 4), max(2, k // 2), k}):
        if w <= k:
            a = p[-w:]
            out.extend([a.mean(), a.std(ddof=0), a.min(), a.max(),
                        (a[-1] - a[0]) / ((w - 1) * resolution_ms / 1000)])
    return np.asarray(out, dtype=np.float32)


def fit_encoder(meta: pd.DataFrame, ids: np.ndarray, condition: str) -> dict:
    groups = [] if condition == "history_only" else list(META_COLUMNS)
    enc = {"groups": groups, "vocab": {}, "batch_median": None, "names": []}
    fit = meta.iloc[ids]
    for col in groups:
        if col == "batch":
            v = pd.to_numeric(fit[col], errors="coerce")
            enc["batch_median"] = float(v.median()) if v.notna().any() else 0.0
            enc["names"] += ["batch", "batch_missing"]
        else:
            vocab = sorted(fit[col].fillna("__MISSING__").astype(str).unique().tolist())
            enc["vocab"][col] = vocab
            enc["names"] += [f"{col}={x}" for x in vocab] + [f"{col}=__UNKNOWN__"]
    return enc


def transform(power: np.ndarray, meta: pd.DataFrame, enc: dict) -> np.ndarray:
    cols = [np.asarray(power, dtype=np.float32)]
    for col in enc["groups"]:
        if col == "batch":
            v = pd.to_numeric(meta[col], errors="coerce")
            cols += [v.fillna(enc["batch_median"]).to_numpy(np.float32)[:, None],
                     v.isna().to_numpy(np.float32)[:, None]]
        else:
            v = meta[col].fillna("__MISSING__").astype(str)
            vocab = enc["vocab"][col]
            cols += [(v == x).to_numpy(np.float32)[:, None] for x in vocab]
            cols.append((~v.isin(vocab)).to_numpy(np.float32)[:, None])
    return np.column_stack(cols).astype(np.float32)


def transform_one(power: np.ndarray, metadata: dict, enc: dict) -> np.ndarray:
    frame = pd.DataFrame([{k: metadata.get(k) for k in META_COLUMNS}])
    return transform(np.asarray(power, dtype=np.float32)[None, :], frame, enc)


def choose_per_run(meta: pd.DataFrame, ids: np.ndarray, cap: int) -> np.ndarray:
    chosen: list[int] = []
    for _, group in meta.iloc[ids].groupby("run_id", sort=True):
        ii = group.index.to_numpy()
        n = min(len(ii), int(cap))
        chosen.extend(ii[np.linspace(0, len(ii) - 1, n, dtype=int)])
    return np.asarray(sorted(chosen), dtype=int)


def equal_run_weights(meta: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    counts = meta.iloc[ids].groupby("run_id").size()
    w = meta.iloc[ids].run_id.map(1.0 / counts).to_numpy() * len(counts)
    totals = pd.DataFrame({"run_id": meta.iloc[ids].run_id.to_numpy(), "w": w}).groupby("run_id").w.sum()
    np.testing.assert_allclose(totals, len(counts))
    return w


def fit_models(x, y, ids, weights, quantiles, params, seed):
    models = []
    for q in quantiles:
        model = LGBMRegressor(objective="quantile", alpha=float(q),
            n_estimators=int(params["n_estimators"]), learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]), min_child_samples=int(params["min_child_samples"]),
            reg_alpha=float(params["reg_alpha"]), reg_lambda=float(params["reg_lambda"]),
            random_state=int(seed), n_jobs=int(params.get("n_jobs", 1)), verbosity=-1)
        model.fit(x[ids], y[ids], sample_weight=weights)
        models.append(model)
    return models


def raw_predict(models, x):
    return np.column_stack([m.predict(x, num_iteration=m.booster_.num_trees()) for m in models])


def rearrange(pred):
    pred = np.asarray(pred)
    if pred.ndim != 2 or not np.isfinite(pred).all():
        raise ValueError("invalid quantile predictions")
    crossing = float(np.mean(np.any(np.diff(pred, axis=1) < 0, axis=1))) if len(pred) else 0.0
    return np.sort(pred, axis=1), crossing


def crps(y, pred, quantiles):
    q = np.r_[0.0, np.asarray(quantiles), 1.0]
    v = np.column_stack([pred[:, 0], pred, pred[:, -1]])
    e = np.asarray(y)[:, None] - v
    loss = np.maximum(q * e, (q - 1) * e)
    return 2 * np.trapz(loss, q, axis=1)


def score_frame(meta, ids, pred, quantiles):
    q = list(map(float, quantiles)); y = meta.iloc[ids].y.to_numpy()
    out = meta.iloc[ids][["run_id", "cohort", "issue_s", "target_s", "target_bin", "y", "current"]].copy()
    out["target_index"] = ids
    out["crps_w"] = crps(y, pred, quantiles)
    out["mae_w"] = np.abs(y - pred[:, q.index(.5)])
    out["covered_80"] = (y >= pred[:, q.index(.1)]) & (y <= pred[:, q.index(.9)])
    out["width_80_w"] = pred[:, q.index(.9)] - pred[:, q.index(.1)]
    if .05 in q and .95 in q:
        out["covered_90"] = (y >= pred[:, q.index(.05)]) & (y <= pred[:, q.index(.95)])
        out["width_90_w"] = pred[:, q.index(.95)] - pred[:, q.index(.05)]
    for j, value in enumerate(q):
        out[f"q_{value:g}"] = pred[:, j]
    return out


def summarize(frame):
    target = {"target_weighted_crps_w": float(frame.crps_w.mean()),
              "target_weighted_mae_w": float(frame.mae_w.mean()),
              "n_targets": int(len(frame)), "n_runs": int(frame.run_id.nunique())}
    per_run = frame.groupby("run_id").agg(crps_w=("crps_w", "mean"), mae_w=("mae_w", "mean"),
        coverage_80=("covered_80", "mean"), width_80_w=("width_80_w", "mean"),
        n_targets=("y", "size"), cohort=("cohort", "first"))
    if "covered_90" in frame:
        extra = frame.groupby("run_id").agg(coverage_90=("covered_90", "mean"), width_90_w=("width_90_w", "mean"))
        per_run = per_run.join(extra)
    run = {f"run_balanced_{c}": float(per_run[c].mean()) for c in per_run.columns if c not in ("cohort", "n_targets")}
    return target | run, per_run.reset_index()


def serialized_size(models, encoder, parameters, quantiles) -> int:
    stream = io.BytesIO()
    joblib.dump({"models": models, "encoder": encoder, "parameters": parameters,
                 "quantiles": list(map(float, quantiles))}, stream, compress=3)
    return len(stream.getvalue())


def latency_us(models, x, trials=30):
    sample = x[:min(64, len(x))]
    for _ in range(3): raw_predict(models, sample)
    values = []
    for _ in range(trials):
        start = time.perf_counter_ns(); raw_predict(models, sample); values.append((time.perf_counter_ns()-start)/1000/len(sample))
    return {"latency_us_p50": float(np.percentile(values, 50)),
            "latency_us_p95": float(np.percentile(values, 95)),
            "latency_us_p99": float(np.percentile(values, 99))}
