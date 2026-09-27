"""Pure metric helpers for the frozen-model retrospective validation."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


RESOLUTION_S = 0.2


def error_values(prediction: Iterable[float], observed: Iterable[float]) -> np.ndarray:
    """Signed error: negative is underprediction, positive is overprediction."""
    return np.asarray(prediction, dtype=float) - np.asarray(observed, dtype=float)


def pinball_crps(y: np.ndarray, pred: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Numerical CRPS used by the frozen experiment, including constant endpoints."""
    q = np.r_[0.0, np.asarray(quantiles, dtype=float), 1.0]
    values = np.column_stack([pred[:, 0], pred, pred[:, -1]])
    residual = np.asarray(y, dtype=float)[:, None] - values
    loss = np.maximum(q * residual, (q - 1.0) * residual)
    return 2.0 * np.trapz(loss, q, axis=1)


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Central (1-alpha) interval score; lower is better."""
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return (upper - lower) + 2.0 / alpha * np.maximum(lower - y, 0.0) + 2.0 / alpha * np.maximum(y - upper, 0.0)


def weighted_quantile(values: np.ndarray, probabilities: Iterable[float], weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(list(probabilities), dtype=float)
    if weights is None:
        return np.quantile(values, probabilities)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(probabilities, cumulative, values, left=values[0], right=values[-1])


def distribution_metrics(errors: np.ndarray, weights: np.ndarray | None = None) -> dict:
    e = np.asarray(errors, dtype=float)
    a = np.abs(e)
    if weights is None:
        weights = np.ones(len(e), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    q = weighted_quantile(a, [0.5, 0.9, 0.95, 0.99], weights)
    cutoff = weighted_quantile(a, [0.95], weights)[0]
    tail = a >= cutoff
    under = e < 0
    over = e > 0
    return {
        "n_targets": int(len(e)),
        "mae_w": float(np.sum(weights * a)),
        "rmse_w": float(np.sqrt(np.sum(weights * e * e))),
        "mean_signed_error_w": float(np.sum(weights * e)),
        "abs_error_p50_w": float(q[0]),
        "abs_error_p90_w": float(q[1]),
        "abs_error_p95_w": float(q[2]),
        "abs_error_p99_w": float(q[3]),
        "abs_error_max_w": float(a.max()),
        "underprediction_rate": float(np.sum(weights[under])),
        "overprediction_rate": float(np.sum(weights[over])),
        "mean_underprediction_magnitude_w": float(np.average(-e[under], weights=weights[under])) if under.any() else np.nan,
        "mean_overprediction_magnitude_w": float(np.average(e[over], weights=weights[over])) if over.any() else np.nan,
        "worst_5pct_mean_abs_error_w": float(np.average(a[tail], weights=weights[tail])),
        "worst_5pct_cutoff_w": float(cutoff),
    }


def interval_metrics(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.2) -> dict:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    low = y < lower
    high = y > upper
    widths = upper - lower
    return {
        "n_targets": int(len(y)),
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "lower_violation_rate": float(low.mean()),
        "upper_violation_rate": float(high.mean()),
        "width_mean_w": float(widths.mean()),
        "width_p50_w": float(np.quantile(widths, 0.5)),
        "width_p90_w": float(np.quantile(widths, 0.9)),
        "lower_violation_mean_w": float(np.mean(lower[low] - y[low])) if low.any() else 0.0,
        "upper_violation_mean_w": float(np.mean(y[high] - upper[high])) if high.any() else 0.0,
        "interval_score_mean_w": float(interval_score(y, lower, upper, alpha).mean()),
    }


def transition_labels(y: Iterable[float], issue_power: Iterable[float], threshold_w: float = 100.0) -> np.ndarray:
    change = np.asarray(y, dtype=float) - np.asarray(issue_power, dtype=float)
    return np.where(change >= threshold_w, "rising", np.where(change <= -threshold_w, "falling", "stable"))


def change_labels(change: Iterable[float], threshold_w: float = 100.0) -> np.ndarray:
    change = np.asarray(change, dtype=float)
    return np.where(change >= threshold_w, "rising", np.where(change <= -threshold_w, "falling", "stable"))


def confusion_rows(observed: np.ndarray, predicted: np.ndarray) -> list[dict]:
    classes = ("falling", "stable", "rising")
    rows = []
    for actual in classes:
        for estimate in classes:
            rows.append({"observed_class": actual, "predicted_class": estimate,
                         "count": int(np.sum((observed == actual) & (predicted == estimate)))})
    for cls in classes:
        tp = sum(r["count"] for r in rows if r["observed_class"] == cls and r["predicted_class"] == cls)
        pred_n = sum(r["count"] for r in rows if r["predicted_class"] == cls)
        obs_n = sum(r["count"] for r in rows if r["observed_class"] == cls)
        for r in rows:
            if r["observed_class"] == cls and r["predicted_class"] == cls:
                r["class_precision"] = tp / pred_n if pred_n else np.nan
                r["class_recall"] = tp / obs_n if obs_n else np.nan
            else:
                r["class_precision"] = np.nan
                r["class_recall"] = np.nan
    return rows


def failure_episodes(frame: pd.DataFrame, threshold_w: float, resolution_s: float = RESOLUTION_S) -> pd.DataFrame:
    """Segment error exceedances; gaps and group boundaries break and censor episodes."""
    required = {"issue_s", "error_w", "run_id", "segment_key"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing episode columns: {sorted(missing)}")
    rows: list[dict] = []
    group_columns = [c for c in ("assessment", "run_id", "segment_key", "horizon_ms", "condition", "representation", "seed") if c in frame]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        g = group.sort_values("issue_s", kind="mergesort").reset_index(drop=True)
        issue = g.issue_s.to_numpy(float)
        exceed = np.abs(g.error_w.to_numpy(float)) > threshold_w
        block_start = np.r_[True, ~np.isclose(np.diff(issue), resolution_s, atol=1e-7)]
        block_id = np.cumsum(block_start)
        for block in np.unique(block_id):
            ii = np.flatnonzero(block_id == block)
            mask = exceed[ii]
            starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
            ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
            for start, end in zip(starts, ends):
                selected = ii[start:end + 1]
                errs = g.error_w.to_numpy(float)[selected]
                rows.append(identity | {
                    "threshold_w": float(threshold_w),
                    "start_issue_s": float(issue[selected[0]]),
                    "end_issue_s": float(issue[selected[-1]]),
                    "n_slots": int(len(selected)),
                    "duration_s": float(len(selected) * resolution_s),
                    "max_abs_error_w": float(np.max(np.abs(errs))),
                    "mean_signed_error_w": float(np.mean(errs)),
                    "left_censored": bool(start == 0),
                    "right_censored": bool(end == len(ii) - 1),
                })
    return pd.DataFrame(rows)


def match_full_compact(raw_full: np.ndarray, raw_compact: np.ndarray,
                       full_indices: tuple[int, ...] = (3, 7, 11),
                       atol: float = 1e-10) -> dict:
    """Verify that compact estimators are the selected full-bundle estimators."""
    full = np.asarray(raw_full, dtype=float)
    compact = np.asarray(raw_compact, dtype=float)
    selected = full[:, list(full_indices)]
    if selected.shape != compact.shape:
        return {"passed": False, "shape_match": False, "max_abs_difference_w": np.nan}
    difference = float(np.max(np.abs(selected - compact))) if selected.size else 0.0
    return {"passed": difference <= atol, "shape_match": True,
            "max_abs_difference_w": difference}


def assert_unique_and_paired(frame: pd.DataFrame, keys: list[str], condition_column: str = "condition") -> dict:
    duplicates = int(frame.duplicated(keys + [condition_column]).sum())
    counts = frame.groupby(keys, dropna=False)[condition_column].nunique()
    unmatched = int((counts != frame[condition_column].nunique()).sum())
    return {"duplicate_prediction_keys": duplicates, "unmatched_paired_keys": unmatched,
            "passed": duplicates == 0 and unmatched == 0}
