from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = np.array([.01, .025, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99])
STATIC = ("gpu_model", "workload_type", "model_name", "model_size", "batch_size", "sequence_length", "image_size", "zero_stage")


def crps_quantile(y, predictions, quantiles=QUANTILES):
    """Piecewise-linear pinball integral, with endpoint quantiles held constant."""
    q = np.asarray(quantiles, float)
    p = np.asarray(predictions, float)
    y = np.asarray(y, float)
    if p.ndim != 2 or p.shape != (len(y), len(q)) or not np.all(np.diff(q) > 0):
        raise ValueError("invalid quantile array")
    if np.any(np.diff(p, axis=1) < -1e-8):
        raise ValueError("quantile crossing; correct or report before scoring")
    grid = np.r_[0., q, 1.]
    values = np.column_stack([p[:, 0], p, p[:, -1]])
    residual = y[:, None] - values
    loss = np.maximum(grid * residual, (grid - 1) * residual)
    return 2 * np.trapz(loss, grid, axis=1)


def target_rows(frame: pd.DataFrame, horizons=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), history=25, threshold_w=100.):
    """Use only consecutive completed 200 ms bins within one valid segment."""
    rows = []
    for segment, g in frame.groupby("segment_id", sort=False):
        g = g.sort_values("available_s").reset_index(drop=True)
        valid = g.is_valid.to_numpy(bool) & np.isfinite(g.power_w.to_numpy(float))
        t = g.available_s.to_numpy(float)
        p = g.power_w.to_numpy(float)
        contiguous = np.r_[False, np.isclose(np.diff(t), .2, atol=1e-5)]
        for end in range(history - 1, len(g) - max(horizons)):
            if not valid[end-history+1:end+max(horizons)+1].all():
                continue
            if not contiguous[end-history+2:end+max(horizons)+1].all():
                continue
            trail = p[end-history+1:end+1]
            for h in horizons:
                future = p[end+1:end+h+1]
                slopes = np.diff(np.r_[p[end], future]) / .2
                delta = future[-1] - p[end]
                rows.append((g.run_id.iloc[0], segment, t[end], float(g.bin_start_s.iloc[end+h]),
                             t[end+h], h * 200, p[end], future[-1], delta,
                             int(np.any(future-p[end] >= threshold_w)),
                             int(np.any(future-p[end] <= -threshold_w)),
                             float(max(slopes)), float(min(slopes)),
                             trail.copy()))
    cols = ["run_id", "segment_id", "issue_s", "target_start_s", "target_end_s", "horizon_ms",
            "current", "y", "delta_w", "rise_event", "drop_event", "max_ramp_up_w_s",
            "max_ramp_down_w_s", "history"]
    return pd.DataFrame(rows, columns=cols)


def history_features(history):
    a = np.asarray(history, float)
    features = [a[-1]]
    for k in (5, 10, 25):
        z = a[-k:]
        features.extend([z.mean(), z.std(), z.min(), z.max(), (z[-1] - z[0]) / ((k-1) * .2)])
    return np.asarray(features, np.float32)


def static_features(meta: pd.DataFrame, train_ids, test_ids, columns=STATIC):
    """Fit categorical vocabulary and numeric imputation on training runs only."""
    train, test = meta.iloc[train_ids], meta.iloc[test_ids]
    categorical = [c for c in columns if c in meta and c not in ("batch_size", "sequence_length", "image_size")]
    numeric = [c for c in ("batch_size", "sequence_length", "image_size") if c in meta and c in columns]
    result = {}
    for c in categorical:
        vocab = sorted(train[c].fillna("__missing__").astype(str).unique())
        for name, data in (("train", train), ("test", test)):
            v = data[c].fillna("__missing__").astype(str)
            result.setdefault(name, []).extend([(v == item).to_numpy(np.float32)[:, None] for item in vocab])
            result[name].append((~v.isin(vocab)).to_numpy(np.float32)[:, None])
    for c in numeric:
        median = pd.to_numeric(train[c], errors="coerce").median()
        median = 0. if pd.isna(median) else float(median)
        for name, data in (("train", train), ("test", test)):
            v = pd.to_numeric(data[c], errors="coerce")
            result.setdefault(name, []).extend([v.fillna(median).to_numpy(np.float32)[:, None],
                                                 v.isna().to_numpy(np.float32)[:, None]])
    return tuple(np.column_stack(result.get(name, [])) if result.get(name) else np.empty((len(data), 0), np.float32)
                 for name, data in (("train", train), ("test", test)))


def replay_filler(power, prediction, dt=.2, ramp_down=200., capacity=500., delay_bins=1,
                  slew_w_s=200.):
    """Idealized filler only; commands use completed observations and delayed forecasts."""
    power = np.asarray(power, float)
    prediction = np.asarray(prediction, float)
    if len(power) != len(prediction) or delay_bins < 0:
        raise ValueError("invalid replay input")
    grid = power.copy()
    filler = np.zeros(len(power))
    queue = [0.] * delay_bins
    reference = power[0]
    command = 0.
    for k in range(1, len(power)):
        observed = power[k-1]
        reference = max(observed, reference - ramp_down * dt)
        desired = float(np.clip(reference - prediction[k-1], 0, capacity))
        command += float(np.clip(desired-command, -slew_w_s*dt, slew_w_s*dt))
        if delay_bins:
            queue.append(command)
            filler[k] = queue.pop(0)
        else:
            filler[k] = command
        grid[k] += filler[k]
    return grid, filler


def ramp_metrics(power, dt=.2, limit=200.):
    ramp = np.diff(np.asarray(power, float)) / dt
    exceed = np.maximum(np.abs(ramp) - limit, 0)
    return {"ramp_exceed_fraction": float(np.mean(exceed > 0)),
            "mean_exceed_w_s": float(exceed.mean()), "max_exceed_w_s": float(exceed.max())}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n")
