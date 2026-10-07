from __future__ import annotations
import numpy as np
import pandas as pd


def build_features(bins: pd.DataFrame, history_bins=150) -> pd.DataFrame:
    """At issuance right_s, only completed bins through that time enter features."""
    p = bins.power_w
    seg = bins.segment
    d = pd.DataFrame(index=bins.index)
    d["current_w"] = p
    for lag in [1, 2, 5, 10, 25, 50, 150]:
        if lag > history_bins:
            continue
        d[f"lag_{lag}"] = p.shift(lag).where(seg.eq(seg.shift(lag)))
    for width in [5, 25, 50, 150]:
        if width > history_bins:
            continue
        d[f"mean_{width}"] = p.rolling(width, min_periods=width).mean().where(seg.eq(seg.shift(width-1)))
        d[f"std_{width}"] = p.rolling(width, min_periods=width).std().where(seg.eq(seg.shift(width-1)))
    d["slope_1"] = p.diff().where(seg.eq(seg.shift()))
    d["eligible_history"] = p.notna() & seg.eq(seg.shift(history_bins))
    return d


def feature_columns(model: str, columns: list[str]) -> list[str]:
    power = [c for c in columns if c == "current_w" or c.startswith(("lag_", "mean_", "std_", "slope_"))]
    basic = ["hardware", "workload"]
    extra = ["model_size_token", "image_size", "batch_size", "sequence_length", "zero_stage"]
    if model == "P1": return ["current_w"]
    if model == "P2": return power
    if model == "P3": return power + [c for c in basic if c in columns]
    if model == "P4": return power + [c for c in basic + extra if c in columns]
    raise ValueError(model)
