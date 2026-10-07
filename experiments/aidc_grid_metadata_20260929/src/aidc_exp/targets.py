from __future__ import annotations
import numpy as np
import pandas as pd


def build_targets(bins: pd.DataFrame, horizons=(5, 10)) -> pd.DataFrame:
    """Endpoint is future completed bin; path starts at the next bin after issuance."""
    out = pd.DataFrame(index=bins.index)
    p, seg = bins.power_w, bins.segment
    for h in horizons:
        future = pd.concat([p.shift(-j) for j in range(1, h + 1)], axis=1)
        same = seg.eq(seg.shift(-h)) & future.notna().all(axis=1)
        delta = future.sub(p, axis=0)
        out[f"y_{h}"] = p.shift(-h).where(same)
        out[f"up_max_{h}"] = delta.max(axis=1).where(same)
        out[f"down_min_{h}"] = delta.min(axis=1).where(same)
        out[f"endpoint_delta_{h}"] = (p.shift(-h) - p).where(same)
        # Adjacent completed-bin changes over (issuance, target], including issuance bin.
        path = pd.concat([(p.shift(-j) - p.shift(-(j-1))) for j in range(1, h + 1)], axis=1)
        out[f"max_ramp_02_{h}"] = path.max(axis=1).where(same)
        out[f"min_ramp_02_{h}"] = path.min(axis=1).where(same)
        if h >= 5:
            ramps = pd.concat([(p.shift(-j) - p.shift(-(j-5))) for j in range(5, h + 1)], axis=1)
            out[f"max_ramp_1s_{h}"] = ramps.max(axis=1).where(same)
            out[f"min_ramp_1s_{h}"] = ramps.min(axis=1).where(same)
    return out
