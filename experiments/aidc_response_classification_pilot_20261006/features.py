"""Causal features on original bins; future columns serve labels only."""
import numpy as np
import pandas as pd
from common import legacy


def continuous(raw: pd.DataFrame, future_bins: int) -> pd.DataFrame:
    """Return causal W features and observed future W, respecting 200ms gaps."""
    pieces = []
    for _, g in raw.groupby("run_id", sort=True):
        g = g.sort_values("issuance_s").copy()
        breaks = np.r_[
            True,
            (g.segment.to_numpy()[1:] != g.segment.to_numpy()[:-1])
            | (np.diff(g.bin) != 1)
            | ~np.isclose(np.diff(g.issuance_s), 0.2, atol=1e-7, rtol=0),
        ]
        for block, x in g.groupby(np.cumsum(breaks), sort=False):
            x = x.copy()
            p = x.current_w
            x["continuous_block"] = block
            for lag in [1, 2, 5, 10, 25]:
                x[f"lag_{lag}"] = p.shift(lag)
            for w in [5, 25, 50]:
                x[f"mean_{w}"] = p.rolling(w, min_periods=w).mean()
                x[f"std_{w}"] = p.rolling(w, min_periods=w).std()
            x["slope_1"] = p.diff()
            for k in range(1, future_bins + 1):
                x[f"future_w_{k}"] = p.shift(-k)
            x["valid_anchor"] = (
                p.rolling(50 + future_bins, min_periods=50 + future_bins)
                .count()
                .shift(-future_bins)
                .eq(50 + future_bins)
            )
            x["future_end_s"] = x.issuance_s + future_bins * 0.2
            pieces.append(x)
    if not pieces:
        raise ValueError("stage=features: no continuous input blocks")
    return pd.concat(pieces, ignore_index=True)


feature_columns = legacy.columns
encode_metadata = legacy.encode
