from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_splits(manifest: pd.DataFrame, out: Path, folds=5) -> list[dict]:
    # Exact duplicate paths share run_id; independence of collection sessions remains unverified.
    groups = manifest.run_id.to_numpy()
    n = min(folds, len(set(groups)))
    if n < 2: raise ValueError("at least two run groups required")
    result = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n).split(manifest, groups=groups)):
        result.append({"fold": fold, "train_runs": sorted(groups[train].tolist()), "test_runs": sorted(groups[test].tolist())})
    out.write_text(json.dumps({"group_rule": "source SHA256; collection session independence unverified",
                               "folds": result}, indent=2))
    return result


def sample_training(df: pd.DataFrame, max_per_run=5000, seed=2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("run_id", sort=True):
        if len(g) > max_per_run:
            ix = np.sort(rng.choice(g.index.to_numpy(), max_per_run, replace=False))
            g = g.loc[ix]
        g = g.copy(); g["sample_weight"] = 1. / len(g)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
