from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core import QUANTILES, STATIC, crps_quantile, history_features, static_features
from train import sample_indices

HERE = Path(__file__).resolve().parent


def run(out):
    out = Path(out)
    frame = pd.read_parquet(out / "targets.parquet")
    frame = frame[frame.horizon_ms == 1000].reset_index(drop=True)
    split = pd.read_csv(out / "split_manifest.csv").set_index("group_id").split
    eligible = np.flatnonzero(frame.group_id.map(split) == "train")
    test = np.flatnonzero(frame.group_id.map(split) == "test")
    ids = sample_indices(frame, eligible)
    counts = frame.iloc[ids].groupby("run_id").size()
    weights = frame.iloc[ids].run_id.map(1 / counts).to_numpy() * len(counts)
    history = np.vstack([history_features(x) for x in frame.history])
    source = frame.iloc[ids].drop_duplicates("run_id").set_index("run_id")
    run_ids = sorted(source.index)
    rows = []
    for seed in (2026, 2027, 2028, 2029, 2030):
        rng = np.random.default_rng(seed)
        donor = dict(zip(run_ids, rng.permutation(run_ids)))
        permuted = frame.copy()
        for recipient, provider in donor.items():
            recipient_rows = permuted.index[permuted.run_id == recipient]
            for col in STATIC:
                permuted.loc[recipient_rows, col] = source.loc[provider, col]
        static_train, static_test = static_features(permuted, ids, test)
        xtrain = np.column_stack([history[ids], static_train])
        xtest = np.column_stack([history[test], static_test])
        raw = []
        for q in QUANTILES:
            model = lgb.LGBMRegressor(objective="quantile", alpha=float(q), n_estimators=100,
                num_leaves=15, learning_rate=.07, min_child_samples=30, random_state=42,
                n_jobs=1, verbosity=-1)
            model.fit(xtrain, frame.y.to_numpy()[ids], sample_weight=weights)
            raw.append(model.predict(xtest))
        pred = np.sort(np.column_stack(raw), axis=1)
        scored = frame.iloc[test][["run_id", "y"]].copy()
        scored["crps_w"] = crps_quantile(scored.y.to_numpy(), pred)
        for run_id, group in scored.groupby("run_id"):
            rows.append(dict(permutation_seed=seed, run_id=run_id, crps_w=group.crps_w.mean(),
                n_targets=len(group), mapping="training run metadata shuffled; test metadata original"))
        pd.DataFrame(rows).to_csv(out / "metadata_permutation.csv", index=False)
        print("permutation", seed, "complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/actual_20260926"))
    run(parser.parse_args().out)
