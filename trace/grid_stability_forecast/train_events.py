from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from core import history_features, static_features
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
    power = np.vstack([history_features(x) for x in frame.history])
    s1_train, s1_test = static_features(frame, ids, test,
        columns=("gpu_model", "workload_type", "model_name", "batch_size"))
    s2_train, s2_test = static_features(frame, ids, test)
    conditions = {"history_only": (power[ids], power[test]),
                  "history_static_s1": (np.column_stack([power[ids], s1_train]),
                                        np.column_stack([power[test], s1_test])),
                  "history_static_s2": (np.column_stack([power[ids], s2_train]),
                                        np.column_stack([power[test], s2_test]))}
    forecasts, metrics = [], []
    for label in ("rise_event", "drop_event"):
        truth = frame[label].to_numpy(int)
        for condition, (xtrain, xtest) in conditions.items():
            model = lgb.LGBMClassifier(objective="binary", n_estimators=100, num_leaves=15,
                learning_rate=.07, min_child_samples=30, random_state=42, n_jobs=1, verbosity=-1)
            model.fit(xtrain, truth[ids], sample_weight=weights)
            probability = model.predict_proba(xtest)[:, 1]
            scored = frame.iloc[test][["run_id", "group_id", "issue_s", "target_end_s"]].copy()
            scored["condition"] = condition
            scored["event"] = label
            scored["y_true"] = truth[test]
            scored["probability"] = probability
            forecasts.append(scored)
            for run_id, group in scored.groupby("run_id"):
                actual = group.y_true.to_numpy(bool)
                detected = group.probability.to_numpy() >= .5
                tp = int(np.sum(actual & detected))
                fp = int(np.sum(~actual & detected))
                fn = int(np.sum(actual & ~detected))
                duration_h = (group.issue_s.max()-group.issue_s.min()+.2) / 3600
                metrics.append(dict(run_id=run_id, condition=condition, event=label,
                    n_windows=len(group), n_positive=int(actual.sum()), threshold=.5,
                    brier=brier_score_loss(actual, group.probability),
                    pr_auc=average_precision_score(actual, group.probability) if actual.any() and (~actual).any() else np.nan,
                    precision=tp/(tp+fp) if tp+fp else np.nan,
                    recall=tp/(tp+fn) if tp+fn else np.nan,
                    missed_window_fraction=fn/actual.sum() if actual.sum() else np.nan,
                    false_positive_windows_per_hour=fp/duration_h if duration_h else np.nan,
                    assessment="window-level event labels; not one-to-one onset matching"))
    pd.concat(forecasts, ignore_index=True).to_parquet(out / "event_predictions.parquet", index=False)
    pd.DataFrame(metrics).to_csv(out / "event_metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/actual_20260926"))
    run(parser.parse_args().out)
