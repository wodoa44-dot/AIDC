"""Pure classification metrics and saved-prediction aggregation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, log_loss


def evaluate(predictions: pd.DataFrame, balanced: bool) -> tuple:
    """Return dimensionless ordinal/class metrics; optionally weight each run equally."""
    actual_class = predictions.label.str[1:].astype(int).to_numpy()
    predicted_class = predictions.predicted_class.to_numpy()
    n_rows = len(predictions)
    if balanced:
        count = predictions.groupby("run_id").size()
        sample_weight = predictions.run_id.map(
            1 / (predictions.run_id.nunique() * count)
        ).to_numpy()
    else:
        sample_weight = np.ones(n_rows) / n_rows
    cm = confusion_matrix(
        actual_class, predicted_class, labels=np.arange(4), sample_weight=sample_weight
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_class,
        predicted_class,
        labels=np.arange(4),
        sample_weight=sample_weight,
        zero_division=0,
    )
    support = np.bincount(actual_class, minlength=4)
    present = support > 0
    recall = recall.astype(float)
    recall[~present] = np.nan
    metrics = {
        "ordinal_mae": float(
            np.sum(sample_weight * abs(predicted_class - actual_class))
        ),
        "accuracy": float(np.sum(sample_weight * (predicted_class == actual_class))),
        "macro_f1": float(f1[present].mean()),
        "under_fraction": float(
            np.sum(sample_weight * (predicted_class < actual_class))
        ),
        "under_2plus_fraction": float(
            np.sum(sample_weight * (actual_class - predicted_class >= 2))
        ),
        "severe_as_low_fraction_all": float(
            np.sum(sample_weight * ((actual_class >= 2) & (predicted_class <= 1)))
        ),
        "severe_as_low_fraction_severe": float(
            np.sum(sample_weight * ((actual_class >= 2) & (predicted_class <= 1)))
            / np.sum(sample_weight * (actual_class >= 2))
        )
        if np.any(actual_class >= 2)
        else None,
        "over_fraction": float(
            np.sum(sample_weight * (predicted_class > actual_class))
        ),
        "log_loss": float(
            log_loss(
                actual_class,
                predictions[[f"p{c}" for c in range(4)]].to_numpy(),
                labels=np.arange(4),
                sample_weight=sample_weight,
            )
        ),
        "n": n_rows,
        "n_runs": predictions.run_id.nunique(),
        "macro_classes": ",".join(str(i) for i in np.flatnonzero(present)),
    }
    rows = [
        {
            "class": c,
            "precision": float(precision[c]),
            "recall": None if not present[c] else float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
        }
        for c in range(4)
    ]
    return metrics, rows, cm


def aggregate(out: Path) -> None:
    """Read saved predictions and write cohort/run/class metric tables."""
    pred = pd.read_parquet(out / "predictions.parquet")
    allmetrics = []
    perrun = []
    classes = []
    cms = []
    for (scope, model), predictions in pred.groupby(["scope", "model"]):
        for balanced in [False, True]:
            metric, rows, cm = evaluate(predictions, balanced)
            base = {
                "scope": scope,
                "model": model,
                "aggregation": "run_balanced" if balanced else "row_pooled",
            }
            allmetrics.append({**base, **metric})
            classes += [{**base, **r} for r in rows]
            for actual_class in range(4):
                for predicted_class in range(4):
                    cms.append(
                        {
                            **base,
                            "actual": actual_class,
                            "predicted": predicted_class,
                            "weight": cm[actual_class, predicted_class],
                        }
                    )
        for run, x in predictions.groupby("run_id"):
            metric, _, _ = evaluate(x, False)
            perrun.append(
                {
                    "scope": scope,
                    "model": model,
                    "run_id": run,
                    "collection_day": x.collection_day.iloc[0],
                    **metric,
                }
            )
    pd.DataFrame(allmetrics).to_csv(out / "aggregate_metrics.csv", index=False)
    pd.DataFrame(perrun).to_csv(out / "per_run_metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(out / "class_metrics.csv", index=False)
    pd.DataFrame(cms).to_csv(out / "confusion_plotting.csv", index=False)
    paired = (
        pd.DataFrame(perrun)
        .pivot(
            index=["scope", "run_id", "collection_day"],
            columns="model",
            values="ordinal_mae",
        )
        .reset_index()
    )
    paired["M1_minus_M2"] = paired.M1 - paired.M2
    paired.to_csv(out / "metadata_effect.csv", index=False)
    pred.groupby(["scope", "model", "run_id", "direction"]).apply(
        lambda predictions: pd.Series(evaluate(predictions, False)[0])
    ).to_csv(out / "direction_error_analysis.csv")
