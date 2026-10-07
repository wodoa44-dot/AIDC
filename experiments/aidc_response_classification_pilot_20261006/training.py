"""Exactly two fixed configurations, three day folds plus final; argmax only."""
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import legacy, KEY, parquet, atomic_json, digest
from labeling import judge
from config import load_config


def fit_predict(
    train: pd.DataFrame,
    ev: pd.DataFrame,
    history: int,
    metadata: bool,
    smoke: bool = False,
    config: dict | None = None,
) -> tuple:
    """Fit on development rows only; return probabilities in fixed R0..R3 coordinates."""
    config = config or load_config()["experiment"]
    params = config["model_parameters"]
    feature_columns = legacy.columns(history, metadata)
    training_features, evaluation_features, mapping, diagnostics = legacy.encode(
        train, ev, feature_columns
    )
    # Treat mapped categories as categorical; unseen code -1 is missing in LightGBM.
    categorical_columns = [d["column"] for d in diagnostics if d["categorical"]]
    target_class = train.label.str[1:].astype(int).to_numpy()
    classes = np.unique(target_class)
    prob = np.zeros((len(ev), 4))
    model = None
    if len(classes) == 1:
        prob[:, classes[0]] = 1
    else:
        model = lgb.LGBMClassifier(
            **{
                **params,
                "n_estimators": config["smoke_trees"]
                if smoke
                else params["n_estimators"],
                "num_class": len(classes),
            }
        )
        # Encoding the present classes makes restoration to global coordinates explicit.
        local_target_class = np.searchsorted(classes, target_class)
        model.fit(
            training_features,
            local_target_class,
            sample_weight=legacy.weights(train),
            categorical_feature=categorical_columns,
        )
        # Installed LightGBM 2.3.1 wraps a two-class multiclass matrix as binary
        # in sklearn predict_proba. The Booster API returns the intended matrix.
        local = np.asarray(model.booster_.predict(evaluation_features)).reshape(
            len(ev), len(classes)
        )
        prob[:, classes] = local
    return (
        prob,
        model,
        {
            "classes": classes.tolist(),
            "missing_classes": sorted(set(range(4)) - set(classes)),
            "features": feature_columns,
            "mapping": mapping,
            "unseen_diagnostics": diagnostics,
            "resolved_parameters": (model or lgb.LGBMClassifier(**params)).get_params(),
        },
    )


def train_models(
    out, train, ev, splits, history, spec, spec_hash, stop_check=None
) -> None:
    """Save eight fixed fold/model configurations with atomic fit checkpoints."""
    records = []
    frames = []

    def output(g, prob, name, split):
        z = g[
            KEY
            + [
                "sample_key",
                "collection_day",
                "issuance_s",
                "current_w",
                "future_end_s",
                "direction",
                "label",
            ]
        ].copy()
        z["scope"] = split["scope"]
        z["fold"] = str(split["fold"])
        z["model"] = name
        z["predicted_class"] = prob.argmax(axis=1)
        for c in range(4):
            z[f"p{c}"] = prob[:, c]
        frames.append(z)

    for s in splits:
        if stop_check is not None:
            stop_check()
        tr = train[
            train.collection_day.isin(s["train_days"]) & train.label.ne("U")
        ].copy()
        va = ev[ev.collection_day.isin(s["validation_days"]) & ev.label.ne("U")].copy()
        if tr.empty or va.empty:
            raise RuntimeError(
                f'No determined train/evaluation labels in fold {s["fold"]}'
            )
        assert_split(tr, va, s["fold"])
        for name, meta in [("M1", False), ("M2", True)]:
            if stop_check is not None:
                stop_check()
            path = out / f'fit_{s["fold"]}_{name}.pkl'
            predpath = out / f'fit_{s["fold"]}_{name}.parquet'
            checkpath = path.with_suffix(".checkpoint.json")
            if checkpath.exists():
                check = json.loads(checkpath.read_text())
                if (
                    not path.exists()
                    or not predpath.exists()
                    or check != {"model": digest(path), "prediction": digest(predpath)}
                ):
                    raise RuntimeError(
                        f'stage=training fold={s["fold"]} model={name}: fit checkpoint corruption'
                    )
                saved = pickle.loads(path.read_bytes())
                frames.append(pd.read_parquet(predpath))
                records.append(saved["record"])
                continue
            prob, model, record = fit_predict(
                tr, va, history, meta, config=spec["experiment"]
            )
            record.update(
                model=name,
                fold=s["fold"],
                scope=s["scope"],
                training_rows=len(tr),
                training_runs=tr.run_id.nunique(),
                fit_performed=model is not None,
            )
            output(va, prob, name, s)
            parquet(predpath, frames[-1])
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps({"model": model, "record": record}))
            tmp.replace(path)
            atomic_json(
                checkpath, {"model": digest(path), "prediction": digest(predpath)}
            )
            records.append(record)
            atomic_json(out / "fit_registry.json", records)
            print(
                f'stage=training fold={s["fold"]} model={name} rows={len(tr)} saved={path}',
                flush=True,
            )
        majority = int(
            np.bincount(
                tr.label.str[1:].astype(int), weights=legacy.weights(tr), minlength=4
            ).argmax()
        )
        for name, level in [("Always-R0", 0), ("Training-majority", majority)]:
            prob = np.zeros((len(va), 4))
            prob[:, level] = 1
            output(va, prob, name, s)
        const = []
        keep = []
        n = spec["scenario"]["future_bins"] + round(
            spec["requirements"]["recovery_s"] / spec["scenario"]["dt_s"]
        )
        for index, r in va.iterrows():
            result = judge([r.current_w] * n, r.current_w, spec, spec_hash)
            if result["label"] != "U":
                keep.append(index)
                const.append(int(result["label"][1]))
        if len(keep) != len(va):
            raise RuntimeError(
                "Constant-load baseline undetermined: report failure rather than compare different cohorts"
            )
        prob = np.zeros((len(va), 4))
        prob[np.arange(len(va)), const] = 1
        output(va, prob, "Constant-load", s)
    assert len(records) == 8
    atomic_json(out / "fit_registry.json", records)
    parquet(out / "predictions.parquet", pd.concat(frames, ignore_index=True))


def assert_split(
    train: pd.DataFrame, evaluation: pd.DataFrame, fold: str | int
) -> None:
    """Reject day/run/key overlap between training and validation cohorts."""
    for column in ("collection_day", "run_id", "sample_key"):
        if set(train[column]) & set(evaluation[column]):
            raise ValueError(f"stage=training fold={fold}: {column} leakage")
