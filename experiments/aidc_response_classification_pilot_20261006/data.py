"""Continuous history and observed future paths, sampled independently of power/label."""
import json
import numpy as np
import pandas as pd
from features import continuous
from config import load_config
from common import PROJECT, legacy, KEY, META, atomic_json, parquet

PREPARED = (
    PROJECT
    / "experiments/aidc_grid_metadata_20260929/outputs/full_20260928T180455Z/prepared.parquet"
)
GROUPS = (
    PROJECT
    / "experiments/aidc_control_fair_budget_20260930/nested/outputs/full_v2/group_manifest.csv"
)
SELECTION = (
    PROJECT
    / "experiments/prediction_models/aidc_forecast_pilot_20261001/outputs/pilot_v1/selected_configuration.json"
)


def history_selection(config: dict | None = None) -> tuple:
    """Reuse verified development-only history selection, otherwise fixed fallback."""
    config = config or load_config()["experiment"]
    record = json.loads(SELECTION.read_text()) if SELECTION.exists() else {}
    from common import digest, LEGACY

    identity_path = SELECTION.parent / "identity.json"
    prior = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    verified = all(
        prior.get("inputs", {}).get(str(p)) == digest(p) for p in [PREPARED, GROUPS]
    ) and prior.get("code", {}).get("data.py") == digest(LEGACY)
    if (
        verified
        and record.get("selection_scope") == "development only"
        and record.get("history") in (1, 5, 10)
    ):
        return record["history"], {
            "source": str(SELECTION),
            "source_sha256": digest(SELECTION),
            "prior_identity": str(identity_path),
            "input_and_feature_hashes_verified": True,
            "record": record,
        }
    return config["history_fallback_s"], {
        "source": "preregistered fallback design",
        "reason": "verifiable development selection unavailable",
    }


def prepare(
    out, future_bins: int = 5, provisional: bool = False, config: dict | None = None
) -> tuple:
    """Read bins and save independently sampled anchors; future power is observed W."""
    config = config or load_config()["experiment"]
    development = config["development_days"]
    holdout = config["holdout_day"]
    # Do not carry old regression targets or solver fields into this frame.
    raw = pd.read_parquet(PREPARED, columns=KEY + ["current_w", "issuance_s"] + META)
    groups = pd.read_csv(GROUPS, dtype={"collection_day": str})
    assert raw.run_id.nunique() == 32 and not groups.run_id.duplicated().any()
    assert set(raw.run_id) == set(groups.run_id)
    dates = sorted(groups.collection_day.unique())
    mismatches = (
        []
        if set(dates) == set(development + [holdout])
        else ["manifest dates differ from requested split"]
    )
    raw["collection_day"] = raw.run_id.map(groups.set_index("run_id").collection_day)
    data = continuous(raw, future_bins)
    valid = data[data.valid_anchor].copy()
    candidates = pd.concat(
        [
            legacy.spaced(g)
            for _, g in valid.groupby(["run_id", "continuous_block"], sort=True)
        ],
        ignore_index=True,
    )
    counts = candidates.groupby("run_id").size()
    included = set(counts[counts >= config["minimum_candidates"]].index)
    tr = []
    ev = []
    for run, g in candidates[candidates.run_id.isin(included)].groupby(
        "run_id", sort=True
    ):
        if g.collection_day.iloc[0] in development:
            tr.append(legacy.sample(g, config["train_cap"], True, config["seed"]))
        if g.collection_day.iloc[0] in development + [holdout]:
            ev.append(legacy.sample(g, config["evaluation_cap"], False, config["seed"]))
    if not tr or not ev:
        raise ValueError("stage=data_preparation: no eligible training/evaluation runs")
    train = pd.concat(tr, ignore_index=True)
    evaluation = pd.concat(ev, ignore_index=True)
    for d in (train, evaluation):
        d["sample_key"] = (
            d.run_id + ":" + d.segment.astype(str) + ":" + d.bin.astype(str)
        )
        d["issuance_rule"] = "end_of_latest_completed_200ms_bin"
        d["anchor_status"] = (
            "provisional_recovery_length_unresolved"
            if provisional
            else "full_spec_future_available"
        )
        assert not d.duplicated(KEY).any()
        assert np.isfinite(d[legacy.columns(10)]).all().all()
    inventory = groups.merge(
        raw.drop_duplicates("run_id")[["run_id"] + META],
        on="run_id",
        validate="one_to_one",
    )
    for name, frame in [
        ("original_bins", raw),
        ("valid_anchors", valid),
        ("spaced_candidates", candidates),
        ("training_rows", train),
        ("evaluation_rows", evaluation),
    ]:
        inventory[name] = (
            inventory.run_id.map(frame.groupby("run_id").size()).fillna(0).astype(int)
        )
    inventory["included"] = inventory.run_id.isin(included)
    inventory["exclusion_reason"] = np.where(
        inventory.included,
        "",
        f'fewer_than_{config["minimum_candidates"]}_continuous_candidates',
    )
    inventory["full_label_anchor_validity_confirmed"] = not provisional
    inventory.to_csv(out / "run_inventory.csv", index=False)
    inventory[~inventory.included].to_csv(out / "excluded_runs.csv", index=False)
    parquet(out / "continuous_features.parquet", data)
    parquet(out / "train_anchors.parquet", train)
    parquet(out / "evaluation_anchors.parquet", evaluation)
    splits = [
        {
            "fold": i,
            "scope": "development",
            "train_days": [x for x in development if x != day],
            "validation_days": [day],
        }
        for i, day in enumerate(development)
    ]
    splits.append(
        {
            "fold": "final",
            "scope": "holdout",
            "train_days": development,
            "validation_days": [holdout],
        }
    )
    atomic_json(
        out / "split_manifest.json",
        {
            "folds": splits,
            "actual_dates": dates,
            "mismatches": mismatches,
            "independent_sessions_verified": False,
            "holdout_used_in_previous_experiments": True,
        },
    )
    atomic_json(
        out / "feature_schema.json",
        {
            "M1": legacy.columns(history_selection(config)[0]),
            "M2": legacy.columns(history_selection(config)[0], True),
            "forbidden": [
                "run_id",
                "collection_day",
                "source_file",
                "future_*",
                "solver_*",
            ],
            "future_bins": future_bins,
            "provisional": provisional,
        },
    )
    return train, evaluation, inventory, splits, mismatches
