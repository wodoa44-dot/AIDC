"""Recount the legacy NLR training 200 ms recovery statistic with offline v2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PREP = HERE.parent / "preprocessing"
sys.path.insert(0, str(PREP))

from powerprep.config import load_config  # noqa: E402
from powerprep.events_v2 import detect_all  # noqa: E402


def run(source: Path, output: Path) -> pd.DataFrame:
    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    resolutions = pd.read_csv(source / "resolution_manifest.csv")
    chosen = resolutions[(resolutions.dataset_id == "nlr_training")
                         & (resolutions.resolution_id == "0.2s")
                         & (resolutions.status == "supported")]
    if chosen.empty or chosen.run_id.duplicated().any():
        raise ValueError("no unique supported NLR training 200 ms inputs")
    chosen.to_csv(output / "resolution_manifest.csv", index=False)
    inventory = pd.read_csv(source / "manifest.csv")
    inventory = inventory[inventory.run_id.isin(chosen.run_id)]
    if len(inventory) != len(chosen):
        raise ValueError("resolution run missing from inventory")
    inventory.to_csv(output / "manifest.csv", index=False)
    config = load_config(PREP / "configs/forecast_v2.yaml")
    config["events"]["resolution_s"] = 0.2
    config["detector"]["delta_event_s"] = [1.0]
    config["detector"]["relative_drop_thresholds"] = [0.2]
    detect_all(output, config)
    events = pd.read_parquet(output / "events.parquet")
    primary = events[(events.relative_threshold == 0.2) & (events.delta_event_s == 1.0)]
    qualified = primary[(primary.low_duration_s >= 30)
                        & (primary.recovery_rise_time_10_90_s <= 2)]
    old = pd.read_parquet(PREP / "outputs/events.parquet")
    old = old[(old.dataset == "nlr_training")
              & (old.relative_threshold == 0.2)
              & (old.delta_event_s == 1.0)]
    old_qualified = old[(old.low_duration_s >= 30)
                        & (old.recovery_rise_time_10_90_s <= 2)]
    summary = pd.DataFrame([
        dict(version="legacy_200ms", events=len(old), long_fast_events=len(old_qualified),
             long_fast_runs=old_qualified.run_id.nunique(),
             resolved_recovery=old.recovery_rise_time_10_90_s.notna().sum()),
        dict(version="v2_final_recovery_200ms", events=len(primary),
             long_fast_events=len(qualified), long_fast_runs=qualified.run_id.nunique(),
             resolved_recovery=primary.recovery_rise_time_10_90_s.notna().sum()),
    ])
    summary.to_csv(output / "comparison.csv", index=False)
    (output / "run_manifest.json").write_text(json.dumps({
        "source": str(source.resolve()), "source_resolution_id": "0.2s",
        "dataset": "nlr_training", "input_runs": len(chosen),
        "detector_relative_threshold": 0.2, "delta_event_s": 1.0,
        "event_definition": "offline v2 final recovery; centered smoothing",
        "training_inputs_available_online": False
    }, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
        default=PREP / "outputs/forecast_v2_complete_20260916")
    parser.add_argument("--out", type=Path,
        default=HERE / "results/event_v2_nlr_training_20260926")
    args = parser.parse_args()
    print(run(args.source, args.out).to_string(index=False))
