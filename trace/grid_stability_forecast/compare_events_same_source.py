"""Compare legacy and final-episode recovery on identical NLR training bins."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "preprocessing"))
from powerprep.config import load_config  # noqa: E402
from powerprep.events import config_grid, detect_segment  # noqa: E402


def run(out: Path) -> pd.DataFrame:
    out = Path(out)
    manifest = pd.read_csv(out / "manifest.csv")
    manifest = manifest.loc[~manifest.is_duplicate].set_index("run_id")
    resolutions = pd.read_csv(out / "resolution_manifest.csv")
    config = load_config(HERE.parent / "preprocessing/configs/forecast_v2.yaml")
    config["detector"]["delta_event_s"] = [1.0]
    config["detector"]["relative_drop_thresholds"] = [0.2]
    detector = next(config_grid(config))
    records = []
    for item in resolutions.itertuples():
        series = pd.read_parquet(item.path)
        row = SimpleNamespace(**manifest.loc[item.run_id].to_dict(), run_id=item.run_id)
        for segment_id, segment in series.groupby("segment_id", sort=False):
            if segment.is_valid.all():
                records.extend(detect_segment(segment, row, segment_id, detector,
                                              "offline same-source comparison"))
    legacy = pd.DataFrame(records)
    legacy.to_parquet(out / "legacy_same_source_events.parquet", index=False)
    v2 = pd.read_parquet(out / "events.parquet")
    key = ["run_id", "segment_id", "event_id", "low_entry_s", "recovery_onset_s"]
    a = legacy[key].fillna(-1).drop_duplicates()
    b = v2[key].fillna(-1).drop_duplicates()
    same_events = len(a) == len(b) and len(a.merge(b, on=key, validate="one_to_one")) == len(a)
    def count(frame):
        selected = frame[(frame.low_duration_s >= 30)
                         & (frame.recovery_rise_time_10_90_s <= 2)]
        return dict(events=len(frame), long_fast_events=len(selected),
                    long_fast_runs=selected.run_id.nunique(),
                    resolved_recovery=frame.recovery_rise_time_10_90_s.notna().sum())
    comparison = pd.DataFrame([
        dict(version="legacy_same_v2_source", **count(legacy)),
        dict(version="v2_same_source", **count(v2)),
    ])
    comparison.to_csv(out / "controlled_comparison.csv", index=False)
    (out / "controlled_verification.json").write_text(json.dumps({
        "same_event_identity_and_onset": bool(same_events),
        "input_runs": int(resolutions.run_id.nunique()),
        "legacy_event_count": len(legacy), "v2_event_count": len(v2),
        "difference_is_recovery_measurement_only": bool(same_events),
    }, indent=2) + "\n")
    if not same_events:
        raise AssertionError("legacy and v2 event identities differ on the same source")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=HERE / "results/event_v2_nlr_training_20260926")
    print(run(parser.parse_args().out).to_string(index=False))
