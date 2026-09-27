from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "trace/preprocessing/outputs"
PREP = OLD / "forecast_v2_complete_20260916"


def inventory(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(OLD / "manifest.csv")
    source = source[~source.is_duplicate].copy()
    resolution = pd.read_csv(PREP / "resolution_manifest.csv")
    analysis = resolution[(resolution.resolution_id == "0.2s") & (resolution.status == "supported")]
    analysis = analysis[["run_id", "samples", "interval_s", "aggregation", "availability", "realtime_interpretation"]]
    analysis = analysis.rename(columns={"samples": "analysis_rows", "interval_s": "analysis_grid_s"})
    meta = pd.read_parquet(PREP / "run_metadata.parquet").drop_duplicates("run_id")
    meta = meta[["run_id", "group_id", "dt_median_s", "source_sensor_cadence_s", "sensor_averaging_window_s",
                 "request_rate", "gpu_model", "model_name", "model_size", "batch_size"]]
    meta = meta.rename(columns={"dt_median_s": "native_recorded_step_s"})
    joined = source.merge(analysis, on="run_id", how="left", validate="one_to_one")
    joined = joined.merge(meta, on="run_id", how="left", validate="one_to_one")
    joined["source_rows"] = joined.row_count
    joined["source_grid_s"] = joined.native_recorded_step_s
    joined["physical_sensor_cadence_s"] = joined.source_sensor_cadence_s
    joined["source_grid_note"] = "median source-file timestamp step; interpolation may be present"
    joined["physical_sensor_cadence_note"] = "unknown unless source documentation or measurement verifies it"
    joined[["dataset", "run_id", "group_id", "power_scope", "source_path", "source_hash",
            "source_rows", "source_grid_s", "analysis_rows", "analysis_grid_s",
            "physical_sensor_cadence_s", "sensor_averaging_window_s", "aggregation", "availability",
            "realtime_interpretation", "gpu_model", "model_name", "model_size", "batch_size", "request_rate",
            "source_grid_note", "physical_sensor_cadence_note"]].to_csv(out / "data_inventory.csv", index=False)
    return joined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/actual_20260926"))
    inventory(parser.parse_args().out)
