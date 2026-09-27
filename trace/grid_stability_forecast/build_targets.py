from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core import target_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "trace/preprocessing/outputs/forecast_v2_complete_20260916"


def build(out, limit_runs=None, horizons=(5,)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(PREP / "resolution_manifest.csv")
    manifest = manifest[(manifest.dataset_id == "highres_node") & (manifest.resolution_id == "0.2s") &
                        (manifest.status == "supported")].sort_values("run_id")
    if limit_runs:
        manifest = manifest.head(limit_runs)
    metadata = pd.read_parquet(PREP / "run_metadata.parquet").drop_duplicates("run_id").set_index("run_id")
    frames = []
    for row in manifest.itertuples():
        frame = pd.read_parquet(row.path)
        targets = target_rows(frame, horizons=horizons)
        source = metadata.loc[row.run_id]
        for col in ("gpu_model", "workload_type", "model_name", "model_size", "batch_size", "sequence_length", "group_id"):
            targets[col] = source[col]
        try:
            details = json.loads(source.source_metadata_json)
        except (TypeError, ValueError):
            details = {}
        targets["image_size"] = details.get("image_size")
        name = str(source.original_run_id)
        targets["zero_stage"] = name.split("Deepspeedds_z")[-1].split("_")[0] if "Deepspeedds_z" in name else None
        frames.append(targets)
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(out / "targets.parquet", index=False)
    write_json(out / "target_manifest.json", {"rows": len(result), "runs": result.run_id.nunique(),
        "horizons_ms": sorted(result.horizon_ms.unique().tolist()), "threshold_w": 100,
        "history_s": 5, "measurement": "8 GPU sum, 200 ms completed bins", "source": str(PREP),
        "status": "retrospective; upstream sensor latency unknown"})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent / "results/actual_20260926"))
    parser.add_argument("--limit-runs", type=int)
    args = parser.parse_args()
    build(args.out, args.limit_runs)
