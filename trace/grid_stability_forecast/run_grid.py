from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from build_targets import build
from core import write_json
from evaluate import evaluate
from train import run as train
from verify import verify

HERE = Path(__file__).resolve().parent
HORIZONS = tuple(range(200, 2001, 200))


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for horizon in HORIZONS:
        cell = out / f"h{horizon}"
        cell.mkdir(exist_ok=True)
        key = f"h{horizon}"
        source_code = hashlib.sha256(Path(__file__).read_bytes() + (HERE / "train.py").read_bytes() +
                                      (HERE / "core.py").read_bytes()).hexdigest()
        if progress.get(key, {}).get("status") == "verified" and progress[key].get("code_sha256") == source_code:
            verify(cell)
            print("reused verified", key, flush=True)
            continue
        progress[key] = {"status": "building", "code_sha256": source_code}
        write_json(progress_path, progress)
        build(cell, horizons=(horizon // 200,))
        progress[key]["status"] = "training"
        write_json(progress_path, progress)
        train(cell, horizon_ms=horizon)
        progress[key]["status"] = "evaluating"
        write_json(progress_path, progress)
        evaluate(cell)
        checks = verify(cell)
        resolved = json.loads((cell / "resolved_config.json").read_text())
        progress[key].update(status="verified", test_rows=resolved["test_rows"],
                             train_runs=resolved["train_runs"], test_runs=resolved["test_runs"])
        write_json(progress_path, progress)
        print("verified", key, flush=True)
    for name in ("aggregate_metrics", "paired_gain_ci", "transition_metrics", "calibration_metrics"):
        frames = [pd.read_csv(out / f"h{h}" / f"{name}.csv") for h in HORIZONS]
        pd.concat(frames, ignore_index=True).to_csv(out / f"{name}.csv", index=False)
    write_json(out / "resolved_config.json", {"horizons_ms": HORIZONS,
        "assessment": "fixed-setting grouped outer pilot", "same_split_and_quantile_grid": True,
        "inner_tuning": False, "device_control": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/grid_fixed_20260926"))
    run(parser.parse_args().out)
