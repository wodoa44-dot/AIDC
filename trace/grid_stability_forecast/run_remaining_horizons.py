"""Resume the missing 200 ms-grid nested horizon cells, one CPU-bound cell at a time."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MISSING_HORIZONS_MS = (200, 400, 600, 800, 1200, 1400, 1600, 1800)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def verified(out: Path, source: Path, horizon_ms: int) -> bool:
    progress_path = out / "progress.json"
    prediction_path = out / "per_target_predictions.parquet"
    signature_path = out / "checkpoint_signature.json"
    if not all(p.exists() for p in (progress_path, prediction_path, signature_path)):
        return False
    progress = json.loads(progress_path.read_text())
    signature = json.loads(signature_path.read_text())
    code = signature.get("code_sha256", {})
    current_code = all(code.get(name) == sha256(HERE / name) for name in
                       ("run_nested_horizon.py", "nested_select.py", "core.py",
                        "unseen_condition.py", "verify_nested.py"))
    return (progress.get("status") == "completed"
            and progress.get("horizon_ms") == horizon_ms
            and progress.get("verification", {}).get("all_passed") is True
            and progress.get("prediction_sha256") == sha256(prediction_path)
            and signature.get("target_sha256") == sha256(source / "targets.parquet")
            and signature.get("split_sha256") == sha256(source / "split_manifest.csv")
            and current_code)


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("a") as log:
        log.write(f"\n[{utc_now()}] $ {' '.join(command)}\n")
        log.flush()
        subprocess.run(command, cwd=HERE.parents[1], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def run(tag: str, horizons: tuple[int, ...]) -> None:
    campaign = RESULTS / tag
    campaign.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign / "progress.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "assessment": "reused highres outer holdout; not independent confirmation",
        "horizons_ms": list(horizons), "cells": {}, "started_utc": utc_now(),
    }
    if manifest["horizons_ms"] != list(horizons):
        raise ValueError("Resume requires the same horizon list and campaign tag")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    for horizon in horizons:
        source = RESULTS / "grid_fixed_20260926" / f"h{horizon}"
        out = campaign / f"h{horizon}"
        if not (source / "targets.parquet").exists():
            raise FileNotFoundError(source / "targets.parquet")
        cell = manifest["cells"].setdefault(str(horizon), {})
        cell.update(status="running", started_or_resumed_utc=utc_now(),
                    source=str(source), output=str(out))
        write_json(manifest_path, manifest)
        log_path = campaign / f"h{horizon}.log"
        try:
            if not verified(out, source, horizon):
                run_command([sys.executable, str(HERE / "run_nested_horizon.py"),
                             "--source", str(source), "--out", str(out),
                             "--horizon-ms", str(horizon)], log_path, env)
            if not verified(out, source, horizon):
                raise RuntimeError(f"Horizon {horizon} finished without verified predictions")
            run_command([sys.executable, str(HERE / "summarize_nested.py"),
                         "--out", str(out)], log_path, env)
            run_command([sys.executable, str(HERE / "diagnose_calibration.py"),
                         "--out", str(out)], log_path, env)
            cell.update(status="verified", completed_utc=utc_now(),
                        prediction_sha256=sha256(out / "per_target_predictions.parquet"))
            write_json(manifest_path, manifest)
        except Exception as exc:
            cell.update(status="failed", failed_utc=utc_now(), error=repr(exc))
            write_json(manifest_path, manifest)
            raise
    manifest["status"] = "completed"
    manifest["completed_utc"] = utc_now()
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="nested_remaining_20260928")
    parser.add_argument("--horizons-ms", type=int, nargs="+",
                        default=list(MISSING_HORIZONS_MS))
    args = parser.parse_args()
    if len(set(args.horizons_ms)) != len(args.horizons_ms):
        parser.error("Duplicate horizon")
    run(args.tag, tuple(args.horizons_ms))
