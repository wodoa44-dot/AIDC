from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
from .preprocessing import static_metadata

ROOT = Path(__file__).resolve().parents[4]
OLD_MANIFEST = ROOT / "trace/preprocessing/outputs/manifest.csv"


def make_inventory(out: Path) -> pd.DataFrame:
    m = pd.read_csv(OLD_MANIFEST)
    m["independent_session_verified"] = False
    m["analysis_use"] = m.dataset.eq("highres_node") & ~m.is_duplicate
    m["exclusion_reason"] = ""
    m.loc[m.is_duplicate, "exclusion_reason"] = "identical source SHA256 alias"
    m.loc[m.dataset.str.startswith("nlr"), "exclusion_reason"] = "aggregate interpolated; causal raw reconstruction unverified"
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "inventory.csv", index=False)
    m.loc[~m.analysis_use, ["dataset", "run_id", "source_path", "exclusion_reason"]].to_csv(out / "exclusions.csv", index=False)
    cap = [
      ("highres node GPU W", "available", "raw CSV; timestamp and eight GPU power channels", "E0 prediction; GPU demand E1 proxy", "facility P/Q/V/f or actuator response"),
      ("NLR aggregate", "available, noncausal reconstruction", "utilities.py interpolates asynchronous sources", "separate retrospective audit", "causal raw device sample availability"),
      ("GPU cap/performance", "not identified", "no paired cap experiment found", "assumption sensitivity only", "throughput/latency and controlled consumption"),
      ("ESS/fast storage", "not identified", "no measured response/SoC trace found", "assumption sensitivity only", "delay, efficiency, capacity, SoC and reserve"),
      ("grid dynamics", "not verified", "no validated target facility dynamic case found", "E2 only if sourced benchmark added", "topology, dynamic controls, parameters and verification"),
    ]
    pd.DataFrame(cap, columns=["item", "availability", "provenance", "possible_experiment", "missing_for_claim"]).to_csv(out / "capability_matrix.csv", index=False)
    schema = {"static": {k: "filename token; unknown is missing" for k in ["model_size_token", "image_size", "batch_size", "sequence_length", "zero_stage"]},
              "hardware": "path label", "workload": "path label", "gpu_count": "8 GPU channels, constant", "dynamic": "unavailable"}
    (out / "metadata_schema.json").write_text(json.dumps(schema, indent=2))
    return m[m.analysis_use].copy()


def fingerprint(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(str(p).encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()
