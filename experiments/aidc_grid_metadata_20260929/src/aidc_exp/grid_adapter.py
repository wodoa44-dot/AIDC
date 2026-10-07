from __future__ import annotations
import json
from pathlib import Path


def unavailable(out: Path):
    provenance={"evidence_level":"E1 only", "E2_status":"blocked_dependency",
        "reason":"No validated dynamic network case with topology, controls, parameters and source license was verified in this repository.",
        "unsupported":["frequency dynamics","voltage dynamics","oscillation damping","UPS switching"]}
    (out/"grid_model_provenance.json").write_text(json.dumps(provenance,indent=2))
    (out/"verification.json").write_text(json.dumps({"E2":"blocked_dependency","E3":"blocked_dependency"},indent=2))
    return provenance
