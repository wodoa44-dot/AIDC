"""Load fixed experiment settings; unresolved physical specs remain a separate gate."""
from pathlib import Path
from typing import Any
import yaml
from common import ROOT

DEFAULT_CONFIG = ROOT / "configs/pilot.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Return YAML settings; validate sampling counts, dates and class coordinates."""
    spec = yaml.safe_load(Path(path).read_text())
    if not isinstance(spec, dict):
        raise ValueError(f"stage=config path={path}: expected mapping")
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text())["experiment"]
    cfg = {**defaults, **spec.get("experiment", {})}
    for key in (
        "train_cap",
        "evaluation_cap",
        "minimum_candidates",
        "smoke_rows",
        "smoke_trees",
        "label_checkpoint_rows",
        "severe_minimum_anchors",
    ):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"stage=config field={key}: positive integer required")
    if any(cfg[key] % 10 for key in ("train_cap", "evaluation_cap")):
        raise ValueError("stage=config: sampling caps must be divisible by ten strata")
    days = cfg["development_days"]
    if len(days) != 3 or len(set(days)) != 3 or cfg["holdout_day"] in days:
        raise ValueError(
            "stage=config: three distinct development days and separate holdout required"
        )
    if cfg["class_mapping"] != {f"R{i}": i for i in range(4)}:
        raise ValueError("stage=config: unsupported class mapping")
    if cfg["eligibility_history_s"] != 10 or cfg["anchor_interval_s"] != 1:
        raise ValueError(
            "stage=config: inherited feature/sampling helpers require 10s eligibility and 1s spacing"
        )
    if cfg["history_fallback_s"] not in (1, 5, 10):
        raise ValueError("stage=config: unsupported feature history seconds")
    params = cfg["model_parameters"]
    if params["random_state"] != cfg["seed"] or not 1 <= params["n_jobs"] <= 4:
        raise ValueError("stage=config: seed mismatch or unsupported thread budget")
    spec["experiment"] = cfg
    return spec
