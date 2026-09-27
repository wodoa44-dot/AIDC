from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path.resolve())
    return cfg

def fingerprint(cfg):
    clean = {k:v for k,v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()

