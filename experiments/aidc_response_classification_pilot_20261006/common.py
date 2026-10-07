"""Read-only reuse of the existing pilot's feature/sampling helpers."""
import importlib.util
import sys

sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
LEGACY = PROJECT / "experiments/prediction_models/aidc_forecast_pilot_20261001/data.py"
spec = importlib.util.spec_from_file_location("legacy_pilot_data", LEGACY)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
digest, atomic_json, parquet = legacy.digest, legacy.atomic_json, legacy.parquet
KEY, META = legacy.KEY, legacy.META
