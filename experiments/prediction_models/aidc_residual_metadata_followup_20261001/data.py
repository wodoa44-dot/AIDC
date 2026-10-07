"""Reuse the unchanged pilot's causal features, encoding, weights and hashing."""
import importlib.util
from pathlib import Path
PILOT=Path(__file__).resolve().parent.parent/'aidc_forecast_pilot_20261001'
spec=importlib.util.spec_from_file_location('pilot_data',PILOT/'data.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for name in ['META','KEY','digest','frame_hash','atomic_json','parquet','columns','encode','weights','spaced','sample']:
    globals()[name]=getattr(module,name)
