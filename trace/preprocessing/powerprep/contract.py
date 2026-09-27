"""Versioned contracts and atomic, content-verified artifacts for powerprep v2."""
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
from .discovery import stream_hash
SCHEMA_VERSION = '2.0.0'
ADAPTER_VERSION = '2.0.0'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def code_hash():
    return digest({p.name: stream_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    os.replace(tmp, path)


def save_table(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    if path.suffix == '.parquet': frame.to_parquet(tmp, index=False)
    else: frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def cache_key(source_hash, cfg, transformation=None):
    return digest({'source': source_hash, 'config': cfg, 'schema': SCHEMA_VERSION,
                   'adapter': ADAPTER_VERSION, 'code': transformation or code_hash()})


def cached(side, key, paths):
    side = Path(side)
    if not side.exists(): return False
    value = json.loads(side.read_text())
    return value.get('key') == key and all(Path(p).exists() and value.get('files', {}).get(str(p)) == stream_hash(p) for p in paths)


def commit(side, key, paths):
    atomic_json(side, {'key': key, 'files': {str(p): stream_hash(p) for p in paths}})
