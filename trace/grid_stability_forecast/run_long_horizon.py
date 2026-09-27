"""Resumable fixed-setting 5/10/30 s horizon pilot on highres GPU power."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from build_targets import PREP, build
from core import write_json
from evaluate import evaluate
from train import run as train
from verify import verify

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'results/long_fixed_20260927'
HORIZONS = (5000, 10000, 30000)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def signature():
    manifest = pd.read_csv(PREP / 'resolution_manifest.csv')
    sources = manifest.loc[manifest.dataset_id.eq('highres_node') &
        manifest.resolution_id.eq('0.2s') & manifest.status.eq('supported')].sort_values('run_id')
    return dict(horizons_ms=HORIZONS, fixed_setting='historical 100-tree, 15-leaf, 0.07 learning rate; 5 s history; 15 quantiles; seed 42',
        source_hashes={row.run_id: sha(row.path) for row in sources.itertuples()},
        code_hashes={name: sha(HERE / name) for name in ['run_long_horizon.py', 'build_targets.py',
            'core.py', 'train.py', 'evaluate.py', 'verify.py']},
        source_metadata_sha256=sha(PREP / 'run_metadata.parquet'))


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sig = signature()
    sigpath = out / 'checkpoint_signature.json'
    if sigpath.exists() and json.loads(sigpath.read_text()) != json.loads(json.dumps(sig)):
        raise ValueError('Code/data/config changed; use a new output directory')
    write_json(sigpath, sig)
    progress_path = out / 'progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for horizon in HORIZONS:
        key = f'h{horizon}'
        cell = out / key
        cell.mkdir(exist_ok=True)
        if progress.get(key, {}).get('status') == 'verified':
            verify(cell)
            print('reused verified', key, flush=True)
            continue
        progress[key] = {'status': 'building'}
        write_json(progress_path, progress)
        build(cell, horizons=(horizon//200,))
        progress[key]['status'] = 'training'
        write_json(progress_path, progress)
        train(cell, horizon_ms=horizon)
        progress[key]['status'] = 'evaluating'
        write_json(progress_path, progress)
        evaluate(cell)
        verify(cell)
        progress[key] = {'status': 'verified', 'prediction_sha256': sha(cell / 'per_target_predictions.parquet')}
        write_json(progress_path, progress)
        print('verified', key, flush=True)
    names = ('aggregate_metrics', 'paired_gain_ci', 'transition_metrics', 'calibration_metrics')
    for name in names:
        pd.concat([pd.read_csv(out / f'h{h}' / f'{name}.csv') for h in HORIZONS],
            ignore_index=True).to_csv(out / f'{name}.csv', index=False)
    progress['status'] = 'completed'
    write_json(progress_path, progress)
    print('completed long horizons', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT)
    run(parser.parse_args().out)
