"""Separate NLR training CPU+GPU pilot: history versus declared job metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core import QUANTILES, crps_quantile, history_features, static_features, target_rows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREP = ROOT / 'trace/preprocessing/outputs/forecast_v2_complete_20260916'
DEFAULT = HERE / 'results/nlr_training_pilot_20260927'
CONDITIONS = ('history_only', 'history_model_node')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(out, sources, metadata):
    frames = []
    for row in sources.itertuples():
        source = pd.read_parquet(row.path)
        target = target_rows(source, horizons=(5,))
        meta = metadata.loc[row.run_id]
        target['group_id'] = meta.group_id
        target['model_name'] = meta.model_name
        target['node_count'] = int(meta.node_count)
        frames.append(target)
        print('built', row.run_id, len(target), flush=True)
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(out / 'targets.parquet', index=False)
    return data


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    resolution = pd.read_csv(PREP / 'resolution_manifest.csv')
    sources = resolution.loc[resolution.dataset_id.eq('nlr_training') &
        resolution.resolution_id.eq('0.2s') & resolution.status.eq('supported')].sort_values('run_id')
    if len(sources) != 41:
        raise ValueError(f'Expected 41 NLR training runs, found {len(sources)}')
    metadata = pd.read_parquet(PREP / 'run_metadata.parquet')
    metadata = metadata.loc[metadata.dataset_id.eq('nlr_training')].set_index('run_id')
    if metadata.group_id.nunique() != 41:
        raise ValueError('Job group mapping is not one-to-one')
    signature = dict(code_sha256=sha(Path(__file__)), core_sha256=sha(HERE / 'core.py'),
        source_hashes={row.run_id: sha(row.path) for row in sources.itertuples()},
        metadata_sha256=sha(PREP / 'run_metadata.parquet'),
        config=dict(horizon_ms=1000, history_ms=5000, train_cap_per_run=1500,
            model_seed=42, split_seed=2026, n_estimators=100, num_leaves=15,
            learning_rate=.07, min_child_samples=30, quantiles=QUANTILES.tolist(), threads=1))
    sigpath = out / 'checkpoint_signature.json'
    if sigpath.exists() and json.loads(sigpath.read_text()) != signature:
        raise ValueError('Code/data/config changed; use a new output directory')
    sigpath.write_text(json.dumps(signature, indent=2) + '\n')
    data = pd.read_parquet(out / 'targets.parquet') if (out / 'targets.parquet').exists() else prepare(out, sources, metadata)
    groups = data[['run_id','group_id','model_name','node_count']].drop_duplicates().sort_values('run_id')
    rng = np.random.default_rng(2026)
    groups['split'] = 'train'
    for _, cohort in groups.groupby(['model_name','node_count']):
        chosen = rng.choice(cohort.index.to_numpy(), size=1)[0]
        groups.loc[chosen, 'split'] = 'test'
    groups.to_csv(out / 'split_manifest.csv', index=False)
    test_runs = set(groups.loc[groups.split.eq('test'), 'run_id'])
    train_runs = set(groups.loc[groups.split.eq('train'), 'run_id'])
    if len(test_runs) != 8 or len(train_runs) != 33 or test_runs & train_runs:
        raise AssertionError('Invalid grouped split')
    train_all = data.index[data.run_id.isin(train_runs)].to_numpy()
    test = data.index[data.run_id.isin(test_runs)].to_numpy()
    sample = []
    for _, group in data.loc[train_all].groupby('run_id', sort=True):
        index = group.index.to_numpy()
        sample.extend(index[np.linspace(0, len(index)-1, min(1500,len(index)), dtype=int)])
    sample = np.sort(np.asarray(sample, dtype=int))
    data.loc[sample, ['run_id','issue_s']].to_csv(out / 'sample_manifest.csv', index=False)
    counts = data.loc[sample].groupby('run_id').size()
    weights = data.loc[sample, 'run_id'].map(1/counts).to_numpy() * len(counts)
    hist_train = np.vstack([history_features(x) for x in data.loc[sample, 'history']])
    hist_test = np.vstack([history_features(x) for x in data.loc[test, 'history']])
    static_train, static_test = static_features(data, sample, test, columns=('model_name','node_count'))
    features = {'history_only': (hist_train, hist_test),
        'history_model_node': (np.column_stack([hist_train,static_train]),
                               np.column_stack([hist_test,static_test]))}
    cells = out / 'cells'
    cells.mkdir(exist_ok=True)
    settings = []
    for condition in CONDITIONS:
        train_x, test_x = features[condition]
        for q in QUANTILES:
            file = cells / f'{condition}_q{q:g}.npy'
            record = cells / f'{condition}_q{q:g}.json'
            if file.exists() and record.exists() and len(np.load(file,mmap_mode='r')) == len(test):
                settings.append(json.loads(record.read_text()))
                continue
            model = lgb.LGBMRegressor(objective='quantile', alpha=float(q), n_estimators=100,
                num_leaves=15, learning_rate=.07, min_child_samples=30,
                random_state=42, n_jobs=1, verbosity=-1)
            model.fit(train_x, data.y.to_numpy()[sample], sample_weight=weights)
            np.save(file, model.predict(test_x))
            fitted = dict(condition=condition, quantile=float(q), fitted_trees=model.booster_.num_trees(),
                objective='quantile', n_estimators=100, num_leaves=15,
                learning_rate=.07, min_child_samples=30, seed=42, n_train_rows=len(sample))
            record.write_text(json.dumps(fitted, indent=2) + '\n')
            settings.append(fitted)
            print(condition, q, flush=True)
    pd.DataFrame(settings).to_csv(out / 'selected_hyperparameters.csv',index=False)
    predictions=[]
    for condition in CONDITIONS:
        raw=np.column_stack([np.load(cells / f'{condition}_q{q:g}.npy') for q in QUANTILES])
        crossing=np.any(np.diff(raw,axis=1)<0,axis=1)
        ordered=np.sort(raw,axis=1)
        scored=data.loc[test,['run_id','group_id','segment_id','issue_s','target_start_s',
            'target_end_s','y','delta_w','model_name','node_count']].copy()
        scored['condition']=condition
        scored['crps_w']=crps_quantile(scored.y.to_numpy(),ordered)
        scored['q50']=ordered[:,7]
        scored['absolute_error_w']=abs(scored.y-scored.q50)
        scored['covered_80']=scored.y.between(ordered[:,3],ordered[:,11])
        scored['crossing_before_correction']=crossing
        for j,q in enumerate(QUANTILES): scored[f'q_{q:g}']=ordered[:,j]
        predictions.append(scored)
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_parquet(out / 'per_target_predictions.parquet',index=False)
    per=pred.groupby(['run_id','condition','model_name','node_count']).agg(
        n_targets=('y','size'),crps_w=('crps_w','mean'),mae_w=('absolute_error_w','mean'),
        p95_w=('absolute_error_w',lambda x:x.quantile(.95)),
        coverage_80=('covered_80','mean')).reset_index()
    per.to_csv(out / 'per_run_metrics.csv',index=False)
    per.groupby('condition',as_index=False)[['crps_w','mae_w','p95_w','coverage_80']].mean().to_csv(
        out / 'aggregate_metrics.csv',index=False)
    pivot=per.pivot(index='run_id',columns='condition',values='crps_w')
    gain=100*(1-pivot.history_model_node.mean()/pivot.history_only.mean())
    result=dict(n_train_runs=33,n_test_runs=8,n_test_rows=len(test),
        history_crps_w=float(pivot.history_only.mean()),metadata_crps_w=float(pivot.history_model_node.mean()),
        descriptive_gain_pct=float(gain),improved_runs=int((pivot.history_model_node<pivot.history_only).sum()),
        inference='single grouped split; jobs share one source experiment; no independent-session CI')
    (out / 'paired_gain_descriptive.json').write_text(json.dumps(result,indent=2)+'\n')
    checks=dict(same_rows=bool(pred.groupby(['run_id','issue_s']).condition.nunique().eq(2).all()),
        same_truth=bool(pred.groupby(['run_id','issue_s']).y.nunique().eq(1).all()),
        no_group_overlap=not bool(set(data.loc[sample,'group_id']) & set(data.loc[test,'group_id'])),
        correct_horizon=bool(np.allclose(pred.target_end_s-pred.issue_s,1)),
        score_recomputed=bool(np.allclose(crps_quantile(pred.y.to_numpy(),
            pred[[f'q_{q:g}' for q in QUANTILES]].to_numpy()),pred.crps_w)))
    checks['all_passed']=all(checks.values())
    (out / 'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not checks['all_passed']: raise AssertionError(checks)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT)
    run(parser.parse_args().out)
