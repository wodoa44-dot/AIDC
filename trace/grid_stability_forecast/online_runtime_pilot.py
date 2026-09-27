"""Within-acquisition B1/B2/B3/B4 pilot using past vLLM runtime logs.

The 200 NLR runs share one acquisition group. Results are descriptive and are
not an independent-session validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core import QUANTILES, crps_quantile, history_features, target_rows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREP = ROOT / 'trace/preprocessing/outputs/forecast_v2_complete_20260916'
RUNTIME = HERE / 'results/online_runtime_20260927'
CONDITIONS = ('history_only', 'history_static_rate', 'history_dynamic', 'history_static_dynamic')
DYNAMIC = ('prompt_tokens_s', 'generation_tokens_s', 'running_requests',
           'waiting_requests', 'kv_cache_pct', 'log_age_s')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(out):
    manifest = pd.read_csv(PREP / 'resolution_manifest.csv')
    manifest = manifest.loc[manifest.dataset_id.eq('nlr_online_rate') &
        manifest.resolution_id.eq('0.2s') & manifest.status.eq('supported')].copy()
    if len(manifest) != 200:
        raise ValueError(f'Expected 200 online-rate traces, found {len(manifest)}')
    coverage = pd.read_csv(RUNTIME / 'runtime_coverage.csv', dtype={'id': str})
    coverage.id = coverage.id.str.zfill(6)
    if coverage.group_id.nunique() != 1:
        raise ValueError('Unexpected acquisition grouping')
    manifest = manifest.merge(coverage[['run_id', 'id', 'request_rate']], on='run_id', validate='one_to_one')
    logs = pd.read_parquet(RUNTIME / 'runtime_observations.parquet')
    frames = []
    for row in manifest.itertuples():
        source = pd.read_parquet(row.path)
        targets = target_rows(source, horizons=(5,))
        targets['original_id'] = int(row.id)
        targets['request_rate'] = float(row.request_rate)
        frames.append(targets)
    targets = pd.concat(frames, ignore_index=True)
    # merge_asof needs the global time key sorted, even when matching by run.
    targets = targets.sort_values(['issue_s', 'run_id']).reset_index(drop=True)
    logs = logs.rename(columns={'effective_available_s': 'runtime_available_s'})
    logs = logs.sort_values(['runtime_available_s', 'run_id'])
    joined = pd.merge_asof(targets, logs[['run_id', 'runtime_available_s', *DYNAMIC[:-1]]],
        left_on='issue_s', right_on='runtime_available_s', by='run_id', direction='backward')
    joined['log_age_s'] = joined.issue_s - joined.runtime_available_s
    joined = joined.loc[joined.runtime_available_s.notna() & joined.log_age_s.le(20)].copy()
    if joined.empty or (joined.runtime_available_s > joined.issue_s + 1e-9).any():
        raise AssertionError('No causal, paired rows')
    joined['split'] = np.where(joined.original_id.le(150), 'train', 'test')
    if joined.loc[joined.split.eq('train'), 'run_id'].nunique() != 150 or \
       joined.loc[joined.split.eq('test'), 'run_id'].nunique() != 50:
        raise AssertionError('Chronological run split incomplete')
    joined = joined.sort_values(['original_id', 'issue_s']).reset_index(drop=True)
    joined.to_parquet(out / 'paired_targets.parquet', index=False)
    joined[['run_id', 'original_id', 'request_rate', 'split']].drop_duplicates().to_csv(
        out / 'split_manifest.csv', index=False)
    return joined


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(code_sha256=sha(Path(__file__)), core_sha256=sha(HERE / 'core.py'),
        runtime_sha256=sha(RUNTIME / 'runtime_observations.parquet'),
        coverage_sha256=sha(RUNTIME / 'runtime_coverage.csv'),
        source_manifest_sha256=sha(PREP / 'resolution_manifest.csv'),
        experiment={'train_run_ids': '1-150', 'test_run_ids': '151-200', 'history_s': 5,
            'horizon_s': 1, 'bin_s': .2, 'train_rows_per_run_cap': 500,
            'sampling_seed': 42, 'model_seed': 42, 'trees': 100, 'num_leaves': 15,
            'learning_rate': .07, 'min_child_samples': 30, 'threads': 1,
            'runtime_availability_lag_s': 1, 'max_runtime_age_s': 20,
            'assessment': 'within one acquisition; interpolated source; retrospective clock alignment'})
    signature_path = out / 'checkpoint_signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError('Code, data or configuration changed; use a new output directory')
    signature_path.write_text(json.dumps(signature, indent=2) + '\n')
    frame = pd.read_parquet(out / 'paired_targets.parquet') if (out / 'paired_targets.parquet').exists() else prepare(out)
    train = frame.loc[frame.split.eq('train')]
    test = frame.loc[frame.split.eq('test')]
    rng = np.random.default_rng(42)
    sample = np.concatenate([rng.choice(group.index.to_numpy(), size=min(500, len(group)), replace=False)
        for _, group in train.groupby('run_id', sort=True)])
    sample = np.sort(sample)
    sample_frame = frame.loc[sample]
    counts = sample_frame.groupby('run_id').size()
    weights = sample_frame.run_id.map(1 / counts).to_numpy() * len(counts)
    sample_frame[['run_id', 'issue_s']].to_csv(out / 'sample_manifest.csv', index=False)
    power_train = np.vstack([history_features(h) for h in sample_frame.history])
    power_test = np.vstack([history_features(h) for h in test.history])
    static_train = sample_frame[['request_rate']].to_numpy(float)
    static_test = test[['request_rate']].to_numpy(float)
    dynamic_train = sample_frame[list(DYNAMIC)].to_numpy(float)
    dynamic_test = test[list(DYNAMIC)].to_numpy(float)
    features = {
        'history_only': (power_train, power_test),
        'history_static_rate': (np.column_stack([power_train, static_train]), np.column_stack([power_test, static_test])),
        'history_dynamic': (np.column_stack([power_train, dynamic_train]), np.column_stack([power_test, dynamic_test])),
        'history_static_dynamic': (np.column_stack([power_train, static_train, dynamic_train]),
                                   np.column_stack([power_test, static_test, dynamic_test])),
    }
    cell_dir = out / 'cells'
    cell_dir.mkdir(exist_ok=True)
    settings = []
    for condition in CONDITIONS:
        x_train, x_test = features[condition]
        for q in QUANTILES:
            cell = cell_dir / f'{condition}_q{q:g}.npy'
            record = cell_dir / f'{condition}_q{q:g}.json'
            if cell.exists() and record.exists():
                values = np.load(cell)
                if len(values) == len(test):
                    settings.append(json.loads(record.read_text()))
                    continue
            model = lgb.LGBMRegressor(objective='quantile', alpha=float(q), n_estimators=100,
                num_leaves=15, learning_rate=.07, min_child_samples=30,
                random_state=42, n_jobs=1, verbosity=-1)
            model.fit(x_train, sample_frame.y.to_numpy(), sample_weight=weights)
            prediction = model.predict(x_test)
            np.save(cell, prediction)
            fitted = dict(condition=condition, quantile=float(q), fitted_trees=model.booster_.num_trees(),
                objective='quantile', n_estimators=100, num_leaves=15,
                learning_rate=.07, min_child_samples=30, model_seed=42,
                n_train_rows=len(sample), n_test_rows=len(test))
            record.write_text(json.dumps(fitted, indent=2) + '\n')
            settings.append(fitted)
            print(condition, q, flush=True)
    pd.DataFrame(settings).to_csv(out / 'selected_hyperparameters.csv', index=False)
    results = []
    for condition in CONDITIONS:
        raw = np.column_stack([np.load(cell_dir / f'{condition}_q{q:g}.npy') for q in QUANTILES])
        crossing = np.any(np.diff(raw, axis=1) < 0, axis=1)
        ordered = np.maximum.accumulate(raw, axis=1)
        result = test[['run_id', 'original_id', 'issue_s', 'target_start_s', 'target_end_s',
            'runtime_available_s', 'log_age_s', 'y', 'delta_w']].copy()
        result['condition'] = condition
        result['crps_w'] = crps_quantile(result.y.to_numpy(), ordered)
        result['q50'] = ordered[:, 7]
        result['absolute_error_w'] = abs(result.q50 - result.y)
        result['covered_80'] = result.y.between(ordered[:, 3], ordered[:, 11])
        result['crossing_before_correction'] = crossing
        for i, q in enumerate(QUANTILES):
            result[f'q_{q:g}'] = ordered[:, i]
        results.append(result)
    pred = pd.concat(results, ignore_index=True)
    pred.to_parquet(out / 'per_target_predictions.parquet', index=False)
    per = pred.groupby(['run_id', 'condition']).agg(n_targets=('y', 'size'),
        crps_w=('crps_w', 'mean'), mae_w=('absolute_error_w', 'mean'),
        p95_w=('absolute_error_w', lambda x: x.quantile(.95)),
        coverage_80=('covered_80', 'mean'), crossing_fraction=('crossing_before_correction', 'mean')).reset_index()
    per.to_csv(out / 'per_run_metrics.csv', index=False)
    per.groupby('condition', as_index=False)[['crps_w', 'mae_w', 'p95_w', 'coverage_80',
        'crossing_fraction']].mean().to_csv(out / 'aggregate_metrics.csv', index=False)
    pivot = per.pivot(index='run_id', columns='condition', values='crps_w')
    gains = []
    for condition in CONDITIONS[1:]:
        gains.append(dict(condition=condition, n_runs=len(pivot),
            history_crps_w=pivot.history_only.mean(), metadata_crps_w=pivot[condition].mean(),
            descriptive_gain_pct=100*(1-pivot[condition].mean()/pivot.history_only.mean()),
            n_improved_runs=int((pivot[condition] < pivot.history_only).sum()),
            independence='all 200 runs belong to one acquisition; no independent-session CI'))
    pd.DataFrame(gains).to_csv(out / 'paired_gain_descriptive.csv', index=False)
    checks = dict(n_test_runs=test.run_id.nunique(), n_train_runs=train.run_id.nunique(),
        same_rows=bool(pred.groupby(['run_id', 'issue_s']).condition.nunique().eq(4).all()),
        same_truth=bool(pred.groupby(['run_id', 'issue_s']).y.nunique().eq(1).all()),
        no_future_runtime=bool((pred.runtime_available_s <= pred.issue_s + 1e-9).all()),
        finite_scores=bool(np.isfinite(pred.crps_w).all()),
        one_acquisition_group=bool(pd.read_csv(RUNTIME / 'runtime_coverage.csv').group_id.nunique() == 1))
    checks['all_passed'] = all([checks['n_test_runs']==50, checks['n_train_runs']==150,
        checks['same_rows'], checks['same_truth'], checks['no_future_runtime'],
        checks['finite_scores'], checks['one_acquisition_group']])
    (out / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    if not checks['all_passed']:
        raise AssertionError(checks)
    print(pd.read_csv(out / 'aggregate_metrics.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE / 'results/online_runtime_pilot_20260927')
    run(parser.parse_args().out)
