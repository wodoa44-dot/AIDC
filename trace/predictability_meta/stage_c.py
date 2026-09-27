"""Verified final refits and configuration-aware atomic tasks.

Legacy reference uses pooled, evenly spaced chronological row positions (250
maximum), equal row weights, 12 rounds, and history chosen by history-only inner
validation. It is a newly fitted budget reference, not a historical score.
"""
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from corrected_capacity import choose_per_run, sample_weights, fit_quantiles, pred
from expanded import QCOL

VERSION = 1
CONDITIONS = ['history_only', 'history_all_meta']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class SafeBoundaryStop(RuntimeError):
    pass


class Tasks:
    def __init__(self, out, cfg, stop_after=None):
        self.out = Path(out)
        self.root = self.out / 'tasks'
        self.root.mkdir(parents=True, exist_ok=True)
        self.signature = digest({'version': VERSION, 'config': cfg,
                                 'inputs': file_hash(self.out / 'input_manifest.csv'),
                                 'folds': file_hash(self.out / 'splits/folds.csv')})
        manifest = self.root / 'identity.json'
        if manifest.exists() and json.loads(manifest.read_text())['signature'] != self.signature:
            raise ValueError('Incompatible checkpoint configuration/input/fold manifest')
        manifest.write_text(json.dumps({'signature': self.signature}))
        self.stop_after = stop_after
        self.completed = 0
        self.skipped = 0

    def run(self, identity, compute):
        spec = {'signature': self.signature, **identity}
        key = digest(spec)
        directory = self.root / key
        directory.mkdir(exist_ok=True)
        done = directory / 'complete.json'
        if done.exists():
            record = json.loads(done.read_text())
            if record['identity'] != spec:
                raise ValueError(f'Incompatible checkpoint: {done}')
            valid = all((directory / name).is_file() and file_hash(directory / name) == checksum
                        for name, checksum in record['files'].items())
            if valid and 'result.joblib' in record['files']:
                result = joblib.load(directory / 'result.joblib')
                self.skipped += 1
                return result
            done.unlink()  # damaged/incomplete results are recomputed, never skipped
        result = compute(directory)
        joblib.dump(result, directory / 'result.joblib', compress=3)
        files = {p.name: file_hash(p) for p in directory.iterdir()
                 if p.is_file() and p.name not in ['complete.json', 'complete.tmp']}
        temp = directory / 'complete.tmp'
        temp.write_text(json.dumps({'identity': spec, 'files': files}, sort_keys=True))
        os.replace(temp, done)
        self.completed += 1
        with (self.out / 'task_events.jsonl').open('a') as f:
            f.write(json.dumps({'event': 'completed', 'identity': identity, 'key': key}) + '\n')
        print(f'completed {identity}', flush=True)
        if self.stop_after and self.completed >= self.stop_after:
            raise SafeBoundaryStop('Intentional development interruption at committed task boundary')
        return result


def fit_encoder(meta, train, condition):
    groups = [] if condition == 'history_only' else ['gpu', 'workload', 'batch', 'ai_model']
    enc = {'groups': groups, 'vocab': {}, 'batch_median': None, 'names': []}
    for g in groups:
        if g == 'batch':
            vals = pd.to_numeric(meta.iloc[train][g], errors='coerce')
            enc['batch_median'] = float(vals.median()) if vals.notna().any() else 0.0
            enc['names'].extend(['batch', 'batch_missing'])
        else:
            enc['vocab'][g] = sorted(meta.iloc[train][g].fillna('__MISSING__').astype(str).unique())
            enc['names'].extend([g + '=' + v for v in enc['vocab'][g]] + [g + '=__UNKNOWN__'])
    return enc


def transform(power, meta, enc):
    cols = [power]
    for g in enc['groups']:
        if g == 'batch':
            v = pd.to_numeric(meta[g], errors='coerce')
            cols.extend([v.fillna(enc['batch_median']).to_numpy()[:, None], v.isna().to_numpy()[:, None]])
        else:
            v = meta[g].fillna('__MISSING__').astype(str)
            cols.extend([(v == item).to_numpy()[:, None] for item in enc['vocab'][g]])
            cols.append((~v.isin(enc['vocab'][g])).to_numpy()[:, None])
    return np.column_stack(cols).astype(np.float32)


def outer_groups(out, fold):
    folds = pd.read_csv(Path(out) / 'splits/folds.csv')
    return {role: sorted(folds.loc[(folds.outer_fold == fold) & (folds.role == role), 'run_id'])
            for role in ['outer_train', 'outer_test']}


def selection_provenance(out, cfg, rr, hh, fold, condition, train_groups):
    return {'configuration': digest(cfg), 'resolution_ms': rr, 'horizon_ms': hh,
            'outer_fold': fold, 'condition': condition, 'outer_training_groups': sorted(train_groups),
            'fold_manifest': file_hash(Path(out) / 'splits/folds.csv'),
            'selection_source': 'grouped_inner_validation_only'}


def refit_pair(out, cfg, tasks, meta, fs, rr, hh, fold, selections, train_groups, test_groups,
               assessment='grouped_outer'):
    train = np.flatnonzero(meta.run_id.isin(train_groups))
    test = np.flatnonzero(meta.run_id.isin(test_groups))
    assert len(train) and len(test) and not set(train_groups) & set(test_groups)
    for condition in CONDITIONS:
        expected = selection_provenance(out, cfg, rr, hh, fold, condition, train_groups)
        if selections[condition]['provenance'] != expected:
            raise ValueError(f'Selection provenance mismatch: {rr}/{hh}/{fold}/{condition}')
    results = []
    for pair in ['legacy_budget', 'improved_controlled', 'independently_tuned']:
        reference = None
        for condition in CONDITIONS:
            selected = selections[condition if pair == 'independently_tuned' else 'history_only']
            pa = dict(selected['parameters'])
            if pair == 'legacy_budget':
                pa.update(n_estimators=12, num_leaves=15, learning_rate=.07,
                          min_child_samples=30, reg_alpha=0., reg_lambda=0., budget_per_run='legacy_pooled_250')
                ti = train[np.linspace(0, len(train)-1, min(250, len(train)), dtype=int)]
                weight = np.ones(len(ti))
            else:
                ti = choose_per_run(meta, train, pa['budget_per_run'])
                weight = sample_weights(meta, ti)
            power = fs[str(pa['history_ms'])]
            if pair != 'independently_tuned':
                current = (ti, weight, pa['history_ms'])
                if reference is not None:
                    assert np.array_equal(reference[0], ti) and np.array_equal(reference[1], weight)
                    assert reference[2] == pa['history_ms']
                reference = current
            for seed in cfg['final_seeds']:
                identity = dict(stage='C', assessment=assessment, resolution_ms=rr, horizon_ms=hh,
                                outer_fold=fold, pair=pair, condition=condition,
                                candidate=selected['candidate_id'], seed=seed,
                                parameters=pa, selection=digest(selected))
                def compute(directory):
                    # Learn preprocessing afresh from all eligible training groups, never test groups.
                    enc = fit_encoder(meta, train, condition)
                    x = transform(power, meta, enc)
                    assert np.array_equal(x[:, :power.shape[1]], power, equal_nan=True)
                    models = fit_quantiles(x, meta.y.to_numpy(), ti, weight, cfg['quantiles'], pa, seed)
                    pp = pred(models, x, test)
                    bundle = {'models': models, 'encoder': enc, 'parameters': pa,
                              'quantiles': cfg['quantiles'], 'power_columns': power.shape[1]}
                    joblib.dump(bundle, directory / 'model.joblib', compress=3)
                    np.savez_compressed(directory / 'indices_weights.npz', train=ti, eligible_train=train,
                                        test=test, weights=weight)
                    (directory / 'parameters.json').write_text(json.dumps({
                        'requested': pa, 'resolved': [m.get_params() for m in models],
                        'fitted': [{'trees': m.booster_.num_trees(), 'parameters': m.booster_.params}
                                   for m in models], 'encoder': enc,
                        'provenance': selected['provenance']}, default=str, indent=2))
                    frame = meta.iloc[test].copy()
                    frame['target_index'] = test
                    for key, value in identity.items():
                        if key not in ['parameters', 'selection']:
                            frame[key] = value
                    for j, name in enumerate(QCOL(cfg['quantiles'])):
                        frame[name] = pp[:, j]
                    frame.to_parquet(directory / 'predictions.parquet', index=False)
                    names = [f'power_{i}' for i in range(power.shape[1])] + enc['names']
                    usage = [{'feature': name, 'splits': int(count), 'gain': float(gain), 'quantile': q}
                             for m, q in zip(models, cfg['quantiles'])
                             for name, count, gain in zip(names, m.booster_.feature_importance('split'),
                                                         m.booster_.feature_importance('gain'))]
                    pd.DataFrame(usage).to_csv(directory / 'split_usage.csv', index=False)
                    # Persisted bundle must reproduce predictions, including its stored imputation.
                    loaded = joblib.load(directory / 'model.joblib')
                    again = pred(loaded['models'], transform(power, meta, loaded['encoder']), test)
                    np.testing.assert_allclose(again, pp, atol=1e-10, rtol=1e-10)
                    return {'prediction': str(directory / 'predictions.parquet'),
                            'model': str(directory / 'model.joblib'), 'usage': str(directory / 'split_usage.csv'),
                            'identity': identity}
                results.append(tasks.run(identity, compute))
    return results


def run_stage_c(out, cfg, tasks, selections):
    results = []
    for rr, hh in sorted({(s['provenance']['resolution_ms'], s['provenance']['horizon_ms']) for s in selections}):
        meta = pd.read_parquet(Path(out) / 'features' / f'meta_{rr}_{hh}.parquet')
        with np.load(Path(out) / 'features' / f'power_{rr}_{hh}.npz') as fs:
            for fold in range(1, cfg['outer_folds'] + 1):
                groups = outer_groups(out, fold)
                chosen = {s['provenance']['condition']: s for s in selections
                          if (s['provenance']['resolution_ms'], s['provenance']['horizon_ms'], s['provenance']['outer_fold']) == (rr, hh, fold)}
                results.extend(refit_pair(out, cfg, tasks, meta, fs, rr, hh, fold, chosen,
                                          groups['outer_train'], groups['outer_test']))
    return results
