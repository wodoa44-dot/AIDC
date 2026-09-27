#!/usr/bin/env python
"""Corrected production driver: cached features, resumable A/B/C and reused holdout."""
import argparse
import json
import os
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from corrected_capacity import (HERE, ROOT, continuous_segments, bin_segment, samples,
                                choose_per_run, sample_weights, fit_quantiles, pred, run_balanced)
from stage_b import candidates
from stage_c import (Tasks, SafeBoundaryStop, digest, file_hash, fit_encoder, transform,
                     outer_groups, selection_provenance, run_stage_c, refit_pair, CONDITIONS)


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    os.replace(tmp, path)


def initialize(out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    config = out / 'effective_config.json'
    if config.exists() and json.loads(config.read_text()) != cfg:
        raise ValueError('Incompatible effective configuration; use a new experiment directory')
    for name in ['features', 'models', 'predictions', 'splits', 'checkpoints']:
        (out / name).mkdir(exist_ok=True)
    atomic_json(config, cfg)
    inv = pd.read_csv(HERE / 'outputs/expanded_main_cpu_bounded_20260915/input_inventory.csv')
    assert len(inv) == 32 and inv.run_id.nunique() == 32
    if cfg.get('development_integration'):
        inv = inv.loc[~inv.reserved_confirmation].copy()
    manifest = out / 'input_manifest.csv'
    if manifest.exists():
        pd.testing.assert_frame_equal(pd.read_csv(manifest), inv.reset_index(drop=True))
    else:
        inv.to_csv(manifest, index=False)
    dev = inv.loc[~inv.reserved_confirmation]
    dev.to_csv(out / 'development_runs.csv', index=False)
    inv.loc[inv.reserved_confirmation].to_csv(out / 'reused_holdout_runs.csv', index=False)
    folds = []
    for o in range(cfg['outer_folds']):
        test = [g.iloc[o % len(g)].run_id for _, g in dev.groupby('cohort', sort=True)]
        tr = dev.loc[~dev.run_id.isin(test)]
        for run in dev.run_id:
            folds.append(dict(outer_fold=o+1, inner_fold=np.nan, run_id=run,
                              role='outer_test' if run in test else 'outer_train'))
        for i in range(cfg['inner_folds']):
            val = [g.iloc[i % len(g)].run_id for _, g in tr.groupby('cohort', sort=True)]
            for run in tr.run_id:
                folds.append(dict(outer_fold=o+1, inner_fold=i+1, run_id=run,
                                  role='inner_val' if run in val else 'inner_train'))
    foldpath = out / 'splits/folds.csv'
    frame = pd.DataFrame(folds)
    if foldpath.exists():
        pd.testing.assert_frame_equal(pd.read_csv(foldpath), frame)
    else:
        frame.to_csv(foldpath, index=False)
    grid = pd.DataFrame([dict(resolution_ms=r, horizon_ms=h, included=h>=r and h%r==0)
                         for r in cfg['resolutions_ms'] for h in cfg['horizons_ms']])
    grid.to_csv(out / 'grid_manifest.csv', index=False)
    return inv, frame, grid.query('included')


def prepare(out, cfg, inv, grid):
    raw = None
    for cell in grid.itertuples():
        rr, hh = int(cell.resolution_ms), int(cell.horizon_ms)
        mp = out / 'features' / f'meta_{rr}_{hh}.parquet'
        fp = out / 'features' / f'power_{rr}_{hh}.npz'
        cp = out / 'features' / f'cache_{rr}_{hh}.json'
        signature = digest({'config': cfg, 'manifest': file_hash(out / 'input_manifest.csv')})
        if cp.exists():
            c = json.loads(cp.read_text())
            if c['signature'] != signature:
                raise ValueError('Incompatible feature cache')
            if mp.exists() and fp.exists() and c['meta'] == file_hash(mp) and c['power'] == file_hash(fp):
                continue
        if raw is None:
            raw = {}
            for r in inv.itertuples():
                d = pd.read_parquet(ROOT / 'trace/preprocessing/outputs/prepared' / f'{r.run_id}.parquet')
                raw[r.run_id] = (r, d, continuous_segments(d, float(np.median(np.diff(d.time_s))*5)))
        rows = []
        for run, (r, d, segments) in raw.items():
            c1 = d.time_s.min() + .6*(d.time_s.max()-d.time_s.min())
            c2 = d.time_s.min() + .8*(d.time_s.max()-d.time_s.min())
            for sid, segment in segments:
                z = bin_segment(segment, rr, coverage=cfg['coverage'])
                for subid, zz in z.groupby('subsegment'):
                    got = samples(zz, rr, hh, cfg['histories_ms'], c1, c2, cfg['evaluation_cadence_ms'])
                    if got:
                        m, f = got
                        m['run_id'], m['cohort'] = run, r.cohort
                        m['segment_id'], m['subsegment_id'] = sid, subid
                        for col in ['gpu', 'workload', 'batch', 'ai_model']:
                            m[col] = getattr(r, col)
                        rows.append((m, f))
        meta = pd.concat([x[0] for x in rows], ignore_index=True)
        fs = {str(h): np.concatenate([x[1][h] for x in rows]) for h in cfg['histories_ms'] if h>=rr and h%rr==0}
        meta.to_parquet(mp, index=False)
        np.savez_compressed(fp, **fs)
        atomic_json(cp, {'signature': signature, 'meta': file_hash(mp), 'power': file_hash(fp)})
        print(f'features ready {rr}/{hh}: {len(meta)} targets', flush=True)


def run_a(out, cfg, tasks, folds, grid):
    records = []
    base = dict(learning_rate=cfg['learning_rate'], num_leaves=cfg['num_leaves'],
                min_child_samples=cfg['min_child_samples'], reg_alpha=0., reg_lambda=0., n_jobs=cfg['n_jobs'],
                n_estimators=max(cfg['boosting_rounds']))
    for cell in grid.itertuples():
        rr, hh = int(cell.resolution_ms), int(cell.horizon_ms)
        meta = pd.read_parquet(out / 'features' / f'meta_{rr}_{hh}.parquet')
        with np.load(out / 'features' / f'power_{rr}_{hh}.npz') as fs:
            hist = '5000' if '5000' in fs.files else max(fs.files, key=int)
            for o in range(1, cfg['outer_folds']+1):
                for i in range(1, cfg['inner_folds']+1):
                    sub = folds.loc[(folds.outer_fold==o) & (folds.inner_fold==i)]
                    tr = np.flatnonzero(meta.run_id.isin(sub.loc[sub.role=='inner_train', 'run_id']))
                    va = np.flatnonzero(meta.run_id.isin(sub.loc[sub.role=='inner_val', 'run_id']))
                    for budget in cfg['budgets_per_run']:
                        ti = choose_per_run(meta, tr, budget)
                        weight = sample_weights(meta, ti)
                        for condition in CONDITIONS:
                            identity = dict(stage='A', resolution_ms=rr, horizon_ms=hh, outer_fold=o,
                                            inner_fold=i, condition=condition, candidate=str(budget), seed=17)
                            def compute(directory):
                                enc = fit_encoder(meta, tr, condition)
                                x = transform(fs[hist], meta, enc)
                                models = fit_quantiles(x, meta.y.to_numpy(), ti, weight, cfg['quantiles'], base, 17)
                                joblib.dump({'models': models, 'encoder': enc, 'parameters': base}, directory/'model.joblib', compress=3)
                                np.savez_compressed(directory/'indices_weights.npz', train=ti, val=va, weight=weight)
                                return [identity | dict(budget_per_run=budget, rounds=n, history_ms=int(hist),
                                         run_balanced_crps_w=run_balanced(meta, va, meta.y.to_numpy(), pred(models, x, va, n), cfg['quantiles']),
                                         n_train=len(ti), n_val=len(va)) for n in cfg['boosting_rounds']]
                            records.extend(tasks.run(identity, compute))
    pd.DataFrame(records).to_csv(out/'learning_curves.csv', index=False)


def run_b(out, cfg, tasks, folds, grid, final=False):
    plan = candidates(cfg)
    if cfg.get('development_integration'):
        plan = plan[:2]
    pd.DataFrame(plan).to_csv(out/'stage_b_candidate_plan.csv', index=False)
    selections, scores = [], []
    development = sorted(pd.read_csv(out/'development_runs.csv').run_id)
    for cell in grid.itertuples():
        rr, hh = int(cell.resolution_ms), int(cell.horizon_ms)
        meta = pd.read_parquet(out/'features'/f'meta_{rr}_{hh}.parquet')
        with np.load(out/'features'/f'power_{rr}_{hh}.npz') as fs:
            for fold in ([0] if final else range(1, cfg['outer_folds']+1)):
                train_groups = development if final else outer_groups(out, fold)['outer_train']
                for condition in CONDITIONS:
                    options = []
                    for cid, pa in enumerate(plan):
                        if str(pa['history_ms']) not in fs.files:
                            continue
                        val_runs = []
                        for inner in range(1, cfg['inner_folds']+1):
                            if final:
                                dev = pd.read_csv(out/'development_runs.csv')
                                val = [g.iloc[(inner-1)%len(g)].run_id for _, g in dev.groupby('cohort', sort=True)]
                                trgroups = sorted(set(development)-set(val))
                            else:
                                subset = folds.loc[(folds.outer_fold==fold) & (folds.inner_fold==inner)]
                                val = subset.loc[subset.role=='inner_val', 'run_id'].tolist()
                                trgroups = subset.loc[subset.role=='inner_train', 'run_id'].tolist()
                            tr = np.flatnonzero(meta.run_id.isin(trgroups))
                            va = np.flatnonzero(meta.run_id.isin(val))
                            assert set(trgroups).issubset(train_groups) and set(val).issubset(train_groups)
                            assert not set(trgroups) & set(val)
                            identity = dict(stage='B_final' if final else 'B', resolution_ms=rr, horizon_ms=hh,
                                            outer_fold=fold, inner_fold=inner, condition=condition, candidate=cid, seed=17,
                                            parameters=pa, training_groups=sorted(trgroups), validation_groups=sorted(val))
                            def compute(directory):
                                ti = choose_per_run(meta, tr, pa['budget_per_run'])
                                weight = sample_weights(meta, ti)
                                enc = fit_encoder(meta, tr, condition)
                                x = transform(fs[str(pa['history_ms'])], meta, enc)
                                models = fit_quantiles(x, meta.y.to_numpy(), ti, weight, cfg['quantiles'], pa, 17)
                                pp = pred(models, x, va)
                                from corrected_capacity import crps
                                values = pd.DataFrame({'run_id': meta.iloc[va].run_id.to_numpy(),
                                                       'crps': crps(meta.y.to_numpy()[va], pp, cfg['quantiles'])})
                                perrun = values.groupby('run_id').crps.mean().to_dict()
                                np.savez_compressed(directory/'indices_weights.npz', train=ti, val=va, weight=weight)
                                return {'run_scores': perrun, 'identity': identity}
                            result = tasks.run(identity, compute)
                            for run, value in result['run_scores'].items():
                                scores.append(identity | {'run_id': run, 'crps_w': value})
                                val_runs.append({'run_id': run, 'crps_w': value})
                        # Each independent run has equal weight, including repeated validation membership.
                        score = float(pd.DataFrame(val_runs).groupby('run_id').crps_w.mean().mean())
                        options.append({'candidate_id': cid, 'parameters': pa, 'inner_run_balanced_crps_w': score})
                    winner = min(options, key=lambda v: (v['inner_run_balanced_crps_w'], v['candidate_id']))
                    winner['provenance'] = selection_provenance(out, cfg, rr, hh, fold, condition, train_groups)
                    selections.append(winner)
    stem = 'final_development' if final else 'stage_b'
    atomic_json(out/f'{stem}_selections.json', selections)
    pd.DataFrame(scores).to_csv(out/f'{stem}_candidate_scores.csv', index=False)
    return selections


def execute(out, cfg, stop_after=None):
    inv, folds, grid = initialize(out, cfg)
    atomic_json(out/'status.json', {'status': 'running', 'started_utc': time.strftime('%FT%TZ', time.gmtime())})
    prepare(out, cfg, inv, grid)
    tasks = Tasks(out, cfg, stop_after)
    run_a(out, cfg, tasks, folds, grid)
    selected = run_b(out, cfg, tasks, folds, grid)
    # Load the persisted, frozen selection records before final fitting.
    selected = json.loads((out/'stage_b_selections.json').read_text())
    results = run_stage_c(out, cfg, tasks, selected)
    from corrected_report import report
    report(out, cfg, results)
    if not cfg.get('development_integration'):
        run_b(out, cfg, tasks, folds, grid, final=True)
        chosen = json.loads((out/'final_development_selections.json').read_text())
        atomic_json(out/'holdout_selection_freeze.json', {'selection_hash': file_hash(out/'final_development_selections.json'),
                    'label': 'reused-holdout assessment', 'frozen_utc': time.strftime('%FT%TZ', time.gmtime())})
        dev = sorted(inv.loc[~inv.reserved_confirmation, 'run_id'])
        holdout = sorted(inv.loc[inv.reserved_confirmation, 'run_id'])
        assert len(holdout) == 4
        for cell in grid.itertuples():
            rr, hh = int(cell.resolution_ms), int(cell.horizon_ms)
            meta = pd.read_parquet(out/'features'/f'meta_{rr}_{hh}.parquet')
            sel = {s['provenance']['condition']: s for s in chosen
                   if (s['provenance']['resolution_ms'], s['provenance']['horizon_ms'])==(rr, hh)}
            with np.load(out/'features'/f'power_{rr}_{hh}.npz') as fs:
                results.extend(refit_pair(out, cfg, tasks, meta, fs, rr, hh, 0, sel, dev, holdout, 'reused_holdout'))
        report(out, cfg, results)
    atomic_json(out/'status.json', {'status': 'completed', 'completed_tasks_this_invocation': tasks.completed,
                                   'skipped_valid_tasks': tasks.skipped, 'development_only': cfg.get('development_integration', False)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE/'configs/corrected_capacity_20260915.yaml'))
    ap.add_argument('--experiment-id', required=True)
    ap.add_argument('--development-integration', action='store_true')
    ap.add_argument('--stop-after', type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    # Directory names do not change mathematical task identities.
    cfg.pop('experiment_id', None)
    if args.development_integration:
        cfg.update(development_integration=True, resolutions_ms=[200], horizons_ms=[1000],
                   outer_folds=1, inner_folds=2, budgets_per_run=[250], boosting_rounds=[12])
    elif args.stop_after:
        raise ValueError('Intentional interruption is development-only')
    out = HERE/'outputs'/args.experiment_id
    try:
        execute(out, cfg, args.stop_after)
    except SafeBoundaryStop as exc:
        atomic_json(out/'status.json', {'status': 'intentionally_interrupted', 'message': str(exc)})
        print(str(exc), flush=True)
        raise SystemExit(75)
    except Exception as exc:
        if out.exists():
            atomic_json(out/'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
