#!/usr/bin/env python
"""Corrected-capacity study runner (resumable; no hidden model-budget overrides)."""
import argparse, json, os, sys, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from expanded import parse, X, QCOL
sys.path.insert(0, str(ROOT / 'trace/predictability/src'))
from core import continuous_segments, bin_segment, samples, crps, rearrange

def choose_per_run(meta, ids, cap):
    """Deterministic nested temporal samples; every run contributes up to cap."""
    chosen = []
    for _, g in meta.iloc[ids].groupby('run_id', sort=True):
        ii = g.index.to_numpy()
        n = len(ii) if cap == 'all' else min(len(ii), int(cap))
        chosen.extend(ii[np.linspace(0, len(ii) - 1, n, dtype=int)])
    return np.asarray(sorted(chosen), dtype=int)

def sample_weights(meta, ids):
    counts = meta.iloc[ids].groupby('run_id').size()
    return meta.iloc[ids].run_id.map(1.0 / counts).to_numpy() * len(counts)

def fit_quantiles(x, y, ids, w, quantiles, p, seed):
    models = []
    for q in quantiles:
        m = LGBMRegressor(objective='quantile', alpha=float(q), n_estimators=int(p['n_estimators']),
            learning_rate=float(p['learning_rate']), num_leaves=int(p['num_leaves']),
            min_child_samples=int(p['min_child_samples']), reg_alpha=float(p['reg_alpha']),
            reg_lambda=float(p['reg_lambda']), random_state=seed, n_jobs=int(p['n_jobs']), verbosity=-1)
        assert m.n_estimators == int(p['n_estimators'])
        m.fit(x[ids], y[ids], sample_weight=w)
        models.append(m)
    return models

def pred(models, x, ids, rounds=None):
    z = np.column_stack([m.predict(x[ids], num_iteration=rounds) for m in models])
    return rearrange(z)[0]

def run_balanced(meta, ids, y, p, q):
    s = pd.DataFrame({'run_id': meta.iloc[ids].run_id.to_numpy(), 'crps': crps(y[ids], p, q)})
    return float(s.groupby('run_id').crps.mean().mean())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default=str(HERE/'configs/corrected_capacity_20260915.yaml')); ap.add_argument('--experiment-id'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
    if a.smoke:
        c.update(experiment_id='corrected_smoke_20260915', resolutions_ms=[200], horizons_ms=[1000], outer_folds=1, inner_folds=2, budgets_per_run=[250], boosting_rounds=[12], n_estimators=12, final_seeds=[17])
    if a.experiment_id: c['experiment_id'] = a.experiment_id
    out = HERE/'outputs'/c['experiment_id']; out.mkdir(parents=True, exist_ok=False)
    for d in ['checkpoints','models','predictions','splits','features']: (out/d).mkdir()
    (out/'effective_config.json').write_text(json.dumps(c, indent=2))
    started_monotonic = time.monotonic()
    status = {'status':'running','started_utc':time.strftime('%FT%TZ',time.gmtime())}; (out/'status.json').write_text(json.dumps(status,indent=2))
    def checkpoint(name, rows):
        tmp = out/'checkpoints'/(name+'.tmp'); final = out/'checkpoints'/(name+'.csv'); pd.DataFrame(rows).to_csv(tmp,index=False); os.replace(tmp,final)
    try:
        inv = pd.read_csv(HERE/'outputs/expanded_main_cpu_bounded_20260915/input_inventory.csv')
        assert len(inv) == 32 and inv.run_id.nunique() == 32
        dev = inv[~inv.reserved_confirmation].copy(); reused = inv[inv.reserved_confirmation].copy()
        inv.to_csv(out/'input_manifest.csv',index=False); dev[['run_id','cohort']].to_csv(out/'development_runs.csv',index=False); reused[['run_id','cohort']].to_csv(out/'reused_holdout_runs.csv',index=False)
        folds=[]
        for o in range(c['outer_folds']):
            test=[]
            for _,g in dev.groupby('cohort',sort=True): test.append(g.iloc[o % len(g)].run_id)
            trdev=dev[~dev.run_id.isin(test)]
            for r in dev.run_id: folds.append({'outer_fold':o+1,'inner_fold':np.nan,'run_id':r,'role':'outer_test' if r in test else 'outer_train'})
            for i in range(c['inner_folds']):
                val=[]
                for _,g in trdev.groupby('cohort',sort=True): val.append(g.iloc[i % len(g)].run_id)
                for r in trdev.run_id: folds.append({'outer_fold':o+1,'inner_fold':i+1,'run_id':r,'role':'inner_val' if r in val else 'inner_train'})
        foldf=pd.DataFrame(folds); foldf.to_csv(out/'splits/folds.csv',index=False)
        raw={}
        for r in inv.itertuples():
            d=pd.read_parquet(ROOT/'trace/preprocessing/outputs/prepared'/f'{r.run_id}.parquet')
            raw[r.run_id]=(r,d,continuous_segments(d,float(np.median(np.diff(d.time_s))*5)))
        q=np.array(c['quantiles']); curves=[]; grid=[]
        basep={'n_estimators':c['n_estimators'],'learning_rate':c['learning_rate'],'num_leaves':c['num_leaves'],'min_child_samples':c['min_child_samples'],'reg_alpha':0.,'reg_lambda':0.,'n_jobs':c['n_jobs']}
        for rr in c['resolutions_ms']:
            for hh in c['horizons_ms']:
                ok=hh>=rr and hh%rr==0; grid.append({'resolution_ms':rr,'horizon_ms':hh,'included':ok})
                if not ok: continue
                rows=[]
                for run,(r,d,segs) in raw.items():
                    c1=d.time_s.min()+.6*(d.time_s.max()-d.time_s.min()); c2=d.time_s.min()+.8*(d.time_s.max()-d.time_s.min())
                    for _,s in segs:
                        z=bin_segment(s,rr,coverage=c['coverage'])
                        for _,zz in z.groupby('subsegment'):
                            got=samples(zz,rr,hh,c['histories_ms'],c1,c2,c['evaluation_cadence_ms'])
                            if got:
                                m,f=got; m['run_id']=run; m['cohort']=r.cohort
                                for col in ['gpu','workload','batch','ai_model']: m[col]=getattr(r,col) if hasattr(r,col) else (r.workload if col=='workload' else np.nan)
                                rows.append((m,f))
                meta=pd.concat([x[0] for x in rows],ignore_index=True); fs={h:np.concatenate([x[1][h] for x in rows]) for h in c['histories_ms'] if h>=rr and h%rr==0}; hist=5000 if 5000 in fs else max(fs); y=meta.y.to_numpy()
                meta.to_parquet(out/'features'/f'meta_{rr}_{hh}.parquet',index=False); np.savez_compressed(out/'features'/f'power_{rr}_{hh}.npz',**{str(k):v for k,v in fs.items()})
                for o in range(c['outer_folds']):
                    test=set(foldf.query('outer_fold==@o+1 and role=="outer_test"').run_id)
                    for i in range(c['inner_folds']):
                        val=set(foldf.query('outer_fold==@o+1 and inner_fold==@i+1 and role=="inner_val"').run_id)
                        tr=np.flatnonzero(~meta.run_id.isin(test|val|set(reused.run_id))); va=np.flatnonzero(meta.run_id.isin(val))
                        assert not set(meta.iloc[tr].run_id)&set(meta.iloc[va].run_id)&set(meta.iloc[np.flatnonzero(meta.run_id.isin(test))].run_id)
                        for budget in c['budgets_per_run']:
                            ti=choose_per_run(meta,tr,budget); w=sample_weights(meta,ti)
                            assert np.allclose(pd.DataFrame({'r':meta.iloc[ti].run_id,'w':w}).groupby('r').w.sum().to_numpy(),len(np.unique(meta.iloc[ti].run_id)))
                            np.savez_compressed(out/'splits'/f'r{rr}_h{hh}_o{o+1}_i{i+1}_b{budget}.npz',train=ti,val=va,weight=w)
                            for cond,groups in [('history_only',[]),('history_all_meta',['gpu','workload','batch','ai_model'])]:
                                x,names,vocab=X(fs[hist],meta,ti,groups); assert np.array_equal(x[:,:fs[hist].shape[1]],fs[hist])
                                (out/'features'/f'schema_r{rr}_h{hh}_o{o+1}_i{i+1}_{cond}.json').write_text(json.dumps({'history_ms':hist,'metadata_columns':names,'vocabulary':vocab}))
                                p=basep|{'n_estimators':300}; models=fit_quantiles(x,y,ti,w,q,p,17)
                                joblib.dump(models,out/'models'/f'curve_r{rr}_h{hh}_o{o+1}_i{i+1}_b{budget}_{cond}.joblib',compress=3)
                                for rounds in c['boosting_rounds']:
                                    pp=pred(models,x,va,rounds); curves.append({'resolution_ms':rr,'horizon_ms':hh,'outer_fold':o+1,'inner_fold':i+1,'condition':cond,'budget_per_run':budget,'rounds':rounds,'history_ms':hist,'run_balanced_crps_w':run_balanced(meta,va,y,pp,q),'n_train':len(ti),'n_val':len(va)})
                checkpoint(f'cell_r{rr}_h{hh}',curves)
        pd.DataFrame(grid).to_csv(out/'grid_manifest.csv',index=False); pd.DataFrame(curves).to_csv(out/'learning_curves.csv',index=False)
        from stage_b import run_stage_b
        run_stage_b(out,c,limit_candidates=(2 if a.smoke else None))
        status.update(status='stage_a_completed',elapsed_s=time.monotonic()-started_monotonic); (out/'status.json').write_text(json.dumps(status,indent=2))
    except Exception as e:
        status.update(status='failed',error=str(e)); (out/'status.json').write_text(json.dumps(status,indent=2)); raise
if __name__=='__main__': main()
