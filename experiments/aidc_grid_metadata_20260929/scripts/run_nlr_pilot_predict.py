#!/usr/bin/env python3
"""Exploratory 5-run P0/P2 forecast on reconstructed raw NLR one-node logs."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from aidc_exp.features import build_features
from aidc_exp.targets import build_targets
from aidc_exp.splits import sample_training
from aidc_exp.training import encode,fit_quantile
from aidc_exp.evaluation import crps_grid

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/'outputs/nlr_raw_causal_pilot_v1'
OUT=SOURCE/'forecast_compact3_v1'
Q=[.1,.5,.9]
CONFIG={'n_estimators':100,'num_leaves':15,'learning_rate':.07,'min_child_samples':30,'reg_lambda':0,'history_s':5}


def prepare():
    frames=[]
    for p in sorted(SOURCE.glob('*.parquet')):
        d=pd.read_parquet(p)
        b=pd.DataFrame({'power_w':d.total_w,'segment':d.segment})
        f=build_features(b,history_bins=150)
        y=build_targets(b,[5,10])
        z=pd.concat([d[['run_id','segment','issuance_time','time_s','gpu_age_s','cpu_age_s','device_coverage']],f,y],axis=1)
        z['bin']=np.arange(len(z));frames.append(z)
    return pd.concat(frames,ignore_index=True)


def work(data,fold,h):
    path=OUT/'prediction_parts'/f'f{fold}_h{h}.parquet'
    if path.exists():return
    runs=sorted(data.run_id.unique());test_run=runs[fold]
    eligible=data.eligible_history & data[f'y_{h}'].notna()
    sub=data.loc[eligible]
    train=sample_training(sub[sub.run_id.ne(test_run)],5000,2026)
    test=sub[sub.run_id.eq(test_run)]
    target=f'y_{h}';actual=test[target].to_numpy();current=test.current_w.to_numpy()
    residual=train[target].to_numpy()-train.current_w.to_numpy()
    p0=np.column_stack([current+np.quantile(residual,q) for q in Q])
    x,xv=encode(train,test,'P2',5)
    p2=np.column_stack([fit_quantile(x,train[target],train.sample_weight,q,CONFIG,2026).predict(xv) for q in Q])
    frame=test[['run_id','segment','issuance_time','time_s','gpu_age_s','cpu_age_s','device_coverage']].copy()
    frame['fold']=fold;frame['horizon_s']=h*.2;frame['target_time']=frame.issuance_time+pd.to_timedelta(h*.2,unit='s')
    frame['actual_w']=actual;frame['current_w']=current
    for name,pred in [('P0',p0),('P2',np.sort(p2,axis=1))]:
        for j,q in enumerate(Q):frame[f'{name}_q{q:g}']=pred[:,j]
    tmp=path.with_suffix('.tmp.parquet');frame.to_parquet(tmp,index=False);os.replace(tmp,path)
    print('DONE',path.stem,'rows',len(frame),flush=True)


def summary():
    rows=[]
    for p in sorted((OUT/'prediction_parts').glob('*.parquet')):
        d=pd.read_parquet(p);actual=d.actual_w.to_numpy()
        for name in ('P0','P2'):
            q=d[[f'{name}_q{x:g}' for x in Q]].to_numpy();median=q[:,1]
            rows.append({'fold':int(d.fold.iloc[0]),'run_id':d.run_id.iloc[0],
                'horizon_s':float(d.horizon_s.iloc[0]),'model':name,'n':len(d),
                'compact3_crps_w':float(np.mean(crps_grid(actual,q,Q))),
                'mae_w':float(np.mean(abs(median-actual))),
                'p95_abs_w':float(np.percentile(abs(median-actual),95))})
    if not rows:return
    a=pd.DataFrame(rows);a.to_csv(OUT/'per_run_metrics.csv',index=False)
    a.groupby(['horizon_s','model'],as_index=False)[['compact3_crps_w','mae_w','p95_abs_w']].mean().to_csv(OUT/'aggregate_metrics.csv',index=False)
    (OUT/'status.json').write_text(json.dumps({'tasks_done':len(list((OUT/'prediction_parts').glob('*.parquet'))),
        'tasks_planned':10,'complete':len(rows)==20,
        'scope':'NLR stable diffusion 1node 5 repeated jobs; conditional timestamp-as-availability assumption; not pooled with highres',
        'score':'compact3 CRPS approximation; not comparable to highres full15 CRPS',
        'models':'P0 residual distribution and fixed P2 history quantile LightGBM; no metadata variation; no hyperparameter search'},indent=2))


def main():
    (OUT/'prediction_parts').mkdir(parents=True,exist_ok=True)
    data=prepare();runs=sorted(data.run_id.unique())
    for fold in range(len(runs)):
        for h in (5,10):work(data,fold,h);summary()

if __name__=='__main__':main()
