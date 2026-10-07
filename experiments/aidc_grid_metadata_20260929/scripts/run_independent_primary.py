#!/usr/bin/env python3
"""Independent P2/P3/P4 nested tuning for primary 1/2s E0 targets."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from aidc_exp.training import candidates,encode,fit_quantile
from aidc_exp.splits import sample_training
from aidc_exp.evaluation import pinball,crps_grid

PROJECT=Path(__file__).resolve().parents[1]
MAIN=PROJECT/'outputs/full_20260928T180455Z'
OUT=MAIN/'independent_primary_v1'
Q=[.01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99]
SCREEN_Q=[.1,.5,.9]
OPTIONS=candidates(24)


def save_csv(frame,path):
    tmp=path.with_suffix('.tmp.csv');frame.to_csv(tmp,index=False);os.replace(tmp,path)


def run_balanced(values,runs):
    return float(pd.DataFrame({'run':runs,'loss':values}).groupby('run').loss.mean().mean())


def tune(data,h,name,fold):
    task=f'f{fold}_h{h}_{name}'
    screen_path=OUT/'screens'/f'{task}.csv';full_path=OUT/'full15'/f'{task}.csv';chosen_path=OUT/'selected'/f'{task}.json'
    train=sample_training(data,5000,2026)
    groups=train.run_id.to_numpy();cv=list(GroupKFold(n_splits=min(3,len(np.unique(groups)))).split(train,groups=groups))
    screens=pd.read_csv(screen_path).to_dict('records') if screen_path.exists() else []
    done={int(row['candidate']) for row in screens}
    for i,option in enumerate(OPTIONS):
        if i in done:continue
        fold_scores=[]
        for ti,vi in cv:
            tr,va=train.iloc[ti],train.iloc[vi];x,xv=encode(tr,va,name,option['history_s'])
            loss=np.mean(np.column_stack([pinball(va[f'y_{h}'].to_numpy(),fit_quantile(x,tr[f'y_{h}'],tr.sample_weight,q,option,2026).predict(xv),q) for q in SCREEN_Q]),axis=1)
            fold_scores.append(run_balanced(loss,va.run_id.to_numpy()))
        screens.append({'candidate':i,'screen_run_balanced_mean_pinball':float(np.mean(fold_scores))})
        save_csv(pd.DataFrame(screens),screen_path)
        print('SCREEN',task,i+1,'/24',flush=True)
    ranked=pd.DataFrame(screens).sort_values(['screen_run_balanced_mean_pinball','candidate']).head(4)
    full=pd.read_csv(full_path).to_dict('records') if full_path.exists() else []
    done={int(row['candidate']) for row in full}
    for i in ranked.candidate.astype(int):
        if i in done:continue
        option=OPTIONS[i];scores=[]
        for ti,vi in cv:
            tr,va=train.iloc[ti],train.iloc[vi];x,xv=encode(tr,va,name,option['history_s'])
            p=np.column_stack([fit_quantile(x,tr[f'y_{h}'],tr.sample_weight,q,option,2026).predict(xv) for q in Q])
            scores.append(run_balanced(crps_grid(va[f'y_{h}'].to_numpy(),np.sort(p,axis=1),Q),va.run_id.to_numpy()))
        full.append({'candidate':i,'inner_run_balanced_full15_crps':float(np.mean(scores))})
        save_csv(pd.DataFrame(full),full_path)
        print('FULL15',task,len(full),'/4',flush=True)
    winning=int(pd.DataFrame(full).sort_values(['inner_run_balanced_full15_crps','candidate']).iloc[0].candidate)
    chosen={'fold':fold,'horizon_s':h*.2,'model':name,'candidate':winning,'params':OPTIONS[winning],
            'selection':'minimum inner run-balanced full15 CRPS among top four by compact3 screening',
            'train_runs':int(train.run_id.nunique()),'train_rows':len(train)}
    chosen_path.write_text(json.dumps(chosen,indent=2))
    return train,chosen


def predict(train,test,h,name,fold,choice):
    option=choice['params'];x,xv=encode(train,test,name,option['history_s'])
    for seed in (17,42,2026):
        path=OUT/'prediction_parts'/f'f{fold}_h{h}_s{seed}_{name}.parquet'
        if path.exists():continue
        p=np.column_stack([fit_quantile(x,train[f'y_{h}'],train.sample_weight,q,option,seed).predict(xv) for q in Q])
        frame=test[['dataset','group_id','run_id','segment','issuance_s','current_w',f'y_{h}']].copy().rename(columns={f'y_{h}':'actual_w'})
        frame['fold']=fold;frame['horizon_s']=h*.2;frame['seed']=seed;frame['model']=name
        frame['target_time_s']=frame.issuance_s+h*.2;frame['crossing_pre_correction_rate']=float(np.mean(np.any(np.diff(p,axis=1)<0,axis=1)))
        p=np.sort(p,axis=1)
        for j,q in enumerate(Q):frame[f'q{q:g}']=p[:,j]
        tmp=path.with_suffix('.tmp.parquet');frame.to_parquet(tmp,index=False);os.replace(tmp,path)
        print('PREDICT',path.stem,'rows',len(frame),flush=True)


def summarize():
    rows=[];files=sorted((OUT/'prediction_parts').glob('*.parquet'))
    cols=[f'q{q:g}' for q in Q]
    for file in files:
        d=pd.read_parquet(file)
        for run,g in d.groupby('run_id'):
            actual=g.actual_w.to_numpy();q=g[cols].to_numpy();e=q[:,7]-actual
            rows.append({'fold':int(g.fold.iloc[0]),'horizon_s':float(g.horizon_s.iloc[0]),'seed':int(g.seed.iloc[0]),
                'model':g.model.iloc[0],'run_id':run,'n':len(g),'crps':float(np.mean(crps_grid(actual,q,Q))),
                'mae':float(np.mean(abs(e))),'p95_abs':float(np.percentile(abs(e),95))})
    if not rows:return
    run=pd.DataFrame(rows);run.to_csv(OUT/'per_run_metrics.csv',index=False)
    mean=run.groupby(['horizon_s','model','run_id'],as_index=False)[['crps','mae','p95_abs']].mean()
    mean.groupby(['horizon_s','model'],as_index=False)[['crps','mae','p95_abs']].mean().to_csv(OUT/'aggregate_metrics.csv',index=False)
    (OUT/'status.json').write_text(json.dumps({'prediction_tasks_done':len(files),'prediction_tasks_planned':90,
        'complete':len(files)==90,'screen_candidates':24,'full15_top':4,'seeds':[17,42,2026]},indent=2))


def main():
    OUT.mkdir(exist_ok=True)
    for folder in ('screens','full15','selected','prediction_parts'):(OUT/folder).mkdir(exist_ok=True)
    source_files=[Path(__file__),MAIN/'prepared.parquet',MAIN/'split_manifest.json',PROJECT/'configs/full.yaml']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    manifest=OUT/'fingerprint.json'
    if manifest.exists() and json.loads(manifest.read_text())!=hashes:raise RuntimeError('fingerprint mismatch')
    manifest.write_text(json.dumps(hashes,indent=2))
    (OUT/'candidate_manifest.json').write_text(json.dumps({'candidates':OPTIONS,'screen_quantiles':SCREEN_Q,'full_quantiles':Q,'top_n_full':4,'seed':2026},indent=2))
    data=pd.read_parquet(MAIN/'prepared.parquet');splits=json.loads((MAIN/'split_manifest.json').read_text())['folds']
    for split in splits:
        fold=split['fold']
        for h in (5,10):
            eligible=data.eligible_history & data[f'y_{h}'].notna();sub=data.loc[eligible]
            train_pool=sub[sub.run_id.isin(split['train_runs'])];test=sub[sub.run_id.isin(split['test_runs'])]
            for name in ('P2','P3','P4'):
                train,choice=tune(train_pool,h,name,fold)
                predict(train,test,h,name,fold,choice)
                summarize()

if __name__=='__main__':main()
