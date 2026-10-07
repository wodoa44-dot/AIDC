#!/usr/bin/env python3
"""Primary-horizon path-ramp event study. Inputs are completed, cached E0 bins."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
from aidc_exp.splits import sample_training
from aidc_exp.training import encode

CANDIDATES=[dict(n_estimators=50,num_leaves=7,min_child_samples=100),
            dict(n_estimators=100,num_leaves=15,min_child_samples=100),
            dict(n_estimators=200,num_leaves=15,min_child_samples=300)]
FALSE_ALARMS_PER_HOUR=30.0
DT=.2


def fit(x, y, w, cfg):
    if np.unique(y).size<2:
        return float(np.mean(y))
    model=LGBMClassifier(objective='binary', learning_rate=.07, reg_lambda=1,
        reg_alpha=0, max_depth=-1, colsample_bytree=1, subsample=1,
        subsample_freq=0, n_jobs=1, random_state=2026, verbosity=-1, **cfg)
    model.fit(x,y,sample_weight=w)
    return model


def probability(fitted, x):
    if isinstance(fitted,float):return np.full(len(x),fitted)
    return fitted.predict_proba(x)[:,1]


def labels(frame,h,direction,threshold):
    x=frame[f'up_max_{h}'] if direction=='up' else -frame[f'down_min_{h}']
    return (x.to_numpy()>threshold).astype('int8')


def threshold_w(train,h):
    delta=train[f'endpoint_delta_{h}'].abs().dropna().to_numpy()
    positive=delta[delta>0]
    if len(positive)==0:return float('inf')
    return float(max(np.quantile(delta,.8),positive.min()))


def brier_by_run(y,p,runs):
    z=pd.DataFrame({'run':runs,'loss':(y-p)**2})
    return float(z.groupby('run').loss.mean().mean())


def alarm_cutoff(y,p,runs):
    # Only inner-validation predictions are inspected. At most 30 false alarms/hour.
    cuts=np.r_[np.inf,np.unique(np.quantile(p,np.linspace(0,1,501)))]
    best=(0.,np.inf,np.inf)
    for cut in cuts:
        alert=p>=cut
        fa=np.mean(alert & (y==0))*3600/DT
        if fa>FALSE_ALARMS_PER_HOUR+1e-12:continue
        recall=np.sum(alert & (y==1))/max(1,np.sum(y==1))
        # Higher recall, then fewer false alarms, then higher threshold.
        key=(recall,-fa,cut)
        if key> (best[0],-best[1],best[2]):best=(recall,fa,cut)
    return float(best[2])


def tune(train,h,direction,model_name,y):
    groups=train.run_id.to_numpy()
    cv=list(GroupKFold(n_splits=min(3,len(np.unique(groups)))).split(train,groups=groups))
    records=[]; selected=None
    for i,cfg in enumerate(CANDIDATES):
        oof=np.full(len(train),np.nan)
        for tr_idx,va_idx in cv:
            tr=train.iloc[tr_idx];va=train.iloc[va_idx]
            x,xv=encode(tr,va,model_name,5)
            fitted=fit(x,y[tr_idx],tr.sample_weight.to_numpy(),cfg)
            oof[va_idx]=probability(fitted,xv)
        score=brier_by_run(y,oof,groups)
        records.append({'candidate':i,'inner_run_balanced_brier':score,**cfg})
        if selected is None or (score,i)<(selected[0],selected[1]):selected=(score,i,oof)
    _,idx,oof=selected
    cut=alarm_cutoff(y,oof,groups)
    return CANDIDATES[idx],cut,records


def task(data,split,h,direction,model_name,out):
    fold=split['fold']; key=f'f{fold}_h{h}_{direction}_{model_name}'
    target=out/'prediction_parts'/f'{key}.parquet'
    if target.exists():return
    eligible=data.eligible_history & data[f'y_{h}'].notna()
    sub=data.loc[eligible]
    outer_train=sub[sub.run_id.isin(split['train_runs'])]
    outer_test=sub[sub.run_id.isin(split['test_runs'])]
    if len(outer_train)==0 or len(outer_test)==0:return
    threshold=threshold_w(outer_train,h)
    train=sample_training(outer_train,5000,2026)
    y=labels(train,h,direction,threshold)
    cfg,cut,records=tune(train,h,direction,model_name,y)
    x,xv=encode(train,outer_test,model_name,5)
    start=time.perf_counter()
    fitted=fit(x,y,train.sample_weight.to_numpy(),cfg)
    p=probability(fitted,xv)
    fit_s=time.perf_counter()-start
    actual=labels(outer_test,h,direction,threshold)
    endpoint=(outer_test[f'endpoint_delta_{h}'].to_numpy() * (1 if direction=='up' else -1)>threshold).astype('int8')
    frame=outer_test[['dataset','group_id','run_id','segment','issuance_s']].copy()
    frame['fold']=fold;frame['horizon_s']=h*DT;frame['direction']=direction;frame['model']=model_name
    frame['event_probability']=p;frame['event_actual']=actual;frame['endpoint_event_actual']=endpoint
    frame['event_threshold_w']=threshold;frame['alarm_probability_threshold']=cut
    frame['target_time_s']=frame.issuance_s+h*DT;frame['alarm']=(p>=cut).astype('int8')
    frame['evidence']='E0 GPU sum; path excursion from issuance power'
    tmp=target.with_suffix('.tmp.parquet');frame.to_parquet(tmp,index=False);os.replace(tmp,target)
    detail={'fold':fold,'horizon_s':h*DT,'direction':direction,'model':model_name,
            'event_threshold_w':threshold,'alarm_probability_threshold':cut,
            'positive_train':int(y.sum()),'sampled_train':len(train),'fit_wall_s':fit_s,
            'selected':cfg,'candidates':records,'false_alarm_budget_per_hour':FALSE_ALARMS_PER_HOUR,
            'threshold_source':'outer-train endpoint absolute delta 80th percentile with positive minimum; path event is any future completed-bin excursion from issuance power'}
    (out/'selected'/f'{key}.json').write_text(json.dumps(detail,indent=2))
    print('DONE',key,'test_rows',len(frame),'train_positive',int(y.sum()),flush=True)


def summarize(out):
    files=sorted((out/'prediction_parts').glob('*.parquet'))
    rows=[]
    for file in files:
        d=pd.read_parquet(file)
        for run,g in d.groupby('run_id'):
            y=g.event_actual.to_numpy();p=g.event_probability.to_numpy();a=g.alarm.to_numpy().astype(bool)
            positives=int(y.sum());false=int(np.sum(a & (y==0)));correct=int(np.sum(a & (y==1)))
            rows.append({'fold':int(g.fold.iloc[0]),'horizon_s':float(g.horizon_s.iloc[0]),
                'direction':g.direction.iloc[0],'model':g.model.iloc[0],'run_id':run,'n':len(g),
                'positives':positives,'brier':float(np.mean((p-y)**2)),
                'pr_auc':float(average_precision_score(y,p)) if positives else np.nan,
                'precision':correct/max(1,int(a.sum())) if a.sum() else np.nan,
                'recall':correct/positives if positives else np.nan,
                'missed_event_rate':1-correct/positives if positives else np.nan,
                'false_alarms_per_hour':false*3600/(len(g)*DT),
                'endpoint_path_disagreement_rate':float(np.mean(g.endpoint_event_actual.to_numpy()!=y))})
    if not rows:return
    result=pd.DataFrame(rows)
    result.to_csv(out/'event_per_run_metrics.csv',index=False)
    result.groupby(['horizon_s','direction','model'],as_index=False)[['brier','pr_auc','precision','recall','missed_event_rate','false_alarms_per_hour','endpoint_path_disagreement_rate']].mean().to_csv(out/'event_aggregate_metrics.csv',index=False)
    (out/'status.json').write_text(json.dumps({'tasks_done':len(files),'tasks_planned':40,'complete':len(files)==40,
        'notes':'PR-AUC for zero-positive runs is N/A; pointwise alarm metrics, not one-to-one event matching or lead time. Session independence unverified.'},indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',default='full_20260928T180455Z');ap.add_argument('--one',action='store_true')
    args=ap.parse_args();project=Path(__file__).resolve().parents[1]
    mainout=project/'outputs'/args.id
    out=mainout/'feasible_events_v2';(out/'prediction_parts').mkdir(parents=True,exist_ok=True);(out/'selected').mkdir(exist_ok=True)
    data=pd.read_parquet(mainout/'prepared.parquet')
    splits=json.loads((mainout/'split_manifest.json').read_text())['folds']
    for split in splits:
        for h in (5,10):
            for direction in ('up','down'):
                for model in ('P2','P4'):
                    task(data,split,h,direction,model,out)
                    summarize(out)
                    if args.one:return
    summarize(out)

if __name__=='__main__':main()
