#!/usr/bin/env python3
"""Streaming tail, regime and calibration metrics for available primary E0 parts."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from aidc_exp.evaluation import crps_grid

PROJECT=Path(__file__).resolve().parents[1]
MAIN=PROJECT/'outputs/full_20260928T180455Z'
OUT=MAIN/'extended_primary_eval_v1'
Q=[.01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99]
QCOL=[f'q{q:g}' for q in Q]


def threshold_table():
    data=pd.read_parquet(MAIN/'prepared.parquet',columns=['run_id','eligible_history','endpoint_delta_5','endpoint_delta_10','y_5','y_10'])
    splits=json.loads((MAIN/'split_manifest.json').read_text())['folds']
    result={}
    for split in splits:
        for h in (5,10):
            sub=data[data.run_id.isin(split['train_runs']) & data.eligible_history & data[f'y_{h}'].notna()]
            a=sub[f'endpoint_delta_{h}'].abs().to_numpy();positive=a[a>0]
            result[(split['fold'],h)]=float(max(np.quantile(a,.8),positive.min()))
    return result


def interval_score(y,lo,hi,alpha):
    return (hi-lo)+2/alpha*(lo-y)*(y<lo)+2/alpha*(y-hi)*(y>hi)


def evaluate_file(path,threshold):
    data=pd.read_parquet(path)
    fold=int(data.fold.iloc[0]);h=int(round(float(data.horizon.iloc[0])/.2))
    th=threshold[(fold,h)]
    rows=[];cal=[]
    for run,g in data.groupby('run_id'):
        y=g.actual_w.to_numpy();q=g[QCOL].to_numpy();median=q[:,7];e=median-y;ae=abs(e)
        delta=y-g.current_w.to_numpy();regime=np.where(delta>th,'up',np.where(delta< -th,'down','stable'))
        for r in ('all','up','down','stable'):
            mask=np.ones(len(g),dtype=bool) if r=='all' else regime==r
            if not mask.any():continue
            yy=y[mask];qq=q[mask];ee=e[mask];aa=ae[mask]
            row={'fold':fold,'horizon_s':h*.2,'seed':int(g.seed.iloc[0]),'model':g.model.iloc[0],
                'run_id':run,'regime':r,'n':len(yy),'regime_threshold_w':th,
                'crps':float(np.mean(crps_grid(yy,qq,Q))),'mae':float(np.mean(aa)),
                'rmse':float(np.sqrt(np.mean(ee*ee))),'bias':float(np.mean(ee)),
                'p90_abs':float(np.percentile(aa,90)),'p95_abs':float(np.percentile(aa,95)),
                'p99_abs':float(np.percentile(aa,99)),'max_abs':float(aa.max()),
                'worst_1pct_mean':float(np.mean(np.sort(aa)[-max(1,int(np.ceil(.01*len(aa)))):])),
                'worst_5pct_mean':float(np.mean(np.sort(aa)[-max(1,int(np.ceil(.05*len(aa)))):])),'under_rate':float(np.mean(ee<0)),
                'over_rate':float(np.mean(ee>0))}
            for level,low_idx,high_idx in [(.8,3,11),(.9,2,12)]:
                low=qq[:,low_idx];high=qq[:,high_idx];alpha=1-level
                tag=f'{int(level*100)}'
                row[f'coverage_{tag}']=float(np.mean((yy>=low)&(yy<=high)))
                row[f'width_{tag}']=float(np.mean(high-low))
                row[f'interval_score_{tag}']=float(np.mean(interval_score(yy,low,high,alpha)))
            rows.append(row)
        for j,nominal in enumerate(Q):
            cal.append({'fold':fold,'horizon_s':h*.2,'seed':int(g.seed.iloc[0]),'model':g.model.iloc[0],
                'run_id':run,'nominal_quantile':nominal,'empirical_coverage':float(np.mean(y<=q[:,j])),'n':len(y)})
    return pd.DataFrame(rows),pd.DataFrame(cal)


def aggregate():
    files=list((OUT/'part_metrics').glob('*.parquet'))
    if not files:return
    run=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    cal=pd.concat([pd.read_parquet(p) for p in (OUT/'part_calibration').glob('*.parquet')],ignore_index=True)
    run.to_csv(OUT/'tail_and_interval_per_run.csv',index=False)
    cal.to_csv(OUT/'quantile_reliability_per_run.csv',index=False)
    metrics=['crps','mae','rmse','bias','p90_abs','p95_abs','p99_abs','max_abs','worst_1pct_mean','worst_5pct_mean','under_rate','over_rate',
        'coverage_80','width_80','interval_score_80','coverage_90','width_90','interval_score_90']
    runmean=run.groupby(['horizon_s','model','regime','run_id'],as_index=False)[metrics].mean()
    runmean.groupby(['horizon_s','model','regime'],as_index=False)[metrics].mean().to_csv(OUT/'tail_and_interval_aggregate.csv',index=False)
    calmean=cal.groupby(['horizon_s','model','nominal_quantile','run_id'],as_index=False).empirical_coverage.mean()
    calmean.groupby(['horizon_s','model','nominal_quantile'],as_index=False).empirical_coverage.mean().to_csv(OUT/'quantile_reliability_aggregate.csv',index=False)
    (OUT/'status.json').write_text(json.dumps({'prediction_parts_evaluated':len(files),'prediction_parts_expected':150,
        'complete':len(files)==150,'scope':'1s/2s primary, available full15 controlled-track prediction parts',
        'aggregation':'row -> run -> seed mean -> run mean; source session independence unverified',
        'regime':'endpoint delta beyond outer-train 80th percentile absolute delta; path event reported separately'},indent=2))


def main():
    for folder in ('part_metrics','part_calibration'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    threshold=threshold_table()
    parts=sorted(p for p in (MAIN/'prediction_parts').glob('*.parquet') if re.search(r'_h(?:5|10)_',p.name))
    for p in parts:
        dest=OUT/'part_metrics'/p.name;caldest=OUT/'part_calibration'/p.name
        if dest.exists() and caldest.exists():continue
        a,b=evaluate_file(p,threshold)
        a.to_parquet(dest,index=False);b.to_parquet(caldest,index=False)
        print('EVALUATED',p.stem,flush=True)
    aggregate()

if __name__=='__main__':main()
