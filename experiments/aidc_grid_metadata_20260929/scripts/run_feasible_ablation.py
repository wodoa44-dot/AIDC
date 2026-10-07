#!/usr/bin/env python3
"""Static-metadata ablation and run-level train-only shuffle on primary horizons."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from aidc_exp.training import encode,fit_quantile
from aidc_exp.evaluation import crps_grid
from aidc_exp.splits import sample_training

QUANTILES=[.01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99]
META=['hardware','workload','model_size_token','image_size','batch_size','sequence_length','zero_stage']
VARIANTS=['P4_reference']+[f'without_{c}' for c in META]+[f'run_shuffle_{i}' for i in range(5)]


def shuffled_by_run(train,seed):
    result=train.copy()
    groups=sorted(result.run_id.unique())
    rng=np.random.default_rng(seed)
    permutation=rng.permutation(len(groups))
    run_meta=result.groupby('run_id',sort=True)[META].first()
    source={groups[i]:groups[int(permutation[i])] for i in range(len(groups))}
    for target_run,source_run in source.items():
        mask=result.run_id.eq(target_run)
        for col in META:result.loc[mask,col]=run_meta.loc[source_run,col]
    return result,source


def run_task(data,split,h,out,mainout):
    fold=split['fold']
    selection=mainout/f'selected_f{fold}_h{h}.json'
    if not selection.exists():return False
    chosen=json.loads(selection.read_text())
    eligible=data.eligible_history & data[f'y_{h}'].notna()
    sub=data.loc[eligible]
    train=sub[sub.run_id.isin(split['train_runs'])]
    test=sub[sub.run_id.isin(split['test_runs'])]
    train=sample_training(train,5000,2026)
    if train.empty or test.empty:return False
    for variant in VARIANTS:
        path=out/'prediction_parts'/f'f{fold}_h{h}_{variant}.parquet'
        if path.exists():continue
        if variant.startswith('run_shuffle_'):
            index=int(variant.rsplit('_',1)[-1])
            training,mapping=shuffled_by_run(train,2026+index)
        else:training=train;mapping=None
        x,xv=encode(training,test,'P4',chosen['history_s'])
        if variant.startswith('without_'):
            column=variant[len('without_'):]
            x=x.drop(columns=[column]);xv=xv.drop(columns=[column])
        pred=np.column_stack([fit_quantile(x,training[f'y_{h}'],training.sample_weight,q,chosen,2026).predict(xv) for q in QUANTILES])
        crossing=float(np.mean(np.any(np.diff(pred,axis=1)<0,axis=1)))
        pred=np.sort(pred,axis=1)
        frame=test[['dataset','group_id','run_id','segment','issuance_s',f'y_{h}']].copy()
        frame=frame.rename(columns={f'y_{h}':'actual_w'})
        frame['fold']=fold;frame['horizon_s']=h*.2;frame['variant']=variant;frame['crossing_pre_correction_rate']=crossing
        for j,q in enumerate(QUANTILES):frame[f'q{q:g}']=pred[:,j]
        tmp=path.with_suffix('.tmp.parquet');frame.to_parquet(tmp,index=False);os.replace(tmp,path)
        if mapping is not None:
            (out/'shuffle_maps'/f'f{fold}_h{h}_{variant}.json').write_text(json.dumps(mapping,indent=2))
        print('DONE',path.stem,'rows',len(frame),flush=True)
    return True


def summarize(out):
    rows=[];files=sorted((out/'prediction_parts').glob('*.parquet'))
    qcols=[f'q{q:g}' for q in QUANTILES]
    for path in files:
        d=pd.read_parquet(path)
        for run,g in d.groupby('run_id'):
            rows.append({'fold':int(g.fold.iloc[0]),'horizon_s':float(g.horizon_s.iloc[0]),
                'variant':g.variant.iloc[0],'run_id':run,'n':len(g),
                'crps':float(np.mean(crps_grid(g.actual_w.to_numpy(),g[qcols].to_numpy(),QUANTILES))),
                'crossing_pre_correction_rate':float(g.crossing_pre_correction_rate.iloc[0])})
    if not rows:return
    run=pd.DataFrame(rows);run.to_csv(out/'metadata_ablation_per_run.csv',index=False)
    summary=run.groupby(['fold','horizon_s','variant'],as_index=False).crps.mean()
    reference=summary[summary.variant.eq('P4_reference')][['fold','horizon_s','crps']].rename(columns={'crps':'reference_crps'})
    summary=summary.merge(reference,on=['fold','horizon_s'],how='left')
    summary['gain_vs_reference_pct']=100*(1-summary.crps/summary.reference_crps)
    summary.to_csv(out/'metadata_ablation.csv',index=False)
    (out/'status.json').write_text(json.dumps({'tasks_done':len(files),'tasks_planned':5*2*len(VARIANTS),
        'complete':len(files)==5*2*len(VARIANTS),'notes':'full15 CRPS, seed 2026 only; per-fold selected P2 capacity fixed; run-level training metadata shuffles, test normal; session independence unverified'},indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',default='full_20260928T180455Z');ap.add_argument('--fold',type=int,default=None)
    args=ap.parse_args();project=Path(__file__).resolve().parents[1]
    mainout=project/'outputs'/args.id;out=mainout/'feasible_ablation_v1'
    for name in ('prediction_parts','shuffle_maps'):(out/name).mkdir(parents=True,exist_ok=True)
    data=pd.read_parquet(mainout/'prepared.parquet')
    splits=json.loads((mainout/'split_manifest.json').read_text())['folds']
    for split in splits:
        if args.fold is not None and split['fold']!=args.fold:continue
        for h in (5,10):
            run_task(data,split,h,out,mainout)
            summarize(out)
    summarize(out)

if __name__=='__main__':main()
