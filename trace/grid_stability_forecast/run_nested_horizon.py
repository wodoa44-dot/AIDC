"""Resumable staged grouped inner search for a specified highres horizon.

Uses the candidate, fold-fit, and outer-fit implementations in nested_select.py.
Existing +1 s artifacts remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nested_select as ns
from core import QUANTILES, crps_quantile, history_features, write_json
from unseen_condition import sampled_rows
from verify_nested import verify

HERE=Path(__file__).resolve().parent


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def run(source,out,horizon_ms):
    source,out=Path(source),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    target_path=source/'targets.parquet'
    source_cfg=json.loads((source/'resolved_config.json').read_text())
    source_sha=sha(target_path)
    if source_cfg['source_sha256']!=source_sha:
        raise ValueError('Source target hash mismatch')
    signature=dict(horizon_ms=horizon_ms,target_sha256=source_sha,
        split_sha256=sha(source/'split_manifest.csv'),
        code_sha256={name:sha(HERE/name) for name in ['run_nested_horizon.py','nested_select.py',
            'core.py','unseen_condition.py','verify_nested.py']},
        candidate_count=24,shortlist=ns.SHORTLIST,outer_seeds=[17,42,2026],
        history_ms=5000,train_cap=ns.CAP,threads=1)
    sigpath=out/'checkpoint_signature.json'
    if sigpath.exists() and json.loads(sigpath.read_text())!=signature:
        raise ValueError('Code, data, split or configuration changed; use a new output directory')
    write_json(sigpath,signature)
    frame=pd.read_parquet(target_path)
    frame=frame.loc[frame.horizon_ms.eq(horizon_ms)].reset_index(drop=True)
    if frame.empty:raise ValueError(f'No {horizon_ms} ms targets')
    outer=pd.read_csv(source/'split_manifest.csv').set_index('group_id').split
    if frame.group_id.map(outer).isna().any():raise ValueError('Missing source group in split')
    outer_train=np.flatnonzero(frame.group_id.map(outer).eq('train'))
    outer_test=np.flatnonzero(frame.group_id.map(outer).eq('test'))
    folds=ns.inner_folds(frame,outer_train)
    folds.to_csv(out/'inner_folds.csv',index=False)
    assignment=frame.run_id.map(folds.set_index('run_id').inner_fold)
    power=np.vstack([history_features(h) for h in frame.history])
    inventory=ns.candidates()
    write_json(out/'candidate_manifest.json',dict(candidates=inventory,generation_seed=2026,
        horizon_ms=horizon_ms,screening_quantile=.5,screening_trees=ns.SCREEN_TREES,
        shortlist_per_condition=ns.SHORTLIST,full_quantiles=QUANTILES.tolist(),
        full_tree_cap=ns.FULL_TREES,early_stopping_patience=50,inner_folds=3,
        sample_cap_per_training_run=ns.CAP,
        assessment='reused highres outer holdout; not independent confirmation',source_sha256=source_sha))
    code_hash=hashlib.sha256((HERE/'run_nested_horizon.py').read_bytes()+
        (HERE/'nested_select.py').read_bytes()+(HERE/'core.py').read_bytes()).hexdigest()
    screen_rows=[];full_rows=[]
    for condition in ns.FEATURES:
        for candidate in inventory:
            cid=candidate['candidate_id']
            path=out/'screen'/condition/f'{cid}.json'
            if path.exists():
                record=json.loads(path.read_text())
                if record.get('code_sha256')==code_hash and record.get('source_sha256')==source_sha:
                    screen_rows.append(record);continue
            path.parent.mkdir(parents=True,exist_ok=True)
            scores=[]
            for fold in range(3):
                pool=np.flatnonzero(assignment.ne(fold)&frame.group_id.map(outer).eq('train'))
                sample=sampled_rows(frame,pool,ns.SELECTION_SEED,ns.CAP)
                val=np.flatnonzero(assignment.eq(fold))
                pred,_=ns.fit_fold(frame,power,sample,val,condition,candidate,[.5],ns.SCREEN_TREES,False)
                y=frame.y.to_numpy()[val]
                loss=np.abs(y-pred[:,0])/2
                scored=pd.DataFrame(dict(run_id=frame.iloc[val].run_id.to_numpy(),loss=loss))
                scores.extend(scored.groupby('run_id').loss.mean().tolist())
            record=dict(condition=condition,candidate_id=cid,screen_loss=float(np.mean(scores)),
                code_sha256=code_hash,source_sha256=source_sha)
            write_json(path,record);screen_rows.append(record)
            print('screen',condition,cid,round(record['screen_loss'],3),flush=True)
        screen=pd.DataFrame([x for x in screen_rows if x['condition']==condition])
        shortlist=screen.nsmallest(ns.SHORTLIST,'screen_loss').candidate_id.tolist()
        for cid in shortlist:
            candidate=next(x for x in inventory if x['candidate_id']==cid)
            path=out/'full15'/condition/f'{cid}.json'
            if path.exists():
                record=json.loads(path.read_text())
                if record.get('code_sha256')==code_hash and record.get('source_sha256')==source_sha:
                    full_rows.append(record);continue
            path.parent.mkdir(parents=True,exist_ok=True)
            scores=[];trees=[]
            for fold in range(3):
                pool=np.flatnonzero(assignment.ne(fold)&frame.group_id.map(outer).eq('train'))
                sample=sampled_rows(frame,pool,ns.SELECTION_SEED,ns.CAP)
                val=np.flatnonzero(assignment.eq(fold))
                pred,fitted=ns.fit_fold(frame,power,sample,val,condition,candidate,
                    QUANTILES,ns.FULL_TREES,True)
                value=crps_quantile(frame.y.to_numpy()[val],np.sort(pred,axis=1))
                scored=pd.DataFrame(dict(run_id=frame.iloc[val].run_id.to_numpy(),crps_w=value))
                scores.extend(scored.groupby('run_id').crps_w.mean().tolist())
                trees.append(fitted)
            record=dict(condition=condition,candidate_id=cid,
                inner_run_balanced_crps_w=float(np.mean(scores)),
                tree_count_per_quantile=np.median(trees,axis=0).astype(int).tolist(),
                code_sha256=code_hash,source_sha256=source_sha)
            write_json(path,record);full_rows.append(record)
            print('full15',condition,cid,round(record['inner_run_balanced_crps_w'],3),flush=True)
    pd.DataFrame(screen_rows).to_csv(out/'screening_metrics.csv',index=False)
    pd.DataFrame(full_rows).drop(columns=['tree_count_per_quantile']).to_csv(
        out/'full15_inner_metrics.csv',index=False)
    selected={condition:min((x for x in full_rows if x['condition']==condition),
        key=lambda x:x['inner_run_balanced_crps_w']) for condition in ns.FEATURES}
    write_json(out/'selection.json',selected)
    print('selection',{k:v['candidate_id'] for k,v in selected.items()},flush=True)
    ns.evaluate_outer(frame,power,outer_train,outer_test,inventory,selected,out,source_sha)
    samples=[]
    for seed in (17,42,2026):
        ids=sampled_rows(frame,outer_train,seed,ns.CAP)
        samples.append(frame.iloc[ids][['run_id']].assign(source_row_index=ids,seed=seed))
    pd.concat(samples).to_parquet(out/'outer_sample_manifest.parquet',index=False)
    checks=verify(out,source)
    selected_params=pd.read_csv(out/'selected_hyperparameters.csv')
    selected_params['horizon_ms']=horizon_ms
    selected_params.to_csv(out/'selected_hyperparameters.csv',index=False)
    progress=dict(status='completed',horizon_ms=horizon_ms,verification=checks,
        prediction_sha256=sha(out/'per_target_predictions.parquet'))
    write_json(out/'progress.json',progress)
    print('completed',horizon_ms,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--horizon-ms',type=int,required=True)
    args=parser.parse_args()
    run(args.source,args.out,args.horizon_ms)
