"""Resumable 4-fold, 3-seed grouped NLR training metadata assessment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core import QUANTILES, crps_quantile, history_features, static_features

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'results/nlr_training_pilot_20260927/targets.parquet'
DEFAULT = HERE / 'results/nlr_training_cv4_20260927'
SEEDS = (17,42,2026)
CAP = 1000
CONDITIONS = ('history_only','history_model_node')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, obj):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,default=str)+'\n')
    temp.replace(path)


def split(frame):
    groups=frame[['run_id','group_id','model_name','node_count']].drop_duplicates().sort_values('run_id').copy()
    if len(groups)!=41 or groups.group_id.nunique()!=41:
        raise ValueError('Expected 41 unique job groups')
    groups['outer_fold']=-1
    rng=np.random.default_rng(2026)
    for _,cohort in groups.groupby(['model_name','node_count']):
        ordered=cohort.index.to_numpy().copy()
        rng.shuffle(ordered)
        groups.loc[ordered,'outer_fold']=np.arange(len(ordered))%4+1
    if set(groups.outer_fold)!={1,2,3,4}:
        raise AssertionError('Four stratified folds required')
    for fold in range(1,5):
        test=groups[groups.outer_fold.eq(fold)]
        if len(test.groupby(['model_name','node_count']))!=8:
            raise AssertionError('Each fold must test all eight conditions')
    return groups


def sampled(frame, indices, seed):
    rng=np.random.default_rng(seed)
    parts=[]
    for _,group in frame.iloc[indices].groupby('run_id',sort=True):
        eligible=group.index.to_numpy()
        parts.append(np.sort(rng.choice(eligible,size=min(CAP,len(eligible)),replace=False)))
    return np.sort(np.concatenate(parts))


def fit_cell(frame,power,groups,fold,seed,out):
    cell=out/f'fold{fold}_seed{seed}'
    cell.mkdir(exist_ok=True)
    test_runs=set(groups.loc[groups.outer_fold.eq(fold),'run_id'])
    train_runs=set(groups.run_id)-test_runs
    train_pool=np.flatnonzero(frame.run_id.isin(train_runs))
    test=np.flatnonzero(frame.run_id.isin(test_runs))
    train=sampled(frame,train_pool,seed)
    counts=frame.iloc[train].groupby('run_id').size()
    weights=frame.iloc[train].run_id.map(1/counts).to_numpy()*len(counts)
    metadata_train,metadata_test=static_features(frame,train,test,columns=('model_name','node_count'))
    features={'history_only':(power[train],power[test]),
        'history_model_node':(np.column_stack([power[train],metadata_train]),
                              np.column_stack([power[test],metadata_test]))}
    truth=frame.y.to_numpy()
    predictions=[];parameters=[]
    for condition,(xtrain,xtest) in features.items():
        raw=[]
        for q in QUANTILES:
            params=dict(objective='quantile',alpha=float(q),n_estimators=100,num_leaves=15,
                max_depth=-1,learning_rate=.07,min_child_samples=30,reg_alpha=0.,reg_lambda=0.,
                colsample_bytree=1.,subsample=1.,subsample_freq=0,random_state=seed,n_jobs=1,verbosity=-1)
            model=lgb.LGBMRegressor(**params)
            model.fit(xtrain,truth[train],sample_weight=weights)
            raw.append(model.predict(xtest))
            parameters.append(dict(condition=condition,quantile=float(q),seed=seed,outer_fold=fold,
                fitted_trees=model.booster_.num_trees(),**params))
        ordered=np.sort(np.column_stack(raw),axis=1)
        scored=frame.iloc[test][['run_id','group_id','issue_s','target_start_s','target_end_s',
            'y','delta_w','model_name','node_count']].copy()
        scored['condition']=condition;scored['seed']=seed;scored['outer_fold']=fold
        scored['crps_w']=crps_quantile(truth[test],ordered)
        scored['q50']=ordered[:,7]
        scored['absolute_error_w']=abs(scored.y-scored.q50)
        scored['covered_80']=scored.y.between(ordered[:,3],ordered[:,11])
        for j,q in enumerate(QUANTILES):scored[f'q_{q:g}']=ordered[:,j]
        predictions.append(scored)
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_parquet(cell/'predictions.parquet',index=False)
    pd.DataFrame(parameters).to_csv(cell/'selected_hyperparameters.csv',index=False)
    manifest=dict(fold=fold,seed=seed,train_run_ids=sorted(train_runs),test_run_ids=sorted(test_runs),
        train_rows=len(train),test_rows=len(test),sample_index_sha256=hashlib.sha256(train.tobytes()).hexdigest())
    save(cell/'manifest.json',manifest)
    checks=dict(paired_rows=bool(pred.groupby(['run_id','issue_s']).condition.nunique().eq(2).all()),
        same_truth=bool(pred.groupby(['run_id','issue_s']).y.nunique().eq(1).all()),
        no_group_overlap=not bool(set(frame.iloc[train].group_id)&set(frame.iloc[test].group_id)),
        right_edge_target=bool(np.allclose(pred.target_end_s-pred.issue_s,1)),
        score_recomputed=bool(np.allclose(crps_quantile(pred.y.to_numpy(),
            pred[[f'q_{q:g}' for q in QUANTILES]].to_numpy()),pred.crps_w)))
    checks['all_passed']=all(checks.values())
    save(cell/'verification.json',checks)
    if not checks['all_passed']:raise AssertionError(checks)
    return sha(cell/'predictions.parquet')


def summarize(out,groups):
    parts=[];parameters=[]
    for fold in range(1,5):
        for seed in SEEDS:
            cell=out/f'fold{fold}_seed{seed}'
            if not json.loads((cell/'verification.json').read_text())['all_passed']:
                raise AssertionError('Unverified cell')
            parts.append(pd.read_parquet(cell/'predictions.parquet'))
            parameters.append(pd.read_csv(cell/'selected_hyperparameters.csv'))
    pred=pd.concat(parts,ignore_index=True)
    pred.to_parquet(out/'per_target_predictions.parquet',index=False)
    pd.concat(parameters,ignore_index=True).to_csv(out/'selected_hyperparameters.csv',index=False)
    per=pred.groupby(['run_id','model_name','node_count','seed','outer_fold','condition'],as_index=False).agg(
        n_targets=('y','size'),crps_w=('crps_w','mean'),mae_w=('absolute_error_w','mean'),
        p95_w=('absolute_error_w',lambda x:x.quantile(.95)),
        p99_w=('absolute_error_w',lambda x:x.quantile(.99)),coverage_80=('covered_80','mean'))
    per.to_csv(out/'per_run_metrics.csv',index=False)
    run=per.groupby(['run_id','condition'],as_index=False)[['crps_w','mae_w','p95_w','p99_w','coverage_80']].mean()
    aggregate=run.groupby('condition',as_index=False)[['crps_w','mae_w','p95_w','p99_w','coverage_80']].mean()
    aggregate.to_csv(out/'aggregate_metrics.csv',index=False)
    pivot=run.pivot(index='run_id',columns='condition',values='crps_w')
    result=dict(n_jobs=len(pivot),n_source_experiment_groups=1,
        history_crps_w=float(pivot.history_only.mean()),metadata_crps_w=float(pivot.history_model_node.mean()),
        descriptive_gain_pct=float(100*(1-pivot.history_model_node.mean()/pivot.history_only.mean())),
        improved_jobs=int((pivot.history_model_node<pivot.history_only).sum()),
        assessment='retrospective grouped job CV, four folds, three seeds; one source experiment group; no independent-session CI')
    save(out/'paired_gain_descriptive.json',result)
    checks=dict(n_jobs=pred.run_id.nunique(),seeds=sorted(pred.seed.unique().tolist()),
        each_job_one_test_fold=bool(groups.run_id.nunique()==41 and pred.groupby('run_id').outer_fold.nunique().eq(1).all()),
        paired_conditions=bool(pred.groupby(['run_id','seed','issue_s']).condition.nunique().eq(2).all()),
        same_truth=bool(pred.groupby(['run_id','seed','issue_s']).y.nunique().eq(1).all()),
        score_recomputed=bool(np.allclose(crps_quantile(pred.y.to_numpy(),
            pred[[f'q_{q:g}' for q in QUANTILES]].to_numpy()),pred.crps_w)))
    checks['all_passed']=checks['n_jobs']==41 and checks['seeds']==list(SEEDS) and all(v for k,v in checks.items() if isinstance(v,bool))
    save(out/'verification.json',checks)
    if not checks['all_passed']:raise AssertionError(checks)
    return result


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    signature=dict(code_sha256=sha(Path(__file__)),core_sha256=sha(HERE/'core.py'),
        target_sha256=sha(SOURCE),seeds=SEEDS,cap=CAP,quantiles=QUANTILES.tolist(),
        experiment='four stratified job folds; fixed historical model; one source experiment')
    sigpath=out/'checkpoint_signature.json'
    if sigpath.exists() and json.loads(sigpath.read_text())!=json.loads(json.dumps(signature)):
        raise ValueError('Code/data/config changed; use a new output directory')
    save(sigpath,signature)
    frame=pd.read_parquet(SOURCE)
    groups=split(frame)
    groups.to_csv(out/'split_manifest.csv',index=False)
    power=np.vstack([history_features(h) for h in frame.history])
    progress_path=out/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for fold in range(1,5):
        for seed in SEEDS:
            key=f'fold{fold}_seed{seed}'
            cell=out/key
            previous=progress.get(key,{})
            pred_path=cell/'predictions.parquet'
            if previous.get('status')=='verified' and pred_path.exists() and \
                previous.get('predictions_sha256')==sha(pred_path) and \
                json.loads((cell/'verification.json').read_text()).get('all_passed'):
                print('reused',key,flush=True);continue
            progress[key]={'status':'training'};save(progress_path,progress)
            digest=fit_cell(frame,power,groups,fold,seed,out)
            progress[key]={'status':'verified','predictions_sha256':digest}
            save(progress_path,progress)
            print('verified',key,flush=True)
    result=summarize(out,groups)
    progress['status']='completed';save(progress_path,progress)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT)
    run(parser.parse_args().out)
