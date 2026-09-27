"""Sequential, resumable retrospective outer-5 / inner-3 grouped assessment."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from core import QUANTILES, crps_quantile
from nested_select import run as search, CAP
from unseen_condition import sampled_rows

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'results/nested_cv5_20260926'
ASSESSMENT = 'retrospective grouped outer-5 inner-3; staged 24 q50 candidates then top-six full15; session independence unverified'


def save(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, default=str) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(source, out):
    target = source / 'targets.parquet'
    frame = pd.read_parquet(target, columns=['run_id','group_id','gpu_model','workload_type','horizon_ms'])
    frame = frame[frame.horizon_ms.eq(1000)].reset_index(drop=True)
    groups = frame[['run_id','group_id','gpu_model','workload_type']].drop_duplicates().sort_values('group_id').reset_index(drop=True)
    if groups.group_id.duplicated().any() or groups.run_id.duplicated().any():
        raise ValueError('source group/run mapping is not one-to-one')
    strata = groups.gpu_model.astype(str) + '__' + groups.workload_type.astype(str)
    groups['outer_fold'] = -1
    for fold, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(groups, strata), 1):
        groups.loc[test, 'outer_fold'] = fold
    signature = dict(target_sha256=sha(target), outer_split_sha256=hashlib.sha256(groups.to_csv(index=False).encode()).hexdigest(),
        code_sha256={name:sha(HERE/name) for name in ['run_nested_cv.py','nested_select.py','core.py','unseen_condition.py']},
        config=dict(outer_folds=5, inner_folds=3, seed=2026, outer_seeds=[17,42,2026], candidate_count=24,
                    full15_shortlist=6, history_ms=5000, horizon_ms=1000, train_cap=CAP, threads=1))
    signature_path = out/'checkpoint_signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError('checkpoint config/data/code/split mismatch; choose a new output directory')
    save(signature_path,signature)
    groups.to_csv(out/'split_manifest.csv',index=False)
    for fold in range(1,6):
        input_dir=out/f'fold{fold}'/'input'
        input_dir.mkdir(parents=True,exist_ok=True)
        link=input_dir/'targets.parquet'
        if not link.exists():
            link.symlink_to(target.resolve())
        if link.resolve() != target.resolve():
            raise ValueError('target symlink differs')
        split=groups.copy()
        split['split']=np.where(split.outer_fold.eq(fold),'test','train')
        split.to_csv(input_dir/'split_manifest.csv',index=False)
        save(input_dir/'resolved_config.json',dict(source_sha256=signature['target_sha256'],horizons_ms=[1000]))
    save(out/'protocol.json',dict(assessment=ASSESSMENT, **signature['config'],
        candidates='same 24 candidates per feature condition; q50 200-tree screening, top six full15 1000-tree cap/patience 50',
        tracks=['history_only_tuned','history_static_s2_controlled','history_static_s2_tuned'],
        hardware_control=False, original_data_copied=False, physical_boundary='highres eight GPU sum, excludes CPU and facility'))
    return frame,groups


def verify_fold(out, frame, groups, fold):
    pred=pd.read_parquet(out/'per_target_predictions.parquet')
    hp=pd.read_csv(out/'selected_hyperparameters.csv')
    test_runs=set(groups.loc[groups.outer_fold.eq(fold),'run_id'])
    train_runs=set(groups.run_id)-test_runs
    conditions={'history_only_tuned','history_static_s2_controlled','history_static_s2_tuned'}
    checks=dict(test_runs_match=set(pred.run_id)==test_runs, no_group_leakage=not bool(test_runs & train_runs),
        paired_conditions=set(pred.condition)==conditions and pred.groupby(['run_id','issue_s','seed']).condition.nunique().eq(3).all(),
        seeds=set(pred.seed)=={17,42,2026}, same_truth=pred.groupby(['run_id','issue_s','seed']).y.nunique().eq(1).all())
    q=pred[[f'q_{q:g}' for q in QUANTILES]].to_numpy()
    checks['ordered_quantiles']=np.all(np.diff(q,axis=1)>=-1e-8)
    checks['score_recomputed']=np.allclose(crps_quantile(pred.y.to_numpy(),q),pred.crps_w)
    keys=['seed','quantile']
    compare=['candidate_id','selected_trees','fitted_trees','learning_rate','num_leaves','min_child_samples','reg_lambda']
    a=hp[hp.feature_condition.eq('history_only_tuned')].set_index(keys)[compare].sort_index()
    b=hp[hp.feature_condition.eq('history_static_s2_controlled')].set_index(keys)[compare].sort_index()
    checks['controlled_capacity']=a.equals(b)
    per=pd.read_csv(out/'per_run_metrics.csv').set_index(['run_id','seed','condition'])
    recalculated=pred.groupby(['run_id','seed','condition']).crps_w.mean()
    checks['run_scores_reconcile']=np.allclose(recalculated.loc[per.index],per.crps_w)
    checks={k:bool(v) for k,v in checks.items()}
    checks['all_passed']=all(checks.values())
    save(out/'verification.json',checks)
    if not checks['all_passed']:
        raise AssertionError(checks)
    return checks


def run(source,out,prepare_only=False):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another runner already holds this output lock')
        frame,groups=prepare(source,out)
        if prepare_only:
            print('prepared',groups.groupby('outer_fold').size().to_dict(),flush=True)
            return
        progress_path=out/'progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'folds':{}}
        progress.update(status='running',pid=os.getpid(),started_or_resumed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        save(progress_path,progress)
        try:
            for fold in range(1,6):
                key=f'fold{fold}'; result=out/key/'results'; result.mkdir(exist_ok=True)
                previous=progress['folds'].get(key,{})
                if previous.get('status')=='verified' and previous.get('predictions_sha256')==sha(result/'per_target_predictions.parquet'):
                    verify_fold(result,frame,groups,fold)
                    print('reused verified',key,flush=True)
                    continue
                progress['folds'][key]={'status':'training','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
                save(progress_path,progress)
                print('starting',key,flush=True)
                search(out/key/'input',result)
                verify_fold(result,frame,groups,fold)
                pool=np.flatnonzero(~frame.run_id.isin(groups.loc[groups.outer_fold.eq(fold),'run_id']))
                samples=[]
                for seed in (17,42,2026):
                    ids=sampled_rows(frame,pool,seed,CAP)
                    samples.append(frame.iloc[ids][['run_id']].assign(source_row_index=ids,seed=seed))
                pd.concat(samples).to_parquet(result/'outer_sample_manifest.parquet',index=False)
                for name in ['candidate_manifest.json','resolved_config.json']:
                    value=json.loads((result/name).read_text());value['assessment']=ASSESSMENT
                    save(result/name,value)
                paired=pd.read_csv(result/'paired_gain_ci.csv');paired['assessment']=ASSESSMENT
                paired.to_csv(result/'paired_gain_ci.csv',index=False)
                progress['folds'][key].update(status='verified',predictions_sha256=sha(result/'per_target_predictions.parquet'))
                save(progress_path,progress)
                print('verified',key,flush=True)
            per=pd.concat([pd.read_csv(out/f'fold{fold}'/'results/per_run_metrics.csv').assign(outer_fold=fold) for fold in range(1,6)],ignore_index=True)
            per.to_csv(out/'per_run_metrics.csv',index=False)
            run_mean=per.groupby(['run_id','condition']).crps_w.mean().unstack()
            if len(run_mean)!=32:
                raise AssertionError('expected 32 outer test runs across folds')
            rng=np.random.default_rng(2026); effects=[]
            base=run_mean.history_only_tuned.to_numpy()
            for condition in ['history_static_s2_controlled','history_static_s2_tuned']:
                meta=run_mean[condition].to_numpy();draw=rng.integers(0,len(base),(2000,len(base)))
                boot=100*(1-meta[draw].mean(axis=1)/base[draw].mean(axis=1))
                effects.append(dict(condition=condition,n_runs=len(base),history_crps_w=base.mean(),metadata_crps_w=meta.mean(),gain_pct=100*(1-meta.mean()/base.mean()),ci_low_pct=np.quantile(boot,.025),ci_high_pct=np.quantile(boot,.975)))
            pd.DataFrame(effects).to_csv(out/'paired_gain_ci.csv',index=False)
            per.groupby('condition')[['crps_w','mae_w','p95_w','p99_w']].mean().to_csv(out/'aggregate_metrics.csv')
            progress.update(status='completed',completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            save(progress_path,progress)
            (out/'report_ko.md').write_text('# Outer 5-fold / inner 3-fold 회고적 평가\n\n'+pd.DataFrame(effects).to_markdown(index=False)+'\n\n32개 run의 seed 평균을 run 균형으로 집계했다. 24개 후보의 q50 screening 후 상위 6개만 full15 평가하는 단계적 검색이다. 원본 session 독립성과 새 데이터 일반화는 미검증이며, CI는 저장 예측의 run 변동만 반영한다. 원자료와 fold별 검증은 각 results/에 있다. 실제 장치 제어는 수행하지 않았다.\n')
            print('completed all five outer folds',flush=True)
        except BaseException as error:
            progress.update(status='failed',error=repr(error))
            save(progress_path,progress)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=HERE/'results/actual_20260926')
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    run(args.source.resolve(),args.out.resolve(),args.prepare_only)
