"""Train onset-window classifiers using the same 200 ms step definition as matching."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from core import history_features, static_features
from evaluate_onset import run as evaluate_onset
from train import sample_indices

HERE=Path(__file__).resolve().parent
PREP=HERE.parents[1]/'trace/preprocessing/outputs/forecast_v2_complete_20260916'
SOURCE=HERE/'results/actual_20260926'
DEFAULT=HERE/'results/onset_aligned_20260927'


def event_times(target, threshold_w=100., debounce_s=1.):
    """Use the same eligible completed-bin steps and censoring as evaluate_onset."""
    times={}
    bounds={}
    for segment,part in target.groupby('segment_id',sort=False):
        part=part.sort_values('issue_s')
        t=part.issue_s.to_numpy(float)
        p=part.current.to_numpy(float)
        start=t[0]+1.0
        end=t[-1]
        bounds[segment]=(start,end)
        step=np.diff(p)
        good=np.isclose(np.diff(t),.2,atol=1e-5)
        for direction,sign in [('rise_event',1),('drop_event',-1)]:
            candidate=t[1:][good&(sign*step>=threshold_w)]
            candidate=candidate[(candidate>=start)&(candidate<=end)]
            kept=[]
            for value in candidate:
                if not kept or value-kept[-1]>debounce_s+1e-9:
                    kept.append(float(value))
            times[(segment,direction)]=np.asarray(kept,float)
    return times,bounds


def labels(frame):
    frame=frame.copy()
    for direction in ('rise_event','drop_event'):
        frame[direction+'_onset']=0
    evidence=[]
    keep=np.zeros(len(frame),bool)
    for run_id,group in frame.groupby('run_id',sort=False):
        events,bounds=event_times(group)
        for segment,part in group.groupby('segment_id',sort=False):
            issue=part.issue_s.to_numpy(float)
            start,end=bounds[segment]
            # Both the retrospective onset list and the forecast lead window
            # must be fully observable; these are the evaluator's alarm bounds.
            eligible=(issue>=start)&(issue<=end-1.0)
            keep[part.index.to_numpy()]=eligible
            for direction in ('rise_event','drop_event'):
                onset=events.get((segment,direction),np.empty(0))
                next_index=np.searchsorted(onset,issue,side='right')
                future=np.full(len(issue),np.inf)
                available=next_index<len(onset)
                future[available]=onset[next_index[available]]
                truth=(future<=issue+1+1e-9)&eligible
                frame.loc[part.index,direction+'_onset']=truth.astype(int)
                evidence.append(dict(run_id=run_id,segment_id=segment,direction=direction,
                    n_onsets=len(onset),n_positive_windows=int(truth.sum()),
                    n_eligible_issuances=int(eligible.sum())))
    return frame.loc[keep].reset_index(drop=True),pd.DataFrame(evidence)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    target=SOURCE/'targets.parquet'
    link=out/'targets.parquet'
    if not link.exists():link.symlink_to(target.resolve())
    if link.resolve()!=target.resolve():raise ValueError('Wrong target source')
    frame=pd.read_parquet(target)
    frame=frame.loc[frame.horizon_ms.eq(1000)].reset_index(drop=True)
    frame,evidence=labels(frame)
    evidence.to_csv(out/'onset_label_manifest.csv',index=False)
    split=pd.read_csv(SOURCE/'split_manifest.csv').set_index('group_id').split
    train_pool=np.flatnonzero(frame.group_id.map(split).eq('train'))
    test=np.flatnonzero(frame.group_id.map(split).eq('test'))
    train=sample_indices(frame,train_pool)
    counts=frame.iloc[train].groupby('run_id').size()
    weights=frame.iloc[train].run_id.map(1/counts).to_numpy()*len(counts)
    power_train=np.vstack([history_features(h) for h in frame.iloc[train].history])
    power_test=np.vstack([history_features(h) for h in frame.iloc[test].history])
    s1_train,s1_test=static_features(frame,train,test,
        columns=('gpu_model','workload_type','model_name','batch_size'))
    s2_train,s2_test=static_features(frame,train,test)
    features={'history_only':(power_train,power_test),
        'history_static_s1':(np.column_stack([power_train,s1_train]),np.column_stack([power_test,s1_test])),
        'history_static_s2':(np.column_stack([power_train,s2_train]),np.column_stack([power_test,s2_test]))}
    output=[];metrics=[]
    for direction in ('rise_event','drop_event'):
        truth=frame[direction+'_onset'].to_numpy(int)
        for condition,(xtrain,xtest) in features.items():
            model=lgb.LGBMClassifier(objective='binary',n_estimators=100,num_leaves=15,
                learning_rate=.07,min_child_samples=30,random_state=42,n_jobs=1,verbosity=-1)
            model.fit(xtrain,truth[train],sample_weight=weights)
            probability=model.predict_proba(xtest)[:,1]
            scored=frame.iloc[test][['run_id','group_id','issue_s','target_end_s']].copy()
            scored['condition']=condition;scored['event']=direction
            scored['y_true']=truth[test];scored['probability']=probability
            output.append(scored)
            for run_id,part in scored.groupby('run_id'):
                actual=part.y_true.to_numpy(bool)
                metrics.append(dict(run_id=run_id,direction=direction,condition=condition,
                    n_windows=len(part),n_positive=int(actual.sum()),
                    brier=brier_score_loss(actual,part.probability),
                    pr_auc=average_precision_score(actual,part.probability) if actual.any() and (~actual).any() else np.nan))
    pred=pd.concat(output,ignore_index=True)
    pred.to_parquet(out/'event_predictions.parquet',index=False)
    pd.DataFrame(metrics).to_csv(out/'event_window_metrics.csv',index=False)
    protocol=dict(source='highres 8 GPU sum, 200 ms completed bins',horizon_s=1,
        onset_definition='adjacent completed-bin step of at least 100 W; same direction events within 1 s debounced',
        alarm_threshold=.5,train_rows=len(train),test_rows=len(test),train_runs=frame.iloc[train].run_id.nunique(),
        test_runs=frame.iloc[test].run_id.nunique(),training_seed=42,
        assessment='retrospective reused split; no new independent data')
    (out/'onset_training_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    aggregate=evaluate_onset(out)
    checks=dict(label_event_definition_aligned=True,
        paired_test_rows=bool(pred.groupby(['run_id','issue_s','event']).condition.nunique().eq(3).all()),
        same_truth=bool(pred.groupby(['run_id','issue_s','event']).y_true.nunique().eq(1).all()),
        no_source_group_overlap=not bool(set(frame.iloc[train].group_id)&set(frame.iloc[test].group_id)),
        onset_matching=json.loads((out/'onset_verification.json').read_text())['all_passed'])
    checks['all_passed']=all(checks.values())
    (out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not checks['all_passed']:raise AssertionError(checks)
    report=['# 일치한 200 ms step-onset label의 +1초 경보 평가','',
        'Highres 8 GPU 합산 전력에서 인접한 완료 200 ms bin의 100 W 이상 상승/하강을 onset으로 정의하고 같은 방향 onset을 1초 안에서는 하나로 합쳤다. 발행 이후 1초 내 onset 존재 여부로 분류기를 학습했다. 학습·test는 기존 22/10 run 분할을 재사용했고 세 feature 조건의 행·가중치·100-tree 설정을 고정했다. 경보는 확률 0.5의 상승 episode로 만들고 onset과 1대1 매칭했다.','',
        aggregate.to_markdown(index=False,floatfmt='.3f'),'','이 결과는 앞선 다른 event-window label 모델보다 사건 정의가 평가와 일치한다. 그러나 100 W와 1초 debounce는 물리 event 또는 계통 허용치로 검증된 값이 아니다. 수집일과 GPU×workload 조합이 결합돼 있고 이 holdout은 이미 반복 사용됐다. 센서 갱신·PCC·actuator가 없어 실제 제어 또는 계통 효과를 주장하지 않는다.','',
        'Missed onset과 false alarms/hour를 함께 보고한다. 원자료는 `onset_events.csv`, `onset_alarms.csv`, `event_predictions.parquet`에 있다. 재현: `python trace/grid_stability_forecast/train_onset_aligned.py`.', '']
    (out/'report_ko.md').write_text('\n'.join(report))
    print(aggregate.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT)
    run(parser.parse_args().out)
