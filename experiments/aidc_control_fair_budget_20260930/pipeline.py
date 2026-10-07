"""Sequential, restartable independent experiment. Every completed task is atomic."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np
import pandas as pd
import yaml
from control import simulate, metrics, validate

ROOT=Path(__file__).resolve().parent
BLOCKED={'nested_controller_tuning':'Outer-training grouped-validation predictions and fitted model artifacts unavailable. Other outer-fold predictions may include this fold test runs in training; forbidden. No model retraining performed.',
         'independent_group_inference':'Source SHA256 groups available; collection-session independence unverified. CI conditional on source groups only.'}

def atomic_json(p,obj):
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)); tmp.replace(p)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def config_hash(cfg): return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()

def save_parquet(p,d):
    q=p.with_suffix('.tmp.parquet'); d.to_parquet(q,index=False); q.replace(p)

def windows(d,cfg):
    n=round(cfg['window_s']/cfg['dt'])
    for (run,segment),g in d.groupby(['run_id','segment'],sort=True):
        g=g.sort_values('issuance_s')
        # Never bridge missing timestamps, even within an original segment.
        cuts=np.r_[0,np.flatnonzero(~np.isclose(np.diff(g.issuance_s),cfg['dt'],atol=1e-7))+1,len(g)]
        for block,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
            for k in range((b-a)//n):
                yield run,segment,f'{block}_{k}',g.iloc[a+k*n:a+(k+1)*n].copy()

def inputs(cfg):
    src=Path(cfg['source'])
    paths=[src/x for x in ('prepared.parquet','split_manifest.json','config_snapshot.yaml','control_metrics.csv','completion_status.json','per_target_predictions.parquet','selected_hyperparameters.csv')]
    paths+=sorted((src/'prediction_parts').glob(f'f*_h{cfg["horizon_bins"]}_s{cfg["seed"]}_P[24].parquet'))
    paths+=sorted(src.glob(f'selected_f*_h{cfg["horizon_bins"]}.json'))
    paths+=sorted((src.parent.parent/'src'/'aidc_exp').glob('*.py'))
    return paths

def audit(cfg,out,progress):
    src=Path(cfg['source']); split=json.loads((src/'split_manifest.json').read_text())['folds']
    parts=[]; checks=[]
    for s in split:
        assert not set(s['train_runs'])&set(s['test_runs'])
        for model in ('P2','P4'):
            p=src/'prediction_parts'/f'f{s["fold"]}_h{cfg["horizon_bins"]}_s{cfg["seed"]}_{model}.parquet'
            d=pd.read_parquet(p)
            assert set(d.run_id)<=set(s['test_runs'])
            assert set(d.fold)=={s['fold']} and set(d.model)=={model} and set(d.seed)=={cfg['seed']}
            assert np.allclose(d.target_time_s-d.issuance_s,cfg['horizon_bins']*cfg['dt'])
            assert np.allclose(d.target_start_s-d.issuance_s,(cfg['horizon_bins']-1)*cfg['dt'])
            assert np.all(d.availability_s<=d.issuance_s)
            assert not d.duplicated(['run_id','segment','bin']).any()
            parts.append(d[['fold','group_id','run_id','segment','bin','issuance_s','current_w','actual_w','q0.5','model']])
            checks.append({'fold':s['fold'],'model':model,'rows':len(d),'test_disjoint':True,'alignment':True})
    p=pd.concat(parts); keys=['fold','group_id','run_id','segment','bin','issuance_s']
    a=p[p.model=='P2'].drop(columns='model').rename(columns={'q0.5':'P2'})
    b=p[p.model=='P4'].drop(columns='model').rename(columns={'q0.5':'P4','current_w':'current_w_check','actual_w':'actual_w_check'})
    d=a.merge(b,on=keys,validate='one_to_one')
    assert len(d)==len(a)==len(b)
    assert np.allclose(d.current_w,d.current_w_check) and np.allclose(d.actual_w,d.actual_w_check)
    assert np.isfinite(d[['current_w','actual_w','P2','P4']]).all().all()
    prepared=pd.read_parquet(src/'prepared.parquet',columns=['run_id','segment','bin','issuance_s','current_w','eligible_history',f'y_{cfg["horizon_bins"]}'])
    # Verify actual values against prepared future targets, without feeding them to C2/C3/C4.
    merged=d.merge(prepared[['run_id','segment','bin',f'y_{cfg["horizon_bins"]}']],on=['run_id','segment','bin'],validate='one_to_one')
    assert len(merged)==len(d) and np.allclose(merged.actual_w,merged[f'y_{cfg["horizon_bins"]}'])
    c=pd.read_csv(src/'control_metrics.csv'); e=c.assign(total=c.fast_abs_energy_wh+c.ess_abs_energy_wh).groupby('controller').total.sum()
    uplift=100*(e.C4/e.C2-1)
    pd.DataFrame(checks).to_csv(out/'alignment_oof_checks.csv',index=False)
    atomic_json(out/'source_audit.json',{'legacy_total_throughput_wh':e.to_dict(),'legacy_c4_vs_c2_percent':uplift,'blocked':BLOCKED})
    report=f'''# 구현 감사\n\n기존 결과와 새 실험은 별도 디렉터리에 보존된다. Git repository/.git 및 적용 가능한 AGENTS.md 없음.\n\n- 기존 fast+ESS 절대 처리량: C2 {e.C2:.9f} Wh, C4 {e.C4:.9f} Wh, 차이 {uplift:.9f}%. 전체 segment 합산 비율이며 run-balanced 비율 평균과 다르다.\n- C0/C1은 같은 분기에서 desired=0, GPU 명령 0이다. 측정 GPU cap 반응 자료가 없어 새 실험도 C1=C0이다.\n- 기존 C5는 미래값으로 reference를 이동하는 동일 휴리스틱이다. PCC ramp를 직접 최적화하지 않고 출력 지연도 있어 최선 보장이 없다. perfect-forecast diagnostic으로만 표기.\n- 기존 C3=P2, C4=P4, seed 17, horizon 1.0s, q0.5. 발행 시각의 current_w와 미래 endpoint 예측을 사용한다. queue 길이가 delay_bins+1이라 delay_bins=0도 실제 1 bin 지연이었다.\n- 새 실험은 delay_bins=1을 명시: t에 발행된 명령은 다음 200ms bin에 적용. current_w는 t까지 완료된 bin 값이며 같은 시각 구간의 소급 제어는 없다. 계산·통신·관측 추가 지연은 0, 공통 actuator 지연 200ms. C5만 actual_w 사용.\n- 기존 ESS만 에너지 제약이 있고 fast 저장장치 에너지 상태는 없었다. 종료 SoC 복구도 없다. 새 실험은 두 장치 효율·용량·상하한·출력·slew·장치별 예산을 모두 집행한다.\n- 기존 용량은 outer-training 전력 p95로 정해졌으며 test 전체 분위수를 사용하지 않는다. 새 실험도 이 규칙을 유지한다.\n- 새 정책: 60s 고정 clock window, 40s 제어+20s 복구, 초기 SoC=0.5, 종료=0.5 및 출력=0. 복구 ramp와 처리량을 평가에 포함. 창 종료는 run 종료 예측이 아니다. 잔여 불완전 창은 모든 제어기에서 제외하며 exclusions.csv에 기록한다. 별도 창 경계 ramp는 연결하지 않는다.\n- AC 측 절대 처리량을 장치별 집행. 초기 credit=창 전체 예산, 최대 credit=같은 값, 창 내 추가 credit=0, 이월 없음. 예산=outer-training C2 run별 평균 절대 출력의 평균×60s×배수. 복구 필요 AC 에너지를 예약해 종료 조건 보장. 기준 0이면 예산 0이며 나누지 않는다.\n- 두 장치 slew 한도=Pmax/0.2s는 명시적 E1 가정. p95 기준은 segment별 평균이 아닌 run 내 모든 유효 창의 ramp를 모은 분위수. 창 경계는 연결하지 않음.\n- 기존 P2/P4 예측은 해당 outer test run을 제외하고 학습. 모델 선택은 P2 inner CRPS로 공통 선택된 기존 모델로 고정. outer-training 내부 검증 예측/학습 모델 파일은 발견되지 않음. 다른 fold OOF를 tuning에 쓰면 현재 test run 정보가 유입될 수 있으므로 사용하지 않음. 제어기 tuning 비교는 blocked; 실행 결과는 사전 고정 공통 파라미터 비교이며 tuning 완료로 해석 불가.\n- 기존 평가셋 재사용 분석. SHA256 source group은 독립 수집 session임이 검증되지 않아 cluster CI의 독립성도 제한된다. 2,000회 CI는 고정 예측·정책에 조건부다.\n- E1 actuator 가정의 PCC proxy 시뮬레이션이며 실제 전압·주파수 안정도 증거가 아니다.\n'''
    (out/'implementation_audit.md').write_text(report)
    progress('audit','completed',rows=len(d))
    return split,prepared,d

def calibrate(cfg,out,splits,prepared,progress):
    p=out/'calibration.json'
    if p.exists(): return json.loads(p.read_text())
    result={}
    for s in splits:
        tr=prepared[prepared.run_id.isin(s['train_runs']) & prepared.eligible_history & prepared[f'y_{cfg["horizon_bins"]}'].notna()]
        capacity=cfg['capacity_train_p95_fraction']*float(np.quantile(tr.current_w,.95))
        device={'fast_max_w':capacity/2,'ess_max_w':capacity/2,'fast_capacity_wh':capacity*cfg['fast_capacity_hours'],'ess_capacity_wh':capacity*cfg['ess_capacity_hours']}
        rows=[]
        for run,segment,w,g in windows(tr,cfg):
            demand=g.current_w.to_numpy()
            z=simulate(demand,demand,'C2',cfg,device,[0.,0.],budgeted=False)
            rows.append({'run_id':run,'fast':abs(z.fast_w).mean(),'ess':abs(z.ess_w).mean()})
        assert rows,'No calibration windows'
        means=pd.DataFrame(rows).groupby('run_id')[['fast','ess']].mean().mean()
        result[str(s['fold'])]={'device':device,'rates_w':means.tolist(),'train_runs':s['train_runs'],'calibration_windows':len(rows)}
        progress('calibration','running',fold=s['fold'],rates_w=means.tolist())
    atomic_json(p,result)
    rows=[]
    for fold,x in result.items():
        rows.append({'fold':fold,**x['device'],'fast_budget_rate_w':x['rates_w'][0],'ess_budget_rate_w':x['rates_w'][1],
                     'dt_s':cfg['dt'],'actuator_delay_s':cfg['delay_bins']*cfg['dt'],'eta':cfg['eta'],
                     'initial_and_terminal_soc':cfg['initial_soc'],'soc_min':cfg['soc_min'],'soc_max':cfg['soc_max'],
                     'window_s':cfg['window_s'],'recovery_s':cfg['recovery_s'],'slew_rule':'Pmax/dt','boundary':'AC'})
    pd.DataFrame(rows).to_csv(out/'device_assumptions.csv',index=False)
    progress('calibration','completed')
    return result

def experiment(cfg,out,data,cal,smoke,progress):
    tasks=out/'checkpoints'; tasks.mkdir(exist_ok=True)
    selected=list(windows(data,cfg))
    if smoke: selected=selected[:2]
    assert selected,'No full-length evaluation windows'
    counts={}
    for run,segment,w,g in selected: counts[run]=counts.get(run,0)+len(g)
    pd.DataFrame([{'run_id':run,'available_prediction_bins':len(g),'evaluated_bins':counts.get(run,0),
                   'excluded_bins':len(g)-counts.get(run,0),'reason':'incomplete 60s windows; smoke subset if smoke'} for run,g in data.groupby('run_id')]).to_csv(out/'exclusions.csv',index=False)
    grouped={}
    for item in selected: grouped.setdefault(item[0],[]).append(item)
    total=len(grouped)*len(cfg['controllers'])*len(cfg['budget_multipliers']); done=0
    for run,items in grouped.items():
        fold=str(int(items[0][3].fold.iloc[0])); setting=cal[fold]
        for budget in cfg['budget_multipliers']:
            for mode in cfg['controllers']:
                tag=f'{run}_{mode}_b{budget:g}'; checkpoint=tasks/f'{tag}.json'
                trace=tasks/f'{tag}.parquet'; met=tasks/f'{tag}.csv'
                if checkpoint.exists():
                    record=json.loads(checkpoint.read_text())
                    assert record['trace_hash']==sha(trace) and record['metrics_hash']==sha(met),'Checkpoint corruption'
                    done+=1; continue
                traces=[]; measures=[]
                for _,segment,w,g in items:
                    demand=g.current_w.to_numpy(); forecast=g[{'C3':'P2','C4':'P4','C5':'actual_w'}.get(mode,'current_w')].to_numpy()
                    z=simulate(demand,forecast,mode,cfg,setting['device'],setting['rates_w'],budget)
                    m=metrics(z,cfg,setting['device']); validate(m)
                    assert abs(z.fast_w.iloc[-1])<1e-6 and abs(z.ess_w.iloc[-1])<1e-6
                    ids=dict(run_id=run,group_id=g.group_id.iloc[0],segment=int(segment),window=w,fold=int(fold),controller=mode,budget=budget)
                    measures.append({**ids,**m})
                    for k,v in ids.items(): z[k]=v
                    z['issuance_s']=g.issuance_s.to_numpy(); z['P2']=g.P2.to_numpy(); z['P4']=g.P4.to_numpy(); z['actual_future_w']=g.actual_w.to_numpy()
                    traces.append(z)
                save_parquet(trace,pd.concat(traces,ignore_index=True)); pd.DataFrame(measures).to_csv(met,index=False)
                atomic_json(checkpoint,{'trace_hash':sha(trace),'metrics_hash':sha(met),'windows':len(items)})
                done+=1; progress('experiment','running',completed_tasks=done,total_tasks=total,run_id=run,controller=mode,budget=budget)
    progress('experiment','completed',completed_tasks=done,total_tasks=total)

def aggregate(cfg,out,progress):
    from analysis import aggregate_results
    aggregate_results(cfg,out)
    progress('aggregate','completed')

def report(cfg,out,progress):
    from analysis import figures_and_report
    figures_and_report(cfg,out)
    progress('figures_and_report','completed')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--resume',action='store_true'); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    cfg=yaml.safe_load(args.config.read_text()); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    identity_cfg={**cfg,'smoke':args.smoke}; digest=config_hash(identity_cfg)
    lockdir=ROOT/'outputs'/'.locks'; lockdir.mkdir(parents=True,exist_ok=True)
    lock=open(lockdir/f'{digest}.lock','a+')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Duplicate config already running; refusing.',flush=True); return 75
    stages={}; started=time.time()
    def progress(stage,status,**detail):
        stages[stage]=status
        atomic_json(out/'progress.json',{'status':'running','stage':stage,'stage_status':status,'pid':os.getpid(),'updated_utc':pd.Timestamp.now(tz='UTC').isoformat(),'elapsed_s':time.time()-started,**detail})
        print(json.dumps({'stage':stage,'status':status,**detail},ensure_ascii=False),flush=True)
    def terminate(signum,frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate); signal.signal(signal.SIGINT,terminate)
    code=1; failure=None
    atomic_json(out/'completion_status.json',{'status':'running','all_completed':False,'available_fixed_policy_pipeline_completed':False,'exit_code':None,'pid':os.getpid(),'blocked':BLOCKED})
    try:
        progress('identity','running')
        fingerprint={'config_hash':digest,'inputs':{str(p):sha(p) for p in inputs(cfg)},
                     'code':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))+sorted(ROOT.glob('*.sh'))},
                     'model_identity':'Frozen P2/P4 q0.5 outer-OOF parquet SHA256 + selected hyperparameters; no retraining'}
        previous=out/'provenance.json'
        if previous.exists():
            assert args.resume,'Output already exists; require --resume'
            assert json.loads(previous.read_text())==fingerprint,'Config/input/model/code identity mismatch; use a new output directory'
        else: atomic_json(previous,fingerprint)
        (out/'resolved_config.yaml').write_text(yaml.safe_dump(identity_cfg,sort_keys=False))
        progress('identity','completed')
        split,prepared,data=audit(cfg,out,progress)
        cal=calibrate(cfg,out,split,prepared,progress)
        experiment(cfg,out,data,cal,args.smoke,progress)
        aggregate(cfg,out,progress); report(cfg,out,progress)
        code=0
    except BaseException as e:
        code=int(e.code) if isinstance(e,SystemExit) and isinstance(e.code,int) else 1
        failure=str(e); traceback.print_exc()
    finally:
        state='completed_with_blocked_scope' if code==0 else ('interrupted' if code in (130,143) else 'failed')
        atomic_json(out/'completion_status.json',{'status':state,'all_completed':False,'available_fixed_policy_pipeline_completed':code==0,
                    'smoke':args.smoke,'exit_code':code,'stages':stages,'blocked':BLOCKED,'failure':failure,
                    'existing_results_visualization':'not_run; legacy numeric audit only','pid':os.getpid(),'elapsed_s':time.time()-started})
        atomic_json(out/'progress.json',{'status':state,'exit_code':code,'pid':os.getpid(),'stages':stages,'updated_utc':pd.Timestamp.now(tz='UTC').isoformat()})
        (out/'exit_code.txt').write_text(str(code)+'\n')
    return code

if __name__=='__main__': sys.exit(main())
