"""Bounded forecast-versus-controller diagnosis using immutable nested artifacts."""
import argparse,fcntl,hashlib,json,os,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parent
NESTED=ROOT.parent/'aidc_control_fair_budget_20260930'/'nested'
sys.path.append(str(NESTED));sys.path.append(str(NESTED.parent))
import nested_pipeline as prior
from aidc_exp.training import encode,fit_quantile
from aidc_exp.evaluation import crps_grid,pinball
from core import simulate,metrics,validate
atomic_json=prior.atomic_json;config_hash=prior.config_hash;sha=prior.sha
INPUTS=['persistence','P2','P4','perfect']


def keyframe(d):return hashlib.sha256(pd.util.hash_pandas_object(d,index=False).values.tobytes()).hexdigest()
def pkey(x):return f'{x:g}'
def canonical(inp,weight,ci):return ('reactive',0.,int(ci)) if inp=='persistence' or weight==0 else (inp,float(weight),int(ci))
def save_table(d,p):
    q=p.with_suffix('.tmp.csv');d.to_csv(q,index=False);q.replace(p)


def load_data(cfg):
    src=Path(cfg['source']);nest=Path(cfg['nested_output']);d=pd.read_parquet(src/'prepared.parquet')
    group=pd.read_csv(nest/'group_manifest.csv');d['group_id']=d.run_id.map(group.set_index('run_id').assumed_group_id)
    return d,group,json.loads((nest/'split_manifest.json').read_text())['folds']


def part_data(cfg,data,fold,inner=None):
    nest=Path(cfg['nested_output']);folder=nest/'predictor_cache'/f'outer{fold}'/('outer_test' if inner is None else f'inner{inner}')
    provenance=json.loads((folder/'provenance.json').read_text());done=json.loads((folder/'complete.json').read_text())
    assert sha(folder/'predictions.parquet')==done['prediction_hash']
    pred=pd.read_parquet(folder/'predictions.parquet')
    lookup=data.reset_index().set_index(['run_id','segment','bin'])['index']
    ix=lookup.reindex(pd.MultiIndex.from_frame(pred[['run_id','segment','bin']])).to_numpy()
    assert not pd.isna(ix).any();va=data.loc[ix].copy()
    pred=prior.restore_prediction_index(pred,va)
    assert not set(provenance['train_groups'])&set(provenance['validation_groups'])
    trids=pd.read_parquet(folder/'training_sample.parquet')
    train=data.set_index(['run_id','segment','bin']).loc[pd.MultiIndex.from_frame(trids[['run_id','segment','bin']])].reset_index()
    train['sample_weight']=trids.sample_weight.to_numpy()
    assert sorted(train.run_id.unique())==provenance['train_runs']
    assert set(train.group_id).isdisjoint(set(va.group_id))
    assert provenance['hyperparameters']==cfg['predictor']
    calfolder=nest/'calibration'/(f'o{fold}_outer' if inner is None else f'o{fold}_i{inner}')
    setting=json.loads((calfolder/'calibration.json').read_text())['0']
    return folder,pred,train,va,setting


def audit(cfg,out,data,groups,splits,log):
    nest=Path(cfg['nested_output']);inventory=pd.read_csv(Path(cfg['source'])/'inventory.csv')
    meta=inventory[inventory.dataset.eq('highres_node')].drop_duplicates('run_id').set_index('run_id')
    rows=[];segments=[]
    for run,g in data.groupby('run_id'):
        valid=g[g.eligible_history & g.y_5.notna()];wins=list(prior.base.windows(valid,cfg))
        se=[]
        for seg,x in g.groupby('segment'):
            length=len(x)*cfg['dt'];se.append(length);segments.append({'run_id':run,'segment':seg,'bins':len(x),'duration_s':length,'first_issuance_s':x.issuance_s.min(),'last_issuance_s':x.issuance_s.max()})
        row={'run_id':run,'source_file':meta.loc[run,'source_path'],'raw_duration_s':float(meta.loc[run,'duration_s']),
             'segment_duration_s':json.dumps(se),'max_segment_s':max(se),'feature_samples':int(g.eligible_history.sum()),
             'feature_target_samples':len(valid),'valid_control_windows':len(wins),'control_included':bool(wins),
             'forecast_included':len(valid)>0,'exclusion_reason':'' if wins else 'No complete 60s contiguous eligible-history plus 1s-target window',
             'collection_day':g.group_id.iloc[0]}
        for k in ['hardware','workload','model_size_token','batch_size','image_size','sequence_length','zero_stage']:row[k]=g[k].iloc[0] if k in g else None
        rows.append(row)
    inc=pd.DataFrame(rows);save_table(inc,out/'run_inclusion_manifest.csv');save_table(pd.DataFrame(segments),out/'segment_manifest.csv')
    save_table(inc.groupby(['control_included','hardware','workload','collection_day'],dropna=False).agg(n_runs=('run_id','size'),mean_max_segment_s=('max_segment_s','mean'),mean_samples=('feature_target_samples','mean')).reset_index(),out/'inclusion_characteristics.csv')
    old=pd.read_csv(nest/'budget_summary.csv');primary=old[(old.policy=='inner_tuned')&(old.budget==1)]
    save_table(primary,out/'verified_previous_primary.csv')
    (out/'implementation_audit.md').write_text('''# 시간·제어 구현 감사

관측값 d(t)는 (t−0.2,t] 완료 bin 평균 전력이다. 발행 시각 t의 endpoint f(t)는 (t+0.8,t+1.0] bin 평균이다. t의 명령은 다음 bin (t,t+0.2]에 적용된다. 초 단위 timestamp 차이를 확인하고 실제 diff(timestamp)로 ramp를 계산한다. 단절된 window/segment는 연결하지 않는다. Endpoint 차이/1초는 endpoint 평균 변화량이며 구간 최대 ramp가 아니다.

기존 core의 수식: aim_t = d_t + w(f_t−d_t); reference_t = reference_(t−1)+clip(aim_t−reference_(t−1), ±reference_slew×dt). desired_t=clip(d_t−reference_t, ±(Pfast+Pess)). 한 bin 지연 후 slow_t=slow_(t−1)+alpha(delayed_t−slow_(t−1)); 요청 fast=delayed−slow, ESS=slow. 따라서 w는 **미래 목표 reference로 이동하는 보간 비중**이며 예측 오차 보정 gain이나 최적 제어 이득이 아니다. w=0이면 같은 파라미터의 reactive와 동일하다.

장치별 출력·slew·SoC bounds를 투영하고, 적용 절대 AC 처리량과 목표 SoC까지 복구할 필요 AC 에너지의 합이 남은 credit 이하가 되도록 요청을 축소한다. 초기 credit은 해당 training-only C2 운전 기준의 60초 예산이고 창 내 추가 보충/이월은 없다. 40초 clock에서 공통 복구를 시작한다. 각 step의 실제 charge/discharge와 효율로 SoC를 갱신하며 마지막 값이나 출력을 강제로 덮어쓰지 않는다. 종료 SoC=0.5와 출력=0 및 에너지 수지를 assertion으로 검사한다. 고정 창 경계는 미지의 run 종료를 예측하는 입력이 아니다.

운전 ramp는 도착 sample <40s, 복구 ramp는 도착 sample >=40s에 배정한다. 따라서 운전→복구 경계 ramp는 복구에 포함된다. 전체 분위수는 모든 실제 ramp를 모아 계산하며 구간 분위수 가중합을 쓰지 않는다. 각 구간의 전력 적분은 그 구간 sample을 사용한다.

새 LP controller는 같은 executor에 다음 bin의 장치별 요청을 전달한다. h=1..5의 PCC 인접 차이 절대합 + 0.01×장치 절대출력 합을 최소화한다. 장치별 출력, slew, 효율을 반영한 에너지 bounds와 모든 계획 prefix의 처리량+복구예약 제약을 둔다. 미래 운전구간 끝에서 horizon을 절단하며 실제 복구는 동일한 20초 정책을 따른다. LP charge/discharge 동시 흐름 해는 실행하지 않고 causal zero-request fallback을 사용한다. solver 실패/시간초과도 같은 fallback을 사용하고 실제 executor 제약 검증 및 요청/적용 차이를 기록한다. Solver 시간은 진단값이며 기본 200ms actuator delay를 늘리지 않는 E1 계산시간 가정이다. deadline 초과 비율을 별도 보고한다. Perfect 입력은 진단용이며 어느 controller도 수학적 최적 상한으로 부르지 않는다.

기존 nested partition의 학습표본·encoding·모델 checksum·OOF median을 확인해 재사용한다. 새로운 horizon q50은 동일 train 표본, 동일 feature와 고정 LightGBM 설정으로만 학습한다. 모델 구조·metadata·loss 추가 탐색은 하지 않는다. Threshold 500 W/s는 사전 고정이며 test 오차로 바꾸지 않는다. 실제 미래값은 평가 라벨 및 perfect 진단 입력에서만 사용한다. 정식 독립 group 검증은 여전히 미완료다.
''')
    log('audit_completed',runs=len(inc),forecast_runs=int(inc.forecast_included.sum()),control_runs=int(inc.control_included.sum()))


def quantiles(cfg,data,fold,out,log):
    folder,pred,train,va,setting=part_data(cfg,data,fold)
    cache=out/'cache'/f'quantiles_outer{fold}.parquet';cache.parent.mkdir(exist_ok=True)
    if cache.exists():return pd.read_parquet(cache)
    import lightgbm as lgb
    frame=pred.reset_index(drop=True).copy()
    for model in ('P2','P4'):
        _,xv=encode(train,va,model,cfg['predictor']['history_s']);qs=[]
        for q in cfg['quantiles']:
            path=folder/f'{model}_q{q:g}.txt';assert sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256']
            qs.append(lgb.Booster(model_file=str(path)).predict(xv))
        mat=np.sort(np.column_stack(qs),axis=1)
        assert np.allclose(mat[:,cfg['quantiles'].index(.5)],pred[model],rtol=1e-10,atol=1e-7),'Restored model median differs'
        for j,q in enumerate(cfg['quantiles']):frame[f'{model}_q{q:g}']=mat[:,j]
    for h in cfg['trajectory_horizons']:frame[f'actual_h{h}']=va[f'y_{h}'].to_numpy()
    prior.base.save_parquet(cache,frame);log('quantile_models_reused',outer=fold,rows=len(frame));return frame


def forecast_evaluation(cfg,data,splits,out,log,smoke):
    rows=[];regions=[];metadata=data.drop_duplicates('run_id').set_index('run_id')
    for s in splits:
        f=quantiles(cfg,data,s['fold'],out,log)
        common=set()
        for run,segment,w,g in prior.base.windows(f,cfg):common.update(zip(g.run_id,g.segment,g.bin))
        f['control_common']=[k in common for k in zip(f.run_id,f.segment,f.bin)]
        actual_path=f[['current_w']+[f'actual_h{h}' for h in cfg['trajectory_horizons']]].to_numpy()
        f['large_transition']=abs(np.diff(actual_path,axis=1)/cfg['dt']).max(axis=1)>cfg['event_threshold_w_s']
        change=(f.actual_w-f.current_w)/(cfg['dt']*cfg['horizon_bins'])
        f['direction']=np.where(change>cfg['event_threshold_w_s'],'rise',np.where(change < -cfg['event_threshold_w_s'],'fall','steady'))
        for scope,subset in [('all_forecast',f),('control_common_timestamps',f[f.control_common])]:
            for run,g in subset.groupby('run_id'):
                for model in ('persistence','P2','P4'):
                    q=np.repeat(g.current_w.to_numpy()[:,None],len(cfg['quantiles']),axis=1) if model=='persistence' else g[[f'{model}_q{x:g}' for x in cfg['quantiles']]].to_numpy()
                    y=g.actual_w.to_numpy();median=q[:,cfg['quantiles'].index(.5)];error=median-y
                    def point(mask):
                        e=error[mask];a=abs(e)
                        return {'n':len(e),'mae':float(a.mean()),'rmse':float(np.sqrt(np.mean(e*e))),'signed_bias':float(e.mean()),'p95_abs_error':float(np.quantile(a,.95)),'p99_abs_error':float(np.quantile(a,.99)),
                                'underprediction_fraction':float((e<0).mean()),'overprediction_fraction':float((e>0).mean()),'mean_underprediction_w':float(np.maximum(-e,0).mean()),'mean_overprediction_w':float(np.maximum(e,0).mean())}
                    ids={'scope':scope,'run_id':run,'group_id':g.group_id.iloc[0],'fold':s['fold'],'model':model,'horizon_s':1.,'hardware':metadata.loc[run,'hardware'],'workload':metadata.loc[run,'workload']}
                    row={**ids,**point(np.ones(len(g),dtype=bool)),'crps_grid':float(crps_grid(y,q,cfg['quantiles']).mean()),
                         'q10_q90_coverage':float(((y>=q[:,cfg['quantiles'].index(.1)])&(y<=q[:,cfg['quantiles'].index(.9)])).mean()),
                         'q10_q90_width_w':float((q[:,cfg['quantiles'].index(.9)]-q[:,cfg['quantiles'].index(.1)]).mean()),'distribution':'degenerate point, same quantile-grid CRPS rule' if model=='persistence' else 'sorted 15 marginal quantiles'}
                    for j,a in enumerate(cfg['quantiles']):row[f'pinball_q{a:g}']=float(pinball(y,q[:,j],a).mean())
                    rows.append(row)
                    for region,mask in [(x,g.direction.eq(x).to_numpy()) for x in ['rise','fall','steady']]+[('large_transition',g.large_transition.to_numpy())]:
                        if mask.any():regions.append({**ids,'region':region,**point(mask)})
    save_table(pd.DataFrame(rows),out/'forecast_per_run.csv');save_table(pd.DataFrame(regions),out/'forecast_regions.csv')
    d=pd.DataFrame(rows)
    numeric=[c for c in d.select_dtypes('number') if c not in ['fold','n','horizon_s']]
    save_table(d.groupby(['scope','model'])[numeric].mean().reset_index(),out/'forecast_summary.csv')
    for dim in ['group_id','hardware','workload']:save_table(d.groupby(['scope',dim,'model'])[numeric].mean().reset_index(),out/f'forecast_by_{dim}.csv')


def window_frames(cfg,pred,smoke,validation=False):
    result=list(prior.base.windows(pred,cfg))
    if smoke:result=result[:cfg['smoke_limits']['validation_windows_per_partition'] if validation else cfg['smoke_limits']['test_windows_per_fold']]
    return result


def rollout(cfg,g,setting,inp,weight,ci,trajectory=False,smoke=False):
    local={**cfg,**cfg['controller_candidates'][ci],'forecast_weight':weight}
    if smoke:local['_smoke_mpc_steps']=cfg['smoke_limits']['mpc_solve_steps_per_window']
    demand=g.current_w.to_numpy()
    name={'reactive':'current_w','persistence':'current_w','perfect':'actual_w'}.get(inp,inp)
    forecast=demand if trajectory else g[name].to_numpy()
    path=None
    if trajectory:
        if inp in ('persistence','reactive'):path=np.repeat(demand[:,None],5,axis=1)
        elif inp.endswith('_endpoint'):
            source=inp.removesuffix('_endpoint');column='actual_w' if source=='perfect' else source
            path=np.repeat(g[column].to_numpy()[:,None],5,axis=1)
        else:path=g[[('actual' if inp=='perfect' else inp)+f'_h{h}' for h in range(1,6)]].to_numpy()
        forecast=demand
    z=simulate(demand,forecast,'C2' if trajectory or inp in ('reactive','persistence') else 'C4',local,setting['device'],setting['rates_w'],g.attrs['budget'],trajectory=path)
    m=metrics(z,local,setting['device']);validate(m)
    assert abs(z.fast_w.iloc[-1])+abs(z.ess_w.iloc[-1])<1e-6
    assert np.allclose(np.diff(g.issuance_s),cfg['dt'],atol=1e-7)
    for col in ['issuance_s','current_w','actual_w','P2','P4']:z[col]=g[col].to_numpy()
    z['command_applied_from_issuance_s']=z.issuance_s-cfg['dt']
    return z,m


def tune_task(cfg,data,out,fold,inner,pred,setting,inp,weight,ci,budget,log,smoke):
    canon=canonical(inp,weight,ci);inp,weight,ci=canon
    dest=out/'tuning_cache'/f'o{fold}_i{inner}_{inp}_w{weight:g}_c{ci}_b{budget:g}.json';dest.parent.mkdir(exist_ok=True)
    identity={'prediction':keyframe(pred),'setting':setting,'input':inp,'weight':weight,'candidate':ci,'budget':budget,'smoke':smoke}
    if dest.exists():
        saved=json.loads(dest.read_text());assert saved['identity']==identity;return saved['rows']
    legacy_mode={'reactive':'C2','P2':'C3','P4':'C4'}.get(inp)
    old=Path(cfg['nested_output'])/'tuning'/f'outer{fold}'/f'i{inner}_{legacy_mode}_b{budget:g}_c{ci}.json'
    if not smoke and legacy_mode and (weight==.5 or inp=='reactive') and old.exists():
        record=json.loads(old.read_text());oldid=record['identity']
        assert oldid['prediction_hash']==prior.frame_hash(pred) and oldid['calibration']==setting and oldid['candidate']==cfg['controller_candidates'][ci]
        rows=record['result'];source={'path':str(old),'sha256':sha(old)}
    else:
        byrun={}
        for run,segment,w,g in window_frames(cfg,pred,smoke,True):
            g.attrs['budget']=budget;z,m=rollout(cfg,g,setting,inp,weight,ci)
            x=byrun.setdefault(run,{'group':g.group_id.iloc[0],'ramp':[],'fast':0.,'ess':0.})
            x['ramp'].append(np.diff(z.pcc_w)/np.diff(z.issuance_s));x['fast']+=m['fast_throughput_wh'];x['ess']+=m['ess_throughput_wh']
        rows=[{'run_id':r,'group_id':x['group'],'p95_ramp':float(np.quantile(abs(np.concatenate(x['ramp'])),.95)),'max_ramp':float(abs(np.concatenate(x['ramp'])).max()),'fast_throughput_wh':x['fast'],'ess_throughput_wh':x['ess']} for r,x in byrun.items()]
        source={'computed':True}
    atomic_json(dest,{'identity':identity,'rows':rows,'source':source});log('candidate',outer=fold,inner=inner,input=inp,weight=weight,candidate=ci,budget=budget)
    return rows


def select_policies(cfg,data,s,out,log,smoke):
    result={};all_scores=[];fold=s['fold'];inner=[]
    for i in s['inner']:
        _,pred,_,_,cal=part_data(cfg,data,fold,i['inner_fold']);inner.append((i['inner_fold'],pred,cal))
    for budget in cfg['budget_multipliers']:
        scores={}
        candidates=sorted(set(canonical(inp,w,c) for inp in INPUTS for w in cfg['weights'] for c in range(len(cfg['controller_candidates']))))
        for inp,w,ci in candidates:
            rows=[]
            for idx,pred,setting in inner:rows.extend(tune_task(cfg,data,out,fold,idx,pred,setting,inp,w,ci,budget,log,smoke))
            d=pd.DataFrame(rows);assert not d.run_id.duplicated().any()
            score=(float(d.p95_ramp.mean()),float(d.max_ramp.mean()),float((d.fast_throughput_wh+d.ess_throughput_wh).mean()),w,ci)
            scores[(inp,w,ci)]=score
            all_scores.append({'fold':fold,'budget':budget,'input':inp,'weight':w,'candidate':ci,'validation_p95':score[0],'validation_max':score[1],'validation_throughput':score[2],'validation_runs':len(d),'validation_groups':d.group_id.nunique()})
        result[pkey(budget)]={}
        for inp in INPUTS:
            for procedure,weights in [('B2_tuned',[.5]),('C_joint',cfg['weights'])]:
                pool=set(canonical(inp,w,c) for w in weights for c in range(len(cfg['controller_candidates'])))
                selected=min(pool,key=lambda x:scores[x]);result[pkey(budget)][procedure+'_'+inp]={'canonical_input':selected[0],'weight':selected[1],'candidate':selected[2],'score':scores[selected],'unique_candidates':len(pool)}
    folder=out/'selection';folder.mkdir(exist_ok=True)
    save_table(pd.DataFrame(all_scores),folder/f'outer{fold}_scores.csv');atomic_json(folder/f'outer{fold}_selected.json',result)
    log('selection_frozen',outer=fold,unique_physical_candidates_per_budget=78)
    return result


def evaluated_task(cfg,out,fold,pred,setting,inp,weight,ci,budget,log,smoke,trajectory=False):
    if not trajectory:inp,weight,ci=canonical(inp,weight,ci)
    mode='mpc' if trajectory else 'endpoint';tag=f'{mode}_o{fold}_{inp}_w{weight:g}_c{ci}_b{budget:g}'
    folder=out/'control_cache'/tag;folder.mkdir(parents=True,exist_ok=True)
    check=folder/'complete.json';identity={'prediction':keyframe(pred),'setting':setting,'input':inp,'weight':weight,'candidate':ci,'budget':budget,'smoke':smoke,'mode':mode}
    if check.exists():
        x=json.loads(check.read_text());assert x['identity']==identity
        for name,digest in x['files'].items():assert sha(folder/name)==digest
        return tag
    files={}
    for run,segment,w,g in window_frames(cfg,pred,smoke):
        name=f'{run}_s{segment}_w{w}.parquet';path=folder/name;cp=folder/(name+'.json')
        if cp.exists():assert json.loads(cp.read_text())['sha256']==sha(path);files[name]=sha(path);continue
        g.attrs['budget']=budget
        z,m=rollout(cfg,g,setting,inp,weight,ci,trajectory,smoke)
        for key,value in {'run_id':run,'group_id':g.group_id.iloc[0],'segment':segment,'window':w,'fold':fold,'budget':budget,'task':tag,'input':inp,'weight':weight,'candidate':ci,'controller_family':mode}.items():z[key]=value
        prior.base.save_parquet(path,z);atomic_json(cp,{'sha256':sha(path),'constraint_metrics':m});files[name]=sha(path)
        log('control_window',task=tag,run=run,window=w)
    assert files;atomic_json(check,{'identity':identity,'files':files});return tag


def endpoint_experiments(cfg,data,splits,out,log,smoke):
    design=[]
    for s in splits:
        selected=select_policies(cfg,data,s,out,log,smoke)
        _,pred,_,_,cal=part_data(cfg,data,s['fold'])
        for budget in cfg['budget_multipliers']:
            for inp in INPUTS:
                choices=[('B1_fixed',.5,0)]+[('C_weight',w,0) for w in cfg['weights']]
                for family in ['B2_tuned','C_joint']:
                    best=selected[pkey(budget)][family+'_'+inp];choices.append((family,best['weight'],best['candidate']))
                for family,w,ci in choices:
                    tag=evaluated_task(cfg,out,s['fold'],pred,cal,inp,w,ci,budget,log,smoke)
                    design.append({'family':family,'fold':s['fold'],'budget':budget,'input':inp,'weight':w,'candidate':ci,'task':tag})
        save_table(pd.DataFrame(design),out/'endpoint_design.csv')
    return design


def trajectories(cfg,data,splits,out,log,smoke):
    import lightgbm as lgb
    atomic_json(out/'trajectory_decision.json',{'execute':True,'reason':cfg['trajectory_decision'],'solver':'scipy HiGHS verified','additional_structures_searched':0,'new_horizon_quantiles':[.5]})
    design=[];forecast_rows=[]
    for s in splits:
        for inner in [x['inner_fold'] for x in s['inner']]+[None]:
            folder,pred,train,va,cal=part_data(cfg,data,s['fold'],inner)
            dest=out/'trajectory_cache'/f'o{s["fold"]}'/('outer' if inner is None else f'inner{inner}');dest.mkdir(parents=True,exist_ok=True)
            target=dest/'predictions.parquet';identity={'training':keyframe(train),'validation':keyframe(va),'hp':cfg['predictor'],'seed':cfg['seed'],'horizons':[1,2,3,4],'quantile':.5,'existing_h5_prediction':sha(folder/'predictions.parquet')}
            if (dest/'identity.json').exists():assert json.loads((dest/'identity.json').read_text())==identity
            else:atomic_json(dest/'identity.json',identity)
            if (dest/'complete.json').exists():
                assert json.loads((dest/'complete.json').read_text())['sha256']==sha(target);frame=pd.read_parquet(target)
            else:
                frame=pred.reset_index(drop=True).copy()
                for h in range(1,6):frame[f'actual_h{h}']=va[f'y_{h}'].to_numpy()
                for model in ['P2','P4']:
                    x,xv=encode(train,va,model,cfg['predictor']['history_s']);frame[f'{model}_h5']=frame[model]
                    for h in range(1,5):
                        mp=dest/f'{model}_h{h}_q50.txt';cp=mp.with_suffix('.json')
                        if cp.exists():
                            assert json.loads(cp.read_text())['sha256']==sha(mp);pr=lgb.Booster(model_file=str(mp)).predict(xv)
                        else:
                            m=fit_quantile(x,train[f'y_{h}'],train.sample_weight,.5,cfg['predictor'],cfg['seed']);pr=m.predict(xv)
                            tmp=mp.with_suffix('.tmp');m.booster_.save_model(str(tmp));tmp.replace(mp);atomic_json(cp,{'sha256':sha(mp)})
                        frame[f'{model}_h{h}']=pr;log('trajectory_q50',fold=s['fold'],inner=inner,model=model,horizon=h*.2)
                train[['run_id','group_id','segment','bin','sample_weight']].to_parquet(dest/'training_sample.parquet',index=False)
                prior.base.save_parquet(target,frame);atomic_json(dest/'complete.json',{'sha256':sha(target)})
            if inner is not None:continue
            for run,g in frame.groupby('run_id'):
                for model in ['persistence','P2','P4']:
                    for h in range(1,6):
                        prediction=g.current_w if model=='persistence' else g[f'{model}_h{h}'];e=prediction-g[f'actual_h{h}']
                        forecast_rows.append({'run_id':run,'group_id':g.group_id.iloc[0],'fold':s['fold'],'model':model,'horizon_s':h*.2,'n':len(g),'mae':abs(e).mean(),'rmse':np.sqrt(np.mean(e**2)),'bias':e.mean(),'h5_note':'reused existing sorted-quantile median; h1..4 new q50 only'})
            for budget in cfg['budget_multipliers']:
                for family in ['D_mpc_trajectory','D_mpc_endpoint']:
                    for inp in INPUTS:
                        passed=inp if family=='D_mpc_trajectory' or inp=='persistence' else inp+'_endpoint'
                        tag=evaluated_task(cfg,out,s['fold'],frame,cal,passed,0,0,budget,log,smoke,True)
                        design.append({'family':family,'fold':s['fold'],'budget':budget,'input':inp,'weight':0,'candidate':0,'task':tag})
            save_table(pd.DataFrame(design),out/'trajectory_design.csv');save_table(pd.DataFrame(forecast_rows),out/'horizon_forecast_metrics.csv')
    return design


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--visualize-only',action='store_true');args=ap.parse_args()
    cfg=yaml.safe_load(args.config.read_text());out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    digest=config_hash({**cfg,'smoke':args.smoke});lockdir=ROOT/'outputs'/'.locks';lockdir.mkdir(parents=True,exist_ok=True);lock=open(lockdir/f'{digest}.lock','a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Duplicate config running');return 75
    stages={x:'pending' for x in ['audit','forecast_evaluation','input_swap','weight_experiments','trajectory_experiment','aggregation','visualization','report','render_verification']}
    stages['independent_group_verification']='blocked_missing_collection_evidence';start=time.time();code=1;failure=None
    def update(status):atomic_json(out/'completion_status.json',{'status':status,'all_completed':all(x=='completed' for x in stages.values()),'smoke':args.smoke,'stages':stages,'exit_code':None if status=='running' else code,'failure':failure,'pid':os.getpid(),'elapsed_s':time.time()-start,'visualization_only':args.visualize_only})
    def log(event,**kw):
        obj={'status':'running','event':event,'pid':os.getpid(),'utc':pd.Timestamp.now(tz='UTC').isoformat(),'elapsed_s':time.time()-start,**kw};print(json.dumps(obj,ensure_ascii=False),flush=True);atomic_json(out/'progress.json',obj);update('running')
    def stop(s,f):raise SystemExit(128+s)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        log('identity_check');nest=Path(cfg['nested_output'])
        paths=[Path(cfg['source'])/x for x in ['prepared.parquet','inventory.csv']]
        paths+=list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh'))
        paths+=[nest/x for x in ['provenance.json','split_manifest.json','group_manifest.csv','budget_summary.csv','completion_status.json']]
        paths+=list((nest/'predictor_cache').rglob('*.json'))+list((nest/'predictor_cache').rglob('*.txt'))+list((nest/'predictor_cache').rglob('*.parquet'))+list((nest/'calibration').rglob('calibration.json'))
        paths+=list(NESTED.glob('*.py'))+list(NESTED.parent.glob('*.py'))
        paths+=list((nest/'tuning').rglob('*.json'))+list((ROOT.parent/'aidc_grid_metadata_20260929'/'src'/'aidc_exp').glob('*.py'))
        identity={'config_hash':digest,'files':{str(p):sha(p) for p in sorted(set(paths))}}
        if (out/'provenance.json').exists():
            assert args.resume or args.visualize_only,'Existing output requires resume'
            assert json.loads((out/'provenance.json').read_text())==identity,'Config/input/model/code identity mismatch'
        else:atomic_json(out/'provenance.json',identity)
        (out/'resolved_config.yaml').write_text(yaml.safe_dump({**cfg,'smoke':args.smoke},sort_keys=False))
        data,groups,splits=load_data(cfg)
        if args.smoke:splits=splits[:cfg['smoke_limits']['outer_folds']]
        if not args.visualize_only:
            stages['audit']='running';audit(cfg,out,data,groups,splits,log);stages['audit']='completed'
            stages['forecast_evaluation']='running';forecast_evaluation(cfg,data,splits,out,log,args.smoke);stages['forecast_evaluation']='completed'
            stages['input_swap']=stages['weight_experiments']='running';endpoint_experiments(cfg,data,splits,out,log,args.smoke);stages['input_swap']=stages['weight_experiments']='completed'
            stages['trajectory_experiment']='running';trajectories(cfg,data,splits,out,log,args.smoke);stages['trajectory_experiment']='completed'
            from reporting import aggregate
            stages['aggregation']='running';aggregate(cfg,out,log);stages['aggregation']='completed'
        else:
            assert (out/'control_per_run.csv').exists(),'No completed results to visualize'
            for x in ['audit','forecast_evaluation','input_swap','weight_experiments','trajectory_experiment','aggregation']:stages[x]='existing_results_not_reexecuted'
        from reporting import visualize_report
        stages['visualization']=stages['report']='running';render=visualize_report(cfg,out,args.smoke,log)
        stages['visualization']=stages['report']='completed';stages['render_verification']='completed' if render else 'blocked_renderer'
        code=0
    except BaseException as e:
        failure=str(e);code=e.code if isinstance(e,SystemExit) and isinstance(e.code,int) else 1
        for k in stages:
            if stages[k]=='running':stages[k]='failed_or_interrupted'
        traceback.print_exc()
    finally:
        status='completed_with_blocked_scope' if code==0 else ('interrupted' if code in [130,143] else 'failed')
        update(status);atomic_json(out/'progress.json',{'status':status,'stages':stages,'exit_code':code,'pid':os.getpid(),'elapsed_s':time.time()-start});(out/'exit_code.txt').write_text(str(code)+'\n')
    return code
if __name__=='__main__':sys.exit(main())
