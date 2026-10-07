"""Run-balanced aggregation, paired effects and fixed descriptive plots."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown
from data import *
plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'figure.dpi':110})
METRICS=['mae','rmse','signed_bias','p95_abs_error','p99_abs_error','gain']

def metric(g):
    e=g.prediction_w-g.actual_w;a=abs(e)
    return dict(n=len(g),mae=a.mean(),rmse=np.sqrt(np.mean(e**2)),signed_bias=e.mean(),p95_abs_error=np.quantile(a,.95),p99_abs_error=np.quantile(a,.99) if len(g)>=100 else np.nan,gain=(abs(g.delta)-a).mean())

def per_run(pred,region=False):
    rows=[]
    for keys,g in pred.groupby(['cap','model','run_id','collection_day']):
        base=dict(zip(['cap','model','run_id','collection_day'],keys))
        masks={'whole':np.ones(len(g),bool)}
        if region:
            masks.update({r:(g.endpoint_region==r).to_numpy() for r in ['normal','rising','falling']})
            masks['large_change']=g.large_change.to_numpy()
            masks.update({r:(g.magnitude_region==r).to_numpy() for r in ['0-q50','q50-q75','q75-q90','>q90']})
        for r,m in masks.items():
            z=g[m]
            if len(z):rows.append({**base,'region':r,**metric(z)})
            else:rows.append({**base,'region':r,'n':0,**{c:np.nan for c in METRICS}})
    return pd.DataFrame(rows)

def aggregate(per,keys):
    s=per.groupby(keys,dropna=False)[METRICS].mean().reset_index()
    n=per.groupby(keys,dropna=False).agg(n_runs=('run_id','nunique'),n_anchors=('n','sum'),valid_region_runs=('mae','count'),p99_valid_runs=('p99_abs_error','count'),improved_runs=('gain',lambda x:int((x>1e-9).sum())),worsened_runs=('gain',lambda x:int((x<-1e-9).sum()))).reset_index()
    s=s.merge(n,on=keys)
    base=s[s.model=='P'][[c for c in keys if c!='model']+['mae']].rename(columns={'mae':'persistence_mae'})
    s=s.merge(base,on=[c for c in keys if c!='model'],validate='many_to_one')
    s['improvement_pct']=100*(s.persistence_mae-s.mae)/s.persistence_mae.replace(0,np.nan)
    return s

def render(out,pilot,cfg):
    figdir=out/'figures';figdir.mkdir(exist_ok=True);plotdir=out/'plotting_data';plotdir.mkdir(exist_ok=True)
    def save(name,fig,data):
        data.to_csv(plotdir/(name+'.csv'),index=False);fig.tight_layout()
        fig.savefig(figdir/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(figdir/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    frames=[];registry=[];audits=[];anchorhashes={}
    for dest in sorted((out/'fits').iterdir()):
        if not (dest/'complete.json').exists():continue
        rec=json.loads((dest/'complete.json').read_text());z=pd.read_parquet(dest/'predictions.parquet');frames.append(z)
        ident=rec['identity'];anchorhashes.setdefault((ident['cap'],ident['fold']),set()).add(ident['train_anchor_hash'])
        registry.append({**ident,**{k:rec[k] for k in ['train_rows','train_runs','validation_rows','fit_seconds','prediction_seconds','new_fit','source']},'feature_count':len(rec['feature_columns'])})
        audits.append(pd.read_csv(dest/'metadata_audit.csv'))
    assert len(registry)==27 and all(len(h)==1 for h in anchorhashes.values())
    pred=pd.concat(frames,ignore_index=True)
    p=pred[pred.model=='A'].copy();p['model']='P';p['prediction_w']=p.current_w;p['delta_hat']=0.;p['abs_error']=abs(p.delta);p['gain']=0.
    pred=pd.concat([pred,p],ignore_index=True)
    assert set(pred.collection_day)==set(cfg['development_days']) and not pred.duplicated(['cap','model']+KEY).any()
    for cap,g in pred.groupby('cap'):
        hashes={frame_hash(z.sort_values(KEY)[KEY]) for _,z in g.groupby('model')};assert len(hashes)==1
    parquet(out/'predictions.parquet',pred);pd.DataFrame(registry).to_csv(out/'fit_registry.csv',index=False)
    audit=pd.concat(audits,ignore_index=True);audit.to_csv(out/'metadata_audit.csv',index=False)
    per=per_run(pred);per.to_csv(out/'metrics_per_run.csv',index=False)
    summary=aggregate(per,['cap','model']);summary.to_csv(out/'metrics_summary.csv',index=False)
    days=aggregate(per,['cap','model','collection_day']);days.to_csv(out/'metrics_by_day.csv',index=False)
    regions=per_run(pred,True);regions.to_csv(out/'region_metrics_per_run.csv',index=False)
    reg=aggregate(regions,['cap','model','region']);reg.to_csv(out/'region_metrics.csv',index=False)
    paired=[]
    for source,dest,file in [('A','R','paired_target_effect'),('R','RM','paired_metadata_effect')]:
        a=per[per.model==source].drop(columns=['model','region']);b=per[per.model==dest].drop(columns=['model','region'])
        z=a.merge(b,on=['cap','run_id','collection_day'],suffixes=('_source','_dest'),validate='one_to_one')
        for m in METRICS:z[m+'_effect']=z[m+'_source']-z[m+'_dest']
        z.to_csv(out/(file+'.csv'),index=False);paired.append(z)
        z.groupby(['cap','collection_day'])[[m+'_effect' for m in METRICS]].mean().reset_index().to_csv(out/(file+'_by_day.csv'),index=False)
    decomp=[]
    for (cap,model,run,day),g in pred.groupby(['cap','model','run_id','collection_day']):
        for r in ['normal','rising','falling']:
            z=g[g.endpoint_region==r];decomp.append(dict(cap=cap,model=model,run_id=run,collection_day=day,region=r,n=len(z),whole_n=len(g),gain_contribution=z.gain.sum()/len(g)))
    dc=pd.DataFrame(decomp);dc.to_csv(out/'gain_decomposition_per_run.csv',index=False)
    checks=dc.groupby(['cap','model','run_id']).gain_contribution.sum().reset_index().merge(per[['cap','model','run_id','gain']],on=['cap','model','run_id'])
    np.testing.assert_allclose(checks.gain_contribution,checks.gain,atol=1e-8)
    de=dc.groupby(['cap','model','region']).gain_contribution.mean().reset_index();de.to_csv(out/'gain_decomposition.csv',index=False)
    # Direction, zero predictions and correction sizes are descriptions, not gating rules.
    direction=[]
    for keys,g in pred[pred.model!='P'].groupby(['cap','model','run_id','collection_day','endpoint_region']):
        direction.append(dict(zip(['cap','model','run_id','collection_day','region'],keys),n=len(g),mean_abs_delta_hat=abs(g.delta_hat).mean(),median_abs_delta_hat=abs(g.delta_hat).median(),direction_agreement=(np.sign(g.delta)==np.sign(g.delta_hat)).mean() if keys[-1]!='normal' else np.nan,zero_prediction_fraction=(g.delta_hat==0).mean()))
    pd.DataFrame(direction).to_csv(out/'change_diagnostics.csv',index=False)
    # Metadata group tables retain day support and training-condition visibility.
    nested=pd.read_parquet(out/'nested_train_anchors.parquet');splits=json.loads((out/'split_manifest.json').read_text())['folds']
    meta=[]
    for field in META:
        for (cap,value),g in pred[pred.model.isin(['R','RM'])].groupby(['cap',field],dropna=False):
            runrows=[];seen=[]
            for (run,model),v in g.groupby(['run_id','model']):runrows.append({'model':model,'run_id':run,**metric(v)})
            for fold,v in g.groupby('fold'):
                tr=nested[(nested.cap==cap)&nested.collection_day.isin(splits[int(fold)]['train_days'])]
                match=tr[field].isna() if pd.isna(value) else tr[field].eq(value)
                seen.append(bool(match.any()))
            rp=pd.DataFrame(runrows);a=rp.groupby('model')[METRICS].mean()
            meta.append(dict(field=field,value=str(value),cap=cap,run_count=g.run_id.nunique(),date_count=g.collection_day.nunique(),anchor_count=len(g)//2,training_observed_all_folds=all(seen),training_observed_any_fold=any(seen),mae_R=a.loc['R','mae'],mae_RM=a.loc['RM','mae'],metadata_mae_gain=a.loc['R','mae']-a.loc['RM','mae'],metadata_p95_gain=a.loc['R','p95_abs_error']-a.loc['RM','p95_abs_error'],evidence_label='기술 통계; 날짜/조건 교락; 인과 추론 없음'))
    metadata=pd.DataFrame(meta);metadata.to_csv(out/'metadata_group_effects.csv',index=False)
    # Charts use numeric cap distances. All plotted values come from saved tables.
    for metricname,name,label in [('mae','learning_mae','Run-balanced MAE (W)'),('p95_abs_error','learning_p95','Run 평균 p95 절대오차 (W)')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for model,g in summary.groupby('model'):ax.plot(g.cap,g[metricname],marker='o',label=model)
        ax.set(xlabel='Run당 anchor cap',ylabel=label,xticks=[100,300,600]);ax.legend();ax.grid(alpha=.2);save(name,fig,summary)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,(day,g) in zip(axes,days.groupby('collection_day')):
        for model,v in g.groupby('model'):ax.plot(v.cap,v.mae,marker='o',label=model)
        ax.set(title=day,xlabel='Cap',xticks=[100,300,600]);ax.grid(alpha=.2)
    axes[0].set_ylabel('MAE (W)');axes[-1].legend();save('learning_by_day',fig,days)
    me=paired[1].groupby('cap').mae_effect.mean().reset_index();fig,ax=plt.subplots(figsize=(7,4));ax.plot(me.cap,me.mae_effect,'o-');ax.axhline(0,color='gray');ax.set(xlabel='Cap',ylabel='MAE_R − MAE_RM (W)',xticks=[100,300,600]);save('metadata_effect',fig,me)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    mags=['0-q50','q50-q75','q75-q90','>q90']
    for ax,cap in zip(axes,[100,300,600]):
        for model in ['A','R','RM']:
            v=reg[(reg.cap==cap)&(reg.model==model)].set_index('region').reindex(mags);ax.plot(range(4),v.gain,'o-',label=model)
        ax.set(title=f'Cap {cap}',xticks=range(4),xticklabels=mags,ylabel='Persistence 대비 gain (W)');ax.axhline(0,color='gray');ax.tick_params(axis='x',rotation=20)
    axes[-1].legend();save('magnitude_gain',fig,reg[reg.region.isin(mags)])
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,cap in zip(axes,[100,300,600]):
        for j,model in enumerate(['A','R','RM']):
            v=de[(de.cap==cap)&(de.model==model)].set_index('region').reindex(['normal','rising','falling']);ax.bar(np.arange(3)+j*.24,v.gain_contribution,width=.24,label=model)
        ax.set(title=f'Cap {cap}',xticks=np.arange(3)+.24,xticklabels=['정상','상승','하강']);ax.axhline(0,color='gray')
    axes[0].set_ylabel('전체 MAE gain 기여 (W)');axes[-1].legend();save('gain_contribution',fig,de)
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for i,(z,label) in enumerate(zip(paired,['A → R','R → RM'])):
        for ax,cap in zip(axes[i],[100,300,600]):
            v=z[z.cap==cap].sort_values(['mae_effect','run_id']);ax.bar(range(len(v)),v.mae_effect);ax.axhline(0,color='gray');ax.set(title=f'{label}, cap={cap}',xlabel='Run (오차 개선량 순위)',ylabel='source MAE − dest MAE (W)')
    save('paired_run_effect',fig,pd.concat([z.assign(comparison=c) for z,c in zip(paired,['A-R','R-RM'])]))
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    scatter=pred[pred.model.isin(['R','RM'])].copy()
    for i,model in enumerate(['R','RM']):
        for ax,cap in zip(axes[i],[100,300,600]):
            v=scatter[(scatter.model==model)&(scatter.cap==cap)];ax.scatter(v.delta,v.delta_hat,s=2,alpha=.1);lim=max(abs(v.delta).max(),abs(v.delta_hat).max());ax.plot([-lim,lim],[-lim,lim],color='gray');ax.set(title=f'{model}, cap={cap}',xlabel='실제 변화 (W)',ylabel='예측 변화 (W)')
    save('delta_scatter',fig,scatter)
    # Original holdout is separate and never used for choices.
    old=pd.read_parquet(out/'prior_diagnostic_predictions.parquet');fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,((scope,model),v) in zip(axes.flat,old.groupby(['scope','model'])):
        ax.scatter(v.delta,v.delta_hat,s=2,alpha=.1);ax.set(title=f'{scope}: {model}',xlabel='실제 변화 (W)',ylabel='예측 변화 (W)')
    save('prior_delta_scatter',fig,old)
    prior=pd.read_csv(out/'prior_diagnostic_per_run.csv');prior.groupby(['analysis_label','model','region'])[['mae','gain','mean_abs_predicted_change','median_abs_predicted_change','direction_agreement','zero_prediction_fraction']].mean().reset_index().to_csv(out/'prior_diagnostic_summary.csv',index=False)
    # Cap300 metadata paired rank chooses best, middle, worst. Continuous original bins.
    order=paired[1][paired[1].cap==300].sort_values(['mae_effect','run_id'],ascending=[False,True]).reset_index(drop=True)
    selected=order.iloc[[0,len(order)//2,len(order)-1]].copy();selected['case']=['개선','중간','악화'];selected.to_csv(out/'case_selection.csv',index=False)
    full=pd.read_parquet(pilot/'continuous_features.parquet');ev=pd.read_parquet(out/'evaluation_anchors.parquet');caseframes=[];inference_s=0.
    for row in selected.itertuples():
        anchors=ev[ev.run_id==row.run_id].sort_values(['issuance_s','segment','bin']);first=anchors.iloc[0]
        z=full[(full.run_id==row.run_id)&(full.segment==first.segment)&full.valid_anchor&(full.issuance_s>=first.issuance_s)&(full.issuance_s<=first.issuance_s+30)].sort_values('issuance_s').copy()
        fold=next(s['fold'] for s in splits if row.collection_day in s['validation_days'])
        z['P']=z.current_w
        for model in ['R','RM']:
            dest=out/'fits'/f'300_{fold}_{model}';rec=json.loads((dest/'complete.json').read_text());cols=rec['feature_columns'];tr=nested[(nested.cap==300)&nested.collection_day.isin(splits[fold]['train_days'])]
            _,x,_,_=encode(tr,z,cols,mapping=rec['encoding']);t=time.perf_counter();z[model]=lgb.Booster(model_file=str(dest/'q50.txt')).predict(x,num_threads=4)+z.current_w;inference_s+=time.perf_counter()-t
        z['target_timestamp']=z.issuance_s+1;z['case']=row.case
        fig,ax=plt.subplots(figsize=(10,4))
        # Break curves at missing bins and segment boundaries.
        blocks=(~np.isclose(z.issuance_s.diff().fillna(.2),.2)).cumsum()
        for col,label in [('actual_w','실제 전력'),('P','Persistence'),('R','R'),('RM','RM')]:
            for j,(_,v) in enumerate(z.groupby(blocks)):ax.plot(v.target_timestamp,v[col],label=label if j==0 else None,alpha=.8)
        ax.set(title=f'{row.case} 사례: {row.run_id} (사후 설명용)',xlabel='Target bin 끝 시각 (run 상대 초)',ylabel='GPU 합산 전력 (W)');ax.legend();save('case_'+row.case,fig,z[KEY+['target_timestamp','actual_w','P','R','RM','case']]);caseframes.append(z)
    atomic_json(out/'case_inference_timing.json',{'prediction_seconds':inference_s,'fits':0})
    # Hypotheses are evaluated from the three development days, without assumed improvement.
    target=paired[0];metadata_pair=paired[1]
    target_all=bool((target.groupby('cap').mae_effect.mean()>0).all());target_days=bool((target.groupby(['cap','collection_day']).mae_effect.mean()>0).all())
    r=summary[summary.model=='R'].sort_values('cap');monotone=bool((np.diff(r.mae)<=0).all());rday=days[days.model=='R']
    normal=de[(de.model=='R')&(de.region=='normal')].gain_contribution;rest=de[(de.model=='R')&(de.region!='normal')].groupby('cap').gain_contribution.sum()
    meta_mean=metadata_pair.groupby('cap').mae_effect.mean();meta_tail=metadata_pair.groupby('cap').p95_abs_error_effect.mean();meta_day=metadata_pair.groupby(['cap','collection_day']).mae_effect.mean()
    hypotheses=[
        dict(hypothesis='H1',result='지지' if target_all and target_days else '혼합' if target_all else '지지하지 않음',evidence='A−R 평균 차이: '+str(target.groupby('cap').mae_effect.mean().round(3).to_dict()),limitation='동일 개발 자료로 history 5초를 선택한 후속 탐색'),
        dict(hypothesis='H2',result='지지' if monotone else '혼합',evidence='R MAE: '+str(dict(zip(r.cap,r.mae.round(3)))),limitation='cap 증가는 동일 run 내 시점 추가; 날짜별 방향 확인 필요'),
        dict(hypothesis='H3',result='지지' if (normal<0).all() and (rest>0).all() else '혼합',evidence='정상/상승/하강 gain 기여도 표',limitation='미래 label 사후 구간; 정상 정의가 Persistence에 유리'),
        dict(hypothesis='H4',result='혼합' if ((meta_mean>0)!=(meta_tail>0)).any() else '지지' if (meta_mean>0).all() and (meta_tail>0).all() else '지지하지 않음',evidence='평균 MAE gain '+str(meta_mean.round(3).to_dict())+'; p95 gain '+str(meta_tail.round(3).to_dict()),limitation='p95는 run별 quantile 평균; p99 100행 이상 run만'),
        dict(hypothesis='H5',result='지지' if meta_day.min()<0<meta_day.max() else '혼합',evidence='날짜 및 metadata 조건별 효과 표',limitation='날짜/hardware/workload 교락, 독립 수집 session 미확인; 인과효과 아님')]
    h=pd.DataFrame(hypotheses);h.to_csv(out/'hypothesis_results.csv',index=False)
    verification={'nested_sets':True,'counts_equal_min_cap_candidates':True,'pilot_300_sampler_restored_identity':True,'no_duplicates':True,'models_share_anchors':True,'weights_equal_total_per_run':True,'evaluation_matches_pilot_development':True,'thresholds_match_pilot':True,'decomposition_sum_matches_whole_gain':True,'max_decomposition_error_w':float(abs(checks.gain_contribution-checks.gain).max()),'new_training_fits':27,'reused_pilot_fits':0,'smoke_fits_excluded':1,'no_holdout_new_inference':True,'source_prediction_metrics_verified':True,'p99_minimum_n':100,'all_training_sequential':True,'figures_png_svg_pairs':len(list(figdir.glob('*.png'))),'all_plotting_data_saved':True}
    atomic_json(out/'verification.json',verification)
    def table(d):return d.to_markdown(index=False,floatfmt='.3f')
    def read(n):return pd.read_csv(out/n)
    text=['# AIDC residual·metadata 후속 개발 실험',
          '완료된 개발 OOF 후속 탐색이다. 기존 pilot의 결론을 재사용하지 않고 동일 평가 anchor의 새 결과로 판단했다. 제어·MPC·저장장치·계통 시뮬레이션은 실행하지 않았다.',
          '## 핵심 결과',table(h),
          '## 경로·자료·분할', '실제 경로는 resolved_paths.json에 기록했다. Git 저장소가 없으며 적용되는 AGENTS.md도 발견되지 않았다. 원본과 사용자 변경은 보존했다. 기존 pilot 완료 상태·config·feature·모델·prediction을 확인했고 기존 저장 MAE를 원본 per-run 표와 대조했다.',
          '표준화 GPU 전력의 완료 200 ms bin에서 발행한다. Target은 (t+0.8,t+1.0] 평균 전력이다. 모델 feature는 현재 포함 25 bins 이내이며, 후보는 과거 50 bins와 미래 5 bins의 연속 유효성을 요구한다. 10초 eligibility는 pilot 비교 표본 일관성 조건이다. 보간 없이 segment 내 최소 1초 간격 후보, 30개 미만 run 제외 조건을 유지했다.',
          table(pd.DataFrame(json.loads((out/'split_manifest.json').read_text())['folds'])),
          table(read('run_inventory.csv')[['run_id','collection_day','hardware','workload','spaced_candidates','included','exclusion_reason']]),
          '## 중첩 학습 표본',
          '기존 300 manifest를 정확히 보존했다. 시간순 후보의 10 strata를 복원해 pilot sampler·seed identity를 검증했다. 100은 기존 선택 안에서 stratum당 우선 10개, 600은 기존 선택을 보존하며 우선 60개를 배분한다. 부족분은 시간순 strata 순회로 하나씩 배분한다. SHA256(17,run,stratum,anchor key) rank를 사용하고 label은 선택에 쓰지 않는다. 모든 count=min(cap,M), 100⊆300⊆600, 중복 없음·후보 소속·모델 anchor hash·run 균등 가중치를 assertion으로 검증했다.',
          table(read('sampling_summary.csv')[['collection_day','run_id','cap','candidates','actual_rows']]),
          table(pd.DataFrame(registry)[['cap','fold','model','train_rows','train_runs','feature_count','fit_seconds','prediction_seconds']]),
          '27회 신규 fit, pilot 재사용 0회. 기존 모델 재사용 대신 새 경로에서 고정 설정을 재학습했다. 별도 smoke는 100행 이하·5 trees·1 fit로 본 성능에서 제외했다. 최종 개발 refit·09-22 신규 평가·새 seed·ablation·hyperparameter 탐색을 하지 않았다.',
          '## Metadata 감사',
          'pilot encoding을 유지했다: training fold 범주를 정렬해 float code로 입력하며 범주형 native categorical로 바꾸지 않았다. unseen·missing 범주는 −1, 수치 결측은 NaN으로 유지했다. Run ID·날짜·파일명·경로는 feature에서 제외했다. missing은 추정하지 않는다. 전체 fold/cap/model별 feature·상수·training/validation 범주·결측·unseen은 metadata_audit.csv에 있다.',
          table(audit[(audit.model=='RM')&(audit.cap==300)&audit.column.isin(META)].drop(columns=['training_categories','validation_categories'])),
          table(read('date_hardware_workload.csv')),
          '날짜는 독립 수집 session으로 확인되지 않았다. 날짜/hardware/workload가 교락되며 history도 겹친다. Cap 증가는 기존 run 내 시점 수 증가이지 독립 run·workload 추가가 아니다.',
          '## 개발 OOF 성능',
          '모든 지표는 run별로 계산한 뒤 23개 run에 동일 가중을 주었다. Fold MAE 단순 평균이 아니다. RMSE도 run별 RMSE 평균이다. 개선율은 평균 오차들의 비율이며 run별 비율 평균이 아니다. p95/p99는 run quantile 평균으로 전체 row pooled quantile과 다르다. p99는 평가 100개 미만 run에서 N/A이며 유효 run 수를 병기한다.',
          table(summary),table(days),
          '## Paired target 및 metadata 효과','양의 source−dest는 개선이다. Target 효과 A→R와 metadata 효과 R→RM를 별도 저장했다.',
          table(read('paired_target_effect_by_day.csv')),table(read('paired_metadata_effect_by_day.csv')),
          '## 변화 구간 및 오차 기여도',
          '각 fold 기존 300 training의 abs(delta) q50·q75·q90를 고정했다. 정상 a≤q50, 상승 delta>q50, 하강 delta<−q50. 큰 변화 a>q90는 중첩 보조 구간이다. 모든 cap·구성은 같은 threshold·평가 anchor다. 빈 구간은 N/A로 보존했다. 미래 label 구간은 사후 설명이며 oracle gating·선택에는 사용하지 않았다.',
          table(pd.DataFrame(json.loads((out/'thresholds.json').read_text()))),
          table(reg[reg.region.isin(['normal','rising','falling','large_change'])][['cap','model','region','mae','gain','n_anchors','valid_region_runs']]),table(de),
          '각 run의 구간 gain 합 / 전체 평가 표본 수를 run 간 균등 평균했다. 정상+상승+하강 기여 합이 전체 Persistence MAE−모델 MAE와 1e−8 W 이내에서 일치한다. 큰 변화를 다시 더하지 않는다. 방향 일치율은 상승/하강만 계산하고 정확히 0 예측 비율을 별도 보존한다. 방향 일치는 크기 정확성을 의미하지 않는다.',
          table(pd.DataFrame(direction).groupby(['cap','model','region'])[['mean_abs_delta_hat','median_abs_delta_hat','direction_agreement','zero_prediction_fraction']].mean().reset_index()),
          '## Metadata 조건별 효과',
          'Run·날짜·anchor 수와 training 관측 여부를 병기한다. 작은 범주는 기술 통계이며 관찰된 연관을 metadata 인과 효과로 해석하지 않는다. 조건별 모델 오차 차이와 metadata 입력 추가 효과를 구분한다.',table(metadata),
          '## 기존 저장 예측 재분석',
          '새 학습 전에 수행했다. 기존 보류 자료 재분석은 09-22 저장 예측과 저장 threshold만 사용했다. 신규 학습·평가·feature/threshold/cap 선택에 쓰지 않았다. 상세 날짜/run/hardware/workload는 prior_diagnostic_per_run.csv에, anchor별 metadata gain 차이는 prior_metadata_gain_difference.csv에 있다.',table(read('prior_diagnostic_summary.csv')),
          '## 설명용 연속 시계열',
          'Cap300 R→RM run MAE 개선량 순위의 개선·중간·악화 사례다. 동률은 run ID로 처리한다. 첫 유효 evaluation anchor가 있는 첫 segment의 해당 anchor부터 최대 30초, 원본 연속 유효 200 ms bin에 저장 fold 모델로 별도 inference했다. 예측선을 target bin 끝 시각에 배치하고 결측·segment 경계를 연결하지 않는다. 사후 설명용이며 재선택에 사용하지 않았다.',table(selected[['case','run_id','collection_day','mae_effect']]),
          '## 해석과 다음 단계',
          'Residual 대 absolute 효과, Persistence 대비 효과, metadata 효과를 분리해 읽어야 한다. 정상 구간 불필요한 보정 손해와 큰 변화 이득이 함께 존재할 수 있다. 더 많은 시점이 주는 효과는 learning curve로 평가하며 평균 개선만으로 모든 날짜의 성공을 주장하지 않는다. 남은 날짜·조건 의존성은 같은 run의 표본 확대만으로 검증할 수 없다.',
          '다음 단계의 우선순위는 독립 수집 session과 날짜·hardware·workload 다양성 확보다. 이 자료로 history를 선택한 뒤 다시 비교했으므로 독립 최종 검증이 아니다. Metadata 검증은 조건 간 중복과 독립 평가 자료를 확보한 뒤 수행하는 것이 타당하다. 이번 결과만으로 feature 개선이나 모델 확장의 우월성을 판단하지 않는다. 이번 실험 후 추가 학습이나 보류 평가를 자동 실행하지 않는다.',
          '## 검증·시간', 'verification.json 및 timing.json 참조. Fit 시간 합은 전체 wall-clock이 아니다. 데이터/표본 전처리, 모델 학습, 평가 예측, 설명용 inference, 보고서 생성 시간을 별도 저장한다. Report-only는 저장 모델·예측을 이용해 집계와 설명용 inference만 하며 신규 fit 0회다.']
    for file in sorted(figdir.glob('*.png')):text.append(f'![{file.stem}](figures/{file.name})')
    md='\n\n'.join(text);(out/'report_ko.md').write_text(md)
    html=markdown.markdown(md,extensions=['tables','fenced_code'])
    (out/'report_ko.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>AIDC residual metadata follow-up</title><style>body{font-family: NanumGothic,sans-serif;margin:32px;line-height:1.6}img{max-width:100%;height:auto}table{display:block;overflow-x:auto;border-collapse:collapse;max-width:100%}td,th{padding:6px;border:1px solid #ddd}h1,h2{line-height:1.3}</style>'+html+'</html>')
