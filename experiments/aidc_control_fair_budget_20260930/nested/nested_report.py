"""Run-balanced paired statistics with explicitly assumed collection clusters."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis import aggregate_results

METRICS=['p95_ramp','max_ramp','fast_throughput_wh','ess_throughput_wh','fast_mean_abs_w','ess_mean_abs_w']

def paired_cluster(a,b,metric,groups,rng,reps):
    diff=a[metric]-b[metric]
    labels=groups.reindex(diff.index)
    assert labels.notna().all() and diff.notna().all()
    d=pd.DataFrame({'difference':diff,'group':labels})
    grouped=d.groupby('group').difference.agg(['sum','count','mean'])
    n=len(grouped)
    lo=hi=np.nan
    if n>=2:
        draws=rng.integers(0,n,(reps,n))
        values=grouped['sum'].to_numpy()[draws].sum(axis=1)/grouped['count'].to_numpy()[draws].sum(axis=1)
        lo,hi=np.quantile(values,[.025,.975])
    row={'mean_difference':float(diff.mean()),'ci_low':lo,'ci_high':hi,'n_runs':len(diff),'n_assumed_groups':n,
         'n_verified_independent_groups':0,'replicates':reps if n>=2 else 0,
         'status':'assumption_based_only' if n>=2 else 'insufficient_groups_no_ci',
         'estimand':'run-balanced mean paired difference; resampled cluster sums / resampled run counts',
         'improved':int((diff < -1e-8).sum()),'worsened':int((diff > 1e-8).sum()),'tied':int((abs(diff)<=1e-8).sum())}
    logo=[]; group_rows=[]
    for name,g in grouped.iterrows():
        remaining=d[d.group!=name].difference
        logo.append({'omitted_group':name,'remaining_runs':len(remaining),'leave_one_group_out_difference':float(remaining.mean()) if len(remaining) else np.nan})
        group_rows.append({'group':name,'n_runs_in_group':int(g['count']),'group_mean_difference':g['mean']})
    return row,logo,group_rows


def build_results(cfg,out,manifest,log):
    runs=[]
    for policy in ('fixed_policy','inner_tuned'):
        aggregate_results(cfg,out/policy)
        d=pd.read_csv(out/policy/'per_run_metrics.csv');d['policy']=policy;runs.append(d)
        log('aggregation',policy=policy,rows=len(d))
    d=pd.concat(runs,ignore_index=True);d.to_csv(out/'per_run_metrics.csv',index=False)
    device=[]
    for p in sorted((out/'calibration').glob('o*_outer/device_assumptions.csv')):
        x=pd.read_csv(p);x['corrected_outer_fold']=int(p.parent.name.split('_')[0][1:]);device.append(x)
    pd.concat(device).to_csv(out/'device_assumptions.csv',index=False)
    allmetrics=[];checks=[]
    for policy in ('fixed_policy','inner_tuned'):
        for fname, dest in [('fair_budget_metrics.csv',allmetrics),('constraint_checks.csv',checks)]:
            x=pd.read_csv(out/policy/fname);x['policy']=policy;dest.append(x)
    pd.concat(allmetrics).to_csv(out/'fair_budget_metrics.csv',index=False)
    pd.concat(checks).to_csv(out/'constraint_checks.csv',index=False)
    summary=d.groupby(['policy','controller','budget'])[METRICS].mean().reset_index()
    summary.to_csv(out/'budget_summary.csv',index=False)
    legacy=pd.read_csv(Path(cfg['fixed_policy_dir'])/'per_run_metrics.csv');legacy['policy']='fixed_policy_legacy';legacy['split_scope']='original 5 run folds; not matched to corrected day splits'
    legacy.to_csv(out/'fixed_policy_legacy_reference.csv',index=False)
    keys=['run_id','controller','budget'];a=d[d.policy=='inner_tuned'].set_index(keys);b=d[d.policy=='fixed_policy'].set_index(keys).reindex(a.index)
    assert not b.p95_ramp.isna().any()
    changes=(a[METRICS]-b[METRICS]).reset_index();changes['comparison']='inner_tuned-minus-fixed_policy; identical corrected split/predictor/devices/windows'
    changes.to_csv(out/'tuning_changes_per_run.csv',index=False)
    changes.groupby(['controller','budget'])[METRICS].mean().reset_index().to_csv(out/'tuning_changes.csv',index=False)
    rng=np.random.default_rng(cfg['bootstrap_seed']);stats=[];logos=[];individual=[]
    mapping=manifest.set_index('run_id').assumed_group_id
    for policy in ('fixed_policy','inner_tuned'):
        for budget in cfg['budget_multipliers']:
            sub=d[(d.policy==policy)&(d.budget==budget)]
            for comparator in ('C2','C3'):
                a=sub[sub.controller=='C4'].set_index('run_id');b=sub[sub.controller==comparator].set_index('run_id').reindex(a.index)
                assert len(a)==len(b) and not b.p95_ramp.isna().any()
                for scheme in ('source_recording_optimistic','collection_day_assumption','single_campaign_conservative'):
                    groups=pd.Series(a.index,index=a.index) if scheme=='source_recording_optimistic' else (mapping if scheme=='collection_day_assumption' else pd.Series('single_campaign',index=a.index))
                    for metric in METRICS:
                        ids={'policy':policy,'budget':budget,'comparison':f'C4-{comparator}','metric':metric,'grouping':scheme,'result_family':'grouping_sensitivity'}
                        r,l,g=paired_cluster(a,b,metric,groups,rng,cfg['bootstrap_replicates'])
                        stats.append({**ids,**r});logos.extend([{**ids,**x} for x in l]);individual.extend([{**ids,**x} for x in g])
    stats=pd.DataFrame(stats);stats.to_csv(out/'grouping_sensitivity.csv',index=False)
    day=stats[stats.grouping=='collection_day_assumption'];day.to_csv(out/'paired_comparisons.csv',index=False)
    for policy in ('fixed_policy','inner_tuned'):
        day[day.policy==policy].to_csv(out/policy/'paired_comparisons.csv',index=False)
    pd.DataFrame(logos).to_csv(out/'leave_one_group_out.csv',index=False)
    pd.DataFrame(individual).to_csv(out/'individual_group_differences.csv',index=False)
    (out/'independent_group_statistics.json').write_text(json.dumps({'status':'blocked_independence_evidence','formal_ci_produced':False,'verified_independent_groups':0,
              'assumed_grouping_results':'grouping_sensitivity.csv','run_balanced_estimand_preserved':True,'replicates':cfg['bootstrap_replicates']},indent=2))
    scores=pd.concat([pd.read_csv(p) for p in sorted((out/'tuning').glob('outer*/validation_scores.csv'))]);scores.to_csv(out/'validation_scores.csv',index=False)
    selected=[]
    for p in sorted((out/'tuning').glob('outer*/selected.json')):
        s=json.loads(p.read_text())['selected']
        for budget,controllers in s.items():
            for mode,c in controllers.items(): selected.append({'outer_fold':int(p.parent.name[5:]),'budget':float(budget),'controller':mode,'candidate':c['candidate'],**c['parameters']})
    pd.DataFrame(selected).to_csv(out/'selected_parameters.csv',index=False)
    log('statistics_completed_assumption_only',formal_independent_ci='blocked')


def plots_and_report(cfg,out,manifest,smoke):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    runs=pd.read_csv(out/'per_run_metrics.csv');summary=pd.read_csv(out/'budget_summary.csv');paired=pd.read_csv(out/'paired_comparisons.csv')
    ng=runs.group_id.nunique();nr=runs.run_id.nunique()
    caption=f'Reused evaluation set | evaluated runs={nr}, assumed day groups={ng}, verified independent groups=0\nCI: 2,000 day-cluster bootstrap, run-balanced estimand; assumption-based, not formal independent-group validation'
    def save(fig,name):
        fig.text(.5,.01,caption,ha='center',fontsize=7)
        fig.tight_layout(rect=(0,.07,1,1));fig.savefig(figdir/f'{name}.png',dpi=160);fig.savefig(figdir/f'{name}.pdf');plt.close(fig)
    colors={'C2':'tab:blue','C3':'tab:orange','C4':'tab:green'}
    fig,ax=plt.subplots(figsize=(9,5))
    for (policy,mode),g in summary.groupby(['policy','controller']):
        ax.plot(g.budget,g.p95_ramp,marker='o',linestyle='--' if policy=='fixed_policy' else '-',color=colors[mode],label=f'{policy} {mode}')
    ax.set(xlabel='Separate-device budget multiplier',ylabel='Run-balanced p95 |PCC ramp| (W/s)',title='Matched corrected splits and predictors');ax.legend(ncol=2,fontsize=8);save(fig,'budget_p95_fixed_vs_tuned')
    for metric in ('p95_ramp','max_ramp'):
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,name in zip(axes,('fast','ess')):
            for (policy,mode),g in summary.groupby(['policy','controller']):
                ax.plot(g[f'{name}_mean_abs_w'],g[metric],marker='o',linestyle='--' if policy=='fixed_policy' else '-',color=colors[mode],label=f'{policy} {mode}')
            ax.set(xlabel=f'{name} actual Wh/h (mean |W|)',ylabel=metric+' W/s');ax.legend(fontsize=7,ncol=2)
        save(fig,f'actual_device_usage_{metric}')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,comp in zip(axes,('C4-C2','C4-C3')):
        for policy in ('fixed_policy','inner_tuned'):
            g=paired[(paired.comparison==comp)&(paired.policy==policy)&(paired.metric=='p95_ramp')].sort_values('budget')
            ax.plot(g.budget,g.mean_difference,'o-',label=policy)
            ax.fill_between(g.budget.to_numpy(),g.ci_low.to_numpy(),g.ci_high.to_numpy(),alpha=.15)
        ax.axhline(0,color='black',lw=.7);ax.set(title=comp,xlabel='Budget multiplier',ylabel='Paired p95 difference W/s');ax.legend()
    save(fig,'paired_p95_day_assumption_ci')
    fig,axes=plt.subplots(2,1,figsize=(12,8))
    p=runs[(runs.policy=='inner_tuned')&(runs.budget==1)]
    for ax,comp in zip(axes,('C2','C3')):
        a=p[p.controller=='C4'].set_index('run_id');b=p[p.controller==comp].set_index('run_id').reindex(a.index)
        delta=(a.p95_ramp-b.p95_ramp).sort_index()
        ax.bar(delta.index.str[-8:],delta.values);ax.set(title=f'inner_tuned C4-{comp}',ylabel='p95 difference W/s');ax.tick_params(axis='x',rotation=90);ax.axhline(0,color='black',lw=.7)
    save(fig,'paired_run_differences')
    sens=pd.read_csv(out/'grouping_sensitivity.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,comp in zip(axes,('C4-C2','C4-C3')):
        g=sens[(sens.policy=='inner_tuned')&(sens.budget==1)&(sens.comparison==comp)&(sens.metric=='p95_ramp')].reset_index(drop=True)
        for i,r in g.iterrows():
            ax.scatter(i,r.mean_difference)
            if np.isfinite(r.ci_low): ax.vlines(i,r.ci_low,r.ci_high)
            ax.annotate(f'G={int(r.n_assumed_groups)}'+(' no CI' if not np.isfinite(r.ci_low) else ''),(i,r.mean_difference),xytext=(5,8),textcoords='offset points',fontsize=8)
        ax.set_xticks(range(len(g)));ax.set_xticklabels(g.grouping,rotation=20,ha='right',fontsize=8);ax.set(title=comp,ylabel='p95 difference W/s');ax.axhline(0,color='black',lw=.7)
    save(fig,'grouping_sensitivity_ci')
    logo=pd.read_csv(out/'leave_one_group_out.csv');lg=logo[(logo.policy=='inner_tuned')&(logo.budget==1)&(logo.grouping=='collection_day_assumption')&(logo.metric=='p95_ramp')]
    fig,ax=plt.subplots(figsize=(10,5))
    for comp,g in lg.groupby('comparison'): ax.plot(g.omitted_group,g.leave_one_group_out_difference,'o-',label=comp)
    ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Omitted assumed day group',ylabel='Remaining run-balanced p95 difference W/s',title='Leave-one-group-out sensitivity');ax.legend();save(fig,'leave_one_day_out')
    # Opposite ends of control benefit among forecast-improved runs, not chosen timestamps.
    a=p[p.controller=='C4'].set_index('run_id');b=p[p.controller=='C3'].set_index('run_id').reindex(a.index)
    pool=a.index[a.forecast_P4_mae<a.forecast_P2_mae]
    if not len(pool): pool=a.index
    order=(a.p95_ramp-b.p95_ramp).reindex(pool).sort_values();cases=[]
    for label,run in [('best_control_delta',order.index[0]),('worst_control_delta',order.index[-1])]:
        traces={m:pd.read_parquet(out/'inner_tuned'/'checkpoints'/f'{run}_{m}_b1.parquet') for m in cfg['controllers']}
        first=traces['C4'].iloc[0];traces={m:z[(z.segment==first.segment)&(z.window==first.window)] for m,z in traces.items()};z=traces['C4'];t=z.issuance_s.to_numpy()
        fig,axes=plt.subplots(6,1,figsize=(12,14),sharex=True)
        axes[0].plot(t,z.baseline_demand_w,label='load');target=t+cfg['dt']*cfg['horizon_bins']
        axes[0].plot(target,z.P2,label='P2 at target');axes[0].plot(target,z.P4,label='P4 at target');axes[0].plot(target,z.actual_future_w,label='actual at target',alpha=.5)
        for m,q in traces.items():
            axes[1].plot(t,q.pcc_w,label=m);axes[2].plot(t,q.fast_w,label=m);axes[3].plot(t,q.ess_w,label=m)
            axes[4].plot(t,q.fast_soc,label=m+' fast');axes[4].plot(t,q.ess_soc,'--',label=m+' ESS')
            axes[5].plot(t,q.fast_credit_wh,label=m+' fast');axes[5].plot(t,q.ess_credit_wh,'--',label=m+' ESS')
        for ax,y in zip(axes,['Load/forecast W','PCC W','fast W','ESS W','SoC','Remaining Wh']):
            ax.set_ylabel(y);ax.legend(ncol=3,fontsize=7);ax.axvspan(t[0]+40,t[-1],alpha=.08,color='gray')
        axes[0].set_title(f'{label}: {run}; first eligible window (case contains one assumed day group)');axes[-1].set_xlabel('Time s')
        save(fig,f'case_{label}');cases.append({'case':label,'run_id':run,'group':a.loc[run,'group_id'],'p95_C4_minus_C3':order[run],
                   'P2_mae':a.loc[run,'forecast_P2_mae'],'P4_mae':a.loc[run,'forecast_P4_mae'],'selection':'forecast-improved run pool, min/max run p95 delta; fallback all runs; first eligible window'})
    pd.DataFrame(cases).to_csv(out/'case_selection.csv',index=False)
    def table(x): return x.to_markdown(index=False,floatfmt='.5g')
    primary=paired[(paired.budget==1)&(paired.policy=='inner_tuned')&paired.metric.isin(['p95_ramp','max_ramp'])]
    statements=[]
    for comp in ('C4-C2','C4-C3'):
        g=primary[(primary.comparison==comp)&(primary.metric=='p95_ramp')].iloc[0]
        mx=primary[(primary.comparison==comp)&(primary.metric=='max_ramp')].iloc[0]
        statements.append(f'- {comp}: p95 차이 {g.mean_difference:.4f} W/s, 가정 기반 95% CI [{g.ci_low:.4f}, {g.ci_high:.4f}]; 최대 ramp 차이 {mx.mean_difference:.4f} W/s. p95 개선 run {g.improved}, 악화 {g.worsened}, 동률 {g.tied}. 음수는 C4가 유리하다.')
    checks=pd.read_csv(out/'constraint_checks.csv');violation=[c for c in checks if 'violation' in c];bad=(checks[violation].abs()>1e-6).any(axis=1).sum()
    changes=pd.read_csv(out/'tuning_changes.csv');selected=pd.read_csv(out/'selected_parameters.csv')
    indi=pd.read_csv(out/'individual_group_differences.csv');indi=indi[(indi.policy=='inner_tuned')&(indi.budget==1)&(indi.grouping=='collection_day_assumption')&(indi.metric=='p95_ramp')]
    models=len(list((out/'predictor_cache').rglob('P*_q*.txt')))
    old=json.loads((out/'group_evidence.json').read_text())
    report=f'''# Nested grouped validation 기반 동일 예산 제어 비교

**기존 평가셋 재사용 분석 · E1 actuator 가정 · {'smoke 제한 표본' if smoke else 'full 실행'}**

## 완료 범위와 남은 항목

내부 OOF 예측 → controller grouped tuning → 수정된 outer 평가 → 표·그림·보고서 경로를 수행했다. 실제 독립 수집 group을 확정할 session/reset/randomization 기록은 없다. collection_group_verification과 independent_group_statistics는 **blocked_independence_evidence**이며 all_completed는 false다. 다른 단계 완료와 정식 독립-group 통계 완료를 혼동하지 않는다. 수집 문서·원본 timestamp 감사 자체는 완료했다.

## 기존 결과 보존과 공정한 비교

원래 fixed_policy 디렉터리 `{cfg['fixed_policy_dir']}`를 읽기 전용으로 보존했다. 새 `fixed_policy/`는 수정된 날짜 split, 새 predictor, 같은 장치·예산·평가창에서 기존 고정 controller를 재평가한 **matched baseline**이다. `inner_tuned/`와의 차이가 controller tuning 전후 비교다. `fixed_policy_legacy_reference.csv`는 원래 결과를 별도로 보관한 참조 표로, 장치 calibration·predictor까지 바뀐 차이를 tuning 효과로 해석하지 않는다. `grouping_sensitivity.csv`는 같은 outer 평가 결과의 통계 group 가정 민감도이며, 다른 그룹 가정마다 정책을 다시 선택한 실험이 아니다.

## 재학습 범위와 누출 방지

- 기존 LightGBM P2/P4 모델 계열, feature 정의, 1초 endpoint target, quantile loss, seed 17만 사용했다. 새 metadata/구조/target/loss 탐색 없음.
- 기존 기본 후보(candidate 0)의 100 trees, 15 leaves, learning rate .07, min_child_samples 30, reg_lambda 0, history 1s를 사전 고정했다. 기존 outer-training 전체 label로 선택된 HP를 inner-validation에 가져오지 않았다. 추가 predictor HP 탐색 없음.
- 원래 15개 quantile을 동일하게 학습하고 정렬 후 q0.5를 사용했다. 생성 모델 파일 수: **{models}**. full 의도 범위는 4 outer × (3 inner + 1 outer) × P2/P4 × 15 quantiles=480 fits다. smoke는 별도 표본·fold 제한이다.
- outer-test 수집일은 inner-training, inner-validation, encoding, budget/calibration에서 제외. 각 inner predictor는 validation 수집일을 제외한 2개 수집일만 학습한다. 분할별 training_sample.parquet, validation_sample.parquet, feature schema, 입력 hash와 모델 checksum을 보존했다.
- 범주 encoding은 해당 training 표본에서만 학습, unseen=-1; numeric 결측은 기존 LightGBM 처리를 유지한다. normalization을 추가하지 않았다. 원래 rolling feature는 run·segment 내 인과적 입력이다.
- 장치 출력·용량 및 C2 기준 예산은 각 inner-training에서만 계산한다. outer 평가에서는 outer-training에서만 계산하고 C2/C3/C4·fixed/inner_tuned 모두 같은 값을 사용한다. threshold 500 W/s, window 60s, recovery 20s는 사전 상수다.
- C2/C3/C4 각각 6개 후보, 같은 budget 5개, 같은 validation 창. 선택 순서는 모든 제약 충족 → run-balanced p95 ramp → mean max ramp → 처리량 → 고정 candidate index다. 후보는 reference slew 250/500/1000 W/s와 slow alpha .075/.15 조합이며 forecast weight는 .5로 고정했다.
- selected.json을 저장한 뒤에만 outer-test predictor 생성·평가를 진행했다. test 결과에 따라 후보·모델·group을 재선택하지 않았다. task/quantile checkpoint 재개는 입력·코드·config·모델 identity 검사 후만 가능하다.

## 수집 group 근거

원본 경로 {old['raw_files_including_aliases']}개, 고유 기록 {old['canonical_runs']}개. 동일 SHA256 별칭 {old['exact_duplicate_aliases']}개는 중복 표본으로 쓰지 않았다. 원본 timestamp 구간 중첩은 {old['overlapping_pairs']}쌍. README는 32개 node training sessions를 설명하지만 명시적인 session/job ID와 독립 초기화 기록이 없다. `collection_session_id`는 원본 기록을 구분하는 추정 ID이며 독립성 확인 ID가 아니다. 확인된 `independent_group_id`는 비워 두었다.

수집일 4개로 보수적으로 묶었다. hardware·model·batch 일치만으로 grouping하지 않았다. 날짜와 hardware/workload가 교락되어 날짜 holdout과 조합 일반화 효과를 분리할 수 없다. 기존 5 outer split은 이 날짜 가정 아래 모두 train/test group을 공유하므로 4 leave-day-out outer split으로 교체했다. 영향받은 inner/outer 예측을 새로 생성했다. 자세한 근거·미확인 사항은 group_manifest.csv, group_audit_ko.md, group_evidence.json에 기록했다.

## 동일 예산에서의 결과

실제 평가 run **{nr}**, 가정 수집일 group **{ng}**, 확인된 독립 group **0**. Run 내 유효 창 ramp를 경계 연결 없이 모아 p95를 계산한 뒤 run별 동일 가중 평균을 구했다. Segment/window p95 평균은 별도 열이다.

{chr(10).join(statements)}

{table(summary[summary.budget==1])}

## Tuning 전후 변화

아래는 같은 새 split·predictor·장치 조건에서 inner_tuned−fixed_policy의 run-balanced 차이다. 개선이 없거나 악화한 결과도 그대로 남겼다.

{table(changes[changes.budget==1])}

{table(selected)}

## 처리량·종료 SoC·최대 ramp

장치별 AC 절대 에너지 처리량 Wh를 분리한다. 동일 예산은 실제 사용량 일치를 뜻하지 않으며 불필요한 충방전으로 맞추지 않았다. 출력·SoC·효율·지연·slew·종료 SoC 조건을 유지했다. 고정 60초 창의 마지막 20초 복구도 ramp 및 처리량에 포함한다. 모든 창에서 종료 SoC=0.5, 종료 출력=0을 확인했다. 1e-6 허용오차를 넘은 제약 위반 행 **{bad}**. 구체적인 종료 에너지 차이와 constraint 활성은 constraint_checks.csv와 fair_budget_metrics.csv를 참조한다.

실제 사용량이 다르므로 같은 사용량에서의 인과적 우위는 판정할 수 없다. 장치별 실제 사용량 대 p95/maximum 그림을 함께 봐야 한다. p95와 최대 ramp는 별개 지표이며 평균 p95 개선이 최대 ramp 악화를 정당화하지 않는다.

{table(pd.read_csv(out/'device_assumptions.csv'))}

## 가정 기반 통계 및 group 집중도

정식 독립-group CI는 만들지 않았다. 아래는 수집일 독립성 가정에 의존한 2,000회 cluster bootstrap이다. group을 재표집할 때 group 합계/포함 run 수로 계산하여 원래 run-balanced estimand를 유지했다. timestamp·segment·seed를 표본으로 부풀리지 않았다. group-balanced 효과로 바꾸지 않았다. 4개 이하 group의 CI는 매우 불안정하며 고정 예측·선택된 정책에 조건부다.

{table(primary[['comparison','metric','mean_difference','ci_low','ci_high','n_runs','n_assumed_groups','status']])}

{table(indi[['comparison','group','n_runs_in_group','group_mean_difference']])}

{table(lg[['comparison','omitted_group','remaining_runs','leave_one_group_out_difference']])}

낙관적 source-recording grouping, 보수적 날짜 grouping, 전체를 한 campaign으로 보는 가정을 grouping_sensitivity.csv에 분리했다. 1개 group에는 CI를 계산하지 않았다. 개별 group 차이와 leave-one-group-out 민감도는 특정 group 집중 여부를 보여주며 독립성 증명은 아니다.

## 예측과 제어 이득 사례

{table(pd.DataFrame(cases))}

선택 규칙: 예측 MAE 개선 run 중 C4−C3 제어 p95 차이의 최소/최대 run, 조건 run이 없으면 전체에서 선택. 각 run의 첫 유효 창을 표시했다. 두 사례가 모두 같은 부호이면 성공/실패 사례가 모두 존재한다고 주장하지 않는다. 예측은 target 시각으로 이동해 그렸고 부하·PCC·출력·SoC·budget 잔량을 정렬했다.

관측 결과와 원인 가설을 구분한다. endpoint 예측 loss와 PCC ramp 목적 불일치, actuator 지연·포화·예산 예약·복구가 예측 개선을 제어 개선으로 전달하지 못할 가능성이 있다. 제약 활성 및 requested/applied 차이는 진단 자료이며 원인별 인과 기여를 증명하지 않는다.

## 남은 미완료 및 해석 한계

독립 수집 group 확인과 그에 기반한 정식 CI는 필요한 수집 기록 부재로 미완료다. 기존 평가셋을 이미 보았으므로 새로운 독립 검증이 아니다. 실제 GPU cap 반응이나 전압·주파수 안정도 개선을 주장하지 않는다. 이번 결과는 기존 simulator의 E1 actuator 가정 내 PCC proxy 비교다.

전체 예산별 표: budget_summary.csv. 상세 tuning 점수: validation_scores.csv. 모든 그림은 figures/에 PNG/PDF로 보존하며 group 수·CI 근거를 표기했다. 최종 단계 상태는 completion_status.json을 기준으로 확인한다.
'''
    (out/'report_ko.md').write_text(report)
