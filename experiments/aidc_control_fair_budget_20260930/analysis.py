import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def aggregate_results(cfg,out):
    parts=sorted((out/'checkpoints').glob('*.csv'))
    windows=pd.concat([pd.read_csv(p) for p in parts],ignore_index=True)
    windows.to_csv(out/'fair_budget_metrics.csv',index=False)
    checkcols=[c for c in windows if any(s in c for s in ('violation','residual','soc_end','end_energy_delta'))]
    windows[['run_id','segment','window','controller','budget']+checkcols].to_csv(out/'constraint_checks.csv',index=False)
    rows=[]
    for p in sorted((out/'checkpoints').glob('*.parquet')):
        z=pd.read_parquet(p); first=z.iloc[0]; keys=['run_id','group_id','fold','controller','budget']
        m={k:first[k].item() if isinstance(first[k],np.generic) else first[k] for k in keys}
        ramps=np.concatenate([np.diff(g.pcc_w)/cfg['dt'] for _,g in z.groupby(['segment','window'])])
        segp=[np.quantile(abs(np.concatenate([np.diff(w.pcc_w)/cfg['dt'] for _,w in g.groupby('window')])),.95) for _,g in z.groupby('segment')]
        sub=windows[(windows.run_id==first.run_id)&(windows.controller==first.controller)&(windows.budget==first.budget)]
        m.update(p95_ramp=np.quantile(abs(ramps),.95),p99_ramp=np.quantile(abs(ramps),.99),max_ramp=abs(ramps).max(),
                 segment_p95_mean=np.mean(segp),window_p95_mean=sub.p95_ramp.mean(),peak_to_peak=np.ptp(z.pcc_w),
                 duration_s=len(z)*cfg['dt'],threshold_exceed_fraction=np.mean(abs(ramps)>cfg['ramp_threshold_w_s']),
                 threshold_exceed_s=np.sum(abs(ramps)>cfg['ramp_threshold_w_s'])*cfg['dt'],
                 forecast_P2_mae=np.mean(abs(z.P2-z.actual_future_w)),forecast_P4_mae=np.mean(abs(z.P4-z.actual_future_w)))
        for name in ('fast','ess'):
            for suffix in ('throughput_wh','loss_wh','budget_wh','unused_wh'):
                m[f'{name}_{suffix}']=sub[f'{name}_{suffix}'].sum()
            for suffix in ('budget_active_fraction','output_active_fraction','soc_active_fraction','request_applied_mae_w'):
                m[f'{name}_{suffix}']=sub[f'{name}_{suffix}'].mean()
            m[f'{name}_mean_abs_w']=m[f'{name}_throughput_wh']*3600/m['duration_s']
            m[f'{name}_max_abs_end_energy_delta_wh']=abs(sub[f'{name}_end_energy_delta_wh']).max()
        rows.append(m)
    runs=pd.DataFrame(rows); runs.to_csv(out/'per_run_metrics.csv',index=False)
    numeric=[c for c in runs.select_dtypes('number') if c not in ('fold','budget')]
    summary=runs.groupby(['budget','controller'])[numeric].mean().reset_index()
    summary.to_csv(out/'budget_summary.csv',index=False)
    rng=np.random.default_rng(cfg['bootstrap_seed']); paired=[]; differences=[]; counts=[]
    primary=runs[runs.budget==1]
    for comparator in ('C2','C3'):
        a=primary[primary.controller=='C4'].set_index(['run_id','group_id'])
        b=primary[primary.controller==comparator].set_index(['run_id','group_id']).reindex(a.index)
        assert not b.p95_ramp.isna().any()
        for metric in ('p95_ramp','max_ramp','fast_throughput_wh','ess_throughput_wh','fast_mean_abs_w','ess_mean_abs_w'):
            diff=a[metric]-b[metric]
            # Resample source groups, retaining all runs of a sampled group.
            groups=list(diff.groupby(level='group_id')); sums=np.array([v.sum() for _,v in groups]); ns=np.array([len(v) for _,v in groups])
            draws=rng.integers(0,len(groups),(cfg['bootstrap_replicates'],len(groups)))
            boot=sums[draws].sum(axis=1)/ns[draws].sum(axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            paired.append({'comparison':f'C4-{comparator}','metric':metric,'mean_difference':diff.mean(),'ci_low':lo,'ci_high':hi,
                           'n_runs':len(diff),'n_source_groups':len(groups),'replicates':cfg['bootstrap_replicates'],
                           'ci_scope':'conditional fixed predictions/policy; source-group independence unverified'})
            counts.append({'comparison':f'C4-{comparator}','metric':metric,'improved':int((diff < -1e-8).sum()),'worsened':int((diff>1e-8).sum()),'tied':int((abs(diff)<=1e-8).sum())})
            for (run,group),value in diff.items(): differences.append({'comparison':f'C4-{comparator}','metric':metric,'run_id':run,'group_id':group,'difference':value})
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame(differences).to_csv(out/'paired_per_run.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'improvement_counts.csv',index=False)
    # No interpolation to claim equal usage. Matched-use subset is explicitly descriptive.
    matched=[]
    for other in ('C2','C3'):
        a=primary[primary.controller=='C4'].set_index('run_id'); b=primary[primary.controller==other].set_index('run_id').reindex(a.index)
        ok=np.ones(len(a),dtype=bool)
        for name in ('fast','ess'):
            x=a[f'{name}_throughput_wh'].to_numpy(); y=b[f'{name}_throughput_wh'].to_numpy()
            ok &= abs(x-y)<=.05*np.maximum(x,y)+1e-10
        matched.append({'comparison':f'C4-{other}','rule':'both devices within 5% max(actual uses), descriptive only',
                        'n_runs':int(ok.sum()),'mean_p95_difference':float((a.p95_ramp-b.p95_ramp)[ok].mean()) if ok.any() else None})
    pd.DataFrame(matched).to_csv(out/'matched_usage_descriptive.csv',index=False)


def figures_and_report(cfg,out):
    figdir=out/'figures'; figdir.mkdir(exist_ok=True)
    summary=pd.read_csv(out/'budget_summary.csv'); runs=pd.read_csv(out/'per_run_metrics.csv')
    paired=pd.read_csv(out/'paired_comparisons.csv'); diff=pd.read_csv(out/'paired_per_run.csv')
    def save(fig,name):
        fig.tight_layout(); fig.savefig(figdir/f'{name}.png',dpi=160); fig.savefig(figdir/f'{name}.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for mode in ('C2','C3','C4'):
        g=summary[summary.controller==mode]; ax.plot(g.budget,g.p95_ramp,'o-',label=mode)
    ax.set(xlabel='Separate-device budget multiplier',ylabel='Run-balanced p95 |PCC ramp| (W/s)',title='Fixed policies, reused evaluation set'); ax.legend(); save(fig,'budget_p95')
    for metric in ('p95_ramp','max_ramp'):
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,name in zip(axes,('fast','ess')):
            for mode in ('C2','C3','C4'):
                g=summary[summary.controller==mode]; ax.plot(g[f'{name}_mean_abs_w'],g[metric],'o-',label=mode)
                for _,r in g.iterrows(): ax.annotate(f'{r.budget:g}',(r[f'{name}_mean_abs_w'],r[metric]),fontsize=7)
            ax.set(xlabel=f'{name}: actual Wh per hour (mean |W|)',ylabel=f'{metric} (W/s)'); ax.legend()
        save(fig,f'actual_usage_{metric}')
    fig,axes=plt.subplots(2,1,figsize=(11,7))
    for ax,comp in zip(axes,('C4-C3','C4-C2')):
        g=diff[(diff.comparison==comp)&(diff.metric=='p95_ramp')].sort_values('run_id')
        ax.bar(g.run_id.str[-8:],g.difference); ax.axhline(0,color='black',lw=.8); ax.tick_params(axis='x',rotation=90); ax.set(title=comp,ylabel='p95 difference (W/s)')
    save(fig,'paired_run_differences')
    primary=runs[(runs.budget==1)&(runs.controller=='C4')].set_index('run_id')
    gain=diff[(diff.comparison=='C4-C3')&(diff.metric=='p95_ramp')].set_index('run_id').difference
    pool=primary[primary.forecast_P4_mae<primary.forecast_P2_mae].index
    pool=pool if len(pool) else primary.index
    ordered=gain.reindex(pool).sort_values(); cases=[]
    for label,run in [('best_control_delta',ordered.index[0]),('worst_control_delta',ordered.index[-1])]:
        traces={m:pd.read_parquet(out/'checkpoints'/f'{run}_{m}_b1.parquet') for m in ('C2','C3','C4')}
        first=traces['C4'].iloc[0]; traces={m:z[(z.segment==first.segment)&(z.window==first.window)] for m,z in traces.items()}
        z=traces['C4']; t=z.issuance_s.to_numpy(); fig,axes=plt.subplots(6,1,figsize=(12,13),sharex=True)
        axes[0].plot(t,z.baseline_demand_w,label='load at t'); axes[0].plot(t+cfg['horizon_bins']*cfg['dt'],z.P2,label='P2 forecast at target'); axes[0].plot(t+cfg['horizon_bins']*cfg['dt'],z.P4,label='P4 forecast at target'); axes[0].plot(t+cfg['horizon_bins']*cfg['dt'],z.actual_future_w,label='future actual',alpha=.5)
        for mode,q in traces.items():
            axes[1].plot(t,q.pcc_w,label=mode)
            axes[2].plot(t,q.fast_w,label=f'{mode} fast'); axes[3].plot(t,q.ess_w,label=f'{mode} ESS')
            axes[4].plot(t,q.fast_soc,label=f'{mode} fast'); axes[4].plot(t,q.ess_soc,'--',label=f'{mode} ESS')
            axes[5].plot(t,q.fast_credit_wh,label=f'{mode} fast'); axes[5].plot(t,q.ess_credit_wh,'--',label=f'{mode} ESS')
        for ax,lab in zip(axes,('load/forecast W','PCC W','fast W','ESS W','SoC','credit Wh')):
            ax.set_ylabel(lab); ax.axvspan(t[0]+cfg['window_s']-cfg['recovery_s'],t[-1],alpha=.08,color='gray'); ax.legend(ncol=3,fontsize=7)
        axes[-1].set_xlabel('time s'); axes[0].set_title(f'{label}: {run}; first eligible window; gray = recovery')
        save(fig,f'case_{label}')
        cases.append({'case':label,'run_id':run,'p95_C4_minus_C3':float(gain[run]),'P4_mae':float(primary.loc[run,'forecast_P4_mae']),'P2_mae':float(primary.loc[run,'forecast_P2_mae']),
                      'rule':'Among forecast-improved runs, minimum/maximum C4-C3 run p95; fallback all runs; show first eligible window (not cherry-picked event)'})
    pd.DataFrame(cases).to_csv(out/'case_selection.csv',index=False)
    def table(df):
        return df.to_markdown(index=False,floatfmt='.5g')
    primary_summary=summary[(summary.budget==1)&summary.controller.isin(['C2','C3','C4'])]
    statements=[]
    for comp in ('C4-C2','C4-C3'):
        r=paired[(paired.comparison==comp)&(paired.metric=='p95_ramp')].iloc[0]
        mx=paired[(paired.comparison==comp)&(paired.metric=='max_ramp')].iloc[0]
        statements.append(f'- {comp}: p95 차이 {r.mean_difference:.4f} W/s, 95% CI [{r.ci_low:.4f}, {r.ci_high:.4f}]. 최대 ramp 차이 {mx.mean_difference:.4f} W/s (음수=감소). CI가 0을 포함하면 일관된 우위를 단정하지 않는다.')
    constraints=pd.read_csv(out/'constraint_checks.csv')
    violation=[c for c in constraints if 'violation' in c]
    bad=int((constraints[violation].abs()>1e-6).any(axis=1).sum())
    audit=json.loads((out/'source_audit.json').read_text())
    report=f'''# 동일 장치별 처리량 예산 제어 실험\n\n**기존 평가셋 재사용 분석 · E1 actuator 가정 · 사전 고정 정책 비교**\n\n## 완료 범위와 차단 범위\n\n이 보고서는 이번 파이프라인에서 실제 완료한 고정 정책 시뮬레이션만 분석한다. smoke 설정 여부는 resolved_config.yaml을 확인한다. Outer-training 내부 grouped-validation 예측과 저장 학습 모델이 없어 **요청된 tuning 후 비교는 blocked**이다. 기존 outer OOF를 다른 fold의 내부 validation에 돌려 쓰지 않았고 모델을 재학습하지 않았다. 전체 요구사항 완료 또는 독립 검증으로 해석할 수 없다.\n\n## 동일 예산에서 C4가 나은가?\n\nPrimary budget=1.0, run 내 유효 창들의 ramp를 모아 p95를 계산한 뒤 run별 동일 가중 평균이다. 창/segment 경계를 연결하지 않는다. segment p95 평균과 window p95 평균은 per_run_metrics.csv에 별도 열이다.\n\n{chr(10).join(statements)}\n\n{table(primary_summary[['controller','p95_ramp','max_ramp','fast_throughput_wh','ess_throughput_wh','fast_mean_abs_w','ess_mean_abs_w']])}\n\n## 실제 사용량을 고려한 해석\n\n같은 예산은 실제 사용량이 같다는 뜻이 아니다. 처리량은 AC 측 충·방전 절대합 Wh이며 순 소비·손실·열화량이 아니다. 장치별 사용량과 p95/maximum trade-off 그림을 함께 제공한다. 실제 사용량을 맞추기 위한 추가 충·방전은 하지 않았다. 아래 동일 사용량 근접 부분집합은 양 장치 모두 실제 사용량 5% 이내인 run만 남긴 **기술적 보조 분석**이며 인과적 동일 처리량 비교가 아니다. 표본이 없거나 적으면 실제 처리량까지 통제한 이점은 판정할 수 없다.\n\n{table(pd.read_csv(out/'matched_usage_descriptive.csv'))}\n\n## 최대 ramp와 run 집중도\n\n{table(pd.read_csv(out/'improvement_counts.csv').query("metric in ['p95_ramp','max_ramp']"))}\n\n평균 개선만으로 모든 run 개선을 주장할 수 없다. paired_run_differences 그림과 paired_per_run.csv에서 부호·크기 분포를 확인한다. p95와 최대 ramp는 별개로 판단하며 복구 구간의 큰 ramp도 포함된다.\n\n## 공통 조건·예산·종료 조건\n\n{table(pd.read_csv(out/'device_assumptions.csv'))}\n\n60초 clock window의 첫 40초 제어, 마지막 20초 SoC 복구. 모든 장치는 초기·종료 SoC 0.5, 최종 출력 0. 예산을 복구 에너지까지 예약하며 복구 ramp와 처리량을 숨기지 않는다. 장치별 기준 처리량은 각 outer-training C2 운전의 run-balanced 평균 절대 출력에서 정한다. 창 초기 credit=전체 예산, 중간 보충 없음, 이월 없음. 잔여 불완전 창은 제외하며 exclusions.csv로 남긴다. 테스트 run의 미래 길이·분위수·에너지는 제어 판단에 사용하지 않는다.\n\n제약 위반 행: **{bad}**. 제약 상세와 종료 에너지 차이는 constraint_checks.csv.\n\n## 예산 전체 결과\n\n{table(summary[summary.controller.isin(['C2','C3','C4'])][['budget','controller','p95_ramp','max_ramp','fast_throughput_wh','ess_throughput_wh']])}\n\n## Paired 차이와 조건부 CI\n\n{table(paired)}\n\n2,000회 source-group cluster bootstrap. timestamp나 segment를 독립 표본으로 취급하지 않았다. source SHA256 group과 run은 이 자료에서 대응하지만 수집 session 독립성은 검증되지 않았다. 따라서 CI는 고정 예측·고정 정책 및 source-group 독립성 가정에 조건부이며 독립 group 검증 완료를 의미하지 않는다.\n\n## 예측 이득과 제어 이득이 연결되지 않는 이유\n\n관측된 사실: 다음 사례는 P4 예측 MAE가 P2보다 작은 run 중 제어 p95 차이가 가장 좋은/나쁜 run이다. 해당 조건 run이 없으면 전체 run으로 대체한다. 각 run의 첫 유효 창을 보여주며 좋은 이벤트만 선택하지 않는다. 부호가 모두 같으면 이득 연결/비연결 사례가 둘 다 있다고 주장하지 않는다.\n\n{table(pd.DataFrame(cases))}\n\n원인 가설: endpoint MAE와 p95 ramp 최소화 목적이 다르고, 예측이 reference를 이동시키는 방향이 PCC ramp에 항상 유리하지 않다. 공통 지연, 장치 포화, 예산 예약, 복구 동작이 예측 이득 전달을 제한할 수 있다. 제약 활성 비율과 요청-적용 출력 차이는 이 가설의 진단 자료이며 원인별 인과 기여를 분리해 입증한 것은 아니다. C5는 perfect-forecast diagnostic이며 최적화 상한이 아니다.\n\n## 기존 실험과 구분·현재 판정 불가\n\n기존 전체 segment 처리량 합계의 C4/C2 증가는 {audit['legacy_c4_vs_c2_percent']:.6f}%. 새 실험은 fast 에너지 제약, 장치별 예산, 공통 복구와 평가창이 추가되어 기존 수치와 직접 혼합하지 않는다. C0/C1 동일 이유, 예측 시각, 모델 선택 및 구현 변경은 implementation_audit.md 참조.\n\n누출 없는 controller tuning 후 우열, 재학습 변동을 포함한 CI, 실제 동일 사용량에서의 인과 이점, 독립 수집 session 재현성, 측정 GPU cap 효과, 실제 전압·주파수 안정도 개선은 현재 자료로 판정할 수 없다. 실험은 실제 장치에 명령을 보내지 않는다.\n\n## 산출물\n\n- fair_budget_metrics.csv: run·segment·window·controller·budget 지표\n- per_run_metrics.csv, budget_summary.csv: run-balanced 집계\n- paired_comparisons.csv, paired_per_run.csv, improvement_counts.csv\n- constraint_checks.csv, device_assumptions.csv, case_selection.csv\n- checkpoints/: 적용 출력·SoC·예산 잔량을 포함한 전체 평가 시계열과 task checksum\n- figures/: 예산 곡선, 장치별 실제 사용량 trade-off, run별 paired 차이, 양쪽 사례의 PNG/PDF\n'''
    (out/'report_ko.md').write_text(report)
