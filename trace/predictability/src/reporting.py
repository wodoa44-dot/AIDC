from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(out,cfg):
    out=Path(out);plots=out/'plots';plots.mkdir(exist_ok=True)
    m=pd.read_csv(out/'metrics.csv');s=pd.read_csv(out/'transition_metrics.csv');ci=pd.read_csv(out/'confidence_intervals.csv')
    chosen=pd.read_csv(out/'selected_validation_configs.csv'); audit=pd.read_csv(out/'data_audit.csv')
    colors=dict(unconditional='#888888',current_only='#d97816',history='#1479b8')
    for cohort,g in m.groupby('cohort'):
        rs=sorted(g.resolution_ms.unique())
        for metric,label,file in [('mean_crps_w','Numerical CRPS (W)','crps'),('crpss_vs_unconditional','CRPSS vs unconditional','crpss')]:
            fig,axes=plt.subplots(1,len(rs),figsize=(5*len(rs),4),squeeze=False)
            for ax,r in zip(axes[0],rs):
                for cond,z in g[g.resolution_ms==r].groupby('condition'):
                    z=z.sort_values('horizon_ms');ax.plot(z.horizon_ms,z[metric],'-o',label=cond,color=colors[cond])
                ax.set(title=f'Completed-bin resolution: {r} ms',xlabel='Forecast horizon (ms)',ylabel=label);ax.grid(alpha=.25);ax.legend(fontsize=8)
            fig.suptitle(cohort+' | 8 GPU sum');fig.tight_layout();fig.savefig(plots/f'{cohort}_{file}.png',dpi=140);plt.close(fig)
        mat=g[g.condition=='history'].pivot(index='resolution_ms',columns='horizon_ms',values='crpss_vs_current_only')
        fig,ax=plt.subplots(figsize=(7,4));lim=max(.01,np.nanmax(np.abs(mat.to_numpy())))
        im=ax.imshow(mat.to_numpy(),cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
        ax.set(xticks=range(len(mat.columns)),xticklabels=mat.columns,yticks=range(len(mat.index)),yticklabels=mat.index,xlabel='Forecast horizon (ms)',ylabel='Completed-bin resolution (ms)',title=cohort+' | History gain vs current only')
        for (i,j),v in np.ndenumerate(mat.to_numpy()):
            if np.isfinite(v):ax.text(j,i,f'{100*v:.1f}%',ha='center',va='center',fontsize=9)
        fig.colorbar(im,ax=ax,label='CRPSS');fig.tight_layout();fig.savefig(plots/f'{cohort}_heatmap.png',dpi=140);plt.close(fig)
        # Predefined representative: 200 ms bins, shortest available horizon >=1000 ms,
        # lexicographically first run, first 20 seconds of eligible test targets.
        rr=200 if 200 in rs else rs[0];hh=sorted(g[(g.resolution_ms==rr)&(g.horizon_ms>=1000)].horizon_ms.unique())
        hh=hh[0] if len(hh) else int(g[g.resolution_ms==rr].horizon_ms.min())
        p=pd.read_parquet(out/'predictions'/f'{cohort}_{rr}ms_{hh}ms.parquet');p=p[p.condition=='history'];p=p[p.run_id==sorted(p.run_id.unique())[0]].sort_values('target_s');p=p[p.target_s<=p.target_s.min()+20]
        fig,ax=plt.subplots(figsize=(11,4));t=p.target_s.to_numpy();ax.plot(t,p.y,label='Observed future completed-bin power',color='black',lw=1)
        ax.plot(t,p['q_0.5'],label='Forecast median',color=colors['history']);ax.fill_between(t,p['q_0.05'].to_numpy(),p['q_0.95'].to_numpy(),alpha=.25,label='90% interval')
        ax.set(xlabel='Target time from run start (s)',ylabel='8 GPU power sum (W)',title=f'{cohort} | bin {rr} ms | horizon {hh} ms');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(plots/f'{cohort}_example.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,st in zip(axes,['stable','rising','falling']):
            for cond,z in s[(s.cohort==cohort)&(s.resolution_ms==rr)&(s.stratum==st)].groupby('condition'):
                z=z.sort_values('horizon_ms');ax.plot(z.horizon_ms,z.mean_crps_w,'-o',label=cond,color=colors[cond])
            ax.set(title=st,xlabel='Forecast horizon (ms)',ylabel='Numerical CRPS (W)');ax.legend(fontsize=8);ax.grid(alpha=.25)
        fig.suptitle(f'{cohort} | completed bins {rr} ms');fig.tight_layout();fig.savefig(plots/f'{cohort}_transitions.png',dpi=140);plt.close(fig)
    wide=m.pivot(index=['cohort','resolution_ms','horizon_ms'],columns='condition',values='mean_crps_w').reset_index()
    wide['history_gain_vs_current_pct']=100*(1-wide.history/wide.current_only)
    wide['current_gain_vs_unconditional_pct']=100*(1-wide.current_only/wide.unconditional)
    wide.to_csv(out/'comparison_summary.csv',index=False)
    lines=['# 전력 시계열 predictability 실험 결과','',
      '## 범위와 실제 입력','',
      f"실제 highres_node {len(audit)}개 run, {audit.usable_observations.sum():,}개 유효 기록을 사용했다. B200/H100 × image_generation/llm_training별로 독립 모델을 적합했다. 측정량은 한 노드의 **8 GPU 전력 합계(W), CPU 제외**다. 각 cohort에서 source path 사전순 첫 {cfg['runs_per_cohort']}개 canonical run을 결과 확인 전에 선택했다. 전체 workload를 대표하는 무작위 표본이 아니다.",
      '', '입력 경로·SHA256·컬럼·quality는 `input_manifest.csv`, `data_audit.csv`에 저장했다. 기존 `outputs/prepared/*.parquet`를 읽고 기존 `load_native`로 원본 CSV를 다시 읽어 time_s/power_w 일치를 검증했다. `events.csv/parquet`는 입력으로 사용하지 않았다. 기존 파일은 수정하지 않았다.',
      '', '## Sampling 및 leakage 감사','',
      'Highres nominal recording cadence는 20 ms이고 실제 median은 약 19.999 ms다. 센서 내부 update/averaging resolution은 별도 검증되지 않았다. CSV GPU 값의 엄격한 합계 이외 보간·중앙 rolling·forward fill은 없다. 기록 timestamp를 가용 시각으로 가정하며 실제 telemetry 전달 지연은 측정되지 않았다.',
      '', '전체 32개 canonical highres를 사전 조사한 20 ms bin count coverage 충족 비율은 run별 약 79.1–98.8%, 50 ms는 99.0–99.9%였다. jitter로 부적합한 20 ms regular grid는 primary에서 제외했다. 50/200/1000 ms만 선택하여 fine/coarse resolution을 비교하고 CPU 예산을 제한했다. 100/500 ms는 추가 비용 대비 중복을 줄이기 위해 생략했다. smoke run은 별도 디렉터리에 저장된다.',
      '', '각 bin은 **(end−resolution, end]** 구간 기록들의 산술평균이며 timestamp=end에서 가용하다. 이는 연속 시간 적분 평균이 아니라 관측 sample mean이다. nominal 20 ms 대비 기록 개수가 80% 이상인 bin만 사용한다. sensor freshness 80%를 보장한다는 뜻은 아니다. segment 양 끝 불완전 bin, 큰 gap(각 run median interval의 5배 초과), invalid/nonfinite/negative power, 역행/중복 timestamp를 분리한다. 결측 bin에서 재분할하며 채우거나 upsample하지 않는다. 입력과 target 모두 같은 segment에 속한다.',
      '', 'NLR training은 CPU+GPU whole-job sum으로 200 ms grid이나, `training/postprocess.py:resample_series`가 `utilities.py:create_dataframe_multinode`의 양쪽 관측값 기반 polynomial(order=1) interpolation을 호출한다. online-rate도 같은 함수로 1 ms grid를 만든 뒤 기존 preprocessing에서 100 ms 평균을 저장했다. 양쪽 모두 미래 source 관측 사용으로 primary 제외했다. 이미 보간된 파일을 split하거나 timestamp를 이동하는 것으로 해결하지 않았다. raw asynchronous device 로그의 별도 causal 재구성은 이번 실험 범위에서 수행하지 않았다. NLR에 대한 예측 성능이나 native 1 ms 예측 가능성을 주장하지 않는다.',
      '', '모든 발견된 파일 및 exclusion 이유는 `inventory_all_files.csv`, `excluded_inputs.csv`에 있다. optional NLR offline/finite는 기존 default inventory에 포함되지 않아 이번 후보가 아니다. 원본 quality summary는 전체 canonical run에 대해 보존했고 선택한 run은 재감사했다.',
      '', '## 실험 설계','',
      '각 run 실제 시간 범위를 약 60/20/20으로 분리했다. train label은 validation에 걸치지 않고, issuance와 target이 같은 partition이어야 한다. validation/test의 trailing history에는 실제로 이미 관측된 이전 partition 데이터를 허용하는 walk-forward 정책이다. 서로 다른 run 또는 segment는 이어 붙이지 않는다. 동일 조건 비교에서 longest eligible history를 공통 warm-up으로 사용해 target을 맞췄다. bin alignment는 run-relative 0 기준이며 같은 horizon/해상도의 모든 조건 target은 같다. 해상도 간 wall-clock split과 가능한 target 시각은 정렬되지만 coverage·gap·warm-up 차이로 완전히 동일한 target 집합은 아니다.',
      '', f"학습은 cohort/cell당 deterministic evenly-spaced 최대 {cfg['max_train_samples']}개, validation 선택은 최대 {cfg['max_validation_samples']}개를 사용한다. 실제 index를 모델 디렉터리에 저장했다. test target cadence는 max(resolution, {cfg['evaluation_cadence_ms']} ms)이다. 중첩 window를 독립 관측으로 간주하지 않는다. 4개 chronological test block/run 성능도 저장했다. 이는 알려진 run의 후기 구간 예측이며 held-out-run/workload 일반화 실험이 아니다.",
      '', f"unconditional은 해당 cell의 전체 training target empirical quantile이다. current_only는 최신 완료 bin 하나만, history는 최신 값·lag·strict trailing mean/std/min/max/slope만 사용한다. 두 ML 조건은 동일한 LightGBM quantile model, {cfg['n_estimators']} trees, {cfg['num_leaves']} leaves, learning rate {cfg['learning_rate']}, {cfg['n_jobs']} CPU threads를 사용한다. history 200/1000/5000 ms 중 resolution과 호환되는 후보를 validation CRPS로 선택하고 선택 파일을 test 평가 전에 기록했다. 1000 ms bin에서 200 ms history는 생략하며 1000 ms history는 current_only와 동등할 수 있다. train+validation 재적합은 하지 않는다. scaling은 사용하지 않는다.",
      '', '## Scoring 및 uncertainty','',
      f"Quantiles: {cfg['quantiles']}. 각 예측을 오름차순 재배열하며 보정 전 crossing 비율은 validation_metrics.csv와 timing.csv에 기록했다. CRPS는 2×pinball loss를 quantile 축에서 trapezoidal integration한 **numerical CRPS approximation (W)**이다. q=0/1에는 가장 바깥쪽 예측 quantile을 상수 연장했다. 극단 tail을 절단한 근사이고 모든 조건에서 동일하다. degenerate distribution=absolute error, uniform 사례를 테스트했다. CRPSS=1−model/reference이며 reference=0은 NaN으로 저장한다. median MAE와 80/90% coverage·width도 저장한다.",
      '', f"paired CRPS 차이의 95% interval은 run별로 segment 경계를 넘지 않는 {cfg['bootstrap_block_s']}초 nonoverlapping temporal block을 복원추출하여 {cfg['bootstrap_replicates']}회 계산했다. 같은 run은 고정하고 run 내부 block을 추출하므로 run 모집단의 uncertainty가 아니다. 짧은 경계 block은 실제 sample 수로 가중한다. 10초는 최대 history 5초와 horizon 2초보다 길지만 장기 의존을 모두 포착한다고 보장하지 않는다. CI는 다중 비교 보정 없는 exploratory interval이며 dense row 독립 bootstrap이 아니다. 값이 음수일수록 앞의 model이 더 우수하다.",
      '', '## 관측 결과 (numerical CRPS, W)','',wide.round(3).to_markdown(index=False),
      '', '### Paired block-bootstrap intervals (W)','',ci[['cohort','resolution_ms','horizon_ms','comparison','difference_w','ci_low_w','ci_high_w','n_blocks']].round(3).to_markdown(index=False),
      '', '## Rising/falling retrospective 평가','',
      f"각 cell training의 |target−current| {100*cfg['transition_quantile']:.0f} percentile을 threshold로 고정했다. delta>threshold는 rising, delta<−threshold는 falling, 나머지는 stable이다. 미래 target은 이 평가 층화에만 사용했다. threshold는 selected_validation_configs.csv에 있다. 이 정의는 horizon 내 순 변화로서 내부 왕복 transition을 모두 감지하지 않는다. 100 sample 미만은 insufficient evidence이며 더 큰 n도 독립 event 개수가 아니다.",
      '',s[['cohort','resolution_ms','horizon_ms','stratum','condition','n','mean_crps_w','coverage_90','status']].round(3).to_markdown(index=False),
      '', '## 해석','']
    for cohort,g in wide.groupby('cohort'):
        c=ci[ci.cohort==cohort]
        cur=c[c.comparison=='current_only-minus-unconditional'];hist=c[c.comparison=='history-minus-current_only']
        positive_cur=cur[cur.ci_high_w<0];positive_hist=hist[hist.ci_high_w<0];negative_hist=hist[hist.ci_low_w>0]
        fmt=lambda z:', '.join(f"{int(v.resolution_ms)}/{int(v.horizon_ms)} ms" for v in z.itertuples()) or '없음'
        lines += [f"- **{cohort}**: current_only 개선 CI가 0 아래인 resolution/horizon: {fmt(positive_cur)}. history가 current_only보다 추가 개선한 셀: {fmt(positive_hist)}. history가 악화된 CI 셀: {fmt(negative_hist)}. 나머지는 이 설정에서 일관된 추가 이득을 입증하지 못했다."]
    lines += ['', '### Transition 결과 해석 (200 ms bin, 1000 ms horizon)', '']
    for cohort,g in s[(s.resolution_ms==200)&(s.horizon_ms==1000)].groupby('cohort'):
        for st,z in g.groupby('stratum'):
            z=z.set_index('condition');u=z.loc['unconditional','mean_crps_w'];c=z.loc['current_only','mean_crps_w'];hh=z.loc['history','mean_crps_w'];n=int(z.loc['history','n'])
            lines.append(f'- {cohort}, {st}: n={n}, CRPS unconditional/current/history = {u:.2f}/{c:.2f}/{hh:.2f} W. history의 current 대비 개선율은 {100*(1-hh/c):.1f}%다. 층화별 CI는 계산하지 않아 transition 효과의 통계적 확정을 주장하지 않는다.')
    lines += ['', '### Calibration과 test block 안정성', '']
    for cond,g in m.groupby('condition'):
        lines.append(f'- {cond}: 명목 90% interval의 cell별 실제 coverage 범위는 {100*g.coverage_90.min():.1f}–{100*g.coverage_90.max():.1f}%다. 명목값과의 차이는 CRPS 개선이 calibration 보장을 뜻하지 않음을 보여준다.')
    bm=pd.read_csv(out/'test_block_metrics.csv')
    bw=bm.pivot(index=['cohort','resolution_ms','horizon_ms','run_id','test_block'],columns='condition',values='mean_crps_w')
    for cohort,g in bw.groupby(level='cohort'):
        lines.append(f'- {cohort}: history가 current_only보다 낮은 CRPS를 보인 run/test block 조합은 {int((g.history<g.current_only).sum())}/{len(g)}개다. horizon/resolution 간 중복 block이며 독립 반복 수가 아니다.')
    lines += ['', '## 한계 및 산출물','',
      '본 관측은 선택한 GPU-only traces·모델·bin target·horizon에 국한된다. 작은 모델 budget, history 후보 선택 효과, run 내 distribution shift, 센서 update 불확실성, 기록 수 기반 coverage, 긴 temporal dependence, 선택한 run subset의 대표성 한계가 있다. coarser bin은 target 자체가 부드러워지므로 raw CRPS 감소만으로 predictability 증가를 결론 내릴 수 없다. calibration과 제어 가능성은 별개이며 interval coverage를 함께 검토해야 한다. 예측 이득 부재는 근본적 unpredictability의 증명이 아니다.',
      '', '코드/재현 명령은 ../../README.md, quantile prediction은 predictions/*.parquet, 모델은 models/*.joblib, 연속 bin은 continuous/*.parquet, 전체 성능은 metrics.csv, test block은 test_block_metrics.csv, transition은 transition_metrics.csv, CI는 confidence_intervals.csv, 환경은 environment.json, execution 상태는 status.json에 있다. inference timing은 feature 추출·disk I/O를 제외한 quantile 모델 순차 예측과 rearrangement의 CPU wall time(단일 batch)이며 cold/warm latency benchmark가 아니다. 학습 시간은 각 후보별 timing.csv에 기록했다.',
      '', '대표 plot은 결과와 무관하게 cohort별 사전 규칙(200 ms bin, 가능한 ≥1000 ms 중 최소 horizon, run_id 사전순 첫 run, 첫 eligible test target부터 20초)으로 선택했다. forecast를 **target timestamp**에 정렬했다. plots/에는 CRPS, CRPSS, history-gain heatmap, interval 예시, transition 비교 5종이 있다.',
      '', 'NLR causal reconstruction, 20 ms regular-grid 분석, unseen-run generalization, hyperparameter 대규모 탐색은 수행하지 않았다. primary 실험 결과를 smoke 결과로 대체하지 않았다.']
    (out/'report_ko.md').write_text('\n'.join(lines)+'\n')
