#!/usr/bin/env python3
"""Generate evidence-labelled figures and a Korean report from saved follow-up data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT=Path(__file__).resolve().parents[1]
MAIN=PROJECT/'outputs/full_20260928T180455Z'
FOLLOW=MAIN/'feasible_followup_v1'
NLR=PROJECT/'outputs/nlr_raw_causal_pilot_v1'


def save(fig,name):
    fig.tight_layout();fig.savefig(FOLLOW/f'{name}.png',dpi=300);fig.savefig(FOLLOW/f'{name}.pdf');plt.close(fig)


def figures(event,control,nlr):
    e=event.copy();e['label']=e.direction+' '+e.horizon_s.map(lambda x:f'{x:g}s')
    e.to_csv(FOLLOW/'figure_data_event_brier.csv',index=False)
    labels=['up 1s','down 1s','up 2s','down 2s'];fig,ax=plt.subplots(figsize=(7,4));x=np.arange(len(labels))
    for j,model in enumerate(('P2','P4')):
        z=e.set_index(['label','model']).loc[[(label,model) for label in labels]].brier.to_numpy()
        ax.bar(x+(j-.5)*.36,z,width=.36,label=model)
    ax.set(xticks=x,xticklabels=labels,ylabel='Brier score',title='E0: path excursion events, run-balanced')
    ax.legend();save(fig,'event_brier_e0')
    c=control[(control.controller.isin(['C3','C4']))].copy()
    c.to_csv(FOLLOW/'figure_data_control_ramp.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for (factor,mode),g in c.groupby(['capacity_factor_training_p95','controller']):
        ax.plot(g.delay_s,g.pcc_ramp_p95_w_per_s,marker='o',label=f'{mode}, {factor:g}×P95')
    ax.set(xlabel='Assumed delay (s)',ylabel='PCC proxy p95 ramp (W/s)',title='E1: assumed storage sensitivity')
    ax.legend();save(fig,'control_ramp_e1')
    n=nlr.copy();n.to_csv(FOLLOW/'figure_data_nlr_crps.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4))
    for model,g in n.groupby('model'):
        ax.plot(g.horizon_s,g.compact3_crps_w,marker='o',label=model)
    ax.set(xlabel='Horizon (s)',ylabel='Compact3 CRPS approximation (W)',title='E0 pilot: NLR raw reconstruction, 5 jobs')
    ax.legend();save(fig,'nlr_compact3_e0_pilot')


def main():
    FOLLOW.mkdir(exist_ok=True)
    event=pd.read_csv(MAIN/'feasible_events_v2/event_aggregate_metrics.csv')
    event_ci=pd.read_csv(MAIN/'feasible_events_v2/paired_brier_gain_ci.csv')
    matching=pd.read_csv(MAIN/'feasible_events_v2/event_matching_aggregate.csv')
    control=pd.read_csv(MAIN/'feasible_control_sensitivity_v1/control_sensitivity_aggregate.csv')
    control_ci=pd.read_csv(MAIN/'feasible_control_sensitivity_v1/paired_c4_vs_c3_ramp_ci.csv')
    nlr=pd.read_csv(NLR/'forecast_compact3_v1/aggregate_metrics.csv')
    nlr_ci=pd.read_csv(NLR/'forecast_compact3_v1/paired_gain_ci.csv')
    figures(event,control,nlr)
    parts=len(list((MAIN/'prediction_parts').glob('*.parquet')))
    ablation_status=MAIN/'feasible_ablation_v1/status.json'
    ablation=json.loads(ablation_status.read_text()) if ablation_status.exists() else {'tasks_done':0,'tasks_planned':130,'complete':False}
    lines=['# 가능한 실험 후속 실행 보고','',
      '## 현재 범위','',f'- 기존 highres full15 예측: {parts}/750 prediction parts. 전체 E0 결과는 아직 미완료.',
      '- Highres path event 분류: 5 outer folds × 1/2 s × 상승/하강 × P2/P4 = 40/40 완료.',
      f"- Highres static metadata ablation: {ablation['tasks_done']}/{ablation['tasks_planned']} 변형 완료. Fold 0부터 진행 중.",
      '- NLR raw 1-node 파일럿: 5개 job 인과적 as-of 재구성, P0/P2 × 1/2 s = 10/10 완료.',
      '- E1 제어 용량·지연 sensitivity: fold 0의 7개 run, 1 s P2/P4 paired 예측으로 완료. 설비 응답은 관측이 아닌 가정.',
      '', '## 1. Metadata와 event 예측 (E0)','',
      'Path event는 issuance 전력 대비 미래 완료 bin의 최대 상승/하강 편차가 training-only 임계값을 넘는 경우다. Endpoint event와 별도로 저장했다.',
      event.to_markdown(index=False),'',
      'P2 대비 P4 Brier 감소율과 고정 예측 run bootstrap 95% CI:',event_ci.to_markdown(index=False),'',
      'Alarm threshold는 inner 데이터에서 30 false alarms/hour 예산으로 정했으나, test에서 이를 초과한 run이 있다. PR-AUC는 양성 event가 없는 run에서 N/A다.',
      '', '실제 완료-bin crossing과 alarm episode의 일대일 매칭. Lead는 매칭된 event만의 평균이며 miss를 함께 제시한다:',matching.to_markdown(index=False),
      '매칭은 1 s dead time과 event 전 horizon~후 0.2 s window를 고정했다. 긴 선행 alarm episode는 onset이 window 밖이면 unmatched로 남는다.',
      '', '## 2. 같은 controller에서 예측 입력 효과 (E1 가정)','',
      control.to_markdown(index=False),'',
      'C3 대비 C4의 PCC proxy p95 ramp 감소율. 음수는 metadata 예측 사용 시 악화:',control_ci.to_markdown(index=False),'',
      'GPU cap 후 전력과 처리량·지연의 counterfactual 자료가 없어 GPU 제어는 비활성이다. 저장장치 용량·지연·효율·SoC는 시나리오 가정이다.',
      '', '## 3. 계통 응답','',
      '검증된 계통 동특성 case가 없으므로 E2/E3 주파수·전압·진동 개선은 판정할 수 없다. 위 E1 PCC proxy를 계통 안정도 증거로 해석하지 않는다.',
      '', '## 4. NLR raw 별도 파일럿 (E0 조건부)','',
      'Raw GPU/CPU 로그의 timestamp 이전 최신 샘플만 사용하고 0.4 s 초과값은 버렸다. CPU package에 core 값을 중복 합산하지 않았다. Timestamp가 실제 수집 완료 시각인지, 센서 평균 window와 통신 지연은 확인되지 않았다.',
      nlr.to_markdown(index=False),'',
      'Compact3 CRPS의 P0 대비 P2 감소율과 5개 job paired bootstrap:',nlr_ci.to_markdown(index=False),'',
      'NLR은 CPU+GPU target이며 highres GPU-only full15 CRPS와 직접 비교하지 않는다. 동일 조건 반복 5개 job으로 metadata 효과나 unseen configuration 일반화는 평가할 수 없다.',
      '', '## 5. 결론 범위','',
      'Metadata event Brier는 4개 primary cell에서 개선됐지만, 제어 PCC proxy의 C4–C3 효과는 용량·지연 조건에 따라 악화와 개선이 섞인다. 계통 동특성 및 실제 actuator 성능은 식별되지 않는다. Run별 수집 session 독립성도 미검증이다.',
      '', '## 재현','',
      '`PYTHONPATH=src /usr/bin/python scripts/run_feasible_events_v2.py`',
      '`PYTHONPATH=src /usr/bin/python scripts/run_feasible_ablation.py --fold 0`',
      '`/usr/bin/python scripts/reconstruct_nlr_causal_pilot.py`',
      '`PYTHONPATH=src /usr/bin/python scripts/run_nlr_pilot_predict.py`',
      '`PYTHONPATH=src /usr/bin/python scripts/run_feasible_control_sensitivity.py`',
      '`/usr/bin/python scripts/summarize_feasible.py`',
      '`/usr/bin/python scripts/report_feasible.py`']
    extended_dir=MAIN/'extended_primary_eval_v1'
    extended_status=json.loads((extended_dir/'status.json').read_text())
    extended=pd.read_csv(extended_dir/'tail_and_interval_aggregate.csv')
    extended=extended[extended.regime.eq('all')&extended.model.isin(['P2','P4'])].copy()
    reliability=pd.read_csv(extended_dir/'quantile_reliability_aggregate.csv')
    reliability=reliability[(reliability.horizon_s==1)&reliability.model.isin(['P2','P4'])].copy()
    extended.to_csv(FOLLOW/'figure_data_primary_tail_partial.csv',index=False)
    reliability.to_csv(FOLLOW/'figure_data_primary_reliability_partial.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4))
    for mode,g in reliability.groupby('model'):
        ax.plot(g.nominal_quantile,g.empirical_coverage,marker='o',label=mode)
    ax.plot([0,1],[0,1],color='black',linestyle='--',label='ideal')
    ax.set(xlabel='Nominal quantile',ylabel='Empirical coverage',title='E0: 1s reliability, available folds only')
    ax.legend();save(fig,'primary_reliability_partial_e0')
    fig,ax=plt.subplots(figsize=(6,4))
    for mode,g in extended.groupby('model'):
        ax.plot(g.horizon_s,g.p99_abs,marker='o',label=mode)
    ax.set(xlabel='Horizon (s)',ylabel='Run-balanced p99 absolute error (W)',title='E0: available primary folds only')
    ax.legend();save(fig,'primary_p99_partial_e0')
    lines+=['', '## 예측 tail·구간 신뢰도 (진행 중인 E0)', '',
      f"현재 primary 예측 파일 {extended_status['prediction_parts_evaluated']}/{extended_status['prediction_parts_expected']}개를 집계했다. Fold 수가 horizon마다 다르므로 전체 결과 또는 horizon 간 최종 비교가 아니다.",
      extended[['horizon_s','model','crps','p95_abs','p99_abs','coverage_80','coverage_90','width_80','width_90']].to_markdown(index=False),
      'Nominal-empirical quantile reliability와 p99 그림은 이용 가능한 fold만을 표시한다.']
    frequency_dir=MAIN/'frequency_only_exploratory_v1'
    frequency=pd.read_csv(frequency_dir/'frequency_metrics_aggregate.csv')
    frequency_ci=pd.read_csv(frequency_dir/'paired_frequency_gain_ci.csv')
    lines+=['', '## 공개 파라미터 저차 모델의 주파수 전용 탐색', '',
      'Aslami et al. (2024)의 M=4 s, D=1.5%, R=5%, Tg=0.2 s를 사용했다. 100 MVA·60 Hz base와 동일 trace 1000개 동시 집계는 시나리오 가정이다. 입력은 200 ms마다 유지하고 ODE는 20 ms로 적분한다.',
      frequency.to_markdown(index=False),'',
      'C3 대비 C4 최대 절대 주파수 편차의 paired run 변화:',frequency_ci.to_markdown(index=False),'',
      '평형, 양의 부하 step 부호·정상값, 20/10 ms 수렴을 통과했다. 이 모델은 실제 네트워크 case가 아니며 전압, 공간 모드, GFM 안정도와 현장 계통 성능을 지원하지 않는다.',
      '모델 출처: https://www.osti.gov/servlets/purl/2404598','']
    frequency.to_csv(FOLLOW/'figure_data_frequency_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4))
    for mode,g in frequency.groupby('controller'):
        ax.bar(mode,float(g.max_abs_delta_f_hz.iloc[0]),label=mode)
    ax.set(ylabel='Maximum |frequency deviation| (Hz)',title='Frequency-only exploratory model, assumed scaling')
    save(fig,'frequency_max_deviation_exploratory')
    freq_segment=pd.read_csv(frequency_dir/'frequency_metrics_per_segment.csv')
    worst=freq_segment[freq_segment.controller.eq('C0')].sort_values(['max_abs_delta_f_hz','run_id','segment'],ascending=[False,True,True]).iloc[0]
    freq_trace=pd.read_parquet(frequency_dir/'frequency_traces.parquet')
    chosen=freq_trace[freq_trace.run_id.eq(worst.run_id)&freq_trace.segment.eq(worst.segment)].copy()
    chosen.to_csv(FOLLOW/'figure_data_frequency_worst_c0_case.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for mode,g in chosen.groupby('controller'):
        ax.plot(g.issuance_s-g.issuance_s.min(),g.frequency_hz,label=mode)
    ax.set(xlabel='Time since segment start (s)',ylabel='Frequency (Hz)',title='Frequency-only exploratory: worst C0 segment')
    ax.legend();save(fig,'frequency_worst_c0_case_exploratory')
    lines+=['사례 그림은 C0 최대 |Δf|가 가장 큰 segment를 자동 선택했다. 전체 run 집계와 함께 해석해야 한다.']
    if ablation['tasks_done']:
        table=pd.read_csv(MAIN/'feasible_ablation_v1/metadata_ablation.csv')
        lines+=['','## 진행 중인 metadata ablation','',table.to_markdown(index=False)]
    print(FOLLOW)

if __name__=='__main__':main()
