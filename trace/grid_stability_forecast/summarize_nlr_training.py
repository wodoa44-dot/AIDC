"""Summarize the separate NLR training job-metadata pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'results/nlr_training_pilot_20260927'


def summarize(out):
    out = Path(out)
    checks = json.loads((out / 'verification.json').read_text())
    if not checks.get('all_passed'):
        raise ValueError('Pilot verification failed')
    pred = pd.read_parquet(out / 'per_target_predictions.parquet')
    per = pd.read_csv(out / 'per_run_metrics.csv')
    aggregate = pd.read_csv(out / 'aggregate_metrics.csv')
    gain = json.loads((out / 'paired_gain_descriptive.json').read_text())
    pred['transition'] = np.select([pred.delta_w >= 100,pred.delta_w <= -100],
                                   ['rise','drop'], default='steady')
    transition = pred.groupby(['run_id','condition','transition'],as_index=False).agg(
        n_targets=('y','size'),crps_w=('crps_w','mean'),mae_w=('absolute_error_w','mean'),
        p95_w=('absolute_error_w',lambda x:x.quantile(.95)),
        p99_w=('absolute_error_w',lambda x:x.quantile(.99)))
    transition.to_csv(out / 'transition_metrics.csv',index=False)
    transition_summary = transition.groupby(['condition','transition'],as_index=False).agg(
        n_runs=('run_id','nunique'),n_targets=('n_targets','sum'),
        crps_w=('crps_w','mean'),mae_w=('mae_w','mean'),
        p95_w=('p95_w','mean'),p99_w=('p99_w','mean'))
    transition_summary.to_csv(out / 'transition_summary.csv',index=False)
    tail = pred.groupby(['run_id','condition'],as_index=False).agg(
        p95_w=('absolute_error_w',lambda x:x.quantile(.95)),
        p99_w=('absolute_error_w',lambda x:x.quantile(.99)),
        worst_1pct_mean_w=('absolute_error_w',lambda x:x.nlargest(max(1,int(np.ceil(len(x)*.01)))).mean()))
    tail.to_csv(out / 'tail_metrics.csv',index=False)
    by_condition = per.pivot(index=['run_id','model_name','node_count'],columns='condition',values='crps_w').reset_index()
    by_condition['gain_pct'] = 100*(1-by_condition.history_model_node/by_condition.history_only)
    by_condition.to_csv(out / 'condition_metrics.csv',index=False)
    lines=['# NLR training +1초 CPU+GPU 전력 예측 파일럿','',
        'NLR training의 41개 job을 별도 데이터셋으로 평가했다. target은 모든 참여 node의 CPU+GPU 합산 전력에서 200 ms 완료 bin 기준 +1초 값이다. 각 model×node-count 조건에서 job 하나를 test로 남겨 33개 학습/8개 test job을 구성했다. 같은 8개 test job의 시점을 history-only와 model name+node count 조건에 짝지었고, run당 최대 1,500행의 동일 학습 표본·균등 run weight·15 quantiles·고정 100-tree LightGBM을 사용했다. Highres GPU-only 점수와 절대 크기를 비교하지 않는다.','',
        aggregate.to_markdown(index=False,floatfmt='.3f'),'','Metadata 조건의 기술적 CRPS 변화: '
        f"{gain['descriptive_gain_pct']:+.2f}% (개선 job {gain['improved_runs']}/8).",'',
        '조건별 test 결과(각 model×node 조건에서 test job 하나뿐임):','',
        by_condition.to_markdown(index=False,floatfmt='.2f'),'','100 W endpoint ΔP 호환성 구간별 run 균형 결과(계통 허용치 아님):','',
        transition_summary.to_markdown(index=False,floatfmt='.2f'),'','Run별 p95/p99와 worst 1% 오차는 `tail_metrics.csv`에 저장했다. 80% 구간 coverage는 0.89 안팎으로 nominal 0.8보다 높아 calibration이 맞는다고 볼 수 없다. 원본 job은 같은 NLR training 수집 프로젝트에 속하며 test 조건당 독립 반복이 하나라 subgroup 효과 또는 CI를 확정하지 않는다. 독립 세션 관계·원시 센서 갱신·PCC 및 실제 장치 응답도 확인되지 않았다.','',
        'NLR long-valley recovery의 별도 offline 재측정은 `../event_v2_nlr_training_20260926/report_ko.md`에 있다. 이 +1초 endpoint 예측 파일럿은 30초 valley onset·duration 또는 제어 성공을 평가하지 않는다.','',
        '재현: `python trace/grid_stability_forecast/run_nlr_training_pilot.py`; `python trace/grid_stability_forecast/summarize_nlr_training.py`.','']
    (out / 'report_ko.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT)
    summarize(parser.parse_args().out)
