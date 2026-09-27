"""Summarize the completed NLR 4-fold, 3-seed assessment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DEFAULT=HERE/'results/nlr_training_cv4_20260927'


def summarize(out):
    out=Path(out)
    progress=json.loads((out/'progress.json').read_text())
    checks=json.loads((out/'verification.json').read_text())
    if progress.get('status')!='completed' or not checks.get('all_passed'):
        raise ValueError('Four-fold experiment is incomplete or unverified')
    pred=pd.read_parquet(out/'per_target_predictions.parquet')
    per=pd.read_csv(out/'per_run_metrics.csv')
    overall=pd.read_csv(out/'aggregate_metrics.csv')
    effect=json.loads((out/'paired_gain_descriptive.json').read_text())
    # Run is the descriptive unit; sampling seeds are repeated fits of that run.
    run=per.groupby(['run_id','model_name','node_count','condition'],as_index=False).agg(
        crps_w=('crps_w','mean'),mae_w=('mae_w','mean'),p95_w=('p95_w','mean'),
        p99_w=('p99_w','mean'),coverage_80=('coverage_80','mean'))
    cohort=run.pivot(index=['run_id','model_name','node_count'],columns='condition',values='crps_w').reset_index()
    rows=[]
    for (model,nodes),part in cohort.groupby(['model_name','node_count']):
        rows.append(dict(model_name=model,node_count=int(nodes),n_jobs=len(part),
            history_crps_w=part.history_only.mean(),metadata_crps_w=part.history_model_node.mean(),
            gain_pct=100*(1-part.history_model_node.mean()/part.history_only.mean()),
            improved_jobs=int((part.history_model_node<part.history_only).sum())))
    cohorts=pd.DataFrame(rows).sort_values(['model_name','node_count'])
    cohorts.to_csv(out/'cohort_metrics.csv',index=False)
    seed=per.groupby(['seed','condition'],as_index=False).crps_w.mean()
    seed.to_csv(out/'seed_variability.csv',index=False)
    pred['transition']=np.select([pred.delta_w>=100,pred.delta_w<=-100],
        ['rise','drop'],default='steady')
    transition=pred.groupby(['run_id','seed','condition','transition'],as_index=False).agg(
        n_targets=('y','size'),crps_w=('crps_w','mean'),mae_w=('absolute_error_w','mean'),
        p95_w=('absolute_error_w',lambda x:x.quantile(.95)),
        p99_w=('absolute_error_w',lambda x:x.quantile(.99)))
    transition.to_csv(out/'transition_metrics.csv',index=False)
    trans_run=transition.groupby(['run_id','condition','transition'],as_index=False).agg(
        n_targets=('n_targets','first'),crps_w=('crps_w','mean'),
        mae_w=('mae_w','mean'),p95_w=('p95_w','mean'),p99_w=('p99_w','mean'))
    trans_summary=trans_run.groupby(['condition','transition'],as_index=False).agg(
        n_jobs=('run_id','nunique'),n_targets=('n_targets','sum'),
        crps_w=('crps_w','mean'),mae_w=('mae_w','mean'),
        p95_w=('p95_w','mean'),p99_w=('p99_w','mean'))
    trans_summary.to_csv(out/'transition_summary.csv',index=False)
    figures=out/'figures';data_dir=out/'figure_data'
    figures.mkdir(exist_ok=True);data_dir.mkdir(exist_ok=True)
    cohorts.to_csv(data_dir/'node_count_gain.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.5,4))
    colors={'llama2_70b_lora':'#0072B2','stable_diffusion':'#D55E00'}
    for model,part in cohorts.groupby('model_name'):
        ax.plot(part.node_count,part.gain_pct,marker='o',lw=2,label=model,
            color=colors.get(model,'#009E73'))
    ax.axhline(0,color='0.3',ls='--',lw=1)
    ax.set(xlabel='Participating nodes',ylabel='Descriptive metadata CRPS gain (%)',
        xticks=[2,4,8,16],title='NLR training; four job folds, three seeds')
    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(figures/f'node_count_gain.{ext}',dpi=300)
    plt.close(fig)
    report=['# NLR training: job-grouped 4-fold·3-seed 평가','',
        '41개 job을 model×node-count 8개 조건 안에서 4개 outer fold로 나눠 각 job을 정확히 한 번 test에 사용했다. 학습 표본 seed 17/42/2026을 반복하고, run당 최대 1,000행·균등 job weight·15 quantiles·고정 100-tree LightGBM을 두 feature 조건에 동일하게 적용했다. 이 평가는 모델 선택을 하지 않은 controlled fixed-setting 비교다. 전력 경계는 모든 참여 node의 CPU+GPU 합이다.','',
        overall.to_markdown(index=False,floatfmt='.2f'),'','전체 job 균형 CRPS 변화: '
        f"{effect['descriptive_gain_pct']:+.2f}% (개선 job {effect['improved_jobs']}/41).",'',
        '조건별 결과(각 cohort에 4–6 job, seed는 독립 job 수에 포함하지 않음):','',
        cohorts.to_markdown(index=False,floatfmt='.2f'),'','100 W endpoint ΔP 호환성 전이 구간(계통 허용치 아님):','',
        trans_summary.to_markdown(index=False,floatfmt='.2f'),'','수집 프로젝트의 원본 session 독립성은 확인되지 않았다. 따라서 job 41개를 독립 session으로 간주한 CI는 제시하지 않는다. 조건별 차이는 model·node metadata의 인과 효과가 아니며, 고정 설정·재사용 데이터의 기술 통계다. 80% 구간 coverage도 명목치와 별도로 확인해야 한다. GPU-only highres 결과와 절대 점수 수준을 비교하지 않는다.','',
        '그림 PNG/PDF 및 원자료는 `figures/node_count_gain.*`, `figure_data/node_count_gain.csv`에 있다. 재현: `python trace/grid_stability_forecast/run_nlr_training_cv.py`; `python trace/grid_stability_forecast/summarize_nlr_training_cv.py`.', '']
    (out/'report_ko.md').write_text('\n'.join(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT)
    summarize(parser.parse_args().out)
