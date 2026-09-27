"""Descriptive day-cluster sensitivity for completed highres nested CV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'results/nested_cv5_20260926'


def run(out):
    out = Path(out)
    per = pd.read_csv(out / 'per_run_metrics.csv')
    dates = pd.read_csv(HERE / 'results/actual_20260926/highres_source_intervals.csv')
    dates = dates[['run_id', 'collection_day', 'cohort']]
    run = per.groupby(['run_id', 'condition'], as_index=False).crps_w.mean()
    pivot = run.pivot(index='run_id', columns='condition', values='crps_w').reset_index()
    frame = pivot.merge(dates, on='run_id', validate='one_to_one')
    rows = []
    for condition in ('history_static_s2_controlled', 'history_static_s2_tuned'):
        for day in sorted(frame.collection_day.unique()):
            for scope, part in [('day_only', frame.loc[frame.collection_day.eq(day)]),
                                ('omit_day', frame.loc[~frame.collection_day.eq(day)])]:
                rows.append(dict(condition=condition, scope=scope, collection_day=day,
                    n_runs=len(part), n_days=part.collection_day.nunique(),
                    history_crps_w=part['history_only_tuned'].mean(), metadata_crps_w=part[condition].mean(),
                    gain_pct=100*(1-part[condition].mean()/part['history_only_tuned'].mean())))
    result = pd.DataFrame(rows)
    result.to_csv(out / 'day_sensitivity.csv', index=False)
    lines = ['# Highres nested CV: 수집일 민감도', '',
        '원본 timestamp audit에서 수집일 네 개와 GPU×workload 조합 네 개가 일대일로 결합됐다. 아래 값은 이미 학습된 outer-fold 예측의 run-balanced CRPS를 날짜별 또는 날짜 하나 제외 후 다시 계산한 기술 통계다. Day-only 값은 day-held-out 모델 성능이 아니다. 네 날짜로 유의성 CI를 만들지 않았다.', '',
        result.to_markdown(index=False, floatfmt='.2f'), '',
        '같은 수집일 기록이 기존 outer-fold의 train/test 양쪽에 있으므로 이 민감도도 날짜 밖 일반화 성능이 아니다. 날짜 밖 평가에 해당하는 별도 고정 설정 결과는 `../unseen_condition_20260926/report_ko.md`에 있다.', '',
        '재현: `python trace/grid_stability_forecast/assess_day_sensitivity.py`', '']
    (out / 'day_sensitivity_ko.md').write_text('\n'.join(lines))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT)
    print(run(parser.parse_args().out).to_string(index=False))
