"""Report and plot the separate fixed-setting 5/10/30 s forecast pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'results/long_fixed_20260927'
COLORS = {'history_only': '#0072B2', 'history_static_s1': '#E69F00',
          'history_static_s2': '#009E73'}


def summarize(out):
    out = Path(out)
    progress = json.loads((out / 'progress.json').read_text())
    if progress.get('status') != 'completed':
        raise ValueError('Long-horizon pilot is still running')
    for name in ('selected_hyperparameters', 'tail_metrics', 'per_run_metrics'):
        pd.concat([pd.read_csv(out / f'h{h}' / f'{name}.csv') for h in (5000, 10000, 30000)],
            ignore_index=True).to_csv(out / f'{name}.csv', index=False)
    aggregate = pd.read_csv(out / 'aggregate_metrics.csv')
    paired = pd.read_csv(out / 'paired_gain_ci.csv')
    for horizon in (5000, 10000, 30000):
        if progress[f'h{horizon}']['status'] != 'verified':
            raise ValueError(f'h{horizon} is unverified')
        checks = json.loads((out / f'h{horizon}' / 'verification.json').read_text())
        if not checks['all_passed']:
            raise ValueError(f'h{horizon} failed verification')
    figure_data = aggregate[['condition', 'horizon_ms', 'n_runs', 'crps_w', 'mae_w',
        'p95_w', 'p99_w', 'coverage_80']].sort_values(['condition', 'horizon_ms'])
    (out / 'figure_data').mkdir(exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    figure_data.to_csv(out / 'figure_data' / 'long_horizon_crps.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for condition, data in figure_data.groupby('condition'):
        ax.plot(data.horizon_ms / 1000, data.crps_w, marker='o', lw=2,
            label=condition, color=COLORS[condition])
    ax.set(xlabel='Forecast horizon (s)', ylabel='Run-balanced CRPS approximation (W)',
           xticks=[5, 10, 30], title='Fixed-setting highres GPU forecast; 10 reused test runs')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / 'figures' / f'long_horizon_crps.{ext}', dpi=300)
    plt.close(fig)
    scores = aggregate.pivot(index='horizon_ms', columns='condition', values='crps_w')
    lines = ['# 5/10/30초 장기 horizon 고정 설정 평가', '',
        'Highres 8 GPU 합산 전력의 완료된 200 ms 평균 bin을 5·10·30초 뒤에 예측했다. 세 horizon을 +0.2–2초 곡선과 별도 실험으로 저장했다. 5초 history, 100 trees, 15 leaves, learning rate 0.07, 15 quantiles, seed 42를 고정했다. 22개 학습 run/10개 이전에 사용한 test run이므로 새 독립 확인은 아니다. Inner tuning을 하지 않아 최적 history 또는 최적 모델 설정의 비교가 아니다.', '',
        '| Horizon | History CRPS (W) | S1 CRPS (W) | S2 CRPS (W) | S2 gain (%) | 95% paired run CI (%) |',
        '|---:|---:|---:|---:|---:|---:|']
    for horizon in (5000, 10000, 30000):
        ci = paired.loc[(paired.horizon_ms.eq(horizon)) & paired.condition.eq('history_static_s2')].iloc[0]
        row = scores.loc[horizon]
        lines.append(f'| {horizon/1000:g} s | {row.history_only:.2f} | {row.history_static_s1:.2f} | {row.history_static_s2:.2f} | {ci.gain_of_run_balanced_scores_pct:+.2f} | {ci.ci_low_pct:+.2f} to {ci.ci_high_pct:+.2f} |')
    lines += ['', 'CI는 저장 예측의 paired test-run 재표집만 반영한다. 같은 10개 test run을 이전 파일럿에서 이미 사용했고 원본 32개 trace의 수집일 네 개가 GPU×workload 조합과 일대일로 결합되며 같은 날짜 기록이 train/test에 함께 있다. session 독립성도 미확인이다. 전이·tail·coverage는 각 horizon의 CSV와 합친 `transition_metrics.csv`, `calibration_metrics.csv`에 있다. 그림 PNG/PDF와 원자료 CSV는 `figures/`, `figure_data/`에 저장했다. GPU 전력 대리지표이므로 PCC 또는 실제 장치 제어 효과를 뜻하지 않는다.', '',
        '재현: `python trace/grid_stability_forecast/run_long_horizon.py`; `python trace/grid_stability_forecast/summarize_long_horizon.py`.', '']
    (out / 'report_ko.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT)
    summarize(parser.parse_args().out)
