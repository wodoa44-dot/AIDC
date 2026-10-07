"""Compare verified grouped-inner highres forecasts across the 200 ms horizon grid."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
CAMPAIGN = RESULTS / 'nested_remaining_20260928'
OUT = RESULTS / 'nested_grid_comparison_20260928'
HORIZONS = (200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000)
CONDITIONS = {
    'history_only_tuned': ('History', '#0072B2'),
    'history_static_s2_controlled': ('S2 controlled', '#E69F00'),
    'history_static_s2_tuned': ('S2 tuned', '#009E73'),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_for(horizon: int) -> Path:
    if horizon == 1000:
        return RESULTS / 'nested_select_20260926'
    if horizon == 2000:
        return RESULTS / 'nested_h2000_20260927'
    return CAMPAIGN / f'h{horizon}'


def main() -> None:
    score_rows, gain_rows = [], []
    for horizon in HORIZONS:
        source = source_for(horizon)
        verification = json.loads((source / 'verification.json').read_text())
        if verification.get('all_passed') is not True:
            raise ValueError(f'Unverified horizon: {horizon}')
        if horizon not in (1000, 2000):
            progress = json.loads((source / 'progress.json').read_text())
            if progress.get('status') != 'completed' or progress.get('horizon_ms') != horizon:
                raise ValueError(f'Incomplete horizon: {horizon}')
            if sha256(source / 'per_target_predictions.parquet') != progress['prediction_sha256']:
                raise ValueError(f'Prediction hash mismatch: {horizon}')
        scores = pd.read_csv(source / 'summary_metrics.csv')
        gains = pd.read_csv(source / 'paired_gain_ci.csv')
        if set(scores.condition) != set(CONDITIONS):
            raise ValueError(f'Condition mismatch: {horizon}')
        scores['horizon_ms'] = horizon
        gains['horizon_ms'] = horizon
        score_rows.append(scores)
        gain_rows.append(gains)
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.concat(score_rows, ignore_index=True).sort_values(['horizon_ms', 'condition'])
    gains = pd.concat(gain_rows, ignore_index=True).sort_values(['horizon_ms', 'condition'])
    scores.to_csv(OUT / 'horizon_metrics.csv', index=False)
    gains.to_csv(OUT / 'paired_gain.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for condition, (label, color) in CONDITIONS.items():
        rows = scores.loc[scores.condition.eq(condition)].sort_values('horizon_ms')
        x = rows.horizon_ms / 1000
        axes[0].plot(x, rows.crps_w, marker='o', color=color, label=label)
        axes[1].plot(x, rows.coverage_80, marker='o', color=color, label=label)
    axes[1].axhline(.8, color='0.4', ls='--', label='Nominal 80%')
    axes[0].set(ylabel='Run-balanced CRPS (W)', xlabel='Horizon (s)')
    axes[1].set(ylabel='80% interval coverage', xlabel='Horizon (s)', ylim=(0, 1))
    for ax in axes:
        ax.set_xticks([h / 1000 for h in HORIZONS])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Highres grouped inner selection; reused holdout')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'horizon_scores.{ext}', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for condition in ('history_static_s2_controlled', 'history_static_s2_tuned'):
        label, color = CONDITIONS[condition]
        rows = gains.loc[gains.condition.eq(condition)].sort_values('horizon_ms')
        ax.errorbar(rows.horizon_ms / 1000, rows.gain_pct,
                    yerr=[rows.gain_pct - rows.ci_low_pct,
                          rows.ci_high_pct - rows.gain_pct],
                    marker='o', color=color, capsize=2, label=label)
    ax.axhline(0, color='0.4', ls='--')
    ax.set(xlabel='Horizon (s)', ylabel='Paired run CRPS gain (%)',
           xticks=[h / 1000 for h in HORIZONS])
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'horizon_gain.{ext}', dpi=300)
    plt.close(fig)
    report = [
        '# 200–2000 ms grouped-inner horizon comparison', '',
        '32 highres run의 8 GPU 합산 전력에 대한 회고적 결과다. 200 ms 완료 bin을 사용했고, 각 horizon에서 22개 개발 run 내부 선택과 기존 10개 test run을 사용했다. +1초/+2초 결과는 기존 완료 실험에서 읽었다. Horizon별 유효 target 집합이 달라 절대 CRPS를 완전한 paired 비교로 취급하지 않는다.', '',
        scores[['horizon_ms', 'condition', 'n_runs', 'crps_w', 'mae_w', 'p95_w', 'p99_w', 'coverage_80']].to_markdown(index=False, floatfmt='.3f'), '',
        '## 동일 horizon 내 metadata gain', '',
        gains[['horizon_ms', 'condition', 'gain_pct', 'ci_low_pct', 'ci_high_pct']].to_markdown(index=False, floatfmt='.3f'), '',
        'CI는 저장된 예측의 test run을 재표집한 기술적 범위다. 같은 holdout을 반복 사용했으며 원본 수집일과 GPU×workload 조합이 결합돼 있다. 80% coverage와 CRPS를 함께 판단해야 한다. GPU-only 자료로 시설 PCC 제어 효과를 주장할 수 없다.', '',
        '원자료는 horizon_metrics.csv와 paired_gain.csv, 그림은 horizon_scores 및 horizon_gain의 PNG/PDF다.',
    ]
    (OUT / 'report_ko.md').write_text('\n'.join(report) + '\n')


if __name__ == '__main__':
    main()
