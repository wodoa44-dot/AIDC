"""Report the same-acquisition dynamic-metadata pilot from stored predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import QUANTILES

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'results/online_runtime_pilot_20260927'
CONDITIONS = ('history_only', 'history_static_rate', 'history_dynamic', 'history_static_dynamic')


def raw_integrated_pinball(y, predictions):
    """Score uncorrected crossing quantiles; this is not a valid quantile function."""
    grid = np.r_[0., QUANTILES, 1.]
    values = np.column_stack([predictions[:, 0], predictions, predictions[:, -1]])
    residual = y[:, None] - values
    return 2 * np.trapz(np.maximum(grid * residual, (grid-1) * residual), grid, axis=1)


def summarize(out):
    out = Path(out)
    pred = pd.read_parquet(out / 'per_target_predictions.parquet')
    per = pd.read_csv(out / 'per_run_metrics.csv')
    agg = pd.read_csv(out / 'aggregate_metrics.csv')
    gain = pd.read_csv(out / 'paired_gain_descriptive.csv')
    split = pd.read_csv(out / 'split_manifest.csv')
    crossing_rows = []
    for condition in CONDITIONS:
        frame = pred.loc[pred.condition.eq(condition)]
        raw = np.column_stack([np.load(out / 'cells' / f'{condition}_q{q:g}.npy') for q in QUANTILES])
        if len(frame) != len(raw):
            raise AssertionError('Prediction cell order changed')
        initial = raw_integrated_pinball(frame.y.to_numpy(), raw)
        by_run = pd.DataFrame({'run_id': frame.run_id.to_numpy(), 'before': initial,
            'after': frame.crps_w.to_numpy(), 'crossing': np.any(np.diff(raw, axis=1)<0, axis=1),
            'q50_shift_w': np.maximum.accumulate(raw, axis=1)[:, 7] - raw[:, 7]})
        runmean = by_run.groupby('run_id')[['before', 'after', 'crossing', 'q50_shift_w']].mean()
        crossing_rows.append(dict(condition=condition, n_runs=len(runmean),
            uncorrected_integrated_pinball_w=runmean.before.mean(),
            corrected_crps_w=runmean.after.mean(),
            crossing_fraction=runmean.crossing.mean(), mean_q50_correction_w=runmean.q50_shift_w.mean()))
    crossing = pd.DataFrame(crossing_rows)
    crossing.to_csv(out / 'crossing_effects.csv', index=False)
    test_runs = split.loc[split.split.eq('test'), ['run_id', 'request_rate']]
    strata = per.merge(test_runs, on='run_id', validate='many_to_one')
    strata['request_rate_band'] = np.where(strata.request_rate.le(75), '51-75', '76-100')
    by_band = strata.groupby(['request_rate_band', 'condition'], as_index=False).agg(
        n_runs=('run_id', 'nunique'), crps_w=('crps_w', 'mean'),
        coverage_80=('coverage_80', 'mean'))
    by_band.to_csv(out / 'rate_strata.csv', index=False)
    report = [
        '# NLR online-rate: timestamped vLLM runtime metadata 파일럿', '',
        '원본 `vllm_server.log`의 약 10초 간격 Running/Waiting 요청 수, prompt/generation token throughput, KV cache 사용률을 추출했다. `postprocess.py`가 원래 시작 시각에서 1분을 제외한 `start_time_exc`로 3분 전력창을 만들었음을 확인했다. 이 시각으로 서버 로그를 상대 시간에 맞추고 로그 시각에 1초 가용성 여유를 더했다. 첫 원시 전력 샘플의 subsecond offset과 장치 간 clock 동기화는 독립 검증되지 않았다. 200개 run 모두에서 17–18개 로그가 정렬됐고, 모델 입력은 발행 시점보다 이전에 가용한 마지막 기록만 사용했다.', '',
        '원본 자료 설명의 upstream cadence는 약 200 ms이며, 제공된 power 파일은 1 ms 보간 grid다. 물리 센서 cadence는 직접 검증되지 않았다. 이 파일럿에서는 200 ms 완료 bin으로 분석했다. 보간을 더 빠른 실측 근거로 해석할 수 없다. 전체 200 run은 모두 `acquisition-nlr_online_rate` 하나의 수집 그룹이다. ID 1–150을 학습, 151–200을 test로 둔 시간순 탐색이며 독립 session 검증이 아니다. Test에는 같은 42,957개 발행 시점을 모든 조건에 사용하고 학습은 run당 최대 500행, 균등 run weight, 15 quantiles, 고정 100-tree LightGBM을 적용했다.', '',
        '| 조건 | run 균형 CRPS (W) | MAE (W) | p95 (W) | 80% coverage | history 대비 변화 |',
        '|---|---:|---:|---:|---:|---:|'
    ]
    gains = gain.set_index('condition')
    for row in agg.itertuples():
        change = '기준' if row.condition == 'history_only' else f"{gains.loc[row.condition, 'descriptive_gain_pct']:+.2f}%"
        report.append(f'| {row.condition} | {row.crps_w:.2f} | {row.mae_w:.2f} | {row.p95_w:.1f} | {row.coverage_80:.3f} | {change} |')
    report += ['', 'Configured request rate만 더한 B2는 이 test에서 악화했고, 동적 신호만 더한 B3는 0.59% 낮은 CRPS를 보였다. B4는 history-only와 거의 같았다. 개선 run 수는 B2/B3/B4 각각 3/32/26개다. 독립 수집 단위가 하나여서 50 run을 독립 표본으로 bootstrap하지 않았고 통계적 유용성 판정은 불충분하다. 특히 B3 coverage는 0.714로 history-only 0.760보다 낮다.', '',
        'Quantile crossing 보정: 독립 quantile 모델의 예측을 행별 누적 최대값으로 정렬했다. 아래의 보정 전 값은 crossing이 있는 상태의 적분 pinball loss이며 정식 CRPS가 아니다.', '',
        crossing.to_markdown(index=False, floatfmt='.3f'), '',
        '요청률 구간별 기술 통계(사후 탐색이며 새 효과 주장이 아님):', '',
        by_band.to_markdown(index=False, floatfmt='.3f'), '',
        '이 파일럿은 약 10초 시점별 queue/throughput 로그의 존재와 causal join 가능성만 확인한다. 원시 개별 요청 arrival/latency, serving 시계 동기화, 다른 수집 session으로의 일반화, 시설/PCC 제어 효과는 확인하지 못했다. 로그의 rolling metric window가 정확히 무엇인지는 원본 logger 구현으로 추가 확인해야 한다.', '',
        '재현: `python trace/grid_stability_forecast/extract_online_runtime.py`; `python trace/grid_stability_forecast/online_runtime_pilot.py`; `python trace/grid_stability_forecast/summarize_online_runtime.py`.', ''
    ]
    (out / 'report_ko.md').write_text('\n'.join(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT)
    summarize(parser.parse_args().out)
