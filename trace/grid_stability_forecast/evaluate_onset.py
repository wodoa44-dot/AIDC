"""Exploratory one-to-one matching of window alarms to 200 ms step onsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'results/actual_20260926'


def match_one_to_one(onsets, alarms, horizon_s=1.0):
    """Chronological matching; each earlier alarm and each onset is used at most once."""
    onsets = np.asarray(onsets, float)
    alarms = np.asarray(alarms, float)
    used = np.zeros(len(alarms), bool)
    pairs = []
    for onset in onsets:
        eligible = np.flatnonzero(~used & (alarms < onset) & (alarms >= onset - horizon_s - 1e-9))
        if len(eligible):
            index = int(eligible[0])
            used[index] = True
            pairs.append((float(onset), float(alarms[index]), float(onset - alarms[index])))
    return pairs, int(len(onsets) - len(pairs)), int((~used).sum())


def run(out: Path, threshold_w=100., alarm_threshold=.5, horizon_s=1., debounce_s=1.):
    target = pd.read_parquet(out / 'targets.parquet', columns=['run_id', 'segment_id',
        'issue_s', 'current', 'horizon_ms'])
    target = target.loc[target.horizon_ms.eq(1000)].drop(columns='horizon_ms')
    probability = pd.read_parquet(out / 'event_predictions.parquet')
    prediction_runs = set(probability.run_id)
    target = target.loc[target.run_id.isin(prediction_runs)]
    joined = probability.merge(target, on=['run_id', 'issue_s'], validate='many_to_one')
    if len(joined) != len(probability):
        raise AssertionError('Prediction/target pairing changed')
    records, events, alarms_out = [], [], []
    for (run_id, segment), data in target.groupby(['run_id', 'segment_id'], sort=False):
        data = data.sort_values('issue_s')
        times = data.issue_s.to_numpy(float)
        power = data.current.to_numpy(float)
        if len(times) < 10:
            continue
        valid_step = np.isclose(np.diff(times), .2, atol=1e-5)
        slope = np.diff(power)
        start = times[0] + horizon_s
        end = times[-1]
        for label, direction in [('rise_event', 1), ('drop_event', -1)]:
            candidate = times[1:][valid_step & (direction * slope >= threshold_w)]
            candidate = candidate[(candidate >= start) & (candidate <= end)]
            truth = []
            for t in candidate:
                if not truth or t - truth[-1] > debounce_s + 1e-9:
                    truth.append(float(t))
            event_rows = joined.loc[joined.run_id.eq(run_id) & joined.segment_id.eq(segment) & joined.event.eq(label)]
            for condition, scored in event_rows.groupby('condition'):
                scored = scored.sort_values('issue_s')
                alarm = scored.probability.to_numpy() >= alarm_threshold
                issue = scored.issue_s.to_numpy(float)
                consecutive = np.r_[False, np.isclose(np.diff(issue), .2, atol=1e-5)]
                onset_mask = alarm & ~(np.r_[False, alarm[:-1]] & consecutive)
                alerts = issue[onset_mask]
                # Full future verification is unavailable at the right edge.
                alerts = alerts[(alerts >= start) & (alerts <= end - horizon_s)]
                pairs, misses, false = match_one_to_one(truth, alerts, horizon_s)
                duration_h = max(0., end - start) / 3600
                records.append(dict(run_id=run_id, segment_id=segment, condition=condition,
                    direction=label, n_onsets=len(truth), n_alarms=len(alerts), n_matched=len(pairs),
                    n_missed=misses, n_false_alarms=false, observed_hours=duration_h,
                    mean_lead_s=np.mean([p[2] for p in pairs]) if pairs else np.nan))
                used_alarm = {p[1] for p in pairs}
                matched_onset = {p[0] for p in pairs}
                for onset in truth:
                    events.append(dict(run_id=run_id, segment_id=segment, condition=condition,
                        direction=label, onset_s=onset, matched=onset in matched_onset))
                for alert in alerts:
                    alarms_out.append(dict(run_id=run_id, segment_id=segment, condition=condition,
                        direction=label, alarm_issue_s=float(alert), matched=float(alert) in used_alarm))
    per_segment = pd.DataFrame(records)
    per_segment.to_csv(out / 'onset_segment_metrics.csv', index=False)
    pd.DataFrame(events).to_csv(out / 'onset_events.csv', index=False)
    pd.DataFrame(alarms_out).to_csv(out / 'onset_alarms.csv', index=False)
    per_run = per_segment.groupby(['run_id', 'condition', 'direction'], as_index=False).agg(
        n_onsets=('n_onsets', 'sum'), n_alarms=('n_alarms', 'sum'),
        n_matched=('n_matched', 'sum'), n_missed=('n_missed', 'sum'),
        n_false_alarms=('n_false_alarms', 'sum'), observed_hours=('observed_hours', 'sum'))
    per_run['recall'] = per_run.n_matched / per_run.n_onsets.replace(0, np.nan)
    per_run['precision'] = per_run.n_matched / per_run.n_alarms.replace(0, np.nan)
    per_run['false_alarms_per_hour'] = per_run.n_false_alarms / per_run.observed_hours.replace(0, np.nan)
    per_run.to_csv(out / 'onset_per_run_metrics.csv', index=False)
    aggregate = per_run.groupby(['condition', 'direction'], as_index=False).agg(
        n_runs=('run_id', 'nunique'), n_onsets=('n_onsets', 'sum'),
        n_alarms=('n_alarms', 'sum'), n_matched=('n_matched', 'sum'),
        n_missed=('n_missed', 'sum'), n_false_alarms=('n_false_alarms', 'sum'),
        observed_hours=('observed_hours', 'sum'))
    aggregate['recall'] = aggregate.n_matched / aggregate.n_onsets.replace(0, np.nan)
    aggregate['precision'] = aggregate.n_matched / aggregate.n_alarms.replace(0, np.nan)
    aggregate['false_alarms_per_hour'] = aggregate.n_false_alarms / aggregate.observed_hours
    aggregate.to_csv(out / 'onset_aggregate_metrics.csv', index=False)
    checks = dict(one_to_one=bool((aggregate.n_matched <= aggregate.n_alarms).all() and
        (aggregate.n_matched <= aggregate.n_onsets).all()),
        counts_reconcile=bool((aggregate.n_matched + aggregate.n_missed == aggregate.n_onsets).all() and
        (aggregate.n_matched + aggregate.n_false_alarms == aggregate.n_alarms).all()),
        paired_conditions=bool(aggregate.condition.nunique() == 3),
        test_runs=int(aggregate.n_runs.min()), threshold_w=threshold_w,
        alarm_threshold=alarm_threshold, lead_window_s=horizon_s,
        event_debounce_s=debounce_s, event_definition='adjacent completed 200 ms bin step',
        limitation='classifier target is any future excursion relative to issue; this onset definition differs')
    checks['all_passed'] = checks['one_to_one'] and checks['counts_reconcile'] and checks['paired_conditions'] and checks['test_runs'] == 10
    (out / 'onset_verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    if not checks['all_passed']:
        raise AssertionError(checks)
    report = [
        '# +1초 event-window 확률의 200 ms step onset 탐색 평가', '',
        '기존 10개 highres test run의 저장 확률을 다시 사용했다. 인접한 완료 200 ms bin의 전력 차이가 100 W 이상인 시점을 onset으로 정의하고, 같은 방향의 onset을 1초 이내에서는 하나로 합쳤다. 확률 0.5 이상으로 올라가는 첫 발행 시점을 경보로 삼고, 다음 1초 내 onset 하나에만 연결했다. 처음 1초의 onset과 마지막 1초의 경보는 완전한 관측창이 없어 제외했다. 경보 threshold와 이 규칙은 test 결과를 보고 최적화하지 않았다.', '',
        aggregate.to_markdown(index=False, floatfmt='.3f'), '',
        '표의 false_alarms_per_hour는 중복 window가 아니라 일대일 경보 episode 기준이다. 검출된 onset만의 lead time으로 성능을 요약하지 않고 missed/false alarm 수를 병기한다. 사건 ID와 경보 ID는 onset_events.csv, onset_alarms.csv에 있다.', '',
        '중요한 한계: 학습 label은 발행 시점 전력에 대한 미래 1초 내 누적 excursion이며 여기의 인접-bin step과 정의가 다르다. 따라서 이 평가는 학습된 onset 예측기의 성능이 아니다. 원본 약 20 ms 기록의 물리 센서 평균/지연, 독립 수집 session, 시설/PCC 부하는 확인되지 않았다. 100 W는 계통 허용치가 아닌 호환성 scenario 값이다.', '',
        '재현: `python trace/grid_stability_forecast/evaluate_onset.py`', ''
    ]
    (out / 'onset_report_ko.md').write_text('\n'.join(report))
    return aggregate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT)
    print(run(parser.parse_args().out).to_string(index=False))
