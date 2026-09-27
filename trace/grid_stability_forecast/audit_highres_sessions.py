"""Audit highres collection-time overlap and date/cohort confounding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT = HERE / 'results/actual_20260926'


def audit(out):
    out = Path(out)
    manifest = pd.read_csv(ROOT / 'trace/preprocessing/outputs/manifest.csv')
    manifest = manifest.loc[manifest.dataset.eq('highres_node') & ~manifest.is_duplicate]
    metadata = pd.read_parquet(ROOT / 'trace/preprocessing/outputs/forecast_v2_complete_20260916/run_metadata.parquet')
    metadata = metadata[['run_id', 'gpu_model', 'workload_type']].drop_duplicates('run_id')
    rows = []
    for record in manifest.itertuples():
        timestamps = pd.to_datetime(pd.read_csv(record.source_path, usecols=['timestamp']).timestamp,
                                    errors='coerce')
        rows.append(dict(run_id=record.run_id, source_path=record.source_path,
            first_timestamp=timestamps.min(), last_timestamp=timestamps.max(),
            valid_timestamps=int(timestamps.notna().sum())))
    intervals = pd.DataFrame(rows).merge(metadata, on='run_id', validate='one_to_one')
    intervals['collection_day'] = intervals.first_timestamp.dt.strftime('%Y-%m-%d')
    intervals['cohort'] = intervals.gpu_model + '__' + intervals.workload_type
    intervals = intervals.sort_values('first_timestamp').reset_index(drop=True)
    intervals.to_csv(out / 'highres_source_intervals.csv', index=False)
    mapping = intervals.groupby(['collection_day', 'cohort'], as_index=False).agg(
        n_runs=('run_id', 'size'), first_timestamp=('first_timestamp', 'min'),
        last_timestamp=('last_timestamp', 'max'))
    mapping.to_csv(out / 'highres_day_cohort.csv', index=False)
    overlap = []
    for i, a in intervals.iterrows():
        for b in intervals.iloc[i+1:].itertuples():
            if a.first_timestamp < b.last_timestamp and b.first_timestamp < a.last_timestamp:
                overlap.append((a.run_id, b.run_id))
    checks = dict(canonical_runs=len(intervals), collection_days=intervals.collection_day.nunique(),
        cohorts=intervals.cohort.nunique(), overlapping_pairs=len(overlap),
        each_day_single_cohort=bool(mapping.groupby('collection_day').cohort.nunique().eq(1).all()),
        each_cohort_single_day=bool(mapping.groupby('cohort').collection_day.nunique().eq(1).all()),
        no_overlapping_timestamps=not overlap)
    checks['all_passed'] = checks['canonical_runs']==32 and checks['collection_days']==4 and \
        checks['cohorts']==4 and checks['each_day_single_cohort'] and \
        checks['each_cohort_single_day'] and checks['no_overlapping_timestamps']
    (out / 'highres_session_audit.json').write_text(json.dumps(checks, indent=2) + '\n')
    report = ['# highres 원본 수집 시각 audit', '',
        '원본 CSV timestamp를 직접 읽어 canonical node 32개 trace의 시작·끝을 비교했다. 네 날짜의 기록은 서로 겹치지 않는다. 다만 한 날짜에는 GPU×workload 조합 하나만 있고 각 조합도 날짜 하나에만 있다.', '',
        mapping.to_markdown(index=False), '',
        '따라서 수집일과 조합의 효과는 이 자료에서 분리할 수 없다. 기존 `unseen_condition_20260926`의 조합 holdout은 정확히 날짜 holdout이기도 하다. 반대로 기존 5-fold run-level CV는 같은 날짜·조합의 기록을 train/test 양쪽에 둔다. 같은 날짜의 독립 session 여부는 timestamp 비중첩만으로 확정할 수 없으므로 run-cluster CI를 독립 수집일 CI로 읽어서는 안 된다.', '',
        '재현: `python trace/grid_stability_forecast/audit_highres_sessions.py`', '']
    (out / 'highres_session_report_ko.md').write_text('\n'.join(report))
    if not checks['all_passed']:
        raise AssertionError(checks)
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT)
    print(json.dumps(audit(parser.parse_args().out), indent=2))
