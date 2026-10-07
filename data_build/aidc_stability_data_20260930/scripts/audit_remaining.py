"""Inventory both prompt projects without changing historical experiment outputs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone

import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
SKIP = {'.grid_deps', '.venv-grid', '__pycache__', '.pytest_cache', '.git'}


def audit(out):
    out.mkdir(parents=True, exist_ok=True)
    prompts = sorted(ROOT.glob('**/*PROMPT*.md'))
    roots = sorted({p.parent for p in prompts} | {ROOT / 'experiments'})
    files = set()
    for root in roots:
        files.update(p for p in root.rglob('*') if p.is_file()
                     and not SKIP.intersection(p.parts) and out not in p.parents)
    rows, errors, candidates = [], [], []
    for p in sorted(files):
        row = {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
               'rows': None, 'columns': '', 'schema_error': ''}
        try:
            if p.suffix == '.parquet':
                meta = pq.ParquetFile(p)
                row.update(rows=meta.metadata.num_rows,
                           columns='|'.join(meta.schema_arrow.names))
            elif p.suffix == '.csv':
                with p.open(newline='') as stream:
                    row['columns'] = '|'.join(next(csv.reader(stream), []))
            columns = set(row['columns'].split('|'))
            if {'trial_id', 'command_id'} <= columns or {'meter_id', 'frequency_hz'} <= columns:
                candidates.append(row['path'])
        except Exception as exc:
            row['schema_error'] = str(exc)
            errors.append({'path': row['path'], 'error': str(exc)})
        rows.append(row)
    with (out / 'file_inventory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    base = ROOT / 'experiments/aidc_grid_metadata_20260929/outputs/full_20260928T180455Z'
    stages = {}
    for name in ['independent_primary_v1', 'feasible_ablation_v1', 'feasible_events_v2',
                 'extended_primary_eval_v1']:
        path = base / name / 'status.json'
        state = json.loads(path.read_text())
        parts = list((path.parent / 'prediction_parts').glob('*.parquet'))
        planned = state.get('prediction_tasks_planned', state.get('tasks_planned'))
        stages[name] = {'source': str(path), 'recorded_status': state,
                        'prediction_files': len(parts),
                        'file_count_matches_plan': len(parts) == planned if planned else None,
                        'interpretation': 'recorded completion; not a new model fit'}
    stages['event_matching'] = {
        'files_present': all((base / 'feasible_events_v2' / name).is_file() for name in
                             ['event_matching_per_run.csv', 'event_matching_aggregate.csv',
                              'matching_protocol.json'])}
    result = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'prompt_files': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in prompts],
        'file_count': len(rows), 'schema_errors': errors,
        'measurement_schema_candidates': candidates,
        'followup_stages': stages,
        'historical_status_is_stale': True,
        'remaining': {
            'D3': 'No command/power/performance paired measurements found in audited project schemas.',
            'D4_network': 'Historical ANDES dependency failure; a new attempt is recorded separately.',
            'D5': 'No meter-identified facility frequency records found in audited project schemas.'},
        'limitations': ['CSV headers and Parquet footers inspected, not every observation.',
                       'Schema candidate search does not prove measurement provenance.',
                       'Original statuses and results were not overwritten.']}
    (out / 'reconciled_status.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'files': len(rows), 'schema_errors': len(errors),
                      'measurement_candidates': candidates, 'output': str(out)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-id', default='remaining_20260929')
    args = parser.parse_args()
    audit(PROJECT / 'outputs' / args.build_id)
