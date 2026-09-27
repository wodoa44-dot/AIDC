#!/usr/bin/env python
"""Persistent read-only production monitor, with progress documentation and final checks."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import pandas as pd
from stage_c import file_hash
from stage_b import candidates
from corrected_experiment import HERE, atomic_json


def find_pid(experiment):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p/'cmdline').read_bytes().decode().split('\0')
            if any(x.endswith('/corrected_experiment.py') for x in args) and '--experiment-id' in args:
                if args[args.index('--experiment-id')+1] == experiment:
                    return int(p.name)
        except (OSError, UnicodeError, IndexError):
            pass
    return None


def final_checks(out):
    cfg = json.loads((out/'effective_config.json').read_text())
    assert not cfg.get('development_integration', False)
    grid = pd.read_csv(out/'grid_manifest.csv').query('included')
    assert len(grid) == 19
    assert cfg['budgets_per_run'] == [250, 1000, 5000, 'all']
    assert cfg['boosting_rounds'] == [12, 50, 100, 300]
    assert cfg['final_seeds'] == [17, 42, 2026]
    assert len(candidates(cfg)) == 24
    records = [json.loads(p.read_text()) for p in (out/'tasks').glob('*/complete.json')]
    counts = Counter(r['identity']['stage'] for r in records)
    eligible = sum(sum(p['history_ms'] >= c.resolution_ms and p['history_ms'] % c.resolution_ms == 0
                       for p in candidates(cfg)) for c in grid.itertuples())
    expected = {'A': 19*cfg['outer_folds']*cfg['inner_folds']*4*2,
                'B': eligible*cfg['outer_folds']*cfg['inner_folds']*2,
                'B_final': eligible*cfg['inner_folds']*2,
                'C': 19*(cfg['outer_folds']+1)*3*2*3}
    assert dict(counts) == expected, (counts, expected)
    for p in (out/'tasks').glob('*/complete.json'):
        record = json.loads(p.read_text())
        for name, checksum in record['files'].items():
            assert file_hash(p.parent/name) == checksum, str(p.parent/name)
    metrics = pd.read_csv(out/'per_run_seed_metrics.csv')
    keys = ['assessment','resolution_ms','horizon_ms','outer_fold','pair','condition','seed','run_id']
    assert not metrics.duplicated(keys).any()
    assert set(metrics.assessment) == {'grouped_outer', 'reused_holdout'}
    freeze = json.loads((out/'holdout_selection_freeze.json').read_text())
    assert file_hash(out/'final_development_selections.json') == freeze['selection_hash']
    assert (out/'report_en.md').is_file()
    result = {'passed': True, 'task_counts': dict(counts), 'expected_task_counts': expected,
              'all_artifact_checksums_verified': True, 'duplicate_metric_keys': 0,
              'finished_utc': time.strftime('%FT%TZ', time.gmtime())}
    atomic_json(out/'production_verification.json', result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment-id', required=True)
    args = ap.parse_args()
    out = HERE/'outputs'/args.experiment_id
    pid = find_pid(args.experiment_id)
    while True:
        counts = Counter()
        latest = 0.
        for p in (out/'tasks').glob('*/complete.json'):
            counts[json.loads(p.read_text())['identity']['stage']] += 1
            latest = max(latest, p.stat().st_mtime)
        status = json.loads((out/'status.json').read_text())
        progress = {'experiment_id': args.experiment_id, 'pid': pid, 'status': status['status'],
                    'committed_tasks': dict(counts), 'updated_utc': time.strftime('%FT%TZ', time.gmtime()),
                    'last_commit_utc': time.strftime('%FT%TZ', time.gmtime(latest)) if latest else None}
        atomic_json(out/'production_progress.json', progress)
        resume = HERE/'RESUME.md'
        text = resume.read_text().split('\n## Live production progress\n')[0]
        resume.write_text(text+'\n## Live production progress\n\n```json\n'+json.dumps(progress, indent=2)+'\n```\n')
        print(json.dumps(progress), flush=True)
        if status['status'] == 'completed':
            result = final_checks(out)
            print(json.dumps(result), flush=True)
            with resume.open('a') as f:
                f.write('\nProduction completed and final integrity checks passed. See '
                        '`outputs/'+args.experiment_id+'/production_verification.json` and `report_en.md`.\n')
            return
        if status['status'] == 'failed':
            raise RuntimeError(f'Production failed: {status}')
        if pid is None or not Path(f'/proc/{pid}').exists():
            atomic_json(out/'monitor_failure.json', {'error': 'Production process disappeared without completed status',
                                                     'last_progress': progress})
            raise RuntimeError('Production process disappeared without completed status')
        time.sleep(30)


if __name__ == '__main__':
    main()
