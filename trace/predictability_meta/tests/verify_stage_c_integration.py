"""Verify the intentionally interrupted run against an independent uninterrupted run."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_c import file_hash, Tasks
from corrected_experiment import HERE

resume = HERE/'outputs/stage_c_dev_resume_20260915'
baseline = HERE/'outputs/stage_c_dev_uninterrupted_20260915'
assert json.loads((resume/'status.json').read_text())['status'] == 'intentionally_interrupted'
completed = list((resume/'tasks').glob('*/complete.json'))
assert len(completed) == 15
snapshot = {str(p): (p.stat().st_mtime_ns, file_hash(p)) for c in completed for p in c.parent.iterdir() if p.is_file()}
# An incomplete task directory must never count as complete.
incomplete = resume/'tasks'/'incomplete_probe'
incomplete.mkdir(exist_ok=True)
(incomplete/'result.joblib').write_bytes(b'incomplete')
for out in [resume, baseline]:
    subprocess.run([sys.executable, str(HERE/'corrected_experiment.py'), '--experiment-id', out.name,
                    '--development-integration'], check=True)
for path, expected in snapshot.items():
    p = Path(path)
    assert (p.stat().st_mtime_ns, file_hash(p)) == expected, f'Completed task refitted: {p}'
status = json.loads((resume/'status.json').read_text())
assert status['skipped_valid_tasks'] == 15
assert status['completed_tasks_this_invocation'] == 15
for name in ['per_run_seed_metrics.csv', 'per_run_metrics.csv', 'aggregate_metrics.csv', 'comparison_summary.csv']:
    pd.testing.assert_frame_equal(pd.read_csv(resume/name), pd.read_csv(baseline/name), atol=1e-10, rtol=1e-10)
maximum = 0.
for p in (resume/'tasks').glob('*/predictions.parquet'):
    other = baseline/'tasks'/p.parent.name/p.name
    a, b = pd.read_parquet(p), pd.read_parquet(other)
    q = [col for col in a if col.startswith('q_')]
    np.testing.assert_allclose(a[q], b[q], atol=1e-10, rtol=1e-10)
    maximum = max(maximum, float(np.max(np.abs(a[q].to_numpy()-b[q].to_numpy()))))
    assert not a.duplicated(['assessment','resolution_ms','horizon_ms','outer_fold','pair','condition','seed','target_index']).any()
# Configuration mismatch must reject, not silently reuse checkpoints.
cfg = json.loads((resume/'effective_config.json').read_text())
try:
    Tasks(resume, cfg | {'final_seeds': [17]})
except ValueError:
    pass
else:
    raise AssertionError('Incompatible checkpoints were accepted')
# A matching incomplete checkpoint is recomputed; a valid one is skipped.
tasks = Tasks(resume, cfg)
identity = {'stage': 'integration_probe', 'cell': '200/1000', 'outer_fold': 1,
            'condition': 'history_only', 'candidate': 0, 'seed': 17}
calls = []
def compute(directory):
    calls.append(1)
    (directory/'payload.txt').write_text('complete')
    return 7
assert tasks.run(identity, compute) == 7
probe = next(p for p in (resume/'tasks').glob('*/complete.json')
             if json.loads(p.read_text())['identity'].get('stage') == 'integration_probe')
probe.unlink()
assert tasks.run(identity, compute) == 7
assert tasks.run(identity, compute) == 7
assert len(calls) == 2
result = {'passed': True, 'atol': 1e-10, 'rtol': 1e-10, 'max_prediction_difference': maximum,
          'completed_tasks_preserved': 15, 'remaining_tasks_completed': 15,
          'incomplete_task_recomputed': True, 'incompatible_config_rejected': True,
          'assessment': 'development_only', 'holdout_runs_used': 0}
(resume/'integration_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
