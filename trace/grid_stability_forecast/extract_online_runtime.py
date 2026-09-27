"""Extract timestamped vLLM runtime counters without reading future log lines."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / 'trace/dataset/nlr_312/extracted/00_raw_datasets/inference_online_rate_llama3_70b'
META = ROOT / 'trace/dataset/nlr_312/extracted/01_aggregated_datasets/inference_online_rate_llama3_70b/metadata.csv'
RUN_META = ROOT / 'trace/preprocessing/outputs/forecast_v2_complete_20260916/run_metadata.parquet'
DEFAULT = HERE / 'results/online_runtime_20260927'
STAMP = re.compile(r'INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*?Engine 000: Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%')


def extract(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(META, dtype={'id': str})
    metadata['id'] = metadata.id.str.zfill(6)
    metadata['start'] = pd.to_datetime(metadata.start_time_exc)
    metadata['end'] = pd.to_datetime(metadata.end_time_exc)
    runmeta = pd.read_parquet(RUN_META)
    runmeta = runmeta.loc[runmeta.dataset_id.eq('nlr_online_rate'), ['run_id', 'original_run_id', 'group_id']]
    runmeta = runmeta.rename(columns={'original_run_id': 'id'})
    metadata = metadata.merge(runmeta, on='id', validate='one_to_one')
    year = int(metadata.start.dt.year.mode().iloc[0])
    observations = []
    with (RAW / 'vllm_server.log').open(errors='replace') as stream:
        for line in stream:
            match = STAMP.search(line)
            if match:
                stamp, prompt, generation, running, waiting, cache = match.groups()
                observations.append((pd.Timestamp(f'{year}-{stamp}'), float(prompt),
                    float(generation), int(running), int(waiting), float(cache)))
    logs = pd.DataFrame(observations, columns=['log_time', 'prompt_tokens_s',
        'generation_tokens_s', 'running_requests', 'waiting_requests', 'kv_cache_pct'])
    logs = logs.sort_values('log_time').drop_duplicates('log_time', keep='last')
    starts = metadata[['id', 'run_id', 'group_id', 'request_rate', 'start', 'end']].sort_values('start')
    aligned = pd.merge_asof(logs, starts, left_on='log_time', right_on='start', direction='backward')
    aligned = aligned.loc[aligned.log_time.le(aligned.end)].copy()
    aligned['time_since_power_start_s'] = (aligned.log_time - aligned.start).dt.total_seconds()
    aligned['effective_available_s'] = aligned.time_since_power_start_s + 1.0
    aligned.to_parquet(out / 'runtime_observations.parquet', index=False)
    audit = metadata[['id', 'run_id', 'group_id', 'request_rate', 'start', 'end']].copy()
    counts = aligned.groupby('run_id').agg(n_logs=('log_time', 'size'),
        first_log_s=('time_since_power_start_s', 'min'), last_log_s=('time_since_power_start_s', 'max'),
        max_waiting=('waiting_requests', 'max')).reset_index()
    audit = audit.merge(counts, on='run_id', how='left', validate='one_to_one')
    audit.to_csv(out / 'runtime_coverage.csv', index=False)
    (out / 'source_audit.json').write_text(json.dumps({
        'server_log': str(RAW / 'vllm_server.log'), 'metadata': str(META),
        'raw_log_records': len(logs), 'aligned_records': len(aligned),
        'runs_with_logs': int(audit.n_logs.notna().sum()), 'runs_total': len(audit),
        'timestamp_resolution_s': 1, 'availability_lag_assumption_s': 1,
        'power_time_origin': 'metadata.start_time_exc; source postprocess.py adds 1 minute to experiment start, then extracts a 3 minute power interval; first raw sample may add subsecond offset',
        'metric_window': 'vLLM rolling throughput interval not verified from raw logger implementation',
        'independence': 'all runs share one acquisition group; no independent-session validation',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT)
    extract(parser.parse_args().out)
