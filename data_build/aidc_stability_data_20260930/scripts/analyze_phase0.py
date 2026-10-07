"""Analyze only supplied command/power pairs; never create measured samples."""
from pathlib import Path
import argparse
import json
import pandas as pd
from aidc_data.phase0.response import response_metrics


def analyze(source, out):
    out.mkdir(parents=True, exist_ok=True)
    results, missing = [], []
    for path in sorted(source.glob('*_commands.parquet')):
        sample_path = path.with_name(path.name.replace('_commands.parquet', '_nvml.parquet'))
        if not sample_path.exists():
            missing.append(str(sample_path))
            continue
        samples = pd.read_parquet(sample_path)
        commands = pd.read_parquet(path).sort_values('intended_monotonic_ns')
        for _, group in commands.groupby(['trial_id', 'gpu_uuid']):
            records = group.to_dict('records')
            for i, command in enumerate(records):
                end = records[i + 1]['intended_monotonic_ns'] if i + 1 < len(records) else None
                results.append(response_metrics(samples, command, end_ns=end))
    if results:
        pd.DataFrame(results).to_parquet(out / 'phase0_response_metrics.parquet', index=False)
    report = {'status': 'requires_paired_trials' if results else 'blocked_source',
              'response_analysis_implemented': True, 'commands_analyzed': len(results),
              'valid_observed_responses': sum(r['valid'] for r in results),
              'missing_sample_files': missing, 'causal_effect_identified': False,
              'performance_effect_identified': False,
              'reason': 'No-treatment matched trials and service performance logs are required for attribution.',
              'source': str(source.resolve()),
              'definitions': {'baseline_s': 1., 'plateau_s': 1., 'dwell_s': .2,
                              'response_fraction': .9, 'minimum_delta_w': 5.}}
    (out / 'phase0_identification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output)
