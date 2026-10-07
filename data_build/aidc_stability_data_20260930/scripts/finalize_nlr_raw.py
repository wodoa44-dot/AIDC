#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import pandas as pd
from aidc_data.inventory import PROJECT,sha256
from aidc_data.splits import make
from aidc_data.events import detect
import yaml

def main():
    a=argparse.ArgumentParser();a.add_argument('--build-id',default='main_20260929');args=a.parse_args()
    out=PROJECT/'outputs'/args.build_id/'nlr_raw';q=pd.read_csv(out/'quality_audit.csv')
    if len(q)!=302:raise RuntimeError(f'expected 302 parsed raw logs, got {len(q)}')
    q['exact_duplicate_source']=q.duplicated('source_hash',keep='first')
    source_records=[]
    for path in q.source_file:
        with Path(path).open('rb') as f:
            source_records.append(sum(bool(line.strip()) and not line.lstrip().startswith(b'#') for line in f))
    q['source_data_lines']=source_records
    q['parser_dropped_rows']=q.source_data_lines-q.source_rows
    q.to_csv(out/'quality_audit.csv',index=False)
    metadata=[]
    for r in q.itertuples():
        p=Path(r.source_file)
        values={'workload_family_token':p.relative_to(PROJECT.parents[1]/'trace/dataset/nlr_312/extracted/00_raw_datasets').parts[0],
            'node_id_token':re.search(r'x\w+n\d+',p.name).group(0) if re.search(r'x\w+n\d+',p.name) else None,
            'gpu_channel_count':4}
        for field,value in values.items():
            metadata.append({'run_id':r.run_id,'split_group_id':r.parent_group,'field':field,'value':str(value) if value is not None else None,
                'unit':None,'source':str(p),'extraction_method':'path_regex' if field!='gpu_channel_count' else 'raw_header',
                'confidence':'medium' if field!='gpu_channel_count' else 'high','valid_from':None,'valid_to':None,'available_at':None,
                'availability_assumption':'static only if job config disclosed before run','evidence_type':'derived','parent_evidence_type':'measured_existing'})
    pd.DataFrame(metadata).to_parquet(out/'metadata.parquet',index=False)
    canonical=q[~q.exact_duplicate_source]
    split=make(canonical.rename(columns={'parent_group':'split_group_id'}),5,3)
    (out/'split_manifest.json').write_text(json.dumps(split,indent=2))
    files={kind:str(out/kind) for kind in ['canonical_runs','bins','forecast_index','targets','features_current_power','features_power_history','events']}
    manifest={'schema_version':'1.0','dataset_id':'nlr_raw_nvml','evidence_type':'derived','parent_evidence_type':'measured_existing',
        'boundary':'sum_of_4_gpu_channels_per_node','source_file_count':len(q),'exact_duplicate_files':int(q.exact_duplicate_source.sum()),
        'canonical_source_files':len(canonical),'conservative_parent_job_groups':canonical.parent_group.nunique(),
        'source_quality_sha256':sha256(out/'quality_audit.csv'),'files':files,'split_manifest':str(out/'split_manifest.json'),
        'feature_availability':'bin end assumed; sensor latency and timezone unverified','input_interpolation':False,
        'target':'future completed 200ms GPU-only bin average W'}
    (out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
    # Verify all per-file partitions and their row keys without concatenating large data.
    checks={'no_parser_dropped_rows':bool(q.parser_dropped_rows.eq(0).all()),'partition_count':True,'row_id_unique_per_partition':True,'causal_feature_time':True,'target_key_match':True,
        'separate_from_interpolated_aggregates':True,'split_group_disjoint':True}
    total_rows=0;total_events=0
    event_dir=out/'events';event_dir.mkdir(exist_ok=True)
    cfg=yaml.safe_load((PROJECT/'configs/events.yaml').read_text())
    cfg.update(yaml.safe_load((PROJECT/'configs/existing.yaml').read_text()))
    for r in q.itertuples():
        bins=pd.read_parquet(out/'bins'/f'{r.run_id}.parquet')
        ev=detect(bins,cfg);ev.to_parquet(event_dir/f'{r.run_id}.parquet',index=False);total_events+=len(ev)
        idx=pd.read_parquet(out/'forecast_index'/f'{r.run_id}.parquet')
        t=pd.read_parquet(out/'targets'/f'{r.run_id}.parquet')
        total_rows+=len(idx)
        checks['row_id_unique_per_partition'] &= bool(idx.row_id.is_unique)
        checks['causal_feature_time'] &= bool((idx.max_feature_available_at<=idx.issuance_time).all())
        checks['target_key_match'] &= set(idx.row_id)==set(t.row_id)
    for f in split['outer_folds']:
        checks['split_group_disjoint'] &= not bool(set(f['train_groups'])&set(f['test_groups']))
    checks['partition_count']=len(list((out/'canonical_runs').glob('*.parquet')))==len(q)
    validation={'checks':checks,'passed':all(checks.values()),'forecast_rows':total_rows,'event_count':total_events,
        'limitations':['conservative job group is not verified independent session','raw timestamp timezone and sensor bandwidth unknown','no paired control or service logs']}
    (out/'validation.json').write_text(json.dumps(validation,indent=2))
    if not validation['passed']:raise RuntimeError(validation)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
