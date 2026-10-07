#!/usr/bin/env python3
from pathlib import Path
import sys,json
import pandas as pd
import yaml
from aidc_data.canonical import completed_bins
from aidc_data.inventory import PROJECT

out=PROJECT/'outputs'/sys.argv[1];base=yaml.safe_load((PROJECT/'configs/existing.yaml').read_text())
summary=[]
for width in [.05,.1,.5,1.]:
    cfg=dict(base,bin_seconds=width)
    dst=out/f'bins_{int(width*1000)}ms';dst.mkdir(exist_ok=True)
    valid=total=0
    for p in sorted((out/'canonical_runs').glob('*.parquet')):
        c=pd.read_parquet(p)
        med=float(c.elapsed_s.diff().loc[lambda x:x.gt(0)].median())
        b=completed_bins(c,med,cfg)
        b.to_parquet(dst/p.name,index=False)
        total+=len(b);valid+=int(b.valid.sum())
    summary.append({'bin_seconds':width,'bin_count':total,'valid_bins':valid,'coverage_required':cfg['min_time_coverage'],
        'max_hold_native_median_factor':cfg['max_hold_dt_factor'],'sensor_bandwidth_verified':False,'evidence_type':'derived',
        'parent_evidence_type':'measured_existing'})
pd.DataFrame(summary).to_csv(out/'resolution_export_summary.csv',index=False)
print(json.dumps(summary,indent=2))
