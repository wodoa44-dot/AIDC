#!/usr/bin/env python3
"""Independent NLR raw NVML track. Never uses published interpolated aggregates."""
from __future__ import annotations
import argparse, hashlib, json, re
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from aidc_data.canonical import completed_bins
from aidc_data.forecasting import build_views
from aidc_data.inventory import ROOT,PROJECT,sha256

RAW=ROOT/'trace/dataset/nlr_312/extracted/00_raw_datasets'
COLS=['timestamp','reading_time_ns']+[f'gpu{i}_mw' for i in range(4)]+[f'gpu{i}_c' for i in range(4)]
def file_id(p):return hashlib.sha256(str(p.relative_to(RAW)).encode()).hexdigest()[:16]
def convert(p,cfg):
    data=pd.read_csv(p,comment='#',sep=r'\s+',names=COLS,on_bad_lines='skip')
    ts=pd.to_datetime(data.timestamp,format='%Y-%m-%d_%H:%M:%S.%f',errors='coerce')
    power=data[[f'gpu{i}_mw' for i in range(4)]].apply(pd.to_numeric,errors='coerce')
    complete=power.notna().all(axis=1)&np.isfinite(power).all(axis=1)
    watts=(power.sum(axis=1)/1000).where(complete)
    elapsed=(ts-ts.iloc[0]).dt.total_seconds();delta=elapsed.diff();median=float(delta[delta.gt(0)].median())
    gap=delta.gt(5*median);reset=delta.le(0)|gap|ts.isna();seg=reset.cumsum().astype(int)
    rel=p.relative_to(RAW);job=re.search(r'(?:slurmid_|nvml_wattameter_)(\d+)',p.name)
    group=f'nlr_raw::{rel.parts[0]}::{job.group(1) if job else rel.parent.as_posix()}'
    run=f'nlr_raw-{file_id(p)}';h=sha256(p)
    flags=np.where(ts.isna(),'bad_timestamp',np.where(delta.le(0),'nonmonotonic',np.where(gap,'gap_start',np.where(~complete,'missing_gpu',''))))
    c=pd.DataFrame({'dataset_id':'nlr_raw_nvml','source_file':str(p),'source_hash':h,'session_id':group,'split_group_id':group,
        'run_id':run,'segment_id':seg,'timestamp':ts,'elapsed_s':elapsed,'sample_index':np.arange(len(data)),
        'node_id':re.search(r'x\w+n\d+',p.name).group(0) if re.search(r'x\w+n\d+',p.name) else None,
        'gpu_uuid':None,'measurement_boundary':'sum_of_4_gpu_channels','evidence_type':'measured_existing',
        'gpu_power_w':watts,'gpu_energy_j':np.nan,'cpu_power_w':np.nan,'node_power_w':np.nan,
        'gpu_channel_count':4,'valid_gpu_count':power.notna().sum(axis=1),'quality_flags':flags,
        'nvml_reading_time_ns':pd.to_numeric(data.reading_time_ns,errors='coerce')})
    b=completed_bins(c,median,cfg)
    if len(b):b['dataset_id']='nlr_raw_nvml';b['measurement_boundary']='sum_of_4_gpu_channels'
    quality={'run_id':run,'source_file':str(p),'source_hash':h,'source_rows':len(c),'median_dt_s':median,
        'missing_gpu_rows':int(c.gpu_power_w.isna().sum()),'gap_starts':int(gap.sum()),'nonmonotonic':int(delta.le(0).sum()),
        'segments':int(seg.nunique()),'valid_200ms_bins':int(b.valid.sum()) if len(b) else 0,'parent_group':group,
        'measurement_boundary':'sum_of_4_gpu_channels','timestamp_timezone_verified':False,
        'sensor_bandwidth_verified':False,'input_interpolated':False}
    return c,b,quality

def main():
    a=argparse.ArgumentParser();a.add_argument('--build-id',default='main_20260929');a.add_argument('--limit',type=int,default=0);args=a.parse_args()
    cfg=yaml.safe_load((PROJECT/'configs/existing.yaml').read_text());out=PROJECT/'outputs'/args.build_id/'nlr_raw';out.mkdir(parents=True,exist_ok=True)
    (out/'canonical_runs').mkdir(exist_ok=True);(out/'bins').mkdir(exist_ok=True)
    paths=sorted(RAW.rglob('nvml_wattameter*.log'));paths=paths[:args.limit] if args.limit else paths
    script_hash=sha256(Path(__file__).resolve());config_hash=sha256(PROJECT/'configs/existing.yaml')
    statefile=out/'build_status.json'
    old=json.loads(statefile.read_text()) if statefile.exists() else {}
    if old and (old.get('script_sha256')!=script_hash or old.get('config_sha256')!=config_hash):
        raise RuntimeError('script/config changed; use a new build ID')
    status={'status':'running','script_sha256':script_hash,'config_sha256':config_hash,'requested_files':len(paths),
        'started_utc':old.get('started_utc',datetime.now(timezone.utc).isoformat())}
    quality=[];view_counts=[]
    for j,p in enumerate(paths,1):
        run=f'nlr_raw-{file_id(p)}';cp=out/'canonical_runs'/f'{run}.parquet';bp=out/'bins'/f'{run}.parquet'
        if cp.exists() and bp.exists():
            c=pd.read_parquet(cp);b=pd.read_parquet(bp)
            q={'run_id':run,'source_file':str(p),'source_hash':c.source_hash.iloc[0],'source_rows':len(c),
               'median_dt_s':float(c.elapsed_s.diff().loc[lambda x:x.gt(0)].median()),'missing_gpu_rows':int(c.gpu_power_w.isna().sum()),
               'gap_starts':int(c.quality_flags.eq('gap_start').sum()),'nonmonotonic':int(c.quality_flags.eq('nonmonotonic').sum()),
               'segments':int(c.segment_id.nunique()),'valid_200ms_bins':int(b.valid.sum()),'parent_group':c.split_group_id.iloc[0],
               'measurement_boundary':'sum_of_4_gpu_channels','timestamp_timezone_verified':False,'sensor_bandwidth_verified':False,'input_interpolated':False}
        else:
            c,b,q=convert(p,cfg);c.to_parquet(cp,index=False);b.to_parquet(bp,index=False)
        quality.append(q)
        if len(b):
            views=build_views(b,cfg)
            for name,frame in views.items():
                folder=out/name;folder.mkdir(exist_ok=True)
                frame.to_parquet(folder/f'{run}.parquet',index=False)
            view_counts.append(len(views['forecast_index']))
        status.update(completed_files=j,forecast_rows=int(sum(view_counts)),heartbeat_utc=datetime.now(timezone.utc).isoformat(),last_run_id=run)
        statefile.write_text(json.dumps(status,indent=2))
    q=pd.DataFrame(quality);q.to_csv(out/'quality_audit.csv',index=False)
    groups=q.parent_group.nunique()
    status.update(status='completed',source_rows=int(q.source_rows.sum()),valid_200ms_bins=int(q.valid_200ms_bins.sum()),
        conservative_group_count=int(groups),finished_utc=datetime.now(timezone.utc).isoformat())
    statefile.write_text(json.dumps(status,indent=2))
    (out/'README.md').write_text(f'# NLR raw NVML track\n\n{len(q)} source node log files, {groups} conservative parent job groups. Direct 4-GPU NVML mW converted to W; no published 1 ms interpolation used. Native timestamp intervals are in quality_audit.csv. 200 ms bins use forward hold, >=95% coverage, and assumed end-of-bin availability. Node logs are not summed into job/PCC power; clock/timezone and sensor bandwidth are unverified. D2 per-run forecast views are in directories; metadata is not joined. This is a separate track from highres and must be evaluated by job group, not pooled blindly.\n')
    print(json.dumps(status,indent=2))
if __name__=='__main__':main()
