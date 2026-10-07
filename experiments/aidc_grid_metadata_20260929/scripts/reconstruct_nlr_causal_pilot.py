#!/usr/bin/env python3
"""Raw NLR one-node training pilot: last arrived sample only, no interpolation."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'trace/dataset/nlr_312/extracted/00_raw_datasets/training_stable_diffusion/1node'
OUT=Path(__file__).resolve().parents[1]/'outputs/nlr_raw_causal_pilot_v1'
DT=pd.Timedelta(milliseconds=200)
MAX_AGE=pd.Timedelta(milliseconds=400)


def read_device(path,kind):
    with path.open() as fh:header=fh.readline().lstrip('#').split()
    d=pd.read_csv(path,sep=r'\s+',comment='#',header=None,names=header)
    d['sample_time']=pd.to_datetime(d['timestamp'],format='%Y-%m-%d_%H:%M:%S.%f',errors='coerce')
    if kind=='nvml':
        cols=[c for c in d if re.fullmatch(r'gpu-\d+\[mW\]',c)]
        value=d[cols].apply(pd.to_numeric,errors='coerce').sum(axis=1,min_count=len(cols))*1e-3
    else:
        cols=[c for c in d if re.fullmatch(r'cpu-\d+\[W\]',c)]
        value=d[cols].apply(pd.to_numeric,errors='coerce').sum(axis=1,min_count=len(cols))
    valid=d.sample_time.notna() & value.notna()
    z=pd.DataFrame({'sample_time':d.loc[valid,'sample_time'],'power_w':value.loc[valid]})
    z=z.sort_values('sample_time').drop_duplicates('sample_time',keep='last').reset_index(drop=True)
    return z,{'samples':len(z),'power_columns':cols,'first':str(z.sample_time.iloc[0]),'last':str(z.sample_time.iloc[-1]),
              'source':str(path),'nonmonotonic_original':int(np.sum(np.diff(d.sample_time.dropna().to_numpy())<np.timedelta64(0,'ns')))}


def reconstruct(pair,job):
    gpu,gpu_info=read_device(pair['nvml'],'nvml')
    cpu,cpu_info=read_device(pair['rapl'],'rapl')
    start=max(gpu.sample_time.iloc[0],cpu.sample_time.iloc[0]).ceil('200ms')
    end=min(gpu.sample_time.iloc[-1],cpu.sample_time.iloc[-1]).floor('200ms')
    if end<=start:return None,None
    grid=pd.DataFrame({'issuance_time':pd.date_range(start,end,freq=DT)})
    for kind,device in [('gpu',gpu),('cpu',cpu)]:
        local=device.rename(columns={'sample_time':f'{kind}_sample_time','power_w':f'{kind}_w'})
        grid=pd.merge_asof(grid,local,left_on='issuance_time',right_on=f'{kind}_sample_time',direction='backward',tolerance=MAX_AGE)
        grid[f'{kind}_age_s']=(grid.issuance_time-grid[f'{kind}_sample_time']).dt.total_seconds()
    grid['device_coverage']=grid[['gpu_w','cpu_w']].notna().sum(axis=1)
    grid['total_w']=(grid.gpu_w+grid.cpu_w).where(grid.device_coverage.eq(2))
    grid['run_id']=f'nlr_training_stable_diffusion_1node_{job}'
    grid['segment']=(grid.total_w.isna() | grid.total_w.shift().isna()).cumsum()
    grid['time_s']=(grid.issuance_time-grid.issuance_time.iloc[0]).dt.total_seconds()
    metadata={'run_id':grid.run_id.iloc[0],'rows':len(grid),'valid_rows':int(grid.total_w.notna().sum()),
              'missing_fraction':float(grid.total_w.isna().mean()),
              'gpu_age_p95_s':float(grid.gpu_age_s.quantile(.95)),'cpu_age_p95_s':float(grid.cpu_age_s.quantile(.95)),
              'gpu':gpu_info,'cpu':cpu_info}
    return grid,metadata


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pairs={}
    for path in SOURCE.glob('*.log'):
        name=path.name
        kind='nvml' if name.startswith('nvml_') else 'rapl' if name.startswith('rapl_') else None
        match=re.search(r'slurmid_(\d+)_node_',name)
        if kind and match:pairs.setdefault(match.group(1),{})[kind]=path
    rows=[]
    for job,pair in sorted(pairs.items()):
        if set(pair)!={'nvml','rapl'}:continue
        grid,metadata=reconstruct(pair,job)
        if grid is None:continue
        grid.to_parquet(OUT/f'{job}.parquet',index=False)
        rows.append(metadata)
        print(job,'valid',metadata['valid_rows'],'/',metadata['rows'])
    summary=pd.DataFrame([{k:v for k,v in row.items() if k not in ('gpu','cpu')} for row in rows])
    summary.to_csv(OUT/'inventory.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps({'rows':rows,'reconstruction':'last sample timestamp <= 200ms right-edge, at most 400ms old, no backward fill or interpolation',
        'power':'GPU sum of gpu-N[mW] / 1000 plus CPU package cpu-N[W]; core terms excluded to avoid nested-package double count',
        'timestamp_limit':'Source timestamp is treated as availability only for this pilot; measurement window and collection latency are unverified. Therefore online deployability and causal predictor performance are not established.',
        'scope':'NLR training stable diffusion 1-node raw job, separate from highres GPU-only target; not pooled'},indent=2))

if __name__=='__main__':main()
