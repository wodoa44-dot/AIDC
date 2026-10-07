from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

GPU_COLS=[f'gpu{i}_power_W' for i in range(8)]

def canonicalize(row, cfg):
    raw=pd.read_csv(row.source_path)
    ts=pd.to_datetime(raw['timestamp'], errors='coerce')
    if not all(c in raw for c in GPU_COLS):
        raise ValueError(f'missing GPU channels: {row.source_path}')
    gpu=raw[GPU_COLS].apply(pd.to_numeric,errors='coerce')
    complete=gpu.notna().all(axis=1) & np.isfinite(gpu).all(axis=1)
    watts=gpu.sum(axis=1).where(complete)
    cpu=pd.to_numeric(raw['cpu_power_W'],errors='coerce') if 'cpu_power_W' in raw else pd.Series(np.nan,index=raw.index)
    elapsed=(ts-ts.iloc[0]).dt.total_seconds()
    dt=elapsed.diff()
    median=float(dt[dt.gt(0)].median())
    gap=dt.gt(cfg['gap_dt_factor']*median)
    reset=dt.le(0)|gap|ts.isna()
    segment=reset.cumsum().astype(int)
    flags=np.where(ts.isna(),'bad_timestamp',np.where(dt.le(0),'nonmonotonic',np.where(gap,'gap_start',np.where(~complete,'missing_gpu',''))))
    c=pd.DataFrame({'dataset_id':'highres_node','source_file':str(row.source_path),'source_hash':row.source_hash,
        'session_id':row.session_id,'split_group_id':row.split_group_id,'run_id':row.run_id,
        'segment_id':segment,'timestamp':ts,'elapsed_s':elapsed,'sample_index':np.arange(len(raw)),
        'node_id':pd.NA,'gpu_uuid':pd.NA,'measurement_boundary':'sum_of_8_gpu_channels',
        'evidence_type':'measured_existing','gpu_power_w':watts,'cpu_power_w':cpu,
        'node_power_w':np.nan,'gpu_channel_count':8,'valid_gpu_count':gpu.notna().sum(axis=1),
        'quality_flags':flags})
    c['gpu_energy_j']=np.nan # no cumulative GPU energy measurement
    return c,median

def completed_bins(c, median, cfg):
    width=float(cfg['bin_seconds']); hold=float(cfg['max_hold_dt_factor'])*median
    records=[]
    for seg,g in c.groupby('segment_id',sort=False):
        g=g[g.timestamp.notna()].sort_values('elapsed_s',kind='stable')
        if len(g)<2:continue
        t=g.elapsed_s.to_numpy(float); p=g.gpu_power_w.to_numpy(float)
        cpu=g.cpu_power_w.to_numpy(float)
        first=int(np.ceil(t[0]/width)); last=int(np.floor(t[-1]/width))
        # Values are held forward for at most hold seconds. This is an explicit
        # observation-time assumption; actual NVML sensor window is unknown.
        for k in range(first,last+1):
            left=(k-1)*width; right=k*width
            i=max(0,int(np.searchsorted(t,left,side='right')-1))
            e=0.; ec=0.; covered=0.; source_max=-1
            while i<len(t) and t[i]<right:
                a=max(left,t[i]); b=min(right,t[i]+hold,t[i+1] if i+1<len(t) else right)
                if b>a and np.isfinite(p[i]):
                    e+=p[i]*(b-a); covered+=b-a;source_max=max(source_max,int(g.sample_index.iloc[i]))
                    if np.isfinite(cpu[i]):ec+=cpu[i]*(b-a)
                i+=1
            coverage=covered/width
            valid=coverage+1e-9>=float(cfg['min_time_coverage'])
            records.append({'dataset_id':'highres_node','run_id':g.run_id.iloc[0], 'session_id':g.session_id.iloc[0],
                'split_group_id':g.split_group_id.iloc[0], 'segment_id':int(seg),'bin_index':k,
                'timestamp':g.timestamp.iloc[0]+pd.to_timedelta(right-t[0],unit='s'),
                'elapsed_s':right,'available_at':g.timestamp.iloc[0]+pd.to_timedelta(right-t[0],unit='s'),
                'gpu_power_w':e/covered if valid else np.nan,'cpu_power_w':ec/covered if valid else np.nan,
                'coverage':coverage,'valid':valid,'max_source_row':source_max,
                'measurement_boundary':'sum_of_8_gpu_channels','evidence_type':'derived',
                'parent_evidence_type':'measured_existing','lineage_source_hash':g.source_hash.iloc[0]})
    return pd.DataFrame.from_records(records)
