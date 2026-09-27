from __future__ import annotations
import numpy as np, pandas as pd

def assess(d,row,cfg):
    t=pd.to_numeric(d.time_s,errors="coerce").to_numpy(float); p=pd.to_numeric(d.power_w,errors="coerce").to_numpy(float)
    dt=np.diff(t); pos=dt[dt>0]; med=float(np.nanmedian(pos)) if len(pos) else np.nan
    gap_thr=med*float(cfg["analysis"]["gap_multiplier"]) if np.isfinite(med) else np.nan
    repeated=float(np.mean(np.diff(p[np.isfinite(p)])==0)) if np.isfinite(p).sum()>1 else np.nan
    maxconst=0.; start=0
    for i in range(1,len(p)):
        if not(np.isfinite(p[i]) and np.isfinite(p[i-1]) and p[i]==p[i-1]): start=i
        elif np.isfinite(t[i]) and np.isfinite(t[start]): maxconst=max(maxconst,t[i]-t[start])
    q=dict(run_id=row.run_id,dataset=row.dataset,row_count=len(d),duration_s=float(np.nanmax(t)-np.nanmin(t)) if len(t) else np.nan,
      invalid_timestamp_count=int((~np.isfinite(t)).sum()),missing_power_count=int((~np.isfinite(p)).sum()),negative_power_count=int(np.sum(p<0)),
      duplicate_timestamp_count=int(np.sum(dt==0)),nonmonotonic_count=int(np.sum(dt<0)),reset_count=int(np.sum(dt<0)),
      dt_median_s=med,dt_p05_s=float(np.nanpercentile(pos,5)) if len(pos) else np.nan,dt_p95_s=float(np.nanpercentile(pos,95)) if len(pos) else np.nan,
      dt_max_s=float(np.nanmax(pos)) if len(pos) else np.nan,gap_threshold_s=gap_thr,large_gap_count=int(np.sum(dt>gap_thr)) if np.isfinite(gap_thr) else 0,
      repeated_power_fraction=repeated,max_constant_stretch_s=maxconst,effective_resolution="unverified" if row.dataset=="highres_node" else ("~0.2s source interpolated to 1ms" if row.dataset=="nlr_online_rate" else "0.2s aggregation grid" if row.dataset=="nlr_training" else "0.1s aggregation grid"))
    return q

def segments(d,gap_threshold):
    t=d.time_s.to_numpy(float); p=d.power_w.to_numpy(float); starts=[0]
    bad=(~np.isfinite(t[1:]))|(~np.isfinite(t[:-1]))|(~np.isfinite(p[1:]))|(~np.isfinite(p[:-1]))|(np.diff(t)<=0)|(np.diff(t)>gap_threshold)
    starts.extend((np.flatnonzero(bad)+1).tolist()); starts=sorted(set(starts)); ends=starts[1:]+[len(d)]
    return [(i,d.iloc[a:b].reset_index(drop=True)) for i,(a,b) in enumerate(zip(starts,ends)) if b-a>=3 and d.iloc[a:b].power_w.notna().all()]
