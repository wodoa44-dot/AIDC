from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

GPU_COLS=[f"gpu{i}_power_W" for i in range(8)]

def load_native(row):
    p=Path(row.source_path)
    if row.dataset=="highres_node":
        use=["timestamp"]+GPU_COLS
        d=pd.read_csv(p,usecols=use)
        d["timestamp_original"]=d["timestamp"]
        d["time_s"]=(pd.to_datetime(d.timestamp,errors="coerce")-pd.to_datetime(d.timestamp,errors="coerce").iloc[0]).dt.total_seconds()
        valid=d[GPU_COLS].notna().all(axis=1)
        d["power_w"]=d[GPU_COLS].sum(axis=1,min_count=8).where(valid)
        d["gpu_power_std_w"]=d[GPU_COLS].std(axis=1).where(valid)
        return d[["time_s","timestamp_original","power_w","gpu_power_std_w"]+GPU_COLS]
    d=pd.read_parquet(p)
    if "timestep[s]" in d.columns: t=d["timestep[s]"]
    else: t=pd.Series(d.index,index=d.index)
    return pd.DataFrame({"time_s":np.asarray(pd.to_numeric(t,errors="coerce"),dtype=float),"timestamp_original":np.asarray(pd.to_numeric(t,errors="coerce"),dtype=float),"power_w":pd.to_numeric(d["power[W]"],errors="coerce").to_numpy()})

def analysis_series(d,row,cfg):
    if row.dataset!="nlr_online_rate": return d.copy(),"native"
    bin_s=float(cfg["analysis"]["online_rate_bin_s"]); x=d.copy()
    x["bin"]=np.floor(x.time_s.astype(float)/bin_s + 1e-9).astype("int64")
    g=x.groupby("bin",sort=False).agg(time_s=("time_s","mean"),power_w=("power_w","mean"),n=("power_w","count"))
    native_dt=np.nanmedian(np.diff(x.time_s)); expected=max(1,round(bin_s/native_dt))
    g["coverage"]=g.n/expected
    g.loc[g.coverage<float(cfg["analysis"]["min_bin_coverage"]),"power_w"]=np.nan
    g["timestamp_original"]=g.time_s
    return g.reset_index(drop=True)[["time_s","timestamp_original","power_w","coverage"]],f"time-bin mean {bin_s}s; min coverage {cfg['analysis']['min_bin_coverage']}"
