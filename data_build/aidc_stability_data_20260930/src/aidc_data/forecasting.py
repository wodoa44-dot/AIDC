from __future__ import annotations
import numpy as np
import pandas as pd

def build_views(bins, cfg):
    width=float(cfg['bin_seconds']); horizons=[int(round(h/width)) for h in cfg['horizons_seconds']]
    frames=[]
    for (_,seg),g in bins.groupby(['run_id','segment_id'],sort=False):
        g=g.sort_values('bin_index').reset_index(drop=True)
        p=g.gpu_power_w.to_numpy(float); idx=g.bin_index.to_numpy(int)
        n=len(g); out=g[['dataset_id','run_id','session_id','split_group_id','segment_id','bin_index','timestamp','available_at','gpu_power_w','coverage','lineage_source_hash']].copy()
        out=out.rename(columns={'timestamp':'issuance_time','gpu_power_w':'current_power_w','available_at':'max_feature_available_at'})
        out['row_id']=[f'{g.run_id.iloc[0]}:{int(seg)}:{int(x)}' for x in idx]
        for h,steps in zip(cfg['horizons_seconds'],horizons):
            future=np.full(n,np.nan); valid=(idx[steps:]-idx[:-steps]==steps)&np.isfinite(p[steps:])&np.isfinite(p[:-steps]);
            future[:-steps]=np.where(valid,p[steps:],np.nan)
            out[f'target_power_{h:g}s_w']=future
        for seconds in cfg['history_seconds']:
            k=int(round(seconds/width)); s=pd.Series(p)
            contiguous=np.r_[np.zeros(k-1,dtype=bool),idx[k-1:]-idx[:n-k+1]==k-1] if n>=k else np.zeros(n,dtype=bool)
            for field,series in [('mean',s.rolling(k,min_periods=k).mean()),('std',s.rolling(k,min_periods=k).std(ddof=0)),('min',s.rolling(k,min_periods=k).min()),('max',s.rolling(k,min_periods=k).max())]:
                out[f'hist_{seconds}s_{field}_w']=series.where(contiguous).to_numpy()
            out[f'hist_{seconds}s_slope_w_per_s']=(s-s.shift(k-1)).where(contiguous).to_numpy()/max((k-1)*width,width)
        max_h=max(horizons); finite=np.isfinite(p)
        for H in [5,10,30]:
            k=int(round(H/width)); plus=np.full(n,np.nan);minus=np.full(n,np.nan);up=np.full(n,np.nan);down=np.full(n,np.nan)
            for i in range(n-k):
                if finite[i:i+k+1].all() and idx[i+k]-idx[i]==k:
                    d=p[i+1:i+k+1]-p[i]
                    plus[i]=np.maximum(d,0).sum()*width;minus[i]=np.maximum(-d,0).sum()*width
                    up[i]=np.max(np.diff(p[i:i+k+1])/width);down[i]=np.min(np.diff(p[i:i+k+1])/width)
            out[f'energy_above_{H}s_j']=plus;out[f'energy_below_{H}s_j']=minus
            out[f'max_ramp_up_{H}s_w_per_s']=up;out[f'max_ramp_down_{H}s_w_per_s']=down
        frames.append(out)
    all_rows=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if all_rows.empty:return {}
    target_cols=['row_id']+[x for x in all_rows if x.startswith(('target_','energy_','max_ramp_'))]
    keys=['row_id','dataset_id','run_id','session_id','split_group_id','segment_id','bin_index','issuance_time','max_feature_available_at','lineage_source_hash']
    base=all_rows[keys+['current_power_w','coverage']]
    valid=base.current_power_w.notna() & all_rows['hist_30s_mean_w'].notna()
    for h in cfg['horizons_seconds']:
        valid &= all_rows[f'target_power_{h:g}s_w'].notna()
    index=base.loc[valid,keys+['coverage']].copy()
    targets=all_rows.loc[valid,target_cols].copy()
    current=base.loc[valid,keys+['current_power_w']].copy()
    hist_cols=[x for x in all_rows if x.startswith('hist_')]
    history=all_rows.loc[valid,keys+['current_power_w']+hist_cols].copy()
    history['history_30s_available']=history['hist_30s_mean_w'].notna()
    return {'forecast_index':index,'targets':targets,'features_current_power':current,'features_power_history':history}
