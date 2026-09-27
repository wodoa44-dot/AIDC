from __future__ import annotations
import hashlib, json
import numpy as np, pandas as pd

def config_grid(cfg):
    base=cfg["detector"]
    for rel in base["relative_drop_thresholds"]:
      for delta in base["delta_event_s"]:
        c=dict(base)
        c.pop("relative_drop_thresholds",None); c.pop("delta_event_s",None)
        c["relative_drop_threshold"]=float(rel); c["delta_event_s"]=float(delta)
        c["detector_config_id"]=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12]
        yield c

def _runs(mask,n):
    a=np.r_[False,mask,False]; z=np.flatnonzero(a[1:]!=a[:-1]); return [(s,e) for s,e in zip(z[::2],z[1::2]) if e-s>=n]

def detect_segment(seg,row,segment_id,c,timing_flag):
    t=seg.time_s.to_numpy(float); raw=seg.power_w.to_numpy(float)
    if len(t)<5:return []
    dt=float(np.median(np.diff(t))); smooth_n=max(1,round(c["noise_median_window_s"]/dt))
    p=pd.Series(raw).rolling(smooth_n,center=True,min_periods=1).median().to_numpy()
    bw=max(3,round(c["baseline_window_s"]/dt)); low_n=max(1,round(c["low_persistence_s"]/dt)); rec_n=max(1,round(c["recovery_persistence_s"]/dt))
    ps=pd.Series(p)
    baseline=ps.rolling(bw,min_periods=bw).median().shift(1).to_numpy()
    # Robust trailing dispersion proxy, implemented with fast rolling medians.
    residual=np.abs(p-baseline)
    mad=pd.Series(residual).rolling(bw,min_periods=bw).median().shift(1).to_numpy()
    stable=mad<=baseline*c["baseline_stability_mad_fraction"]
    low=(baseline-p>=c["absolute_drop_w"]) & ((baseline-p)/baseline>=c["relative_drop_threshold"]) & stable
    out=[]; occupied_until=-1
    for s,e0 in _runs(low,low_n):
      if s<=occupied_until: continue
      b=float(baseline[s]); threshold=b*c["recovery_fraction"]
      rs=rc=None; consecutive=0
      for jj in range(e0,len(p)):
        if p[jj]>=threshold:
          consecutive+=1
          if consecutive>=rec_n:
            rs=jj-rec_n+1; rc=jj; break
        else: consecutive=0
      censored=rs is None
      if not censored: occupied_until=rc+bw
      else: occupied_until=len(p)
      drop_on=max(0,s-1); low_end=rs if rs is not None else len(p)
      low_power=float(np.median(p[s:low_end])) if low_end>s else float(p[s])
      amp=b-low_power; rise=np.nan; qflags=[]
      if rs is not None:
        p10=low_power+.1*amp; p90=low_power+.9*amp
        i10=i90=None
        for jj in range(s,min(len(p),rc+bw+1)):
          if i10 is None and p[jj]>=p10: i10=jj
          if i10 is not None and p[jj]>=p90: i90=jj; break
        if i10 is not None and i90 is not None: rise=float(t[i90]-t[i10])
        else:qflags.append("ambiguous_10_90")
      de=max(1,round(c["delta_event_s"]/dt)); j=min(len(p)-1,(rs if rs is not None else e0)+de)
      k=rs if rs is not None else e0; slope=float((p[j]-p[k])/(t[j]-t[k])) if j>k and t[j]>t[k] else np.nan
      original=seg.timestamp_original.astype(str).to_numpy() if "timestamp_original" in seg else t.astype(str)
      out.append(dict(dataset=row.dataset,run_id=row.run_id,segment_id=segment_id,event_id=f"{row.run_id}-s{segment_id}-e{len(out)}",detector_config_id=c["detector_config_id"],power_scope=row.power_scope,timing_quality_flag=timing_flag,
       drop_onset_s=float(t[drop_on]),low_entry_s=float(t[s]),recovery_onset_s=float(t[rs]) if rs is not None else np.nan,recovery_confirmation_s=float(t[rc]) if rc is not None else np.nan,
       drop_onset_timestamp=original[drop_on],low_entry_timestamp=original[s],recovery_onset_timestamp=original[rs] if rs is not None else None,recovery_confirmation_timestamp=original[rc] if rc is not None else None,
       baseline_power_w=b,low_power_w=low_power,drop_amplitude_w=amp,drop_fraction=amp/b,low_duration_s=float(t[rs]-t[s]) if rs is not None else np.nan,observed_low_duration_s=float(t[(rs if rs is not None else len(t)-1)]-t[s]),
       recovery_rise_time_10_90_s=rise,recovery_slope_w_per_s=slope,delta_event_s=c["delta_event_s"],relative_threshold=c["relative_drop_threshold"],right_censored=censored,completed=not censored,quality_flags=";".join(qflags)))
    return out

def duration_bin(x):
    if pd.isna(x):return "censored"
    if x<1:return "under 1 s"
    if x<5:return "1–5 s"
    if x<30:return "5–30 s"
    if x<120:return "30–120 s"
    return "at least 120 s"
