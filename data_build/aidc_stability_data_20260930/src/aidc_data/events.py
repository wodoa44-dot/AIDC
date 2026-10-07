from __future__ import annotations
import numpy as np
import pandas as pd

def detect(bins,cfg):
    width=float(cfg['bin_seconds']); records=[]
    for (run,seg),g in bins.groupby(['run_id','segment_id'],sort=False):
        g=g.sort_values('bin_index').reset_index(drop=True); p=g.gpu_power_w.to_numpy(float)
        active=None; baseline=None; confirm=int(cfg['event_confirm_bins']); end_bins=int(cfg['event_end_bins'])
        for i in range(5,len(g)):
            if not np.isfinite(p[i]) or g.bin_index.iloc[i]-g.bin_index.iloc[i-1]!=1:
                if active is not None:active['right_censored']=True;records.append(active);active=None
                continue
            if active is None:
                hist=p[i-5:i]
                if not np.isfinite(hist).all():continue
                b=float(np.median(hist)); th=max(float(cfg['event_min_absolute_w']),abs(b)*float(cfg['event_relative_threshold']))
                direction=int(np.sign(p[i]-b))
                if direction==0 or abs(p[i]-b)<th:continue
                if i+confirm>len(g) or not np.isfinite(p[i:i+confirm]).all() or not np.all(direction*(p[i:i+confirm]-b)>=th):continue
                active={'event_id':f'{run}:{seg}:{int(g.bin_index.iloc[i])}','parent_session':g.session_id.iloc[0],
                    'run_id':run,'segment_id':int(seg),'direction':'rise' if direction>0 else 'fall',
                    'onset_time':g.timestamp.iloc[i],'detected_at':g.available_at.iloc[i+confirm-1],
                    'end_time':pd.NaT,'baseline_w':b,'extreme_w':p[i],'delta_w':p[i]-b,
                    'duration_s':np.nan,'max_ramp_w_per_s':0.,'energy_deviation_j':0.,
                    'label_source':'power_shape_only','confidence':'exploratory','right_censored':False,
                    'threshold_version':'relative_0.10_abs_100w_v1','evidence_type':'derived',
                    'parent_evidence_type':'measured_existing','start_bin':i,'end_count':0,'threshold_w':th}
                baseline=b
            if active is not None:
                d=int(1 if active['direction']=='rise' else -1)
                if d*(p[i]-baseline)>d*(active['extreme_w']-baseline):active['extreme_w']=p[i]
                active['delta_w']=active['extreme_w']-baseline
                active['energy_deviation_j']+=abs(p[i]-baseline)*width
                active['max_ramp_w_per_s']=max(active['max_ramp_w_per_s'],abs(p[i]-p[i-1])/width)
                active['end_count']=active['end_count']+1 if d*(p[i]-baseline)<active['threshold_w']*float(cfg['event_hysteresis_fraction']) else 0
                if active['end_count']>=end_bins:
                    active['end_time']=g.timestamp.iloc[i];active['duration_s']=(active['end_time']-active['onset_time']).total_seconds()
                    records.append(active);active=None
        if active is not None:active['right_censored']=True;records.append(active)
    result=pd.DataFrame(records)
    return result.drop(columns=['start_bin','end_count','threshold_w'],errors='ignore')
