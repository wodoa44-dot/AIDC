"""Optional offline v2 events: final recovery episode, explicit unresolved values."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from .events import detect_segment,config_grid
from .contract import save_table


def recovery_measurement(t,p,start,threshold_index,confirmation,low,baseline):
    result={'recovery_rise_time_10_90_s':np.nan,'recovery_10_s':np.nan,'recovery_90_s':np.nan,'recovery_measurement_status':'unresolved'}
    if threshold_index is None:return result|{'recovery_measurement_status':'right_censored'}
    amp=baseline-low
    if not np.isfinite(amp) or amp<=0:return result|{'recovery_measurement_status':'nonpositive_excursion'}
    lo,hi=low+.1*amp,low+.9*amp
    up=np.flatnonzero((p[:-1]<lo)&(p[1:]>=lo))+1
    up=up[(up>start)&(up<=threshold_index)]
    if not len(up):return result|{'recovery_measurement_status':'no_directed_10_crossing'}
    i10=int(up[-1]);relapse=np.flatnonzero(p[i10+1:]<lo)
    stop=i10+1+int(relapse[0]) if len(relapse) else len(p)
    up90=np.flatnonzero((p[:-1]<hi)&(p[1:]>=hi))+1
    up90=up90[(up90>=i10)&(up90<stop)]
    if not len(up90):return result|{'recovery_measurement_status':'no_90_crossing_in_final_episode'}
    i90=int(up90[0])
    if i90<threshold_index:return result|{'recovery_measurement_status':'ambiguous_early_90_crossing'}
    if i10==i90:return result|{'recovery_measurement_status':'sensor_unresolved_same_sample','recovery_10_s':float(t[i10]),'recovery_90_s':float(t[i90])}
    return result|{'recovery_rise_time_10_90_s':float(t[i90]-t[i10]),'recovery_10_s':float(t[i10]),'recovery_90_s':float(t[i90]),'recovery_measurement_status':'resolved_sample_crossings'}


def detect_v2(segment,row,sid,cfg):
    segment=segment.reset_index(drop=True)
    if 'coverage' in segment and (segment.coverage<cfg['min_valid_coverage']).any():raise ValueError('Insufficient event coverage')
    events=detect_segment(segment,row,sid,cfg,'offline only; source timing limits apply')
    t=segment.time_s.to_numpy(float);dt=np.median(np.diff(t)) if len(t)>1 else np.nan
    n=max(1,round(cfg['noise_median_window_s']/dt)) if np.isfinite(dt) else 1
    p=segment.power_w.rolling(n,center=True,min_periods=1).median().to_numpy()
    for e in events:
        s=int(np.searchsorted(t,e['low_entry_s']));rs=None if e['right_censored'] else int(np.searchsorted(t,e['recovery_onset_s']))
        rc=None if rs is None else int(np.searchsorted(t,e['recovery_confirmation_s']))
        e.update(recovery_measurement(t,p,s,rs,rc,e['low_power_w'],e['baseline_power_w']))
        e['event_definition_version']='2.0.0';e['time_until_recovery_threshold_s']=e['low_duration_s']
        slope=e.pop('recovery_slope_w_per_s');e['post_threshold_slope_w_per_s']=slope if rs is not None else np.nan
        end=len(t)-1 if rs is None else rs
        e['low_state_persistence_s']=float(sum(np.diff(t[s:end+1])*(p[s:end]<e['baseline_power_w']*cfg['recovery_fraction'])))
        e['quality_flags']=e['recovery_measurement_status']
    return events


def export_events(out,events):
    save_table(Path(out)/'events.parquet',events);save_table(Path(out)/'events.csv',events)


def detect_all(out,cfg):
    out=Path(out)
    if not(out/'resolution_manifest.csv').exists():raise ValueError('Run resolutions before detect')
    m=pd.read_csv(out/'manifest.csv').query('not is_duplicate').set_index('run_id')
    v=pd.read_csv(out/'resolution_manifest.csv');rid=f"{cfg['events']['resolution_s']:g}s";records=[]
    for r in v[(v.resolution_id==rid)&(v.status=='supported')].itertuples():
        d=pd.read_parquet(r.path);row=SimpleNamespace(**m.loc[r.run_id].to_dict(),run_id=r.run_id)
        for sid,g in d.groupby('segment_id',sort=False):
            if not g.is_valid.all():continue
            for dc in config_grid(cfg):records.extend(detect_v2(g,row,sid,dc))
    events=pd.DataFrame(records)
    if events.empty:events=pd.DataFrame(columns=['dataset','run_id','segment_id','event_id','detector_config_id','completed','right_censored','recovery_rise_time_10_90_s'])
    export_events(out,events)
    (out/'event_dictionary.md').write_text('''# Offline event definitions 2.0.0

Low entry is the first sample of the persistent low state. Recovery threshold
entry starts the persistent recovered state; confirmation ends its confirming
samples. Time until recovery threshold includes the intervening episode. Low-state
persistence integrates observed time below that threshold. Censored events retain
observed duration but no fabricated recovery values.

Directed 10–90% crossings belong to the final recovery episode: use the last 10%
departure before threshold entry and the following 90% crossing before relapse.
Earlier 90% crossings are ambiguous. Same-sample crossings are sensor-unresolved,
not exact zero-time recovery. Unresolved rise times stay null. Post-threshold slope
is explicitly AFTER threshold entry, not the recovery-rise slope. Sample crossings
have finite temporal resolution. Centered smoothing and events are offline only;
these annotations must never be forecasting inputs.
''')
    if len(events):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for (dataset,config),g in events.groupby(['dataset','detector_config_id']):
            fig,ax=plt.subplots();ax.hist(g.low_duration_s.dropna(),bins=20)
            ax.set(title=f'{dataset} / {config}',xlabel='Time until recovery threshold (s)',ylabel='Events')
            fig.tight_layout();fig.savefig(out/'plots'/f'events_{dataset}_{config}.png');plt.close(fig)
