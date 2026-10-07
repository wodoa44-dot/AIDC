#!/usr/bin/env python3
"""One-to-one matching of E0 path-event crossing times and alert episodes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
MAIN=PROJECT/'outputs/full_20260928T180455Z'
EVENT=MAIN/'feasible_events_v2'
DT=.2


def onsets(times,mask):
    idx=np.flatnonzero(mask)
    if len(idx)==0:return np.array([],dtype=int)
    starts=np.r_[True,np.diff(times[idx])>DT+1e-6]
    return idx[starts]


def crossing_time(prepared,run,segment,issuance,h,threshold,direction):
    p=prepared[(prepared.run_id==run)&(prepared.segment==segment)].set_index('issuance_s').current_w
    base=float(p.loc[issuance]);sign=1 if direction=='up' else -1
    for j in range(1,h+1):
        future=round(issuance+j*DT,6)
        index=np.argmin(abs(p.index.to_numpy()-future))
        if abs(p.index[index]-future)>1e-5:continue
        if sign*(float(p.iloc[index])-base)>threshold:return float(p.index[index])
    raise ValueError('Positive path label lacks crossing')


def main():
    prepared=pd.read_parquet(MAIN/'prepared.parquet',columns=['run_id','segment','issuance_s','current_w'])
    # Shared actual crossing times are calculated once per fold/horizon/direction/run/segment.
    rows=[];cache={}
    for file in sorted((EVENT/'prediction_parts').glob('*.parquet')):
        d=pd.read_parquet(file)
        for (run,segment),g in d.groupby(['run_id','segment']):
            g=g.sort_values('issuance_s');times=g.issuance_s.to_numpy();h=int(round(g.horizon_s.iloc[0]/DT))
            threshold=float(g.event_threshold_w.iloc[0]);direction=g.direction.iloc[0]
            key=(int(g.fold.iloc[0]),h,direction,run,int(segment))
            if key not in cache:
                true_starts=onsets(times,g.event_actual.to_numpy().astype(bool))
                event_times=[]
                local=prepared[(prepared.run_id==run)&(prepared.segment==segment)].set_index('issuance_s').current_w
                local_times=local.index.to_numpy();local_power=local.to_numpy();sign=1 if direction=='up' else -1
                for idx in true_starts:
                    issue=times[idx]
                    pos=np.searchsorted(local_times,issue)
                    if pos>=len(local_times) or abs(local_times[pos]-issue)>1e-6:raise ValueError('issuance missing')
                    base=local_power[pos]
                    found=None
                    for j in range(1,h+1):
                        if pos+j<len(local_times) and abs(local_times[pos+j]-(issue+j*DT))<1e-5 and sign*(local_power[pos+j]-base)>threshold:
                            found=float(local_times[pos+j]);break
                    if found is None:raise ValueError('positive label lacks physical crossing')
                    event_times.append(found)
                # Repeated issuance windows can refer to the same transition; 1s dead time.
                event_times=sorted(event_times);merged=[]
                for t in event_times:
                    if not merged or t-merged[-1]>1.0:merged.append(t)
                cache[key]=merged
            event_times=cache[key]
            alert_times=times[onsets(times,g.alarm.to_numpy().astype(bool))]
            used=set();leads=[]
            for event_time in event_times:
                eligible=[i for i,t in enumerate(alert_times) if i not in used and event_time-h*DT-1e-6<=t<=event_time+DT+1e-6]
                if eligible:
                    idx=min(eligible,key=lambda i:abs(event_time-alert_times[i]))
                    used.add(idx);leads.append(float(event_time-alert_times[idx]))
            rows.append({'fold':key[0],'horizon_s':h*DT,'direction':direction,'model':g.model.iloc[0],
                'run_id':run,'segment':segment,'n_issuances':len(g),'actual_events':len(event_times),
                'alert_episodes':len(alert_times),'matched_events':len(leads),
                'misses':len(event_times)-len(leads),'unmatched_alert_episodes':len(alert_times)-len(used),
                'matched_lead_mean_s':float(np.mean(leads)) if leads else np.nan,
                'matched_lead_p05_s':float(np.percentile(leads,5)) if leads else np.nan})
    result=pd.DataFrame(rows);result.to_csv(EVENT/'event_matching_per_segment.csv',index=False)
    by_run=result.groupby(['fold','horizon_s','direction','model','run_id'],as_index=False).agg(
        actual_events=('actual_events','sum'),alert_episodes=('alert_episodes','sum'),
        matched_events=('matched_events','sum'),misses=('misses','sum'),
        unmatched_alert_episodes=('unmatched_alert_episodes','sum'),matched_lead_mean_s=('matched_lead_mean_s','mean'))
    by_run['episode_recall']=by_run.matched_events/by_run.actual_events.replace(0,np.nan)
    by_run['episode_precision']=by_run.matched_events/by_run.alert_episodes.replace(0,np.nan)
    by_run.to_csv(EVENT/'event_matching_per_run.csv',index=False)
    by_run.groupby(['horizon_s','direction','model'],as_index=False)[['actual_events','matched_events','misses','unmatched_alert_episodes','episode_recall','episode_precision','matched_lead_mean_s']].mean().to_csv(EVENT/'event_matching_aggregate.csv',index=False)
    (EVENT/'matching_protocol.json').write_text(json.dumps({'evidence':'E0 offline','actual_event':'first physical completed-bin crossing from onset of contiguous positive-issuance path labels',
        'event_dead_time_s':1.0,'alert_event':'onset of contiguous positive-alarm issuance labels',
        'matching_window':'alert onset from event_time minus forecast horizon through event_time plus 0.2s',
        'matching':'one-to-one, nearest alert onset in window; unmatched events are misses; unmatched alert episodes are false alarms',
        'censoring':'segments separated; incomplete future windows excluded upstream; no event inferred after segment end',
        'lead_sign':'positive means alert before crossing; matched events only, misses reported separately',
        'limitation':'a sustained early alarm whose onset precedes the matching window is unmatched; source session independence unverified'},indent=2))
    print('rows',len(result),'events',int(result.actual_events.sum()),'matched',int(result.matched_events.sum()))

if __name__=='__main__':main()
