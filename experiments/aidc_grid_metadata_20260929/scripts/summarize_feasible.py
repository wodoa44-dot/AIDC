#!/usr/bin/env python3
"""Recalculate paired uncertainty and summarize completed feasible tracks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
MAIN=PROJECT/'outputs/full_20260928T180455Z'
NLR=PROJECT/'outputs/nlr_raw_causal_pilot_v1'
RNG_SEED=2026


def paired_ci(frame,group_cols,key_col,base,other,value,out):
    rng=np.random.default_rng(RNG_SEED);rows=[]
    for keys,g in frame.groupby(group_cols,dropna=False):
        z=g.pivot(index=key_col,columns='model',values=value)
        if not {base,other}.issubset(z.columns):continue
        a=z[[base,other]].dropna().to_numpy();n=len(a)
        if not n:continue
        ids=rng.integers(0,n,size=(2000,n));boot=a[ids].mean(axis=1)
        gain=np.where(boot[:,0]>0,100*(1-boot[:,1]/boot[:,0]),np.nan)
        actual=100*(1-a[:,1].mean()/a[:,0].mean()) if a[:,0].mean()>0 else np.nan
        if not isinstance(keys,tuple):keys=(keys,)
        rows.append({**dict(zip(group_cols,keys)),'groups':n,'gain_pct':actual,
            'ci_low':float(np.nanpercentile(gain,2.5)),'ci_high':float(np.nanpercentile(gain,97.5)),
            'ci_kind':'fixed-prediction paired run bootstrap, 2000 draws; session independence unverified'})
    pd.DataFrame(rows).to_csv(out,index=False)


def main():
    event=MAIN/'feasible_events_v2';er=pd.read_csv(event/'event_per_run_metrics.csv')
    assert len(list((event/'prediction_parts').glob('*.parquet')))==40
    assert er.run_id.nunique()==32
    paired_ci(er,['horizon_s','direction'],'run_id','P2','P4','brier',event/'paired_brier_gain_ci.csv')
    for key,g in er.groupby(['fold','horizon_s','direction','run_id']):
        assert len(g)==2 and g.positives.nunique()==1
    nlr=NLR/'forecast_compact3_v1';nr=pd.read_csv(nlr/'per_run_metrics.csv')
    assert nr.run_id.nunique()==5 and len(nr)==20
    paired_ci(nr,['horizon_s'],'run_id','P0','P2','compact3_crps_w',nlr/'paired_gain_ci.csv')
    for p in NLR.glob('*.parquet'):
        d=pd.read_parquet(p)
        assert (d.loc[d.gpu_sample_time.notna(),'gpu_sample_time']<=d.loc[d.gpu_sample_time.notna(),'issuance_time']).all()
        assert (d.loc[d.cpu_sample_time.notna(),'cpu_sample_time']<=d.loc[d.cpu_sample_time.notna(),'issuance_time']).all()
        assert d.gpu_age_s.dropna().between(0,.4+1e-9).all()
        assert d.cpu_age_s.dropna().between(0,.4+1e-9).all()
    control=MAIN/'feasible_control_sensitivity_v1';c=pd.read_csv(control/'control_sensitivity_per_run.csv')
    trace=pd.read_parquet(control/'control_traces_base_case.parquet')
    assert abs(trace.power_balance_error_w).max()<1e-8
    assert trace.soc.between(.2-1e-9,1+1e-9).all()
    pivot=c[c.controller.isin(['C3','C4'])].pivot(index=['capacity_factor_training_p95','delay_s','run_id'],columns='controller',values='pcc_ramp_p95_w_per_s')
    assert len(pivot)==28
    cc=c[c.controller.isin(['C3','C4'])].rename(columns={'controller':'model'})
    paired_ci(cc,['capacity_factor_training_p95','delay_s'],'run_id','C3','C4','pcc_ramp_p95_w_per_s',control/'paired_c4_vs_c3_ramp_ci.csv')
    verification={'event_tasks':40,'event_run_groups':32,'event_paired_labels':'verified',
        'nlr_raw_runs':5,'nlr_no_future_sample_and_age':'verified',
        'nlr_forecast_tasks':10,'control_pairs':28,'control_power_balance_and_soc':'verified',
        'limitations':['source timestamps may precede acquisition completion','collection sessions not independently verified','E1 assumptions not measured']}
    (MAIN/'feasible_verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
