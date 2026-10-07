#!/usr/bin/env python3
"""Published-parameter single-area frequency-only exploratory response."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
MAIN=PROJECT/'outputs/full_20260928T180455Z'
OUT=MAIN/'frequency_only_exploratory_v1'
M=4.0;D=.015;R=.05;TG=.2;BASE_W=100e6;F0=60.;SCALE=1000.


def deriv(state,load_pu):
    omega,pm=state
    return np.array([(pm-load_pu-D*omega)/M,(-pm-omega/R)/TG])


def rk4_step(state,load_pu,step):
    k1=deriv(state,load_pu);k2=deriv(state+.5*step*k1,load_pu)
    k3=deriv(state+.5*step*k2,load_pu);k4=deriv(state+step*k3,load_pu)
    return state+step*(k1+2*k2+2*k3+k4)/6


def simulate(pcc_w,step=.02):
    values=np.asarray(pcc_w,float)
    assert .2/step==round(.2/step)
    nsub=int(round(.2/step));state=np.zeros(2);freq=np.empty(len(values));pm=np.empty(len(values))
    reference=values[0]
    for i,p in enumerate(values):
        load_pu=(p-reference)*SCALE/BASE_W
        for _ in range(nsub):state=rk4_step(state,load_pu,step)
        freq[i]=F0*(1+state[0]);pm[i]=state[1]
    return freq,pm


def verify():
    assert abs(simulate(np.full(100,4000.))[0]-60).max()<1e-12
    signal=np.r_[np.zeros(20),np.full(300,1e6/SCALE)]
    f,_=simulate(signal)
    expected=F0*(1-(1e6/BASE_W)/(D+1/R))
    assert f[-1]<F0 and abs(f[-1]-expected)<1e-5
    a,_=simulate(signal,.02);b,_=simulate(signal,.01)
    assert np.max(abs(a-b))<1e-5
    return {'equilibrium_hz':60.,'step_expected_steady_hz':expected,'step_simulated_steady_hz':float(f[-1]),
            'step_size_max_abs_difference_hz':float(np.max(abs(a-b))),
            'step_test_load_MW':1.,'verification':'baseline equilibrium, positive load lowers frequency, steady droop and 20/10ms convergence passed'}


def main():
    OUT.mkdir(exist_ok=True)
    verification=verify();data=pd.read_parquet(MAIN/'feasible_control_sensitivity_v1/control_traces_base_case.parquet')
    metrics=[];traces=[]
    for (run,segment,mode),g in data.groupby(['run_id','segment','controller']):
        g=g.sort_values('issuance_s');pcc=g.pcc_w.to_numpy();t=g.issuance_s.to_numpy()
        if len(g)<3:continue
        f,pm=simulate(pcc)
        rocof=np.diff(f)/.2
        metrics.append({'run_id':run,'segment':segment,'controller':mode,'n':len(g),
            'f_nadir_hz':float(f.min()),'f_zenith_hz':float(f.max()),
            'max_abs_delta_f_hz':float(np.max(abs(f-F0))),
            'rocof_200ms_max_abs_hz_s':float(np.max(abs(rocof))),
            'bound_0p2hz_exceedance_s':float(np.sum(abs(f-F0)>.2)*.2),
            'frequency_only_evidence':'exploratory published-parameter low-order model; scale assumed'})
        traces.append(pd.DataFrame({'run_id':run,'segment':segment,'controller':mode,
            'issuance_s':t,'pcc_w':pcc,'frequency_hz':f,'governor_pm_pu':pm}))
    result=pd.DataFrame(metrics);result.to_csv(OUT/'frequency_metrics_per_segment.csv',index=False)
    run=result.groupby(['run_id','controller'],as_index=False)[['max_abs_delta_f_hz','rocof_200ms_max_abs_hz_s','bound_0p2hz_exceedance_s']].mean()
    run.to_csv(OUT/'frequency_metrics_per_run.csv',index=False)
    run.groupby('controller',as_index=False)[['max_abs_delta_f_hz','rocof_200ms_max_abs_hz_s','bound_0p2hz_exceedance_s']].mean().to_csv(OUT/'frequency_metrics_aggregate.csv',index=False)
    pd.concat(traces,ignore_index=True).to_parquet(OUT/'frequency_traces.parquet',index=False)
    provenance={'source':'Aslami et al., Power System Frequency Dynamics Modeling, State Estimation, and Control using NODEs and SAC, ACM SIGAPP Applied Computing Review 24(1), 2024, DOI 10.1145/3663652.3663655',
        'public_parameter_table':'https://www.osti.gov/servlets/purl/2404598',
        'parameters':{'M_s':M,'D_pu':D,'droop_R_pu':R,'governor_Tg_s':TG,'secondary_integral_gain':0},
        'equations':'M domega/dt=pm-load_pu-D*omega; Tg dpm/dt=-pm-omega/R',
        'units':'omega and pm per unit; frequency_hz=60*(1+omega)',
        'nominal_hz':F0,'base_MVA_scenario':BASE_W/1e6,'trace_scale_synchronized_copies':SCALE,
        'load_mapping':'(PCC(t)-PCC(first sample))*1000/100 MW. Each segment starts at its own equilibrium; zero-order-hold 200ms measured input at 20ms solver steps.',
        'case_type':'single-area low-order frequency response only, exploratory; not a network benchmark or facility validation',
        'unsupported':['voltage','spatial mode','GFM stability','real operating point','UPS disturbance','frequency protection and certified bounds'],
        'validation':verification}
    (OUT/'grid_model_provenance.json').write_text(json.dumps(provenance,indent=2))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    print('segments',len(result),'runs',run.run_id.nunique(),'trace_rows',sum(len(t) for t in traces))
    print(run.groupby('controller').max_abs_delta_f_hz.mean().to_string())

if __name__=='__main__':main()
