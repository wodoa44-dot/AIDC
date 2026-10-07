from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from ..inventory import PROJECT

SOURCE='https://www.osti.gov/servlets/purl/2404598'
def params():return yaml.safe_load((PROJECT/'configs/grid.yaml').read_text())
def derivative(x,load_pu,cfg):
    omega,pm=x
    return np.array([(pm-load_pu-cfg['damping_pu']*omega)/cfg['inertia_m_s'],(-pm-omega/cfg['droop_r_pu'])/cfg['governor_tg_s']])
def rk4(x,u,h,c):
    a=derivative(x,u,c);b=derivative(x+h*a/2,u,c);d=derivative(x+h*b/2,u,c);e=derivative(x+h*d,u,c)
    return x+h*(a+2*b+2*d+e)/6
def solve(power_w,cfg,dt=None):
    p=np.asarray(power_w,float);h=dt or cfg['simulation_dt_s'];control=.2
    if not np.isclose(control/h,round(control/h)):raise ValueError('solver step must divide 200ms control step')
    state=np.zeros(2);frequency=[];mechanical=[];ref=p[0];copies=cfg['synchronized_copies_scenario'];base=cfg['base_mva_scenario']*1e6
    for w in p:
        u=(w-ref)*copies/base
        for _ in range(int(round(control/h))):state=rk4(state,u,h,cfg)
        frequency.append(cfg['nominal_frequency_hz']*(1+state[0]));mechanical.append(state[1])
    return np.asarray(frequency),np.asarray(mechanical)
def verify(cfg):
    eq=np.full(100,500.);f,_=solve(eq,cfg)
    step=np.r_[np.zeros(20),np.full(300,1000.)];a,_=solve(step,cfg);b,_=solve(step,cfg,cfg['simulation_dt_s']/2)
    loadpu=1000*cfg['synchronized_copies_scenario']/(cfg['base_mva_scenario']*1e6)
    expected=cfg['nominal_frequency_hz']*(1-loadpu/(cfg['damping_pu']+1/cfg['droop_r_pu']))
    checks={'equilibrium':bool(np.max(abs(f-cfg['nominal_frequency_hz']))<1e-10),
        'positive_step_reduces_frequency':bool(a[-1]<cfg['nominal_frequency_hz']),
        'steady_state':bool(abs(a[-1]-expected)<1e-5),
        'half_step_convergence':bool(np.max(abs(a-b))<1e-5)}
    return {'checks':checks,'passed':all(checks.values()),'steady_expected_hz':expected,'half_step_max_difference_hz':float(np.max(abs(a-b)))}
def prepare(out:Path):
    out.mkdir(exist_ok=True);c=params();v=verify(c)
    provenance={'model_version':'single_area_frequency_v1','source':'Aslami et al., Power System Frequency Dynamics Modeling, State Estimation, and Control using NODEs and SAC, ACM SIGAPP Applied Computing Review 24(1), 2024, DOI 10.1145/3663652.3663655',
        'public_parameter_source':SOURCE,'equations':'M dω/dt=Pm-ΔPL-Dω; Tg dPm/dt=-Pm-ω/R',
        'parameters':c,'parameter_type':'published example plus scenario assumptions',
        'scenario_assumptions':['100 MW base','1000 synchronized copies of a node GPU power deviation','each segment begins in equilibrium'],
        'output_scope':'aggregate frequency only; no PCC meter, voltage, Q, ESS, UPS, weak-grid mode, or facility validation',
        'reused_from':'experiments/aidc_grid_metadata_20260929/scripts/run_frequency_exploratory.py; refactored into standalone adapter',
        'validation':v}
    (out/'grid_model_provenance.json').write_text(json.dumps(provenance,indent=2))
    if not v['passed']:raise RuntimeError('frequency model verification failed')
def simulate(out:Path):
    c=params();b=pd.read_parquet(out/'completed_bins_200ms.parquet')
    groups=b[b.valid].groupby(['run_id','segment_id'],sort=False)
    manifest=[]; traces=[];metrics=[]
    for (run,seg),g in groups:
        g=g.sort_values('bin_index')
        if len(g)<100:continue
        # Use one contiguous block per segment. Gaps are already segmented.
        g=g.iloc[:min(len(g),1500)]
        if (g.bin_index.diff().dropna()!=1).any():continue
        p=g.gpu_power_w.to_numpy(float);f,pm=solve(p,c)
        scenario=f'{run}:{seg}:uncontrolled'
        manifest.append({'scenario_id':scenario,'parent_trace_id':run,'split_group_id':g.split_group_id.iloc[0],
            'model_version':'single_area_frequency_v1','solver':'RK4','simulation_dt_s':c['simulation_dt_s'],
            'control_dt_s':c['control_dt_s'],'forecast_dt_s':None,'scaling_factor':c['synchronized_copies_scenario'],
            'parameter_source':SOURCE,'initial_state':'equilibrium','disturbance_id':'held_trace_gpu_deviation',
            'input_evidence_type':'derived_from_measured_existing','output_evidence_type':'simulated'})
        traces.append(pd.DataFrame({'scenario_id':scenario,'timestamp':g.timestamp,'parent_run_id':run,
            'p_it_node_gpu_w':p,'p_load_scenario_w':(p-p[0])*c['synchronized_copies_scenario'],
            'frequency_hz':f,'evidence_type':'simulated','parent_evidence_type':'measured_existing'}))
        rocof=np.diff(f)/c['control_dt_s']
        metrics.append({'scenario_id':scenario,'max_abs_delta_frequency_hz':float(np.max(abs(f-c['nominal_frequency_hz']))),
            'max_abs_rocof_200ms_hz_per_s':float(np.max(abs(rocof))),
            'frequency_nadir_hz':float(f.min()),'frequency_zenith_hz':float(f.max()),'evidence_type':'simulated'})
    pd.DataFrame(manifest).to_csv(out/'grid_scenario_manifest.csv',index=False)
    pd.concat(traces,ignore_index=True).to_parquet(out/'grid_frequency_traces.parquet',index=False) if traces else None
    pd.DataFrame(metrics).to_csv(out/'grid_frequency_metrics.csv',index=False)
    (out/'grid_status.json').write_text(json.dumps({'status':'completed' if traces else 'blocked_source','scenarios':len(traces),'frequency_only':True},indent=2))
