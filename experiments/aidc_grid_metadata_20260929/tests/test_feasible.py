from pathlib import Path
import numpy as np
import pandas as pd
from reconstruct_nlr_causal_pilot import reconstruct
from run_feasible_ablation import shuffled_by_run,META
from run_frequency_exploratory import simulate,M,D,R,F0,BASE_W,SCALE
from match_feasible_events import onsets


def _write_log(path,kind,values):
    times=pd.date_range('2025-01-01',periods=len(values),freq='200ms')
    if kind=='nvml':
        lines=['# timestamp reading-time[ns] gpu-0[mW]']
        lines += [f'{t.strftime("%Y-%m-%d_%H:%M:%S.%f")} 1000 {v}' for t,v in zip(times,values)]
    else:
        lines=['# timestamp reading-time[ns] cpu-0[W] cpu-0-core[W]']
        lines += [f'{t.strftime("%Y-%m-%d_%H:%M:%S.%f")} 1000 {v} 9999' for t,v in zip(times,values)]
    path.write_text('\n'.join(lines)+'\n')


def test_nlr_future_perturbation_and_cpu_package_only(tmp_path):
    nv=tmp_path/'nvml.log';rapl=tmp_path/'rapl.log'
    _write_log(nv,'nvml',[100000,200000,300000,400000,500000])
    _write_log(rapl,'rapl',[10,20,30,40,50])
    first,_=reconstruct({'nvml':nv,'rapl':rapl},'x')
    _write_log(nv,'nvml',[100000,200000,300000,400000,999999])
    second,_=reconstruct({'nvml':nv,'rapl':rapl},'x')
    pd.testing.assert_series_equal(first.total_w.iloc[:4],second.total_w.iloc[:4])
    assert np.isclose(first.total_w.iloc[0],110)
    assert first.gpu_age_s.dropna().ge(0).all()


def test_static_shuffle_preserves_run_tuple_and_test_input():
    d=pd.DataFrame({'run_id':['a','a','b','b','c','c']})
    for i,col in enumerate(META):d[col]=[i,i,i+10,i+10,i+20,i+20]
    original=d.copy(deep=True)
    shuffled,mapping=shuffled_by_run(d,2026)
    pd.testing.assert_frame_equal(d,original)
    assert sorted(mapping)==['a','b','c']
    for _,g in shuffled.groupby('run_id'):
        assert all(g[col].nunique()==1 for col in META)
    assert sorted(shuffled[META[0]].unique())==sorted(original[META[0]].unique())


def test_frequency_step_equilibrium_sign_and_convergence():
    zero=np.full(40,4000.)
    assert np.max(abs(simulate(zero,.02)[0]-F0))<1e-10
    step=np.r_[np.zeros(20),np.full(300,1e6/SCALE)]
    coarse,_=simulate(step,.02);fine,_=simulate(step,.01)
    expected=F0*(1-(1e6/BASE_W)/(D+1/R))
    assert coarse[-1]<F0
    assert abs(coarse[-1]-expected)<1e-5
    assert max(abs(coarse-fine))<1e-5


def test_alarm_episode_onsets_respect_gaps():
    time=np.array([0.,.2,.4,.6,.8,1.4,1.6])
    alarm=np.array([False,True,True,False,True,True,True])
    assert onsets(time,alarm).tolist()==[1,4,5]
