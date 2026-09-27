from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
from powerprep.adapters import load_native
from powerprep.discovery import stream_hash
from powerprep.events import detect_segment
from powerprep.quality import segments

def cfg(**kw):
    x=dict(delta_event_s=.5,relative_drop_threshold=.2,absolute_drop_w=50,
      baseline_window_s=5,baseline_stability_mad_fraction=.08,low_persistence_s=1,
      recovery_fraction=.9,recovery_persistence_s=.5,noise_median_window_s=.2,
      min_valid_coverage=.8,detector_config_id="test")
    x.update(kw); return x

def row(): return SimpleNamespace(dataset="synthetic",run_id="r",power_scope="test")
def series(low_end=30,recovery=400,noise=0):
    t=np.arange(0,50,.1); p=np.full(len(t),500.)
    p[(t>=10)&(t<low_end)]=250
    z=(t>=low_end)&(t<low_end+recovery)
    if recovery>0: p[z]=250+(t[z]-low_end)/recovery*250
    if noise: p+=np.random.default_rng(3).normal(0,noise,len(p))
    return pd.DataFrame({"time_s":t,"power_w":p})

def test_long_low_abrupt_recovery():
    e=detect_segment(series(recovery=.1),row(),0,cfg(),"ok")
    assert len(e)==1 and e[0]["completed"] and e[0]["low_duration_s"]>19

def test_gradual_recovery_has_positive_rise_time():
    e=detect_segment(series(recovery=5),row(),0,cfg(),"ok")
    assert len(e)==1 and e[0]["recovery_rise_time_10_90_s"]>3

def test_noise_is_not_sustained_event():
    d=series(low_end=10,recovery=.1,noise=8)
    assert detect_segment(d,row(),0,cfg(),"ok")==[]

def test_right_censored():
    d=series(low_end=80,recovery=.1)
    e=detect_segment(d,row(),0,cfg(),"ok")
    assert len(e)==1 and e[0]["right_censored"] and np.isnan(e[0]["low_duration_s"])

def test_gap_and_reset_split():
    d=pd.DataFrame({"time_s":[0,.1,.2,4,4.1,4.2,0,.1,.2],"power_w":[1]*9})
    assert len(segments(d,.5))==3

def test_missing_gpu_channel_invalidates_sum(tmp_path):
    p=tmp_path/"x.csv"; d=pd.DataFrame({"timestamp":pd.date_range("2020",periods=2,freq="20ms")})
    for i in range(8): d[f"gpu{i}_power_W"]=[1,1]
    d.loc[1,"gpu3_power_W"]=np.nan; d.to_csv(p,index=False)
    r=SimpleNamespace(source_path=str(p),dataset="highres_node")
    assert np.isnan(load_native(r).power_w.iloc[1])

def test_exact_duplicates_not_parameter_duplicates(tmp_path):
    a=tmp_path/"a"; b=tmp_path/"b"; c=tmp_path/"c"
    a.write_text("same"); b.write_text("same"); c.write_text("different")
    assert stream_hash(a)==stream_hash(b) and stream_hash(a)!=stream_hash(c)

