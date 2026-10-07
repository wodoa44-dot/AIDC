import numpy as np
import pandas as pd
from aidc_data.forecasting import build_views
from aidc_data.events import detect
from aidc_data.splits import make
from aidc_data.phase0.collect import sample_nvml
from aidc_data.phase0.controls import temporary_power_limit
from aidc_data.grid.adapter import verify,params

CFG={'bin_seconds':.2,'history_seconds':[1,2,5,10,30],'horizons_seconds':[.2,.4,.6,.8,1.,1.2,1.4,1.6,1.8,2.],
     'event_relative_threshold':.1,'event_min_absolute_w':100.,'event_confirm_bins':2,'event_end_bins':2,'event_hysteresis_fraction':.5}
def fixture(n=240):
    t=pd.date_range('2026-01-01',periods=n,freq='200ms')
    return pd.DataFrame({'dataset_id':'synthetic_test','run_id':'r','session_id':'s','split_group_id':'s','segment_id':0,'bin_index':np.arange(n),
        'timestamp':t,'available_at':t,'gpu_power_w':np.full(n,500.),'coverage':1.,'lineage_source_hash':'fixture','valid':True})
def test_future_change_does_not_change_past_features():
    a=fixture();v1=build_views(a,CFG)
    a.loc[220:,'gpu_power_w']=900.
    v2=build_views(a,CFG)
    h1=v1['features_power_history'];h2=v2['features_power_history']
    cols=[x for x in h1 if x.startswith('hist_') or x=='current_power_w']
    pd.testing.assert_frame_equal(h1.loc[h1.bin_index<200,cols].reset_index(drop=True),h2.loc[h2.bin_index<200,cols].reset_index(drop=True))
def test_gap_and_missing_power_excluded():
    a=fixture();a.loc[180,'gpu_power_w']=np.nan
    v=build_views(a,CFG)
    assert not v['forecast_index'].bin_index.isin(range(170,181)).any()
    a=fixture();a.loc[190:,'bin_index']+=2
    v=build_views(a,CFG)
    assert not v['forecast_index'].bin_index.isin(range(180,191)).any()
def test_event_onset_detection_and_censor():
    a=fixture();a.loc[190:,'gpu_power_w']=800.
    ev=detect(a,CFG)
    assert len(ev)>=1
    e=ev.iloc[0]
    assert e.detected_at>e.onset_time and bool(e.right_censored)
    assert e.direction=='rise'
def test_group_split_disjoint():
    idx=pd.DataFrame({'split_group_id':np.repeat(list('ABCDE'),10)})
    for fold in make(idx)['outer_folds']:
        assert not set(fold['train_groups'])&set(fold['test_groups'])
def test_frequency_model_checks():
    assert verify(params())['passed']
class FakeNvml:
    NVML_TEMPERATURE_GPU=0
    def __init__(self):self.cap=250000
    def nvmlDeviceGetPowerUsage(self,d):return 100000
    def nvmlDeviceGetTotalEnergyConsumption(self,d):raise RuntimeError('unsupported')
    def nvmlDeviceGetTemperature(self,d,t):return 40
    def nvmlDeviceGetUUID(self,d):return 'fake'
    def nvmlDeviceGetPowerManagementLimit(self,d):return self.cap
    def nvmlDeviceSetPowerManagementLimit(self,d,c):self.cap=c

def test_fake_nvml_unsupported_and_restore():
    f=FakeNvml();rows=sample_nvml(f,object(),'synthetic_test',.001,.002)
    assert len(rows)>0 and rows.energy_j.isna().all()
    assert rows.error.str.contains('energy_unsupported').all()
    try:
        with temporary_power_limit(f,object(),180000,150000,250000):
            assert f.cap==180000
            raise RuntimeError('test')
    except RuntimeError:pass
    assert f.cap==250000

def test_missing_gpu_not_summed(tmp_path):
    from types import SimpleNamespace
    from aidc_data.canonical import canonicalize
    raw=pd.DataFrame({'timestamp':['2026-01-01 00:00:00.000','2026-01-01 00:00:00.020']})
    for i in range(8):raw[f'gpu{i}_power_W']=[100.,100.]
    raw.loc[1,'gpu7_power_W']=np.nan
    p=tmp_path/'raw.csv';raw.to_csv(p,index=False)
    c,_=canonicalize(SimpleNamespace(source_path=str(p),source_hash='fake',session_id='s',split_group_id='s',run_id='r'),{'gap_dt_factor':5})
    assert c.gpu_power_w.iloc[0]==800
    assert pd.isna(c.gpu_power_w.iloc[1]) and c.valid_gpu_count.iloc[1]==7
