from types import SimpleNamespace
import numpy as np
import pandas as pd
from powerprep.standardize import validate,aggregate,read_native
from powerprep.forecast import split_assignments,window_indices,training_normalization,bin_splits
from powerprep.contract import cache_key,commit,cached
from powerprep.events_v2 import export_events,recovery_measurement


def native(t,p=None):return pd.DataFrame({'time_s':t,'power_w':np.ones(len(t)) if p is None else p,'source_row':np.arange(len(t))})


def test_reset_invalid_negative_zero_duration():
    d,s,q=validate(native([0,1,2,0,1,2,3,4,5],[1,1,1,1,1,-1,0,np.nan,1]),'r','d')
    assert q['valid_duration_s']==3
    assert d.loc[6,'is_valid'] and not d.loc[5,'is_valid']
    v=aggregate(d,1)
    assert len(v)==3 and v.segment_id.nunique()==2
    assert not v.duplicated(['segment_id','bin_index']).any()


def test_missing_entire_gpu_column(tmp_path):
    d=pd.DataFrame({'timestamp':pd.date_range('2020',periods=3,freq='s')})
    for i in range(7):d[f'gpu{i}_power_W']=1
    path=tmp_path/'x.csv';d.to_csv(path,index=False)
    x=read_native(SimpleNamespace(dataset='highres_node',source_path=path))
    assert x.power_w.isna().all()


def test_bin_availability_and_conservation():
    d,_,_=validate(native([0,.3,1,1.7,2],[2,4,8,10,12]),'r','d')
    v=aggregate(d,1)
    assert np.allclose(v.power_w,[3.4,8.6])
    assert np.allclose(v.time_s,[1,2]) and np.allclose(v.available_s,v.bin_end_s)
    assert len(aggregate(d,.1))==0


def test_invalid_gap_windows_and_strict_split():
    p=np.ones(60);p[20]=np.nan
    d,_,_=validate(native(np.arange(60),p),'r','d');v=aggregate(d,1)
    a=SimpleNamespace(protocol='within_run',run_id='r',dataset_id='d',group_id='g',split='chronological',cut1_row=30,cut2_row=45)
    spec=dict(context_s=5,horizon_s=3,stride_s=1,output_interval_s=1,resolution_s=1,target_definition='future_trajectory')
    rows,error=window_indices(v,a,spec);assert rows and not error
    labels=bin_splits(v,a)
    for r in rows:
        z=v.loc[r['context_start_index']:r['target_end_index']]
        assert z.segment_id.nunique()==1
        assert len(set(labels[r['context_start_index']:r['target_end_index']+1]))==1
        assert r['feature_available_s']==r['issue_s']<r['target_available_s']


def test_alias_group_splits_and_small_group_rejection():
    m=pd.DataFrame({'dataset':['d']*5,'run_id':['a','aa','b','c','e'],'is_duplicate':[False,True,False,False,False],'group_id':['g','g','h','i','j']})
    q=pd.DataFrame({'run_id':['a','b','c','e'],'native_samples':[100]*4})
    cfg={'fractions':[.6,.2,.2],'seed':1,'protocols':['held_out_group'],'import_manifest':None}
    a=split_assignments(m,q,cfg);assert set(a.split)=={'train','validation','test'} and len(a)==4
    m.loc[:,'group_id']='g';a=split_assignments(m,q,cfg);assert (a.support_status=='unsupported').all()


def test_training_normalization_does_not_read_test():
    frame=pd.DataFrame({'source_row_start':range(10),'source_row_end':range(10),'power_w':[1.,2.,3.]+[100000.]*7})
    a=SimpleNamespace(protocol='within_run',cut1_row=3,cut2_row=6)
    mask=bin_splits(frame,a)=='train';before=training_normalization(frame.loc[mask,'power_w'])
    frame.loc[~mask,'power_w']=1e12
    assert training_normalization(frame.loc[mask,'power_w'])==before and before['mean_w']==2.


def test_code_change_and_corruption_invalidate_cache(tmp_path):
    a=cache_key('source',{},'v1');b=cache_key('source',{},'v2');assert a!=b
    p=tmp_path/'data';p.write_text('valid');side=tmp_path/'cache.json';commit(side,a,[p])
    assert cached(side,a,[p]) and not cached(side,b,[p])
    p.write_text('truncated');assert not cached(side,a,[p])


def test_complete_event_export(tmp_path):
    frame=pd.DataFrame({'event_id':np.arange(100005)});export_events(tmp_path,frame)
    assert len(pd.read_csv(tmp_path/'events.csv'))==100005


def test_recovery_final_episode_and_unresolved():
    t=np.arange(10,dtype=float);p=np.array([0,0,2,0,0,2,5,9,10,10],float)
    r=recovery_measurement(t,p,0,7,8,0,10)
    assert r['recovery_10_s']==5 and r['recovery_90_s']==7 and r['recovery_rise_time_10_90_s']==2
    r=recovery_measurement(t,np.array([0]*7+[10]*3),0,7,8,0,10)
    assert np.isnan(r['recovery_rise_time_10_90_s']) and 'unresolved' in r['recovery_measurement_status']
    assert recovery_measurement(t,p,0,None,None,0,10)['recovery_measurement_status']=='right_censored'


def test_dynamic_report_join_failure_and_changed_counts(tmp_path):
    from powerprep.pipeline_v2 import report_v2,mark
    from powerprep.contract import save_table
    (tmp_path/'plots').mkdir();(tmp_path/'prepared').mkdir()
    manifest=pd.DataFrame({'dataset':['synthetic'],'run_id':['r'],'is_duplicate':[False],'metadata_join_status':['unmatched']})
    save_table(tmp_path/'manifest.csv',manifest)
    d,s,q=validate(native([0,1,2]),'r','synthetic');save_table(tmp_path/'prepared/r.parquet',d)
    save_table(tmp_path/'native_quality.csv',pd.DataFrame([q]));cfg={'plots':{'max_full_points':100}}
    mark(tmp_path,cfg,'preprocess');report_v2(tmp_path,cfg)
    first=(tmp_path/'preprocessing_report.md').read_text();assert 'unmatched' in first and 'All NLR' not in first
    q['native_samples']=111;q['valid_samples']=100
    save_table(tmp_path/'native_quality.csv',pd.DataFrame([q]));report_v2(tmp_path,cfg)
    second=(tmp_path/'preprocessing_report.md').read_text();assert second!=first and '111' in second


def test_cli_stages_dispatch_independently(tmp_path,monkeypatch):
    from powerprep import pipeline_v2 as v2
    calls=[]
    monkeypatch.setattr(v2,'inventory_v2',lambda *a:calls.append('inventory'))
    monkeypatch.setattr(v2,'standardize_v2',lambda *a:calls.append('preprocess'))
    monkeypatch.setattr(v2,'resolutions_v2',lambda *a:calls.append('resolutions'))
    cfg={'datasets':['synthetic'],'events':{'enabled':False}}
    v2.run('inventory',cfg,tmp_path);assert calls==['inventory']
    calls.clear();v2.run('preprocess',cfg,tmp_path);assert calls==['preprocess']
    calls.clear();v2.run('resolutions',cfg,tmp_path);assert calls==['resolutions']


def test_complete_fractional_bins_not_excluded_by_float_roundoff():
    d,_,_=validate(native(np.arange(0,100,.1)),'r','d')
    v=aggregate(d,.2,min_coverage=1.)
    assert len(v)==499 and v.is_valid.all() and (v.coverage==1.).all()


def test_single_machine_native_channel_and_bad_footer(tmp_path):
    path=tmp_path/'machine.CSV'
    pd.DataFrame({'Date':['30.9.2025','30.9.2025','Date'],'Time':['20:46:9.906','20:46:10.006','Time'],
                  'GPU Power [W]':['20','0','GPU Power [W]'],'CPU Package Power [W]':['80','90','CPU']}).to_csv(path,index=False,encoding='cp1252')
    row=SimpleNamespace(dataset='highres_single',source_path=path)
    raw=read_native(row);d,s,q=validate(raw,'r','highres_single')
    assert d.power_w.iloc[0]==20 and d['CPU Package Power [W]'].iloc[0]==80
    assert d.is_valid.tolist()==[True,True,False]
    assert np.isclose(q['valid_duration_s'],.1)


def test_partial_clock_recovers_only_relative_time_and_splits_wrap(tmp_path):
    p=tmp_path/'short.CSV'
    pd.DataFrame({'Date':['29.9.2025']*5,'Time':['59:59.8','59:59.9','00:00.0','00:00.1','00:00.2'],'GPU Power [W]':[10]*5}).to_csv(p,index=False)
    raw=read_native(SimpleNamespace(dataset='highres_single',source_path=p))
    d,s,q=validate(raw,'r','highres_single')
    assert not d.timestamp_hour_known.any() and d.is_valid.all()
    assert len(s)==2 and np.isclose(q['valid_duration_s'],.3)
    assert q['hour_unknown_valid_samples']==5
