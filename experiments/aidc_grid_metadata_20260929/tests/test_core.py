import numpy as np
import pandas as pd
from aidc_exp.preprocessing import bin_samples, static_metadata
from aidc_exp.features import build_features
from aidc_exp.targets import build_targets
from aidc_exp.evaluation import crps_grid, aggregate
from aidc_exp.control import rollout


def test_future_perturbation_cannot_change_past_features():
    t=np.arange(500)*.02
    a=bin_samples(t,np.ones(len(t)),min_coverage=.8)
    b=bin_samples(t,np.where(t>5,10.,1.),min_coverage=.8)
    x=build_features(a,history_bins=5);y=build_features(b,history_bins=5)
    pd.testing.assert_frame_equal(x.loc[:24],y.loc[:24])


def test_gap_resets_history_and_target():
    t=np.r_[np.arange(150)*.02, 5+np.arange(150)*.02]
    b=bin_samples(t,np.ones(len(t)),gap_factor=5)
    x=build_features(b,history_bins=5)
    y=build_targets(b,[5])
    assert not x.loc[25:29,'eligible_history'].any()
    assert y.loc[12:15,'y_5'].isna().all()


def test_target_boundary():
    b=pd.DataFrame({'power_w':np.arange(15,dtype=float),'segment':np.zeros(15,dtype=int)})
    y=build_targets(b,[1,5])
    assert y.loc[0,'y_1']==1
    assert y.loc[0,'y_5']==5
    assert y.loc[0,'max_ramp_02_5']==1
    assert y.loc[0,'max_ramp_1s_5']==5
    assert np.isnan(y.loc[11,'y_5'])


def test_unknown_metadata():
    m=static_metadata('/tmp/H100/ModelSize128_BatchSize32.csv')
    assert m['model_size_token']==128
    assert np.isnan(m['zero_stage'])


def test_quantile_score_and_order():
    q=[.1,.5,.9]
    assert crps_grid([0],np.array([[0,0,0]]),q)[0]==0
    a=crps_grid([0],np.array([[-1,0,1]]),q)[0]
    b=crps_grid([0],np.array([[-2,0,2]]),q)[0]
    assert a>0 and b>a
    assert np.all(np.diff(np.sort([[3,1,2]],axis=1),axis=1)>=0)


def test_run_balanced_aggregation():
    m=pd.DataFrame({'model':['P2']*3,'horizon':[1.]*3,'run_id':['a','a','b'],
       'crps':[0,2,10],'mae':[0,2,10],'rmse':[0,2,10],'bias':[0,2,10],
       'p95_abs':[0,2,10],'p99_abs':[0,2,10]})
    assert aggregate(m).iloc[0].crps==5.5


def test_storage_balance_delay_soc():
    d=np.r_[np.ones(10)*100,np.ones(10)*500]
    z=rollout(d,np.ones(20)*100,mode='C3',capacity_w=100,delay_bins=2,ess_wh=1,reserve=.2)
    assert abs(z.power_balance_error_w).max()<1e-9
    assert z.soc.min()>=.2-1e-12 and z.soc.max()<=1+1e-12
    assert z.fast_storage_w.iloc[10]==0
    assert (z.fast_storage_w.abs()<=50+1e-9).all()


def test_split_groups_do_not_overlap(tmp_path):
    from aidc_exp.splits import make_splits
    m=pd.DataFrame({'run_id':[f'run{i}' for i in range(6)]})
    folds=make_splits(m,tmp_path/'split.json',folds=3)
    assert len(folds)==3
    for fold in folds:
        assert not set(fold['train_runs']) & set(fold['test_runs'])
    assert sorted(sum((f['test_runs'] for f in folds),[]))==sorted(m.run_id)


def test_training_sampling_is_reproducible_and_run_balanced():
    from aidc_exp.splits import sample_training
    d=pd.DataFrame({'run_id':['a']*100+['b']*20,'bin':list(range(100))+list(range(20))})
    x=sample_training(d,10,2026);y=sample_training(d,10,2026)
    pd.testing.assert_frame_equal(x,y)
    assert np.allclose(x.groupby('run_id').sample_weight.sum(),1)
