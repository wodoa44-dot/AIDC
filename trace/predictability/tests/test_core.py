import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from src.core import steps,continuous_segments,bin_segment,features,samples,crps,rearrange,skill


def test_crps_known_cases():
    q=np.array([.01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99])
    np.testing.assert_allclose(crps(np.array([2.,-3.,0.]),np.zeros((3,len(q))),q),[2,3,0])
    dense=np.linspace(.0001,.9999,10001)
    np.testing.assert_allclose(crps([.5],dense[None,:],dense),[1/12],atol=1e-7)
    np.testing.assert_allclose(crps([0.],np.ones((1,len(q)))*4,q),[4])
    assert np.isnan(skill(0,0))


def test_physical_horizons():
    assert steps(2000,50)==40
    for h,r in [(20,50),(50,200),(50,20)]:
        with pytest.raises(ValueError):steps(h,r)


def test_boundaries_and_no_gap_fill():
    d=pd.DataFrame({'time_s':[0,.02,.04,.5,.52,.54,.56,.58,.60,.62], 'power_w':[1,2,3,4,5,np.nan,6,7,8,9]})
    segs=continuous_segments(d,.1)
    assert [len(x) for _,x in segs]==[3,2,4]
    for _,z in segs:assert np.diff(z.time_s).max()<=.1
    b=bin_segment(pd.DataFrame({'time_s':np.arange(0,2,.02),'power_w':np.arange(100)}),200)
    assert (b.source_max_s<=b.time_s+1e-9).all()
    assert (b.source_min_s>b.time_s-.2-1e-9).all()
    assert (b.time_s<=1.98).all()


def test_features_causal():
    p=np.arange(100,dtype=float);a=features(p,50,1000);p[51:]+=10000;b=features(p,50,1000)
    np.testing.assert_allclose(a[:51],b[:51],equal_nan=True)


def test_split_and_target_identity():
    z=pd.DataFrame({'time_s':np.arange(1,1001)*.05,'power_w':np.arange(1000), 'bin':np.arange(1,1001)})
    meta,f=samples(z,50,1000,[200,1000,5000],30,40,200)
    np.testing.assert_allclose(meta.target_s-meta.issue_s,1)
    assert (meta[meta.split==0].target_s<30).all()
    assert (meta[meta.split==1].issue_s>=30).all()
    assert (meta[meta.split==2].issue_s>=40).all()
    assert all(len(v)==len(meta) for v in f.values())
    assert not meta.target_bin.duplicated().any()
    assert meta[meta.split==1].issue_s.min()<35 # earlier-partition history allowed


def test_prediction_order_shape():
    p,cross=rearrange([[3,1,2],[1,2,3]])
    assert p.shape==(2,3) and cross==.5
    assert (np.diff(p,axis=1)>=0).all()
    with pytest.raises(ValueError):rearrange([[np.nan]])


def test_bin_future_change_cannot_affect_past():
    d=pd.DataFrame({'time_s':np.arange(100)*.02,'power_w':np.arange(100,dtype=float)})
    a=bin_segment(d,200);d.loc[d.time_s>1,'power_w']+=1e6;b=bin_segment(d,200)
    np.testing.assert_allclose(a[a.time_s<=1].power_w,b[b.time_s<=1].power_w)


def test_one_bin_history_is_current_only():
    x=features(np.arange(10.),200,200)
    assert np.isfinite(x).all()
    np.testing.assert_allclose(x[:,0],np.arange(10.))
    assert (x[:,1:]==0).all() # only zero slope if present
