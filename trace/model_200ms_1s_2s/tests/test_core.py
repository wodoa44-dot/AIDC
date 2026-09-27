import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/"trace/predictability/src")]
from src.core import power_features, fit_encoder, transform, choose_per_run, equal_run_weights, crps, rearrange
from core import features

def test_online_feature_matches_verified_builder():
    p=np.arange(40,dtype=float)**1.2
    for h in (1000,5000): np.testing.assert_allclose(power_features(p,200,h),features(p,200,h)[-1])

def test_train_only_encoder_unknown_and_missing():
    m=pd.DataFrame({"gpu":["A","B"],"workload":["x","y"],"batch":[1,np.nan],"ai_model":["m","n"]})
    e=fit_encoder(m,np.array([0]),"history_meta"); x=transform(np.ones((2,1)),m,e)
    assert e["vocab"]["gpu"]==["A"] and x.shape[0]==2 and np.isfinite(x).all()
    assert x[1,e["names"].index("gpu=__UNKNOWN__")+1]==1

def test_equal_run_weights_and_deterministic_sampling():
    m=pd.DataFrame({"run_id":["a"]*7+["b"]*3}); ids=np.arange(10); pick=choose_per_run(m,ids,3); w=equal_run_weights(m,pick)
    assert np.array_equal(pick,choose_per_run(m,ids,3)); np.testing.assert_allclose(pd.DataFrame({"r":m.iloc[pick].run_id,"w":w}).groupby("r").w.sum(),2)

def test_crps_and_ordering():
    p,c=rearrange([[3,1,2]]); assert c==1 and np.array_equal(p,[[1,2,3]])
    assert crps(np.array([2]),p,np.array([.1,.5,.9]))[0]>=0
