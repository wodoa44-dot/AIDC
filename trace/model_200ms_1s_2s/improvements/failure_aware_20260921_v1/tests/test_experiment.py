from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
BASE = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(BASE / "validation"))
from experiment import (longest_exceedance_seconds, ordinary_equal_run_weights,
                        reconstruct_power_predictions, select_candidates,
                        transition_weight_table, validate_or_create_signature)
from metrics import failure_episodes
from runtime import predict_item


def config():
    return {"transition_threshold_w":100.0,"weight_ratio_cap":4.0,
            "crps_feasibility_tolerance":.05}


def test_residual_reconstruction():
    raw=np.array([[-10.,0.,20.],[5.,10.,15.]])
    current=np.array([100.,200.])
    np.testing.assert_allclose(reconstruct_power_predictions(raw,current,"residual"),
                               [[90,100,120],[205,210,215]])
    np.testing.assert_allclose(reconstruct_power_predictions(raw,current,"absolute"),raw)


def test_transition_weights_use_only_selected_rows_and_equalize_runs():
    meta=pd.DataFrame({"run_id":["a"]*5+["b"]*4+["outside"],
                       "current":[0]*10,"y":[0,0,0,200,-200,0,0,200,-200,9999]})
    ids=np.arange(9)
    weights,summary=transition_weight_table(meta,ids,config())
    totals=pd.DataFrame({"run":meta.iloc[ids].run_id,"w":weights}).groupby("run").w.sum()
    np.testing.assert_allclose(totals,[1,1])
    changed=meta.copy(); changed.loc[9,"y"]=-9999
    weights2,_=transition_weight_table(changed,ids,config())
    np.testing.assert_allclose(weights,weights2)
    assert summary.groupby("run_id").run_total_weight.first().eq(1).all()
    assert weights.max()/weights.min() <= 4.0+1e-12


def test_ordinary_weights_equal_total_per_run():
    meta=pd.DataFrame({"run_id":["a","a","b","b","b"]})
    w=ordinary_equal_run_weights(meta,np.arange(5))
    totals=pd.DataFrame({"run":meta.run_id,"w":w}).groupby("run").w.sum()
    np.testing.assert_allclose(totals,[1,1])


def test_grouped_split_manifest_is_isolated():
    folds=pd.read_csv(BASE.parents[1]/"trace/predictability_meta/outputs/corrected_full_20260915/splits/folds.csv")
    for _,group in folds.dropna(subset=["inner_fold"]).groupby(["outer_fold","inner_fold"]):
        train=set(group.loc[group.role=="inner_train","run_id"])
        val=set(group.loc[group.role=="inner_val","run_id"])
        assert train.isdisjoint(val)


def test_reliability_selection_is_deterministic(tmp_path):
    rows=[]
    for i,(crps,tail,exceed,episode,size) in enumerate([
        (100,30,.3,3,1000),(102,20,.2,2,2000),(106,1,.01,.1,10)]):
        rows.append({"horizon_ms":1000,"candidate_id":f"c{i}","family":"absolute",
            "history_ms":1000,"n_estimators":100,"num_leaves":15,"run_balanced_crps_w":crps,
            "run_balanced_mae_w":1,"mean_run_worst_5pct_mae_w":tail,
            "mean_run_exceedance_500_fraction":exceed,"mean_run_longest_500_episode_s":episode,
            "model_bytes_full15":size,"fit_s":1,"n_validation_runs":3,"n_validation_targets":30})
    table=pd.DataFrame(rows)
    a=select_candidates(table,config(),"final",tmp_path)
    b=select_candidates(table,config(),"final",tmp_path)
    pd.testing.assert_frame_equal(a,b)
    assert a.set_index("procedure").loc["mean_crps_selection","candidate_id"]=="c0"
    assert a.set_index("procedure").loc["reliability_selection","candidate_id"]=="c1"


def test_continuous_episode_segmentation_breaks_on_gaps_and_segments():
    frame=pd.DataFrame({"assessment":"x","run_id":"r","segment_key":["a","a","a","a","b"],
        "issue_s":[0,.2,.6,.8,1.0],"error_w":[600,700,800,900,1000]})
    episodes=failure_episodes(frame,500)
    assert sorted(episodes.n_slots.tolist())==[1,2,2]
    assert longest_exceedance_seconds(frame.assign(segment_id=frame.segment_key,subsegment_id=0),500)==.4


class FakeModel:
    def __init__(self,value): self.value=value
    def predict(self,x,num_iteration=None): return np.full(len(x),self.value)
    class Booster:
        @staticmethod
        def num_trees(): return 1
    booster_=Booster()


def test_offline_and_deployed_quantile_postprocessing_match():
    encoder={"groups":[],"vocab":{},"batch_median":None,"names":[]}
    item={"models":[FakeModel(2),FakeModel(-1),FakeModel(1)],"encoder":encoder,
          "parameters":{"family":"residual"},"family":"residual","quantiles":[.1,.5,.9]}
    raw,ordered,_=predict_item(item,np.array([100.]),{})
    np.testing.assert_allclose(raw,[[102,99,101]])
    np.testing.assert_allclose(ordered,[[99,101,102]])


def test_configuration_aware_resume_rejects_changes(tmp_path):
    path=tmp_path/"signature.json"
    validate_or_create_signature(path,{"x":1})
    validate_or_create_signature(path,{"x":1})
    with pytest.raises(ValueError): validate_or_create_signature(path,{"x":2})
