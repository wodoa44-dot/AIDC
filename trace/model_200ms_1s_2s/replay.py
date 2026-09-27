#!/usr/bin/env python
"""Chronological real-run replay and batch/export equivalence check."""
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
CORRECTED = ROOT / "trace/predictability_meta/outputs/corrected_full_20260915"
sys.path.insert(0, str(HERE))
from src.core import raw_predict, rearrange, transform_one
from src.online import ForecastService

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--package",required=True); ap.add_argument("--output",required=True); ap.add_argument("--run-id")
    a=ap.parse_args(); pkg=joblib.load(a.package); service=ForecastService(a.package)
    held=pd.read_csv(CORRECTED/"reused_holdout_runs.csv"); run=a.run_id or sorted(held.run_id)[0]
    metas={}; powers={}
    for h in (1000,2000):
        m=pd.read_parquet(CORRECTED/f"features/meta_200_{h}.parquet"); ids=np.flatnonzero(m.run_id.eq(run)); metas[h]=(m,ids)
        z=np.load(CORRECTED/f"features/power_200_{h}.npz"); powers[h]={k:z[k] for k in z.files}
    m,ids=metas[1000]; stream=m.iloc[ids].sort_values(["segment_id","subsegment_id","issue_s"])
    rows=[]; maxdiff=0.; compared=0; last_group=None
    for row in stream.itertuples():
        group=(row.segment_id,row.subsegment_id)
        if group != last_group: service.reset(); last_group=group
        md={"gpu":row.gpu,"workload":row.workload,"batch":row.batch,"ai_model":row.ai_model}
        got=service.update(round(row.issue_s*1000),row.current,md)
        if got["status"] != "ready": continue
        for forecast in got["forecasts"]:
            h=forecast["horizon_ms"]; hm,hids=metas[h]
            match=hids[np.isclose(hm.iloc[hids].issue_s.to_numpy(),row.issue_s)]
            if not len(match): continue
            idx=int(match[0]); item=pkg["horizons"][str(h)]
            enc=item["encoder"]; x=transform_one(powers[h][str(item["history_ms"])][idx],md,enc)
            direct,_=rearrange(raw_predict(item["models"],x)); online=np.array(list(forecast["quantiles"].values()))
            diff=float(np.max(np.abs(direct[0]-online))); maxdiff=max(maxdiff,diff); compared+=1
            rows.append({"run_id":run,"issue_s":row.issue_s,"target_s":row.issue_s+h/1000,"horizon_ms":h,
                         "q10":online[0],"q50":online[1],"q90":online[2],"batch_replay_max_abs_diff_w":diff})
    if not rows: raise RuntimeError("replay produced no comparable predictions")
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_parquet(a.output,index=False)
    result={"passed":maxdiff<=1e-10,"run_id":run,"n_compared":compared,"max_abs_difference_w":maxdiff,"output":a.output}
    print(json.dumps(result,indent=2));
    if not result["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
