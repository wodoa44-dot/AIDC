from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .inventory import make_inventory, fingerprint
from .preprocessing import load_highres, static_metadata
from .features import build_features
from .targets import build_targets
from .splits import make_splits
from .training import train_all
from .reporting import make_tables,make_figures,report
from .grid_adapter import unavailable
from .control import rollout,summarize

PROJECT=Path(__file__).resolve().parents[2]
STAGES=["audit","prepare","train","evaluate","control","grid","report"]


def save_json(path, obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)


def state(out, stage, status, detail=""):
    path=out/"status.json"
    s=json.loads(path.read_text()) if path.exists() else {"stages":{}}
    s["experiment_id"]=out.name;s["updated_utc"]=datetime.now(timezone.utc).isoformat()
    s["stages"][stage]={"status":status,"detail":detail,"updated_utc":s["updated_utc"]}
    save_json(path,s)
    print(f"{stage}: {status} {detail}",flush=True)


def prepare(out, cfg):
    manifest=make_inventory(out)
    if cfg.get("smoke_runs"):
        manifest=manifest.sort_values("run_id").head(cfg["smoke_runs"])
    splits=make_splits(manifest,out/"split_manifest.json",cfg["outer_folds"])
    cache=PROJECT/"data_cache"; cache.mkdir(exist_ok=True)
    frames=[]
    for _,r in manifest.iterrows():
        prep_key=hashlib.sha256(json.dumps({k:cfg[k] for k in ["horizons_bins","history_bins","min_coverage","gap_factor"]},sort_keys=True).encode()).hexdigest()[:12]
        path=cache/f"{r.run_id}_{prep_key}.parquet"
        if path.exists():d=pd.read_parquet(path)
        else:
            b=load_highres(r.source_path,bin_s=.2,min_coverage=cfg["min_coverage"],gap_factor=cfg["gap_factor"])
            f=build_features(b,history_bins=cfg["history_bins"])
            y=build_targets(b,cfg["horizons_bins"])
            d=pd.concat([b[["bin","segment"]],f,y],axis=1)
            d["issuance_s"]=b.right_s
            d["dataset"]="highres_node";d["run_id"]=r.run_id;d["group_id"]=r.run_id
            for k,v in static_metadata(r.source_path).items():d[k]=v
            tmp=path.with_suffix(".tmp.parquet");d.to_parquet(tmp,index=False);tmp.replace(path)
        frames.append(d)
        print(f"PREPARED {r.run_id} rows={len(d)}",flush=True)
    data=pd.concat(frames,ignore_index=True)
    data.to_parquet(out/"prepared.parquet",index=False)
    return data,splits


def do_control(out,cfg):
    pred_path=out/"per_target_predictions.parquet"
    if not pred_path.exists():return "blocked_dependency"
    p=pd.read_parquet(pred_path)
    p=p[np.isclose(p.horizon,1.) & p.seed.eq(cfg["seeds"][0])]
    if p.empty:return "blocked_dependency"
    prepared=pd.read_parquet(out/"prepared.parquet",columns=["run_id","current_w","eligible_history"])
    split=json.loads((out/"split_manifest.json").read_text())["folds"]
    rows=[]; traces=[]
    for (run,segment),g in p.groupby(["run_id","segment"]):
        wide=g.pivot(index="issuance_s",columns="model",values="q0.5").dropna(subset=["P2","P4"])
        demand=g[g.model=="P2"].set_index("issuance_s").current_w.reindex(wide.index).to_numpy()
        if len(demand)<2:continue
        fold=int(g.fold.iloc[0]); train_runs=split[fold]["train_runs"]
        training_power=prepared.loc[prepared.run_id.isin(train_runs)&prepared.eligible_history,"current_w"].dropna().to_numpy()
        capacity=.3*np.percentile(training_power,95)
        for mode in ["C0","C1","C2","C3","C4","C5"]:
            forecast={"C3":wide["P2"].to_numpy(),"C4":wide["P4"].to_numpy(),"C5":g[g.model=="P2"].set_index("issuance_s").actual_w.reindex(wide.index).to_numpy()}.get(mode)
            z=rollout(demand,forecast,mode,capacity_w=capacity,ess_wh=capacity*2,delay_bins=0)
            z["run_id"]=run;z["segment"]=segment;z["controller"]=mode;z["issuance_s"]=wide.index.to_numpy();traces.append(z)
            rows.append({**summarize(z,mode,run),"segment":segment,"capacity_w":capacity})
    if not rows:return "blocked_dependency"
    pd.DataFrame(rows).to_csv(out/"control_metrics.csv",index=False)
    pd.concat(traces,ignore_index=True).to_parquet(out/"control_traces.parquet",index=False)
    pd.DataFrame([{"assumption":"capacity","value":"0.3 x outer-training P95 W"},
        {"assumption":"delay","value":"0 s"},{"assumption":"round-trip efficiency","value":"0.95 each direction"},
        {"assumption":"ESS energy","value":"2 h at assumed capacity"}]).to_csv(out/"assumptions.csv",index=False)
    return "completed_assumed_E1"


def main():
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=STAGES+["all","status","resume","make_tables","make_figures"])
    ap.add_argument("--config",default=str(PROJECT/"configs/full.yaml"));ap.add_argument("--id",default=None)
    a=ap.parse_args();cfg=yaml.safe_load(Path(a.config).read_text())
    eid=a.id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=PROJECT/"outputs"/eid;out.mkdir(parents=True,exist_ok=True)
    if a.command=="status":
        print((out/"status.json").read_text() if (out/"status.json").exists() else "no status")
        return
    code_files=list((PROJECT/"src").rglob("*.py"))
    code_hash=fingerprint(code_files);data_hash=hashlib.sha256((PROJECT.parent.parent/"trace/preprocessing/outputs/manifest.csv").read_bytes()).hexdigest()
    fp=hashlib.sha256((Path(a.config).read_text()+code_hash+data_hash).encode()).hexdigest()
    fp_path=out/"fingerprint.json"
    if fp_path.exists() and json.loads(fp_path.read_text())["fingerprint"]!=fp:
        raise RuntimeError("checkpoint fingerprint mismatch; use a new experiment ID")
    save_json(fp_path,{"fingerprint":fp,"code_hash":code_hash,"manifest_hash":data_hash,"config":str(Path(a.config).resolve()),"python":sys.executable})
    (out/"config_snapshot.yaml").write_text(Path(a.config).read_text())
    requested=STAGES if a.command in ("all","resume") else [a.command]
    for stage in requested:
        s=json.loads((out/"status.json").read_text()).get("stages",{}) if (out/"status.json").exists() else {}
        if s.get(stage,{}).get("status") in ("completed","completed_assumed_E1","blocked_dependency") and a.command=="resume":continue
        state(out,stage,"running")
        try:
            if stage=="audit":make_inventory(out);result="completed"
            elif stage=="prepare":prepare(out,cfg);result="completed"
            elif stage=="train":
                data=pd.read_parquet(out/"prepared.parquet")
                splits=json.loads((out/"split_manifest.json").read_text())["folds"]
                train_all(data,splits,out,cfg);result="completed"
            elif stage=="evaluate":result="completed" if make_tables(out,cfg["quantiles"]) else "blocked_dependency"
            elif stage=="control":result=do_control(out,cfg)
            elif stage=="grid":unavailable(out);result="blocked_dependency"
            elif stage=="report":make_figures(out);report(out);result="completed"
            state(out,stage,result)
        except Exception as e:
            state(out,stage,"failed",str(e));traceback.print_exc()
    status=json.loads((out/"status.json").read_text())
    save_json(out/"completion_status.json",{"all_completed":all(v["status"]=="completed" for v in status["stages"].values()),"stages":status["stages"]})

if __name__=="__main__":main()
