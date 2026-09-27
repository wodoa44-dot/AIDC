from __future__ import annotations
import hashlib, re
from collections import defaultdict
from pathlib import Path
import pandas as pd

def stream_hash(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while b:=f.read(chunk): h.update(b)
    return h.hexdigest()

def _high_metadata(path):
    s=str(path); name=path.name
    hardware="B200" if "B200" in s else "H100" if "H100" in s else "unknown"
    workload="image_generation" if "Image Generation" in s else "llm_training"
    model="diffusion" if workload=="image_generation" else "Llama"
    return dict(workload=workload, model=model, hardware=hardware, gpu_count=8,
                node_count=1, power_scope="GPU sum (8 GPUs); CPU excluded")

def _normalize_result_key(value):
    if pd.isna(value): return None
    p=Path(str(value).replace("\\","/"))
    try:
        i=p.parts.index("01_aggregated_datasets")
        return "/".join(p.parts[i+1:])
    except ValueError:
        return "/".join(p.parts[-3:])

def discover(cfg, selected=None, limit=None):
    selected=selected or cfg["datasets"]
    rows=[]
    if "highres_node" in selected:
        root=Path(cfg["inputs"]["highres"])
        files=sorted(root.rglob("Node_Dataset/**/*.csv"))
        for p in files:
            d=_high_metadata(p); d.update(dataset="highres_node",source_path=str(p.resolve()),
                metadata_join_status="not_applicable",completion_status="unknown",
                timestamp_column="timestamp",source_units="W",
                scope_evidence="CSV gpu0..gpu7_power_W columns; dataset README")
            rows.append(d)
    groups=[]
    if "nlr_training" in selected: groups.append(("nlr_training","training"))
    if "nlr_online_rate" in selected: groups.append(("nlr_online_rate","inference_online_rate_llama3_70b"))
    if "nlr_offline" in selected: groups.append(("nlr_offline","inference_offline_llama3_70b"))
    if "nlr_online_finite" in selected: groups.append(("nlr_online_finite","inference_online_finite_llama3_70b"))
    agg=Path(cfg["inputs"]["nlr"])/"01_aggregated_datasets"
    for dataset,group in groups:
        gd=agg/group; meta=pd.read_csv(gd/"metadata.csv")
        pathcol="path_run" if "path_run" in meta else "path_save"
        keys=defaultdict(list)
        for i,v in meta[pathcol].items(): keys[_normalize_result_key(v)].append(i)
        for p in sorted((gd/"results").glob("*.parquet")):
            key=f"{group}/results/{p.name}"; matches=keys.get(key,[])
            md=meta.loc[matches[0]].to_dict() if len(matches)==1 else {}
            complete=md.get("run_completed","unknown")
            rows.append(dict(dataset=dataset,source_path=str(p.resolve()),workload=("training" if dataset=="nlr_training" else "inference"),
                model=md.get("model","Llama3 70B"),hardware="unknown",gpu_count="unknown",
                node_count=md.get("nodes",1),power_scope="whole job CPU+GPU sum" if dataset=="nlr_training" else "single node CPU+GPU sum",
                metadata_join_status="matched" if len(matches)==1 else ("unmatched" if not matches else "ambiguous"),
                completion_status=complete,timestamp_column="timestep[s]",source_units="W",
                scope_evidence="NLR postprocess.py sums NVML GPU W and RAPL CPU W; utilities.py aligns/interpolates"))
    if limit:
        out=[]
        for ds in selected:
            subset=[r for r in rows if r["dataset"]==ds][:limit]
            out.extend(subset)
        rows=out
    for r in rows:
        p=Path(r["source_path"]); r["source_hash"]=stream_hash(p); r["source_bytes"]=p.stat().st_size
    byhash=defaultdict(list)
    for r in rows: byhash[r["source_hash"]].append(r)
    for hs,grp in byhash.items():
        canonical=min(x["source_path"] for x in grp)
        for r in grp:
            r["canonical_path"]=canonical; r["is_duplicate"]=r["source_path"]!=canonical
            r["run_id"]=f'{r["dataset"]}-{hs[:16]}'
    return pd.DataFrame(rows)

def duplicate_map(manifest):
    d=manifest[manifest.duplicated("source_hash",False)].copy()
    if d.empty: return pd.DataFrame(columns=["source_hash","canonical_path","alias_path","run_id"])
    return d.rename(columns={"source_path":"alias_path"})[["source_hash","canonical_path","alias_path","run_id"]]
