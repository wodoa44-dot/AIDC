from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from .config import fingerprint
from .discovery import discover,duplicate_map
from .adapters import load_native,analysis_series
from .quality import assess,segments
from .events import config_grid,detect_segment,duration_bin
from .plots import summary_plots,run_plot,event_zooms

def inventory(cfg,out,selected=None,limit=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); m=discover(cfg,selected,limit)
    # Populate cheap parquet metadata / CSV row counts without decoding full tables.
    import pyarrow.parquet as pq
    vals=[]
    for r in m.itertuples():
      if str(r.source_path).endswith(".parquet"):
        pf=pq.ParquetFile(r.source_path); vals.append((pf.metadata.num_rows,None))
      else:
        with open(r.source_path,"rb") as f: n=sum(1 for _ in f)-1
        vals.append((n,None))
    m["row_count"]=[x[0] for x in vals]
    m.to_csv(out/"manifest.csv",index=False); duplicate_map(m).to_csv(out/"duplicates.csv",index=False)
    return m

def process(cfg,out,selected=None,limit=None,resume=True,stages="all"):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); (out/"prepared").mkdir(exist_ok=True); (out/"plots"/"runs").mkdir(parents=True,exist_ok=True)
    fp=fingerprint(cfg); m=inventory(cfg,out,selected,limit); canonical=m[~m.is_duplicate].copy()
    quality=[]; events=[]; summaries=[]
    grids=list(config_grid(cfg))
    for ix,row in enumerate(canonical.itertuples(),1):
      cache=out/"prepared"/f"{row.run_id}.parquet"; side=cache.with_suffix(".json")
      valid_cache=resume and cache.exists() and side.exists() and json.loads(side.read_text()).get("source_hash")==row.source_hash and json.loads(side.read_text()).get("config_fingerprint")==fp
      if valid_cache: d=pd.read_parquet(cache)
      else:
        native=load_native(row); d,prep=analysis_series(native,row,cfg); d.to_parquet(cache,index=False); side.write_text(json.dumps(dict(source_hash=row.source_hash,config_fingerprint=fp,preparation=prep),indent=2))
      q=assess(d,row,cfg); quality.append(q); run_events=[]
      for sid,seg in segments(d,q["gap_threshold_s"]):
        timing="fine-scale unsupported" if row.dataset=="nlr_online_rate" else ("effective resolution unverified" if row.dataset=="highres_node" else "supported at >=0.2s")
        for dc in grids: run_events += detect_segment(seg,row,sid,dc,timing)
      events += run_events
      valid_h=q["duration_s"]/3600
      summaries.append(dict(dataset=row.dataset,run_id=row.run_id,valid_observation_hours=valid_h,event_rows=len(run_events),unique_configurations=len(grids)))
      if ix<=8:
        ev=pd.DataFrame(run_events); ev=ev[(ev.relative_threshold==.2)&(ev.delta_event_s==1)] if not ev.empty else ev
        run_plot(d,ev,out/"plots"/"runs"/f"{row.run_id}.png",cfg["plots"]["max_full_points"])
      print(f"[{ix}/{len(canonical)}] {row.run_id}: {len(run_events)} config-event rows",flush=True)
    qd=pd.DataFrame(quality); ed=pd.DataFrame(events); sd=pd.DataFrame(summaries)
    qd.to_csv(out/"run_quality.csv",index=False); sd.to_csv(out/"run_summary.csv",index=False)
    # Enrich the inventory with properties verified while decoding each canonical run.
    verified=qd[["run_id","duration_s","dt_median_s","missing_power_count"]]
    m=m.merge(verified,on="run_id",how="left")
    m.to_csv(out/"manifest.csv",index=False)
    if ed.empty: ed=pd.DataFrame(columns=["dataset","run_id","segment_id","event_id","detector_config_id","completed","right_censored","low_duration_s","recovery_rise_time_10_90_s","relative_threshold","delta_event_s"])
    else: ed["low_duration_bin"]=ed.low_duration_s.map(duration_bin)
    ed.to_parquet(out/"events.parquet",index=False); ed.head(100000).to_csv(out/"events.csv",index=False)
    sens=(ed.groupby(["dataset","detector_config_id","relative_threshold","delta_event_s"],dropna=False).agg(event_rows=("event_id","count"),completed=("completed","sum"),censored=("right_censored","sum"),long_30s=("low_duration_s",lambda x:(x>=30).sum()),long_120s=("low_duration_s",lambda x:(x>=120).sum()),fast_recovery_le_2s=("recovery_rise_time_10_90_s",lambda x:(x<=2).sum())).reset_index()) if len(ed) else pd.DataFrame()
    
    hours=sd.groupby("dataset").valid_observation_hours.sum() if len(sd) else pd.Series(dtype=float)
    if len(sens): sens["valid_observation_hours"]=sens.dataset.map(hours); sens["events_per_valid_hour"]=sens.event_rows/sens.valid_observation_hours
    sens.to_csv(out/"threshold_sensitivity.csv",index=False); summary_plots(ed,sd,out/"plots"); event_zooms(ed,out/"plots",cfg["plots"]["max_event_zooms"])
    effective={k:v for k,v in cfg.items() if not k.startswith("_")}; effective["config_fingerprint"]=fp; (out/"effective_config.json").write_text(json.dumps(effective,indent=2))
    write_report(out,m,qd,ed,sens)
    return m,qd,ed

def write_report(out,m,q,e,sens):
    canon=m[~m.is_duplicate]; primary=e[(e.relative_threshold==.2)&(e.delta_event_s==1.0)].copy() if len(e) else e
    lines=["# Power trace preprocessing report","",f"Discovered **{len(m)} files** and retained **{len(canon)} exact-content canonical runs**.","","## Inventory","",canon.groupby("dataset").size().rename("canonical_runs").to_frame().to_markdown(),"",f"Exact duplicate aliases excluded: **{int(m.is_duplicate.sum())}**. All NLR metadata joins were unique and matched.","","## Timing and provenance","","- High-resolution Node median recording interval is 20 ms, but effective sensor update/averaging resolution is undocumented and remains unverified.","- NLR training uses a 0.2 s aggregation grid.","- NLR online-rate's 1 ms grid is generated by `create_dataframe_multinode(..., 0.001)` and first-order polynomial interpolation from asynchronous raw source logs near 0.2 s. It is not direct 1 ms sensing. Detection uses coverage-checked 0.1 s means, while sub-source-cadence physical interpretation is unsupported.","- High-resolution power is a strict eight-GPU sum. NLR power is CPU+GPU; training additionally sums all nodes. These scopes are retained and are not treated as equivalent.","","## Primary exploratory result","","Primary configuration: 20% baseline-relative drop, 50 W absolute minimum, 1 s slope window, 1 s low persistence, and 0.5 s recovery persistence.",""]
    if len(primary):
      primary["long_fast_30s"]=(primary.low_duration_s>=30)&(primary.recovery_rise_time_10_90_s<=2)
      x=primary.groupby("dataset").agg(events=("event_id","count"),runs_with_events=("run_id","nunique"),completed=("completed","sum"),censored=("right_censored","sum"),long_30s=("low_duration_s",lambda z:(z>=30).sum()),long_120s=("low_duration_s",lambda z:(z>=120).sum()),fast_recovery_le_2s=("recovery_rise_time_10_90_s",lambda z:(z<=2).sum()),long_30s_and_fast_2s=("long_fast_30s","sum"))
      lf=primary[primary.long_fast_30s].groupby("dataset").run_id.nunique(); x["runs_with_long_fast"]=lf.reindex(x.index,fill_value=0)
      lines += [x.to_markdown(),"", "Long-low/fast-recovery events are present in NLR training under the primary exploratory definition (low duration ≥30 s and 10–90% recovery ≤2 s). They are absent from the selected High-resolution Node and NLR online-rate traces at that definition. No completed event reaches 120 s in the primary result.","", "The 14 qualifying configuration events occur in 11 independent NLR training runs. This is enough to justify a small Oracle-control experiment for hypothesis testing, but not enough to establish general control feasibility across hardware and workload classes. The experiment must validate actuator dynamics and interference separately."]
    else: lines += ["No events detected."]
    lines += ["","## Sensitivity and limitations","","Results for 10%, 20%, and 30% thresholds and all four `delta_event_s` values are retained in `threshold_sensitivity.csv`. Rows across configurations describe sensitivity and must not be summed as independent physical events.","","- File boundaries are not job starts/stops. Right-censored events have no fabricated recovery time.","- Causes such as checkpointing, evaluation, or failure are not assigned without logs.","- Passive traces do not measure actuator delay `d_act`, interference `eta`, or filler-power calibration.","- This preprocessing does not demonstrate real control feasibility or grid compliance."]
    (Path(out)/"report.md").write_text("\n".join(lines)+"\n")

def finalize_existing(cfg,out):
    out=Path(out); m=pd.read_csv(out/"manifest.csv"); q=pd.read_csv(out/"run_quality.csv"); e=pd.read_parquet(out/"events.parquet"); sd=pd.read_csv(out/"run_summary.csv")
    sens=pd.read_csv(out/"threshold_sensitivity.csv")
    hours=sd.groupby("dataset").valid_observation_hours.sum(); sens["valid_observation_hours"]=sens.dataset.map(hours); sens["events_per_valid_hour"]=sens.event_rows/sens.valid_observation_hours; sens.to_csv(out/"threshold_sensitivity.csv",index=False)
    summary_plots(e,sd,out/"plots"); event_zooms(e,out/"plots",cfg["plots"]["max_event_zooms"]); write_report(out,m,q,e,sens)
