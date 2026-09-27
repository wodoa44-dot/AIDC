from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def summary_plots(events,summaries,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if events.empty:return
    done=events[events.completed]
    fig,ax=plt.subplots(); ax.hist(done.low_duration_s.dropna(),bins=30); ax.set(xlabel="Low duration (s)",ylabel="Events"); fig.tight_layout(); fig.savefig(out/"low_duration.png"); plt.close(fig)
    fig,ax=plt.subplots(); ax.scatter(done.low_duration_s,done.recovery_rise_time_10_90_s,s=9,alpha=.5); ax.set(xlabel="Low duration (s)",ylabel="10–90% recovery time (s)"); fig.tight_layout(); fig.savefig(out/"duration_vs_recovery.png"); plt.close(fig)
    sens=events.groupby(["dataset","relative_threshold","delta_event_s"]).size().unstack(fill_value=0)
    fig,ax=plt.subplots(figsize=(8,4)); sens.plot.bar(ax=ax); ax.set_ylabel("Detected intervals"); fig.tight_layout(); fig.savefig(out/"threshold_sensitivity.png"); plt.close(fig)
    primary=events[(events.relative_threshold==.2)&(events.delta_event_s==1.0)]
    hours=summaries.groupby("dataset").valid_observation_hours.sum()
    counts=primary.groupby("dataset").size().reindex(hours.index,fill_value=0)
    fig,ax=plt.subplots(figsize=(7,4)); (counts/hours).plot.bar(ax=ax); ax.set(ylabel="Events per valid observation hour",xlabel="Dataset"); fig.tight_layout(); fig.savefig(out/"event_rate.png"); plt.close(fig)

def event_zooms(events,out,max_zooms=12):
    out=Path(out); zdir=out/"event_zooms"; zdir.mkdir(parents=True,exist_ok=True)
    for stale in zdir.glob("*.png"): stale.unlink()
    primary=events[(events.relative_threshold==.2)&(events.delta_event_s==1.0)].copy()
    primary["kind"]="ordinary"
    primary.loc[primary.right_censored,"kind"]="censored"
    primary.loc[(primary.low_duration_s>=30)&(primary.recovery_rise_time_10_90_s<=2),"kind"]="long_fast"
    primary.loc[(primary.low_duration_s>=30)&~(primary.recovery_rise_time_10_90_s<=2),"kind"]="long_slow"
    chosen=pd.concat([g.head(2) for _,g in primary.groupby("kind")])
    weak=events[(events.relative_threshold==.1)&(events.delta_event_s==1.0)].copy()
    if len(weak):
      primary_keys=set(zip(primary.run_id,(primary.low_entry_s*10).round().astype(int)))
      weak=weak[[ (r,int(round(t*10))) not in primary_keys for r,t in zip(weak.run_id,weak.low_entry_s) ]].head(2); weak["kind"]="weak_10pct_only"; chosen=pd.concat([chosen,weak])
    chosen=chosen.head(max_zooms)
    for _,e in chosen.iterrows():
      path=out.parent/"prepared"/f"{e.run_id}.parquet"
      if not path.exists(): continue
      d=pd.read_parquet(path); end=e.recovery_confirmation_s if pd.notna(e.recovery_confirmation_s) else min(d.time_s.max(),e.low_entry_s+120)
      a=max(d.time_s.min(),e.drop_onset_s-15); b=min(d.time_s.max(),end+15); x=d[(d.time_s>=a)&(d.time_s<=b)]
      fig,ax=plt.subplots(figsize=(9,3)); ax.plot(x.time_s,x.power_w,lw=.8); ax.axvline(e.low_entry_s,color="tab:red",label="low entry")
      if pd.notna(e.recovery_onset_s): ax.axvline(e.recovery_onset_s,color="tab:green",label="recovery onset")
      ax.axhline(e.baseline_power_w,color="black",ls="--",lw=.7,label="baseline"); ax.set(title=f"{e.kind}: {e.dataset}",xlabel="Elapsed time (s)",ylabel="Power (W)"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(zdir/f"{e.kind}_{e.run_id}_{e.segment_id}_{e.event_id.split('-e')[-1]}.png"); plt.close(fig)
    prepared=list((out.parent/"prepared").glob("*.parquet"))
    detected=set(primary.run_id)
    missing=next((x for x in prepared if x.stem not in detected),None)
    if missing is not None:
      d=pd.read_parquet(missing); step=max(1,len(d)//5000); fig,ax=plt.subplots(figsize=(9,3)); ax.plot(d.time_s.iloc[::step],d.power_w.iloc[::step],lw=.7); ax.set(title="failed/no primary detection",xlabel="Elapsed time (s)",ylabel="Power (W)"); fig.tight_layout(); fig.savefig(zdir/f"failed_no_primary_detection_{missing.stem}.png"); plt.close(fig)

def run_plot(d,ev,path,max_points=5000):
    step=max(1,len(d)//max_points); fig,ax=plt.subplots(figsize=(11,3)); ax.plot(d.time_s.iloc[::step],d.power_w.iloc[::step],lw=.6)
    for _,e in ev.iterrows(): ax.axvspan(e.low_entry_s,e.recovery_onset_s if pd.notna(e.recovery_onset_s) else d.time_s.iloc[-1],alpha=.15,color="tab:red")
    ax.set(xlabel="Elapsed time (s)",ylabel="Power (W)"); fig.tight_layout(); fig.savefig(path); plt.close(fig)
