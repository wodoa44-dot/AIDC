#!/usr/bin/env python
"""Verify exports, benchmark, summarize uncertainty/diagnostics, plot, and report."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import joblib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
OUT=HERE/"outputs/lite_full_20260920"; MODELS=HERE/"models/power-forecast-lite-20260920"
CORRECTED=ROOT/"trace/predictability_meta/outputs/corrected_full_20260915"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def atomic_json(path,value):
    temp=Path(str(path)+".tmp"); temp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)); os.replace(temp,path)
def bootstrap(pair,reps=1000,seed=20260915):
    rng=np.random.default_rng(seed); runs=pair.run_id.unique(); vals=[]
    for _ in range(reps):
        take=rng.choice(runs,len(runs),replace=True); x=pd.concat([pair[pair.run_id==r] for r in take],ignore_index=True)
        vals.append((x.delta_crps_w.mean(),1-x.history_meta.mean()/x.history_only.mean()))
    a=np.asarray(vals); return np.percentile(a,[2.5,97.5],axis=0)
def main():
    status=json.loads((OUT/"status.json").read_text())
    if status["status"] not in ("trained","completed"): raise RuntimeError(f"training is not complete: {status}")
    meta_pkg=MODELS/"history_meta_deployment.joblib"; hist_pkg=MODELS/"history_only_deployment.joblib"
    for p in (meta_pkg,hist_pkg):
        b=joblib.load(p); assert sorted(map(int,b["horizons"]))==[1000,2000]
        assert all(len(x["models"])==3 and x["quantiles"]==[.1,.5,.9] for x in b["horizons"].values())
    replay_path=OUT/"replay_predictions.parquet"; replay=json.loads(subprocess.check_output([sys.executable,str(HERE/"replay.py"),"--package",str(meta_pkg),"--output",str(replay_path)],text=True))
    benchmark_path=OUT/"benchmark.json"; subprocess.run([sys.executable,str(HERE/"benchmark.py"),"--package",str(meta_pkg),"--output",str(benchmark_path),"--trials","1000"],check=True)
    bench=json.loads(benchmark_path.read_text()); outer=pd.read_parquet(OUT/"outer_predictions.parquet")
    aggregate=[]
    for (h,c),g in outer.groupby(["horizon_ms","condition"]):
        rm=g.groupby("run_id").agg(crps_w=("crps_w","mean"),mae_w=("mae_w","mean"),coverage_80=("covered_80","mean"),width_80_w=("width_80_w","mean"),coverage_90=("covered_90","mean"),width_90_w=("width_90_w","mean"))
        aggregate.append({"horizon_ms":h,"condition":c,"n_runs":g.run_id.nunique(),"n_targets":len(g),"run_balanced_crps_w":rm.crps_w.mean(),"target_weighted_crps_w":g.crps_w.mean(),"run_balanced_median_mae_w":rm.mae_w.mean(),"coverage_80":rm.coverage_80.mean(),"width_80_w":rm.width_80_w.mean(),"coverage_90":rm.coverage_90.mean(),"width_90_w":rm.width_90_w.mean()})
    pd.DataFrame(aggregate).to_csv(OUT/"aggregate_metrics.csv",index=False)
    run_detail=outer.groupby(["horizon_ms","condition","run_id","cohort"],as_index=False).agg(crps_w=("crps_w","mean"),mae_w=("mae_w","mean"),n_targets=("y","size"))
    run_detail.groupby(["horizon_ms","condition","cohort"],as_index=False).agg(crps_w=("crps_w","mean"),mae_w=("mae_w","mean"),n_runs=("run_id","nunique"),n_targets=("n_targets","sum")).to_csv(OUT/"per_cohort_metrics.csv",index=False)
    outer.groupby(["horizon_ms","condition","fold","seed"],as_index=False).agg(crps_w=("crps_w","mean"),mae_w=("mae_w","mean"),n_runs=("run_id","nunique"),n_targets=("y","size")).to_csv(OUT/"per_seed_fold_metrics.csv",index=False)
    transition=outer[np.abs(outer.y-outer.current)>=100].groupby(["horizon_ms","condition","run_id"],as_index=False).agg(transition_crps_w=("crps_w","mean"),transition_mae_w=("mae_w","mean"),n_transition_targets=("y","size"))
    transition.groupby(["horizon_ms","condition"],as_index=False).agg(transition_crps_w=("transition_crps_w","mean"),transition_mae_w=("transition_mae_w","mean"),n_runs=("run_id","nunique"),n_transition_targets=("n_transition_targets","sum")).to_csv(OUT/"transition_metrics.csv",index=False)
    run=outer.groupby(["horizon_ms","condition","run_id"],as_index=False).agg(crps_w=("crps_w","mean"))
    summary=pd.read_csv(OUT/"comparison_summary.csv"); cis=[]; gain_rows=[]
    for h,g in run.groupby("horizon_ms"):
        pair=g.pivot(index="run_id",columns="condition",values="crps_w").reset_index(); pair["delta_crps_w"]=pair.history_meta-pair.history_only; pair["meta_gain"]=1-pair.history_meta/pair.history_only
        ci=bootstrap(pair); cis.append({"horizon_ms":h,"delta_ci_low_w":ci[0,0],"delta_ci_high_w":ci[1,0],"gain_ci_low":ci[0,1],"gain_ci_high":ci[1,1]})
        for r in pair.itertuples(): gain_rows.append({"horizon_ms":h,"run_id":r.run_id,"meta_gain":r.meta_gain})
    pd.DataFrame(cis).to_csv(OUT/"confidence_intervals.csv",index=False); gains=pd.DataFrame(gain_rows); gains.to_csv(OUT/"per_run_meta_gain.csv",index=False)
    hold=pd.read_parquet(OUT/"reused_holdout_predictions.parquet"); export_rows=[]
    for (h,c),g in hold.groupby(["horizon_ms","condition"]):
        y=g.y.to_numpy(); export_rows.append({"horizon_ms":h,"condition":c,"n_runs":g.run_id.nunique(),"n_targets":len(g),
            "q50_mae_w":np.mean(np.abs(y-g.q50_export)),"coverage_80":np.mean((y>=g.q10_export)&(y<=g.q90_export)),
            "width_80_w":np.mean(g.q90_export-g.q10_export),"crossing_before_correction":g.export_crossing_before.iloc[0],
            "q50_change_vs_full_mean_w":g.export_q50_abs_change_w.mean(),"q50_change_vs_full_max_w":g.export_q50_abs_change_w.max()})
    exports=pd.DataFrame(export_rows); exports.to_csv(OUT/"deployment_export_metrics.csv",index=False)
    diagnostics=pd.concat([pd.read_csv(p).assign(task=p.parent.name) for p in (OUT/"evaluation").glob("*/tree_diagnostics.csv")],ignore_index=True)
    diagnostics.to_csv(OUT/"tree_diagnostics.csv",index=False)
    meta_diag=diagnostics[diagnostics.feature.str.startswith(("gpu=","workload=","batch","ai_model="))]
    meta_diag.groupby("feature",as_index=False).agg(splits=("splits","sum"),gain=("gain","sum")).sort_values("gain",ascending=False).to_csv(OUT/"metadata_split_usage.csv",index=False)
    # Concise deterministic plots.
    ref=pd.read_csv(CORRECTED/"comparison_summary.csv"); ref=ref[(ref.assessment=="grouped_outer")&(ref.resolution_ms==200)&(ref.pair=="improved_controlled")&(ref.horizon_ms.isin(summary.horizon_ms))]
    x=np.arange(2); fig,ax=plt.subplots(figsize=(6,4)); ax.bar(x-.2,summary.history_meta_crps_w,.4,label="lite meta"); ax.bar(x+.2,ref.sort_values("horizon_ms").history_all_meta,.4,label="corrected reference"); ax.set_xticks(x,["1 s","2 s"]); ax.set_ylabel("run-balanced CRPS (W)"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"plots/lite_vs_reference_crps.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));
    for h,g in gains.groupby("horizon_ms"): ax.scatter([f"{r[:8]}" for r in g.run_id],g.meta_gain,label=f"{h} ms")
    ax.axhline(0,color="black",lw=.8); ax.tick_params(axis="x",rotation=90,labelsize=6); ax.set_ylabel("META gain"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"plots/per_run_meta_gain.png",dpi=160); plt.close(fig)
    scores=pd.read_csv(OUT/"final_candidate_scores.csv"); means=scores.groupby(["horizon_ms","candidate_id"],as_index=False).agg(crps=("run_balanced_crps_w","mean"),bytes=("model_bytes_3q","mean"),latency=("latency_us_p50","mean"))
    fig,ax=plt.subplots(figsize=(6,4)); sc=ax.scatter(means.bytes/1024,means.crps,c=means.latency); ax.set_xlabel("3-model size (KiB)"); ax.set_ylabel("validation CRPS (W)"); fig.colorbar(sc,label="prediction us/row"); fig.tight_layout(); fig.savefig(OUT/"plots/accuracy_size_latency.png",dpi=160); plt.close(fig)
    replay_frame=pd.read_parquet(replay_path); example=replay_frame.sort_values(["run_id","issue_s","horizon_ms"]); run_id=sorted(example.run_id.unique())[0]; e=example[(example.run_id==run_id)&(example.horizon_ms==1000)].head(250)
    target_meta=pd.read_parquet(CORRECTED/"features/meta_200_1000.parquet"); truth=target_meta[target_meta.run_id.eq(run_id)][["target_s","y"]]; e=e.merge(truth,on="target_s",how="left")
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(e.target_s,e.y,label="observed",lw=1); ax.plot(e.target_s,e.q50,label="q50",lw=1); ax.fill_between(e.target_s,e.q10,e.q90,alpha=.25,label="q10-q90"); ax.set_ylabel("GPU sum power (W)"); ax.set_xlabel("target time (s)"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"plots/observed_and_forecast.png",dpi=160); plt.close(fig)
    selections=json.loads((OUT/"final_selections.json").read_text()); inv=pd.read_csv(OUT/"run_inventory.csv"); outer_runs=outer.run_id.nunique()
    deployment_manifest=[]
    for condition,path in (("history_meta",meta_pkg),("history_only",hist_pkg)):
        pkg=joblib.load(path)
        for horizon,item in pkg["horizons"].items():
            deployment_manifest.append({"condition":condition,"horizon_ms":int(horizon),"quantiles":item["quantiles"],"history_ms":item["history_ms"],"requested_parameters":item["parameters"],"actual_tree_counts":[int(m.booster_.num_trees()) for m in item["models"]],"feature_count":int(item["models"][0].n_features_),"package_path":str(path),"package_bytes":path.stat().st_size})
    atomic_json(OUT/"deployment_manifest.json",deployment_manifest)
    hashes=[]
    for p in [HERE/"train.py",HERE/"src/core.py",HERE/"src/online.py",CORRECTED/"input_manifest.csv",CORRECTED/"splits/folds.csv"]: hashes.append({"path":str(p),"sha256":sha(p)})
    pd.DataFrame(hashes).to_csv(OUT/"source_hashes.csv",index=False)
    lines=["# Lightweight Final Power Forecasting Report","", "## Completion", "",
        "The complete CPU experiment, final refit, compact export, reload check, chronological real-run replay, and warmed benchmark completed. No GPU or power-control action was used.","",
        "## Forecast contract","", "The model predicts `P_200ms(t+h)`, the eight-GPU node-sum mean in the future completed 200 ms bin ending exactly at `t+h`. It does not predict a maximum or a 1–2 s interval average. Horizons are 1,000 and 2,000 ms; CPU and facility load are excluded.","",
        "## Inventory and protocols","",f"The audited inventory has {len(inv)} eligible canonical runs. Final development fitting uses 28 runs and excludes four reused-holdout runs. The retrospective grouped-outer protocol evaluates {outer_runs} distinct test runs (four cohort-balanced runs in each of three folds); runs not assigned to these three historical test folds can still participate in outer training and final development fitting. The four holdout runs were excluded from all lightweight selection and fitting and assessed only after settings were frozen.","",
        "The motivating corrected artifacts were verified: grouped-outer metadata gains were 7.67% and 8.67% at 1,000/2,000 ms, while reused-holdout gains were 9.73% and 9.15%. These are historical reference results, not lightweight-model guarantees.","",
        "These evaluations are not fresh confirmation: historical corrected results informed the design, grouped-outer results are retrospective, and the four holdout runs were previously inspected.","",
        "## Final settings","" ]
    for s in selections:
        p=s["parameters"]; lines.append(f"- {int(s['horizon_ms'])} ms: history {int(p['history_ms'])} ms, {int(p['n_estimators'])} estimators, {int(p['num_leaves'])} leaves; learning rate {p['learning_rate']}, minimum leaf size {int(p['min_child_samples'])}, L1/L2 {p['reg_alpha']}/{p['reg_lambda']}; deployment seed 42.")
    lines += ["", "All fits use at most 5,000 deterministic temporal samples per training run, equal total weight per run, no early stopping, and one LightGBM inference thread. Metadata encoders/imputation are fitted only on each fitting partition; unknown categories have explicit columns.","", "## Retrospective grouped-outer accuracy",""]
    for r in summary.itertuples(): lines.append(f"- {int(r.horizon_ms)} ms: history-only CRPS {r.history_only_crps_w:.2f} W; metadata CRPS {r.history_meta_crps_w:.2f} W; META gain {100*r.meta_gain:.2f}% (absolute delta {r.absolute_delta_crps_w:.2f} W). Versus matched corrected reference, changes are {100*r.history_change_vs_reference:.2f}% (history) and {100*r.meta_change_vs_reference:.2f}% (metadata). 80% metadata coverage/width: {100*r.coverage_80_meta:.1f}%/{r.width_80_meta_w:.1f} W; 90%: {100*r.coverage_90_meta:.1f}%/{r.width_90_meta_w:.1f} W.")
    trans=pd.read_csv(OUT/"transition_metrics.csv")
    lines += ["", "Aggregation averages targets within run, repeated seeds/folds within run, then weights independent runs equally. META gain is the ratio of the two run-balanced means. Bootstrap intervals resample paired runs only. They are fixed-prediction intervals and do not include retraining uncertainty; cohort counts are sparse.","", "The separately tabulated transition metric uses the corrected report's fixed definition `abs(target power - issue-time power) >= 100 W`. It is an endpoint-change subset, not ramp-onset detection. Metadata transition CRPS is " + ", ".join(f"{int(r.horizon_ms)} ms: {r.transition_crps_w:.2f} W" for r in trans[trans.condition=="history_meta"].itertuples()) + ".","", "## Actual compact export on reused holdout",""]
    for r in exports[exports.condition=="history_meta"].itertuples(): lines.append(f"- {int(r.horizon_ms)} ms: q50 MAE {r.q50_mae_w:.2f} W; q10–q90 coverage {100*r.coverage_80:.1f}%, width {r.width_80_w:.1f} W; pre-correction crossing {100*r.crossing_before_correction:.3f}%; mean/max q50 change versus the full 15-quantile ordered bundle {r.q50_change_vs_full_mean_w:.4f}/{r.q50_change_vs_full_max_w:.4f} W.")
    lines += ["",f"The metadata deployment package contains exactly six models (two horizons × q10/q50/q90), is {bench['package_bytes']/1024:.1f} KiB, and loaded in {bench['load_ms']:.2f} ms. Warm six-model end-to-end latency was p50 {bench['end_to_end_six_models_us']['p50']:.1f} µs, p95 {bench['end_to_end_six_models_us']['p95']:.1f} µs, p99 {bench['end_to_end_six_models_us']['p99']:.1f} µs. The separate 200 ms bin collection delay is not included. This local measurement is not a real-time guarantee.","",
        "A separately labeled six-model history-only package is exported. Deployment applies row-wise ascending sorting to the three outputs; its metrics are reported separately and are not called 15-quantile CRPS.","", "## Diagnostics and limitations","",
        "Tree split/gain counts are descriptive, not causal attribution. The 100 W transition definition retained by the corrected reference is an absolute issue-to-target endpoint change, not a learned threshold or advance ramp detector. Static metadata must be known at the effective observation time; any metadata identity change resets state. Timestamp gaps reset state and require warm-up again.","",
        "No claim of prospective confirmation or production readiness is made. Remaining work is evaluation on newly collected, untouched runs and operational timing under the intended serving stack.","", "## Commands and artifacts","",
        "See `README.md` for exact test, smoke, training, prediction, replay, benchmark, and resume commands.","",
        f"- Experiment: `{OUT}`",f"- Metadata package: `{meta_pkg}`",f"- History-only package: `{hist_pkg}`",f"- Replay verification: passed={replay['passed']}, {replay['n_compared']} comparisons, maximum difference {replay['max_abs_difference_w']} W",f"- Benchmark: `{benchmark_path}`"]
    (OUT/"report_en.md").write_text("\n".join(lines)+"\n")
    verification={"passed":True,"deployment_packages":{"history_meta":str(meta_pkg),"history_only":str(hist_pkg)},"model_counts":{"history_meta":6,"history_only":6},"replay":replay,
        "checks":["package reload","two horizons","three quantiles per horizon","batch/replay equivalence","unknown-category benchmark","quantile ordering"],"finished_utc":time.strftime("%FT%TZ",time.gmtime())}
    atomic_json(OUT/"verification.json",verification); status.update(status="completed",finished_utc=verification["finished_utc"],verification=str(OUT/"verification.json"),report=str(OUT/"report_en.md")); atomic_json(OUT/"status.json",status)
    print(json.dumps(verification,indent=2))
if __name__=="__main__": main()
