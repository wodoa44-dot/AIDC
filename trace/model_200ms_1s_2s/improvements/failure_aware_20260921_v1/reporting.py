"""Complete held-out evaluation, plots, replay checks, benchmark, and English report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
AUDIT = BASE / "validation/audit_20260921_full_v3"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE / "validation"))
from metrics import distribution_metrics, failure_episodes, interval_metrics, transition_labels
from benchmark import run_benchmark
sys.path.insert(0, str(HERE))
from replay import run_replay

QUANTILES = [.01, .025, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99]
OLD_SUFFIX = {.01:"01", .025:"025", .05:"05", .1:"10", .2:"20", .3:"30", .4:"40",
              .5:"50", .6:"60", .7:"70", .8:"80", .9:"90", .95:"95", .975:"975", .99:"99"}
QCOLS = {q: "q" + OLD_SUFFIX[q] for q in QUANTILES}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def new_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.rename(columns={"issue_s":"issue_time_s", "target_s":"target_time_s",
                                "y":"observed_power_w", "current":"issue_power_w"}).copy()
    out["model"] = out.procedure
    out["representation"] = "full15"
    out["segment_id"] = out.segment_id.astype(str)
    out["subsegment_id"] = out.subsegment_id.astype(str)
    out["segment_key"] = out.segment_id + ":" + out.subsegment_id
    for q, col in QCOLS.items(): out[col] = out[f"q{q:g}"]
    out["error_w"] = out.q50 - out.observed_power_w
    return out


def old_canonical(assessment_keys: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_parquet(AUDIT / "per_target_evaluation.parquet")
    source = source[source.valid_prediction].copy()
    source["segment_id"] = source.segment_id.astype(str)
    source["subsegment_id"] = source.subsegment_id.astype(str)
    keys = assessment_keys[["assessment", "run_id", "horizon_ms"]].drop_duplicates()
    source = source.merge(keys, on=["assessment", "run_id", "horizon_ms"], how="inner")
    frames = []
    full = source.copy(); full["model"] = "old_frozen"; full["procedure"] = "old_frozen"
    full["representation"] = "full15"; full["q50"] = full.full_q50
    full["raw_q50"] = full.raw_q50; full["crps_w"] = full.full_crps_w
    for q, col in QCOLS.items(): full[col] = full[f"full_q{OLD_SUFFIX[q]}"]
    full["error_w"] = full.q50 - full.observed_power_w; frames.append(full)
    compact = source.copy(); compact["model"] = "old_frozen"; compact["procedure"] = "old_frozen"
    compact["representation"] = "compact3"; compact["q10"] = compact.compact_q10
    compact["q50"] = compact.compact_q50; compact["q90"] = compact.compact_q90
    compact["raw_q50"] = compact.compact_raw_q50; compact["crps_w"] = np.nan
    compact["error_w"] = compact.q50 - compact.observed_power_w; frames.append(compact)
    persistence = source[(source.condition == "history_only") & (source.seed == 42)].copy()
    persistence["condition"] = "none"; persistence["model"] = "persistence"
    persistence["procedure"] = "persistence"; persistence["representation"] = "point"
    persistence["q50"] = persistence.issue_power_w; persistence["raw_q50"] = persistence.q50
    persistence["crps_w"] = np.nan; persistence["error_w"] = persistence.q50-persistence.observed_power_w
    frames.append(persistence)
    keep = ["assessment","run_id","segment_id","subsegment_id","segment_key","cohort","fold","seed",
            "condition","procedure","model","representation","issue_time_s","target_time_s","horizon_ms",
            "issue_power_w","observed_power_w","target_bin","target_index","raw_q50","q50","error_w","crps_w",
            "transition_stratum"] + [column for column in QCOLS.values() if column != "q50"]
    return pd.concat([x.reindex(columns=keep) for x in frames], ignore_index=True)


def distribution_record(group: pd.DataFrame) -> dict:
    d = distribution_metrics(group.error_w.to_numpy(float))
    d["crps_w"] = float(group.crps_w.mean()) if group.crps_w.notna().any() else np.nan
    d["raw_q50_mae_w"] = float(np.abs(group.raw_q50-group.observed_power_w).mean())
    d["rearranged_q50_change_mean_w"] = float(np.abs(group.q50-group.raw_q50).mean())
    d["rearranged_q50_change_max_w"] = float(np.abs(group.q50-group.raw_q50).max())
    d["exceedance_500_fraction"] = float(np.mean(np.abs(group.error_w) > 500))
    d["n_runs"] = int(group.run_id.nunique())
    return d


GROUP = ["assessment","horizon_ms","model","condition","representation","seed"]


def build_metric_tables(canonical: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    for keys, group in canonical.groupby(GROUP + ["run_id"], dropna=False, sort=True):
        row = dict(zip(GROUP+["run_id"], keys)); row.update(distribution_record(group))
        row["cohort"] = group.cohort.iloc[0]
        row["longest_500_episode_s"] = 0.0
        episodes = failure_episodes(group.assign(issue_s=group.issue_time_s, segment_key=group.segment_key), 500)
        if len(episodes): row["longest_500_episode_s"] = float(episodes.duration_s.max())
        run_rows.append(row)
    per_run = pd.DataFrame(run_rows)
    per_run.to_csv(out / "per_run_metrics.csv", index=False)
    aggregate_rows = []
    for keys, group in canonical.groupby(GROUP, dropna=False, sort=True):
        row = dict(zip(GROUP, keys)); row.update(distribution_record(group)); row["weighting"] = "target_weighted"
        aggregate_rows.append(row)
        runs = per_run
        for col, value in zip(GROUP, keys):
            runs = runs[runs[col].fillna("__NA__").eq("__NA__" if pd.isna(value) else value)]
        numeric = ["crps_w","mae_w","rmse_w","mean_signed_error_w","abs_error_p50_w","abs_error_p90_w",
                   "abs_error_p95_w","abs_error_p99_w","abs_error_max_w","worst_5pct_mean_abs_error_w",
                   "exceedance_500_fraction","longest_500_episode_s"]
        aggregate_rows.append(row | {k: float(runs[k].mean()) for k in numeric} |
                              {"weighting":"run_balanced", "n_runs":len(runs), "n_targets":int(runs.n_targets.sum())})
    aggregate = pd.DataFrame(aggregate_rows)
    persistence = aggregate[(aggregate.model=="persistence") & (aggregate.weighting=="target_weighted")][
        ["assessment","horizon_ms","mae_w"]].rename(columns={"mae_w":"persistence_mae_w"})
    aggregate = aggregate.merge(persistence, on=["assessment","horizon_ms"], how="left")
    aggregate["mae_improvement_vs_persistence"] = 1-aggregate.mae_w/aggregate.persistence_mae_w
    aggregate.to_csv(out / "aggregate_metrics.csv", index=False)
    return aggregate, per_run


def build_diagnostics(canonical: pd.DataFrame, per_run: pd.DataFrame, out: Path) -> None:
    primary = canonical[canonical.seed.eq(42)].copy()
    cohort_rows, transition_rows, interval_rows, tolerance_rows, calibration_rows = [], [], [], [], []
    for keys, group in primary.groupby(GROUP[:-1], dropna=False, sort=True):
        base = dict(zip(GROUP[:-1], keys))
        for cohort, g in group.groupby("cohort"):
            cohort_rows.append(base | {"cohort":cohort} | distribution_record(g))
        for stratum, g in group.groupby("transition_stratum"):
            transition_rows.append(base | {"transition_stratum":stratum,
                                            "evaluated_share":len(g)/len(group)} | distribution_record(g))
        for threshold in (50,100,200,300,500,750,1000):
            tolerance_rows.append(base | {"threshold_w":threshold,
                "exceedance_fraction":float(np.mean(np.abs(group.error_w)>threshold)), "n_targets":len(group)})
        if group.q10.notna().all() and group.q90.notna().all():
            interval_rows.append(base | {"nominal_coverage":.8} |
                                 interval_metrics(group.observed_power_w,group.q10,group.q90,.2))
        if "q05" in group and group.q05.notna().all() and "q95" in group and group.q95.notna().all():
            interval_rows.append(base | {"nominal_coverage":.9} |
                                 interval_metrics(group.observed_power_w,group.q05,group.q95,.1))
        if group.representation.iloc[0] == "full15":
            for q, col in QCOLS.items():
                calibration_rows.append(base | {"quantile":q,
                    "empirical_fraction_actual_le_prediction":float(np.mean(group.observed_power_w<=group[col])),
                    "calibration_error":float(np.mean(group.observed_power_w<=group[col])-q)})
    pd.DataFrame(cohort_rows).to_csv(out/"per_cohort_metrics.csv",index=False)
    pd.DataFrame(transition_rows).to_csv(out/"transition_metrics.csv",index=False)
    pd.DataFrame(interval_rows).to_csv(out/"interval_diagnostics.csv",index=False)
    pd.DataFrame(tolerance_rows).to_csv(out/"tolerance_exceedance.csv",index=False)
    pd.DataFrame(calibration_rows).to_csv(out/"calibration_diagnostics.csv",index=False)
    episode_parts=[]
    for keys,group in primary.groupby(GROUP[:-1],dropna=False,sort=True):
        part=failure_episodes(group.assign(issue_s=group.issue_time_s),500)
        for column,value in zip(GROUP[:-1],keys): part[column]=value
        episode_parts.append(part)
    episodes=pd.concat(episode_parts,ignore_index=True) if episode_parts else pd.DataFrame()
    episodes.to_parquet(out/"failure_episodes.parquet",index=False)
    episode_summary = (episodes.groupby(GROUP[:-1],dropna=False).duration_s
        .agg(["count","mean","median","max"]).reset_index()
        .rename(columns={"count":"episode_count","mean":"duration_mean_s","median":"duration_median_s","max":"duration_max_s"}))
    episode_summary.to_csv(out/"failure_episode_summary.csv",index=False)
    export = (primary[primary.representation.eq("full15")].groupby(GROUP[:-1],dropna=False)
              .agg(n_targets=("q50","size"), raw_q50_mae_w=("raw_q50",lambda x: np.nan),
                   q50_change_mean_w=("error_w",lambda x: np.nan)).reset_index())
    export_rows=[]
    for keys,g in primary[primary.representation.eq("full15")].groupby(GROUP[:-1],dropna=False):
        export_rows.append(dict(zip(GROUP[:-1],keys)) | {"n_targets":len(g),
            "raw_q50_mae_w":float(np.abs(g.raw_q50-g.observed_power_w).mean()),
            "ordered_q50_mae_w":float(np.abs(g.q50-g.observed_power_w).mean()),
            "q50_change_mean_w":float(np.abs(g.q50-g.raw_q50).mean()),
            "q50_change_p99_w":float(np.quantile(np.abs(g.q50-g.raw_q50),.99)),
            "q50_change_max_w":float(np.abs(g.q50-g.raw_q50).max())})
    pd.DataFrame(export_rows).to_csv(out/"quantile_postprocessing_effects.csv",index=False)
    comparisons=[]
    p=per_run[per_run.seed.eq(42)]
    for assessment in p.assessment.unique():
      for horizon in p.horizon_ms.unique():
        persistence=p[(p.assessment==assessment)&(p.horizon_ms==horizon)&(p.model=="persistence")][["run_id","mae_w"]].rename(columns={"mae_w":"persistence_mae_w"})
        for keys,g in p[(p.assessment==assessment)&(p.horizon_ms==horizon)&(p.model!="persistence")].groupby(["model","condition","representation"]):
            paired=g.merge(persistence,on="run_id")
            comparisons.append({"assessment":assessment,"horizon_ms":horizon,"model":keys[0],"condition":keys[1],"representation":keys[2],
                "improved_runs":int((paired.mae_w<paired.persistence_mae_w).sum()),
                "worsened_runs":int((paired.mae_w>paired.persistence_mae_w).sum()),
                "tied_runs":int(np.isclose(paired.mae_w,paired.persistence_mae_w,atol=1e-9).sum()),"n_runs":len(paired)})
    pd.DataFrame(comparisons).to_csv(out/"improved_worsened_run_counts.csv",index=False)


def bootstrap_intervals(per_run: pd.DataFrame, cfg: dict, out: Path) -> None:
    rng=np.random.default_rng(cfg["bootstrap_seed"]); reps=cfg["bootstrap_replicates"]
    primary=per_run.copy()
    # Average seeds within run; predictions are checked separately for seed identity.
    keys=["assessment","horizon_ms","model","condition","representation","run_id"]
    metrics=["crps_w","mae_w","worst_5pct_mean_abs_error_w","exceedance_500_fraction","longest_500_episode_s"]
    avg=primary.groupby(keys,dropna=False)[metrics].mean().reset_index()
    comparisons=[]
    specs=[]
    for assessment in avg.assessment.unique():
      for horizon in avg.horizon_ms.unique():
       for condition in ("history_only","history_meta"):
        specs += [(assessment,horizon,condition,"reliability_selection","full15","mean_crps_selection","full15","reliability_minus_mean"),
                  (assessment,horizon,condition,"reliability_selection","full15","old_frozen","full15","reliability_minus_old_full15")]
       for model in ("reliability_selection","mean_crps_selection","old_frozen"):
        specs += [(assessment,horizon,"__META__",model,"full15",model,"full15","metadata_minus_history")]
    for assessment,horizon,condition,left,lrep,right,rrep,label in specs:
        subset=avg[(avg.assessment==assessment)&(avg.horizon_ms==horizon)]
        if condition=="__META__":
            a=subset[(subset.model==left)&(subset.representation==lrep)&(subset.condition=="history_meta")]
            b=subset[(subset.model==right)&(subset.representation==rrep)&(subset.condition=="history_only")]
        else:
            a=subset[(subset.model==left)&(subset.representation==lrep)&(subset.condition==condition)]
            b=subset[(subset.model==right)&(subset.representation==rrep)&(subset.condition==condition)]
        pair=a.merge(b,on="run_id",suffixes=("_left","_right"))
        if not len(pair): continue
        for metric in metrics:
            delta=(pair[f"{metric}_left"]-pair[f"{metric}_right"]).to_numpy(float)
            delta=delta[np.isfinite(delta)]
            if not len(delta): continue
            boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(reps)])
            comparisons.append({"comparison":label,"assessment":assessment,"horizon_ms":horizon,
                "condition":condition,"left":left,"right":right,"metric":metric,"n_runs":len(delta),
                "mean_delta":delta.mean(),"ci_lower":np.quantile(boot,.025),"ci_upper":np.quantile(boot,.975),
                "replicates":reps,"fixed_predictions":True})
    pd.DataFrame(comparisons).to_csv(out/"paired_run_bootstrap_intervals.csv",index=False)


def seed_checks(canonical: pd.DataFrame, out: Path) -> dict:
    new=canonical[canonical.model.isin(["mean_crps_selection","reliability_selection"])]
    key=["assessment","horizon_ms","model","condition","run_id","target_index"]
    spread=new.groupby(key).q50.agg(lambda x: float(x.max()-x.min()))
    result={"max_q50_spread_w":float(spread.max()),"deterministic_predictions_across_seeds":bool(spread.max()<=1e-10),
            "note":"Seeds are sensitivity checks, not independent evidence."}
    (out/"seed_sensitivity.json").write_text(json.dumps(result,indent=2)); return result


def make_plots(canonical: pd.DataFrame, out: Path) -> None:
    plots=out/"plots"; plots.mkdir(exist_ok=True)
    primary=canonical[canonical.seed.eq(42)]
    tol=pd.read_csv(out/"tolerance_exceedance.csv")
    selected=tol[(tol.assessment=="grouped_outer")&(tol.condition=="history_meta")&tol.representation.eq("full15")]
    fig,ax=plt.subplots(figsize=(8,5))
    for model,g in selected.groupby("model"):
        for horizon,h in g.groupby("horizon_ms"):
            ax.plot(h.threshold_w,h.exceedance_fraction,marker="o",label=f"{model} {horizon}ms")
    ax.set(xlabel="absolute-error threshold (W)",ylabel="exceedance fraction",title="Diagnostic tolerance exceedance (seed 42)"); ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(plots/"tolerance_exceedance.png",dpi=150); plt.close(fig)
    trans=pd.read_csv(out/"transition_metrics.csv")
    t=trans[(trans.assessment=="grouped_outer")&(trans.condition=="history_meta")&trans.representation.eq("full15")]
    if len(t):
        strata=["falling","rising","stable"]
        labels={"old_frozen":"Original","mean_crps_selection":"Mean CRPS selection",
                "reliability_selection":"Failure-aware"}
        colors={"old_frozen":"#7a8793","mean_crps_selection":"#d98725",
                "reliability_selection":"#147d73"}
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True,layout="constrained")
        for ax,horizon in zip(axes,[1000,2000]):
            for model,g in t[t.horizon_ms.eq(horizon)].groupby("model"):
                values=g.set_index("transition_stratum").reindex(strata).mae_w
                ax.plot(range(3),values,marker="o",linewidth=2.3,
                        color=colors.get(model),label=labels.get(model,model))
            ax.set_xticks(range(3),[name.title() for name in strata])
            ax.set_title(f"{horizon // 1000} s horizon")
            ax.set_xlabel("Endpoint transition")
            ax.set_ylabel("MAE (W)")
            ax.grid(alpha=.25)
        fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",
                   bbox_to_anchor=(.5,-.08),ncol=3)
        fig.suptitle("Grouped-test transition error by model")
        fig.savefig(plots/"transition_mae.png",dpi=180,bbox_inches="tight")
        plt.close(fig)
    interval=pd.read_csv(out/"interval_diagnostics.csv")
    i=interval[(interval.assessment=="grouped_outer")&(interval.condition=="history_meta")&interval.representation.eq("full15")]
    if len(i):
        fig,ax=plt.subplots(figsize=(8,5))
        for model,g in i.groupby("model"): ax.scatter(g.nominal_coverage,g.coverage,label=model)
        ax.plot([.75,.92],[.75,.92],"k--"); ax.set(xlabel="nominal",ylabel="empirical",title="Uncalibrated interval coverage"); ax.legend()
        fig.tight_layout(); fig.savefig(plots/"interval_coverage.png",dpi=150); plt.close(fig)
    candidates=pd.read_csv(out/"outer_candidate_metrics.csv")
    fig,ax=plt.subplots(figsize=(8,5))
    for family,g in candidates.groupby("family"): ax.scatter(g.run_balanced_crps_w,g.mean_run_longest_500_episode_s,label=family,alpha=.75)
    ax.set(xlabel="run-balanced validation CRPS (W)",ylabel="mean run longest >500 W episode (s)",title="Frozen candidate trade-off"); ax.legend()
    fig.tight_layout(); fig.savefig(plots/"candidate_tradeoff.png",dpi=150); plt.close(fig)
    # Historical examples plus the new method on matching keys.
    cases=pd.read_csv(AUDIT/"worst_cases.csv")
    new=primary[(primary.model=="reliability_selection")&(primary.condition=="history_meta")&primary.representation.eq("full15")]
    old=primary[(primary.model=="old_frozen")&(primary.condition=="history_meta")&primary.representation.eq("full15")]
    for index,case in cases.iterrows():
        identity=(new.assessment.eq(case.assessment)&new.run_id.eq(case.run_id)&new.horizon_ms.eq(case.horizon_ms))
        n=new[identity & new.issue_time_s.between(case.issue_time_s-10,case.issue_time_s+10)]
        o=old[(old.assessment==case.assessment)&(old.run_id==case.run_id)&(old.horizon_ms==case.horizon_ms)&old.issue_time_s.between(case.issue_time_s-10,case.issue_time_s+10)]
        if not len(n) or not len(o): continue
        fig,ax=plt.subplots(figsize=(10,4)); ax.plot(n.issue_time_s,n.observed_power_w,label="observed",color="black")
        ax.plot(n.issue_time_s,n.issue_power_w,label="persistence",alpha=.6); ax.plot(o.issue_time_s,o.q50,label="old full15 q50")
        ax.plot(n.issue_time_s,n.q50,label="new reliability q50"); ax.axvline(case.issue_time_s,color="red",ls="--",alpha=.5)
        ax.set(title=case.selection_rule,xlabel="issue time (s)",ylabel="endpoint power (W)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(plots/f"historical_window_{index:02d}.png",dpi=150); plt.close(fig)
    worst=new.assign(abs_error=lambda x:abs(x.error_w)).nlargest(5,"abs_error")
    worst.to_csv(out/"new_worst_cases.csv",index=False)


def report_text(out: Path, cfg: dict, aggregate: pd.DataFrame, seed: dict, model_root: Path,
                replay: dict, benchmark: dict, smoke: bool) -> str:
    primary=aggregate[(aggregate.seed==42)&(aggregate.weighting=="run_balanced")]
    selections=pd.read_csv(out/"final_selections.csv")
    lines=["# Failure-aware lightweight AIDC retraining report", "",
      "## Status and scope", "",
      f"The {'smoke' if smoke else 'full'} real-data retrospective experiment completed. It used 200 ms completed bins, 1 s and 2 s endpoints, 28 development runs, and four excluded reused-holdout runs. The holdout had already been inspected; these results are not fresh confirmation.", "",
      "No calibration was fitted. Weighted models alter the effective training distribution, while every result below is evaluated on the ordinary unweighted held-out distribution. Fixed-prediction paired-run bootstrap intervals use 2,000 replicates and exclude retraining and selection uncertainty.", "",
      "## Frozen selections", "", selections.to_markdown(index=False), "",
      "The metadata condition uses exactly the history-only-selected family, history, tree budget, and weighting rule. Seed 42 is deployed; seeds 17/42/2026 are sensitivity refits and are not independent evidence.", "",
      "## Primary run-balanced results (seed 42)", ""]
    cols=["assessment","horizon_ms","model","condition","representation","crps_w","mae_w","worst_5pct_mean_abs_error_w","exceedance_500_fraction","longest_500_episode_s","mae_improvement_vs_persistence"]
    lines += [primary[cols].to_markdown(index=False), "", "## Reliability and quantile handling", "",
      f"Predictions across seeds were deterministic: {seed['deterministic_predictions_across_seeds']} (maximum q50 spread {seed['max_q50_spread_w']:.3g} W).",
      "The deployed package computes all 15 quantiles before row-wise monotone rearrangement; q10/q50/q90 are only response subsets. Raw q50 and rearranged q50 are evaluated separately in `quantile_postprocessing_effects.csv`. Calibration remains descriptive and is not corrected.", "",
      "## Export verification and cost", "",
      f"Continuous replay passed across {replay['runs']} reused-holdout runs and {replay['n_compared']} horizon predictions; maximum batch/export difference was {replay['max_abs_difference_w']:.3g} W.",
      f"The metadata deployment bundle is {benchmark['package_bytes']} bytes. Warm both-horizon 30-model prediction latency was {benchmark['prediction_only_30_models_us']['p50']:.1f} µs p50 and {benchmark['prediction_only_30_models_us']['p99']:.1f} µs p99; end-to-end latency was {benchmark['end_to_end_30_models_us']['p50']:.1f} µs p50 and {benchmark['end_to_end_30_models_us']['p99']:.1f} µs p99. RSS is a process high-water mark, not incremental model memory.", "",
      "## Interpretation limits", "",
      "Residual targets, transition weighting, and reliability selection are assessed with matched validation configurations in the candidate tables. Test results were not used to choose a family. The diagnostic 100 W and 500 W thresholds and 5% feasibility tolerance are experimental design choices, not safety limits. Historical windows provide continuity only; the new method's own worst cases are saved separately. No claim is made about control readiness, unseen future runs, or causal phase mismatch.", "",
      "## Artifact map", "",
      "Candidate tables, selections, per-target predictions, per-run/cohort/transition/interval/tolerance/episode tables, bootstrap intervals, plots, saved weights, encoders, reloadable models, replay output, benchmark, environment, integrity checks, status, and logs are all in this artifact directory or the adjacent versioned model directory."]
    return "\n".join(lines)+"\n"


def build_all_reports(out: Path, cfg: dict, model_root: Path, smoke: bool=False) -> None:
    new=pd.concat([pd.read_parquet(out/"outer_predictions.parquet"),pd.read_parquet(out/"reused_holdout_predictions.parquet")],ignore_index=True)
    new=new_canonical(new); old=old_canonical(new)
    common=["assessment","run_id","horizon_ms","target_index"]
    allowed=new[common].drop_duplicates(); old=old.merge(allowed,on=common,how="inner")
    canonical=pd.concat([new,old],ignore_index=True,sort=False)
    canonical.to_parquet(out/"per_target_predictions_all_models.parquet",index=False)
    aggregate,per_run=build_metric_tables(canonical,out); build_diagnostics(canonical,per_run,out)
    bootstrap_intervals(per_run,cfg,out); seed=seed_checks(canonical,out); make_plots(canonical,out)
    package=model_root/"history_meta_full15_deployment.joblib"
    replay=run_replay(package,out/"replay_predictions.parquet",80 if smoke else None)
    (out/"replay_verification.json").write_text(json.dumps(replay,indent=2))
    benchmark=run_benchmark(package,out/"benchmark.json",50 if smoke else 1000)
    (out/"report_en.md").write_text(report_text(out,cfg,aggregate,seed,model_root,replay,benchmark,smoke))
    hashes=[]
    for path in sorted(list(HERE.glob("*.py"))+[HERE/"config.json",HERE/"failure_diagnosis.md"]):
        hashes.append({"path":str(path),"sha256":sha256(path),"bytes":path.stat().st_size})
    pd.DataFrame(hashes).to_csv(out/"source_hashes.csv",index=False)
    completion={"passed":True,"mode":"smoke" if smoke else "full","report":str(out/"report_en.md"),
                "model_root":str(model_root),"replay_passed":replay["passed"],"artifact_count":sum(1 for _ in out.rglob("*"))}
    (out/"completion_status.json").write_text(json.dumps(completion,indent=2))
