from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from .features import feature_columns
from .evaluation import pinball, crps_grid
from .splits import sample_training

FULL_Q = [.01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99]


def candidates(n=24):
    base = dict(n_estimators=100,num_leaves=15,learning_rate=.07,min_child_samples=30,reg_lambda=0,history_s=1)
    import itertools
    grid = list(itertools.product([100,300,600],[7,15,31],[.03,.07],[30,100,300],[0,1,10],[1,2,5,10,30]))
    rng = np.random.default_rng(2026)
    picks = [base]
    for i in rng.permutation(len(grid)):
        x = dict(zip(base,grid[i]))
        if x not in picks:
            picks.append(x)
        if len(picks) >= n: break
    return picks


def encode(train: pd.DataFrame, test: pd.DataFrame, model: str, history_s: int):
    cols = feature_columns(model, list(train.columns))
    limit = int(history_s / .2)
    cols = [c for c in cols if not ((c.startswith(("lag_","mean_","std_"))) and int(c.split("_")[-1]) > limit)]
    xtr, xte = train[cols].copy(), test[cols].copy()
    for c in cols:
        if xtr[c].dtype == object:
            names = {v:i for i,v in enumerate(sorted(xtr[c].dropna().unique()))}
            xtr[c]=xtr[c].map(names).fillna(-1).astype(float)
            xte[c]=xte[c].map(names).fillna(-1).astype(float)
        else:
            xtr[c]=pd.to_numeric(xtr[c],errors="coerce")
            xte[c]=pd.to_numeric(xte[c],errors="coerce")
    return xtr, xte


def fit_quantile(x, y, weight, q, cfg, seed):
    model = lgb.LGBMRegressor(objective="quantile", alpha=q,
        n_estimators=cfg["n_estimators"], num_leaves=cfg["num_leaves"],
        learning_rate=cfg["learning_rate"], min_child_samples=cfg["min_child_samples"],
        reg_lambda=cfg["reg_lambda"], reg_alpha=0, max_depth=-1,
        colsample_bytree=1, subsample=1, subsample_freq=0,
        n_jobs=1, random_state=seed, verbosity=-1)
    model.fit(x,y,sample_weight=weight)
    return model


def tune(train: pd.DataFrame, h: int, model: str, qs: list[float], options: list[dict], inner_folds=3):
    ycol=f"y_{h}"
    g=train.run_id.to_numpy()
    nf=min(inner_folds,len(set(g)))
    if nf < 2: return options[0], pd.DataFrame()
    cv=list(GroupKFold(n_splits=nf).split(train,groups=g))
    screen=[]
    for ci,c in enumerate(options):
        losses=[]
        for ti,vi in cv:
            tr, va = train.iloc[ti], train.iloc[vi]
            x, xv=encode(tr,va,model,c["history_s"])
            for q in [.1,.5,.9]:
                m=fit_quantile(x,tr[ycol],tr.sample_weight,q,c,2026)
                losses.append(float(np.mean(pinball(va[ycol],m.predict(xv),q))))
        screen.append({"candidate":ci,"screen_weighted_pinball":float(np.mean(losses))})
    ranked=sorted(screen,key=lambda x:x["screen_weighted_pinball"])
    full=[]
    for item in ranked[:min(4,len(ranked))]:
        ci=item["candidate"]; c=options[ci]; scores=[]
        for ti,vi in cv:
            tr,va=train.iloc[ti],train.iloc[vi]
            x,xv=encode(tr,va,model,c["history_s"])
            pred=np.column_stack([fit_quantile(x,tr[ycol],tr.sample_weight,q,c,2026).predict(xv) for q in qs])
            scores.append(float(np.mean(crps_grid(va[ycol],np.sort(pred,axis=1),qs))))
        full.append((float(np.mean(scores)),ci))
    chosen=min(full)[1]
    for row in screen:
        row["full15_crps"] = next((v for v,i in full if i==row["candidate"]),np.nan)
        row["selected"] = row["candidate"] == chosen
    return options[chosen],pd.DataFrame(screen)


def _atomic_parquet(frame: pd.DataFrame, path: Path):
    tmp=path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp,index=False)
    tmp.replace(path)


def train_all(data: pd.DataFrame, splits: list[dict], out: Path, config: dict):
    qs=config["quantiles"]; horizons=config["horizons_bins"]
    opts=candidates(config["candidates"])
    (out/"candidate_manifest.json").write_text(json.dumps({"candidates":opts,"screen_quantiles":[.1,.5,.9],
        "screen_metric":"mean pinball (three quantiles)","top_n_full_crps":4,"search_seed":2026},indent=2))
    parts=out/"prediction_parts"; parts.mkdir(exist_ok=True)
    selections=[]; screens=[]
    for split in splits:
        fold=split["fold"]
        for h in horizons:
            sub=data[data[f"y_{h}"].notna() & data.eligible_history].copy()
            tr=sub[sub.run_id.isin(split["train_runs"])]; te=sub[sub.run_id.isin(split["test_runs"])]
            if tr.empty or te.empty: continue
            tr=sample_training(tr,config["max_train_per_run"])
            tr[["run_id","bin","sample_weight"]].assign(fold=fold,horizon=h).to_parquet(out/f"training_sample_f{fold}_h{h}.parquet",index=False)
            tune_path=out/f"tuning_f{fold}_h{h}.csv"
            if tune_path.exists():
                tuned=json.loads((out/f"selected_f{fold}_h{h}.json").read_text())
            else:
                tuned,screen=tune(tr,h,"P2",qs,opts,config["inner_folds"])
                screen.assign(fold=fold,horizon=h,model="P2").to_csv(tune_path,index=False)
                (out/f"selected_f{fold}_h{h}.json").write_text(json.dumps(tuned))
            selections.append({"fold":fold,"horizon":h,"model":"P2/P3/P4 controlled","selection":"inner P2 CRPS",**tuned})
            for seed in config["seeds"]:
                for model in ["P0","P1","P2","P3","P4"]:
                    target=parts/f"f{fold}_h{h}_s{seed}_{model}.parquet"
                    if target.exists(): continue
                    tic=time.perf_counter()
                    if model=="P0":
                        residual=tr[f"y_{h}"].to_numpy()-tr.current_w.to_numpy()
                        pred=np.column_stack([te.current_w.to_numpy()+np.quantile(residual,q) for q in qs])
                    else:
                        x,xv=encode(tr,te,model,tuned["history_s"])
                        pred=np.column_stack([fit_quantile(x,tr[f"y_{h}"],tr.sample_weight,q,tuned,seed).predict(xv) for q in qs])
                    before=np.mean(np.any(np.diff(pred,axis=1)<0,axis=1))
                    pred=np.sort(pred,axis=1)
                    frame=te[["dataset","group_id","run_id","segment","bin","issuance_s","current_w",f"y_{h}"]].copy()
                    frame=frame.rename(columns={f"y_{h}":"actual_w"})
                    frame["experiment_id"]=out.name;frame["fold"]=fold;frame["seed"]=seed;frame["model"]=model
                    frame["target"]="endpoint_mean_power";frame["horizon"]=h*.2
                    frame["target_start_s"]=frame.issuance_s+(h-1)*.2
                    frame["target_time_s"]=frame.issuance_s+h*.2
                    frame["availability_s"]=frame.issuance_s
                    frame["unit"]="W";frame["measurement_boundary"]="completed 200ms right edge"
                    frame["crossing_pre_correction_rate"]=before
                    frame["fit_and_prediction_wall_s"]=time.perf_counter()-tic
                    for j,q in enumerate(qs):frame[f"q{q:g}"]=pred[:,j]
                    _atomic_parquet(frame,target)
                    (out/"heartbeat_utc.txt").write_text(pd.Timestamp.now(tz="UTC").isoformat())
                    print(f"DONE fold={fold} horizon={h} seed={seed} model={model} rows={len(frame)}",flush=True)
    pd.DataFrame(selections).drop_duplicates().to_csv(out/"selected_hyperparameters.csv",index=False)
    return parts
