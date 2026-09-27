"""Grouped-inner, equal-budget Stage B selection built on corrected_capacity interfaces."""
import hashlib,json,os
from pathlib import Path
import joblib,numpy as np,pandas as pd
from corrected_capacity import choose_per_run,sample_weights,fit_quantiles,pred,run_balanced
from expanded import X

def candidates(cfg):
    """Exactly 24 deterministic templates; same list is evaluated for both conditions."""
    rows=[]
    for h in cfg['histories_ms']:
      for budget,leaves,lr,rounds,child,l2 in [
        (250,7,.05,50,20,0.),(1000,15,.07,100,30,0.),
        (5000,31,.05,100,40,.1),('all',15,.03,300,60,.5),
        (1000,7,.10,50,20,.1),(5000,31,.07,300,40,.5)]:
       rows.append(dict(history_ms=h,budget_per_run=budget,num_leaves=leaves,learning_rate=lr,n_estimators=rounds,min_child_samples=child,reg_alpha=0.,reg_lambda=l2,n_jobs=cfg['n_jobs']))
    assert len(rows)==24
    return rows

def atomic_csv(path, frame):
    tmp=path.with_suffix(path.suffix+'.tmp');frame.to_csv(tmp,index=False);os.replace(tmp,path)

def run_stage_b(out,cfg,limit_candidates=None):
    out=Path(out); q=np.array(cfg['quantiles']); folds=pd.read_csv(out/'splits/folds.csv'); inv=pd.read_csv(out/'input_manifest.csv'); reused=set(inv.loc[inv.reserved_confirmation,'run_id'])
    grid=pd.read_csv(out/'grid_manifest.csv').query('included').copy(); allrows=[]; selected=[]
    plan=candidates(cfg); pd.DataFrame(plan).to_csv(out/'stage_b_candidate_plan.csv',index=False)
    if limit_candidates is not None: plan=plan[:limit_candidates]
    for cell in grid.itertuples():
      rr,hh=int(cell.resolution_ms),int(cell.horizon_ms); meta=pd.read_parquet(out/'features'/f'meta_{rr}_{hh}.parquet'); fs=np.load(out/'features'/f'power_{rr}_{hh}.npz'); y=meta.y.to_numpy()
      for outer in range(cfg['outer_folds']):
        test=set(folds.query('outer_fold==@outer+1 and role=="outer_test"').run_id)
        for cond,groups in [('history_only',[]),('history_all_meta',['gpu','workload','batch','ai_model'])]:
          for cid,pa in enumerate(plan):
            records=[]
            h=str(pa['history_ms'])
            if h not in fs.files: continue
            vals=[]
            for inner in range(cfg['inner_folds']):
              val=set(folds.query('outer_fold==@outer+1 and inner_fold==@inner+1 and role=="inner_val"').run_id)
              tr=np.flatnonzero(~meta.run_id.isin(test|val|reused)); va=np.flatnonzero(meta.run_id.isin(val))
              ti=choose_per_run(meta,tr,pa['budget_per_run']); w=sample_weights(meta,ti); x,names,vocab=X(fs[h],meta,ti,groups)
              mods=fit_quantiles(x,y,ti,w,q,pa,17); pp=pred(mods,x,va); score=run_balanced(meta,va,y,pp,q)
              vals.append(score)
              records.append(dict(outer_fold=outer+1,inner_fold=inner+1,resolution_ms=rr,horizon_ms=hh,condition=cond,candidate_id=cid,validation_crps_w=score,n_train=len(ti),n_val=len(va),parameters=json.dumps(pa,sort_keys=True)))
            mean=float(np.mean(vals)); allrows.extend(records)
            selected.append(dict(outer_fold=outer+1,resolution_ms=rr,horizon_ms=hh,condition=cond,candidate_id=cid,mean_inner_run_balanced_crps_w=mean,parameters=json.dumps(pa,sort_keys=True)))
      atomic_csv(out/'checkpoints'/f'stage_b_r{rr}_h{hh}.csv',pd.DataFrame(allrows))
    score=pd.DataFrame(allrows); sel=pd.DataFrame(selected)
    # Deterministic tie-break: score, then candidate ID.
    winners=sel.sort_values(['outer_fold','resolution_ms','horizon_ms','condition','mean_inner_run_balanced_crps_w','candidate_id']).groupby(['outer_fold','resolution_ms','horizon_ms','condition'],as_index=False).first()
    atomic_csv(out/'stage_b_candidate_scores.csv',score);atomic_csv(out/'stage_b_selection.csv',winners)
    # Refit selected artifacts on outer-train development runs; models/encoders are condition-specific.
    artifacts=[]
    for wrow in winners.itertuples():
      rr,hh=int(wrow.resolution_ms),int(wrow.horizon_ms);meta=pd.read_parquet(out/'features'/f'meta_{rr}_{hh}.parquet');fs=np.load(out/'features'/f'power_{rr}_{hh}.npz');test=set(folds.query('outer_fold==@wrow.outer_fold and role=="outer_test"').run_id);tr=np.flatnonzero(~meta.run_id.isin(test|reused));pa=json.loads(wrow.parameters);groups=[] if wrow.condition=='history_only' else ['gpu','workload','batch','ai_model'];ti=choose_per_run(meta,tr,pa['budget_per_run']);weight=sample_weights(meta,ti);x,names,vocab=X(fs[str(pa['history_ms'])],meta,ti,groups);mods=fit_quantiles(x,meta.y.to_numpy(),ti,weight,q,pa,17)
      stem=f'stage_b_r{rr}_h{hh}_o{wrow.outer_fold}_{wrow.condition}';joblib.dump(mods,out/'models'/(stem+'.joblib'),compress=3);(out/'features'/(stem+'.json')).write_text(json.dumps({'parameters':pa,'metadata_columns':names,'vocabulary':vocab,'train_indices_file':stem+'.npz'},indent=2));np.savez_compressed(out/'splits'/(stem+'.npz'),train=ti,weight=weight);artifacts.append({'outer_fold':wrow.outer_fold,'resolution_ms':rr,'horizon_ms':hh,'condition':wrow.condition,'model':str(out/'models'/(stem+'.joblib')),'n_train':len(ti)})
    atomic_csv(out/'stage_b_artifacts.csv',pd.DataFrame(artifacts))
    return winners
