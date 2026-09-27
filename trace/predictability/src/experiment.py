import json,time,joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from .core import rearrange,crps,skill


def deterministic_subset(ids,cap):
    return ids[np.linspace(0,len(ids)-1,min(cap,len(ids)),dtype=int)]


def fit(X,y,q,cfg):
    models=[]; start=time.perf_counter()
    for alpha in q:
        model=LGBMRegressor(objective='quantile',alpha=float(alpha),n_estimators=cfg['n_estimators'],
          num_leaves=cfg['num_leaves'],learning_rate=cfg['learning_rate'],min_child_samples=30,
          n_jobs=cfg['n_jobs'],random_state=cfg['seed'],verbosity=-1)
        model.fit(X,y);models.append(model)
    return models,time.perf_counter()-start


def predict(models,X):
    start=time.perf_counter(); pred=np.column_stack([m.predict(X) for m in models])
    pred,cross=rearrange(pred)
    return pred,cross,time.perf_counter()-start


def summarize(g,q):
    scores=g.crps.to_numpy();y=g.y.to_numpy();pred=g[[f'q_{a:g}' for a in q]].to_numpy()
    median=pred[:,list(q).index(.5)]
    res=dict(n=len(g),mean_crps_w=float(scores.mean()),mae_median_w=float(np.abs(y-median).mean()))
    for level,lo,hi in [(80,.1,.9),(90,.05,.95)]:
        l,u=pred[:,list(q).index(lo)],pred[:,list(q).index(hi)]
        res[f'coverage_{level}']=float(np.mean((y>=l)&(y<=u)))
        res[f'width_{level}_w']=float(np.mean(u-l))
    return res


def bootstrap(meta,difference,cfg,seed):
    """Stratified-by-run nonoverlapping temporal-block bootstrap of paired sums.

    Fixed known runs; resample contiguous blocks within each run. No block bridges
    a gap/segment. Truncated blocks retained with their actual counts.
    """
    b=meta[['run_id','segment_id','bootstrap_block']].copy();b['difference']=difference
    blocks=b.groupby(['run_id','segment_id','bootstrap_block']).difference.agg(['sum','count']).reset_index()
    groups=[x[['sum','count']].to_numpy() for _,x in blocks.groupby('run_id')]
    rng=np.random.default_rng(seed);means=[]
    for _ in range(cfg['bootstrap_replicates']):
        draw=[g[rng.integers(0,len(g),len(g))].sum(axis=0) for g in groups]
        total=np.sum(draw,axis=0);means.append(total[0]/total[1])
    low,high=np.percentile(means,[2.5,97.5])
    return dict(difference_w=float(np.mean(difference)),ci_low_w=low,ci_high_w=high,n_blocks=len(blocks),
      n_runs=len(groups),block_s=cfg['bootstrap_block_s'],replicates=cfg['bootstrap_replicates'],
      status='exploratory' if len(blocks)>=10 else 'insufficient blocks')


def run_cells(cells,cfg,out,log):
    q=np.array(cfg['quantiles']); metrics=[];strata=[];blocks=[];intervals=[];validations=[];timings=[];selection=[]
    (out/'predictions').mkdir(exist_ok=True);(out/'models').mkdir(exist_ok=True)
    for cellno,(key,(meta,features)) in enumerate(sorted(cells.items())):
        cohort,r,h=key;common=dict(cohort=cohort,dataset='highres_node',power_scope='8 GPU sum; CPU excluded',resolution_ms=r,horizon_ms=h)
        name=f'{cohort}_{r}ms_{h}ms';log(f'[{cellno+1}/{len(cells)}] {name}')
        tr=np.flatnonzero(meta.split==0);va=np.flatnonzero(meta.split==1);te=np.flatnonzero(meta.split==2)
        if min(len(tr),len(va),len(te))<50:raise ValueError(f'Insufficient samples {key}')
        train=deterministic_subset(tr,cfg['max_train_samples']);val=deterministic_subset(va,cfg['max_validation_samples'])
        y=meta.y.to_numpy(); current=meta[['current']].to_numpy(dtype=np.float32)
        empirical=np.quantile(y[tr],q) # entire train partition; not validation/test
        predictions={};models_by_condition={};scores={};local_timing=[]
        unconditional=np.tile(empirical,(len(val),1));uncond_score=float(crps(y[val],unconditional,q).mean())
        validations.append({**common,'condition':'unconditional','history_ms':0,'validation_crps_w':uncond_score})
        trained,fit_s=fit(current[train],y[train],q,cfg)
        vp,cross,inf=predict(trained,current[val]);models_by_condition['current_only']=(trained,current)
        validations.append({**common,'condition':'current_only','history_ms':0,'validation_crps_w':float(crps(y[val],vp,q).mean()),'crossing_fraction':cross})
        local_timing.append(dict(condition='current_only',history_ms=0,fit_s=fit_s))
        best=None
        for duration,X in sorted(features.items()):
            trained,fit_s=fit(X[train],y[train],q,cfg);vp,cross,inf=predict(trained,X[val])
            score=float(crps(y[val],vp,q).mean())
            validations.append({**common,'condition':'history','history_ms':duration,'validation_crps_w':score,'crossing_fraction':cross})
            local_timing.append(dict(condition='history',history_ms=duration,fit_s=fit_s))
            if best is None or score<best[0]:best=(score,duration,trained,X)
        models_by_condition['history']=(best[2],best[3])
        threshold=float(np.quantile(np.abs(y[tr]-meta.current.to_numpy()[tr]),cfg['transition_quantile']))
        # Freeze choices on disk before any test forecasts or scoring.
        selected={**common,'history_ms':best[1],'validation_crps_w':best[0], 'transition_threshold_w':threshold,
          'n_train_available':len(tr),'n_train_fit':len(train),'n_validation_available':len(va),'n_validation_select':len(val),'n_test':len(te)}
        selection.append(selected)
        (out/'models'/f'{name}_selection.json').write_text(json.dumps(selected,indent=2))
        np.savez_compressed(out/'models'/f'{name}_sample_indices.npz',train=train,validation=val,test=te)
        joblib.dump(dict(models={k:v[0] for k,v in models_by_condition.items()},empirical_quantiles=empirical,
                         selected_history_ms=best[1],quantiles=q),out/'models'/f'{name}.joblib',compress=3)
        tm=meta.iloc[te].copy().reset_index(drop=True); delta=tm.y-tm.current
        tm['stratum']=np.where(delta>threshold,'rising',np.where(delta< -threshold,'falling','stable'))
        tm['evaluation_cadence_ms']=max(r,cfg['evaluation_cadence_ms'])
        tables=[]
        for condition in ['unconditional','current_only','history']:
            if condition=='unconditional':
                start=time.perf_counter();p=np.tile(empirical,(len(te),1));elapsed=time.perf_counter()-start;cross=0.
            else:p,cross,elapsed=predict(models_by_condition[condition][0],models_by_condition[condition][1][te])
            timings.append({**common,'condition':condition,'test_batch_n':len(te),'inference_batch_s':elapsed,
              'inference_us_per_sample':elapsed/len(te)*1e6,'test_crossing_fraction':cross,'threads':cfg['n_jobs']})
            frame=tm.copy()
            for k,v in common.items():frame[k]=v
            frame['condition']=condition;frame['crps']=crps(frame.y,p,q)
            for j,a in enumerate(q):frame[f'q_{a:g}']=p[:,j]
            scores[condition]=frame.crps.to_numpy();tables.append(frame)
            metrics.append({**common,'condition':condition,**summarize(frame,q)})
            for st,g in frame.groupby('stratum'):
                strata.append({**common,'condition':condition,'stratum':st,**summarize(g,q),
                  'status':'insufficient evidence' if len(g)<100 else 'exploratory; dependent rows'})
            for (run,block),g in frame.groupby(['run_id','test_block']):
                blocks.append({**common,'condition':condition,'run_id':run,'test_block':block,**summarize(g,q)})
        pd.concat(tables).to_parquet(out/'predictions'/f'{name}.parquet',index=False,compression='zstd')
        for ref in ['unconditional','current_only']:
            intervals.append({**common,'comparison':f'history-minus-{ref}',**bootstrap(tm,scores['history']-scores[ref],cfg,cfg['seed']+cellno)})
        intervals.append({**common,'comparison':'current_only-minus-unconditional',**bootstrap(tm,scores['current_only']-scores['unconditional'],cfg,cfg['seed']+cellno)})
        for item in local_timing:timings.append({**common,**item})
        log(f'  CRPS W: unconditional={scores["unconditional"].mean():.3f}, current_only={scores["current_only"].mean():.3f}, history={scores["history"].mean():.3f}; history={best[1]} ms')
        pd.DataFrame(selection).to_csv(out/'selected_validation_configs.csv',index=False)
        pd.DataFrame(validations).to_csv(out/'validation_metrics.csv',index=False)
    md=pd.DataFrame(metrics)
    for keys,g in md.groupby(['cohort','resolution_ms','horizon_ms']):
        ref=g.set_index('condition').mean_crps_w
        for idx in g.index:
            md.loc[idx,'crpss_vs_unconditional']=skill(md.loc[idx,'mean_crps_w'],ref['unconditional'])
            md.loc[idx,'crpss_vs_current_only']=skill(md.loc[idx,'mean_crps_w'],ref['current_only'])
    md.to_csv(out/'metrics.csv',index=False)
    for filename,rows in [('transition_metrics',strata),('test_block_metrics',blocks),('confidence_intervals',intervals),('timing',timings)]:
        pd.DataFrame(rows).to_csv(out/f'{filename}.csv',index=False)
    return md
