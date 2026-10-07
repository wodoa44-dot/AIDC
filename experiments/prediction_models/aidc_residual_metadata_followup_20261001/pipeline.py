"""Fixed 27-cell follow-up; prediction only, development only, no tuning."""
import argparse,fcntl,json,os,signal,sys,time,traceback,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml
from data import *
ROOT=Path(__file__).resolve().parent
PARAMS=dict(objective='quantile',alpha=.5,n_estimators=100,num_leaves=15,learning_rate=.07,min_child_samples=30,reg_alpha=0,reg_lambda=0,max_depth=-1,colsample_bytree=1.,subsample=1.,subsample_freq=0,n_jobs=4,random_state=17,verbosity=-1)
KEEP=KEY+['collection_day','issuance_s','current_w','actual_w','time_stratum']+columns(5,True)
KEEP=list(dict.fromkeys(KEEP))

def rank(row,stratum):
    return hashlib.sha256(f'17:{row.run_id}:{stratum}:{row.segment}:{row.bin}'.encode()).hexdigest()

def quotas(sizes,target,initial,lower=None):
    q=np.minimum(sizes,initial) if lower is None else np.maximum(lower,np.minimum(sizes,initial))
    assert q.sum()<=target
    while q.sum()<target:
        for s in range(10):
            if q[s]<sizes[s]:q[s]+=1
            if q.sum()==target:break
    return q

def prepare(pilot,out,cfg):
    t=time.perf_counter()
    full=pd.read_parquet(pilot/'continuous_features.parquet')
    tr=pd.read_parquet(pilot/'train_anchors.parquet');ev=pd.read_parquet(pilot/'evaluation_anchors.parquet')
    dev=full[full.collection_day.isin(cfg['development_days'])]
    cand=pd.concat([spaced(g) for _,g in dev[dev.valid_anchor].groupby(['run_id','segment'],sort=True)],ignore_index=True)
    nested=[];summ=[]
    for run,g in cand.groupby('run_id',sort=True):
        g=g.sort_values(['issuance_s','segment','bin']).reset_index(drop=True)
        if len(g)<30:continue
        g['time_stratum']=0
        for s,ix in enumerate(np.array_split(np.arange(len(g)),10)):g.loc[ix,'time_stratum']=s
        old=tr[tr.run_id==run]
        restored=sample(g,300,True,17)
        assert frame_hash(restored[KEY])==frame_hash(old[KEY]),'pilot sampler identity mismatch'
        oldkeys=set(map(tuple,old[KEY].to_numpy()))
        selected=g[g[KEY].apply(tuple,axis=1).isin(oldkeys)]
        assert len(selected)==min(300,len(g))
        np.testing.assert_array_equal(selected.time_stratum,old.time_stratum)
        sizes=g.groupby('time_stratum').size().reindex(range(10),fill_value=0).to_numpy()
        oldsizes=selected.groupby('time_stratum').size().reindex(range(10),fill_value=0).to_numpy()
        for cap in cfg['caps']:
            if cap==300:z=old[KEEP].copy()
            else:
                source=selected if cap==100 else g
                q=quotas(oldsizes if cap==100 else sizes,min(cap,len(g)),10 if cap==100 else 60,None if cap==100 else oldsizes)
                chunks=[]
                for s in range(10):
                    pool=source[source.time_stratum==s].copy()
                    pool['_rank']=[rank(row,s) for row in pool.itertuples()]
                    if cap==100:chosen=pool.sort_values('_rank').head(q[s])
                    else:
                        oldmask=pool[KEY].apply(tuple,axis=1).isin(oldkeys)
                        chosen=pd.concat([pool[oldmask],pool[~oldmask].sort_values('_rank').head(q[s]-int(oldmask.sum()))])
                    chunks.append(chosen)
                z=pd.concat(chunks).sort_values(['issuance_s','segment','bin'])[KEEP]
            assert len(z)==min(cap,len(g)) and not z.duplicated(KEY).any()
            assert set(map(tuple,z[KEY].to_numpy()))<=set(map(tuple,g[KEY].to_numpy()))
            nested.append(z.assign(cap=cap))
            summ.append(dict(run_id=run,collection_day=g.collection_day.iloc[0],cap=cap,candidates=len(g),actual_rows=len(z),anchor_hash=frame_hash(z[KEY])))
        sets=[set(map(tuple,nested[-i][KEY].to_numpy())) for i in [3,2,1]]
        assert sets[0]<=sets[1]<=sets[2]
    nested=pd.concat(nested,ignore_index=True)
    ev=ev[ev.collection_day.isin(cfg['development_days'])][KEEP].copy()
    assert set(ev.collection_day)==set(cfg['development_days'])
    for run,g in ev.groupby('run_id'):assert len(g)<=600
    parquet(out/'nested_train_anchors.parquet',nested);parquet(out/'evaluation_anchors.parquet',ev)
    pd.DataFrame(summ).to_csv(out/'sampling_summary.csv',index=False)
    splits=json.loads((pilot/'split_manifest.json').read_text())['folds'][:3]
    atomic_json(out/'split_manifest.json',{'folds':splits,'development_exploratory':True,'holdout_new_evaluation':False})
    thresholds=[]
    for s in splits:
        z=tr[tr.collection_day.isin(s['train_days'])];q=np.quantile(abs(z.actual_w-z.current_w),[.5,.75,.9])
        old=json.loads((pilot/f"fits/development_{s['fold']}_residual_h5/complete.json").read_text())
        np.testing.assert_allclose(q[[0,2]],[old['threshold_q50_w'],old['threshold_q90_w']])
        thresholds.append(dict(fold=s['fold'],q50=float(q[0]),q75=float(q[1]),q90=float(q[2])))
    atomic_json(out/'thresholds.json',thresholds)
    inventory=pd.read_csv(pilot/'run_inventory.csv');inventory.to_csv(out/'run_inventory.csv',index=False)
    inventory.groupby(['collection_day','hardware','workload'],dropna=False).agg(run_count=('run_id','size')).reset_index().to_csv(out/'date_hardware_workload.csv',index=False)
    atomic_json(out/'sampling_checkpoint.json',{n:digest(out/n) for n in ['nested_train_anchors.parquet','evaluation_anchors.parquet','sampling_summary.csv','split_manifest.json','thresholds.json']})
    return time.perf_counter()-t

def decorate(z,q):
    z=z.copy();z['delta']=z.actual_w-z.current_w;z['delta_hat']=z.prediction_w-z.current_w
    z['abs_error']=abs(z.actual_w-z.prediction_w);z['gain']=abs(z.delta)-z.abs_error
    z['endpoint_region']=np.where(abs(z.delta)<=q['q50'],'normal',np.where(z.delta>q['q50'],'rising','falling'))
    z['large_change']=abs(z.delta)>q['q90']
    z['magnitude_region']=np.select([abs(z.delta)<=q['q50'],abs(z.delta)<=q['q75'],abs(z.delta)<=q['q90']],['0-q50','q50-q75','q75-q90'],default='>q90')
    return z

def prior_diagnosis(pilot,out):
    t=time.perf_counter();pred=pd.read_parquet(pilot/'pilot_predictions.parquet')
    pred=pred[(pred.postprocess=='raw')&(pred.region=='whole')&pred.model.isin(['residual_h5','residual_h5_metadata'])].copy()
    frames=[]
    for (scope,fold,model),g in pred.groupby(['scope','fold','model']):
        r=json.loads((pilot/f'fits/{scope}_{fold}_{model}/complete.json').read_text())
        q={'q50':r['threshold_q50_w'],'q90':r['threshold_q90_w'],'q75':r['threshold_q90_w']}
        z=decorate(g,q);z['analysis_label']='개발 OOF' if scope=='development' else '기존 보류 자료 재분석';frames.append(z)
    old=pd.concat(frames,ignore_index=True)
    # Validate stored per-run errors against the original pilot, not report prose.
    saved=pd.read_csv(pilot/'per_run_metrics.csv');saved=saved[(saved.postprocess=='raw')&(saved.region=='whole')&saved.model.isin(old.model.unique())]
    for key,g in old.groupby(['scope','model','run_id']):
        ref=saved[(saved.scope==key[0])&(saved.model==key[1])&(saved.run_id==key[2])]
        np.testing.assert_allclose(g.abs_error.mean(),ref.mae.iloc[0],atol=1e-8)
    inventory=pd.read_csv(pilot/'run_inventory.csv')
    old=old.merge(inventory[['run_id','hardware','workload']],on='run_id',validate='many_to_one')
    parquet(out/'prior_diagnostic_predictions.parquet',old)
    rows=[]
    for keys,g in old.groupby(['analysis_label','model','collection_day','run_id','hardware','workload','endpoint_region']):
        rows.append(dict(zip(['analysis_label','model','collection_day','run_id','hardware','workload','region'],keys),n=len(g),mae=g.abs_error.mean(),gain=g.gain.mean(),mean_abs_predicted_change=abs(g.delta_hat).mean(),median_abs_predicted_change=abs(g.delta_hat).median(),direction_agreement=float((np.sign(g.delta)==np.sign(g.delta_hat)).mean()) if keys[-1]!='normal' else np.nan,zero_prediction_fraction=float((g.delta_hat==0).mean())))
    pd.DataFrame(rows).to_csv(out/'prior_diagnostic_per_run.csv',index=False)
    pivot=old.pivot(index=['scope']+KEY,columns='model',values='gain').reset_index()
    pivot['metadata_gain_difference']=pivot.residual_h5_metadata-pivot.residual_h5
    pivot.to_csv(out/'prior_metadata_gain_difference.csv',index=False)
    atomic_json(out/'prior_diagnosis_checkpoint.json',{'completed_before_new_training':True,'original_metrics_verified':True,'seconds':time.perf_counter()-t,'holdout_thresholds_retained':True})

def fit(out,cfg,cap,s,model,smoke=False):
    nested=pd.read_parquet(out/'nested_train_anchors.parquet');ev=pd.read_parquet(out/'evaluation_anchors.parquet')
    tr=nested[(nested.cap==cap)&nested.collection_day.isin(s['train_days'])];va=ev[ev.collection_day.isin(s['validation_days'])]
    if smoke:tr=tr.head(100);va=va.head(100)
    assert not set(tr.run_id)&set(va.run_id) and not set(tr.collection_day)&set(va.collection_day)
    cols=columns(5,model=='RM');x,xv,maps,audit=encode(tr,va,cols);w=weights(tr)
    ident={'train_anchor_hash':frame_hash(tr[KEY]),'evaluation_anchor_hash':frame_hash(va[KEY]),'feature_hash':frame_hash(x),'weight_hash':hashlib.sha256(w.tobytes()).hexdigest(),'cap':cap,'fold':s['fold'],'model':model}
    dest=out/'fits'/f"{cap}_{s['fold']}_{model}";dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        rec=json.loads((dest/'complete.json').read_text());assert rec['identity']==ident
        for n,h in rec['files'].items():assert digest(dest/n)==h
        return rec
    params={**PARAMS,'n_estimators':5 if smoke else 100};reg=lgb.LGBMRegressor(**params)
    y=tr.actual_w.to_numpy()-(tr.current_w.to_numpy() if model!='A' else 0)
    begin=time.perf_counter();reg.fit(x,y,sample_weight=w);train_s=time.perf_counter()-begin
    begin=time.perf_counter();p=reg.predict(xv);prediction_s=time.perf_counter()-begin
    if model!='A':p+=va.current_w.to_numpy()
    assert np.isfinite(p).all()
    q=json.loads((out/'thresholds.json').read_text())[s['fold']]
    z=decorate(va[KEY+['collection_day','issuance_s','current_w','actual_w']+META].assign(prediction_w=p,model=model,cap=cap,fold=s['fold'],target_timestamp=va.issuance_s+1),q)
    parquet(dest/'predictions.parquet',z);reg.booster_.save_model(str(dest/'q50.txt'))
    for d in audit:
        c=d['column'];d.update(fold=s['fold'],cap=cap,model=model,feature_count=len(cols),training_categories=json.dumps(sorted(tr[c].dropna().astype(str).unique().tolist())) if d['categorical'] else '',validation_categories=json.dumps(sorted(va[c].dropna().astype(str).unique().tolist())) if d['categorical'] else '')
    pd.DataFrame(audit).to_csv(dest/'metadata_audit.csv',index=False)
    parquet(dest/'weights.parquet',tr[KEY].assign(sample_weight=w))
    rec={'identity':ident,'train_rows':len(tr),'train_runs':int(tr.run_id.nunique()),'validation_rows':len(va),'feature_columns':cols,'encoding':maps,'fit_seconds':train_s,'prediction_seconds':prediction_s,'parameters':reg.get_params(),'new_fit':True,'source':'new follow-up fit; no pilot model reuse','files':{n:digest(dest/n) for n in ['predictions.parquet','q50.txt','metadata_audit.csv','weights.parquet']}}
    atomic_json(dest/'complete.json',rec);print(json.dumps({'event':'fit_completed','cap':cap,'fold':s['fold'],'model':model,'train_rows':len(tr),'fit_seconds':train_s}),flush=True)
    return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'config.yaml');ap.add_argument('--output-dir',type=Path,required=True)
    for flag in ['resume','smoke','report-only']:ap.add_argument('--'+flag,action='store_true')
    args=ap.parse_args();cfg=yaml.safe_load(args.config.read_text());project=ROOT.parents[2];pilot=(project/cfg['pilot']).resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lockdir=ROOT/'outputs/.locks';lockdir.mkdir(parents=True,exist_ok=True);lock=open(lockdir/('smoke.lock' if args.smoke else 'full.lock'),'a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.perf_counter();stage='audit';code=1;times={}
    def progress(**kw):atomic_json(out/'progress.json',{'status':'running','stage':stage,'pid':os.getpid(),'elapsed_seconds':time.perf_counter()-start,**kw})
    def stop(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        progress()
        inputs=[p for p in pilot.glob('*') if p.is_file() and p.suffix in ['.parquet','.json','.yaml','.csv']]
        inputs += list(pilot.glob('fits/*/complete.json'))+list(pilot.glob('fits/*/q50.txt'))
        identity={'config':digest(args.config),'code':{str(p):digest(p) for p in list(ROOT.glob('*.py'))+list(PILOT.glob('*.py'))},'inputs':{str(p):digest(p) for p in inputs},'lightgbm':lgb.__version__,'smoke':args.smoke}
        if (out/'identity.json').exists():
            assert args.resume or args.report_only;assert identity==json.loads((out/'identity.json').read_text()),'identity mismatch'
        else:assert not args.report_only;atomic_json(out/'identity.json',identity)
        atomic_json(out/'resolved_paths.json',{'project_root':str(project),'pilot':str(pilot),'pilot_code':str(PILOT),'metadata_reference':str(project/'experiments/aidc_grid_metadata_20260929'),'diagnosis_reference':str(project/'experiments/aidc_forecast_diagnosis_20261001'),'continuous_cache':str(pilot/'continuous_features.parquet'),'output':str(out),'git_status':'not a git repository','AGENTS':'none found','missing_pilot_artifacts':[]})
        (out/'resolved_config.yaml').write_text(yaml.safe_dump({**cfg,'api_parameters':lgb.LGBMRegressor(**PARAMS).get_params(),'lightgbm_version':lgb.__version__,'encoding':'pilot ordinal float codes; training fold only; unseen and missing categorical=-1; numeric missing=NaN','threshold_source':'pilot 300 training anchors per fold','feature_columns':{'A':columns(5),'R':columns(5),'RM':columns(5,True)},'no_early_stopping':True,'sequential_fits':True,'pilot_reuse_reason':'conservative new training of all 27 cells; existing artifacts preserved'},allow_unicode=True,sort_keys=False))
        if not args.report_only:
            if not (out/'sampling_checkpoint.json').exists():
                stage='prior_saved_prediction_diagnosis';progress();prior_diagnosis(pilot,out)
                stage='nested_sampling';progress();times['preprocessing_seconds']=prepare(pilot,out,cfg)
            else:
                for n,h in json.loads((out/'sampling_checkpoint.json').read_text()).items():assert digest(out/n)==h
            splits=json.loads((out/'split_manifest.json').read_text())['folds']
            if args.smoke:
                stage='smoke';progress();fit(out,cfg,100,splits[0],'RM',True)
            else:
                stage='training';progress()
                for cap in cfg['caps']:
                    for s in splits:
                        for model in ['A','R','RM']:
                            fit(out,cfg,cap,s,model);progress(cap=cap,fold=s['fold'],model=model)
        if not args.smoke:
            stage='aggregation_report';progress();t=time.perf_counter()
            from report import render
            render(out,pilot,cfg);times['report_seconds']=time.perf_counter()-t
        recs=[json.loads(p.read_text()) for p in (out/'fits').glob('*/complete.json')]
        times.update(fit_seconds=sum(r['fit_seconds'] for r in recs),prediction_seconds=sum(r['prediction_seconds'] for r in recs),wall_clock_seconds=time.perf_counter()-start)
        atomic_json(out/('report_only_timing.json' if args.report_only else 'timing.json'),times)
        code=0;atomic_json(out/'completion_status.json',{'status':'completed','exit_code':0,'smoke':args.smoke,'report_only':args.report_only,'new_fits_this_invocation':0 if args.report_only else len(recs),'reused_pilot_fits':0,'new_holdout_predictions':0,'times':times})
        atomic_json(out/'progress.json',{'status':'completed','stage':'completed','pid':os.getpid()})
    except BaseException as e:
        code=e.code if isinstance(e,SystemExit) and isinstance(e.code,int) else 1
        atomic_json(out/'completion_status.json',{'status':'failed','stage':stage,'error':str(e),'exit_code':code});traceback.print_exc()
    finally:(out/'exit_code.txt').write_text(str(code)+'\n')
    sys.exit(code)
if __name__=='__main__':main()
