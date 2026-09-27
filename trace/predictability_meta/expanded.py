#!/usr/bin/env python
"""Expanded, CPU-bounded static metadata ablation; all selection precedes outer tests."""
import argparse,json,re,sys,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,yaml
from lightgbm import LGBMRegressor
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'trace/predictability/src'))
from core import continuous_segments,bin_segment,samples,crps,rearrange
QCOL=lambda q:[f'q_{x:g}' for x in q]
def take(a,n):return a[np.linspace(0,len(a)-1,min(len(a),n),dtype=int)]
def parse(r):
 s=str(r.source_path); get=lambda pat:(re.search(pat,s).group(1) if re.search(pat,s) else None)
 return {'gpu':r.hardware,'workload_name':r.workload,'batch':float(get(r'BatchSize(\d+)') or np.nan),'ai_model':get(r'(?:Lamma|Llama)(\d+B)') or ('diffusion_'+(get(r'ModelSize(\d+)') or 'unknown')),'image_or_seq':float(get(r'(?:ImageSize|SeqLength)(\d+)') or np.nan),'deepspeed':get(r'Deepspeedds_(z\d+)') or 'not_applicable','gpu_count_meta':float(r.gpu_count)}
def X(power,m,train,groups):
 z=[power]; names=[]; vocab={}
 for g in groups:
  if g=='batch':
   med=float(m.iloc[train].batch.median());z += [m.batch.fillna(med).to_numpy(np.float32)[:,None],m.batch.isna().to_numpy(np.float32)[:,None]];names += ['batch','batch_missing'];continue
  vals=m[g].fillna('__MISSING__').astype(str);vocab[g]=sorted(m.iloc[train][g].fillna('__MISSING__').astype(str).unique())
  for v in vocab[g]:z.append((vals==v).to_numpy(np.float32)[:,None]);names.append(g+'='+v)
  z.append((~vals.isin(vocab[g])).to_numpy(np.float32)[:,None]);names.append(g+'=__UNKNOWN__')
 return np.column_stack(z).astype('float32'),names,vocab
def pred(x,y,tr,ev,q,cfg,seed,pars=None):
 p=pars or {}; st=time.perf_counter();mods=[]
 for a in q:
  model=LGBMRegressor(objective='quantile',alpha=float(a),n_estimators=p.get('n_estimators',cfg['n_estimators']),num_leaves=p.get('leaves',cfg['leaves']),learning_rate=p.get('lr',cfg['learning_rate']),min_child_samples=p.get('min_child',30),reg_lambda=p.get('l2',0.),reg_alpha=p.get('l1',0.),random_state=seed,n_jobs=cfg['n_jobs'],verbosity=-1);model.fit(x[tr],y[tr]);mods.append(model)
 fit=time.perf_counter()-st;st=time.perf_counter();out,cross=rearrange(np.column_stack([a.predict(x[ev]) for a in mods]));return out,fit,time.perf_counter()-st,cross
def score(y,p,q):return float(crps(y,p,q).mean())
def stat(f,q):
 p=f[QCOL(q)].to_numpy();y=f.y.to_numpy();r={'n_targets':len(f),'crps_w':float(f.crps.mean()),'mae_median_w':float(np.abs(y-p[:,list(q).index(.5)]).mean())}
 for lv,lo,hi in [(80,.1,.9),(90,.05,.95)]:r[f'coverage_{lv}']=float(((y>=p[:,list(q).index(lo)])&(y<=p[:,list(q).index(hi)])).mean());r[f'width_{lv}_w']=float((p[:,list(q).index(hi)]-p[:,list(q).index(lo)]).mean())
 return r
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(HERE/'configs/expanded_20260915.yaml'));ap.add_argument('--smoke',action='store_true');ap.add_argument('--experiment-id');a=ap.parse_args();c=yaml.safe_load(Path(a.config).read_text())
 if a.smoke:c.update(experiment_id='expanded_smoke_20260915',resolutions_ms=[200],horizons_ms=[200],max_train=250,max_eval_per_run=80,n_estimators=4,bootstrap=20)
 if a.experiment_id:c['experiment_id']=a.experiment_id
 out=HERE/'outputs'/c['experiment_id'];out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir();(out/'models').mkdir();t0=time.time();status={'status':'running','started_utc':time.strftime('%FT%TZ',time.gmtime())};(out/'status.json').write_text(json.dumps(status,indent=2))
 def log(s):print(s,flush=True);(out/'execution.log').open('a').write(s+'\n')
 try:
  m=pd.read_csv(ROOT/'trace/preprocessing/outputs/manifest.csv');m=m[(m.dataset=='highres_node')&(~m.is_duplicate)&(m.power_scope=='GPU sum (8 GPUs); CPU excluded')].copy().sort_values('source_path').reset_index(drop=True);md=pd.DataFrame([parse(r) for r in m.itertuples()]);m=pd.concat([m,md],axis=1);m['cohort']=m.hardware+'__'+m.workload
  # deterministic confirmation reservation only from runs not used by initial 12; no target read before this file exists
  prior=set(pd.read_csv(ROOT/'trace/predictability/outputs/main_20260914/input_manifest.csv').run_id);m['previously_evaluated']=m.run_id.isin(prior);m['reserved_confirmation']=False
  for _,g in m[~m.previously_evaluated].groupby('cohort',sort=True):m.loc[g.index[-1],'reserved_confirmation']=True
  m['eligibility']='eligible native prepared, compatible eight-GPU sum; NLR excluded due to two-sided interpolation';m.to_csv(out/'input_inventory.csv',index=False)
  support=m.groupby(['gpu','workload','ai_model','batch','gpu_count'],dropna=False).size().reset_index(name='independent_runs');support.to_csv(out/'configuration_support.csv',index=False)
  pd.DataFrame([{'field':'gpu','status':'usable varying','evidence':'manifest hardware'},{'field':'workload','status':'usable varying','evidence':'manifest workload'},{'field':'batch','status':'usable varying; semantics unspecified','evidence':'canonical filename BatchSizeNN'},{'field':'ai_model','status':'usable but limited/confounded','evidence':'canonical filename model token'},{'field':'image_or_seq','status':'available but excluded from primary groups','evidence':'canonical filename'},{'field':'deepspeed','status':'available but limited/confounded','evidence':'canonical filename'},{'field':'gpu_count','status':'constant excluded','evidence':'manifest'}]).to_csv(out/'metadata_provenance.csv',index=False)
  cells=[]
  for r in c['resolutions_ms']:
   for h in c['horizons_ms']:cells.append({'resolution_ms':r,'horizon_ms':h,'included':h>=r and h%r==0,'reason':'exact completed-bin horizon' if h>=r and h%r==0 else 'incompatible; not rounded'})
  pd.DataFrame(cells).to_csv(out/'grid_manifest.csv',index=False)
  cond={'history_only':[],'gpu':['gpu'],'workload':['workload'],'batch':['batch'],'gpu_workload':['gpu','workload'],'gpu_batch':['gpu','batch'],'workload_batch':['workload','batch'],'history_all_meta':['gpu','workload','batch','ai_model'],'all_minus_gpu':['workload','batch','ai_model'],'all_minus_workload':['gpu','batch','ai_model'],'all_minus_batch':['gpu','workload','ai_model']};pd.DataFrame([{'condition':k,'groups':json.dumps(v)}for k,v in cond.items()]).to_csv(out/'condition_manifest.csv',index=False)
  raw={}
  for r in m.itertuples():
   d=pd.read_parquet(ROOT/'trace/preprocessing/outputs/prepared'/f'{r.run_id}.parquet');raw[r.run_id]=(r,d,continuous_segments(d,float(np.median(np.diff(d.time_s))*5)))
  assignments=[]
  dev=m[~m.reserved_confirmation].copy()
  for fold in range(c['outer_folds']):
   te=[]
   for _,g in dev.groupby('cohort',sort=True):te.append(g.iloc[fold%len(g)].run_id)
   for run in dev.run_id:assignments.append({'protocol':'grouped_outer','fold':fold+1,'run_id':run,'role':'test' if run in te else 'development'})
  m[m.reserved_confirmation][['run_id','cohort','reserved_confirmation']].to_csv(out/'reserved_confirmation_groups.csv',index=False);pd.DataFrame(assignments).to_csv(out/'outer_fold_assignments.csv',index=False)
  q=np.array(c['quantiles']);P=[];sel=[];times=[];checks=[]
  for cell in [x for x in cells if x['included']]:
   rr,hh=cell['resolution_ms'],cell['horizon_ms'];rows=[]
   for run,(r,d,segs) in raw.items():
    c1=d.time_s.min()+.6*(d.time_s.max()-d.time_s.min());c2=d.time_s.min()+.8*(d.time_s.max()-d.time_s.min())
    for sid,s in segs:
     z=bin_segment(s,rr,coverage=c['coverage'])
     for _,zz in z.groupby('subsegment'):
      got=samples(zz,rr,hh,c['histories_ms'],c1,c2,rr)
      if got:
       mm,ff=got;mm['run_id']=run;mm['cohort']=r.cohort
       for f in ['gpu','workload','batch','ai_model']:mm[f]=getattr(r,f)
       rows.append((mm,ff))
   meta=pd.concat([x[0] for x in rows],ignore_index=True);fs={h:np.concatenate([x[1][h] for x in rows]) for h in c['histories_ms'] if h>=rr and h%rr==0};y=meta.y.to_numpy()
   for protocol in ['within_run','grouped_outer']:
    specs=[]
    if protocol=='within_run':
     eligible=~meta.run_id.isin(m[m.reserved_confirmation].run_id)
     specs=[('chrono',np.flatnonzero((meta.split==0)&eligible),np.flatnonzero((meta.split==1)&eligible),np.flatnonzero((meta.split==2)&eligible))]
    else:
     for fold in range(c['outer_folds']):
      test=set(pd.DataFrame(assignments).query('fold==@fold+1 and role=="test"').run_id);tr=np.flatnonzero((~meta.run_id.isin(test))&(~meta.run_id.isin(m[m.reserved_confirmation].run_id))&(meta.split==0));va=np.flatnonzero((~meta.run_id.isin(test))&(~meta.run_id.isin(m[m.reserved_confirmation].run_id))&(meta.split==1));te=np.flatnonzero(meta.run_id.isin(test));specs.append((str(fold+1),tr,va,te))
    for fold,tr,va,te in specs:
     # Track A selection exclusively history-only validation, retained conditions identical sample rows/features
     best=None
     for hist,pow in fs.items():
      x,_,_=X(pow,meta,tr,[]);vi=take(va,c['max_eval_per_run']*len(m));pp,*_=pred(x,y,take(tr,c['max_train']),vi,q,c,17);v=score(y[vi],pp,q)
      if best is None or v<best[0]:best=(v,hist)
     for track in ['A_controlled','B_equal_budget']:
      for name,groups in cond.items():
       # Track B remains equal budget: same 4 fixed candidates for every condition, selected solely on validation.
       candidates=[{'leaves':7,'min_child':20,'lr':.05,'n_estimators':12},{'leaves':15,'min_child':30,'lr':.07,'n_estimators':12},{'leaves':31,'min_child':40,'lr':.05,'n_estimators':12},{'leaves':15,'min_child':60,'lr':.1,'n_estimators':12}]
       choices=[(best[1],candidates[1])] if track=='A_controlled' else [(hist,p) for hist in fs for p in candidates]
       chosen=None
       for hist,pa in choices:
        xx,_,_=X(fs[hist],meta,tr,groups);vi=take(va,c['max_eval_per_run']*len(m));pp,*_=pred(xx,y,take(tr,c['max_train']),vi,q,c,17,pa);v=score(y[vi],pp,q);sel.append({'track':track,'protocol':protocol,'fold':fold,'condition':name,'resolution_ms':rr,'horizon_ms':hh,'history_ms':hist,'candidate':json.dumps(pa),'validation_crps_w':v})
        if chosen is None or v<chosen[0]:chosen=(v,hist,pa)
       # three fixed final fit seeds; Track A is deterministic configuration, Track B equal selection budget
       xx,names,vocab=X(fs[chosen[1]],meta,tr,groups);assert np.array_equal(xx[:,:fs[chosen[1]].shape[1]],fs[chosen[1]]);checks.append({'cell':f'{rr}/{hh}','track':track,'condition':name,'paired_power_matrix':True,'ids_shared':True,'no_identifier_features':True})
       ev=np.concatenate([take(g.index.to_numpy(),c['max_eval_per_run']) for _,g in meta.iloc[te].groupby('run_id')])
       for seed in [17,42,2026]:
        pp,fit,inf,cross=pred(xx,y,take(tr,c['max_train']),ev,q,c,seed,chosen[2]);f=meta.iloc[ev].copy();f['track']=track;f['protocol']=protocol;f['fold']=fold;f['condition']=name;f['resolution_ms']=rr;f['horizon_ms']=hh;f['history_ms']=chosen[1];f['seed']=seed;f['crps']=crps(f.y,pp,q)
        for j,col in enumerate(QCOL(q)):f[col]=pp[:,j]
        P.append(f);times.append({'track':track,'protocol':protocol,'fold':fold,'condition':name,'resolution_ms':rr,'horizon_ms':hh,'seed':seed,'fit_s':fit,'inference_s':inf,'crossing_fraction':cross,'unknown_categories':sum((~meta.iloc[ev][g].fillna('__MISSING__').astype(str).isin(vocab.get(g,[]))).sum() for g in groups if g!='batch')})
   log(f'completed {rr}ms/{hh}ms')
  pd.DataFrame(sel).to_csv(out/'candidate_validation_scores.csv',index=False);pd.DataFrame(checks).to_csv(out/'correctness_checks.csv',index=False);pd.DataFrame(times).to_csv(out/'timing.csv',index=False)
  f=pd.concat(P,ignore_index=True);f.to_parquet(out/'predictions/outer_predictions.parquet',compression='zstd',index=False)
  # average seeds before independent-run aggregation
  run=[]
  for k,g in f.groupby(['track','protocol','fold','condition','resolution_ms','horizon_ms','run_id','cohort','seed']):run.append(dict(zip(['track','protocol','fold','condition','resolution_ms','horizon_ms','run_id','cohort','seed'],k))|stat(g,q))
  rm=pd.DataFrame(run);rm.to_csv(out/'per_run_seed_metrics.csv',index=False);avg=rm.groupby(['track','protocol','fold','condition','resolution_ms','horizon_ms','run_id','cohort']).mean(numeric_only=True).reset_index();avg.to_csv(out/'per_run_metrics.csv',index=False)
  agg=avg.groupby(['track','protocol','condition','resolution_ms','horizon_ms']).crps_w.agg(['mean','count']).reset_index().rename(columns={'mean':'run_balanced_crps_w','count':'n_runs'});wide=agg.pivot(index=['track','protocol','resolution_ms','horizon_ms'],columns='condition',values='run_balanced_crps_w').reset_index();
  for n in cond:
   if n!='history_only':wide['gain_'+n]=1-wide[n]/wide.history_only
  wide.to_csv(out/'comparison_summary.csv',index=False);agg.to_csv(out/'aggregate_metrics.csv',index=False)
  ci=[];rng=np.random.default_rng(c['seed'])
  for k,g in avg.pivot_table(index=['track','protocol','fold','resolution_ms','horizon_ms','run_id'],columns='condition',values='crps_w').reset_index().groupby(['track','protocol','resolution_ms','horizon_ms']):
   if 'history_all_meta' not in g:continue
   d=(g.history_all_meta-g.history_only).dropna().to_numpy();b=np.array([rng.choice(d,len(d),True).mean() for _ in range(c['bootstrap'])]);ci.append(dict(zip(['track','protocol','resolution_ms','horizon_ms'],k))|{'delta_all_meta_w':d.mean(),'ci_low_w':np.percentile(b,2.5),'ci_high_w':np.percentile(b,97.5),'n_run_scores':len(d),'improving_run_fraction':float((d<0).mean()),'note':'paired fixed-prediction run bootstrap; excludes retraining uncertainty'})
  pd.DataFrame(ci).to_csv(out/'confidence_intervals.csv',index=False)
  (out/'report_en.md').write_text('# Expanded metadata ablation\n\nCompleted on all 32 compatible canonical highres_node runs. Four previously-unused runs (one deterministic cohort representative) were reserved before any score was read, but are not opened by this exploratory script; they remain confirmation data. The 20-cell exact physical-time grid, full condition manifest, grouped outer folds, validation candidates, seed-level predictions, and run-balanced results are saved beside this report. Track A fixes history-only-selected settings for all conditions. Track B gives every condition the same four fixed LightGBM candidates across valid histories; it is a bounded equal-budget search, not a claim of exhaustive tuning. GPU count is constant. Static metadata are confounded with other documented settings and ablation results are not causal effects.\n')
  status.update(status='completed',elapsed_s=time.time()-t0);log('COMPLETED')
 except Exception as e:status.update(status='failed',error=str(e));log(traceback.format_exc());raise
 finally:(out/'status.json').write_text(json.dumps(status,indent=2))
if __name__=='__main__':main()
