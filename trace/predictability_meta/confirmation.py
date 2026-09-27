#!/usr/bin/env python
"""One-shot untouched-run confirmation for the predeclared primary comparison."""
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,yaml
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from expanded import parse,X,pred,stat,QCOL
sys.path.insert(0,str(ROOT/'trace/predictability/src'))
from core import continuous_segments,bin_segment,samples,crps

SOURCE=HERE/'outputs/expanded_main_cpu_bounded_20260915'
def take(a,n): return a[np.linspace(0,len(a)-1,min(len(a),n),dtype=int)]
def main():
 c=yaml.safe_load((HERE/'configs/expanded_cpu_bounded_20260915.yaml').read_text());out=HERE/'outputs/confirmation_v2_20260915';out.mkdir(exist_ok=False);(out/'predictions').mkdir();start=time.time();status={'status':'running'};(out/'status.json').write_text(json.dumps(status))
 m=pd.read_csv(SOURCE/'input_inventory.csv');reserved=set(m.loc[m.reserved_confirmation,'run_id']);dev=~m.run_id.isin(reserved); sel=pd.read_csv(SOURCE/'candidate_validation_scores.csv')
 # Frozen solely from development grouped-outer validation scores, before creating confirmation targets.
 plan=(sel[sel.protocol.eq('grouped_outer') & sel.condition.isin(['history_only','history_all_meta'])]
       .groupby(['track','condition','resolution_ms','horizon_ms','history_ms','candidate'],as_index=False).validation_crps_w.mean()
       .sort_values('validation_crps_w').groupby(['track','condition','resolution_ms','horizon_ms'],as_index=False).first())
 plan.to_csv(out/'frozen_confirmation_plan.csv',index=False);m[m.run_id.isin(reserved)].to_csv(out/'confirmation_groups.csv',index=False)
 raw={}
 for r in m.itertuples():
  d=pd.read_parquet(ROOT/'trace/preprocessing/outputs/prepared'/f'{r.run_id}.parquet');raw[r.run_id]=(r,d,continuous_segments(d,float(np.median(np.diff(d.time_s))*5)))
 q=np.array(c['quantiles']);rowsout=[];tim=[]
 for rr in c['resolutions_ms']:
  for hh in c['horizons_ms']:
   if hh<rr or hh%rr:continue
   rows=[]
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
   meta=pd.concat([x[0] for x in rows],ignore_index=True);fs={h:np.concatenate([x[1][h] for x in rows]) for h in c['histories_ms'] if h>=rr and h%rr==0};tr=np.flatnonzero(meta.run_id.isin(m.loc[~m.run_id.isin(reserved),'run_id']).to_numpy() & meta.split.isin([0,1]).to_numpy());ev=np.flatnonzero(meta.run_id.isin(reserved)&(meta.split==2))
   ev=np.concatenate([take(g.index.to_numpy(),c['max_eval_per_run']) for _,g in meta.iloc[ev].groupby('run_id')])
   for p in plan.query('resolution_ms==@rr and horizon_ms==@hh').itertuples():
    groups=[] if p.condition=='history_only' else ['gpu','workload','batch','ai_model'];pa=json.loads(p.candidate);x,names,vocab=X(fs[p.history_ms],meta,tr,groups)
    for seed in [17,42,2026]:
     pp,fit,inf,cross=pred(x,meta.y.to_numpy(),take(tr,c['max_train']),ev,q,c,seed,pa);f=meta.iloc[ev].copy();f['track']=p.track;f['condition']=p.condition;f['resolution_ms']=rr;f['horizon_ms']=hh;f['history_ms']=p.history_ms;f['seed']=seed;f['crps']=crps(f.y,pp,q)
     for j,col in enumerate(QCOL(q)):f[col]=pp[:,j]
     rowsout.append(f);tim.append({'track':p.track,'condition':p.condition,'resolution_ms':rr,'horizon_ms':hh,'seed':seed,'fit_s':fit,'inference_s':inf,'crossing_fraction':cross})
   print(f'completed {rr}/{hh}',flush=True)
 f=pd.concat(rowsout,ignore_index=True);f.to_parquet(out/'predictions/confirmation_predictions.parquet',compression='zstd',index=False);pd.DataFrame(tim).to_csv(out/'timing.csv',index=False)
 run=[]
 for k,g in f.groupby(['track','condition','resolution_ms','horizon_ms','run_id','cohort','seed']):run.append(dict(zip(['track','condition','resolution_ms','horizon_ms','run_id','cohort','seed'],k))|stat(g,q))
 rm=pd.DataFrame(run);rm.to_csv(out/'per_run_seed_metrics.csv',index=False);avg=rm.groupby(['track','condition','resolution_ms','horizon_ms','run_id','cohort']).mean(numeric_only=True).reset_index();avg.to_csv(out/'per_run_metrics.csv',index=False)
 a=avg.groupby(['track','condition','resolution_ms','horizon_ms']).crps_w.agg(['mean','count']).reset_index();w=a.pivot(index=['track','resolution_ms','horizon_ms'],columns='condition',values='mean').reset_index();w['delta_crps_w']=w.history_all_meta-w.history_only;w['meta_gain']=1-w.history_all_meta/w.history_only;w.to_csv(out/'comparison_summary.csv',index=False)
 status.update(status='completed',elapsed_s=time.time()-start);(out/'status.json').write_text(json.dumps(status,indent=2))
if __name__=='__main__':main()
