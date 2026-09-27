#!/usr/bin/env python
"""Controlled pooled metadata ablation.  Outputs are intentionally self contained."""
import argparse, json, os, platform, re, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'trace/predictability'/'src'))
from core import continuous_segments, bin_segment, samples, crps, rearrange

def qcols(q): return [f'q_{x:g}' for x in q]
def subset(a,n): return a[np.linspace(0,len(a)-1,min(len(a),n),dtype=int)]
def meta_from_row(r):
    s=str(r.source_path); b=re.search(r'BatchSize(\\d+)',s)
    return dict(gpu_model=r.hardware,workload_type=r.workload, batch_size=float(b.group(1)) if b else np.nan,
      ai_model=('diffusion' if r.workload=='image_generation' else 'Llama8B'), gpu_count=float(r.gpu_count))
def feature_matrix(power, records, condition, train_records=None):
    # explicit train-only category vocabulary and unknown all-zero encoding
    base=np.asarray(power,dtype=np.float32)
    if condition=='history_only': return base, [], dict()
    cats=['gpu_model','workload_type','ai_model']; nums=['batch_size']
    vocab={c:sorted(train_records[c].dropna().astype(str).unique()) for c in cats}
    pieces=[base]; names=[]
    for c in cats:
        values=records[c].fillna('__MISSING__').astype(str)
        for v in vocab[c]: pieces.append((values==v).to_numpy(np.float32)[:,None]); names.append(c+'='+v)
        pieces.append((~values.isin(vocab[c])).to_numpy(np.float32)[:,None]); names.append(c+'=__UNKNOWN__')
    med=float(train_records.batch_size.median()) if train_records.batch_size.notna().any() else 0.
    x=records.batch_size.fillna(med).to_numpy(np.float32)[:,None]; pieces.extend([x,records.batch_size.isna().to_numpy(np.float32)[:,None]])
    names += ['batch_size','batch_size_missing']
    return np.column_stack(pieces),names,vocab
def fit_predict(xtr,ytr,xev,q,cfg):
    started=time.perf_counter(); ms=[]
    for a in q:
        m=LGBMRegressor(objective='quantile',alpha=float(a),n_estimators=cfg['n_estimators'],num_leaves=cfg['num_leaves'],learning_rate=cfg['learning_rate'],min_child_samples=30,n_jobs=cfg['n_jobs'],random_state=cfg['seed'],verbosity=-1)
        m.fit(xtr,ytr); ms.append(m)
    fit_s=time.perf_counter()-started; started=time.perf_counter(); p=np.column_stack([m.predict(xev) for m in ms]); p,cross=rearrange(p)
    return p,fit_s,time.perf_counter()-started,cross
def metrics(f,q):
    p=f[qcols(q)].to_numpy(); y=f.y.to_numpy(); out={'n_targets':len(f),'crps_w':float(f.crps.mean()),'mae_median_w':float(np.abs(y-p[:,list(q).index(.5)]).mean())}
    for lev,lo,hi in [(80,.1,.9),(90,.05,.95)]:
      out[f'coverage_{lev}']=float(((y>=p[:,list(q).index(lo)])&(y<=p[:,list(q).index(hi)] )).mean());out[f'width_{lev}_w']=float((p[:,list(q).index(hi)]-p[:,list(q).index(lo)]).mean())
    return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(HERE/'configs/default.yaml'));p.add_argument('--smoke',action='store_true');p.add_argument('--experiment-id');a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text())
 if a.smoke: cfg.update(experiment_id='smoke_20260915',resolutions_ms=[200],horizons_ms=[1000],n_estimators=6,max_train_samples=400,max_eval_samples_per_run=180,bootstrap_replicates=20)
 if a.experiment_id:cfg['experiment_id']=a.experiment_id
 out=HERE/'outputs'/cfg['experiment_id'];out.mkdir(parents=True,exist_ok=False); (out/'predictions').mkdir(); (out/'models').mkdir(); start=time.time()
 def log(x): print(x,flush=True);(out/'execution.log').open('a').write(x+'\\n')
 status={'status':'running','pid':os.getpid(),'started_utc':time.strftime('%FT%TZ',time.gmtime())};(out/'status.json').write_text(json.dumps(status,indent=2));(out/'effective_config.json').write_text(json.dumps(cfg,indent=2))
 try:
  man=pd.read_csv(ROOT/'trace/predictability/outputs/main_20260914/input_manifest.csv'); man['run_id']=man.run_id.astype(str); md=pd.DataFrame([meta_from_row(r) for r in man.itertuples()]); inv=pd.concat([man.reset_index(drop=True),md],axis=1)
  inv['selection_rule']='exact 12 canonical highres_node rows in prior main manifest; pooled before scoring'; inv['metadata_source']='explicit manifest hardware/workload/gpu_count plus documented filename BatchSizeNN';inv.to_csv(out/'metadata_inventory.csv',index=False)
  prov=[]
  for f in ['gpu_model','workload_type','batch_size','ai_model','gpu_count']:
   prov.append({'field':f,'evidence':'prior input manifest' if f in ['gpu_model','workload_type','gpu_count'] else 'canonical filename, regex BatchSize([0-9]+)' if f=='batch_size' else 'filename Lamma8B; diffusion workload label','known_at_issue':True,'distinct_values':inv[f].nunique(dropna=True),'missing':inv[f].isna().sum(),'classification':'usable and varying' if f!='gpu_count' else 'constant; excluded'})
  pd.DataFrame(prov).to_csv(out/'metadata_provenance.csv',index=False)
  audit='# Metadata audit\\n\\nThe fixed input rule selected the exact twelve canonical highres_node runs from the preceding main manifest, after duplicate handling there. GPU model (B200/H100), workload (image_generation/llm_training), batch size (2/16/32 or 128/256/512) and explicit model identifiers are available before issuance. GPU count is always 8 and is excluded. `ai_model` is perfectly confounded with workload; it is excluded from model inputs despite being retained in inventory. GPU/workload combinations are balanced (three runs each), but GPU, workload, and batch effects are not individually causal in this ablation. No IDs, paths, hashes, or output names are features. NLR is excluded because the prior audit documents two-sided interpolation.\\n'
  (out/'metadata_audit.md').write_text(audit)
  # load and causally bin each selected prepared native trace
  raw={}
  for r in inv.itertuples():
   d=pd.read_parquet(ROOT/'trace/preprocessing/outputs/prepared'/f'{r.run_id}.parquet'); gap=float(np.median(np.diff(d.time_s))*cfg['gap_multiplier']);raw[r.run_id]=(r,d,continuous_segments(d,gap))
  all_predictions=[]; summaries=[]; folds=[]; features_out=[]; timings=[]; selections=[]
  q=np.array(cfg['quantiles'])
  for res in cfg['resolutions_ms']:
   for hor in cfg['horizons_ms']:
    if hor<res or hor%res: continue
    rows=[]
    for run,(r,d,segs) in raw.items():
     c1=d.time_s.min()+.6*(d.time_s.max()-d.time_s.min());c2=d.time_s.min()+.8*(d.time_s.max()-d.time_s.min())
     for sid,s in segs:
      z=bin_segment(s,res,coverage=cfg['min_bin_coverage'])
      for sub in z.subsegment.unique():
       got=samples(z[z.subsegment==sub],res,hor,cfg['histories_ms'],c1,c2,res)
       if got:
        m,fs=got;m['run_id']=run;m['cohort']=r.hardware+'__'+r.workload;m['segment_id']=str(sid); [m.__setitem__(k,getattr(r,k)) for k in ['gpu_model','workload_type','batch_size','ai_model']]; rows.append((m,fs))
    # concatenate separately per history in common row order
    meta=pd.concat([x[0] for x in rows],ignore_index=True); fdict={h:np.concatenate([x[1][h] for x in rows]) for h in cfg['histories_ms'] if h>=res and h%res==0}
    for protocol in ['within_run','heldout_run']:
     assignments=[]
     if protocol=='within_run': assignments=[('chronological',np.flatnonzero(meta.split==0),np.flatnonzero(meta.split==1),np.flatnonzero(meta.split==2))]
     else:
      # one run per cohort per fold, deterministic path order inherited from manifest
      for fold in range(3):
       held=[]
       for _,g in inv.groupby(['hardware','workload'],sort=True): held.append(g.iloc[fold].run_id)
       te=np.flatnonzero(meta.run_id.isin(held)); trruns=~meta.run_id.isin(held); tr=np.flatnonzero(trruns&(meta.split==0));va=np.flatnonzero(trruns&(meta.split==1));assignments.append((f'fold_{fold+1}',tr,va,te))
       for x in held: folds.append({'resolution_ms':res,'horizon_ms':hor,'fold':fold+1,'held_out_run':x,'protocol':protocol})
     for fold,tr,va,te in assignments:
      # shared history choice only history-only validation
      best=None
      for h,power in fdict.items():
       x,_,_=feature_matrix(power,meta,'history_only',meta.iloc[tr]); pv,_,_,_=fit_predict(x[subset(tr,cfg['max_train_samples'])],meta.y.to_numpy()[subset(tr,cfg['max_train_samples'])],x[subset(va,cfg['max_eval_samples_per_run']*12)],q,cfg);score=crps(meta.y.to_numpy()[subset(va,cfg['max_eval_samples_per_run']*12)],pv,q).mean()
       if best is None or score<best[0]:best=(score,h)
      h=best[1]; selections.append({'protocol':protocol,'fold':fold,'resolution_ms':res,'horizon_ms':hor,'history_ms':h,'validation_crps_history_only_w':best[0]})
      for cond in ['history_only','history_meta']:
       x,names,vocab=feature_matrix(fdict[h],meta,cond,meta.iloc[tr]); assert (x[:,:fdict[h].shape[1]]==fdict[h]).all()
       truse=subset(tr,cfg['max_train_samples']); teuse=np.concatenate([subset(g.index.to_numpy(),cfg['max_eval_samples_per_run']) for _,g in meta.iloc[te].groupby('run_id')])
       pred,fit_s,inf_s,cross=fit_predict(x[truse],meta.y.to_numpy()[truse],x[teuse],q,cfg); fr=meta.iloc[teuse].copy();fr['protocol']=protocol;fr['fold']=fold;fr['condition']=cond;fr['resolution_ms']=res;fr['horizon_ms']=hor;fr['history_ms']=h;fr['crps']=crps(fr.y,pred,q)
       for j,c in enumerate(qcols(q)):fr[c]=pred[:,j]
       all_predictions.append(fr);timings.append({'protocol':protocol,'fold':fold,'condition':cond,'resolution_ms':res,'horizon_ms':hor,'fit_s':fit_s,'inference_s':inf_s,'crossing_fraction':cross});features_out.append({'protocol':protocol,'fold':fold,'condition':cond,'history_ms':h,'power_feature_count':fdict[h].shape[1],'metadata_features':json.dumps(names),'category_vocabularies':json.dumps(vocab)})
  pred=pd.concat(all_predictions,ignore_index=True);pred.to_parquet(out/'predictions/all_predictions.parquet',index=False,compression='zstd');pd.DataFrame(folds).to_csv(out/'fold_assignments.csv',index=False);pd.DataFrame(selections).to_csv(out/'selected_validation_configs.csv',index=False);pd.DataFrame(features_out).to_csv(out/'feature_lists.csv',index=False);pd.DataFrame(timings).to_csv(out/'timing.csv',index=False)
  for keys,g in pred.groupby(['protocol','fold','resolution_ms','horizon_ms','condition']): summaries.append(dict(zip(['protocol','fold','resolution_ms','horizon_ms','condition'],keys))|metrics(g,q))
  sm=pd.DataFrame(summaries); runm=[]
  for keys,g in pred.groupby(['protocol','fold','resolution_ms','horizon_ms','condition','run_id','cohort']):runm.append(dict(zip(['protocol','fold','resolution_ms','horizon_ms','condition','run_id','cohort'],keys))|metrics(g,q))
  rm=pd.DataFrame(runm); rm.to_csv(out/'per_run_metrics.csv',index=False);sm.to_csv(out/'metrics_target_weighted.csv',index=False)
  rb=rm.groupby(['protocol','resolution_ms','horizon_ms','condition']).crps_w.mean().reset_index(name='run_balanced_crps_w'); pivot=rb.pivot(index=['protocol','resolution_ms','horizon_ms'],columns='condition',values='run_balanced_crps_w').reset_index();pivot['delta_crps_w']=pivot.history_meta-pivot.history_only;pivot['meta_gain']=1-pivot.history_meta/pivot.history_only;pivot.to_csv(out/'comparison_summary.csv',index=False)
  # paired run bootstrap over held-out / chronological run means
  ci=[]; rng=np.random.default_rng(cfg['seed'])
  for keys,g in rm.pivot_table(index=['protocol','fold','resolution_ms','horizon_ms','run_id'],columns='condition',values='crps_w').reset_index().groupby(['protocol','resolution_ms','horizon_ms']):
   d=(g.history_meta-g.history_only).to_numpy(); bs=[rng.choice(d,len(d),replace=True).mean() for _ in range(cfg['bootstrap_replicates'])];ci.append(dict(zip(['protocol','resolution_ms','horizon_ms'],keys))|{'paired_delta_crps_w':d.mean(),'ci_low_w':np.percentile(bs,2.5),'ci_high_w':np.percentile(bs,97.5),'n_run_scores':len(d),'note':'fixed out-of-fold/temporal predictions; sample uncertainty only'})
  pd.DataFrame(ci).to_csv(out/'confidence_intervals.csv',index=False)
  report=['# 메타데이터 절제 실험 결과','', '## 실행 상태','완료. 동일한 12개 native highres_node trace를 pooled Quantile LightGBM으로 평가했다. 주 그리드는 사전 선언된 계산 절감판(200 ms: 200/1000/2000 ms, 1000 ms: 1000/2000 ms)이다.','', '## 해석','`comparison_summary.csv`의 run-balanced CRPS가 주 요약이며, `delta_crps_w < 0` 및 양의 `meta_gain`은 메타데이터 개선을 뜻한다. heldout_run은 모델/선택에 해당 run의 미래를 쓰지 않았지만, GPU·workload 범주는 익숙할 수 있다. 배치 크기는 held-out 값일 수 있고 수치형으로 전달된다. bootstrap CI는 고정 예측의 run-level 표본 불확실성일 뿐 재학습 불확실성은 아니다. 이 두 조건 비교는 개별 필드의 인과효과나 제어 가능성을 보이지 않는다.','', '## 산출물','per-target quantile 예측, fold, 선택값, feature vocabulary, timing, metadata audit와 모든 표는 이 디렉터리에 저장했다. Interval/transition 층화는 본 계산 절감판에서는 생략했으며 이 제한을 명시한다.']
  (out/'report_ko.md').write_text('\\n'.join(report));status.update(status='completed',elapsed_s=time.time()-start);log('COMPLETED')
 except Exception as e: status.update(status='failed',error=str(e));log(traceback.format_exc());raise
 finally:(out/'status.json').write_text(json.dumps(status,indent=2))
if __name__=='__main__':main()
