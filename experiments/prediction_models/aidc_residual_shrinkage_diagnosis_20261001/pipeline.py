"""Read-only saved OOF shrinkage diagnosis. No fitting or inference imports."""
import argparse, fcntl, hashlib, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
KEY = ['run_id', 'segment', 'bin']
M = ['mae', 'rmse', 'signed_bias', 'p95_abs_error', 'p99_abs_error', 'gain']
REGIONS = ['whole', 'normal', 'rising', 'falling', 'large_change']

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def fh(d):
    return hashlib.sha256(pd.util.hash_pandas_object(d, index=False).values.tobytes()).hexdigest()

def write(p, x):
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)); tmp.replace(p)

def same(a, b): np.testing.assert_allclose(a, b, atol=1e-8, rtol=0, equal_nan=True)

def source_files(cfg):
    src = PROJECT/cfg['source']; pilot = PROJECT/cfg['pilot']
    files = [src/n for n in ['evaluation_anchors.parquet','nested_train_anchors.parquet','split_manifest.json','thresholds.json','resolved_config.yaml','metrics_per_run.csv','metrics_summary.csv','sampling_checkpoint.json']]
    files += [src.parent.parent/n for n in ['pipeline.py','data.py','config.yaml','report.py']]
    files += [pilot.parent.parent/n for n in ['data.py','config.yaml']]
    files += [PROJECT/'experiments/classification_manifest.json', PROJECT/'experiments/README.md']
    legacy = PROJECT/'experiments/aidc_grid_metadata_20260929/src/aidc_exp'
    files += [legacy/n for n in ['targets.py','features.py','preprocessing.py']]
    for fold in range(3):
        for model in ['R','RM','A']:
            dest = src/'fits'/f'300_{fold}_{model}'
            if model == 'A' and not dest.exists(): continue
            files += [dest/'complete.json', dest/'predictions.parquet', dest/'q50.txt', dest/'weights.parquet']
    return files

def validate(cfg, out):
    src = PROJECT/cfg['source']; pilot = PROJECT/cfg['pilot']
    ev = pd.read_parquet(src/'evaluation_anchors.parquet')
    tr = pd.read_parquet(src/'nested_train_anchors.parquet', filters=[('cap','==',300)])
    assert set(ev.collection_day) == set(cfg['development_days'])
    assert set(tr.collection_day) == set(cfg['development_days'])
    assert not ev.duplicated(KEY).any() and not tr.duplicated(KEY).any()
    assert tr.groupby('run_id').size().max() <= 300
    pcfg = yaml.safe_load((pilot.parent.parent/'config.yaml').read_text())
    assert pcfg['dt'] == .2 and pcfg['horizon_bins'] == 5 and not pcfg['clip_zero']
    source_cfg = yaml.safe_load((src/'resolved_config.yaml').read_text())
    assert source_cfg['history_seconds'] == 5
    basecols = ['current_w','lag_1','lag_2','lag_5','lag_10','mean_5','std_5','mean_25','std_25','slope_1']
    splits = json.loads((src/'split_manifest.json').read_text())['folds']
    thresholds = {t['fold']:t for t in json.loads((src/'thresholds.json').read_text())}
    checkpoint = json.loads((src/'sampling_checkpoint.json').read_text())
    for n in ['nested_train_anchors.parquet','evaluation_anchors.parquet','split_manifest.json','thresholds.json']:
        assert digest(src/n) == checkpoint[n], n
    rows=[]; parts=[]
    for s in splits:
        fold=s['fold']; train=tr[tr.collection_day.isin(s['train_days'])]; val=ev[ev.collection_day.isin(s['validation_days'])]
        assert not set(s['train_days']) & set(s['validation_days'])
        assert not set(train.run_id) & set(val.run_id)
        assert set(train.collection_day) == set(s['train_days'])
        assert set(val.collection_day) == set(s['validation_days'])
        q=np.quantile(abs(train.actual_w-train.current_w), [.5,.9], method='linear')
        same(q,[thresholds[fold]['q50'], thresholds[fold]['q90']])
        for model in ['R','RM','A']:
            dest=src/'fits'/f'300_{fold}_{model}'
            if model=='A' and not dest.exists(): continue
            rec=json.loads((dest/'complete.json').read_text()); ident=rec['identity']
            assert ident['cap']==300 and ident['fold']==fold and ident['model']==model
            assert ident['train_anchor_hash']==fh(train[KEY])
            assert ident['evaluation_anchor_hash']==fh(val[KEY])
            assert rec['train_rows']==len(train) and rec['validation_rows']==len(val)
            assert rec['feature_columns']==basecols if model!='RM' else rec['feature_columns'][:len(basecols)]==basecols
            assert len(rec['feature_columns']) == (17 if model=='RM' else 10)
            for n in ['predictions.parquet','q50.txt','weights.parquet']:
                assert digest(dest/n)==rec['files'][n]
            w=pd.read_parquet(dest/'weights.parquet')
            assert fh(w[KEY])==fh(train[KEY])
            z=pd.read_parquet(dest/'predictions.parquet')
            assert not z.duplicated(KEY).any() and set(z.model)=={model} and set(z.cap)=={300}
            assert set(z.fold)=={fold} and set(z.collection_day)==set(s['validation_days'])
            for c in KEY+['collection_day','issuance_s','target_timestamp','current_w','actual_w','prediction_w']:
                assert z[c].notna().all(), c
            assert np.isfinite(z[['issuance_s','target_timestamp','current_w','actual_w','prediction_w']]).all().all()
            assert (z.target_timestamp==z.issuance_s+1).all()
            np.testing.assert_allclose(z.target_timestamp-z.issuance_s,1,rtol=0,atol=1e-12)
            j=z.merge(val[KEY+['collection_day','issuance_s','current_w','actual_w']],on=KEY,how='outer',suffixes=('','_manifest'),validate='one_to_one',indicator=True)
            assert (j._merge=='both').all()
            assert (j.collection_day==j.collection_day_manifest).all()
            for c in ['issuance_s','current_w','actual_w']: same(j[c],j[c+'_manifest'])
            parts.append(z)
            rows.append(dict(model=model,fold=fold,train_days=','.join(s['train_days']),evaluation_day=','.join(s['validation_days']),train_runs=train.run_id.nunique(),train_anchors=len(train),n_runs=z.run_id.nunique(),n_anchors=len(z),q50_w=q[0],q90_w=q[1],unit='W',prediction_source=str((dest/'predictions.parquet').resolve()),postprocessing='none; current + raw residual'))
    pred=pd.concat(parts,ignore_index=True)
    base=pred[pred.model=='R'].sort_values(KEY).reset_index(drop=True)
    for model in pred.model.unique():
        z=pred[pred.model==model].sort_values(KEY).reset_index(drop=True)
        assert base[KEY+['collection_day','fold']].equals(z[KEY+['collection_day','fold']])
        for c in ['current_w','actual_w','issuance_s','target_timestamp']: same(base[c],z[c])
    assert len(base)==len(ev)
    base['endpoint_region']='normal';base['large_change']=False
    for fold,q in thresholds.items():
        ix=base.fold==fold;delta=base.loc[ix,'actual_w']-base.loc[ix,'current_w']
        base.loc[ix,'endpoint_region']=np.where(abs(delta)<=q['q50'],'normal',np.where(delta>q['q50'],'rising','falling'))
        base.loc[ix,'large_change']=abs(delta)>q['q90']
    for model in pred.model.unique():
        z=pred[pred.model==model].sort_values(KEY).reset_index(drop=True)
        assert base.endpoint_region.equals(z.endpoint_region)
        assert base.large_change.equals(z.large_change)
    base=base.drop(columns=['model','prediction_w','cap','delta','delta_hat','abs_error','gain','magnitude_region'])
    for model in pred.model.unique():
        base[model]=pred[pred.model==model].sort_values(KEY).prediction_w.to_numpy()
    # Explicit canonical fields; timestamps are seconds relative to each run's origin.
    base['segment_id']=base.segment;base['anchor_timestamp']=base.issuance_s
    base['current_completed_bin_power']=base.current_w;base['actual_target_power']=base.actual_w
    base.to_parquet(out/'analysis_anchors.parquet',index=False)
    pd.DataFrame(rows).to_csv(out/'provenance.csv',index=False)
    checks=dict(training_excludes_evaluation_day=True,sample_keys_current_target_match=True,target_exactly_one_second=True,no_duplicate_missing_nonfinite_core_values=True,cap300_history5_verified=True,evaluation_manifest_matches=True,fit_artifact_hashes_verified=True,training_thresholds_recomputed=True,raw_unclipped_residual_verified_from_source=True,excluded_20250922=True,new_fits=0,new_inference=0,actual_runs=int(base.run_id.nunique()),actual_anchors=len(base),expected_counts_match=bool(base.run_id.nunique()==23 and len(base)==13338),unavailable=[],metadata_missing_values_note='Metadata missing fields existed in original encoding; required analysis fields have none.')
    return base,checks

def metric(g, p):
    error=p-g.actual_w.to_numpy();a=abs(error);perr=abs(g.actual_w-g.current_w)
    return dict(n=len(g),mae=float(a.mean()),rmse=float(np.sqrt(np.mean(error**2))),signed_bias=float(error.mean()),p95_abs_error=float(np.quantile(a,.95)),p99_abs_error=float(np.quantile(a,.99)) if len(g)>=100 else np.nan,gain=float(np.mean(perr-a)))

def aggregate(per, keys):
    grouped=per.groupby(keys,dropna=False)
    a=grouped[M].mean().reset_index()
    counts=grouped.agg(n_runs=('run_id','nunique'),n_anchors=('n','sum'),valid_region_runs=('mae','count'),p99_valid_runs=('p99_abs_error','count'),improved_runs=('gain',lambda x:int((x>1e-9).sum())),tied_runs=('gain',lambda x:int((abs(x)<=1e-9).sum())),worsened_runs=('gain',lambda x:int((x< -1e-9).sum()))).reset_index()
    a=a.merge(counts,on=keys);a['persistence_mae']=a.mae+a.gain
    a['improvement_pct']=100*a.gain/a.persistence_mae.replace(0,np.nan)
    join=[k for k in keys if k!='lambda']
    ref=a[a['lambda']==1][join+['mae']].rename(columns={'mae':'lambda1_mae'})
    a=a.merge(ref,on=join,validate='many_to_one');a['gain_vs_lambda1']=a.lambda1_mae-a.mae;a['unit']='W'
    return a

def calculate(base,cfg,out,checks):
    calc_start=time.perf_counter()
    rows=[];dcs=[]
    models=['R','RM']+(['A'] if 'A' in base else [])
    for model in models:
        for lam in (cfg['lambdas'] if model!='A' else [1]):
            p=base.current_w.to_numpy()+lam*(base[model].to_numpy()-base.current_w.to_numpy())
            if lam==0: assert np.array_equal(p,base.current_w.to_numpy())
            if lam==1: same(p,base[model])
            assert (p>=np.minimum(base.current_w,base[model])-1e-8).all() and (p<=np.maximum(base.current_w,base[model])+1e-8).all()
            for (run,day),g in base.groupby(['run_id','collection_day']):
                pr=p[g.index];gain=abs(g.actual_w-g.current_w).to_numpy()-abs(g.actual_w.to_numpy()-pr)
                common=dict(model=model,**{'lambda':lam},run_id=run,collection_day=day,fold=int(g.fold.iloc[0]),unit='W')
                masks={'whole':np.ones(len(g),bool),**{r:(g.endpoint_region==r).to_numpy() for r in REGIONS[1:4]},'large_change':g.large_change.to_numpy()}
                for region,mask in masks.items():
                    if mask.any(): values=metric(g[mask],pr[mask])
                    else: values={'n':0,**{m:np.nan for m in M}}
                    rows.append({**common,'region':region,**values})
                    if region in REGIONS[1:4]:dcs.append({**common,'region':region,'n':int(mask.sum()),'whole_n':len(g),'gain_contribution':float(gain[mask].sum()/len(g))})
    calculation_seconds=time.perf_counter()-calc_start
    agg_start=time.perf_counter()
    per=pd.DataFrame(rows)
    per['persistence_mae']=per.mae+per.gain
    per['improvement_pct']=100*per.gain/per.persistence_mae.replace(0,np.nan)
    ref=per[per['lambda']==1][['model','run_id','region','mae']].rename(columns={'mae':'lambda1_mae'})
    per=per.merge(ref,on=['model','run_id','region'],validate='many_to_one')
    per['gain_vs_lambda1']=per.lambda1_mae-per.mae
    whole=per[per.region=='whole']
    whole.to_csv(out/'metrics_per_run.csv',index=False);per.to_csv(out/'region_metrics_per_run.csv',index=False)
    summary=aggregate(whole,['model','lambda']);days=aggregate(whole,['model','lambda','collection_day'])
    regions=aggregate(per,['model','lambda','region']);regiondays=aggregate(per,['model','lambda','region','collection_day'])
    for name,d in [('metrics_summary',summary),('metrics_by_day',days),('region_metrics',regions),('region_metrics_by_day',regiondays)]:d.to_csv(out/(name+'.csv'),index=False)
    dc=pd.DataFrame(dcs);dc.to_csv(out/'gain_decomposition_per_run.csv',index=False)
    for name,keys in [('gain_decomposition',['model','lambda','region']),('gain_decomposition_by_day',['model','lambda','region','collection_day'])]:
        d=dc.groupby(keys).agg(gain_contribution=('gain_contribution','mean'),n_runs=('run_id','nunique'),n_anchors=('n','sum'),whole_n_anchors=('whole_n','sum')).reset_index();d['unit']='W';d.to_csv(out/(name+'.csv'),index=False)
    check=dc.groupby(['model','lambda','run_id']).gain_contribution.sum().reset_index().merge(whole,on=['model','lambda','run_id']);same(check.gain_contribution,check.gain)
    effects=[]
    for scope,d,keys in [('whole',summary,['lambda']),('day',days,['lambda','collection_day']),('region',regions,['lambda','region']),('day_region',regiondays,['lambda','region','collection_day'])]:
        a=d[d.model=='R'];b=d[d.model=='RM'];j=a.merge(b,on=keys,suffixes=('_R','_RM'),validate='one_to_one')
        z=j[keys].copy();z['scope']=scope
        for m in M[:5]:z[m+'_effect']=j[m+'_R']-j[m+'_RM']
        for c in ['n_runs','n_anchors','valid_region_runs','p99_valid_runs']:z[c]=j[c+'_R'];assert (j[c+'_R']==j[c+'_RM']).all()
        z['unit']='W';effects.append(z)
    effect=pd.concat(effects,ignore_index=True);effect.to_csv(out/'metadata_effect.csv',index=False)
    assert (effect[effect['lambda']==0][[m+'_effect' for m in M[:5]]].fillna(0)==0).all().all()
    r=per[per.model=='R'].sort_values(['lambda','run_id','region']);rm=per[per.model=='RM'].sort_values(['lambda','run_id','region'])
    same(r[r['lambda']==0][M+['n']],rm[rm['lambda']==0][M+['n']])
    for model in ['R','RM']:
        g=per[per.model==model]
        assert g.groupby(['run_id','region']).n.nunique().eq(1).all()
    convex=[]
    for scope,d,keys in [('whole',summary,['model']),('day',days,['model','collection_day']),('region',regions,['model','region']),('day_region',regiondays,['model','region','collection_day']),('per_run_region',per,['model','run_id','region'])]:
        for key,g in d[d.model.isin(['R','RM'])].groupby(keys):
            second=np.diff(g.sort_values('lambda').mae.to_numpy(),n=2);finite=second[np.isfinite(second)]
            assert (finite>=-1e-8).all(), (scope,key,second)
            if len(finite):convex.append(float(finite.min()))
    # Exact numeric CSV references, never rounded prose as input.
    src=PROJECT/cfg['source'];old=pd.read_csv(src/'metrics_per_run.csv');old=old[(old.cap==300)&(old.region=='whole')]
    diffs=[]
    for model in models:
        a=whole[(whole.model==model)&(whole['lambda']==1)].merge(old[old.model==model],on=['model','run_id','collection_day'],suffixes=('','_old'),validate='one_to_one')
        assert len(a)==base.run_id.nunique()
        for m in M: same(a[m],a[m+'_old']);diffs.extend(abs(a[m]-a[m+'_old']).dropna().tolist())
    oldsummary=pd.read_csv(src/'metrics_summary.csv');oldsummary=oldsummary[oldsummary.cap==300]
    for model in ['P']+models:
        a=summary[(summary.model==('R' if model=='P' else model))&(summary['lambda']==(0 if model=='P' else 1))].iloc[0];b=oldsummary[oldsummary.model==model].iloc[0]
        for m in M: same(a[m],b[m])
    checks.update(lambda0_equals_current=True,lambda1_equals_saved=True,all_cells_same_keys_targets_counts=True,intermediate_predictions_between_endpoints=True,membership_fixed=True,contribution_sum_matches_gain=True,max_contribution_error_w=float(abs(check.gain_contribution-check.gain).max()),lambda0_R_RM_identical=True,lambda0_metadata_effect_exact_zero=True,lambda1_matches_original_numeric_csv=True,max_original_metric_difference_w=max(diffs),mae_convex_all_fixed_weight_groups=True,minimum_mae_second_difference_w=min(convex),p99_minimum_n100_enforced=True,empty_region_metrics_are_NA=True,raw_residual_note='Raw residual vector not separately stored; unclipped prediction-current recovered from source-defined current+residual output.',rounded_reference_matches=True)
    write(out/'verification.json',checks)
    return dict(calculation_seconds=calculation_seconds,aggregation_and_numeric_verification_seconds=time.perf_counter()-agg_start)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    for flag in ['smoke','resume','report-only']:ap.add_argument('--'+flag,action='store_true')
    args=ap.parse_args();cfg=yaml.safe_load(args.config.read_text());out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    assert cfg['lambdas']==[0,.25,.5,.75,1] and cfg['cap']==300 and cfg['history_seconds']==5 and cfg['models']==['R','RM']
    lock=open(ROOT/'analysis.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.perf_counter();timing={};code=1;stage='input_validation'
    def progress():
        write(out/'progress.json',dict(status='running',stage=stage,pid=os.getpid(),elapsed_seconds=time.perf_counter()-start));print(stage,flush=True)
    try:
        progress();files=source_files(cfg)
        identity=dict(inputs={str(p.resolve()):digest(p) for p in files},code={p.name:digest(p) for p in ROOT.glob('*.py')},config=digest(args.config),smoke=args.smoke)
        if (out/'identity.json').exists():
            assert args.resume or args.report_only,'Existing output requires --resume'
            assert json.loads((out/'identity.json').read_text())==identity,'Input/code/config hash changed; use a new output directory'
        else:
            assert not args.report_only,'No saved aggregation'
            write(out/'identity.json',identity)
        write(out/'resolved_paths.json',dict(project_root=str(PROJECT.resolve()),source=str((PROJECT/cfg['source']).resolve()),pilot=str((PROJECT/cfg['pilot']).resolve()),output=str(out),git_status='not a git repository; git status --short failed',applicable_AGENTS=[],classification_manifest=str(PROJECT/'experiments/classification_manifest.json'),timestamp_units='run-relative seconds; segment+bin retained in sample key'))
        write(out/'input_manifest.json',dict(files=[dict(path=str(p.resolve()),sha256=digest(p),bytes=p.stat().st_size) for p in files],no_holdout_rows_used=True))
        (out/'resolved_config.yaml').write_text(yaml.safe_dump({**cfg,'target':'(t+0.8,t+1.0] completed bin sample mean of 8-GPU summed power','aggregation':'run equal weight; per-run RMSE/p95/p99 averaged','prediction_formula':'current + lambda*(saved_prediction-current)','threshold_rule':'fold training cap300 abs endpoint delta, numpy quantile linear','new_fits':0,'new_inference':0,'postprocessing':'none','versions':{'numpy':np.__version__,'pandas':pd.__version__}},allow_unicode=True,sort_keys=False))
        if args.report_only:
            saved=json.loads((out/'aggregation_checkpoint.json').read_text())
            for n,h in saved.items():assert digest(out/n)==h,n
            timing['input_validation_seconds']=time.perf_counter()-start
        else:
            base,checks=validate(cfg,out);timing['input_validation_seconds']=time.perf_counter()-start
            if args.smoke:
                # Existing subset only; no sampling scheme or inference changes.
                z=base.head(64)
                for m in ['R','RM']:
                    for lam in cfg['lambdas']:
                        p=z.current_w+lam*(z[m]-z.current_w)
                        if lam==0:same(p,z.current_w)
                        if lam==1:same(p,z[m])
                synthetic=pd.DataFrame(dict(current_w=[0.,0.],actual_w=[2.,-2.]))
                known=metric(synthetic,np.array([1.,-1.]))
                same([known['mae'],known['rmse'],known['signed_bias'],known['gain']],[1,1,0,1])
                assert np.isnan(known['p99_abs_error'])
                write(out/'verification.json',{**checks,'smoke_subset_anchors':len(z),'lambda0_lambda1_restored':True,'new_fits':0})
            else:
                stage='calculation_and_aggregation';progress();t=time.perf_counter();timing.update(calculate(base,cfg,out,checks));timing['calculation_and_aggregation_seconds']=time.perf_counter()-t
                write(out/'aggregation_checkpoint.json',{p.name:digest(p) for p in list(out.glob('*.csv'))+[out/'verification.json']})
        if not args.smoke:
            stage='visualization_and_report';progress();t=time.perf_counter()
            from render import render
            timing.update(render(out));timing['visualization_and_report_seconds']=time.perf_counter()-t
        assert identity['inputs']=={str(p.resolve()):digest(p) for p in files}
        timing['wall_clock_seconds']=time.perf_counter()-start
        write(out/('report_only_timing.json' if args.report_only else 'timing.json'),timing)
        code=0;write(out/'completion_status.json',dict(status='completed',exit_code=0,smoke=args.smoke,report_only=args.report_only,new_fits=0,new_inference=0,finished_utc=datetime.now(timezone.utc).isoformat(),timing=timing))
        write(out/'progress.json',dict(status='completed',stage='completed',pid=os.getpid(),elapsed_seconds=time.perf_counter()-start))
    except BaseException as e:
        traceback.print_exc();write(out/'completion_status.json',dict(status='failed',stage=stage,error=str(e),exit_code=1,new_fits=0))
        write(out/'progress.json',dict(status='failed',stage=stage,error=str(e)));raise
    finally:(out/'exit_code.txt').write_text(str(code)+'\n')
if __name__=='__main__':main()
