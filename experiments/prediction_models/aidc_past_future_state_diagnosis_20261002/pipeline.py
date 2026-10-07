"""Saved OOF and completed-bin diagnosis; no model fitting or inference."""
import argparse, fcntl, importlib.util, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
SHRINK=ROOT.parent/'aidc_residual_shrinkage_diagnosis_20261001'
spec=importlib.util.spec_from_file_location('saved_shrinkage',SHRINK/'pipeline.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
KEY=old.KEY; digest=old.digest; write=old.write; same=old.same
CELLS=[p+'_'+f for p in ['low','high'] for f in ['small','medium','large']]
METRICS=['mae','rmse','signed_bias','p95_abs_error','p99_abs_error','gain','actual_abs_delta','predicted_abs_delta','direction_match']
MODELS=[('P',0),('A',1)]+[(m,l) for m in ['R','RM'] for l in [.25,.5,.75,1]]

def bins_join(anchors,raw):
    """Every offset joins by run/segment/bin, never by input row position."""
    z=anchors.copy().reset_index(drop=True)
    for off in range(-24,6):
        r=raw[KEY+['issuance_s','current_w']].copy();r['bin']-=off
        r=r.rename(columns={'issuance_s':f't_{off}','current_w':f'x_{off}'})
        z=z.merge(r,on=KEY,how='left',validate='one_to_one')
        missing=z[f'x_{off}'].isna()|z[f't_{off}'].isna()
        if missing.any():
            raise ValueError(f'Missing/nonfinite completed bins offset={off}, affected keys='+z.loc[missing,KEY].to_json(orient='records'))
        assert np.isfinite(z[[f'x_{off}',f't_{off}']]).all().all()
        same(z[f't_{off}'],z.issuance_s+.2*off)
    same(z.x_0,z.current_w);same(z.x_5,z.actual_w)
    for seconds,width in [(5,25),(1,5)]:
        a=z[[f'x_{off}' for off in range(-width+1,1)]].to_numpy()
        z[f'past_range_{seconds}s']=np.ptp(a,axis=1)
    assert (z[[f't_{off}' for off in range(-24,1)]].max(axis=1)<=z.issuance_s+1e-10).all()
    return z

def prepare(cfg,out):
    base,checks=old.validate(cfg,out)
    src=PROJECT/cfg['source']
    tr=pd.read_parquet(src/'nested_train_anchors.parquet',filters=[('cap','==',300)])
    pcfg=yaml.safe_load((PROJECT/cfg['pilot']).parent.parent.joinpath('config.yaml').read_text())
    rawpath=PROJECT/pcfg['prepared']
    runs=sorted(set(base.run_id)|set(tr.run_id))
    # Arrow filter excludes holdout runs before any bin rows are materialized.
    raw=pd.read_parquet(rawpath,filters=[('run_id','in',runs)],columns=KEY+['issuance_s','current_w'])
    assert set(raw.run_id)==set(runs) and not raw.duplicated(KEY).any()
    base=bins_join(base,raw);train=bins_join(tr,raw)
    thresholds=[]
    splits=json.loads((src/'split_manifest.json').read_text())['folds']
    for s in splits:
        g=train[train.collection_day.isin(s['train_days'])]
        q50,q90=np.quantile(abs(g.actual_w-g.current_w),[.5,.9],method='linear')
        thresholds.append(dict(fold=s['fold'],train_days=','.join(s['train_days']),evaluation_day=','.join(s['validation_days']),train_runs=g.run_id.nunique(),train_anchors=len(g),past_q50_5s=np.quantile(g.past_range_5s,.5,method='linear'),past_q50_1s=np.quantile(g.past_range_1s,.5,method='linear'),endpoint_q50=q50,endpoint_q90=q90,unit='W'))
    th=pd.DataFrame(thresholds);th.to_csv(out/'thresholds.csv',index=False)
    base=base.merge(th[['fold','past_q50_5s','past_q50_1s','endpoint_q50','endpoint_q90']],on='fold',validate='many_to_one')
    base['delta_future']=base.actual_w-base.current_w;base['abs_delta_future']=abs(base.delta_future)
    base['future_state']=np.where(base.abs_delta_future<=base.endpoint_q50,'small',np.where(base.abs_delta_future<=base.endpoint_q90,'medium','large'))
    for seconds in [5,1]:
        base[f'past_state_{seconds}s']=np.where(base[f'past_range_{seconds}s']<=base[f'past_q50_{seconds}s'],'low','high')
        base[f'cell_{seconds}s']=base[f'past_state_{seconds}s']+'_'+base.future_state
        assert base[f'cell_{seconds}s'].isin(CELLS).all()
        assert sum((base[f'cell_{seconds}s']==c).astype(int) for c in CELLS).eq(1).all()
    path=base[[f'x_{j}' for j in range(1,6)]].to_numpy()
    exceed=abs(path-base.current_w.to_numpy()[:,None])>base.endpoint_q90.to_numpy()[:,None]
    base['first_large_departure']=np.where(base.future_state=='large',np.argmax(exceed,axis=1)+1,0)
    assert exceed[base.future_state=='large'].any(axis=1).all()
    for j in range(1,6):
        ix=(base.future_state=='large')&(base.first_large_departure==j)
        assert exceed[ix,j-1].all() and not exceed[ix,:j-1].any()
    base.to_parquet(out/'anchor_states.parquet',index=False)
    train[KEY+['collection_day','past_range_5s','past_range_1s']].to_parquet(out/'training_ranges.parquet',index=False)
    checks.update(past_reference_max_time_le_issuance=True,target_exactly_five_bins=True,same_segment_continuous_bins=True,thresholds_training_only=True,six_cells_exclusive_exhaustive=True,all_models_same_membership=True,completed_bins_read_filter='run_id IN development training/evaluation runs; no holdout bin rows materialized',missing_history_anchors=0,partial_analysis=False)
    return base,checks

def metric(g,p):
    e=p-g.actual_w.to_numpy();a=abs(e);d=p-g.current_w.to_numpy();n=len(g)
    v=dict(n=n,mae=a.mean(),rmse=np.sqrt(np.mean(e**2)),signed_bias=e.mean(),p95_abs_error=np.quantile(a,.95) if n>=20 else np.nan,p99_abs_error=np.quantile(a,.99) if n>=100 else np.nan,gain=np.mean(g.abs_delta_future.to_numpy()-a),actual_abs_delta=g.abs_delta_future.mean(),predicted_abs_delta=np.mean(abs(d)))
    large=(g.future_state=='large').to_numpy()
    v['direction_match']=np.mean(np.sign(d[large])==np.sign(g.delta_future.to_numpy()[large])) if large.any() else np.nan
    return v

def aggregate(per,keys):
    gb=per.groupby(keys,dropna=False)
    z=gb[METRICS].mean().reset_index()
    c=gb.agg(n_runs=('run_id','nunique'),n_anchors=('n','sum'),valid_runs=('mae','count'),n_days=('collection_day','nunique'),p95_valid_runs=('p95_abs_error','count'),p99_valid_runs=('p99_abs_error','count'),direction_valid_runs=('direction_match','count'),small_n_runs=('small_n','sum'),improved_runs=('gain',lambda x:int((x>1e-9).sum())),tied_runs=('gain',lambda x:int((abs(x)<=1e-9).sum())),worsened_runs=('gain',lambda x:int((x< -1e-9).sum()))).reset_index()
    z=z.merge(c,on=keys)
    valid_days=per[per.n>0].groupby(keys,dropna=False).collection_day.nunique().reset_index(name='valid_n_days')
    z=z.drop(columns='n_days').merge(valid_days,on=keys,how='left').rename(columns={'valid_n_days':'n_days'})
    z['n_days']=z.n_days.fillna(0).astype(int)
    z['unit']='W; direction_match=fraction';return z

def calculate(base,out,checks,reference=True):
    rows=[];con=[];counts=[];compositions=[]
    for seconds in [5,1]:
        col=f'cell_{seconds}s'
        for run,g in base.groupby('run_id',sort=True):
            for c in CELLS:
                x=g[g[col]==c]
                counts.append(dict(history_seconds=seconds,cell=c,run_id=run,collection_day=g.collection_day.iloc[0],hardware=g.hardware.iloc[0],workload=g.workload.iloc[0],n=len(x),whole_n=len(g),cell_fraction=len(x)/len(g),small_n=0<len(x)<30))
            for model,lam in MODELS:
                p=g.current_w.to_numpy() if model=='P' else g.current_w.to_numpy()+lam*(g[model].to_numpy()-g.current_w.to_numpy())
                if model=='P':assert np.array_equal(p,g.current_w.to_numpy())
                if lam==1:same(p,g[model])
                masks={'whole':np.ones(len(g),bool),**{c:(g[col]==c).to_numpy() for c in CELLS}}
                for past in ['low','high']:
                    for direction,sign in [('rising',1),('falling',-1)]:
                        masks[past+'_large_'+direction]=(g[col].eq(past+'_large')&(np.sign(g.delta_future)==sign)).to_numpy()
                for j in range(1,6):masks[f'first_departure_{j}']=(g[col].eq('low_large')&g.first_large_departure.eq(j)).to_numpy()
                gain=g.abs_delta_future.to_numpy()-abs(g.actual_w.to_numpy()-p)
                for cell,mask in masks.items():
                    values=metric(g[mask],p[mask]) if mask.any() else dict(n=0,**{m:np.nan for m in METRICS})
                    rows.append(dict(history_seconds=seconds,model=model,**{'lambda':lam},cell=cell,run_id=run,collection_day=g.collection_day.iloc[0],small_n=0<mask.sum()<30,**values))
                    if cell in CELLS:con.append(dict(history_seconds=seconds,model=model,**{'lambda':lam},cell=cell,run_id=run,collection_day=g.collection_day.iloc[0],n=int(mask.sum()),whole_n=len(g),gain_contribution=gain[mask].sum()/len(g)))
    per=pd.DataFrame(rows);per.to_csv(out/'metrics_per_run_state.csv',index=False)
    summary=aggregate(per,['history_seconds','model','lambda','cell'])
    days=aggregate(per,['history_seconds','model','lambda','cell','collection_day'])
    summary.to_csv(out/'metrics_summary_state.csv',index=False);days.to_csv(out/'metrics_by_day_state.csv',index=False)
    summary[summary.cell.str.startswith('first_departure')].to_csv(out/'first_departure_metrics.csv',index=False)
    days[days.cell.str.startswith('first_departure')].to_csv(out/'first_departure_metrics_by_day.csv',index=False)
    summary[summary.cell.str.contains('large_rising|large_falling')].to_csv(out/'large_direction_metrics.csv',index=False)
    cnt=pd.DataFrame(counts);cnt.to_csv(out/'state_counts_per_run.csv',index=False)
    z=cnt.groupby(['history_seconds','cell']).agg(n_anchors=('n','sum'),n_runs=('n',lambda x:int((x>0).sum())),small_n_runs=('small_n','sum')).reset_index()
    nd=cnt[cnt.n>0].groupby(['history_seconds','cell']).collection_day.nunique()
    z['n_days']=[int(nd.get((r.history_seconds,r.cell),0)) for r in z.itertuples()]
    z.to_csv(out/'state_counts.csv',index=False)
    for keys,name in [(['collection_day'],'day'),(['hardware','workload'],'hardware_workload'),(['collection_day','hardware','workload'],'day_hardware_workload')]:
        cnt.groupby(['history_seconds','cell']+keys).agg(n_anchors=('n','sum'),n_runs=('n',lambda x:int((x>0).sum())),small_n_runs=('small_n','sum')).reset_index().to_csv(out/f'state_composition_{name}.csv',index=False)
    dc=pd.DataFrame(con);dc.to_csv(out/'gain_contributions_per_run.csv',index=False)
    d=dc.groupby(['history_seconds','model','lambda','cell']).agg(gain_contribution=('gain_contribution','mean'),n_anchors=('n','sum'),n_runs=('run_id','nunique'),valid_cell_runs=('n',lambda x:int((x>0).sum()))).reset_index();d['unit']='W';d.to_csv(out/'gain_contributions.csv',index=False)
    joined=dc.groupby(['history_seconds','model','lambda','run_id']).gain_contribution.sum().reset_index().merge(per[per.cell=='whole'],on=['history_seconds','model','lambda','run_id'])
    same(joined.gain_contribution,joined.gain)
    effects=[]
    for scope,table,keys in [('state',summary,['history_seconds','lambda','cell']),('day_state',days,['history_seconds','lambda','cell','collection_day'])]:
        j=table[table.model=='R'].merge(table[table.model=='RM'],on=keys,suffixes=('_R','_RM'),validate='one_to_one')
        e=j[keys].copy();e['scope']=scope
        for m in METRICS:e[m+'_effect']=j[m+'_R']-j[m+'_RM']
        for c in ['n_anchors','valid_runs','p95_valid_runs','p99_valid_runs','small_n_runs']:
            assert (j[c+'_R']==j[c+'_RM']).all();e[c]=j[c+'_R']
        effects.append(e)
        # The lambda-zero performance is stored once as P, while the paired
        # metadata effect is known to be exactly zero wherever metrics exist.
        p=table[table.model=='P'];e=p[keys].copy();e['scope']=scope
        for m in METRICS:e[m+'_effect']=np.where(p[m].notna(),0.,np.nan)
        for c in ['n_anchors','valid_runs','p95_valid_runs','p99_valid_runs','small_n_runs']:e[c]=p[c]
        effects.append(e)
    pd.concat(effects).to_csv(out/'metadata_effect.csv',index=False)
    summary[summary.cell.isin(CELLS)].to_csv(out/'sensitivity_comparison.csv',index=False)
    # All model/coefficients have exactly the same valid run sets and membership.
    assert per.groupby(['history_seconds','cell','run_id']).n.nunique().eq(1).all()
    assert per.loc[per.n==0,METRICS].isna().all().all()
    assert per.loc[per.n<20,'p95_abs_error'].isna().all()
    assert per.loc[per.n<100,'p99_abs_error'].isna().all()
    maxdiff=0.
    if reference:
        ref=pd.read_csv(SHRINK/'outputs/full_v1/metrics_summary.csv')
        refper=pd.read_csv(SHRINK/'outputs/full_v1/metrics_per_run.csv')
        for model,lam in MODELS:
            refmodel='R' if model=='P' else model
            a=summary[(summary.history_seconds==5)&(summary.cell=='whole')&(summary.model==model)&(summary['lambda']==lam)].iloc[0]
            b=ref[(ref.model==refmodel)&(ref['lambda']==lam)].iloc[0]
            for m in METRICS[:6]:same(a[m],b[m]);maxdiff=max(maxdiff,abs(a[m]-b[m]))
            a=per[(per.history_seconds==5)&(per.cell=='whole')&(per.model==model)&(per['lambda']==lam)].merge(refper[(refper.model==refmodel)&(refper['lambda']==lam)],on='run_id',suffixes=('','_old'))
            for m in METRICS[:6]:same(a[m],a[m+'_old'])
    checks.update(lambda0_equals_persistence=True,lambda1_equals_saved=True,cell_contribution_sum_matches_whole=True,max_contribution_error_w=float(abs(joined.gain_contribution-joined.gain).max()),whole_metrics_match_shrinkage=reference,max_reference_difference_w=float(maxdiff),same_valid_run_sets_all_models=True,p95_minimum_n20=True,p99_minimum_n100=True,empty_cells_NA=True,new_fits=0,new_inference=0)
    write(out/'verification.json',checks)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    for f in ['smoke','resume','report-only']:ap.add_argument('--'+f,action='store_true')
    a=ap.parse_args();cfg=yaml.safe_load(a.config.read_text());out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    assert cfg['lambdas']==[0,.25,.5,.75,1]
    lock=open(ROOT/'analysis.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.perf_counter();timing={};code=1;stage='input_validation'
    def progress():
        write(out/'progress.json',dict(status='running',stage=stage,pid=os.getpid(),elapsed_seconds=time.perf_counter()-start));print(stage,flush=True)
    try:
        progress()
        pcfg=yaml.safe_load((PROJECT/cfg['pilot']).parent.parent.joinpath('config.yaml').read_text())
        files=old.source_files(cfg)+[SHRINK/'pipeline.py',SHRINK/'outputs/full_v1/metrics_summary.csv',SHRINK/'outputs/full_v1/metrics_per_run.csv',PROJECT/pcfg['prepared']]
        identity=dict(inputs={str(p.resolve()):digest(p) for p in files},code={p.name:digest(p) for p in ROOT.glob('*.py')},config=digest(a.config),smoke=a.smoke)
        if (out/'identity.json').exists():
            assert a.resume or a.report_only,'Use --resume for existing output'
            assert json.loads((out/'identity.json').read_text())==identity,'Input/code/config changed: use a new output directory'
        else:
            assert not a.report_only;write(out/'identity.json',identity)
        write(out/'resolved_paths.json',dict(project_root=str(PROJECT),source=str((PROJECT/cfg['source']).resolve()),original_bins=str((PROJECT/pcfg['prepared']).resolve()),shrinkage=str(SHRINK.resolve()),output=str(out),applicable_AGENTS=[],git_status='not a git repository; verified git status --short',reference_paths={str(p):p.exists() for p in [ROOT.parent/'aidc_residual_metadata_followup_20261001',SHRINK,ROOT.parent/'aidc_forecast_pilot_20261001',PROJECT/'experiments/aidc_forecast_pilot_20261001']}))
        write(out/'input_manifest.json',dict(files=[dict(path=str(p.resolve()),sha256=identity['inputs'][str(p.resolve())],bytes=p.stat().st_size) for p in files],holdout_power_rows_loaded=0))
        (out/'resolved_config.yaml').write_text(yaml.safe_dump({**cfg,'primary_history_seconds':5,'sensitivity_history_seconds':1,'past_threshold':'unweighted linear quantile q50 of original cap300 training anchors for each fold','future_threshold':'existing training endpoint q50/q90','p95_minimum_n':20,'p99_minimum_n':100,'small_n_threshold':30,'target':'x[k+5], completed bin (t+0.8,t+1.0]','history_1s':'5 bins cover 1s; first-to-last end-time difference 0.8s','aggregation':'equal run weight within nonempty cell; contribution denominator whole run; empty contribution zero','new_fits':0,'new_inference':0},allow_unicode=True,sort_keys=False))
        if a.report_only:
            for n,h in json.loads((out/'aggregation_checkpoint.json').read_text()).items():assert digest(out/n)==h
        else:
            base,checks=prepare(cfg,out);timing['input_validation_seconds']=time.perf_counter()-start
            stage='smoke' if a.smoke else 'calculation';progress();t=time.perf_counter()
            if a.smoke:
                subset=base.groupby('run_id',group_keys=False).head(8).reset_index(drop=True)
                calculate(subset,out,checks,reference=False)
                checks=json.loads((out/'verification.json').read_text());checks['smoke_subset_anchors']=len(subset);write(out/'verification.json',checks)
            else:calculate(base,out,checks)
            timing['calculation_seconds']=time.perf_counter()-t
            write(out/'aggregation_checkpoint.json',{p.name:digest(p) for p in list(out.glob('*.csv'))+[out/'verification.json',out/'anchor_states.parquet']})
        if not a.smoke:
            stage='figures_and_report';progress();t=time.perf_counter()
            from render import render
            render(out);timing['figures_and_report_seconds']=time.perf_counter()-t
        assert identity['inputs']=={str(p.resolve()):digest(p) for p in files},'Input changed during analysis'
        timing['wall_clock_seconds']=time.perf_counter()-start;write(out/('report_only_timing.json' if a.report_only else 'timing.json'),timing)
        code=0;write(out/'completion_status.json',dict(status='completed',exit_code=0,smoke=a.smoke,report_only=a.report_only,new_fits=0,new_inference=0,finished_utc=datetime.now(timezone.utc).isoformat(),timing=timing));write(out/'progress.json',dict(status='completed',stage='completed',pid=os.getpid()))
    except BaseException as e:
        traceback.print_exc();write(out/'completion_status.json',dict(status='failed',exit_code=1,stage=stage,error=str(e)));write(out/'progress.json',dict(status='failed',stage=stage,error=str(e)));raise
    finally:(out/'exit_code.txt').write_text(str(code)+'\n')
if __name__=='__main__':main()
