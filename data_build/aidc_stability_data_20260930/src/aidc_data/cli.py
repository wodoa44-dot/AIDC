from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .inventory import PROJECT, MANIFEST, audit, sha256
from .canonical import canonicalize, completed_bins
from .metadata import extract
from .events import detect
from .forecasting import build_views
from .splits import make

UTC=timezone.utc; KST=timezone(timedelta(hours=9))
def now():return datetime.now(UTC).isoformat()
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
def config():return yaml.safe_load((PROJECT/'configs/existing.yaml').read_text())
def code_hash():
    h=hashlib.sha256()
    for p in sorted((PROJECT/'src').rglob('*.py')):h.update(p.relative_to(PROJECT).as_posix().encode()+p.read_bytes())
    return h.hexdigest()
def signature():
    return {'source_manifest_sha256':sha256(MANIFEST),'config_sha256':sha256(PROJECT/'configs/existing.yaml'), 'code_sha256':code_hash()}
def read_status(out):
    p=out/'build_status.json'
    return json.loads(p.read_text()) if p.exists() else {}
def set_status(out,**kw):
    s=read_status(out);s.update(kw);s['updated_utc']=now();dump(out/'build_status.json',s)
def do_audit(out):
    m=audit(out);return m

def build_existing(out,smoke=0):
    cfg=config(); sig=signature(); m=do_audit(out)
    selected=m[m.available_for_d1].copy()
    if smoke:selected=selected.head(smoke)
    out.mkdir(parents=True,exist_ok=True)
    prior=read_status(out)
    if prior.get('signature') and prior['signature']!=sig:raise RuntimeError('Input/config/code changed: choose a new build_id; old cache cannot be resumed')
    set_status(out,signature=sig,started_utc=prior.get('started_utc',now()),started_kst=datetime.now(KST).isoformat(),
               build_existing='running',pid=os.getpid(),requested_run_count=len(selected),completed_run_count=prior.get('completed_run_count',0))
    (out/'canonical_runs').mkdir(exist_ok=True);(out/'bins').mkdir(exist_ok=True)
    quality=[];canpaths=[];binpaths=[]
    for j,r in enumerate(selected.itertuples(),1):
        cp=out/'canonical_runs'/f'{r.run_id}.parquet';bp=out/'bins'/f'{r.run_id}.parquet'
        if not(cp.exists() and bp.exists()):
            c,dt=canonicalize(r,cfg);b=completed_bins(c,dt,cfg)
            c.to_parquet(cp,index=False);b.to_parquet(bp,index=False)
        else:
            c=pd.read_parquet(cp);b=pd.read_parquet(bp);dt=float(c.elapsed_s.diff().loc[lambda x:x.gt(0)].median())
        quality.append({'run_id':r.run_id,'source_file':r.source_path,'source_rows':len(c),'median_dt_s':dt,
            'duplicate_or_reverse_timestamps':int(c.quality_flags.eq('nonmonotonic').sum()),
            'gap_starts':int(c.quality_flags.eq('gap_start').sum()),'missing_gpu_rows':int(c.gpu_power_w.isna().sum()),
            'segments':int(c.segment_id.nunique()),'bins_total':len(b),'bins_valid':int(b.valid.sum()),
            'bins_excluded_coverage':int((~b.valid).sum()),'sensor_bandwidth_verified':False,
            'timezone_verified':False})
        canpaths.append(cp);binpaths.append(bp)
        set_status(out,completed_run_count=j,heartbeat_utc=now(),last_run_id=r.run_id)
    pd.DataFrame(quality).to_csv(out/'quality_audit.csv',index=False)
    canonical=pd.concat([pd.read_parquet(p) for p in canpaths],ignore_index=True)
    canonical.to_parquet(out/'canonical_power.parquet',index=False)
    bins=pd.concat([pd.read_parquet(p) for p in binpaths],ignore_index=True)
    bins.to_parquet(out/'completed_bins_200ms.parquet',index=False)
    meta=extract(selected);meta.to_parquet(out/'metadata.parquet',index=False)
    ev=detect(bins,cfg);ev.to_parquet(out/'events.parquet',index=False)
    views=build_views(bins,cfg)
    # Event occurrence is a future label; publication time never enters features.
    index=views['forecast_index']; targets=views['targets']
    ev_groups={(run,int(seg)):np.sort(g.onset_time.to_numpy(dtype='datetime64[ns]').astype('int64')) for (run,seg),g in ev.groupby(['run_id','segment_id'])} if len(ev) else {}
    for H in [5,10,30]:
        occurred=[];first=[];censored=[]
        for r,energy in zip(index.itertuples(),targets[f'energy_above_{H}s_j'].to_numpy()):
            event_times=ev_groups.get((r.run_id,int(r.segment_id)),np.array([],dtype='int64'))
            t=int(r.issuance_time.value); j=np.searchsorted(event_times,t,side='right')
            upcoming=(event_times[j]-t)/1e9 if j<len(event_times) else np.nan
            has=bool(np.isfinite(upcoming) and upcoming<=H)
            occurred.append(has);first.append(upcoming if has else np.nan)
            censored.append(pd.isna(energy))
        targets[f'event_within_{H}s']=occurred;targets[f'first_transition_{H}s_s']=first;targets[f'event_label_right_censored_{H}s']=censored
    for name,frame in views.items():frame.to_parquet(out/f'{name}.parquet',index=False)
    split=make(views['forecast_index'],cfg['outer_folds'],cfg['inner_folds']);dump(out/'split_manifest.json',split)
    # Metadata is joined by run key only. Raw filename and identifiers never enter features.
    wide=meta[meta.field.isin(['workload_family','hardware_label','batch_size_token','image_size_token','model_size_token','sequence_length_token','zero_stage_token'])].pivot(index='run_id',columns='field',values='value').reset_index()
    ph=views['features_power_history'].merge(wide,on='run_id',how='left')
    ph=ph.drop(columns=['run_id','session_id','split_group_id','lineage_source_hash'])
    ph.to_parquet(out/'features_history_static_metadata.parquet',index=False)
    ph.to_parquet(out/'features_history_static_dynamic_metadata.parquet',index=False)
    dump(out/'dataset_manifest.json',{'schema_version':'1.0','build_id':out.name,'evidence_type':'derived','parent_evidence_type':'measured_existing',
        'measurement_boundary':'sum_of_8_gpu_channels','native_timestamp_median_s':'per-run quality_audit.csv',
        'comparison_bin_s':cfg['bin_seconds'],'sensor_bandwidth_verified':False,
        'target':'completed future 200ms bin average GPU power W','feature_availability':'assumed at completed bin end; arrival delay unmeasured',
        'files':{name:{'path':str(out/f'{name}.parquet'),'sha256':sha256(out/f'{name}.parquet')} for name in ['canonical_power','completed_bins_200ms','metadata','events','forecast_index','targets','features_current_power','features_power_history','features_history_static_metadata','features_history_static_dynamic_metadata']},
        'split_manifest':str(out/'split_manifest.json')})
    set_status(out,build_existing='completed',completed_run_count=len(selected),canonical_rows=len(canonical),valid_bins=int(bins.valid.sum()),
        forecast_rows=len(views['forecast_index']),event_count=len(ev),metadata_rows=len(meta),finished_utc=now())
    return out

def validate(out):
    manifest=json.loads((out/'dataset_manifest.json').read_text())
    idx=pd.read_parquet(out/'forecast_index.parquet');targets=pd.read_parquet(out/'targets.parquet');bins=pd.read_parquet(out/'completed_bins_200ms.parquet')
    checks={'index_unique':bool(idx.row_id.is_unique),'targets_unique':bool(targets.row_id.is_unique),
        'keys_match':set(idx.row_id)==set(targets.row_id),'feature_time_causal':bool((idx.max_feature_available_at<=idx.issuance_time).all()),
        'bin_coverage':bool((bins.loc[bins.valid,'coverage']>=.95-1e-8).all()),
        'measured_simulated_separate':not any(x in idx.columns for x in ['voltage_v','frequency_hz']),
        'no_interpolated_1ms_input':bool((bins.dataset_id=='highres_node').all()),
        'split_disjoint':True,'manifest_hashes':True}
    split=json.loads((out/'split_manifest.json').read_text())
    for f in split['outer_folds']:
        if set(f['train_groups'])&set(f['test_groups']):checks['split_disjoint']=False
    for f in manifest['files'].values():
        if sha256(Path(f['path']))!=f['sha256']:checks['manifest_hashes']=False
    report={'checks':checks,'passed':all(checks.values()),'checked_utc':now(),'limitations':['GPU sensor refresh and arrival delay unverified','session proxy does not establish independent experiments','source timestamps have no verified timezone']}
    dump(out/'validation.json',report);set_status(out,validation='completed' if report['passed'] else 'failed')
    if not report['passed']:raise RuntimeError(f'Validation failed: {checks}')
    return report

def status(out):
    print(json.dumps(read_status(out),ensure_ascii=False,indent=2));return read_status(out)

def main():
    a=argparse.ArgumentParser();a.add_argument('command',choices=['audit','build-existing','label','export','validate','prepare-phase0','collect','identify','prepare-grid','simulate-grid','report','all','status','resume'])
    a.add_argument('--build-id',default='main_20260929');a.add_argument('--smoke-runs',type=int,default=0);a.add_argument('--live',action='store_true');a.add_argument('--config')
    args=a.parse_args();out=PROJECT/'outputs'/args.build_id
    if args.command=='audit':do_audit(out)
    elif args.command in ['build-existing','all','resume']:
        build_existing(out,args.smoke_runs);validate(out)
        if args.command=='all':
            from .phase0.collect import prepare as prepare_phase0
            from .phase0.identify import identify
            from .grid.adapter import prepare as prepare_grid,simulate
            from .export import report
            prepare_phase0(out);identify(out);prepare_grid(out);simulate(out);report(out)
    elif args.command=='validate':validate(out)
    elif args.command=='status':status(out)
    elif args.command in ['label','export']:
        if not (out/'dataset_manifest.json').exists():raise RuntimeError('Run build-existing first')
        print('Already generated by build-existing:',out)
    elif args.command=='collect':
        from .phase0.collect import collect
        collect(out,args.live,args.config)
    elif args.command=='prepare-phase0':
        from .phase0.collect import prepare
        prepare(out)
    elif args.command=='identify':
        from .phase0.identify import identify
        identify(out)
    elif args.command in ['prepare-grid','simulate-grid']:
        from .grid.adapter import prepare,simulate
        prepare(out)
        if args.command=='simulate-grid':simulate(out)
    elif args.command=='report':
        from .export import report
        report(out)
    print(out)
if __name__=='__main__':main()
