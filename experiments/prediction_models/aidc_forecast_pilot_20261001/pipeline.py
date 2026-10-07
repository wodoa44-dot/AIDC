"""Bounded sequential prediction-only pilot. No controller imports or simulation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from data import (META,KEY,digest,frame_hash,atomic_json,parquet,prepare,encode,columns,weights,metrics,summarize,prediction_frame,spaced)

ROOT=Path(__file__).resolve().parent
PARAMS=dict(objective='quantile',alpha=.5,n_estimators=100,num_leaves=15,learning_rate=.07,min_child_samples=30,reg_alpha=0,reg_lambda=0,max_depth=-1,colsample_bytree=1.,subsample=1.,subsample_freq=0,n_jobs=4,random_state=17,verbosity=-1)


def project_root():
    candidates=[Path.cwd(),*Path.cwd().parents,ROOT,*ROOT.parents]
    for p in candidates:
        if (p/'experiments/classification_manifest.json').is_file() and (p/'experiments/aidc_grid_metadata_20260929').is_dir():return p
    raise RuntimeError('Verified AIDC project root unavailable')


def resolve(config):
    project=project_root();cfg=yaml.safe_load(config.read_text())
    for k in ['prepared','group_manifest','nested_models','legacy_src']:
        relative=Path(cfg[k]);assert not relative.is_absolute(),(k,'expected project-relative path')
        cfg[k]=str((project/relative).resolve());assert Path(cfg[k]).exists(),cfg[k]
    cfg['project_root']=str(project);cfg['pilot_root']=str(ROOT)
    return cfg


def aggregate(out):
    pieces=[pd.read_parquet(p/'predictions.parquet') for p in sorted((out/'fits').glob('*')) if (p/'complete.json').exists()]
    frames=[];seen=set()
    for d in pieces:
        frames.append(d)
        for scope,g in d.groupby('scope'):
            key=(scope,int(g.fold.iloc[0]) if str(g.fold.iloc[0])!='final' else 'final')
            if key not in seen:
                persistence=g.copy();persistence['model']='persistence';persistence['prediction_w']=persistence.current_w
                frames.append(persistence);seen.add(key)
    pred=pd.concat(frames,ignore_index=True)
    # Every model and persistence share original sample keys for their evaluation partition.
    whole=pred[(pred.region=='whole') & (pred.postprocess=='raw')]
    for (scope,fold),g in whole.groupby(['scope','fold']):
        ref=g[g.model=='persistence'][KEY+['actual_w']].sort_values(KEY).reset_index(drop=True)
        for model,x in g.groupby('model'):pd.testing.assert_frame_equal(ref,x[KEY+['actual_w']].sort_values(KEY).reset_index(drop=True))
    parquet(out/'pilot_predictions.parquet',pred)
    per=metrics(pred);per.to_csv(out/'per_run_metrics.csv',index=False)
    summary=summarize(per);summary.to_csv(out/'aggregate_metrics.csv',index=False)
    days=per.groupby(['scope','model','collection_day','region','postprocess'])[['mae','p95_abs_error','mae_gain_w']].mean().reset_index()
    days.to_csv(out/'per_day_metrics.csv',index=False)
    clipped=whole.assign(clipped=whole.prediction_w<0).groupby(['scope','model']).agg(negative_prediction_fraction=('clipped','mean'),n_samples=('clipped','size')).reset_index()
    clipped.to_csv(out/'clipping_audit.csv',index=False)
    return summary


def fit(cfg,out,train,evaluation,target,history,metadata,fold,scope,log,smoke=False):
    name=f'{target}_h{history}'+('_metadata' if metadata else '')
    folder=out/'fits'/f'{scope}_{fold}_{name}';folder.mkdir(parents=True,exist_ok=True)
    ident={'train_anchor_hash':frame_hash(train[KEY]),'evaluation_anchor_hash':frame_hash(evaluation[KEY]),'target':target,'history':history,'metadata':metadata,'scope':scope,'fold':fold}
    done=folder/'complete.json'
    if done.exists():
        rec=json.loads(done.read_text());assert rec['identity']==ident
        for file,checksum in rec['files'].items():assert digest(folder/file)==checksum
        log('fit_reused',name=name,fold=fold)
        return pd.read_parquet(folder/'predictions.parquet')
    cols=columns(history,metadata);x,xv,maps,diagnostics=encode(train,evaluation,cols)
    y=train.actual_w.to_numpy()-(train.current_w.to_numpy() if target=='residual' else 0)
    if target=='residual':np.testing.assert_allclose(y+train.current_w,train.actual_w,atol=1e-8)
    w=weights(train);params={**PARAMS,'n_jobs':cfg['threads']}
    if smoke:params['n_estimators']=5
    model=lgb.LGBMRegressor(**params)
    log('fit_start',name=name,fold=fold,train_rows=len(train),evaluation_rows=len(evaluation),train_runs=train.run_id.nunique(),evaluation_runs=evaluation.run_id.nunique(),features=len(cols),anchor_hash=ident['train_anchor_hash'],train_runs_by_day=train.drop_duplicates('run_id').groupby('collection_day').size().to_dict(),evaluation_runs_by_day=evaluation.drop_duplicates('run_id').groupby('collection_day').size().to_dict())
    started=time.perf_counter();model.fit(x,y,sample_weight=w);p=model.predict(xv)
    if target=='residual':p+=evaluation.current_w.to_numpy()
    assert np.isfinite(p).all()
    thresholds=np.quantile(abs(train.actual_w-train.current_w),[.5,.9])
    d=prediction_frame(evaluation,p,name,scope,*thresholds);d['fold']=str(fold)
    model.booster_.save_model(str(folder/'q50.txt'));parquet(folder/'predictions.parquet',d)
    pd.DataFrame(diagnostics).assign(model=name,fold=fold,scope=scope).to_csv(folder/'encoding_diagnostics.csv',index=False)
    parquet(folder/'training_weights.parquet',train[KEY].assign(sample_weight=w))
    record={'identity':ident,'train_rows':len(train),'evaluation_rows':len(evaluation),'train_runs':train.run_id.nunique(),'evaluation_runs':evaluation.run_id.nunique(),'feature_columns':cols,'categorical_encoding':maps,'threshold_q50_w':float(thresholds[0]),'threshold_q90_w':float(thresholds[1]),'fit_prediction_seconds':time.perf_counter()-started,'parameters':model.get_params(),'booster_parameters':model.booster_.params,'postprocessing':'primary raw; zero clipping only diagnostic','files':{n:digest(folder/n) for n in ['q50.txt','predictions.parquet','training_weights.parquet','encoding_diagnostics.csv']}}
    atomic_json(done,record);log('fit_completed',name=name,fold=fold,seconds=record['fit_prediction_seconds'])
    return d


def legacy_audit(cfg,out,evaluation,scope,log):
    dest=out/f'legacy_audit_{scope}';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        for name,h in json.loads((dest/'complete.json').read_text())['files'].items():assert digest(dest/name)==h
        return
    sys.path.insert(0,cfg['legacy_src'])
    from aidc_exp.training import encode as old_encode
    raw=pd.read_parquet(cfg['prepared']);groups=pd.read_csv(cfg['group_manifest'])
    raw['group_id']=raw.run_id.map(groups.set_index('run_id').assumed_group_id)
    indexed=raw.set_index(KEY);split=json.loads((Path(cfg['nested_models'])/'split_manifest.json').read_text())['folds']
    rows=[];detail=[];availability=[];provenance=[]
    for s in split:
        folder=Path(cfg['nested_models'])/'predictor_cache'/f'outer{s["fold"]}'/'outer_test'
        try:
            rec=json.loads((folder/'provenance.json').read_text());check=json.loads((folder/'complete.json').read_text())
            assert not set(rec['train_runs'])&set(rec['validation_runs'])
            assert digest(folder/'predictions.parquet')==check['prediction_hash']
            saved=pd.read_parquet(folder/'predictions.parquet')
            ev=evaluation[evaluation.run_id.isin(rec['validation_runs'])]
            if ev.empty:continue
            # Original old models have 30s eligibility; only exactly matching stored sample keys audited.
            overlap=ev.merge(saved[KEY+['P2','P4','actual_w']].rename(columns={'actual_w':'stored_actual_w'}),on=KEY,validate='one_to_one')
            availability.append({'fold':s['fold'],'requested_anchors':len(ev),'matched_anchors':len(overlap),'reason':'original validation sample overlap; old 30s eligibility differs from pilot 10s'})
            if overlap.empty:continue
            np.testing.assert_allclose(overlap.actual_w,overlap.stored_actual_w,atol=1e-7)
            va=indexed.loc[pd.MultiIndex.from_frame(overlap[KEY])].reset_index()
            ids=pd.read_parquet(folder/'training_sample.parquet');train=indexed.loc[pd.MultiIndex.from_frame(ids[KEY])].reset_index();train['sample_weight']=ids.sample_weight.to_numpy()
            assert sorted(train.run_id.unique())==rec['train_runs']
            for model in ['P2','P4']:
                x,xv=old_encode(train,va,model,rec['hyperparameters']['history_s'])
                schema=json.loads((folder/f'{model}_feature_schema.json').read_text())
                assert list(x.columns)==schema['columns']
                h=__import__('hashlib').sha256(pd.util.hash_pandas_object(x,index=True).values.tobytes()).hexdigest()
                assert h==schema['training_feature_hash'],'legacy training encoding mismatch'
                outputs=[];model_hashes={}
                for q in rec['quantiles']:
                    path=folder/f'{model}_q{q:g}.txt';sha=digest(path);assert sha==json.loads(path.with_suffix('.json').read_text())['sha256']
                    outputs.append(lgb.Booster(model_file=str(path)).predict(xv,num_threads=cfg['threads']));model_hashes[path.name]=sha
                mat=np.column_stack(outputs);original=mat[:,rec['quantiles'].index(.5)];sorted_q=np.sort(mat,axis=1)[:,rec['quantiles'].index(.5)]
                np.testing.assert_allclose(sorted_q,overlap[model],atol=1e-7,rtol=1e-10)
                z=overlap[KEY+['collection_day','actual_w','current_w']].copy();z['legacy_model']=model;z['original_q50_w']=original;z['sorted_q50_w']=sorted_q;z['q50_change_w']=sorted_q-original
                detail.append(z)
                for run,g in z.groupby('run_id'):
                    for name,p in [('persistence',g.current_w),('raw_q50',g.original_q50_w),('sorted_q50',g.sorted_q50_w)]:
                        e=p-g.actual_w
                        rows.append({'scope':scope,'legacy_model':model,'run_id':run,'collection_day':g.collection_day.iloc[0],'output':name,'n':len(g),'mae':float(abs(e).mean()),'signed_bias':float(e.mean()),'p95_abs_error':float(np.quantile(abs(e),.95)),'q50_changed_fraction':float((abs(g.q50_change_w)>1e-7).mean()),'mean_abs_q50_change_w':float(abs(g.q50_change_w).mean()),'max_abs_q50_change_w':float(abs(g.q50_change_w).max()),'generalization_label':'existing leave-day-out audit; reused evaluation; old training may include pilot holdout date'})
                provenance.append({'fold':s['fold'],'model':model,'model_hashes':model_hashes,'training_run_overlap_with_audit':False,'training_feature_hash_verified':h,'train_days':rec['train_groups'],'validation_days':rec['validation_groups']})
        except FileNotFoundError as e:
            availability.append({'fold':s['fold'],'requested_anchors':0,'matched_anchors':0,'reason':'unavailable: '+str(e)})
    pd.DataFrame(rows).to_csv(dest/'per_run.csv',index=False);pd.DataFrame(availability).to_csv(dest/'availability.csv',index=False)
    if detail:parquet(dest/'anchor_outputs.parquet',pd.concat(detail,ignore_index=True))
    if rows:
        d=pd.DataFrame(rows);d.groupby(['legacy_model','output'])[['mae','signed_bias','p95_abs_error','q50_changed_fraction','mean_abs_q50_change_w','max_abs_q50_change_w']].mean().reset_index().to_csv(dest/'summary.csv',index=False)
    atomic_json(dest/'provenance.json',{'partitions':provenance,'no_new_quantile_fits':True})
    atomic_json(dest/'complete.json',{'files':{p.name:digest(p) for p in dest.iterdir() if p.is_file()}})
    log('legacy_audit_completed',scope=scope,run_metric_rows=len(rows),status='available' if rows else 'unavailable')


def sampling_regions(data,train,cfg,out,splits):
    candidates=pd.concat([spaced(g) for _,g in data[data.valid_anchor & data.run_id.isin(train.run_id)].groupby(['run_id','segment'])],ignore_index=True)
    rows=[]
    for split in splits:
        tr=train[train.collection_day.isin(split['train_days'])];q50,q90=np.quantile(abs(tr.actual_w-tr.current_w),[.5,.9])
        for name,source in [('before_sampling',candidates[candidates.collection_day.isin(split['train_days'])]),('selected_training',tr)]:
            for run,g in source.groupby('run_id'):
                delta=g.actual_w-g.current_w
                for region,mask in [('normal',abs(delta)<=q50),('rising',delta>q50),('falling',delta < -q50),('large_endpoint_change',abs(delta)>q90)]:
                    rows.append({'fold':split['fold'],'run_id':run,'sampling':name,'region':region,'n':len(g),'region_n':int(mask.sum()),'fraction':float(mask.mean()),'threshold_q50_w':q50,'threshold_q90_w':q90})
    pd.DataFrame(rows).to_csv(out/'sampling_region_audit.csv',index=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'config.yaml');ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--report-only',action='store_true');args=ap.parse_args()
    cfg=resolve(args.config);out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lockdir=ROOT/'outputs/.locks';lockdir.mkdir(parents=True,exist_ok=True)
    cfg_hash=__import__('hashlib').sha256(json.dumps({**cfg,'smoke':args.smoke},sort_keys=True).encode()).hexdigest()
    lock=open(lockdir/f'{cfg_hash}.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();stage='initialization';code=1
    def log(event,**kw):print(json.dumps({'event':event,'stage':stage,'elapsed_s':round(time.time()-started,3),**kw},ensure_ascii=False),flush=True)
    def state(status,**kw):atomic_json(out/'progress.json',{'status':status,'stage':stage,'pid':os.getpid(),'elapsed_s':time.time()-started,**kw})
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        log('resolved_paths',project_root=cfg['project_root'],pilot_root=str(ROOT),data_path=cfg['prepared'],reference_models=cfg['nested_models'],output_dir=str(out),python=sys.executable)
        required=['experiments/aidc_grid_metadata_20260929','experiments/aidc_control_fair_budget_20260930','experiments/aidc_forecast_diagnosis_20261001','experiments/control_models','experiments/prediction_models','experiments/classification_manifest.json','experiments/README.md']
        atomic_json(out/'workspace_audit.json',{'project_root':cfg['project_root'],'missing_required_paths':[x for x in required if not (Path(cfg['project_root'])/x).exists()],'git_status':'root is not a Git repository','applicable_AGENTS':'none found in workspace or ancestors','control_models_modified':False})
        # All referenced legacy artifact identities, not just matching filenames.
        source_files=[Path(cfg['prepared']),Path(cfg['group_manifest']),Path(cfg['nested_models'])/'split_manifest.json',Path(cfg['legacy_src'])/'aidc_exp/training.py',Path(cfg['legacy_src'])/'aidc_exp/features.py']
        for folder in (Path(cfg['nested_models'])/'predictor_cache').glob('outer*/outer_test'):
            source_files+=list(folder.glob('*'))
        identity={'config_hash':cfg_hash,'code':{p.name:digest(p) for p in ROOT.glob('*.py')},'inputs':{str(p):digest(p) for p in source_files if p.is_file()},'lightgbm_version':lgb.__version__}
        if (out/'identity.json').exists():
            assert args.resume or args.report_only,'Existing output requires --resume'
            assert json.loads((out/'identity.json').read_text())==identity,'Input/code/config identity mismatch'
        else:
            assert not args.report_only,'Report-only requires completed source output'
            atomic_json(out/'identity.json',identity)
        resolved={**cfg,'output_dir':str(out),'python':sys.executable,'lightgbm_parameters':lgb.LGBMRegressor(**PARAMS).get_params(),'no_early_stopping':True,'remaining_lightgbm_defaults':'installed LightGBM '+lgb.__version__,'source_config':str(args.config.resolve())}
        (out/'resolved_config.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False,allow_unicode=True))
        if args.report_only:
            stage='report';state('running');aggregate(out)
            from report import render
            render(cfg,out);code=0
        else:
            stage='sampling';state('running')
            if args.resume and (out/'split_manifest.json').exists():
                sampling_checkpoint=json.loads((out/'sampling_checkpoint.json').read_text())
                for name,sha in sampling_checkpoint.items():assert digest(out/name)==sha,'Sampling checkpoint changed: '+name
                data=pd.read_parquet(out/'continuous_features.parquet');train=pd.read_parquet(out/'train_anchors.parquet');ev=pd.read_parquet(out/'evaluation_anchors.parquet');inventory=pd.read_csv(out/'run_inventory.csv');splits=json.loads((out/'split_manifest.json').read_text())['folds']
            else:
                data,train,ev,inventory,splits=prepare(cfg,out)
                atomic_json(out/'sampling_checkpoint.json',{name:digest(out/name) for name in ['continuous_features.parquet','train_anchors.parquet','evaluation_anchors.parquet','run_inventory.csv','sampling_summary.csv','excluded_runs.csv','split_manifest.json','feature_schema.json']})
            assert not set(train.collection_day)&{cfg['holdout_day']}
            log('sampling_completed',original_bins=int(inventory.original_bins.sum()),valid_anchors=int(inventory.valid_anchors.sum()),spaced_candidates=int(inventory.spaced_candidates.sum()),training_rows=len(train),evaluation_rows=len(ev),included_runs=int(inventory.included.sum()),excluded_runs=int((~inventory.included).sum()),common_train_anchor_hash=frame_hash(train[KEY]),common_evaluation_anchor_hash=frame_hash(ev[KEY]))
            if args.smoke:
                stage='smoke';tr=train[train.collection_day.isin(splits[0]['train_days'])].head(100);va=ev[ev.collection_day.isin(splits[0]['validation_days'])].head(100)
                fit(cfg,out,tr,va,'absolute',1,False,0,'smoke',log,True);code=0
            else:
                stage='legacy_development_audit';state('running');legacy_audit(cfg,out,ev[ev.collection_day.isin(cfg['development_days'])],'development',log)
                sampling_regions(data,train,cfg,out,splits)
                stage='target_history_development';state('running')
                for s in splits[:3]:
                    tr=train[train.collection_day.isin(s['train_days'])];va=ev[ev.collection_day.isin(s['validation_days'])]
                    assert not set(tr.run_id)&set(va.run_id)
                    for target in ['absolute','residual']:
                        for history in cfg['histories']:fit(cfg,out,tr,va,target,history,False,s['fold'],'development',log)
                    state('running',completed_development_fold=s['fold'])
                agg=aggregate(out);selection=agg[(agg.scope=='development')&(agg.region=='whole')&(agg.postprocess=='raw')&(agg.model!='persistence')].copy()
                selection['history']=selection.model.str.extract(r'_h(\d+)').astype(int);selection['target_order']=selection.model.str.startswith('residual').astype(int)
                best=selection.sort_values(['mae','p95_abs_error','history','target_order']).iloc[0]
                selected={'model':best.model,'target':best.model.split('_')[0],'history':int(best.history),'rule':'OOF equal-run MAE, p95 absolute error, shorter history, absolute target; exact numeric ties','selection_scope':'development only','development_mae':float(best.mae),'development_p95':float(best.p95_abs_error)}
                if (out/'selected_configuration.json').exists():assert json.loads((out/'selected_configuration.json').read_text())==selected
                else:atomic_json(out/'selected_configuration.json',selected)
                log('configuration_frozen',**selected)
                stage='metadata_development';state('running')
                for s in splits[:3]:fit(cfg,out,train[train.collection_day.isin(s['train_days'])],ev[ev.collection_day.isin(s['validation_days'])],selected['target'],selected['history'],True,s['fold'],'development',log)
                stage='final_holdout';state('running')
                s=splits[3];tr=train;va=ev[ev.collection_day==cfg['holdout_day']]
                assert len(va) and not set(tr.run_id)&set(va.run_id)
                configs=[('absolute',1,False),(selected['target'],selected['history'],False),(selected['target'],selected['history'],True)]
                for target,history,meta in dict.fromkeys(configs):fit(cfg,out,tr,va,target,history,meta,'final','holdout',log)
                legacy_audit(cfg,out,va,'holdout',log)
                stage='aggregation_report';state('running');aggregate(out)
                records=[json.loads(p.read_text()) for p in (out/'fits').glob('*/complete.json')]
                assert len(records)<=24
                atomic_json(out/'fit_summary.json',{'new_q50_fits':len(records),'development_target_history_fits':18,'development_metadata_fits':3,'final_unique_fits':len(dict.fromkeys(configs)),'fit_prediction_seconds':sum(r['fit_prediction_seconds'] for r in records),'final_training_rows':len(tr),'final_training_runs':tr.run_id.nunique(),'maximum_training_rows_per_run':int(tr.groupby('run_id').size().max()),'max_trees':100,'smoke_excluded':True})
                pd.concat([pd.read_csv(p) for p in (out/'fits').glob('*/encoding_diagnostics.csv')],ignore_index=True).to_csv(out/'encoding_diagnostics.csv',index=False)
                from report import render
                render(cfg,out);code=0
        state('completed');atomic_json(out/'completion_status.json',{'status':'completed','exit_code':code,'elapsed_s':time.time()-started,'smoke':args.smoke,'report_only':args.report_only,'independent_group_verification':'unverified_not_required_for_computations','no_control_simulation':True});log('pipeline_completed')
    except BaseException as e:
        code=int(e.code) if isinstance(e,SystemExit) and isinstance(e.code,int) else 1
        state('failed',error=str(e));atomic_json(out/'completion_status.json',{'status':'failed','failed_stage':stage,'exit_code':code,'error':str(e)});traceback.print_exc()
    finally:(out/'exit_code.txt').write_text(str(code)+'\n')
    raise SystemExit(code)


if __name__=='__main__':main()
