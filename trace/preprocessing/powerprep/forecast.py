"""Forecast indices only: no model fitting, no event-derived predictors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .contract import digest, save_table, atomic_json


def split_assignments(manifest, quality, cfg):
    canonical=manifest.loc[~manifest.is_duplicate].copy()
    sizes=quality.set_index('run_id').native_samples.to_dict()
    fractions=cfg['fractions']
    if len(fractions)!=3 or not np.isclose(sum(fractions),1) or min(fractions)<=0:
        raise ValueError('Three positive split fractions summing to 1 are required')
    imported=pd.read_csv(cfg['import_manifest']) if cfg.get('import_manifest') else None
    if imported is not None:
        if imported.groupby(['protocol','group_id']).split.nunique().max()>1:
            raise ValueError('Imported manifest separates aliases/source groups')
    rows=[]
    for dataset, ds in canonical.groupby('dataset'):
        groups=sorted(ds.group_id.unique(),key=lambda g:digest([cfg['seed'],g]))
        n=len(groups); ntr=max(1,int(n*fractions[0])); nval=max(1,int(n*fractions[1]))
        groupmap={g:('train' if i<ntr else 'validation' if i<ntr+nval else 'test') for i,g in enumerate(groups)}
        groupcounts=ds.groupby('group_id').run_id.nunique()
        for r in ds.itertuples():
            for protocol in cfg['protocols']:
                record=dict(dataset_id=dataset,run_id=r.run_id,group_id=r.group_id,protocol=protocol,
                            split='',support_status='supported',reason='',
                            cut1_row=int(sizes[r.run_id]*fractions[0]),
                            cut2_row=int(sizes[r.run_id]*(fractions[0]+fractions[1])))
                if protocol=='within_run':
                    record['split']='chronological'
                    if groupcounts[r.group_id]>1:
                        record.update(support_status='unsupported',reason='shared acquisition group: absolute alignment/overlap isolation not established')
                elif protocol=='held_out_group':
                    if n<3 or ntr+nval>=n:
                        record.update(support_status='unsupported',reason='fewer than three usable source groups')
                    else: record['split']=groupmap[r.group_id]
                else: raise ValueError(f'Unknown split protocol {protocol}')
                if imported is not None:
                    match=imported[(imported.protocol==protocol)&(imported.group_id==r.group_id)]
                    if len(match)==0: raise ValueError(f'Imported split missing group {r.group_id}')
                    if protocol=='within_run': raise ValueError('Imported manifests currently require held_out_group protocol')
                    value=match.split.iloc[0]
                    if value not in ['train','validation','test']:raise ValueError('Invalid imported split')
                    record.update(split=value,support_status='supported',reason='explicit imported assignment')
                rows.append(record)
    return pd.DataFrame(rows)


def bin_splits(frame, assignment):
    if assignment.protocol=='held_out_group':return np.full(len(frame),assignment.split,dtype=object)
    a=frame.source_row_start.to_numpy(); b=frame.source_row_end.to_numpy()
    # A closing endpoint used to determine support must also belong to the same split.
    def label(x):return np.where(x<assignment.cut1_row,'train',np.where(x<assignment.cut2_row,'validation','test'))
    sa,sb=label(a),label(b)
    return np.where(sa==sb,sa,'boundary')


def window_indices(frame, assignment, spec):
    if frame.empty: return [], "no eligible bins"
    interval=float(frame.interval_s.iloc[0])
    values=[spec['context_s']/interval,spec['horizon_s']/interval,spec['stride_s']/interval,spec['output_interval_s']/interval]
    if any(v<1 or not np.isclose(v,round(v),atol=1e-8) for v in values):
        return [],'durations must be exact positive integer multiples of resolution; no rounding'
    nc,nt,stride,output_step=map(lambda x:int(round(x)),values)
    if nt%output_step:return [],'horizon must be divisible by output interval'
    rows=[]
    for sid,g in frame.groupby('segment_id',sort=False):
        valid=g.is_valid.to_numpy(); times=g.available_s.to_numpy(float)
        splits=bin_splits(g,assignment); original=g.index.to_numpy()
        for end in range(nc,len(g)-nt+1,stride):
            start=end-nc; stop=end+nt
            if not valid[start:stop].all():continue
            if not np.allclose(np.diff(times[start:stop]),interval,atol=max(1e-8,interval*1e-6),rtol=0):continue
            split=splits[end-1]
            if split not in ['train','validation','test']:continue
            if not np.all(splits[start:stop]==split):continue
            issue=times[end-1]
            assert np.max(times[start:end])<=issue and times[end]>issue
            rows.append(dict(window_id=digest([assignment.protocol,assignment.run_id,sid,spec, int(original[start])]),
                             dataset_id=assignment.dataset_id,run_id=assignment.run_id,segment_id=sid,
                             group_id=assignment.group_id,protocol=assignment.protocol,split=split,
                             resolution_id=f'{interval:g}s',window_config_id=digest(spec)[:16],
                             context_start_index=int(original[start]),context_end_index=int(original[end-1]),
                             target_start_index=int(original[end+output_step-1]),target_end_index=int(original[stop-1]),
                             target_index_step=output_step,context_samples=nc,target_samples=nt//output_step,
                             context_start_s=float(g.bin_start_s.iloc[start]),issue_s=float(issue),
                             target_end_s=float(times[stop-1]),feature_available_s=float(issue),
                             target_available_s=float(times[stop-1]),target_definition=spec['target_definition'],
                             rounding='exact_multiple_required',boundary_history_reuse=False))
    return rows,''


def training_normalization(values):
    x=np.asarray(values,dtype=float)
    if not len(x):return {'status':'unsupported_no_training_observations'}
    mean=float(x.mean()); std=float(x.std())
    return {'status':'fitted','mean_w':mean,'scale_w':std if std>0 else 1.,'training_observations':len(x),
            'transform':'z=(power_w-mean_w)/scale_w','inverse':'power_w=z*scale_w+mean_w'}


def prepare_windows(out,cfg):
    out=Path(out)
    manifest=pd.read_csv(out/'manifest.csv'); quality=pd.read_csv(out/'native_quality.csv')
    assignments=split_assignments(manifest,quality,cfg['forecast']['splits'])
    save_table(out/'split_manifest.csv',assignments)
    variants=pd.read_csv(out/'resolution_manifest.csv')
    allrows=[]; support=[]; norm_values={}
    for a in assignments.itertuples():
        for spec in cfg['forecast']['windows']:
            if spec['target_definition']!='future_trajectory':raise ValueError('Only explicitly indexed future_trajectory is supported')
            rid=f"{spec['resolution_s']:g}s"
            found=variants[(variants.run_id==a.run_id)&(variants.resolution_id==rid)&(variants.status=='supported')]
            status=a.support_status; reason=a.reason
            frame=None; rows=[]
            if status=='supported' and found.empty:status,reason='unsupported','requested resolution unavailable'
            if status=='supported':
                frame=pd.read_parquet(found.iloc[0].path)
                rows,reason=window_indices(frame,a,spec)
                if reason:status='unsupported'
            allrows.extend(rows)
            # Eligible duration is union of context+target supports, not sum of overlapping windows.
            duration=0.
            for sid in {r['segment_id'] for r in rows}:
                intervals=sorted((r['context_start_s'],r['target_end_s']) for r in rows if r['segment_id']==sid)
                end=None
                for lo,hi in intervals:
                    duration+=hi-lo if end is None else max(0,hi-max(lo,end))
                    end=hi if end is None else max(end,hi)
            support.append(dict(dataset_id=a.dataset_id,run_id=a.run_id,protocol=a.protocol,resolution_id=rid,
                                window_config_id=digest(spec)[:16],status=status,reason=reason,
                                windows=len(rows),eligible_duration_s=duration))
            if frame is not None and cfg['forecast']['normalization']=='train_zscore':
                key=(a.protocol,rid,a.dataset_id)
                # Once per run/representation, irrespective of window multiplicity or configurations.
                bucket=norm_values.setdefault(key,{})
                if a.run_id not in bucket:
                    bucket[a.run_id]=frame.loc[(bin_splits(frame,a)=='train')&frame.is_valid,'power_w'].to_numpy()
    index=pd.DataFrame(allrows)
    if index.empty:index=pd.DataFrame(columns=['window_id','dataset_id','run_id','segment_id','group_id','protocol','split','resolution_id','window_config_id'])
    assert not index.window_id.duplicated().any()
    save_table(out/'window_index.parquet',index)
    save_table(out/'forecast_support.csv',pd.DataFrame(support))
    counts=index.groupby(['dataset_id','protocol','split','resolution_id','window_config_id']).size().rename('windows').reset_index()
    save_table(out/'window_counts.csv',counts)
    if cfg['forecast']['normalization'] not in ['none','train_zscore']:raise ValueError('Unsupported normalization')
    parameters={'mode':cfg['forecast']['normalization'],'scope':'separate protocol/resolution/dataset; training bins once each','parameters':{}}
    for key,runs in norm_values.items():
        parameters['parameters']['/'.join(key)]=training_normalization(np.concatenate(list(runs.values())))
    atomic_json(out/'normalization.json',parameters)
