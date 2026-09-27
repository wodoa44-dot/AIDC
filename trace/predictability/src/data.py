import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .core import continuous_segments,bin_segment,samples
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'preprocessing'))
from powerprep.adapters import load_native
from powerprep.quality import assess
from powerprep.discovery import stream_hash


def audit_and_select(cfg,out):
    pre=ROOT/'preprocessing'/'outputs'
    m=pd.read_csv(pre/'manifest.csv'); q=pd.read_csv(pre/'run_quality.csv')
    # Inspect actual source and prepared schemas for the excluded interpolation grids.
    provenance=[]
    for ds in ['nlr_training','nlr_online_rate']:
        representative=m[(m.dataset==ds)&~m.is_duplicate].sort_values('source_path').iloc[0]
        for kind,path in [('source',Path(representative.source_path)),('prepared',pre/'prepared'/f'{representative.run_id}.parquet')]:
            frame=pd.read_parquet(path)
            t=frame.time_s.to_numpy() if 'time_s' in frame else frame['timestep[s]'].to_numpy() if 'timestep[s]' in frame else np.asarray(frame.index,dtype=float)
            dt=np.diff(t)
            provenance.append(dict(dataset=ds,kind=kind,path=str(path),columns=list(frame.columns),index_name=frame.index.name,
                n=len(frame),duration_s=float(t[-1]-t[0]),dt_p05_s=float(np.quantile(dt,.05)),dt_median_s=float(np.median(dt)),dt_p95_s=float(np.quantile(dt,.95)),
                limitation='two-sided interpolation; grid spacing is not native measurement cadence'))
    (out/'excluded_sampling_provenance.json').write_text(json.dumps(provenance,indent=2))
    m['cohort']=m.dataset+'__'+m.hardware+'__'+m.workload
    candidate=m[(m.dataset=='highres_node') & ~m.is_duplicate].sort_values('source_path')
    chosen=candidate.groupby('cohort',sort=True).head(cfg['runs_per_cohort']).copy()
    selected=set(chosen.source_path)
    m['exclusion_reason']=[('exact-content duplicate alias' if r.is_duplicate else
      'two-sided polynomial interpolation: unavailable future source observations; no causal reconstruction in this bounded study' if r.dataset.startswith('nlr_') else
      'deterministic resource subset: first source paths per hardware/workload cohort' if r.source_path not in selected else '') for r in m.itertuples()]
    m[m.exclusion_reason!=''].to_csv(out/'excluded_inputs.csv',index=False)
    q.to_csv(out/'prior_quality_all_runs.csv',index=False)
    m.to_csv(out/'inventory_all_files.csv',index=False)
    records=[]; series={}
    for row in chosen.itertuples():
        path=pre/'prepared'/f'{row.run_id}.parquet'
        side=json.loads(path.with_suffix('.json').read_text())
        assert side['preparation']=='native' and side['source_hash']==row.source_hash
        assert stream_hash(row.source_path)==row.source_hash
        d=pd.read_parquet(path)
        # Independently compare to the existing source loader, not event summaries.
        raw=load_native(row)
        np.testing.assert_allclose(d[['time_s','power_w']],raw[['time_s','power_w']],equal_nan=True)
        stats=assess(d,row,{'analysis':{'gap_multiplier':cfg['gap_multiplier']}})
        seg=continuous_segments(d,stats['gap_threshold_s'])
        stats.update(cohort=row.cohort,prepared_path=str(path),source_path=row.source_path,
          prepared_sha256=stream_hash(path),source_sha256=row.source_hash,
          timestamp_column='time_s (relative seconds); timestamp_original (recorded datetime)',
          power_column='power_w',power_unit='W',power_scope=row.power_scope,
          provenance='strict sum of eight recorded GPU columns; no interpolation, rolling or fill',
          future_preprocessing=False,sensor_update_resolution='unverified; nominal recording cadence 20 ms',
          segment_count=len(seg),usable_observations=sum(len(s) for _,s in seg))
        records.append(stats);series[row.run_id]=(row,d,seg)
    pd.DataFrame(records).to_csv(out/'data_audit.csv',index=False)
    chosen['prepared_path']=[str(pre/'prepared'/f'{r}.parquet') for r in chosen.run_id]
    chosen.to_csv(out/'input_manifest.csv',index=False)
    return series


def build(series,cfg,out):
    cells={}; binrows=[]; splits=[]; discarded=[]
    continuous=out/'continuous';continuous.mkdir(exist_ok=True)
    for run,(row,d,segs) in series.items():
        start,end=d.time_s.min(),d.time_s.max(); cut1=start+.6*(end-start);cut2=start+.8*(end-start)
        for r in cfg['resolutions_ms']:
            parts=[]
            for sid,s in segs:
                z=bin_segment(s,r,coverage=cfg['min_bin_coverage'])
                for sub,g in z.groupby('subsegment'):
                    g=g.copy(); segid=f'{sid}:{sub}'
                    g['run_id']=run;g['segment_id']=segid;g['cohort']=row.cohort
                    g['dataset']=row.dataset;g['power_scope']=row.power_scope
                    g['resolution_ms']=r;g['quality_flag']='observed sample mean; count coverage >= 0.8; no fill'
                    parts.append(g)
                    for h in cfg['horizons_ms']:
                        if h<r or h%r: continue
                        result=samples(g,r,h,cfg['histories_ms'],cut1,cut2,cfg['evaluation_cadence_ms'])
                        if result is None:
                            discarded.append(dict(run_id=run,resolution_ms=r,horizon_ms=h,segment_id=segid,n_bins=len(g),reason='too short for maximum history and horizon'))
                            continue
                        meta,feat=result
                        if not len(meta):continue
                        meta['run_id']=run;meta['segment_id']=segid;meta['cohort']=row.cohort
                        meta['test_block']=np.minimum(3,np.maximum(0,((meta.target_s-cut2)/((end-cut2)/4)).astype(int)))
                        meta['bootstrap_block']=np.floor((meta.target_s-cut2)/cfg['bootstrap_block_s']).astype(int)
                        key=(row.cohort,r,h)
                        cells.setdefault(key,[]).append((meta,feat))
            combined=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
            combined.to_parquet(continuous/f'{run}_{r}ms.parquet',compression='zstd',index=False)
            binrows.append(dict(run_id=run,resolution_ms=r,n_bins=len(combined),segments=len(parts),
                                cut_train_validation_s=cut1,cut_validation_test_s=cut2,
                                count_coverage_mean=combined.coverage.mean() if len(combined) else 0))
    output={}
    for key,items in cells.items():
        meta=pd.concat([x[0] for x in items],ignore_index=True)
        assert not meta.duplicated(['run_id','target_bin']).any()
        feat={v:np.concatenate([x[1][v] for x in items]) for v in items[0][1]}
        assert all(np.isfinite(f).all() for f in feat.values())
        for (run,split),g in meta.groupby(['run_id','split']):
            splits.append(dict(cohort=key[0],resolution_ms=key[1],horizon_ms=key[2],run_id=run,
               split=['train','validation','test'][split],n=len(g),first_target_s=g.target_s.min(),last_target_s=g.target_s.max(),
               evaluation_cadence_ms=max(key[1],cfg['evaluation_cadence_ms'])))
        output[key]=(meta,feat)
    pd.DataFrame(binrows).to_csv(out/'bin_audit.csv',index=False)
    pd.DataFrame(splits).to_csv(out/'split_summary.csv',index=False)
    pd.DataFrame(discarded).to_csv(out/'excluded_segments.csv',index=False)
    return output
