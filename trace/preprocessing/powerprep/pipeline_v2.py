"""Independent, resumable forecasting-data stages; old output directories are read-only."""
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .contract import SCHEMA_VERSION, code_hash, digest, cache_key, cached, commit, save_table, atomic_json
from .discovery import stream_hash, duplicate_map
from .metadata import recover, clean
from .standardize import read_native, validate, aggregate, FLAGS
from .forecast import prepare_windows


def progress(out,stage,done,total,status='running'):
    atomic_json(out/'progress.json',dict(stage=stage,completed=done,total=total,status=status,
                                       updated_utc=time.strftime('%FT%TZ',time.gmtime())))
    print(f'{stage}: {done}/{total} {status}',flush=True)


def stage_key(out,cfg):
    return cache_key(stream_hash(out/'manifest.csv'),cfg)


def mark(out,cfg,stage):
    atomic_json(out/'stages'/f'{stage}.json',{'key':stage_key(out,cfg),'completed':True})


def require(out,cfg,stage):
    p=out/'stages'/f'{stage}.json'
    if not p.exists() or json.loads(p.read_text()).get('key')!=stage_key(out,cfg):
        raise ValueError(f'Missing/incompatible {stage} stage. Run {stage} explicitly first.')


def inventory_v2(out,cfg,selected,limit):
    m,metadata,evidence=recover(cfg,selected,limit)
    for col in ['gpu_count','node_count','batch_size','sequence_length','request_rate','concurrency','input_length','output_length','power_cap_w']:
        metadata[col]=pd.to_numeric(metadata[col],errors='coerce')
    for col in ['repeat_id','original_run_id','completion_status']:
        metadata[col]=metadata[col].map(lambda v:None if v is None else str(v))
    save_table(out/'manifest.csv',m)
    save_table(out/'source_metadata.parquet',metadata)
    canonical_metadata=metadata.loc[~m.is_duplicate.to_numpy()].copy()
    assert canonical_metadata.run_id.is_unique
    save_table(out/'run_metadata.parquet',canonical_metadata)
    save_table(out/'duplicates.csv',duplicate_map(m))
    path=out/'metadata_evidence.jsonl'; tmp=path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(clean(e),allow_nan=False)+'\n' for e in evidence));os.replace(tmp,path)
    support=[]
    for root in cfg['inputs'].values():
        for p in Path(root).rglob('*'):
            if p.is_file() and (p.name in ['README.md','metadata.csv','config.json'] or p.suffix in ['.py','.sh','.txt']):
                support.append({'path':str(p),'sha256':stream_hash(p)})
    atomic_json(out/'source_documentation.json',support)
    review=m[['dataset','run_id','source_path','source_hash','group_id']].copy()
    review['relationship']='exact aliases only confirmed by hash; shared acquisition groups are conservative, not duplicate claims'
    save_table(out/'source_relationships.csv',review)
    mark(out,cfg,'inventory');progress(out,'inventory',len(m),len(m),'completed')


def standardize_v2(out,cfg,resume):
    require(out,cfg,'inventory')
    manifest=pd.read_csv(out/'manifest.csv');canonical=manifest[~manifest.is_duplicate]
    quality=[];segments=[];prepared=[]
    transform=code_hash()
    for i,r in enumerate(canonical.itertuples(),1):
        nativepath=out/'standardized'/f'{r.run_id}.parquet'
        preparedpath=out/'prepared'/f'{r.run_id}.parquet'
        segpath=out/'quality'/f'{r.run_id}.segments.parquet'
        qpath=out/'quality'/f'{r.run_id}.json'
        side=out/'cache'/f'{r.run_id}.standardize.json'
        if stream_hash(r.source_path)!=r.source_hash:raise ValueError(f'Source changed since inventory: {r.source_path}')
        key=cache_key(r.source_hash,cfg,transform)
        paths=[nativepath,preparedpath,segpath,qpath]
        if not(resume and cached(side,key,paths)):
            native=read_native(r)
            d,seg,q=validate(native,r.run_id,r.dataset,cfg['analysis']['gap_multiplier'])
            save_table(nativepath,d);save_table(preparedpath,d[d.is_valid].copy())
            save_table(segpath,seg);atomic_json(qpath,clean(q));commit(side,key,paths)
        q=json.loads(qpath.read_text());quality.append(q);segments.append(pd.read_parquet(segpath))
        prepared.append({'dataset_id':r.dataset,'run_id':r.run_id,'prepared_samples':q['valid_samples'],
                         'excluded_samples':q['excluded_samples'],'valid_duration_s':q['valid_duration_s'],
                         'segments':q['segments']})
        progress(out,'preprocess',i,len(canonical))
    save_table(out/'native_quality.csv',pd.DataFrame(quality))
    save_table(out/'prepared_quality.csv',pd.DataFrame(prepared))
    save_table(out/'segments.parquet',pd.concat(segments,ignore_index=True))
    metadata=pd.read_parquet(out/'run_metadata.parquet')
    q=pd.DataFrame(quality)[['run_id','dt_median_s','dt_p05_s','dt_p95_s','hour_unknown_valid_samples']]
    metadata=metadata.drop(columns=[c for c in q if c!='run_id' and c in metadata]).merge(q,on='run_id',how='left')
    metadata['observed_interval_status']='verified from native row order; not sensor cadence'
    save_table(out/'run_metadata.parquet',metadata)
    mark(out,cfg,'preprocess')


def resolutions_v2(out,cfg,resume):
    require(out,cfg,'preprocess')
    q=pd.read_csv(out/'native_quality.csv');rows=[];transform=code_hash()
    for i,r in enumerate(q.itertuples(),1):
        path=out/'standardized'/f'{r.run_id}.parquet'; d=None
        source_hash=stream_hash(path)
        for interval in cfg['resolutions']['intervals_s']:
            rid=f'{interval:g}s';dest=out/'resolutions'/rid/f'{r.run_id}.parquet'
            record=dict(dataset_id=r.dataset_id,run_id=r.run_id,resolution_id=rid,interval_s=interval,
                        source_representation='standardized native',path=str(dest),
                        bin_origin='first valid observation in each segment',label='right edge',availability='right edge',
                        aggregation='left_hold_time_weighted',partial_policy=cfg['resolutions']['partial_policy'],
                        realtime_interpretation='unsupported upstream interpolation' if r.dataset_id.startswith('nlr') else 'sensor cadence/latency unverified',
                        raw_source_coverage='unknown',status='supported',reason='',samples=0)
            if pd.isna(r.dt_median_s) or interval+max(1e-9,r.dt_median_s*.01)<r.dt_median_s:
                record.update(status='unsupported',reason='finer than observed native recording interval')
            else:
                side=out/'cache'/f'{r.run_id}.{rid}.json';key=cache_key(source_hash,{'config':cfg,'interval':interval},transform)
                if not(resume and cached(side,key,[dest])):
                    if d is None:d=pd.read_parquet(path)
                    variant=aggregate(d,float(interval),cfg['resolutions']['min_coverage'],cfg['resolutions']['partial_policy'])
                    save_table(dest,variant);commit(side,key,[dest])
                import pyarrow.parquet as pq
                record['samples']=pq.ParquetFile(dest).metadata.num_rows
            rows.append(record)
        progress(out,'resolutions',i,len(q))
    save_table(out/'resolution_manifest.csv',pd.DataFrame(rows));mark(out,cfg,'resolutions')


def write_dictionary(out):
    (out/'data_dictionary.md').write_text('''# Data contract 2.0.0

Native standardized tables preserve original order and invalid rows. `prepared/` is
an explicitly valid-only native view, with the same persistent segment IDs. This
v2 native representation replaces v1's special online-rate 0.1 s prepared view;
use `resolutions/0.1s/` explicitly when that representation is needed. Historical
consumers keep using the untouched old `outputs/prepared/` directory.

Keys: standardized `(run_id, source_row)`; segment `(run_id, segment_id)`;
resolution `(run_id, segment_id, resolution_id, bin_index)`; window `window_id`.
Run metadata has one row per canonical run; source_metadata.parquet and evidence
retain every alias declaration independently. Exact content aliases share a source group; byte-distinct files are not asserted
to be independent. Run IDs retain the legacy dataset/hash convention.

- dataset_id, run_id, segment_id: string; segment_id null only on excluded rows.
- time_s, available_s, bin_start_s, bin_end_s, bin_origin_s: float64 seconds.
  Native time is elapsed source time (may reset); original timestamp is a string.
  Short mm:ss timestamps preserve unknown absolute hours. Only relative differences
  within monotonic epochs are used; wraps are split, never repaired across hours.
  timestamp_hour_known and hour_unknown_valid_samples expose this limitation.
- source_row: int64, zero-based position in the original decoded file.
- power_w and original GPU channel power: float64 watts, physical units unchanged.
- is_valid: bool. quality_flags: integer bitmask, meanings in quality_flags.json.
- coverage: native validity indicator or aggregate observed-duration/bin-duration.
  Full-bin duration equality uses 1 ns absolute / 1e-10 relative numerical
  tolerance; this is floating-point handling, not permission to bridge gaps.
  sample_coverage is count divided by expected recording-grid count; it may exceed
  one under irregular cadence. Neither establishes independent sensor coverage.
  raw_source_coverage remains nullable/unknown for all unverified sensor sources.
- segment duration: last observed time minus first, summed within valid segments.
  Singleton duration is zero. No duration is guessed past a segment endpoint.
  Total timestamp span is max minus min and is reported separately, even on resets.
- Resolution power is a left-held time integral over [start,end), justified as a
  causal convention for irregular recordings, not a claim of physical sensor hold.
  The final observation closes support. No extrapolated tail, upsampling, midpoint
  availability, or aggregation across resets/invalid observations is allowed.
- source_row_start/end delimit contributing/supporting original rows inclusive;
  segment_id and source standardized table resolve exact lineage. Native GPU
  channels remain available; aggregate-only NLR data has no fabricated channels.
- window input/target indices are zero-based inclusive positions in the indicated
  resolution Parquet, with target_index_step specifying trajectory subsampling.
  All context and target bins must be in one valid segment and one split. Durations
  require exact integer multiples; boundary history reuse is disabled. Eligibility
  duration is the union of context+target window supports, counted once per segment.
- Normalization, when enabled, uses training bins once, separately by dataset,
  protocol and resolution; raw watts stay unchanged. Inverse rules are persisted.

Nullable run metadata include hardware, size, batch semantics, sensor cadence,
averaging, precision, accumulation and caps when not established. Numeric counts
and durations are numeric; identifiers and declarations are strings. Full source
table rows are retained as JSON. Every recovered field has separate evidence and
verified/source_declared/inferred/unknown status; filename declarations do not
verify actual execution. Diffusion filename size tokens are not parameter counts.

Input availability: source-declared hardware/model/batch settings are static only
when externally known before prediction; power/channel readings are past-observed
and available no earlier than available_s. Completion, full-run summaries, quality
judgments and events are offline evaluation-only. IDs/groups identify partitions,
not model features. No known-future schedule or retrieval index is fabricated.
NLR interpolation/backfill is not causal; these datasets support offline resolution
comparisons but do not establish real-time input availability or control feasibility.
''')
    atomic_json(out/'quality_flags.json',FLAGS|{'insufficient_bin_coverage':512})


def report_v2(out,cfg):
    require(out,cfg,'preprocess')
    m=pd.read_csv(out/'manifest.csv');q=pd.read_csv(out/'native_quality.csv')
    canonical=m[~m.is_duplicate]
    summary=q.groupby('dataset_id').agg(runs=('run_id','size'),native_samples=('native_samples','sum'),
                prepared_samples=('valid_samples','sum'),excluded_samples=('excluded_samples','sum'),
                valid_duration_s=('valid_duration_s','sum'),segments=('segments','sum'),
                timestamp_span_sum_s=('total_timestamp_span_s','sum')).reset_index()
    save_table(out/'dataset_summary.csv',summary)
    joins=m.groupby(['dataset','metadata_join_status']).size().rename('files').reset_index()
    lines=['# Forecasting data preparation 2.0.0','',f'Source files: {len(m)}; exact-content canonical runs: {len(canonical)}; exact aliases: {int(m.is_duplicate.sum())}.',
           'Byte-distinct files are not assumed statistically independent. No forecasting models or control experiments were trained.','',
           '## Actual data counts','',summary.to_markdown(index=False),'','## Metadata joins','',joins.to_markdown(index=False)]
    if (out/'resolution_manifest.csv').exists():
        v=pd.read_csv(out/'resolution_manifest.csv')
        tab=v.groupby(['dataset_id','resolution_id','status']).agg(runs=('run_id','size'),samples=('samples','sum')).reset_index()
        lines+=['','## Resolution variants','',tab.to_markdown(index=False)]
    if (out/'window_counts.csv').exists():
        w=pd.read_csv(out/'window_counts.csv');sp=pd.read_csv(out/'forecast_support.csv')
        lines+=['','## Forecast windows','',w.to_markdown(index=False),'','Unsupported settings:',
                sp[sp.status!='supported'].groupby(['dataset_id','protocol','reason']).size().rename('run_configurations').reset_index().to_markdown(index=False)]
    if (out/'events.parquet').exists():
        e=pd.read_parquet(out/'events.parquet'); counts=e.groupby('dataset').size() if len(e) else pd.Series(dtype=int)
        lines+=['','## Optional events','',counts.reindex(summary.dataset_id,fill_value=0).rename('event_rows_all_configurations').to_frame().to_markdown(),
                'Configurations are sensitivity analyses, not independent events. Definitions are in event_dictionary.md.']
    else:lines+=['','Optional event detection was not run; event counts are not inferred.']
    lines+=['','## Limitations','',
            'Source sensor update/averaging cadence remains unknown. Observed cadence describes recording only. '
            'NLR supplied series are interpolated; online-rate additionally inherits backward-fill cleaning. '
            'Its grid coverage is not raw sensor coverage. RAPL package/core summation is inherited and not independently calibrated. '
            'Conservative shared-acquisition grouping can make held-out and within-run inference settings unsupported; '
            'no alternate evaluation protocol is silently substituted. No future-derived labels are model inputs. '
            'Use metadata_evidence.jsonl and source_documentation.json for provenance.']
    (out/'preprocessing_report.md').write_text('\n'.join(lines)+'\n');write_dictionary(out)
    # Separate datasets/scopes: one plot per dataset, no mixed detector configurations.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for ds,g in q.groupby('dataset_id'):
        fig,ax=plt.subplots();ax.hist(g.valid_duration_s/60,bins=20);ax.set(title=ds,xlabel='Valid contiguous minutes per run',ylabel='Runs')
        fig.tight_layout();fig.savefig(out/'plots'/f'{ds}_valid_duration.png');plt.close(fig)
        run=g.run_id.iloc[0];d=pd.read_parquet(out/'prepared'/f'{run}.parquet')
        fig,ax=plt.subplots()
        for _,seg in d.groupby('segment_id',sort=False):
            step=max(1,len(seg)//cfg['plots']['max_full_points']);ax.plot(seg.time_s.iloc[::step],seg.power_w.iloc[::step],lw=.7)
        ax.set(title=f'{ds}: {run}',xlabel='Source elapsed seconds',ylabel='Power (W)');fig.tight_layout();fig.savefig(out/'plots'/f'{ds}_example.png');plt.close(fig)
    mark(out,cfg,'report')


def run(stage,cfg,output,selected=None,limit=None,resume=True):
    out=Path(output).resolve()
    legacy=Path(__file__).parents[1]/'outputs'
    if out==legacy.resolve() or ((out/'manifest.csv').exists() and not (out/'provenance.json').exists()):
        raise ValueError('Use a separate versioned v2 output directory; existing outputs are protected')
    out.mkdir(parents=True,exist_ok=True)
    for name in ['standardized','prepared','quality','cache','resolutions','stages','plots']:(out/name).mkdir(exist_ok=True)
    cfg={k:v for k,v in cfg.items() if not k.startswith('_')}
    effective=cfg|{'selected_datasets':selected or cfg['datasets'],'limit_per_dataset':limit}
    cp=out/'effective_config.json'
    if cp.exists() and json.loads(cp.read_text())!=effective:
        raise ValueError('Incompatible output configuration; choose a new versioned directory')
    atomic_json(cp,effective)
    atomic_json(out/'provenance.json',{'schema_version':SCHEMA_VERSION,'transformation_hash':code_hash(),
                'source_policy':'raw inputs read-only; historical experiment outputs untouched','pid':os.getpid()})
    try:
        if stage in ['inventory','all']:inventory_v2(out,cfg,selected,limit)
        if stage in ['preprocess','all']:standardize_v2(out,cfg,resume)
        if stage in ['resolutions','all']:resolutions_v2(out,cfg,resume)
        if stage in ['windows','all']:
            require(out,cfg,'resolutions');prepare_windows(out,cfg);mark(out,cfg,'windows')
        if stage=='detect' or (stage=='all' and cfg['events']['enabled']):
            require(out,cfg,'resolutions')
            from .events_v2 import detect_all
            detect_all(out,cfg);mark(out,cfg,'detect')
        if stage in ['report','all']:report_v2(out,cfg)
        progress(out,stage,1,1,'completed')
    except Exception as exc:
        atomic_json(out/'failure.json',{'stage':stage,'error':repr(exc)})
        progress(out,stage,0,1,'failed');raise
