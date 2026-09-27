"""Native order validation precedes all sorting, binning and analysis views."""
import numpy as np
import pandas as pd
from .adapters import GPU_COLS
FLAGS = {'invalid_time':1,'invalid_power':2,'negative_power':4,'missing_channel':8,
         'duplicate_time':16,'time_reversal':32,'gap_before':64,'repeated_power':128,'irregular_cadence':256}


def read_native(row):
    if row.dataset == 'highres_single':
        d=pd.read_csv(row.source_path,encoding='cp1252',usecols=lambda c:c in ['Date','Time','CPU Package Power [W]'] or (c.startswith('GPU') and ('[W]' in c or 'Clock' in c)))
        original=d.Date.astype(str)+' '+d.Time.astype(str)
        timestamp=pd.to_datetime(original,format='%d.%m.%Y %H:%M:%S.%f',errors='coerce')
        origin=timestamp.dropna().iloc[0] if timestamp.notna().any() else pd.NaT
        for col in d:
            if col not in ['Date','Time']:d[col]=pd.to_numeric(d[col],errors='coerce')
        d['timestamp_original']=original
        d['time_s']=(timestamp-origin).dt.total_seconds()
        short=d.Time.astype(str).str.extract(r'^(\d{1,2}):(\d{1,2}(?:\.\d+)?)$')
        minutes=pd.to_numeric(short[0],errors='coerce');seconds=pd.to_numeric(short[1],errors='coerce')
        relative=(minutes*60+seconds).where((minutes<60)&(seconds<60))
        d['timestamp_hour_known']=timestamp.notna()
        # No absolute hour is fabricated. Without any complete clock stamps,
        # recover relative mm:ss differences; wraps/reversals split epochs below.
        if not timestamp.notna().any() and relative.notna().any():
            d['time_s']=relative-relative.dropna().iloc[0]
        d['timestamp_interpretation']=np.where(d.timestamp_hour_known,'full_source_clock','hour_unknown_relative_minute_second')
        d['power_w']=pd.to_numeric(d.get('GPU Power [W]',pd.Series(np.nan,index=d.index)),errors='coerce')
        d['gpu0_power_W']=d.power_w
        d['missing_channel']=~np.isfinite(d.power_w)
        d['negative_channel']=d.power_w<0
        d=d.drop(columns=['Date','Time'])
    elif row.dataset == 'highres_node':
        d = pd.read_csv(row.source_path, usecols=lambda c: c=='timestamp' or c in GPU_COLS)
        for col in GPU_COLS:
            if col not in d: d[col] = np.nan
            d[col] = pd.to_numeric(d[col], errors='coerce')
        timestamp = pd.to_datetime(d.timestamp, errors='coerce')
        origin = timestamp.dropna().iloc[0] if timestamp.notna().any() else pd.NaT
        d['timestamp_original'] = d.timestamp.astype(str)
        d['time_s'] = (timestamp-origin).dt.total_seconds()
        finite = np.isfinite(d[GPU_COLS]).all(axis=1)
        d['power_w'] = d[GPU_COLS].sum(axis=1, min_count=8).where(finite)
        d['missing_channel'] = ~finite
        d['negative_channel'] = (d[GPU_COLS]<0).any(axis=1)
        d = d.drop(columns='timestamp')
    else:
        source = pd.read_parquet(row.source_path)
        t = source['timestep[s]'] if 'timestep[s]' in source else source.index
        # Preserve every native source channel, without fabricating decompositions.
        d = source.reset_index(drop=True).copy()
        d['timestamp_original'] = np.asarray(t).astype(str)
        d['time_s'] = np.asarray(pd.to_numeric(t, errors='coerce'), dtype=float)
        d['power_w'] = pd.to_numeric(source['power[W]'], errors='coerce').to_numpy()
        d['missing_channel'], d['negative_channel'] = False, False
    d['source_row'] = np.arange(len(d), dtype=np.int64)
    return d


def validate(d, run_id, dataset_id, gap_multiplier=5.):
    d = d.copy().reset_index(drop=True)
    t, p = d.time_s.to_numpy(float), d.power_w.to_numpy(float)
    dt = np.r_[np.nan, np.diff(t)]
    positive = dt[np.isfinite(dt)&(dt>0)]
    median = float(np.median(positive)) if len(positive) else np.nan
    flags = np.zeros(len(d), dtype=np.uint16)
    conditions = {'invalid_time':~np.isfinite(t), 'invalid_power':~np.isfinite(p),
                  'negative_power':(p<0)|d.get('negative_channel',pd.Series(False,index=d.index)).to_numpy(),
                  'missing_channel':d.get('missing_channel',pd.Series(False,index=d.index)).to_numpy(),
                  'duplicate_time':dt==0,'time_reversal':dt<0,'gap_before':dt>median*gap_multiplier,
                  'repeated_power':np.r_[False,np.diff(p)==0],
                  'irregular_cadence':np.isfinite(dt)&(dt>0)&(np.abs(dt-median)>max(1e-9,median*.1))}
    for key, mask in conditions.items(): flags[mask] |= FLAGS[key]
    # Reversal starts a new epoch; its first valid observation remains usable.
    valid = (flags & (1|2|4|8|16)) == 0
    boundary = np.r_[True, (~valid[1:])|(~valid[:-1])|(dt[1:]<=0)|(dt[1:]>median*gap_multiplier)]
    block = np.cumsum(boundary)-1
    epoch = np.cumsum(np.nan_to_num(dt, nan=0)<0)
    d['dataset_id'], d['run_id'] = dataset_id, run_id
    d['epoch_id'], d['quality_flags'], d['is_valid'] = epoch, flags, valid
    d['segment_id'] = [f'{run_id}:s{x}' if ok else None for x,ok in zip(block,valid)]
    d['available_s'] = t
    d['coverage'] = np.where(valid, 1., 0.)
    d['raw_source_coverage'] = np.nan
    segments = []
    for sid, g in d[d.is_valid].groupby('segment_id',sort=False):
        idx = int(g.index[0]); reason=[]
        if idx==0: reason.append('run_start')
        if idx and not valid[idx-1]: reason.append('invalid_predecessor')
        if dt[idx]<0: reason.append('reset')
        if dt[idx]==0: reason.append('duplicate')
        if dt[idx]>median*gap_multiplier: reason.append('gap')
        duration = float(g.time_s.iloc[-1]-g.time_s.iloc[0])
        segments.append(dict(dataset_id=dataset_id,run_id=run_id,segment_id=sid,epoch_id=int(g.epoch_id.iloc[0]),
                             start_s=float(g.time_s.iloc[0]),end_s=float(g.time_s.iloc[-1]),
                             source_row_start=int(g.source_row.iloc[0]),source_row_end=int(g.source_row.iloc[-1]),
                             sample_count=len(g),duration_s=duration,boundary_reason=';'.join(reason) or 'discontinuity',
                             eligible=len(g)>=2,exclusion_reason='' if len(g)>=2 else 'singleton'))
    seg = pd.DataFrame(segments, columns=['dataset_id','run_id','segment_id','epoch_id','start_s','end_s',
                                       'source_row_start','source_row_end','sample_count','duration_s','boundary_reason','eligible','exclusion_reason'])
    finite_t=t[np.isfinite(t)]
    quality = dict(dataset_id=dataset_id,run_id=run_id,native_samples=len(d),valid_samples=int(valid.sum()),
                   excluded_samples=int((~valid).sum()),segments=len(seg),
                   total_timestamp_span_s=float(np.ptp(finite_t)) if len(finite_t) else 0.,
                   valid_duration_s=float(seg.duration_s.sum()),
                   dt_median_s=median if np.isfinite(median) else None,
                   dt_p05_s=float(np.percentile(positive,5)) if len(positive) else None,
                   dt_p95_s=float(np.percentile(positive,95)) if len(positive) else None)
    quality.update({key+'_count':int(mask.sum()) for key,mask in conditions.items()})
    quality['hour_unknown_valid_samples']=int((valid & ~d.get('timestamp_hour_known',pd.Series(True,index=d.index)).to_numpy()).sum())
    # Constant durations are computed within valid segments only.
    constant_max=0.
    for _,g in d[d.is_valid].groupby('segment_id',sort=False):
        change=g.power_w.ne(g.power_w.shift()).cumsum()
        spans=g.groupby(change).time_s.agg(['min','max'])
        if len(spans): constant_max=max(constant_max,float((spans['max']-spans['min']).max()))
    quality['max_constant_stretch_s']=constant_max
    return d, seg, quality


def aggregate(d, interval, min_coverage=.8, partial_policy='drop'):
    """Left-held time integration on observed [first,last], no guessed tail.

    Availability is the right edge. The right-edge observation closes coverage;
    no samples after that edge are required. Each segment has its own origin.
    """
    rows=[]
    for sid,g in d[d.is_valid].groupby('segment_id',sort=False):
        t=g.time_s.to_numpy(float); p=g.power_w.to_numpy(float)
        if len(t)<2: continue
        median=float(np.median(np.diff(t)))
        if interval+max(1e-9,median*.01)<median: continue
        origin=t[0]; span=t[-1]-origin
        count=int(np.floor((span+1e-9)/interval)) if partial_policy=='drop' else int(np.ceil(span/interval))
        if count<=0:continue
        start=origin+np.arange(count)*interval
        stop=np.minimum(start+interval,t[-1])
        def integral_at(x, values):
            cumulative=np.r_[0.,np.cumsum(np.diff(t)*values[:-1])]
            k=np.clip(np.searchsorted(t,x,side='right')-1,0,len(t)-1)
            return cumulative[k]+(x-t[k])*values[k]
        power=(integral_at(stop,p)-integral_at(start,p))/(stop-start)
        first=np.searchsorted(t,start,side='right')-1
        last=np.searchsorted(t,stop,side='right')-1
        n=np.maximum(0,np.searchsorted(t,stop,side='left')-np.searchsorted(t,start,side='left'))
        coverage=(stop-start)/interval
        coverage=np.where(np.isclose(stop-start,interval,atol=1e-9,rtol=1e-10),1.,coverage)
        frame=pd.DataFrame({'dataset_id':g.dataset_id.iloc[0],'run_id':g.run_id.iloc[0],
                            'segment_id':sid,'epoch_id':int(g.epoch_id.iloc[0]),'bin_index':np.arange(count),
                            'resolution_id':f'{interval:g}s','interval_s':interval,'bin_origin_s':origin,
                            'bin_start_s':start,'bin_end_s':stop,'time_s':stop,'available_s':stop,
                            'power_w':power,'coverage':coverage,'sample_coverage':n/np.maximum(1,interval/median),
                            'raw_source_coverage':np.nan,'native_sample_count':n,
                            'source_row_start':g.source_row.to_numpy()[first],
                            'source_row_end':g.source_row.to_numpy()[last],
                            'aggregation':'left_hold_time_weighted','partial_bin':coverage<1-1e-8,
                            'is_valid':coverage>=min_coverage,'quality_flags':np.where(coverage>=min_coverage,0,512)})
        for col in [c for c in GPU_COLS if c in g]:
            v=g[col].to_numpy(float)
            frame[col]=(integral_at(stop,v)-integral_at(start,v))/(stop-start)
        rows.append(frame)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['dataset_id','run_id','segment_id','resolution_id','time_s','available_s','power_w','is_valid'])
