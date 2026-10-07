"""Causal features on original 200ms bins; sample only after feature computation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

META = ['hardware','workload','model_size_token','image_size','batch_size','sequence_length','zero_stage']
KEY = ['run_id','segment','bin']


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def frame_hash(d):
    return hashlib.sha256(pd.util.hash_pandas_object(d,index=False).values.tobytes()).hexdigest()


def atomic_json(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)


def parquet(p,d):
    tmp=p.with_suffix('.tmp.parquet');d.to_parquet(tmp,index=False);tmp.replace(p)


def columns(history,metadata=False):
    bins=round(history/.2)
    # A window of H bins includes current bin and lags 1..H-1.
    # Existing lag exactly H is excluded to keep all observed bins inside H seconds.
    cols=['current_w']+[f'lag_{x}' for x in [1,2,5,10,25,50,150] if x<bins]
    cols += [f'{a}_{w}' for w in [5,25,50,150] if w<=bins for a in ['mean','std']]
    cols+=['slope_1']
    return cols+(META if metadata else [])


def continuous_features(raw):
    pieces=[]
    for run,g in raw.groupby('run_id',sort=True):
        g=g.sort_values('issuance_s').copy()
        power=g.current_w.to_numpy();t=g.issuance_s.to_numpy();b=g.bin.to_numpy();seg=g.segment.to_numpy()
        breaks=np.r_[True,(seg[1:]!=seg[:-1]) | (np.diff(b)!=1) | ~np.isclose(np.diff(t),.2,atol=1e-7,rtol=0)]
        block=np.cumsum(breaks)
        for _,x in g.groupby(block,sort=False):
            x=x.copy();p=x.current_w
            for lag in [1,2,5,10,25]: x[f'lag_{lag}']=p.shift(lag)
            for width in [5,25,50]:
                x[f'mean_{width}']=p.rolling(width,min_periods=width).mean()
                x[f'std_{width}']=p.rolling(width,min_periods=width).std()
            x['slope_1']=p.diff()
            # All 50 observed bins and next 5 target-path bins must be finite.
            all55=p.rolling(55,min_periods=55).count().shift(-5).eq(55)
            future=p.shift(-5)
            x['valid_anchor']=all55 & future.notna()
            np.testing.assert_allclose(x.loc[x.valid_anchor,'y_5'],future[x.valid_anchor],rtol=1e-9,atol=1e-7)
            x['actual_w']=future
            pieces.append(x)
    return pd.concat(pieces,ignore_index=True)


def spaced(g):
    chosen=[];last=-np.inf
    for i,t in zip(g.index,g.issuance_s):
        if t-last >= 1.-1e-8:chosen.append(i);last=t
    return g.loc[chosen].copy()


def sample(g,cap,training,seed):
    g=g.sort_values(['issuance_s','segment','bin']).reset_index(drop=True)
    stable=int.from_bytes(hashlib.sha256((str(seed)+':'+str(g.run_id.iloc[0])).encode()).digest()[:8],'big')
    rule='all candidates' if len(g)<=cap else ('10 strata random without replacement' if training else '10 strata equispaced indices')
    partitions=np.array_split(np.arange(len(g)),10)
    bucket={int(i):k for k,p in enumerate(partitions) for i in p}
    if len(g)<=cap:ix=np.arange(len(g))
    else:
        rng=np.random.default_rng(stable)
        ix=np.concatenate([np.sort(rng.choice(p,cap//10,replace=False)) if training else p[np.linspace(0,len(p)-1,cap//10,dtype=int)] for p in partitions])
    assert len(np.unique(ix))==len(ix)
    d=g.iloc[np.sort(ix)].copy();d['time_stratum']=[bucket[int(i)] for i in np.sort(ix)]
    d['sampling_role']='training' if training else 'evaluation';d['sampling_seed']=str(stable) if training else '';d['sampling_rule']=rule
    return d


def prepare(cfg,out):
    raw=pd.read_parquet(cfg['prepared']);groups=pd.read_csv(cfg['group_manifest'],dtype={'collection_day':str})
    assert raw.run_id.nunique()==32 and not groups.run_id.duplicated().any()
    assert set(raw.run_id)==set(groups.run_id)
    assert set(groups.collection_day)==set(cfg['development_days']+[cfg['holdout_day']]),'Date manifest mismatch'
    raw['collection_day']=raw.run_id.map(groups.set_index('run_id').collection_day)
    data=continuous_features(raw)
    valid=data[data.valid_anchor].copy();candidates=pd.concat([spaced(g) for _,g in valid.groupby(['run_id','segment'],sort=True)],ignore_index=True)
    counts=candidates.groupby('run_id').size(); included=set(counts[counts>=cfg['minimum_candidates']].index)
    tr=[];ev=[]
    for run,g in candidates[candidates.run_id.isin(included)].groupby('run_id',sort=True):
        if g.collection_day.iloc[0]!=cfg['holdout_day']:tr.append(sample(g,cfg['train_cap'],True,cfg['seed']))
        ev.append(sample(g,cfg['evaluation_cap'],False,cfg['seed']))
    train=pd.concat(tr,ignore_index=True);evaluation=pd.concat(ev,ignore_index=True)
    for d in (train,evaluation):
        d['segment_id']=d.segment;d['anchor_timestamp']=d.issuance_s;d['target_timestamp']=d.issuance_s+1
        assert not d.duplicated(KEY).any()
        assert np.isfinite(d[columns(10)]).all().all()
    inventory=groups[['run_id','collection_day','verification_status','source_file','source_sha256']].merge(raw.drop_duplicates('run_id')[['run_id']+META],on='run_id')
    for name,source in [('original_bins',raw),('valid_anchors',valid),('spaced_candidates',candidates),('training_rows',train),('evaluation_rows',evaluation)]:
        inventory[name]=inventory.run_id.map(source.groupby('run_id').size()).fillna(0).astype(int)
    inventory['included']=inventory.run_id.isin(included)
    inventory['exclusion_reason']=np.where(inventory.included,'','fewer than 30 one-second candidates with continuous 50 history bins and 5 future bins')
    inventory['training_role']=np.where(inventory.collection_day==cfg['holdout_day'],'holdout evaluation only','development training and validation')
    inventory.to_csv(out/'run_inventory.csv',index=False);inventory[~inventory.included].to_csv(out/'excluded_runs.csv',index=False)
    inventory.to_csv(out/'sampling_summary.csv',index=False)
    parquet(out/'train_anchors.parquet',train);parquet(out/'evaluation_anchors.parquet',evaluation)
    # Save full causal feature frame for short continuous illustrative inference only.
    parquet(out/'continuous_features.parquet',data)
    splits=[]
    for fold,day in enumerate(cfg['development_days']):
        days=[x for x in cfg['development_days'] if x!=day]
        assert len(train[train.collection_day.isin(days)]) and len(evaluation[evaluation.collection_day==day])
        splits.append({'fold':fold,'train_days':days,'validation_days':[day]})
    splits.append({'fold':'final','train_days':cfg['development_days'],'validation_days':[cfg['holdout_day']]})
    atomic_json(out/'split_manifest.json',{'grouping':'assumed calendar day, not verified independent session','folds':splits})
    atomic_json(out/'feature_schema.json',{str(h):{'history_only':columns(h),'metadata':columns(h,True),'history_bins':round(h/.2),'boundary_lag_excluded':True} for h in cfg['histories']})
    return data,train,evaluation,inventory,splits


def encode(train,evaluation,cols,mapping=None):
    x=train[cols].copy();v=evaluation[cols].copy();maps={} if mapping is None else mapping
    diagnostics=[]
    for c in cols:
        categorical=c in ('hardware','workload') or x[c].dtype==object
        if categorical:
            if mapping is None:maps[c]={str(a):i for i,a in enumerate(sorted(x[c].dropna().astype(str).unique()))}
            names=maps[c]
            unseen=v[c].notna() & ~v[c].astype(str).isin(names)
            x[c]=x[c].map(lambda a:names.get(str(a),-1)).astype(float)
            v[c]=v[c].map(lambda a:names.get(str(a),-1)).astype(float)
        else:
            unseen=pd.Series(False,index=v.index);x[c]=pd.to_numeric(x[c],errors='coerce');v[c]=pd.to_numeric(v[c],errors='coerce')
        diagnostics.append({'column':c,'categorical':categorical,'evaluation_unseen_fraction':float(unseen.mean()),'evaluation_missing_fraction':float(evaluation[c].isna().mean()),'training_missing_fraction':float(train[c].isna().mean()),'training_constant':bool(train[c].nunique(dropna=False)<=1),'evaluation_constant':bool(evaluation[c].nunique(dropna=False)<=1)})
    return x,v,maps,diagnostics


def weights(d):
    counts=d.groupby('run_id').size();n=len(d);r=len(counts)
    w=d.run_id.map(n/(r*counts)).to_numpy()
    np.testing.assert_allclose(w.mean(),1.)
    totals=pd.Series(w,index=d.index).groupby(d.run_id).sum();np.testing.assert_allclose(totals,n/r)
    return w


def metrics(pred):
    rows=[]
    for (scope,model,run,region,postprocess),g in pred.groupby(['scope','model','run_id','region','postprocess']):
        error=g.prediction_w-g.actual_w;a=abs(error);n=len(g)
        rows.append({'scope':scope,'model':model,'run_id':run,'collection_day':g.collection_day.iloc[0],'region':region,'postprocess':postprocess,'n':n,'mae':a.mean(),'rmse':np.sqrt(np.mean(error**2)),'signed_bias':error.mean(),'p95_abs_error':np.quantile(a,.95),'p99_abs_error':np.quantile(a,.99) if n>=100 else np.nan})
    d=pd.DataFrame(rows)
    base=d[d.model=='persistence'][['scope','run_id','region','postprocess','mae']].rename(columns={'mae':'persistence_mae'})
    d=d.merge(base,on=['scope','run_id','region','postprocess'],validate='many_to_one')
    d['mae_gain_w']=d.persistence_mae-d.mae
    return d


def summarize(d):
    group=['scope','model','region','postprocess'];s=d.groupby(group)[['mae','rmse','signed_bias','p95_abs_error','p99_abs_error']].mean().reset_index()
    n=d.groupby(group).agg(n_runs=('run_id','nunique'),n_samples=('n','sum'),p99_n_runs=('p99_abs_error','count'),improved_runs=('mae_gain_w',lambda x:int((x>1e-9).sum())),worsened_runs=('mae_gain_w',lambda x:int((x < -1e-9).sum()))).reset_index();s=s.merge(n,on=group)
    base=s[s.model=='persistence'][['scope','region','postprocess','mae']].rename(columns={'mae':'persistence_mae'})
    s=s.merge(base,on=['scope','region','postprocess'],validate='many_to_one');s['mae_improvement_pct']=100*(s.persistence_mae-s.mae)/s.persistence_mae.replace(0,np.nan)
    return s


def prediction_frame(evaluation,p,model,scope,q50,q90):
    z=evaluation[KEY+['collection_day','issuance_s','current_w','actual_w']].copy();z['target_timestamp']=z.issuance_s+1
    z['prediction_w']=p;z['model']=model;z['scope']=scope;z['postprocess']='raw';z['region']='whole'
    delta=z.actual_w-z.current_w
    labels=np.where(abs(delta)<=q50,'normal',np.where(delta>q50,'rising','falling'))
    rows=[z]+[z[labels==r].assign(region=r) for r in ['normal','rising','falling']]+[z[abs(delta)>q90].assign(region='large_endpoint_change')]
    allrows=pd.concat(rows,ignore_index=True)
    clipped=allrows.copy();clipped['prediction_w']=np.maximum(0,clipped.prediction_w);clipped['postprocess']='zero_clip_diagnostic'
    return pd.concat([allrows,clipped],ignore_index=True)
