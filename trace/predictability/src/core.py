"""Causal binning, trailing features, physical horizons, and numerical CRPS."""
import numpy as np
import pandas as pd


def steps(duration_ms, resolution_ms):
    if duration_ms < resolution_ms or duration_ms % resolution_ms:
        raise ValueError(f'Unsupported duration {duration_ms} ms at {resolution_ms} ms')
    return duration_ms // resolution_ms


def continuous_segments(d, gap):
    t, p = d.time_s.to_numpy(), d.power_w.to_numpy()
    good = np.isfinite(t) & np.isfinite(p) & (p >= 0)
    start = good & np.r_[True, (~good[:-1]) | (np.diff(t) <= 0) | (np.diff(t) > gap)]
    ids = np.cumsum(start)
    return [(int(sid), d.iloc[np.flatnonzero((ids == sid) & good)].copy())
            for sid in np.unique(ids[good])]


def bin_segment(d, resolution_ms, native_ms=20, coverage=.8):
    """Recorded sample mean over (end-r,end], available at end; no fill/interpolation.

    Count coverage is relative to nominal recording cadence, NOT sensor coverage.
    Partial bins at segment edges are excluded, even if their count is sufficient.
    """
    r = resolution_ms / 1000
    x = d[['time_s', 'power_w']].copy()
    x['bin'] = np.ceil(x.time_s / r - 1e-10).astype('int64')
    z = x.groupby('bin').agg(power_w=('power_w', 'mean'), n=('power_w','size'),
                              source_max_s=('time_s', 'max'), source_min_s=('time_s','min'))
    z['time_s'] = z.index.to_numpy() * r
    z['coverage'] = z.n / (resolution_ms / native_ms)
    valid = ((z.coverage >= coverage - 1e-6) &
             (z.time_s - r >= x.time_s.iloc[0] - 1e-9) &
             (z.time_s <= x.time_s.iloc[-1] + 1e-9))
    z = z[valid].copy()
    assert (z.source_max_s <= z.time_s + 1e-9).all()
    z['subsegment'] = (z.index.to_series().diff() != 1).cumsum().to_numpy()
    return z.reset_index(names='bin')


def features(p, resolution_ms, history_ms):
    k = steps(history_ms, resolution_ms)
    if k == 1: return np.asarray(p,dtype=np.float32)[:,None]
    s = pd.Series(p)
    # k bins cover a physical interval of k*r, including the current bin.
    z = {'current': s}
    for lag in sorted(set([max(1,k//4), max(1,k//2), k-1])):
        if 0 < lag < k: z[f'lag_{lag*resolution_ms}ms'] = s.shift(lag)
    for w in sorted(set([max(2,k//4), max(2,k//2), k])):
        if w > k: continue
        roll = s.rolling(w, min_periods=w)
        for name in ['mean','std','min','max']:
            z[f'{name}_{w*resolution_ms}ms'] = roll.std(ddof=0) if name == "std" else getattr(roll,name)()
        z[f'slope_{w*resolution_ms}ms'] = (s-s.shift(w-1))/((w-1)*resolution_ms/1000) if w>1 else s*0
    return pd.DataFrame(z).to_numpy(dtype=np.float32)


def partition(t, cut1, cut2):
    return np.where(t < cut1, 0, np.where(t < cut2, 1, 2))


def samples(z, resolution_ms, horizon_ms, histories_ms, cut1, cut2, cadence_ms):
    """One contiguous run/segment only. All conditions share longest-history targets.

    Issuance and target must both lie in the same split; history may precede split.
    """
    h = steps(horizon_ms, resolution_ms)
    hs = [v for v in histories_ms if v >= resolution_ms and v % resolution_ms == 0]
    k = steps(max(hs), resolution_ms)
    if len(z) <= k+h: return None
    t=z.time_s.to_numpy(); p=z.power_w.to_numpy()
    i=np.arange(k-1, len(z)-h); j=i+h
    cadence=max(resolution_ms,cadence_ms)
    assert cadence % resolution_ms == 0
    mask=((partition(t[i],cut1,cut2)==partition(t[j],cut1,cut2)) &
          (z.bin.to_numpy()[j] % (cadence//resolution_ms)==0))
    i,j=i[mask],j[mask]
    meta=pd.DataFrame({'issue_s':t[i],'target_s':t[j], 'target_bin':z.bin.to_numpy()[j],
                       'y':p[j], 'current':p[i], 'split':partition(t[j],cut1,cut2)})
    assert np.allclose(meta.target_s-meta.issue_s, horizon_ms/1000)
    return meta, {v:features(p,resolution_ms,v)[i] for v in hs}


def rearrange(pred):
    pred=np.asarray(pred)
    if pred.ndim != 2 or not np.isfinite(pred).all(): raise ValueError('Invalid quantiles')
    crossing=float(np.any(np.diff(pred,axis=1)<0,axis=1).mean()) if len(pred) else 0.
    return np.sort(pred,axis=1),crossing


def crps(y, pred, quantiles):
    """Trapezoidal 2*pinball integral; constant Q tails to q=0 and q=1.

    Finite-grid numerical approximation, identical for empirical and ML forecasts.
    """
    q=np.r_[0.,quantiles,1.]
    v=np.column_stack([pred[:,0],pred,pred[:,-1]])
    e=np.asarray(y)[:,None]-v
    loss=np.maximum(q*e,(q-1)*e)
    return 2*np.trapz(loss,q,axis=1)


def skill(model, reference):
    return float(1-model/reference) if reference > 0 else np.nan
