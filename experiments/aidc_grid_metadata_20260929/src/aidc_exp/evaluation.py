from __future__ import annotations
import numpy as np
import pandas as pd


def pinball(y, q, alpha):
    e = np.asarray(y) - np.asarray(q)
    return np.maximum(alpha * e, (alpha - 1) * e)


def crps_grid(y, predictions, alphas):
    """2*trapezoid pinball with constant tail extension to alpha=0 and 1."""
    a = np.asarray(alphas, float)
    q = np.asarray(predictions, float)
    if np.any(np.diff(a) <= 0) or a[0] <= 0 or a[-1] >= 1:
        raise ValueError("strictly increasing interior quantiles required")
    losses = np.stack([pinball(y, q[..., i], v) for i, v in enumerate(a)], axis=-1)
    aa = np.r_[0., a, 1.]
    ll = np.concatenate([losses[..., :1], losses, losses[..., -1:]], axis=-1)
    return 2 * np.trapz(ll, aa, axis=-1)


def metrics(pred: pd.DataFrame, quantiles: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    qcols = [f"q{q:g}" for q in quantiles]
    out, tails = [], []
    for key, g in pred.groupby(["fold", "seed", "model", "horizon", "run_id"], sort=False):
        y = g.actual_w.to_numpy(); qs = g[qcols].to_numpy()
        med = qs[:, np.argmin(abs(np.asarray(quantiles) - .5))]
        e = med - y; ae = abs(e)
        score = crps_grid(y, qs, quantiles)
        row = dict(zip(["fold", "seed", "model", "horizon", "run_id"], key))
        row.update(n=len(g), crps=float(np.mean(score)), mae=float(np.mean(ae)), rmse=float(np.sqrt(np.mean(e*e))),
                   bias=float(np.mean(e)), p95_abs=float(np.percentile(ae,95)), p99_abs=float(np.percentile(ae,99)),
                   max_abs=float(ae.max()), max_issuance_s=float(g.issuance_s.iloc[ae.argmax()]),
                   under_rate=float(np.mean(e<0)), over_rate=float(np.mean(e>0)),
                   under_mean_w=float(np.mean(-e[e<0])) if np.any(e<0) else 0.,
                   over_mean_w=float(np.mean(e[e>0])) if np.any(e>0) else 0.)
        out.append(row)
        tails.append({**row, "worst_1pct_mean_w": float(np.mean(np.sort(ae)[-max(1,int(np.ceil(.01*len(ae)))):])),
                      "worst_5pct_mean_w": float(np.mean(np.sort(ae)[-max(1,int(np.ceil(.05*len(ae)))):]))})
    return pd.DataFrame(out), pd.DataFrame(tails)


def aggregate(run_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = ["crps", "mae", "rmse", "bias", "p95_abs", "p99_abs"]
    # Each run first averages seeds, then every run receives equal weight.
    rs = run_metrics.groupby(["model", "horizon", "run_id"], as_index=False)[numeric].mean()
    return rs.groupby(["model", "horizon"], as_index=False)[numeric].mean()


def paired_bootstrap(run_metrics: pd.DataFrame, b=2000, seed=2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows=[]
    s = run_metrics.groupby(["model", "horizon", "run_id"], as_index=False).crps.mean()
    for h in sorted(s.horizon.unique()):
        z = s[s.horizon==h].pivot(index="run_id", columns="model", values="crps")
        if not {"P2", "P4"}.issubset(z.columns): continue
        z=z[["P2","P4"]].dropna(); n=len(z)
        if not n: continue
        pairs=z.to_numpy(); ids=rng.integers(0,n,size=(b,n)); bs=pairs[ids].mean(axis=1)
        gain=np.where(bs[:,0]>0,100*(1-bs[:,1]/bs[:,0]),np.nan)
        actual=100*(1-pairs[:,1].mean()/pairs[:,0].mean()) if pairs[:,0].mean()>0 else np.nan
        rows.append({"horizon":h,"independent_groups_assumed":n,"gain_pct":actual,
                     "ci_low":np.nanpercentile(gain,2.5),"ci_high":np.nanpercentile(gain,97.5),
                     "ci_type":"fixed-prediction group bootstrap; session independence unverified"})
    return pd.DataFrame(rows)
