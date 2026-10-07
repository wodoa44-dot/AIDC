from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

GPU = [f"gpu{i}_power_W" for i in range(8)]
TOKEN = re.compile(r"(ModelSize|ImageSize|BatchSize|SeqLen|SequenceLength|ZeroStage|ZeROStage)([0-9]+)", re.I)


def static_metadata(path: str) -> dict:
    p = Path(path)
    s = str(p)
    out = {"hardware": "B200" if "B200" in s else "H100" if "H100" in s else "unknown",
           "workload": "image_generation" if "Image Generation" in s else "llm_training",
           "gpu_count": 8}
    names = {"modelsize": "model_size_token", "imagesize": "image_size", "batchsize": "batch_size",
             "seqlen": "sequence_length", "sequencelength": "sequence_length", "zerostage": "zero_stage"}
    for k, v in TOKEN.findall(p.stem):
        out[names[k.lower()]] = int(v)
    for k in set(names.values()) - out.keys():
        out[k] = np.nan
    return out


def bin_samples(t: np.ndarray, p: np.ndarray, bin_s=.2, min_coverage=.8, gap_factor=5.) -> pd.DataFrame:
    """Completed right-edge bins. Samples are instantaneous; sensor averaging is unknown."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    good = np.isfinite(t) & np.isfinite(p)
    t, p = t[good], p[good]
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be finite, strictly increasing and have >=3 observations")
    dt = float(np.median(np.diff(t)))
    ix = np.floor((t - t[0]) / bin_s).astype(int)
    n = ix.max() + 1
    count = np.bincount(ix, minlength=n)
    sums = np.bincount(ix, weights=p, minlength=n)
    mean = np.divide(sums, count, out=np.full(n, np.nan), where=count > 0)
    # A gap invalidates both bordering bins, so no history crosses it.
    bad_gap = np.where(np.diff(t) > gap_factor * dt)[0]
    valid = count >= np.ceil(min_coverage * bin_s / dt)
    valid[ix[bad_gap]] = False
    valid[ix[bad_gap + 1]] = False
    mean[~valid] = np.nan
    segment = np.cumsum(~valid)
    return pd.DataFrame({"bin": np.arange(n), "right_s": (np.arange(n) + 1) * bin_s,
                         "power_w": mean, "count": count, "coverage_count": count * dt / bin_s,
                         "segment": segment, "valid": valid, "observed_dt_s": dt})


def load_highres(path: str, **kwargs) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=["timestamp"] + GPU)
    times = pd.to_datetime(d.timestamp, errors="coerce")
    t = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    power = d[GPU].sum(axis=1, min_count=8).to_numpy()
    return bin_samples(t, power, **kwargs)
