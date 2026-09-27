"""Recompute run-clustered cohort effects from held-out prediction metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def run(out: Path):
    out = Path(out)
    per = pd.read_csv(out / "per_run_metrics.csv")
    pivot = per.pivot(index=["held_out_cohort", "run_id", "seed"],
                      columns="condition", values="crps_w").reset_index()
    averaged = pivot.groupby(["held_out_cohort", "run_id"])[
        ["history_only", "history_static_s1", "history_static_s2"]].mean().reset_index()
    rng = np.random.default_rng(2026)
    rows = []
    for cohort, group in averaged.groupby("held_out_cohort"):
        base = group.history_only.to_numpy()
        for condition in ("history_static_s1", "history_static_s2"):
            meta = group[condition].to_numpy()
            draw = rng.integers(0, len(base), (2000, len(base)))
            boot = 100 * (1 - meta[draw].mean(axis=1) / base[draw].mean(axis=1))
            rows.append(dict(held_out_cohort=cohort, condition=condition,
                n_runs=len(base), history_crps_w=base.mean(), metadata_crps_w=meta.mean(),
                gain_pct=100*(1-meta.mean()/base.mean()),
                improved_runs=int((meta<base).sum()),
                ci_low_pct=np.quantile(boot,.025), ci_high_pct=np.quantile(boot,.975),
                bootstrap_replicates=2000))
    result = pd.DataFrame(rows)
    result.to_csv(out / "cohort_gain_ci.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
        default=HERE / "results/unseen_condition_20260926")
    print(run(parser.parse_args().out).to_string(index=False))
