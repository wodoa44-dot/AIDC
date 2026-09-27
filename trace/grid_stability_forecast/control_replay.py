from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import ramp_metrics, replay_filler

HERE = Path(__file__).resolve().parent


def replay(out):
    out = Path(out)
    pred = pd.read_parquet(out / "per_target_predictions.parquet")
    rows, curves, sensitivity = [], [], []
    for run_id, r in pred.groupby("run_id"):
        reference = r[r.condition == "history_only"].sort_values("target_end_s")
        for segment, seg in reference.groupby("segment_id"):
            seg = seg.sort_values("target_end_s")
            breaks = np.r_[True, ~np.isclose(np.diff(seg.target_end_s), .2, atol=1e-5)]
            for piece, g in seg.groupby(np.cumsum(breaks)):
                if len(g) < 30:
                    continue
                t = g.target_end_s.to_numpy(float)
                power = g.y.to_numpy(float)
                keyed = ["run_id", "segment_id", "target_end_s"]
                aligned = pred[(pred.run_id == run_id) & (pred.segment_id == segment) &
                               (pred.condition == "history_static_s2")].set_index("target_end_s")
                if not np.isin(t, aligned.index).all():
                    continue
                forecasts = {"history_only": g.q50.to_numpy(float),
                             "history_static_s2": aligned.loc[t, "q50"].to_numpy(float)}
                scenarios = {"uncontrolled": (power, np.zeros(len(power))),
                             "reactive": replay_filler(power, power)}
                for condition, forecast in forecasts.items():
                    # Every value here was issued 1 s before its target timestamp.
                    scenarios[condition] = replay_filler(power, forecast)
                for name, (grid, filler) in scenarios.items():
                    metric = ramp_metrics(grid)
                    rows.append(dict(run_id=run_id, segment_id=segment, piece=piece, condition=name,
                        n_bins=len(g), duration_s=len(g)*.2, filler_energy_j=float(filler.sum()*.2),
                        delay_s=.2, capacity_w=500., ramp_limit_w_s=200., assessment="idealized sensitivity",
                        **metric))
                    if len(curves) < 4000:
                        curves.extend(dict(run_id=run_id, time_s=tt, condition=name, power_w=pp,
                                           filler_w=ff) for tt, pp, ff in zip(t[:200], grid[:200], filler[:200]))
                for delay in (0, 1, 3, 5):
                    for cap in (250., 500., 1000.):
                        grid, filler = replay_filler(power, forecasts["history_only"],
                            delay_bins=delay, capacity=cap)
                        sensitivity.append(dict(run_id=run_id, segment_id=segment, piece=piece,
                            delay_s=delay*.2, capacity_w=cap, duration_s=len(g)*.2,
                            filler_energy_j=float(filler.sum()*.2), **ramp_metrics(grid)))
    pd.DataFrame(rows).to_csv(out / "control_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "control_trajectories.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(out / "control_sensitivity.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/actual_20260926"))
    replay(parser.parse_args().out)
