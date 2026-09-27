from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
HORIZONS = range(200, 2001, 200)


def make(out):
    out = Path(out)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    aggregate = pd.read_csv(out / "aggregate_metrics.csv")
    gain = pd.read_csv(out / "paired_gain_ci.csv")
    aggregate.merge(gain, on=["condition", "horizon_ms"], how="left").to_csv(
        tables / "T4_horizon_performance.csv", index=False)
    selection = pd.concat([pd.read_csv(out / f"h{h}" / "selected_hyperparameters.csv") for h in HORIZONS],
                          ignore_index=True)
    selection.to_csv(out / "selected_hyperparameters.csv", index=False)
    selection.to_csv(tables / "T3_selected_parameters.csv", index=False)
    per = pd.concat([pd.read_csv(out / f"h{h}" / "per_run_metrics.csv") for h in HORIZONS], ignore_index=True)
    wide = per.pivot(index=["horizon_ms", "run_id"], columns="condition", values="crps_w").dropna()
    for condition in ("history_static_s1", "history_static_s2"):
        wide[condition + "_gain_pct"] = 100 * (1 - wide[condition]/wide.history_only)
    wide.reset_index().to_csv(tables / "T5_metadata_by_run_horizon.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/grid_fixed_20260926"))
    make(parser.parse_args().out)
