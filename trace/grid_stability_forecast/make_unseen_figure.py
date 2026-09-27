"""Categorical held-out-cohort gain figure from saved summary CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def run(out: Path):
    out = Path(out)
    data = pd.read_csv(out / "cohort_gain_ci.csv")
    data = data.loc[data.condition.eq("history_static_s2")].copy()
    if len(data) != 4:
        raise ValueError("expected four held-out GPU/workload cohorts")
    labels = data.held_out_cohort.str.replace("__", " / ", regex=False)
    y = np.arange(len(data))
    gains = data.gain_pct.to_numpy()
    errors = np.vstack([gains-data.ci_low_pct.to_numpy(),
                        data.ci_high_pct.to_numpy()-gains])
    colors = ["#0072B2" if v >= 0 else "#D55E00" for v in gains]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(y, gains, color=colors, alpha=.85)
    ax.errorbar(gains, y, xerr=errors, fmt="none", ecolor="#333333",
                capsize=3, linewidth=1.2)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("S2 gain in run-balanced CRPS (%)")
    ax.set_title("Held-out GPU/workload combination, +1 s")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "cohort_gain.png", dpi=300)
    fig.savefig(figures / "cohort_gain.pdf")
    plt.close(fig)
    data.to_csv(figures / "cohort_gain_data.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
        default=HERE / "results/unseen_condition_20260926")
    run(parser.parse_args().out)
