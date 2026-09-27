from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
COLORS = {"history_only": "#0072B2", "history_static_s1": "#CC79A7", "history_static_s2": "#D55E00"}


def save(out, name, fig, data):
    (out / "figures").mkdir(exist_ok=True)
    (out / "figure_data").mkdir(exist_ok=True)
    data.to_csv(out / "figure_data" / f"{name}.csv", index=False)
    fig.tight_layout()
    fig.savefig(out / "figures" / f"{name}.png", dpi=300)
    fig.savefig(out / "figures" / f"{name}.pdf")
    plt.close(fig)


def make(out):
    out = Path(out)
    aggregate = pd.read_csv(out / "aggregate_metrics.csv")
    gain = pd.read_csv(out / "paired_gain_ci.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for name, group in aggregate.groupby("condition"):
        group = group.sort_values("horizon_ms")
        axes[0].plot(group.horizon_ms / 1000, group.crps_w, "o-", color=COLORS[name], label=name)
    for name, group in gain.groupby("condition"):
        group = group.sort_values("horizon_ms")
        y = group.gain_of_run_balanced_scores_pct
        axes[1].errorbar(group.horizon_ms / 1000, y,
            yerr=[y-group.ci_low_pct, group.ci_high_pct-y], fmt="o-", color=COLORS[name],
            capsize=2, label=name)
    axes[0].set(xlabel="Forecast horizon (s)", ylabel="Run-balanced CRPS approximation (W)")
    axes[1].axhline(0, color="0.5", ls="--")
    axes[1].set(xlabel="Forecast horizon (s)", ylabel="Paired gain (%) with 95% run-cluster CI")
    axes[0].legend(fontsize=8)
    save(out, "F1_horizon_crps_grid", fig, aggregate.merge(gain, on=["condition", "horizon_ms"], how="left"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, metric, label in zip(axes, ("mae_w", "p95_w", "p99_w"),
                                 ("MAE", "p95 absolute error", "p99 absolute error")):
        for name, group in aggregate.groupby("condition"):
            group = group.sort_values("horizon_ms")
            ax.plot(group.horizon_ms / 1000, group[metric], "o-", color=COLORS[name], label=name)
        ax.set(xlabel="Forecast horizon (s)", ylabel=f"{label} (W)")
    axes[0].legend(fontsize=8)
    save(out, "F2_horizon_errors_grid", fig, aggregate)
    transition = pd.read_csv(out / "transition_metrics.csv")
    summary = transition.groupby(["horizon_ms", "condition", "transition"], as_index=False).agg(
        n_targets=("n_targets", "sum"), n_runs=("run_id", "nunique"), crps_w=("crps_w", "mean"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, class_name in zip(axes, ("rise", "drop", "steady")):
        for name, group in summary[summary.transition == class_name].groupby("condition"):
            group = group.sort_values("horizon_ms")
            ax.plot(group.horizon_ms / 1000, group.crps_w, "o-", color=COLORS[name], label=name)
        ax.set(xlabel="Forecast horizon (s)", ylabel="CRPS approximation (W)", title=class_name)
    axes[0].legend(fontsize=8)
    save(out, "F3_transition_crps_grid", fig, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/grid_fixed_20260926"))
    make(parser.parse_args().out)
