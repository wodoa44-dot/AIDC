from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
COLORS = {"history_only": "#0072B2", "history_static_s1": "#CC79A7", "history_static_s2": "#D55E00",
          "reactive": "#009E73", "uncontrolled": "#333333"}


def save(fig, out, name, data):
    figures = out / "figures"
    source = out / "figure_data"
    figures.mkdir(exist_ok=True)
    source.mkdir(exist_ok=True)
    data.to_csv(source / f"{name}.csv", index=False)
    fig.tight_layout()
    fig.savefig(figures / f"{name}.png", dpi=300)
    fig.savefig(figures / f"{name}.pdf")
    plt.close(fig)


def figures(out):
    out = Path(out)
    agg = pd.read_csv(out / "aggregate_metrics.csv")
    ci = pd.read_csv(out / "paired_gain_ci.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for condition, g in agg.groupby("condition"):
        ax.plot(g.horizon_ms / 1000, g.crps_w, marker="o", color=COLORS[condition], label=condition)
    ax.set(xlabel="Forecast horizon (s)", ylabel="Run-balanced quantile CRPS approximation (W)",
           title=f"F1: +1 s grouped pilot, {int(ci.n_runs.iloc[0])} test runs")
    ax.legend()
    save(fig, out, "F1_horizon_crps_pilot", agg.merge(ci, on=["horizon_ms", "condition"], how="left"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (column, label) in zip(axes, (("mae_w", "MAE"), ("p95_w", "p95 absolute error"),
                                          ("p99_w", "p99 absolute error"))):
        for condition, g in agg.groupby("condition"):
            ax.plot(g.horizon_ms / 1000, g[column], "o-", color=COLORS[condition], label=condition)
        ax.set(xlabel="Horizon (s)", ylabel=f"{label} (W)")
    axes[0].legend(fontsize=8)
    save(fig, out, "F2_horizon_errors_pilot", agg)
    trans = pd.read_csv(out / "transition_metrics.csv")
    t = trans.groupby(["condition", "horizon_ms", "transition"], as_index=False).agg(
        crps_w=("crps_w", "mean"), n_targets=("n_targets", "sum"), n_runs=("run_id", "nunique"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, name in zip(axes, ("rise", "drop", "steady")):
        for condition, g in t[t.transition == name].groupby("condition"):
            ax.plot(g.horizon_ms / 1000, g.crps_w, "o-", color=COLORS[condition], label=condition)
        ax.set(xlabel="Horizon (s)", ylabel="CRPS approximation (W)", title=name)
    axes[0].legend(fontsize=8)
    save(fig, out, "F3_transition_crps_pilot", t)
    cal = pd.read_csv(out / "calibration_metrics.csv")
    c = cal.groupby(["condition", "nominal"], as_index=False).empirical_coverage.mean()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="0.4", label="ideal")
    for condition, g in c.groupby("condition"):
        ax.plot(g.nominal, g.empirical_coverage, "o-", color=COLORS[condition], label=condition)
    ax.set(xlabel="Nominal quantile", ylabel="Empirical coverage", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save(fig, out, "F4_quantile_coverage_pilot", c)
    pred_path = out / "per_target_predictions.parquet"
    if pred_path.exists():
        pred = pd.read_parquet(pred_path)
        hist = pred[pred.condition == "history_only"].sort_values(["run_id", "segment_id", "target_end_s"]).copy()
        hist["step_w"] = hist.groupby(["run_id", "segment_id"]).y.diff()
        hist["step_s"] = hist.groupby(["run_id", "segment_id"]).target_end_s.diff()
        hist.loc[~hist.step_s.between(.19999, .20001), "step_w"] = float("nan")
        hist["baseline_error"] = (hist.y - hist.q50).abs()
        keyed = pred[pred.condition == "history_static_s2"].set_index(["run_id", "segment_id", "target_end_s"])
        candidates = hist.set_index(["run_id", "segment_id", "target_end_s"])
        candidates["meta_error"] = (keyed.y-keyed.q50).abs()
        candidates["improvement"] = candidates.baseline_error-candidates.meta_error
        selections = []
        for direction in ("rise", "drop"):
            sub = candidates[candidates.step_w.ge(100) if direction == "rise" else candidates.step_w.le(-100)]
            if sub.empty:
                continue
            median = sub.baseline_error.median()
            choices = {"median baseline error": (sub.baseline_error-median).abs().idxmin(),
                       "worst baseline error": sub.baseline_error.idxmax(),
                       "largest meta improvement": sub.improvement.idxmax(),
                       "largest meta worsening": sub.improvement.idxmin()}
            selections.extend((direction, rule, key) for rule, key in choices.items())
        if selections:
            fig, axes = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
            waveform = []
            for ax, (direction, rule, key) in zip(axes.T.flat, selections):
                run, segment, onset = key
                view = pred[(pred.run_id == run) & (pred.segment_id == segment) &
                    pred.target_end_s.between(onset-2, onset+2)].copy()
                for condition, g in view.groupby("condition"):
                    g = g.sort_values("target_end_s")
                    relative = g.target_end_s-onset
                    if condition == "history_only":
                        ax.plot(relative, g.y, color="black", label="observed")
                    ax.plot(relative, g.q50, color=COLORS[condition], label=condition)
                    if condition == "history_static_s2":
                        ax.fill_between(relative, g["q_0.1"], g["q_0.9"], color=COLORS[condition], alpha=.15)
                    g["relative_s"] = relative
                    g["selection_rule"] = rule
                    g["direction"] = direction
                    waveform.append(g[["run_id", "segment_id", "target_end_s", "relative_s", "condition",
                                       "direction", "selection_rule", "y", "q50", "q_0.1", "q_0.9"]])
                ax.axvline(0, ls="--", color="0.5")
                ax.set(title=f"{direction}: {rule}", ylabel="GPU power (W)")
            for ax in axes[-1]:
                ax.set_xlabel("Time from observed 200 ms threshold crossing (s)")
            axes[0, 0].legend(fontsize=7)
            save(fig, out, "F5_event_aligned_selected_cases", pd.concat(waveform, ignore_index=True))
    trajectories = out / "control_trajectories.csv"
    if trajectories.exists():
        data = pd.read_csv(trajectories)
        if not data.empty:
            first = data.run_id.iloc[0]
            d = data[data.run_id == first]
            fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
            for name, g in d.groupby("condition"):
                axes[0].plot(g.time_s, g.power_w, color=COLORS[name], label=name)
                if name != "uncontrolled":
                    axes[1].plot(g.time_s, g.filler_w, color=COLORS[name], label=name)
            axes[0].set(ylabel="GPU power proxy (W)", title="F6: idealized filler replay")
            axes[1].set(xlabel="Time (s)", ylabel="Filler command (W)")
            axes[0].legend(ncol=2, fontsize=8)
            save(fig, out, "F6_control_replay_idealized", d)
    path = out / "control_sensitivity.csv"
    if path.exists():
        data = pd.read_csv(path)
        d = data.groupby(["delay_s", "capacity_w"], as_index=False).agg(
            ramp_exceed_fraction=("ramp_exceed_fraction", "mean"), filler_energy_j=("filler_energy_j", "mean"))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for cap, g in d.groupby("capacity_w"):
            for ax, metric, unit in zip(axes, ("ramp_exceed_fraction", "filler_energy_j"), ("Fraction", "J")):
                ax.plot(g.delay_s, g[metric], "o-", label=f"{cap:g} W")
                ax.set(xlabel="Assumed delay (s)", ylabel=unit)
        axes[0].legend(title="Filler capacity")
        save(fig, out, "F8_actuator_sensitivity_idealized", d)
    path = out / "learning_curve.csv"
    if path.exists():
        d = pd.read_csv(path)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for name, g in d.groupby("condition"):
            g = g.sort_values("actual_train_runs")
            for ax, metric, label in zip(axes, ("crps_w", "transition_p95_w"),
                                         ("Run-balanced CRPS approximation (W)", "Transition p95 error (W)")):
                ax.plot(g.actual_train_runs, g[metric], "o-", color=COLORS[name], label=name)
                ax.set(xlabel="Independent train runs", ylabel=label)
        axes[0].legend(fontsize=8)
        save(fig, out, "F7_independent_run_learning_curve", d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/actual_20260926"))
    figures(parser.parse_args().out)
