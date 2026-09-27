#!/usr/bin/env python3
"""Create consolidated error visualizations for forecasting experiments.

The model comparison uses the matched old_frozen and reliability_selection
predictions stored by the failure-aware experiment.  Predictability results are
plotted separately because their sampling grids and evaluation populations are
different from the 200 ms / 1 s / 2 s model experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPROVEMENT = Path(__file__).resolve().parent / "improvements" / "failure_aware_20260921_v1" / "artifacts" / "full"
DEFAULT_PREDICTABILITY = TRACE_ROOT / "predictability" / "outputs" / "main_20260914"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "visualizations" / "error_comparison_20260921"

COLORS = {
    "Original": "#7a8793",
    "Failure-aware": "#147d73",
    "Unconditional": "#8c8c8c",
    "Current only": "#4c78a8",
    "History": "#f58518",
}


def read_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfb",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "font.size": 10,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def select_model_rows(data: pd.DataFrame) -> pd.DataFrame:
    selected = data[
        (data["condition"] == "history_meta")
        & (data["representation"] == "full15")
        & (data["model"].isin(["old_frozen", "reliability_selection"]))
    ].copy()
    if "seed" in selected.columns:
        selected = selected[selected["seed"] == 42]
    if "weighting" in selected.columns:
        selected = selected[selected["weighting"] == "run_balanced"]
    selected["model_label"] = selected["model"].map(
        {"old_frozen": "Original", "reliability_selection": "Failure-aware"}
    )
    return selected


def model_summary(improvement_root: Path, output: Path) -> pd.DataFrame:
    aggregate = select_model_rows(read_csv(improvement_root, "aggregate_metrics.csv"))
    key = ["assessment", "horizon_ms", "model_label"]
    if aggregate.duplicated(key).any():
        duplicates = aggregate.loc[aggregate.duplicated(key, keep=False), key]
        raise ValueError(f"Model selection did not produce unique rows:\n{duplicates}")

    metrics = [
        ("crps_w", "CRPS", "W"),
        ("mae_w", "Mean absolute error", "W"),
        ("abs_error_p95_w", "95th percentile absolute error", "W"),
        ("exceedance_500_fraction", "Absolute error > 500 W", "%"),
    ]
    assessments = ["grouped_outer", "reused_holdout"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), layout="constrained", sharey="col")
    for row, assessment in enumerate(assessments):
        subset = aggregate[aggregate["assessment"] == assessment]
        for col, (metric, title, unit) in enumerate(metrics):
            ax = axes[row, col]
            factor = 100 if unit == "%" else 1
            for label, offset in [("Original", 8), ("Failure-aware", -12)]:
                values = (subset[subset["model_label"] == label]
                          .set_index("horizon_ms").reindex([1000, 2000])[metric]
                          .to_numpy(dtype=float) * factor)
                ax.plot([1, 2], values, marker="o", markersize=7,
                        linewidth=2.5, color=COLORS[label], label=label)
                for x, value in zip([1, 2], values):
                    ax.annotate(f"{value:,.0f}{unit if unit == '%' else ''}",
                                (x, value), xytext=(0, offset),
                                textcoords="offset points", ha="center",
                                va="bottom" if offset > 0 else "top",
                                color=COLORS[label], fontsize=8)
            ax.set_xticks([1, 2], ["1 s", "2 s"])
            ax.set_xlim(0.8, 2.2)
            ax.set_title(title if row == 0 else "", fontsize=11)
            ax.set_ylabel(("Grouped test" if row == 0 else "Reused holdout")
                          + f" · {unit}")
            ax.set_xlabel("Forecast horizon")
            ax.grid(axis="x", visible=False)
    fig.suptitle("Forecast error by horizon · original (gray) vs failure-aware (teal)",
                 fontsize=16, fontweight="bold")
    save_figure(fig, output, "model_error_overview.png")

    original = aggregate[aggregate["model_label"] == "Original"].set_index(["assessment", "horizon_ms"])
    improved = aggregate[aggregate["model_label"] == "Failure-aware"].set_index(["assessment", "horizon_ms"])
    rows: list[dict[str, object]] = []
    for idx in original.index.intersection(improved.index):
        item: dict[str, object] = {"assessment": idx[0], "horizon_ms": idx[1]}
        for metric, _, _ in metrics:
            old = float(original.loc[idx, metric])
            new = float(improved.loc[idx, metric])
            item[f"original_{metric}"] = old
            item[f"failure_aware_{metric}"] = new
            item[f"reduction_{metric}_pct"] = 100.0 * (old - new) / old
        rows.append(item)
    summary = pd.DataFrame(rows).sort_values(["assessment", "horizon_ms"])
    summary.to_csv(output / "model_comparison_summary.csv", index=False)
    return summary


def model_transition_plot(improvement_root: Path, output: Path) -> None:
    data = select_model_rows(read_csv(improvement_root, "transition_metrics.csv"))
    strata = ["falling", "rising", "stable"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained", sharey=True)
    for row, assessment in enumerate(["grouped_outer", "reused_holdout"]):
        for col, horizon in enumerate([1000, 2000]):
            ax = axes[row, col]
            subset = data[(data["assessment"] == assessment) & (data["horizon_ms"] == horizon)]
            for label, offset in [("Original", 8), ("Failure-aware", -12)]:
                values = (subset[subset["model_label"] == label]
                          .set_index("transition_stratum").reindex(strata)["mae_w"]
                          .to_numpy(dtype=float))
                ax.plot(range(3), values, marker="o", markersize=7,
                        linewidth=2.5, color=COLORS[label], label=label)
                for x, value in enumerate(values):
                    ax.annotate(f"{value:,.0f}", (x, value), xytext=(0, offset),
                                textcoords="offset points", ha="center",
                                va="bottom" if offset > 0 else "top",
                                color=COLORS[label], fontsize=8)
            ax.set_xticks(range(3), [name.title() for name in strata])
            ax.set_xlim(-0.2, 2.2)
            ax.set_title(f"{'Grouped test' if assessment == 'grouped_outer' else 'Reused holdout'} · {horizon // 1000} s",
                         loc="left")
            ax.set_ylabel("MAE (W)")
            ax.set_xlabel("Endpoint transition")
            ax.grid(axis="x", visible=False)
    fig.suptitle("Transition error · original (gray) vs failure-aware (teal)",
                 fontsize=16, fontweight="bold")
    save_figure(fig, output, "model_transition_errors.png")


def model_per_run_plot(improvement_root: Path, output: Path) -> None:
    data = select_model_rows(read_csv(improvement_root, "per_run_metrics.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for row, assessment in enumerate(["grouped_outer", "reused_holdout"]):
        for col, horizon in enumerate([1000, 2000]):
            ax = axes[row, col]
            subset = data[(data["assessment"] == assessment) & (data["horizon_ms"] == horizon)]
            wide = subset.pivot(index="run_id", columns="model_label", values="mae_w").dropna()
            cohorts = subset.drop_duplicates("run_id").set_index("run_id")["cohort"]
            cohort_names = sorted(cohorts.loc[wide.index].unique())
            cmap = plt.get_cmap("tab10")
            for index, cohort in enumerate(cohort_names):
                run_ids = cohorts[cohorts == cohort].index.intersection(wide.index)
                ax.scatter(
                    wide.loc[run_ids, "Original"],
                    wide.loc[run_ids, "Failure-aware"],
                    s=45,
                    alpha=0.85,
                    color=cmap(index),
                    label=cohort.replace("__", " / "),
                )
            limit = float(np.nanmax(wide.to_numpy())) * 1.06
            ax.plot([0, limit], [0, limit], color="#222222", linestyle="--", linewidth=1)
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{assessment.replace('_', ' ').title()} | {horizon / 1000:g} s")
            ax.set_xlabel("Original per-run MAE (W)")
            ax.set_ylabel("Failure-aware per-run MAE (W)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle("Paired run errors (points below diagonal improved)", y=1.07, fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output, "model_per_run_paired.png")


def model_error_quantiles(improvement_root: Path, output: Path) -> None:
    """Compare the body and tail of absolute forecast errors."""
    data = select_model_rows(read_csv(improvement_root, "aggregate_metrics.csv"))
    data = data[data["weighting"] == "run_balanced"]
    quantiles = [(50, "abs_error_p50_w"), (90, "abs_error_p90_w"),
                 (95, "abs_error_p95_w"), (99, "abs_error_p99_w")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained", sharey=True)
    for row, assessment in enumerate(["grouped_outer", "reused_holdout"]):
        for col, horizon in enumerate([1000, 2000]):
            ax = axes[row, col]
            pair = data[(data["assessment"] == assessment)
                        & (data["horizon_ms"] == horizon)].set_index("model_label")
            for label in ["Original", "Failure-aware"]:
                values = [float(pair.loc[label, name]) for _, name in quantiles]
                ax.plot(range(len(quantiles)), values, marker="o", linewidth=2.5,
                        markersize=7, color=COLORS[label], label=label)
                ax.annotate(f"{values[-1]:,.0f} W", (3, values[-1]),
                            xytext=(8, 10 if label == "Original" else -10), textcoords="offset points",
                            color=COLORS[label], va="center", fontsize=9)
            ax.set_title(f"{'Grouped test' if assessment == 'grouped_outer' else 'Reused holdout'} · {horizon // 1000} s",
                         loc="left")
            ax.set_xticks(range(len(quantiles)), ["50th", "90th", "95th", "99th"])
            ax.set_xlim(-0.2, 3.8)
            ax.set_xlabel("Absolute error percentile")
            ax.set_ylabel("Absolute error (W)")
    fig.suptitle("Absolute error percentiles · original (gray) vs failure-aware (teal)",
                 fontsize=16, fontweight="bold")
    save_figure(fig, output, "model_error_quantiles.png")


def model_tolerance_curves(improvement_root: Path, output: Path) -> None:
    """Show how often absolute errors exceed several watt thresholds."""
    data = select_model_rows(read_csv(improvement_root, "tolerance_exceedance.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained", sharex=True, sharey=True)
    for row, assessment in enumerate(["grouped_outer", "reused_holdout"]):
        for col, horizon in enumerate([1000, 2000]):
            ax = axes[row, col]
            pair = data[(data["assessment"] == assessment)
                        & (data["horizon_ms"] == horizon)]
            for label in ["Original", "Failure-aware"]:
                values = pair[pair["model_label"] == label].sort_values("threshold_w")
                ax.plot(values["threshold_w"], values["exceedance_fraction"] * 100,
                        marker="o", linewidth=2.5, color=COLORS[label], label=label)
            ax.axvline(500, color="#333333", linestyle=":", linewidth=1)
            ax.set_title(f"{'Grouped test' if assessment == 'grouped_outer' else 'Reused holdout'} · {horizon // 1000} s",
                         loc="left")
            ax.set_xlabel("Absolute error threshold (W)")
            ax.set_ylabel("Targets exceeding threshold (%)")
    fig.suptitle("Large errors · original (gray) vs failure-aware (teal)",
                 fontsize=16, fontweight="bold")
    save_figure(fig, output, "model_tolerance_curves.png")


def model_run_deltas(improvement_root: Path, output: Path) -> None:
    """Connect each run's original and revised MAE."""
    data = select_model_rows(read_csv(improvement_root, "per_run_metrics.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained", sharex=True)
    for row, assessment in enumerate(["grouped_outer", "reused_holdout"]):
        for col, horizon in enumerate([1000, 2000]):
            ax = axes[row, col]
            subset = data[(data["assessment"] == assessment)
                          & (data["horizon_ms"] == horizon)]
            paired = subset.pivot(index="run_id", columns="model_label", values="mae_w").dropna()
            better = 0
            for _, item in paired.iterrows():
                old, new = float(item["Original"]), float(item["Failure-aware"])
                improved = new < old
                better += improved
                ax.plot([0, 1], [old, new], color="#147d73" if improved else "#bd4b41",
                        marker="o", markersize=4, alpha=0.8, linewidth=1.7)
            ax.set_xticks([0, 1], ["Original", "Failure-aware"])
            ax.set_xlim(-0.12, 1.12)
            ax.set_title(f"{'Grouped test' if assessment == 'grouped_outer' else 'Reused holdout'} · {horizon // 1000} s"
                         f" · {better}/{len(paired)} runs improved", loc="left")
            ax.set_ylabel("Run MAE (W)")
            ax.grid(axis="x", visible=False)
    fig.suptitle("Paired run errors · green lines improve, red lines worsen",
                 fontsize=16, fontweight="bold")
    save_figure(fig, output, "model_run_deltas.png")


def model_transition_change_lines(improvement_root: Path, output: Path) -> None:
    """Track transition-specific MAE reduction across the two measured horizons."""
    data = select_model_rows(read_csv(improvement_root, "transition_metrics.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
    colors = {"falling": "#4c78a8", "rising": "#d98725", "stable": "#147d73"}
    for ax, assessment in zip(axes, ["grouped_outer", "reused_holdout"]):
        for stratum in ["falling", "rising", "stable"]:
            reductions = []
            for horizon in [1000, 2000]:
                pair = data[(data["assessment"] == assessment)
                            & (data["horizon_ms"] == horizon)
                            & (data["transition_stratum"] == stratum)].set_index("model_label")
                old = float(pair.loc["Original", "mae_w"])
                new = float(pair.loc["Failure-aware", "mae_w"])
                reductions.append(100 * (old - new) / old)
            ax.plot([1, 2], reductions, marker="o", markersize=8, linewidth=2.5,
                    color=colors[stratum], label=stratum.title())
            for x, value in zip([1, 2], reductions):
                ax.annotate(f"{value:+.1f}%", (x, value), xytext=(0, 7),
                            textcoords="offset points", ha="center",
                            color=colors[stratum], fontsize=9)
        ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
        ax.set_xticks([1, 2], ["1 s", "2 s"])
        ax.set_xlim(0.8, 2.2)
        ax.set_title("Grouped test" if assessment == "grouped_outer" else "Reused holdout")
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("MAE reduction (%) · below zero = worse")
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Transition error change by horizon", fontsize=16, fontweight="bold")
    save_figure(fig, output, "model_transition_change_lines.png")


def predictability_plots(predictability_root: Path, output: Path) -> pd.DataFrame:
    data = read_csv(predictability_root, "metrics.csv")
    labels = {"unconditional": "Unconditional", "current_only": "Current only", "history": "History"}
    grouped = (
        data.groupby(["resolution_ms", "horizon_ms", "condition"], as_index=False)
        .agg(
            mean_crps_w=("mean_crps_w", "mean"),
            min_crps_w=("mean_crps_w", "min"),
            max_crps_w=("mean_crps_w", "max"),
            mean_crpss_vs_current_only=("crpss_vs_current_only", "mean"),
            n_cohorts=("cohort", "nunique"),
        )
    )
    grouped.to_csv(output / "predictability_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, resolution in zip(axes, [50, 200, 1000]):
        subset = grouped[grouped["resolution_ms"] == resolution]
        for condition in ["unconditional", "current_only", "history"]:
            rows = subset[subset["condition"] == condition].sort_values("horizon_ms")
            label = labels[condition]
            x = rows["horizon_ms"].to_numpy(dtype=float)
            y = rows["mean_crps_w"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2, label=label, color=COLORS[label])
            ax.fill_between(
                x,
                rows["min_crps_w"].to_numpy(dtype=float),
                rows["max_crps_w"].to_numpy(dtype=float),
                color=COLORS[label],
                alpha=0.10,
            )
        ax.set_title(f"Resolution {resolution} ms")
        ax.set_xlabel("Forecast horizon (ms)")
        ax.set_xticks(sorted(subset["horizon_ms"].unique()))
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Mean cohort CRPS (W)\n(shading: cohort min-max)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Predictability by sampling resolution and horizon", y=1.09, fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output, "predictability_crps.png")

    history = grouped[grouped["condition"] == "history"].copy()
    gain = history.pivot(index="resolution_ms", columns="horizon_ms", values="mean_crpss_vs_current_only") * 100.0
    gain = gain.reindex(index=[50, 200, 1000], columns=[50, 200, 1000, 2000])
    fig, ax = plt.subplots(figsize=(8, 4.3))
    masked = np.ma.masked_invalid(gain.to_numpy(dtype=float))
    max_abs = max(1.0, float(np.nanmax(np.abs(masked))))
    image = ax.imshow(masked, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, aspect="auto")
    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            value = gain.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontweight="bold")
    ax.set_xticks(range(gain.shape[1]), [f"{value} ms" for value in gain.columns])
    ax.set_yticks(range(gain.shape[0]), [f"{value} ms" for value in gain.index])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Sampling resolution")
    ax.set_title("History CRPSS vs current-only (cohort mean)\npositive = history improves CRPS")
    fig.colorbar(image, ax=ax, label="CRPS skill (%)")
    fig.tight_layout()
    save_figure(fig, output, "predictability_history_gain_heatmap.png")

    transition = read_csv(predictability_root, "transition_metrics.csv")
    transition = transition[transition["condition"] == "history"]
    transition_grouped = (
        transition.groupby(["resolution_ms", "horizon_ms", "stratum"], as_index=False)["mean_crps_w"].mean()
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, resolution in zip(axes, [50, 200, 1000]):
        subset = transition_grouped[transition_grouped["resolution_ms"] == resolution]
        for stratum, color in zip(["falling", "rising", "stable"], ["#4c78a8", "#e45756", "#72b7b2"]):
            rows = subset[subset["stratum"] == stratum].sort_values("horizon_ms")
            ax.plot(rows["horizon_ms"], rows["mean_crps_w"], marker="o", linewidth=2, label=stratum.title(), color=color)
        ax.set_title(f"Resolution {resolution} ms")
        ax.set_xlabel("Forecast horizon (ms)")
        ax.set_xticks(sorted(subset["horizon_ms"].unique()))
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("History-model mean cohort CRPS (W)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Predictability error by endpoint transition stratum", y=1.09, fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output, "predictability_transition_crps.png")
    return grouped


def write_readme(output: Path, model: pd.DataFrame, predictability: pd.DataFrame) -> None:
    grouped = model[model["assessment"] == "grouped_outer"].set_index("horizon_ms")
    history = predictability[predictability["condition"] == "history"]
    best = history.loc[history["mean_crpss_vs_current_only"].idxmax()]
    worst = history.loc[history["mean_crpss_vs_current_only"].idxmin()]
    text = f"""# Error visualization results

Generated from the frozen experiment outputs; no model was retrained.

## Main matched model comparison

- Grouped-outer 1 s: CRPS reduction {grouped.loc[1000, 'reduction_crps_w_pct']:.1f}%, MAE reduction {grouped.loc[1000, 'reduction_mae_w_pct']:.1f}%, >500 W rate reduction {grouped.loc[1000, 'reduction_exceedance_500_fraction_pct']:.1f}%.
- Grouped-outer 2 s: CRPS reduction {grouped.loc[2000, 'reduction_crps_w_pct']:.1f}%, MAE reduction {grouped.loc[2000, 'reduction_mae_w_pct']:.1f}%, >500 W rate reduction {grouped.loc[2000, 'reduction_exceedance_500_fraction_pct']:.1f}%.
- The matched comparison is old_frozen vs reliability_selection, history_meta/full15, seed 42, run-balanced.
- Reused holdout is shown separately because it had already been inspected and is not fresh confirmation.

## Predictability experiment

- Best mean history gain vs current-only: resolution {int(best['resolution_ms'])} ms / horizon {int(best['horizon_ms'])} ms ({100 * best['mean_crpss_vs_current_only']:.1f}% CRPSS).
- Lowest mean history gain vs current-only: resolution {int(worst['resolution_ms'])} ms / horizon {int(worst['horizon_ms'])} ms ({100 * worst['mean_crpss_vs_current_only']:.1f}% CRPSS).
- Values are unweighted means across four cohorts; the min-max band in the CRPS plot exposes cohort spread.
- Predictability and failure-aware results are not pooled because their grids, targets, and evaluation populations differ.

## Files

- `model_error_overview.png`: line charts of CRPS, MAE, p95 error, and >500 W rate across 1 s and 2 s horizons.
- `model_transition_errors.png`: line charts of falling/rising/stable endpoint MAE.
- `model_per_run_paired.png`: per-run original vs failure-aware MAE.
- `model_error_quantiles.png`: absolute-error 50th, 90th, 95th, and 99th percentiles.
- `model_tolerance_curves.png`: fraction of targets above each absolute-error threshold.
- `model_run_deltas.png`: paired per-run MAE lines; red lines show worse runs.
- `model_transition_change_lines.png`: MAE reduction by transition and horizon; below zero means worse.
- `predictability_crps.png`: CRPS by resolution, horizon, and predictor.
- `predictability_history_gain_heatmap.png`: history gain over current-only.
- `predictability_transition_crps.png`: history-model CRPS by transition stratum.
- CSV files contain the exact plotted aggregate values and reductions.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--improvement-root", type=Path, default=DEFAULT_IMPROVEMENT)
    parser.add_argument("--predictability-root", type=Path, default=DEFAULT_PREDICTABILITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()
    model = model_summary(args.improvement_root, args.output)
    model_transition_plot(args.improvement_root, args.output)
    model_per_run_plot(args.improvement_root, args.output)
    model_error_quantiles(args.improvement_root, args.output)
    model_tolerance_curves(args.improvement_root, args.output)
    model_run_deltas(args.improvement_root, args.output)
    model_transition_change_lines(args.improvement_root, args.output)
    predictability = predictability_plots(args.predictability_root, args.output)
    write_readme(args.output, model, predictability)
    print(f"Wrote visualizations to {args.output}")


if __name__ == "__main__":
    main()
