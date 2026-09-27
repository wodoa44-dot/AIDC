#!/usr/bin/env python3
"""Build appendix-ready tables and figures from completed forecast experiments.

No model fitting or hyperparameter search is performed.  All statistics are
derived from frozen CSV/JSON/Parquet artifacts already present in the repo.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PLOT_DATA = OUT / "plot_data"

PRED = REPO / "trace" / "predictability" / "outputs" / "main_20260914"
FAIL = (
    REPO
    / "trace"
    / "model_200ms_1s_2s"
    / "improvements"
    / "failure_aware_20260921_v1"
    / "artifacts"
    / "full"
)
OLD = REPO / "trace" / "model_200ms_1s_2s" / "outputs" / "lite_full_20260920"

MODEL_LABELS = {
    "unconditional": "Unconditional",
    "current_only": "Current only",
    "history": "History",
    "old_frozen": "Original",
    "mean_crps_selection": "Mean-CRPS selection",
    "reliability_selection": "Failure-aware",
    "persistence": "Persistence",
}
COLORS = {
    "Unconditional": "#8c8c8c",
    "Current only": "#0072B2",
    "History": "#009E73",
    "Original": "#6B7280",
    "Mean-CRPS selection": "#56B4E9",
    "Failure-aware": "#D55E00",
    "Persistence": "#CC79A7",
    "Ground truth": "#111111",
}
LINESTYLES = {
    "Original": "--",
    "Failure-aware": "-",
    "Ground truth": "-",
}
MARKERS = {"Original": "o", "Failure-aware": "D", "Ground truth": "x"}


def setup() -> None:
    for directory in [TABLES, FIGURES, PLOT_DATA]:
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "savefig.facecolor": "white",
        }
    )


def require_completed() -> None:
    pred_status = json.loads((PRED / "status.json").read_text())
    fail_status = json.loads((FAIL / "completion_status.json").read_text())
    if pred_status.get("status") != "completed":
        raise RuntimeError("predictability/main_20260914 is not completed")
    if not fail_status.get("passed"):
        raise RuntimeError("failure-aware full experiment did not pass completion checks")


def latex_escape(value: object) -> str:
    if pd.isna(value):
        return "---"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def ranked_strings(row: pd.Series, columns: list[str], decimals: int) -> dict[str, str]:
    values = {column: float(row[column]) for column in columns if pd.notna(row[column])}
    unique = sorted(set(values.values()))
    best = unique[0] if unique else None
    second = unique[1] if len(unique) > 1 else None
    result: dict[str, str] = {}
    for column in columns:
        value = row[column]
        if pd.isna(value):
            result[column] = "---"
        else:
            formatted = f"{float(value):.{decimals}f}"
            if float(value) == best:
                formatted = rf"\textbf{{{formatted}}}"
            elif float(value) == second:
                formatted = rf"\underline{{{formatted}}}"
            result[column] = formatted
    return result


def write_table(
    table_id: str,
    caption: str,
    data: pd.DataFrame,
    notes: str,
    rank_columns: list[str] | None = None,
    decimals: int = 2,
) -> list[Path]:
    base = TABLES / f"table_{table_id}"
    data.to_csv(base.with_suffix(".csv"), index=False)

    display = data.copy()
    numeric_columns = display.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        display[column] = display[column].map(lambda x: "---" if pd.isna(x) else f"{x:.{decimals}f}")
    md = f"**Table {table_id}. {caption}**\n\n{display.to_markdown(index=False)}\n\n_{notes}_\n"
    base.with_suffix(".md").write_text(md, encoding="utf-8")

    tex_rows: list[str] = []
    for _, row in data.iterrows():
        ranking = ranked_strings(row, rank_columns, decimals) if rank_columns else {}
        cells: list[str] = []
        for column in data.columns:
            value = row[column]
            if column in ranking:
                cells.append(ranking[column])
            elif isinstance(value, (float, np.floating)) and not isinstance(value, bool):
                cells.append("---" if pd.isna(value) else f"{value:.{decimals}f}")
            else:
                cells.append(latex_escape(value))
        tex_rows.append(" & ".join(cells) + r" \\")
    alignment = "l" * len(data.columns)
    header = " & ".join(latex_escape(column) for column in data.columns) + r" \\" 
    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:{table_id.lower()}}}\n"
        "\\small\n"
        f"\\begin{{tabular}}{{{alignment}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(tex_rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\vspace{{2pt}}\\par\\footnotesize {notes}\n"
        "\\end{table*}\n"
    )
    base.with_suffix(".tex").write_text(tex, encoding="utf-8")
    return [base.with_suffix(ext) for ext in [".csv", ".md", ".tex"]]


def save_figure(fig: plt.Figure, figure_id: str) -> list[Path]:
    png = FIGURES / f"figure_{figure_id}.png"
    pdf = FIGURES / f"figure_{figure_id}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def predictability_run_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((PRED / "predictions").glob("*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=[
                "run_id",
                "cohort",
                "resolution_ms",
                "horizon_ms",
                "condition",
                "y",
                "q_0.5",
                "crps",
            ],
        )
        frame["abs_error"] = (frame["q_0.5"] - frame["y"]).abs()
        frame["sq_error"] = (frame["q_0.5"] - frame["y"]) ** 2
        grouped = (
            frame.groupby(
                ["run_id", "cohort", "resolution_ms", "horizon_ms", "condition"],
                as_index=False,
            )
            .agg(n_windows=("y", "size"), crps_w=("crps", "mean"), mae_w=("abs_error", "mean"), mse_w2=("sq_error", "mean"))
        )
        grouped["rmse_w"] = np.sqrt(grouped["mse_w2"])
        rows.append(grouped.drop(columns="mse_w2"))
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(PLOT_DATA / "predictability_per_run_metrics.csv", index=False)
    return result


def predictability_macro(run_metrics: pd.DataFrame) -> pd.DataFrame:
    result = (
        run_metrics.groupby(["resolution_ms", "horizon_ms", "condition"], as_index=False)
        .agg(
            n_runs=("run_id", "nunique"),
            n_windows=("n_windows", "sum"),
            crps_w=("crps_w", "mean"),
            mae_w=("mae_w", "mean"),
            rmse_w=("rmse_w", "mean"),
        )
    )
    result.to_csv(PLOT_DATA / "predictability_run_macro.csv", index=False)
    return result


def failure_aggregate() -> pd.DataFrame:
    data = pd.read_csv(FAIL / "aggregate_metrics.csv")
    selected = data[(data["seed"] == 42) & (data["weighting"] == "run_balanced")].copy()
    keep = (
        ((selected["model"].isin(["old_frozen", "mean_crps_selection", "reliability_selection"]))
         & (selected["condition"] == "history_meta")
         & (selected["representation"] == "full15"))
        | ((selected["model"] == "persistence") & (selected["representation"] == "point"))
    )
    selected = selected[keep].copy()
    selected["model_label"] = selected["model"].map(MODEL_LABELS)
    return selected


def add_average_rows(frame: pd.DataFrame, group_columns: list[str], model_columns: list[str]) -> pd.DataFrame:
    rows = [frame]
    for group_key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        avg = {column: value for column, value in zip(group_columns, group_key)}
        for column in frame.columns:
            if column in model_columns:
                avg[column] = group[column].mean()
            elif column not in group_columns:
                avg[column] = "Avg"
        rows.append(pd.DataFrame([avg]))
    return pd.concat(rows, ignore_index=True)


def table_a1(run_metrics: pd.DataFrame, failure: pd.DataFrame) -> pd.DataFrame:
    selected = pd.read_csv(PRED / "selected_validation_configs.csv")
    pred_rows: list[dict[str, object]] = []
    for (resolution, horizon), group in selected.groupby(["resolution_ms", "horizon_ms"]):
        histories = sorted(group["history_ms"].dropna().astype(int).unique())
        metric = run_metrics[
            (run_metrics["resolution_ms"] == resolution)
            & (run_metrics["horizon_ms"] == horizon)
            & (run_metrics["condition"] == "history")
        ]
        pred_rows.append(
            {
                "Dataset": "highres_node",
                "Protocol": "within-run chronological test",
                "Sampling (ms)": int(resolution),
                "Horizon (ms)": int(horizon),
                "Target / unit": "8-GPU power sum / W",
                "Test runs": int(metric["run_id"].nunique()),
                "Test windows": int(metric["n_windows"].sum()),
                "Lookback (steps / ms)": ", ".join(f"{h // resolution}/{h}" for h in histories),
                "Horizon (steps / ms)": f"{horizon // resolution}/{horizon}",
                "Model / config ID": "History / main_20260914",
            }
        )
    selections = json.loads((FAIL / "final_selections.json").read_text())
    rel = {int(row["horizon_ms"]): row for row in selections if row["procedure"] == "reliability_selection"}
    fail_rows: list[dict[str, object]] = []
    chosen = failure[failure["model"] == "reliability_selection"]
    for _, row in chosen.iterrows():
        horizon = int(row["horizon_ms"])
        history = int(rel[horizon]["history_ms"])
        fail_rows.append(
            {
                "Dataset": "highres_node",
                "Protocol": str(row["assessment"]),
                "Sampling (ms)": 200,
                "Horizon (ms)": horizon,
                "Target / unit": "8-GPU power sum / W",
                "Test runs": int(row["n_runs"]),
                "Test windows": int(row["n_targets"]),
                "Lookback (steps / ms)": f"{history // 200}/{history}",
                "Horizon (steps / ms)": f"{horizon // 200}/{horizon}",
                "Model / config ID": rel[horizon]["candidate_id"],
            }
        )
    result = pd.DataFrame(pred_rows + fail_rows).sort_values(["Dataset", "Protocol", "Sampling (ms)", "Horizon (ms)"])
    write_table(
        "A1_evaluation_settings",
        "Evaluation settings (RAFT Table 6 형식의 데이터 맞춤 적용).",
        result,
        "Sampling은 최종 평가 bin 간격이며 원시 기록 간격과 다르다. 모든 target은 8개 GPU 전력 합(W), CPU 제외이다.",
        decimals=0,
    )
    return result


def wide_metric_table(
    source: pd.DataFrame,
    index_columns: list[str],
    model_column: str,
    metric: str,
    model_order: list[str],
    average_groups: list[str],
) -> pd.DataFrame:
    wide = source.pivot(index=index_columns, columns=model_column, values=metric).reset_index()
    wide.columns.name = None
    for model in model_order:
        if model not in wide:
            wide[model] = np.nan
    wide = wide[index_columns + model_order]
    return add_average_rows(wide, average_groups, model_order)


def tables_a2(pred_macro: pd.DataFrame, failure: pd.DataFrame) -> dict[str, pd.DataFrame]:
    products: dict[str, pd.DataFrame] = {}
    pred = pred_macro.copy()
    pred["Protocol"] = "within-run chronological test"
    pred["Model"] = pred["condition"].map(MODEL_LABELS)
    pred = pred.rename(columns={"resolution_ms": "Resolution (ms)", "horizon_ms": "Horizon (ms)"})
    pred_models = ["Unconditional", "Current only", "History"]
    for suffix, metric, label in [("a", "crps_w", "CRPS (W)"), ("b", "mae_w", "MAE (W)"), ("c", "rmse_w", "RMSE (W)")]:
        table = wide_metric_table(
            pred,
            ["Protocol", "Resolution (ms)", "Horizon (ms)"],
            "Model",
            metric,
            pred_models,
            ["Protocol"],
        )
        table = table.rename(columns={model: f"{model} {label}" for model in pred_models})
        score_columns = [f"{model} {label}" for model in pred_models]
        table_id = f"A2{suffix}_predictability_{metric.removesuffix('_w')}"
        write_table(
            table_id,
            f"Full predictability results — run-macro {label}.",
            table,
            "각 run에서 먼저 계산한 뒤 12개 run을 동일 가중 평균했다. Avg는 표에 나열된 9개 resolution–horizon cell의 평균이다. 낮을수록 좋으며 최선은 bold, 차선은 underline이다.",
            rank_columns=score_columns,
        )
        products[table_id] = table

    f = failure.copy()
    f = f.rename(columns={"assessment": "Protocol", "horizon_ms": "Horizon (ms)"})
    f["Resolution (ms)"] = 200
    f["Model"] = f["model_label"]
    failure_models = ["Original", "Mean-CRPS selection", "Failure-aware", "Persistence"]
    for suffix, metric, label in [("d", "crps_w", "CRPS (W)"), ("e", "mae_w", "MAE (W)"), ("f", "rmse_w", "RMSE (W)")]:
        table = wide_metric_table(
            f,
            ["Protocol", "Resolution (ms)", "Horizon (ms)"],
            "Model",
            metric,
            failure_models,
            ["Protocol"],
        )
        table = table.rename(columns={model: f"{model} {label}" for model in failure_models})
        score_columns = [f"{model} {label}" for model in failure_models]
        table_id = f"A2{suffix}_failure_aware_{metric.removesuffix('_w')}"
        write_table(
            table_id,
            f"Full original/failure-aware results — run-macro {label}.",
            table,
            "Seed 42, history+metadata, full 15 quantiles, run-balanced. Avg는 각 protocol의 1 s와 2 s 평균이다. Persistence에는 probabilistic CRPS가 없어 —로 표시한다.",
            rank_columns=score_columns,
        )
        products[table_id] = table
    return products


def tables_a3(pred_macro: pd.DataFrame, failure: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pred = pred_macro.copy()
    pred["Model"] = pred["condition"].map(MODEL_LABELS)
    component = pred[pred["resolution_ms"] == 200].pivot(index="Model", columns="horizon_ms", values="crps_w")
    component = component.reindex(["Unconditional", "Current only", "History"]).reset_index()
    component = component.rename(columns={200: "H=200 ms", 1000: "H=1000 ms", 2000: "H=2000 ms"})
    write_table(
        "A3a_component_ablation",
        "Component comparison at 200 ms sampling — run-macro CRPS (W).",
        component,
        "행 간에는 입력 구성만 달라진다. Unconditional은 입력 없음, Current only는 최신 bin, History는 선택된 causal history feature를 사용한다.",
        rank_columns=["H=200 ms", "H=1000 ms", "H=2000 ms"],
    )

    history = pred[pred["condition"] == "history"]
    sensitivity = history[history["horizon_ms"].isin([1000, 2000])].pivot(
        index="horizon_ms", columns="resolution_ms", values="crps_w"
    ).reset_index()
    sensitivity = sensitivity.rename(
        columns={"horizon_ms": "Horizon (ms)", 50: "R=50 ms", 200: "R=200 ms", 1000: "R=1000 ms"}
    )
    write_table(
        "A3b_resolution_sensitivity",
        "Resolution sensitivity of the History model — run-macro CRPS (W).",
        sensitivity,
        "Horizon을 행에서 고정하고 sampling interval만 열에서 바꾼다. 해상도별 binning으로 target 자체가 달라질 수 있으므로 순수 feature ablation으로 해석하지 않는다.",
        rank_columns=["R=50 ms", "R=200 ms", "R=1000 ms"],
    )

    proc = failure[(failure["assessment"] == "grouped_outer") & (failure["model"] != "persistence")].copy()
    proc_table = proc.pivot(index="model_label", columns="horizon_ms", values="crps_w").reset_index()
    proc_table = proc_table.rename(columns={"model_label": "Procedure", 1000: "H=1000 ms", 2000: "H=2000 ms"})
    order = pd.Categorical(
        proc_table["Procedure"],
        categories=["Original", "Mean-CRPS selection", "Failure-aware"],
        ordered=True,
    )
    proc_table = proc_table.assign(_order=order).sort_values("_order").drop(columns="_order")
    write_table(
        "A3c_selection_procedure",
        "Failure-aware selection procedure comparison — grouped-outer run-macro CRPS (W).",
        proc_table,
        "Mean-CRPS와 reliability selection은 1 s에서 같은 후보를 선택한다. 2 s에서는 candidate가 달라지므로 순수 단일-component ablation이 아닌 선택 절차 비교다.",
        rank_columns=["H=1000 ms", "H=2000 ms"],
    )
    return {"A3a": component, "A3b": sensitivity, "A3c": proc_table}


def table_a4(failure: pd.DataFrame) -> pd.DataFrame:
    transition = pd.read_csv(FAIL / "transition_metrics.csv")
    transition = transition[
        (transition["condition"] == "history_meta")
        & (transition["representation"] == "full15")
        & (transition["model"].isin(["old_frozen", "reliability_selection"]))
    ].copy()
    transition["Model"] = transition["model"].map(MODEL_LABELS)
    all_rows = failure[failure["model"].isin(["old_frozen", "reliability_selection"])].copy()
    records: list[dict[str, object]] = []
    for _, row in all_rows.iterrows():
        record: dict[str, object] = {
            "Protocol": row["assessment"],
            "Horizon (ms)": int(row["horizon_ms"]),
            "Model": row["model_label"],
            "All CRPS": row["crps_w"],
            "All n": int(row["n_targets"]),
        }
        subset = transition[
            (transition["assessment"] == row["assessment"])
            & (transition["horizon_ms"] == row["horizon_ms"])
            & (transition["Model"] == row["model_label"])
        ]
        for source, label in [("stable", "Stable"), ("rising", "Ramp-up"), ("falling", "Ramp-down")]:
            item = subset[subset["transition_stratum"] == source].iloc[0]
            record[f"{label} CRPS"] = item["crps_w"]
            record[f"{label} n"] = int(item["n_targets"])
        records.append(record)
    result = pd.DataFrame(records).sort_values(["Protocol", "Horizon (ms)", "Model"])
    write_table(
        "A4_transition_conditioned",
        "Transition-conditioned CRPS (W) and evaluation-window counts.",
        result,
        "Ramp-up/down은 target power − issuance power가 각각 +100 W 이상/−100 W 이하인 endpoint 변화다. Stable은 그 사이이며 event detection 결과가 아니다; n은 event 수가 아니라 forecast window 수다.",
    )
    result.to_csv(PLOT_DATA / "transition_conditioned_performance.csv", index=False)
    return result


def table_a5(failure: pd.DataFrame) -> pd.DataFrame:
    old_bench = json.loads((OLD / "benchmark.json").read_text())
    new_bench = json.loads((FAIL / "benchmark.json").read_text())
    all_aggregate = pd.read_csv(FAIL / "aggregate_metrics.csv")
    old_mae = all_aggregate[
        (all_aggregate["assessment"] == "grouped_outer")
        & (all_aggregate["model"] == "old_frozen")
        & (all_aggregate["condition"] == "history_meta")
        & (all_aggregate["representation"] == "compact3")
        & (all_aggregate["seed"] == 42)
        & (all_aggregate["weighting"] == "run_balanced")
    ]["mae_w"].mean()
    new_mae = failure[
        (failure["assessment"] == "grouped_outer")
        & (failure["model"] == "reliability_selection")
    ]["mae_w"].mean()
    result = pd.DataFrame(
        [
            {
                "Model": "Original deployment",
                "Representation": "compact 3 quantiles",
                "Hardware": "x86_64 CPU; 1 model thread",
                "Batch": 1,
                "Trials": old_bench["trials"],
                "Models / request": 6,
                "Package (MB)": old_bench["package_bytes"] / 1e6,
                "Inference p95 (ms)": old_bench["prediction_only_six_models_us"]["p95"] / 1000,
                "End-to-end p50 (ms)": old_bench["end_to_end_six_models_us"]["p50"] / 1000,
                "End-to-end p95 (ms)": old_bench["end_to_end_six_models_us"]["p95"] / 1000,
                "Process max RSS (GiB)": old_bench["maxrss_kb"] / 1024**2,
                "Grouped-outer MAE (W)": old_mae,
            },
            {
                "Model": "Failure-aware deployment",
                "Representation": "full 15 quantiles",
                "Hardware": "x86_64 CPU; 1 model thread",
                "Batch": 1,
                "Trials": new_bench["trials"],
                "Models / request": 30,
                "Package (MB)": new_bench["package_bytes"] / 1e6,
                "Inference p95 (ms)": new_bench["prediction_only_30_models_us"]["p95"] / 1000,
                "End-to-end p50 (ms)": new_bench["end_to_end_30_models_us"]["p50"] / 1000,
                "End-to-end p95 (ms)": new_bench["end_to_end_30_models_us"]["p95"] / 1000,
                "Process max RSS (GiB)": new_bench["maxrss_kb"] / 1024**2,
                "Grouped-outer MAE (W)": new_mae,
            },
        ]
    )
    write_table(
        "A5_computational_cost",
        "Measured deployment-package accuracy and computational cost.",
        result,
        "Warm single-row CPU benchmark for both horizons. max RSS is process high-water mark, not incremental model memory. Training time was not recorded here. Different quantile counts are explicit; latency is not a real-time guarantee.",
        decimals=3,
    )
    result.to_csv(PLOT_DATA / "accuracy_cost.csv", index=False)
    return result


def figure_a1(run_metrics: pd.DataFrame) -> pd.DataFrame:
    cohort = (
        run_metrics.groupby(["cohort", "resolution_ms", "horizon_ms", "condition"], as_index=False)["crps_w"].mean()
    )
    wide = cohort.pivot(
        index=["cohort", "resolution_ms", "horizon_ms"], columns="condition", values="crps_w"
    ).reset_index()
    wide["improvement_pct"] = 100.0 * (wide["current_only"] - wide["history"]) / wide["current_only"]
    wide.to_csv(PLOT_DATA / "figure_A1_heatmap.csv", index=False)
    cohorts = sorted(wide["cohort"].unique())
    resolutions = [50, 200, 1000]
    horizons = [50, 200, 1000, 2000]
    limit = float(np.nanmax(np.abs(wide["improvement_pct"])))
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.1), constrained_layout=True)
    image = None
    for index, (ax, name) in enumerate(zip(axes.flat, cohorts)):
        subset = wide[wide["cohort"] == name]
        grid = subset.pivot(index="resolution_ms", columns="horizon_ms", values="improvement_pct").reindex(
            index=resolutions, columns=horizons
        )
        masked = np.ma.masked_invalid(grid.to_numpy(dtype=float))
        image = ax.imshow(masked, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(horizons)), horizons)
        ax.set_yticks(range(len(resolutions)), resolutions)
        ax.set_xlabel("Forecast horizon (ms)")
        ax.set_ylabel("Sampling interval (ms)")
        ax.set_title(name.replace("highres_node__", "").replace("__", " / ").replace("_", " "))
        panel_label(ax, f"({chr(97 + index)})")
        for i in range(len(resolutions)):
            for j in range(len(horizons)):
                value = grid.iloc[i, j]
                ax.text(j, i, "—" if pd.isna(value) else f"{value:.1f}", ha="center", va="center", fontsize=7)
    assert image is not None
    fig.colorbar(image, ax=axes, location="right", shrink=0.83, label="CRPS improvement vs current-only (%)")
    save_figure(fig, "A1_resolution_horizon_heatmap")
    return wide


def figure_a2() -> pd.DataFrame:
    data = pd.read_csv(FAIL / "per_run_metrics.csv")
    data = data[
        (data["assessment"] == "grouped_outer")
        & (data["seed"] == 42)
        & (data["condition"] == "history_meta")
        & (data["representation"] == "full15")
        & (data["model"].isin(["old_frozen", "reliability_selection"]))
    ].copy()
    data["Model"] = data["model"].map(MODEL_LABELS)
    wide = data.pivot(index=["horizon_ms", "run_id", "cohort"], columns="Model", values="mae_w").reset_index()
    wide["improvement_w"] = wide["Original"] - wide["Failure-aware"]
    wide.to_csv(PLOT_DATA / "figure_A2_paired_runs.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 5.0), sharex=False)
    for panel, (ax, horizon) in enumerate(zip(axes, [1000, 2000])):
        subset = wide[wide["horizon_ms"] == horizon].sort_values("improvement_w")
        y = np.arange(len(subset))
        for yi, (_, row) in zip(y, subset.iterrows()):
            color = "#009E73" if row["improvement_w"] >= 0 else "#C44E52"
            ax.plot([row["Original"], row["Failure-aware"]], [yi, yi], color=color, alpha=0.65, linewidth=1)
        ax.scatter(subset["Original"], y, color=COLORS["Original"], marker=MARKERS["Original"], s=22, label="Original", zorder=3)
        ax.scatter(subset["Failure-aware"], y, color=COLORS["Failure-aware"], marker=MARKERS["Failure-aware"], s=22, label="Failure-aware", zorder=3)
        labels = [rid[-8:] for rid in subset["run_id"]]
        ax.set_yticks(y, labels)
        ax.axvline(0, color="#333333", linewidth=0.6)
        ax.set_xlabel("Per-run MAE (W)")
        ax.set_title(f"Horizon {horizon / 1000:g} s\nmedian improvement {subset['improvement_w'].median():.1f} W; n={len(subset)}")
        panel_label(ax, f"({chr(97 + panel)})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, "A2_paired_per_run")
    return wide


def read_case_predictions() -> pd.DataFrame:
    columns = [
        "run_id", "cohort", "segment_key", "issue_time_s", "target_time_s", "observed_power_w",
        "issue_power_w", "q10", "q50", "q90", "transition_stratum", "assessment", "procedure",
        "horizon_ms", "seed", "condition", "representation",
    ]
    path = FAIL / "per_target_predictions_all_models.parquet"
    data = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("assessment", "=", "grouped_outer"),
            ("seed", "=", 42),
            ("condition", "=", "history_meta"),
            ("representation", "=", "full15"),
        ],
    )
    return data[data["procedure"].isin(["old_frozen", "reliability_selection"])].copy()


def figure_a3() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = read_case_predictions()
    keys = ["run_id", "cohort", "segment_key", "issue_time_s", "target_time_s", "horizon_ms", "transition_stratum", "observed_power_w"]
    wide = data.pivot(index=keys, columns="procedure", values="q50").reset_index()
    wide["improvement_w"] = (
        (wide["old_frozen"] - wide["observed_power_w"]).abs()
        - (wide["reliability_selection"] - wide["observed_power_w"]).abs()
    )
    candidates = wide[wide["horizon_ms"] == 2000].copy()
    rules = [
        ("stable", "typical", lambda x: (x["improvement_w"] - x["improvement_w"].median()).abs().idxmin()),
        ("rising", "largest improvement", lambda x: x["improvement_w"].idxmax()),
        ("falling", "largest worsening", lambda x: x["improvement_w"].idxmin()),
    ]
    chosen_rows: list[pd.Series] = []
    for stratum, rule, chooser in rules:
        subset = candidates[candidates["transition_stratum"] == stratum]
        item = subset.loc[chooser(subset)].copy()
        item["selection_rule"] = rule
        chosen_rows.append(item)
    selected = pd.DataFrame(chosen_rows)
    selected.to_csv(PLOT_DATA / "figure_A3_selected_cases.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.6), sharey="row", constrained_layout=True)
    plot_records: list[dict[str, object]] = []
    titles = {"stable": "Stable / typical", "rising": "Ramp-up / largest improvement", "falling": "Ramp-down / largest worsening"}
    for row_index, (_, case) in enumerate(selected.iterrows()):
        run_data = data[
            (data["run_id"] == case["run_id"])
            & (data["segment_key"] == case["segment_key"])
            & (data["procedure"] == "old_frozen")
            & (data["horizon_ms"] == 2000)
        ].sort_values("issue_time_s")
        history = run_data[
            (run_data["issue_time_s"] >= case["issue_time_s"] - 5.0)
            & (run_data["issue_time_s"] <= case["issue_time_s"])
        ].drop_duplicates("issue_time_s")
        ax_history, ax_future = axes[row_index]
        relative = history["issue_time_s"] - case["issue_time_s"]
        ax_history.plot(relative, history["issue_power_w"], color=COLORS["Ground truth"], linewidth=1.2)
        ax_history.axvline(0, color="#555555", linestyle=":", linewidth=0.8)
        ax_history.set_title(f"{titles[case['transition_stratum']]}\n{case['run_id'][-8:]} @ {case['issue_time_s']:.1f} s")
        ax_history.set_xlabel("Time from forecast origin (s)")
        ax_history.set_ylabel("Power (W)")
        panel_label(ax_history, f"({chr(97 + row_index * 2)})")
        for t, power in zip(relative, history["issue_power_w"]):
            plot_records.append({"case": row_index + 1, "part": "history", "relative_time_s": t, "observed_power_w": power})

        future_rows = data[
            (data["run_id"] == case["run_id"])
            & (data["segment_key"] == case["segment_key"])
            & (np.isclose(data["issue_time_s"], case["issue_time_s"]))
            & (data["horizon_ms"].isin([1000, 2000]))
        ].copy()
        for model, label, offset in [("old_frozen", "Original", -0.045), ("reliability_selection", "Failure-aware", 0.045)]:
            model_rows = future_rows[future_rows["procedure"] == model].sort_values("horizon_ms")
            x = model_rows["horizon_ms"].to_numpy(dtype=float) / 1000.0 + offset
            y = model_rows["q50"].to_numpy(dtype=float)
            low = y - model_rows["q10"].to_numpy(dtype=float)
            high = model_rows["q90"].to_numpy(dtype=float) - y
            ax_future.errorbar(
                x, y, yerr=np.vstack([low, high]), color=COLORS[label], linestyle="none",
                marker=MARKERS[label], markersize=4, capsize=2, label=label,
            )
            for _, pred in model_rows.iterrows():
                plot_records.append(
                    {
                        "case": row_index + 1,
                        "part": "future",
                        "relative_time_s": pred["horizon_ms"] / 1000,
                        "model": label,
                        "observed_power_w": pred["observed_power_w"],
                        "q10_w": pred["q10"],
                        "q50_w": pred["q50"],
                        "q90_w": pred["q90"],
                    }
                )
        truth = future_rows[future_rows["procedure"] == "old_frozen"].sort_values("horizon_ms")
        ax_future.scatter(truth["horizon_ms"] / 1000, truth["observed_power_w"], color=COLORS["Ground truth"], marker="x", s=32, label="Ground truth", zorder=5)
        ax_future.axvline(0, color="#555555", linestyle=":", linewidth=0.8)
        ax_future.set_xlim(-0.1, 2.2)
        ax_future.set_xticks([0, 1, 2])
        ax_future.set_xlabel("Time from forecast origin (s)")
        ax_future.set_title("Stored endpoint forecasts (q10–q90)")
        panel_label(ax_future, f"({chr(98 + row_index * 2)})")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.73, 1.02))
    save_figure(fig, "A3_forecast_cases")
    plot_data = pd.DataFrame(plot_records)
    plot_data.to_csv(PLOT_DATA / "figure_A3_case_plot_data.csv", index=False)
    return selected, plot_data


def figure_a4(transition: pd.DataFrame) -> pd.DataFrame:
    data = transition[transition["Protocol"] == "grouped_outer"].copy()
    order = ["All", "Stable", "Ramp-up", "Ramp-down"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, constrained_layout=True)
    for panel, (ax, horizon) in enumerate(zip(axes, [1000, 2000])):
        subset = data[data["Horizon (ms)"] == horizon]
        for model in ["Original", "Failure-aware"]:
            row = subset[subset["Model"] == model].iloc[0]
            values = [row[f"{name} CRPS"] for name in order]
            ax.plot(order, values, color=COLORS[model], linestyle=LINESTYLES[model], marker=MARKERS[model], label=model)
        counts = [int(subset.iloc[0][f"{name} n"]) for name in order]
        for x, count in enumerate(counts):
            ax.text(x, 845, f"n={count:,}", ha="center", va="center", fontsize=6.2)
        ax.set_title(f"Horizon {horizon / 1000:g} s")
        ax.set_xlabel("Endpoint condition")
        ax.tick_params(axis="x", rotation=20)
        panel_label(ax, f"({chr(97 + panel)})")
        ax.set_ylim(70, 880)
    axes[0].set_ylabel("Run-macro CRPS (W)")
    axes[0].legend()
    save_figure(fig, "A4_overall_vs_transition")
    return data


def figure_a5(cost: pd.DataFrame) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(3.4, 2.8), constrained_layout=True)
    labels = ["Original", "Failure-aware"]
    for (_, row), label in zip(cost.iterrows(), labels):
        ax.scatter(
            row["End-to-end p95 (ms)"], row["Grouped-outer MAE (W)"],
            color=COLORS[label], marker=MARKERS[label], s=48, label=label, zorder=3,
        )
        ax.annotate(
            f"{row['Package (MB)']:.2f} MB\n{int(row['Models / request'])} models",
            (row["End-to-end p95 (ms)"], row["Grouped-outer MAE (W)"]),
            xytext=(5, 5), textcoords="offset points", fontsize=7,
        )
    ax.set_xlabel("Measured end-to-end p95 latency (ms)")
    ax.set_ylabel("Grouped-outer mean-horizon MAE (W)")
    ax.legend()
    save_figure(fig, "A5_accuracy_cost")
    return cost


def write_manifest(entries: Iterable[dict[str, str]]) -> None:
    pd.DataFrame(entries).to_csv(OUT / "manifest.csv", index=False)


def write_docs(a2: dict[str, pd.DataFrame], heatmap: pd.DataFrame, paired: pd.DataFrame, cases: pd.DataFrame, cost: pd.DataFrame) -> None:
    grouped = a2["A2d_failure_aware_crps"]
    grouped = grouped[(grouped["Protocol"] == "grouped_outer") & (grouped["Horizon (ms)"] != "Avg")]
    reductions = 100 * (grouped["Original CRPS (W)"] - grouped["Failure-aware CRPS (W)"]) / grouped["Original CRPS (W)"]
    best_heat = heatmap.loc[heatmap["improvement_pct"].idxmax()]
    worsened = int((paired["improvement_w"] < 0).sum())
    total = len(paired)
    case_lines = []
    for _, row in cases.iterrows():
        case_lines.append(
            f"`{row['transition_stratum']}` / {row['selection_rule']}: `{row['run_id']}` at {row['issue_time_s']:.1f} s"
        )

    index = f"""# Appendix visual index

## Table A1. Evaluation Settings

[Markdown](tables/table_A1_evaluation_settings.md) · [CSV](tables/table_A1_evaluation_settings.csv) · [LaTeX](tables/table_A1_evaluation_settings.tex)

Caption: RAFT Table 6의 조건 정렬 원칙을 전력 예측 데이터에 맞게 적용했다. Predictability의 within-run test와 failure-aware의 grouped outer/reused holdout을 분리했고, 최종 bin 간격과 원시 약 20 ms 기록 간격을 구분했다.

관측: Predictability는 50/200/1000 ms 해상도에서 9개 유효 grid cell을 평가한다. Failure-aware는 200 ms bin에서 1 s와 2 s endpoint를 평가하며 선택된 lookback은 각각 1 s와 5 s이다.

## Table A2. Full Forecasting Results

- Predictability: [CRPS](tables/table_A2a_predictability_crps.md), [MAE](tables/table_A2b_predictability_mae.md), [RMSE](tables/table_A2c_predictability_rmse.md)
- Original vs failure-aware: [CRPS](tables/table_A2d_failure_aware_crps.md), [MAE](tables/table_A2e_failure_aware_mae.md), [RMSE](tables/table_A2f_failure_aware_rmse.md)

Caption: 모든 표는 run별 metric을 먼저 구한 run-macro 결과이며 낮을수록 좋다. Predictability Avg는 9개 grid cell, failure-aware Avg는 protocol별 두 horizon 평균이다.

관측: Grouped outer에서 failure-aware CRPS 감소율은 1 s {reductions.iloc[0]:.1f}%, 2 s {reductions.iloc[1]:.1f}%이다. Reused holdout은 이미 확인된 run이므로 fresh confirmation으로 해석하지 않는다.

## Table A3. Component / Sensitivity Analysis

[Input components](tables/table_A3a_component_ablation.md) · [Resolution](tables/table_A3b_resolution_sensitivity.md) · [Selection procedure](tables/table_A3c_selection_procedure.md)

Caption: 한 표에서 하나의 주 비교축만 바꿨다. Resolution 표는 binning에 따라 target이 달라질 수 있고, selection 표는 2 s에서 후보 자체가 달라지므로 순수 단일-component ablation이 아니다.

관측: 긴 horizon에서는 History가 Current only보다 대체로 강하지만, 가장 짧은 50 ms/50 ms 조건에서는 cohort에 따라 history가 악화된다. Failure-aware 1 s의 mean-CRPS와 reliability 절차는 같은 후보를 선택했다.

## Table A4. Transition-Conditioned Performance

[Markdown](tables/table_A4_transition_conditioned.md) · [CSV](tables/table_A4_transition_conditioned.csv) · [LaTeX](tables/table_A4_transition_conditioned.tex)

Caption: All/Stable/Ramp-up/Ramp-down의 CRPS와 forecast-window 수를 함께 제시한다. Ramp는 100 W endpoint 변화 기준이며 event detection precision/recall이 아니다.

관측: 급변 조건의 CRPS는 stable보다 크다. Failure-aware 개선은 전체 평균뿐 아니라 다수 ramp 조건에서도 유지되지만, 조건별 표본 수와 오차 규모가 크게 다르다.

## Table A5. Computational Cost

[Markdown](tables/table_A5_computational_cost.md) · [CSV](tables/table_A5_computational_cost.csv) · [LaTeX](tables/table_A5_computational_cost.tex)

Caption: 동일 CPU 환경의 warm single-row benchmark다. 원 모델은 6개, failure-aware는 30개 quantile model을 실행하므로 accuracy와 함께 representation 차이를 표시했다.

관측: Failure-aware package는 MAE를 낮추는 대신 p95 end-to-end latency가 {cost.iloc[0]['End-to-end p95 (ms)']:.2f} ms에서 {cost.iloc[1]['End-to-end p95 (ms)']:.2f} ms로 증가한다. 이 benchmark만으로 실시간 제어 가능성을 판단할 수 없다.

## Figure A1. Resolution × Horizon Improvement Heatmap

![Figure A1](figures/figure_A1_resolution_horizon_heatmap.png)

[PDF](figures/figure_A1_resolution_horizon_heatmap.pdf) · [Plot data](plot_data/figure_A1_heatmap.csv)

Caption: History의 Current-only 대비 run-macro CRPS 개선율(%). 양수는 개선, 음수는 악화이며 미실험 cell은 회색 —로 표시했다.

관측: 가장 큰 cohort-level 개선은 {best_heat['cohort'].replace('highres_node__', '')}, {int(best_heat['resolution_ms'])} ms/{int(best_heat['horizon_ms'])} ms에서 {best_heat['improvement_pct']:.1f}%다. 짧은 horizon에서는 history feature가 항상 이득이 아니다.

## Figure A2. Paired Per-Run Comparison

![Figure A2](figures/figure_A2_paired_per_run.png)

[PDF](figures/figure_A2_paired_per_run.pdf) · [Plot data](plot_data/figure_A2_paired_runs.csv)

Caption: Grouped-outer, seed 42, history+metadata/full15의 run별 MAE paired dot plot. 초록 연결선은 개선, 적색은 악화이며 horizon을 섞지 않았다.

관측: 두 horizon을 합친 {total} paired run 중 {worsened}개에서 failure-aware MAE가 악화됐다. 평균 개선은 모든 run의 일관된 개선을 뜻하지 않는다.

## Figure A3. Multi-Panel Forecast Cases

![Figure A3](figures/figure_A3_forecast_cases.png)

[PDF](figures/figure_A3_forecast_cases.pdf) · [Selected cases](plot_data/figure_A3_selected_cases.csv) · [Plot data](plot_data/figure_A3_case_plot_data.csv)

Caption: 각 행 왼쪽은 실제 5 s history, 오른쪽은 저장된 1 s/2 s endpoint 예측과 q10–q90이다. 미래 trajectory는 생성하지 않았고 deterministic rule로 stable typical, ramp-up success, ramp-down failure를 선택했다.

관측: {'; '.join(case_lines)}. 급변 사례에서 개선 폭이 클 수 있지만 반대로 baseline보다 나빠지는 실제 사례도 존재한다.

## Figure A4. Overall vs Transition Performance

![Figure A4](figures/figure_A4_overall_vs_transition.png)

[PDF](figures/figure_A4_overall_vs_transition.pdf) · [Plot data](plot_data/transition_conditioned_performance.csv)

Caption: Grouped-outer CRPS를 All/Stable/Ramp-up/Ramp-down으로 나누고 실제 window 수를 표시했다. error bar는 시계열 row를 독립 표본으로 오인할 수 있어 넣지 않았다.

관측: Ramp-up/down CRPS가 stable보다 높아 전체 평균만으로 실패 상황을 설명하기 어렵다. 2 s는 1 s보다 전반적으로 어려운 조건이다.

## Figure A5. Accuracy–Cost Trade-off

![Figure A5](figures/figure_A5_accuracy_cost.png)

[PDF](figures/figure_A5_accuracy_cost.pdf) · [Plot data](plot_data/accuracy_cost.csv)

Caption: 실제 deployment package의 p95 end-to-end latency와 grouped-outer 두-horizon 평균 MAE. quantile model 수와 package 크기를 점 옆에 표시했다.

관측: Failure-aware는 accuracy를 개선하지만 계산량과 package 크기를 늘린다. 두 점만으로 Pareto frontier나 운영 적합성을 주장하지 않는다.
"""
    (OUT / "visual_index.md").write_text(index, encoding="utf-8")

    readme = """# Appendix tables and figures

## Rebuild

```bash
python reports/appendix_visuals/build_visuals.py
```

This command only reads completed artifacts and rewrites this report directory. It does not train or tune a model.

## Selected experiments

- `trace/predictability/outputs/main_20260914`: completed 36-cell within-run chronological experiment.
- `trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/artifacts/full`: completion checks and replay passed; grouped outer and separately labelled reused holdout.
- Smoke, failed, running, and partial outputs are excluded.

CRPS/MAE/RMSE main tables use run-macro aggregation: metric per run, followed by an equal-weight mean over runs. Heatmap improvement is `100 × (current-only CRPS − history CRPS) / current-only CRPS`. Failure-aware comparisons use seed 42, history+metadata, full 15 quantiles, and matched test targets.

## Deliberate omissions and limitations

- No retrieval panel: the selected experiments contain no retrieval implementation or retrieved keys/values. The RAFT layout is only adapted as a visual organization principle.
- No event count: saved transition tables contain forecast-window counts, not independently identified physical-event counts.
- No training-time comparison for deployment packages: comparable training duration was not saved. Process max RSS is reported only as a high-water mark.
- Figure A3 shows only stored 1 s and 2 s endpoints; it does not invent a future trajectory.
- Reused holdout was previously inspected and is not fresh prospective evidence.
- No local RAFT PDF was found; the supplied table/figure description was used. LaTeX was emitted but not compiled because `pdflatex` is unavailable in the environment.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    setup()
    require_completed()
    run_metrics = predictability_run_metrics()
    pred_macro = predictability_macro(run_metrics)
    failure = failure_aggregate()

    table_a1(run_metrics, failure)
    a2 = tables_a2(pred_macro, failure)
    tables_a3(pred_macro, failure)
    transition = table_a4(failure)
    cost = table_a5(failure)

    heatmap = figure_a1(run_metrics)
    paired = figure_a2()
    cases, _ = figure_a3()
    figure_a4(transition)
    figure_a5(cost)

    source_pred = ";".join(
        [
            "trace/predictability/outputs/main_20260914/status.json",
            "trace/predictability/outputs/main_20260914/predictions/*.parquet",
            "trace/predictability/outputs/main_20260914/selected_validation_configs.csv",
        ]
    )
    source_fail = ";".join(
        [
            "trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/artifacts/full/aggregate_metrics.csv",
            "trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/artifacts/full/per_run_metrics.csv",
            "trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/artifacts/full/transition_metrics.csv",
        ]
    )
    entries = [
        {"ID": "Table A1", "source": source_pred + ";" + source_fail, "experiment/config": "main_20260914; failure_aware_20260921_v1", "metric": "settings/counts", "aggregation": "grid", "filter": "completed full runs", "generated": "tables/table_A1_evaluation_settings.*"},
        {"ID": "Table A2a-c", "source": source_pred, "experiment/config": "main_20260914", "metric": "CRPS/MAE/RMSE", "aggregation": "run-macro", "filter": "test predictions", "generated": "tables/table_A2a_predictability_crps.*;tables/table_A2b_predictability_mae.*;tables/table_A2c_predictability_rmse.*"},
        {"ID": "Table A2d-f", "source": source_fail, "experiment/config": "failure_aware_20260921_v1", "metric": "CRPS/MAE/RMSE", "aggregation": "run-balanced", "filter": "seed42; full15; history_meta", "generated": "tables/table_A2d_failure_aware_crps.*;tables/table_A2e_failure_aware_mae.*;tables/table_A2f_failure_aware_rmse.*"},
        {"ID": "Table A3", "source": source_pred + ";" + source_fail, "experiment/config": "main_20260914; failure_aware_20260921_v1", "metric": "CRPS", "aggregation": "run-macro", "filter": "component/resolution/procedure", "generated": "tables/table_A3*"},
        {"ID": "Table A4", "source": source_fail, "experiment/config": "failure_aware_20260921_v1", "metric": "CRPS/windows", "aggregation": "run-balanced", "filter": "endpoint delta threshold 100W", "generated": "tables/table_A4_transition_conditioned.*"},
        {"ID": "Table A5", "source": "trace/model_200ms_1s_2s/outputs/lite_full_20260920/benchmark.json;trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/artifacts/full/benchmark.json", "experiment/config": "deployment packages", "metric": "latency/MAE", "aggregation": "1000 trials; mean horizon MAE", "filter": "warm CPU; batch1", "generated": "tables/table_A5_computational_cost.*"},
        {"ID": "Figure A1", "source": source_pred, "experiment/config": "main_20260914", "metric": "CRPS improvement", "aggregation": "run-macro by cohort", "filter": "History vs Current only", "generated": "figures/figure_A1_resolution_horizon_heatmap.*;plot_data/figure_A1_heatmap.csv"},
        {"ID": "Figure A2", "source": source_fail, "experiment/config": "failure_aware_20260921_v1", "metric": "MAE", "aggregation": "per run", "filter": "grouped_outer; seed42; full15; history_meta", "generated": "figures/figure_A2_paired_per_run.*;plot_data/figure_A2_paired_runs.csv"},
        {"ID": "Figure A3", "source": "trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1/artifacts/full/per_target_predictions_all_models.parquet", "experiment/config": "failure_aware_20260921_v1", "metric": "endpoint q10/q50/q90", "aggregation": "selected targets", "filter": "deterministic typical/best/worst rules", "generated": "figures/figure_A3_forecast_cases.*;plot_data/figure_A3_*"},
        {"ID": "Figure A4", "source": source_fail, "experiment/config": "failure_aware_20260921_v1", "metric": "CRPS", "aggregation": "run-balanced", "filter": "grouped_outer; transition stratum", "generated": "figures/figure_A4_overall_vs_transition.*"},
        {"ID": "Figure A5", "source": "benchmark.json files;" + source_fail, "experiment/config": "deployment packages", "metric": "p95 latency/MAE", "aggregation": "measured trials/run-balanced", "filter": "warm CPU", "generated": "figures/figure_A5_accuracy_cost.*;plot_data/accuracy_cost.csv"},
    ]
    write_manifest(entries)
    write_docs(a2, heatmap, paired, cases, cost)
    print(f"Built appendix visuals in {OUT}")


if __name__ == "__main__":
    main()
