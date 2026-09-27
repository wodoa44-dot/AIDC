#!/usr/bin/env python3
"""Plot matched forecast errors against target time within each run.

The source is the frozen per-target prediction table. No model is retrained.
Each five-second bin is based on target_time_s, not issue_time_s.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "improvements/failure_aware_20260921_v1/artifacts/full/per_target_predictions_all_models.parquet"
DEFAULT_OUTPUT = HERE / "visualizations/error_over_time_20260922"
BIN_SECONDS = 5
MODELS = {"old_frozen": ("Original", "#7a8793"),
          "reliability_selection": ("Failure-aware", "#147d73")}
KEY = ["assessment", "horizon_ms", "run_id", "target_index"]


def load_matched(source: Path) -> pd.DataFrame:
    dataset = ds.dataset(source, format="parquet")
    filt = (
        ds.field("model").isin(list(MODELS))
        & (ds.field("condition") == "history_meta")
        & (ds.field("representation") == "full15")
        & (ds.field("seed") == 42)
    )
    columns = KEY + ["cohort", "segment_key", "target_time_s", "observed_power_w",
                     "q50", "error_w", "model"]
    data = dataset.to_table(filter=filt, columns=columns).to_pandas()
    if data.empty or data.duplicated(KEY + ["model"]).any():
        raise ValueError("Missing or duplicated model predictions")
    discrepancy = (data["q50"] - data["observed_power_w"] - data["error_w"]).abs().max()
    if not np.isfinite(discrepancy) or discrepancy > 1e-8:
        raise ValueError(f"Saved error sign/values differ from q50 - observed: {discrepancy}")
    old = data[data.model.eq("old_frozen")].drop(columns="model")
    new = data[data.model.eq("reliability_selection")].drop(columns="model")
    pair = old.merge(new, on=KEY, suffixes=("_old", "_new"), validate="one_to_one")
    if len(pair) != len(old) or len(pair) != len(new):
        raise ValueError("Original and failure-aware targets are not fully paired")
    for col in ["cohort", "segment_key", "target_time_s", "observed_power_w"]:
        left, right = pair[f"{col}_old"], pair[f"{col}_new"]
        if not left.equals(right):
            raise ValueError(f"Paired targets disagree on {col}")
    pair = pair.rename(columns={"cohort_old": "cohort",
                                "segment_key_old": "segment_key",
                                "target_time_s_old": "target_time_s",
                                "observed_power_w_old": "observed_power_w"})
    pair["abs_error_old_w"] = pair["error_w_old"].abs()
    pair["abs_error_new_w"] = pair["error_w_new"].abs()
    return pair.sort_values(KEY)


def make_bins(pair: pd.DataFrame) -> pd.DataFrame:
    frame = pair[KEY[:-1] + ["cohort", "target_time_s",
                             "abs_error_old_w", "abs_error_new_w", "error_w_old", "error_w_new"]].copy()
    frame["bin_start_s"] = np.floor(frame.target_time_s / BIN_SECONDS).astype(int) * BIN_SECONDS
    rows = []
    for (assessment, horizon, run_id, cohort, start), group in frame.groupby(
        ["assessment", "horizon_ms", "run_id", "cohort", "bin_start_s"], sort=True
    ):
        record = {"assessment": assessment, "horizon_ms": horizon, "run_id": run_id,
                  "cohort": cohort, "bin_start_s": start,
                  "bin_center_s": start + BIN_SECONDS / 2,
                  "n_targets": len(group)}
        for suffix in ["old", "new"]:
            values = group[f"abs_error_{suffix}_w"].to_numpy()
            record[f"mae_{suffix}_w"] = float(values.mean())
            record[f"p95_{suffix}_w"] = float(np.quantile(values, 0.95))
            record[f"bias_{suffix}_w"] = float(group[f"error_w_{suffix}"].mean())
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["assessment", "run_id", "horizon_ms", "bin_start_s"])


def plot_overview(bins: pd.DataFrame, output: Path) -> None:
    # Average run-level bin MAE, so long runs cannot dominate a time bin.
    grouped = bins.groupby(["assessment", "horizon_ms", "bin_start_s"], as_index=False).agg(
        bin_center_s=("bin_center_s", "first"),
        mean_run_mae_old_w=("mae_old_w", "mean"),
        mean_run_mae_new_w=("mae_new_w", "mean"),
        n_runs=("run_id", "nunique"),
    )
    grouped.to_csv(output / "run_balanced_5s_mae.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained", sharex=True, sharey=True)
    for row, assessment in enumerate(["grouped_outer", "reused_holdout"]):
        for col, horizon in enumerate([1000, 2000]):
            ax = axes[row, col]
            part = grouped[(grouped.assessment == assessment) & (grouped.horizon_ms == horizon)]
            for suffix, (label, color) in zip(["old", "new"], MODELS.values()):
                ax.plot(part.bin_center_s, part[f"mean_run_mae_{suffix}_w"],
                        color=color, linewidth=1.8, label=label)
            ax.axhline(500, color="#bd4b41", linestyle="--", linewidth=1, alpha=.7)
            ax.set_title(f"{'Grouped test' if assessment == 'grouped_outer' else 'Reused holdout'} · {horizon // 1000} s",
                         loc="left")
            ax.set_xlabel("Target time within run (s)")
            ax.set_ylabel("Mean of run MAE in each 5 s bin (W)")
            ax.grid(alpha=.2)
    fig.suptitle("Forecast error over target time · 5 s bins · original (gray) vs failure-aware (teal)",
                 fontsize=15, fontweight="bold")
    fig.savefig(output / "mae_over_time_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def with_gaps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN breaks where the 200 ms target series is discontinuous."""
    if len(x) < 2:
        return x, y
    breaks = np.flatnonzero(np.diff(x) > .21) + 1
    return np.insert(x, breaks, np.nan), np.insert(y, breaks, np.nan)


def plot_run(pair: pd.DataFrame, bins: pd.DataFrame, assessment: str,
             run_id: str, cohort: str, output: Path) -> tuple[str, str, str]:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", run_id)
    stem = f"{assessment}__{safe}"
    subset = pair[(pair.assessment == assessment) & (pair.run_id == run_id)]
    binned = bins[(bins.assessment == assessment) & (bins.run_id == run_id)]
    files = []
    for metric, label in [("mae", "5 s mean absolute error"),
                          ("p95", "5 s 95th percentile absolute error"),
                          ("bias", "5 s mean signed error (prediction - observed)")]:
        fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), layout="constrained", sharex=True)
        for ax, horizon in zip(axes, [1000, 2000]):
            raw = subset[subset.horizon_ms.eq(horizon)].sort_values("target_time_s")
            part = binned[binned.horizon_ms.eq(horizon)].set_index("bin_start_s")
            full = part.reindex(range(int(part.index.min()), int(part.index.max()) + BIN_SECONDS, BIN_SECONDS))
            for suffix, (model_label, color) in zip(["old", "new"], MODELS.values()):
                if metric in ("mae", "bias"):
                    raw_col = f"abs_error_{suffix}_w" if metric == "mae" else f"error_w_{suffix}"
                    x, y = with_gaps(raw.target_time_s.to_numpy(),
                                     raw[raw_col].to_numpy())
                    ax.plot(x, y, color=color, alpha=.13, linewidth=.55)
                ax.plot(full.bin_center_s, full[f"{metric}_{suffix}_w"],
                        color=color, linewidth=1.8, label=model_label)
            ax.axhline(0 if metric == "bias" else 500, color="#bd4b41",
                       linestyle="--", linewidth=1, alpha=.7)
            ax.set_title(f"{horizon // 1000} s horizon · {len(raw):,} paired targets", loc="left")
            ax.set_ylabel("Signed error (W)" if metric == "bias" else "Absolute error (W)")
            ax.grid(alpha=.2)
        axes[-1].set_xlabel("Target time within run (s)")
        fig.suptitle(f"{run_id} · {cohort.replace('__', ' / ')}\n{label} over target time · original (gray) vs failure-aware (teal)",
                     fontsize=14, fontweight="bold")
        destination = output / "runs" / metric / f"{stem}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(destination.relative_to(output).as_posix())
    return tuple(files)


def write_readme(output: Path, pair: pd.DataFrame, run_links: list[tuple[str, str, str, str, str, str]]) -> None:
    lines = [
        "# 시간별 예측 오차",
        "",
        "가로축은 각 run 안의 목표 시각(target_time_s, 초)입니다. 1초·2초 horizon은 별도 패널로 표시합니다.",
        "각 시점의 오차는 abs(q50 - observed_power_w)로 다시 계산했습니다.",
        "회색은 기존 old_frozen, 청록색은 개선 reliability_selection입니다. 빨간 점선은 500 W 진단 기준입니다.",
        "두 모델은 같은 target_index에서 일대일로 짝지었고, history_meta/full15/seed 42만 사용했습니다.",
        "",
        "## 요약",
        "",
        "- [전체 시간별 5초 평균 오차](mae_over_time_overview.png): 각 시간 구간에서 run별 평균 절대오차를 먼저 계산한 뒤 run을 동일 가중치로 평균했습니다. 시간 구간별 참여 run 수는 [요약 CSV](run_balanced_5s_mae.csv)의 n_runs에 있습니다.",
        "- [실행별 5초 집계 CSV](per_run_5s_error.csv): 각 run과 horizon의 5초 구간 평균·95번째 백분위 절대오차 및 target 수.",
        "- 실행별 MAE·부호오차 그림의 흐린 선은 200ms 시점별 오차, 굵은 선은 5초 구간 평균입니다. P95 그림은 5초 구간의 95번째 백분위만 표시합니다.",
        "- 부호오차는 q50 - 관측값입니다. 양수는 과대예측, 음수는 저예측이며 빨간 점선은 0 W입니다.",
        "",
        "## 모든 실행",
        "",
        "| 평가 | Cohort | Run | 5초 평균 절대오차 | 5초 P95 절대오차 | 5초 평균 부호오차 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for assessment, cohort, run_id, mae, p95, bias in run_links:
        lines.append(f"| {assessment} | {cohort} | {run_id} | [그래프]({mae}) | [그래프]({p95}) | [그래프]({bias}) |")
    lines += ["", f"총 {pair.run_id.nunique()}개 run, {len(pair):,}개 일대일 대응 target을 사용했습니다.",
              "예측은 저장된 원자료에서 읽었으며 재학습하지 않았습니다.",
              ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pair = load_matched(args.source)
    bins = make_bins(pair)
    bins.to_csv(args.output / "per_run_5s_error.csv", index=False)
    plot_overview(bins, args.output)
    links = []
    for (assessment, run_id, cohort), _ in pair.groupby(["assessment", "run_id", "cohort"], sort=True):
        mae, p95, bias = plot_run(pair, bins, assessment, run_id, cohort, args.output)
        links.append((assessment, cohort, run_id, mae, p95, bias))
    write_readme(args.output, pair, links)
    print(f"Wrote 3 figures for each of {len(links)} runs, plus overview, to {args.output}")


if __name__ == "__main__":
    main()
