#!/usr/bin/env python3
"""Build line-chart summaries for the three-stage forecasting experiment."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PREDICTABILITY = ROOT / "trace/predictability/outputs/main_20260914/comparison_summary.csv"
FAILURE_AWARE = (
    ROOT
    / "trace/model_200ms_1s_2s/improvements/failure_aware_20260921_v1"
    / "artifacts/full/aggregate_metrics.csv"
)
EPISODES = FAILURE_AWARE.with_name("failure_episode_summary.csv")
TRANSITIONS = FAILURE_AWARE.with_name("transition_metrics.csv")
COHORTS = FAILURE_AWARE.with_name("per_cohort_metrics.csv")
FIGURE_DIR = ROOT / "reports/visuals/figures"
DATA_DIR = ROOT / "reports/visuals/plot_data"

COLORS = {
    "blue": "#2673B8",
    "orange": "#E1812C",
    "green": "#3A923A",
    "red": "#C33C54",
    "gray": "#67717E",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def initial_history_gain(rows: list[dict[str, str]]) -> pd.DataFrame:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["resolution_ms"]), int(row["horizon_ms"]))].append(
            float(row["history_gain_vs_current_pct"])
        )
    return pd.DataFrame(
        [
            {
                "resolution_ms": resolution,
                "horizon_ms": horizon,
                "history_gain_pct": float(np.mean(values)),
            }
            for (resolution, horizon), values in sorted(grouped.items())
        ]
    )


def select_one(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def metadata_crps(rows: list[dict[str, str]]) -> pd.DataFrame:
    records = []
    for horizon in (1000, 2000):
        for condition, label in (("history_only", "전력 이력만"), ("history_meta", "이력 + 메타데이터")):
            row = select_one(
                rows,
                assessment="grouped_outer",
                horizon_ms=str(horizon),
                model="old_frozen",
                condition=condition,
                representation="full15",
                seed="42",
                weighting="run_balanced",
            )
            records.append({"horizon_ms": horizon, "series": label, "crps_w": float(row["crps_w"])})
    return pd.DataFrame(records)


def failure_comparison(rows: list[dict[str, str]], assessment: str) -> pd.DataFrame:
    records = []
    specifications = (
        ("old_frozen", "compact3", "기존 모델"),
        ("reliability_selection", "full15", "보완 모델"),
    )
    for horizon in (1000, 2000):
        for model, representation, label in specifications:
            row = select_one(
                rows,
                assessment=assessment,
                horizon_ms=str(horizon),
                model=model,
                condition="history_meta",
                representation=representation,
                seed="42",
                weighting="target_weighted",
            )
            records.append(
                {
                    "assessment": assessment,
                    "horizon_ms": horizon,
                    "series": label,
                    "mae_w": float(row["mae_w"]),
                    "exceedance_500_pct": 100.0 * float(row["exceedance_500_fraction"]),
                    "p95_w": float(row["abs_error_p95_w"]),
                    "p99_w": float(row["abs_error_p99_w"]),
                    "max_w": float(row["abs_error_max_w"]),
                }
            )
    return pd.DataFrame(records)


def episode_comparison(rows: list[dict[str, str]], assessment: str) -> pd.DataFrame:
    records = []
    specifications = (
        ("old_frozen", "compact3", "기존 모델"),
        ("reliability_selection", "full15", "보완 모델"),
    )
    for horizon in (1000, 2000):
        for model, representation, label in specifications:
            row = select_one(
                rows,
                assessment=assessment,
                horizon_ms=str(horizon),
                model=model,
                condition="history_meta",
                representation=representation,
            )
            records.append(
                {
                    "assessment": assessment,
                    "horizon_ms": horizon,
                    "series": label,
                    "longest_episode_s": float(row["duration_max_s"]),
                }
            )
    return pd.DataFrame(records)


def situation_comparison(rows: list[dict[str, str]], group_column: str) -> pd.DataFrame:
    records = []
    specifications = (
        ("old_frozen", "compact3", "기존 모델"),
        ("reliability_selection", "full15", "보완 모델"),
    )
    groups = sorted({row[group_column] for row in rows if row["assessment"] == "grouped_outer"})
    for group in groups:
        for horizon in (1000, 2000):
            for model, representation, label in specifications:
                row = select_one(
                    rows,
                    assessment="grouped_outer",
                    horizon_ms=str(horizon),
                    model=model,
                    condition="history_meta",
                    representation=representation,
                    **{group_column: group},
                )
                records.append(
                    {
                        group_column: group,
                        "horizon_ms": horizon,
                        "series": label,
                        "mae_w": float(row["mae_w"]),
                        "exceedance_500_pct": 100.0 * float(row["exceedance_500_fraction"]),
                        "mean_signed_error_w": float(row["mean_signed_error_w"]),
                        "n_targets": int(row["n_targets"]),
                    }
                )
    return pd.DataFrame(records)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#DCE2E8", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([1000, 2000], ["1 s", "2 s"])


def annotate(ax: plt.Axes, x: list[int], y: list[float], fmt: str, color: str) -> None:
    for x_value, y_value in zip(x, y):
        ax.annotate(
            fmt.format(y_value),
            (x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=color,
        )


def plot_overview(initial: pd.DataFrame, metadata: pd.DataFrame, grouped: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    for resolution, color in zip((50, 200, 1000), (COLORS["blue"], COLORS["orange"], COLORS["green"])):
        part = initial[initial["resolution_ms"] == resolution]
        x = part["horizon_ms"].tolist()
        y = part["history_gain_pct"].tolist()
        ax.plot(x, y, marker="o", linewidth=2, color=color, label=f"{resolution} ms 구간")
    ax.axhline(0, color=COLORS["gray"], linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([50, 200, 1000, 2000], ["50 ms", "200 ms", "1 s", "2 s"])
    ax.set_ylabel("현재값 모델 대비 CRPS 개선율 (%)")
    ax.set_title("A. 초기 예측 가능성 실험")
    ax.grid(axis="y", color="#DCE2E8", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for label, color in (("전력 이력만", COLORS["gray"]), ("이력 + 메타데이터", COLORS["blue"])):
        part = metadata[metadata["series"] == label]
        x = part["horizon_ms"].tolist()
        y = part["crps_w"].tolist()
        ax.plot(x, y, marker="o", linewidth=2.3, color=color, label=label)
        annotate(ax, x, y, "{:.1f}", color)
    style_axis(ax)
    ax.set_ylabel("Run-balanced CRPS (W, 낮을수록 좋음)")
    ax.set_title("B. 200 ms 모델: 메타데이터 효과")
    ax.legend(frameon=False, fontsize=8)

    for ax, metric, title, ylabel, fmt in (
        (axes[1, 0], "mae_w", "C. Failure-aware 재학습: MAE", "Target-weighted MAE (W)", "{:.1f}"),
        (
            axes[1, 1],
            "exceedance_500_pct",
            "D. Failure-aware 재학습: 대오차 비율",
            "|오차|가 500 W를 넘는 예측 (%)",
            "{:.1f}%",
        ),
    ):
        for label, color in (("기존 모델", COLORS["gray"]), ("보완 모델", COLORS["red"])):
            part = grouped[grouped["series"] == label]
            x = part["horizon_ms"].tolist()
            y = part[metric].tolist()
            ax.plot(x, y, marker="o", linewidth=2.3, color=color, label=label)
            annotate(ax, x, y, fmt, color)
        style_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("전력 예측 실험 흐름: 예측 가능성 탐색에서 failure-aware 재학습까지", fontsize=15)
    fig.text(
        0.5,
        -0.015,
        "C-D는 grouped outer 대상에서 기존 compact-3 배포 출력과 보완된 full-15 모델을 비교한다.",
        ha="center",
        fontsize=8,
        color="#4F5965",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"experiment_progression_lines.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(grouped: pd.DataFrame, holdout: pd.DataFrame, episodes: pd.DataFrame) -> None:
    combined = pd.concat([grouped, holdout], ignore_index=True)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.7), constrained_layout=True)
    specifications = (
        ("mae_w", "평균 절대오차(MAE)", "W", "{:.0f}"),
        ("p95_w", "절대오차 95백분위", "W", "{:.0f}"),
        ("p99_w", "절대오차 99백분위", "W", "{:.0f}"),
        ("max_w", "관측된 최대 절대오차", "W", "{:.0f}"),
        ("exceedance_500_pct", "|오차| > 500 W", "전체 예측 중 비율(%)", "{:.1f}"),
        ("longest_episode_s", "500 W 초과 최장 지속구간", "초", "{:.1f}"),
    )
    episode_all = episodes
    for ax, (metric, title, ylabel, fmt) in zip(axes.flat, specifications):
        source = episode_all if metric == "longest_episode_s" else combined
        for assessment, linestyle in (("grouped_outer", "-"), ("reused_holdout", "--")):
            for label, color in (("기존 모델", COLORS["gray"]), ("보완 모델", COLORS["red"])):
                part = source[(source["assessment"] == assessment) & (source["series"] == label)]
                ax.plot(
                    part["horizon_ms"],
                    part[metric],
                    marker="o",
                    linewidth=2,
                    linestyle=linestyle,
                    color=color,
                    label=f"{label} / {'outer' if assessment == 'grouped_outer' else 'holdout'}",
                )
        style_axis(ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("기존 모델과 failure-aware 보완 모델의 오차 진단", fontsize=15)
    fig.text(
        0.5,
        -0.055,
        "실선: grouped outer, 점선: 기존에 확인한 reused holdout. 최대값은 관측 사례이며 미래 오차의 상한이 아니다.",
        ha="center",
        fontsize=8,
        color="#4F5965",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"failure_aware_diagnostics_lines.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_cohort(data: pd.DataFrame) -> None:
    cohorts = [
        ("B200__image_generation", "B200 · 이미지 생성"),
        ("B200__llm_training", "B200 · LLM 학습"),
        ("H100__image_generation", "H100 · 이미지 생성"),
        ("H100__llm_training", "H100 · LLM 학습"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.0), sharex=True, constrained_layout=True)
    for column, (cohort, title) in enumerate(cohorts):
        for row_index, (metric, ylabel) in enumerate(
            (("mae_w", "MAE (W)"), ("exceedance_500_pct", "500 W 초과 비율 (%)"))
        ):
            ax = axes[row_index, column]
            for label, color in (("기존 모델", COLORS["gray"]), ("보완 모델", COLORS["red"])):
                part = data[(data["cohort"] == cohort) & (data["series"] == label)]
                ax.plot(part["horizon_ms"], part[metric], marker="o", linewidth=2.1, color=color, label=label)
            style_axis(ax)
            if row_index == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=2, frameon=False)
    fig.suptitle("GPU·워크로드 상황별 기존 모델과 보완 모델 비교 (grouped outer)", fontsize=15)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"failure_aware_by_cohort_lines.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_transition(data: pd.DataFrame) -> None:
    transitions = [("stable", "안정"), ("rising", "상승"), ("falling", "하락")]
    metrics = (
        ("mae_w", "MAE (W)"),
        ("exceedance_500_pct", "500 W 초과 비율 (%)"),
        ("mean_signed_error_w", "평균 signed error (W)"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 9.2), sharex=True, constrained_layout=True)
    for column, (transition, title) in enumerate(transitions):
        for row_index, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_index, column]
            for label, color in (("기존 모델", COLORS["gray"]), ("보완 모델", COLORS["red"])):
                part = data[(data["transition_stratum"] == transition) & (data["series"] == label)]
                ax.plot(part["horizon_ms"], part[metric], marker="o", linewidth=2.1, color=color, label=label)
            if metric == "mean_signed_error_w":
                ax.axhline(0, color="#AAB2BC", linewidth=0.9)
            style_axis(ax)
            if row_index == 0:
                ax.set_title(f"{title} 구간")
            if column == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=2, frameon=False)
    fig.suptitle("전력 변화 상황별 기존 모델과 보완 모델 비교 (grouped outer)", fontsize=15)
    fig.text(
        0.5,
        -0.04,
        "Signed error는 예측-관측이다. 음수는 과소예측, 양수는 과대예측을 뜻한다.",
        ha="center",
        fontsize=8,
        color="#4F5965",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"failure_aware_by_transition_lines.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    initial = initial_history_gain(read_rows(PREDICTABILITY))
    aggregate_rows = read_rows(FAILURE_AWARE)
    metadata = metadata_crps(aggregate_rows)
    grouped = failure_comparison(aggregate_rows, "grouped_outer")
    holdout = failure_comparison(aggregate_rows, "reused_holdout")
    episodes = pd.concat(
        [
            episode_comparison(read_rows(EPISODES), "grouped_outer"),
            episode_comparison(read_rows(EPISODES), "reused_holdout"),
        ],
        ignore_index=True,
    )
    cohorts = situation_comparison(read_rows(COHORTS), "cohort")
    transitions = situation_comparison(read_rows(TRANSITIONS), "transition_stratum")

    initial.assign(panel="initial_history_gain").to_csv(
        DATA_DIR / "experiment_initial_history_gain.csv", index=False
    )
    metadata.assign(panel="metadata_crps").to_csv(DATA_DIR / "experiment_metadata_crps.csv", index=False)
    pd.concat([grouped, holdout], ignore_index=True).to_csv(
        DATA_DIR / "experiment_failure_aware_metrics.csv", index=False
    )
    episodes.to_csv(DATA_DIR / "experiment_failure_episode_metrics.csv", index=False)
    cohorts.to_csv(DATA_DIR / "experiment_failure_aware_by_cohort.csv", index=False)
    transitions.to_csv(DATA_DIR / "experiment_failure_aware_by_transition.csv", index=False)

    plt.rcParams.update({"font.family": "NanumGothic", "font.size": 9.5, "axes.unicode_minus": False})
    plot_overview(initial, metadata, grouped)
    plot_diagnostics(grouped, holdout, episodes)
    plot_by_cohort(cohorts)
    plot_by_transition(transitions)


if __name__ == "__main__":
    main()
