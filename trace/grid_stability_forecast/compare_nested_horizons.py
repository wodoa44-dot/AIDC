"""Compare verified reused-holdout nested runs across forecast horizons."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "nested_horizon_comparison_20260927"
SOURCES = {
    1: RESULTS / "nested_select_20260926",
    2: RESULTS / "nested_h2000_20260927",
    5: RESULTS / "nested_h5000_20260927",
    10: RESULTS / "nested_h10000_20260927",
    30: RESULTS / "nested_h30000_20260927",
}
CONDITIONS = {
    "history_only_tuned": "History tuned",
    "history_static_s2_controlled": "S2 controlled",
    "history_static_s2_tuned": "S2 tuned",
}


def run() -> None:
    OUT.mkdir(exist_ok=True)
    scores = []
    effects = []
    for horizon, directory in SOURCES.items():
        checks = json.loads((directory / "verification.json").read_text())
        if not checks.get("all_passed"):
            raise ValueError(f"Unverified input: {directory}")
        if horizon != 1:
            progress = json.loads((directory / "progress.json").read_text())
            if progress.get("status") != "completed":
                raise ValueError(f"Incomplete input: {directory}")
        summary = pd.read_csv(directory / "summary_metrics.csv")
        paired = pd.read_csv(directory / "paired_gain_ci.csv")
        if set(summary.condition) != set(CONDITIONS):
            raise ValueError(f"Condition mismatch: {directory}")
        summary["horizon_s"] = horizon
        paired["horizon_s"] = horizon
        scores.append(summary)
        effects.append(paired)
    scores = pd.concat(scores, ignore_index=True).sort_values(["horizon_s", "condition"])
    effects = pd.concat(effects, ignore_index=True).sort_values(["horizon_s", "condition"])
    scores.to_csv(OUT / "horizon_metrics.csv", index=False)
    effects.to_csv(OUT / "paired_gain.csv", index=False)
    colors = {"history_only_tuned":"#1f77b4",
              "history_static_s2_controlled":"#ff7f0e",
              "history_static_s2_tuned":"#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for condition, label in CONDITIONS.items():
        data = scores.loc[scores.condition.eq(condition)].sort_values("horizon_s")
        axes[0].plot(data.horizon_s, data.crps_w, marker="o", lw=1.8,
                     label=label, color=colors[condition])
        axes[1].plot(data.horizon_s, data.coverage_80, marker="o", lw=1.8,
                     label=label, color=colors[condition])
    axes[1].axhline(.8, color="0.3", ls="--", lw=1, label="Nominal 80%")
    axes[0].set(ylabel="Run-balanced CRPS (W)", xlabel="Forecast horizon (s)")
    axes[1].set(ylabel="Run-balanced 80% interval coverage", xlabel="Forecast horizon (s)", ylim=(0, 1))
    for ax in axes:
        ax.set_xticks(list(SOURCES))
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Highres reused holdout: staged grouped inner selection")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"nested_horizon_scores.{ext}", dpi=250)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for condition in ("history_static_s2_controlled", "history_static_s2_tuned"):
        data = effects.loc[effects.condition.eq(condition)].sort_values("horizon_s")
        ax.errorbar(data.horizon_s, data.gain_pct,
            yerr=[data.gain_pct-data.ci_low_pct, data.ci_high_pct-data.gain_pct],
            marker="o", capsize=3, lw=1.5, label=CONDITIONS[condition],
            color=colors[condition])
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.set(xlabel="Forecast horizon (s)", ylabel="Paired run CRPS gain (%)",
           xticks=list(SOURCES), title="Metadata gain; descriptive paired bootstrap intervals")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"nested_horizon_gain.{ext}", dpi=250)
    plt.close(fig)
    compact = scores[["horizon_s", "condition", "crps_w", "coverage_80"]]
    lines = ["# Highres horizon별 grouped inner 평가", "",
        "+1/2/5/10/30초 각각에서 같은 24개 후보, 조건별 shortlist 6개, 개발 run 내부 3-fold 선택과 3개 outer seed를 사용했다. 기존 22/10 run 분할과 동일한 재사용 holdout이다. Horizon마다 eligible target 행 수가 달라 절대 점수의 완전한 짝비교가 아니며, 200 ms 완료 bin 평균 target의 1–30초 미래 수준을 비교한다.", "",
        compact.to_markdown(index=False, floatfmt=".3f"), "",
        "## 동일 horizon 내 paired run CRPS gain", "",
        effects[["horizon_s", "condition", "gain_pct", "ci_low_pct", "ci_high_pct"]].to_markdown(index=False, floatfmt=".3f"), "",
        "Controlled S2는 history에서 선택한 후보·tree 수를 고정하고 metadata만 추가했다. Tuned S2는 같은 inner 예산 안에서 별도 선택했다. CI는 기존 10개 test run을 재표집한 기술 통계다. 원본 session 독립성이 확인되지 않았고 수집일과 GPU×workload 조합이 결합돼 있어 새로운 외부 일반화 CI가 아니다. CRPS가 낮아도 80% coverage가 명목치와 어긋날 수 있다.", "",
        "그림: `nested_horizon_scores.png`(CRPS·coverage), `nested_horizon_gain.png`(paired gain). 표 원자료: `horizon_metrics.csv`, `paired_gain.csv`. 재현: `python trace/grid_stability_forecast/compare_nested_horizons.py`.", ""]
    (OUT / "report_ko.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run()
