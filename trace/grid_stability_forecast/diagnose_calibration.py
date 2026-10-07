"""Run-balanced, read-only calibration diagnostics for stored nested predictions.

This does not fit, select, or recalibrate on the reused outer holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import QUANTILES, write_json

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "results/nested_h2000_20260927"
LABELS = {
    "history_only_tuned": "History tuned",
    "history_static_s2_controlled": "S2 controlled",
    "history_static_s2_tuned": "S2 tuned",
}


def run(out: Path) -> None:
    out = Path(out)
    progress = json.loads((out / "progress.json").read_text())
    if progress.get("status") != "completed" or not progress.get("verification", {}).get("all_passed"):
        raise ValueError("Nested output is not completed and verified")
    prediction_path = out / "per_target_predictions.parquet"
    if hashlib.sha256(prediction_path.read_bytes()).hexdigest() != progress["prediction_sha256"]:
        raise ValueError("Stored prediction hash does not match completed experiment")
    horizon_s = progress["horizon_ms"] / 1000
    pred = pd.read_parquet(prediction_path)
    pred["transition"] = np.select(
        [pred.delta_w >= 100, pred.delta_w <= -100],
        ["rise", "drop"], default="steady")
    pred["covered_80"] = ((pred.y >= pred["q_0.1"]) &
                          (pred.y <= pred["q_0.9"]))
    pred["width_80_w"] = pred["q_0.9"] - pred["q_0.1"]
    pred["signed_error_w"] = pred.q50 - pred.y
    group = ["condition", "run_id", "seed", "transition"]
    per_seed = pred.groupby(group, observed=True).agg(
        n_targets=("y", "size"), coverage_80=("covered_80", "mean"),
        width_80_w=("width_80_w", "mean"), crps_w=("crps_w", "mean"),
        mae_w=("absolute_error_w", "mean"),
        signed_error_w=("signed_error_w", "mean"),
    ).reset_index()
    per_run = per_seed.groupby(["condition", "run_id", "transition"], observed=True).agg(
        n_targets=("n_targets", "first"), coverage_80=("coverage_80", "mean"),
        width_80_w=("width_80_w", "mean"), crps_w=("crps_w", "mean"),
        mae_w=("mae_w", "mean"), signed_error_w=("signed_error_w", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    if not per_run.n_seeds.eq(3).all():
        raise AssertionError("Missing seed for a run and transition")
    per_run.to_csv(out / "calibration_by_run_transition.csv", index=False)
    summary = per_run.groupby(["condition", "transition"], observed=True).agg(
        n_runs=("run_id", "nunique"), n_targets=("n_targets", "sum"),
        coverage_80=("coverage_80", "mean"), width_80_w=("width_80_w", "mean"),
        crps_w=("crps_w", "mean"), mae_w=("mae_w", "mean"),
        signed_error_w=("signed_error_w", "mean"),
    ).reset_index()
    summary.to_csv(out / "calibration_by_transition.csv", index=False)
    qcols = [f"q_{q:g}" for q in QUANTILES]
    actual = pred.y.to_numpy()
    values = pred[qcols].to_numpy()
    # One 0/1 hit per target and nominal quantile, averaged per seed then run.
    hits = (actual[:, None] <= values).astype(np.float32)
    reliability = pd.DataFrame(hits, columns=qcols)
    reliability["condition"] = pred.condition.to_numpy()
    reliability["run_id"] = pred.run_id.to_numpy()
    reliability["seed"] = pred.seed.to_numpy()
    per_seed_q = reliability.groupby(["condition", "run_id", "seed"], observed=True)[qcols].mean()
    per_run_q = per_seed_q.groupby(["condition", "run_id"], observed=True).mean().reset_index()
    per_run_q.to_csv(out / "calibration_quantiles_by_run.csv", index=False)
    overall_q = per_run_q.groupby("condition", observed=True)[qcols].mean().reset_index()
    overall_q.to_csv(out / "calibration_quantiles.csv", index=False)
    scores = []
    for _, rec in overall_q.iterrows():
        observed = rec[qcols].to_numpy(dtype=float)
        scores.append(dict(condition=rec["condition"],
                           mean_abs_quantile_gap=float(np.abs(observed - QUANTILES).mean()),
                           max_abs_quantile_gap=float(np.abs(observed - QUANTILES).max())))
    pd.DataFrame(scores).to_csv(out / "calibration_gap_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    ax.plot([0, 1], [0, 1], color="0.3", linestyle="--", label="Nominal")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (i, (_, rec)) in enumerate(overall_q.iterrows()):
        observed = rec[qcols].to_numpy(dtype=float)
        ax.plot(QUANTILES, observed, marker="o", ms=3, lw=1.5,
                label=LABELS[rec["condition"]], color=colors[i])
    ax.set(xlabel="Nominal quantile", ylabel="Observed fraction below prediction",
           title=f"+{horizon_s:g} s highres: run-balanced quantile reliability", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"calibration_reliability.{ext}", dpi=250)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    conditions = list(LABELS)
    transitions = ["steady", "rise", "drop"]
    x = np.arange(len(transitions))
    for i, condition in enumerate(conditions):
        part = summary.set_index(["condition", "transition"]).loc[condition]
        ax.bar(x + (i-1)*.24, part.loc[transitions, "coverage_80"], .23,
               label=LABELS[condition], color=colors[i])
    ax.axhline(.8, color="0.3", linestyle="--", lw=1)
    ax.set(xticks=x, xticklabels=transitions, ylim=(0, 1),
           ylabel="Empirical 80% interval coverage",
           title=f"+{horizon_s:g} s highres: conditional coverage (outcome groups)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"calibration_transition.{ext}", dpi=250)
    plt.close(fig)
    text = [f"# +{horizon_s:g}초 nested 평가의 사후 calibration 진단", "",
        "검증된 기존 outer 예측을 읽어 각 seed 내 target을 평균하고, seed를 평균한 뒤 run에 동일 가중치를 주었다. 여기서 어떤 보정도 test에 학습·선택하지 않았다. 10개 test run은 이미 여러 파일럿에서 재사용된 holdout이다.", "",
        "## 100 W endpoint ΔP로 구분한 결과", "",
        summary.to_markdown(index=False, floatfmt=".3f"), "",
        "전이 구분은 미래 실제 ΔP로 만들었으므로 예측 시점에 알려진 층이 아니다. 구간별 coverage는 오류 진단용이며 조건부 보정 정책의 근거가 아니다.", "",
        f"S2 tuned의 steady/rise/drop coverage는 " + "/".join(f"{summary.set_index(['condition','transition']).loc[('history_static_s2_tuned',t),'coverage_80']:.3f}" for t in ["steady","rise","drop"]) + ". 명목 0.8과 구간별로 비교해야 하며, 이 분류는 미래 실제 전력으로 정의된다.", "",
        "## Quantile reliability gap", "",
        pd.DataFrame(scores).to_markdown(index=False, floatfmt=".3f"), "",
        "명목 80% 구간은 q10–q90이다. 같은 run의 연속 시점은 독립 표본으로 세지 않았고, 원본 수집일과 GPU×workload 조합은 결합돼 있어 이 표의 차이를 metadata의 인과 효과나 새 환경의 보정 성능으로 해석할 수 없다.", "",
        f"재현: `python trace/grid_stability_forecast/diagnose_calibration.py --out {out}`. CSV와 PNG/PDF가 같은 폴더에 있다.", ""]
    (out / "calibration_diagnostic_ko.md").write_text("\n".join(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT)
    run(parser.parse_args().out)
