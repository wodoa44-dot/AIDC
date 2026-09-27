"""Rebuild detailed outer-CV metrics from the five verified prediction files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "results/nested_cv5_20260926"
BASE = "history_only_tuned"
CONDITIONS = (BASE, "history_static_s2_controlled", "history_static_s2_tuned")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(out: Path) -> None:
    progress = json.loads((out / "progress.json").read_text())
    if progress.get("status") != "completed":
        raise ValueError("All outer folds must be completed before summarizing")
    groups = pd.read_csv(out / "split_manifest.csv")
    rows = []
    selected = []
    for fold in range(1, 6):
        result = out / f"fold{fold}" / "results"
        path = result / "per_target_predictions.parquet"
        recorded = progress["folds"][f"fold{fold}"]
        if recorded["status"] != "verified" or recorded["predictions_sha256"] != sha256(path):
            raise ValueError(f"Fold {fold} is not verified or its predictions changed")
        checks = json.loads((result / "verification.json").read_text())
        if not checks.get("all_passed"):
            raise ValueError(f"Fold {fold} failed verification")
        pred = pd.read_parquet(path)
        expected = set(groups.loc[groups.outer_fold.eq(fold), "run_id"])
        if set(pred.run_id) != expected or set(pred.condition) != set(CONDITIONS):
            raise ValueError(f"Fold {fold} test runs or conditions differ")
        key = ["run_id", "seed", "issue_s"]
        paired = pred.groupby(key, sort=False)
        if not paired.condition.nunique().eq(3).all() or not paired.y.nunique().eq(1).all():
            raise ValueError(f"Fold {fold} has unpaired predictions")
        pred["outer_fold"] = fold
        pred["transition"] = np.select(
            [pred.delta_w >= 100, pred.delta_w <= -100], ["rise", "drop"], default="steady"
        )
        pred["covered_80"] = pred.y.between(pred["q_0.1"], pred["q_0.9"])
        pred["width_80_w"] = pred["q_0.9"] - pred["q_0.1"]
        pred["signed_error_w"] = pred.q50 - pred.y
        for (run_id, seed, condition, transition), part in pred.groupby(
            ["run_id", "seed", "condition", "transition"], sort=False
        ):
            error = part.absolute_error_w.to_numpy()
            rows.append(dict(run_id=run_id, seed=seed, condition=condition,
                transition=transition, outer_fold=fold, n_targets=len(part),
                crps_w=part.crps_w.mean(), mae_w=error.mean(),
                p95_w=np.quantile(error, .95), p99_w=np.quantile(error, .99),
                bias_w=part.signed_error_w.mean(),
                coverage_80=part.covered_80.mean(), width_80_w=part.width_80_w.mean()))
        selected.append(pd.read_csv(result / "selected_hyperparameters.csv").assign(outer_fold=fold))

    detail = pd.DataFrame(rows)
    detail.to_csv(out / "per_run_transition_metrics.csv", index=False)
    selected_hp = pd.concat(selected, ignore_index=True)
    selected_hp.to_csv(out / "selected_hyperparameters.csv", index=False)
    # Average seed repeats within each independent run before averaging runs.
    run = detail.groupby(["run_id", "condition", "transition"], as_index=False).agg(
        n_targets=("n_targets", "first"), crps_w=("crps_w", "mean"),
        mae_w=("mae_w", "mean"), p95_w=("p95_w", "mean"), p99_w=("p99_w", "mean"),
        bias_w=("bias_w", "mean"), coverage_80=("coverage_80", "mean"),
        width_80_w=("width_80_w", "mean"))
    aggregate = run.groupby(["condition", "transition"], as_index=False).agg(
        n_runs=("run_id", "nunique"), n_targets=("n_targets", "sum"),
        crps_w=("crps_w", "mean"), mae_w=("mae_w", "mean"),
        p95_w=("p95_w", "mean"), p99_w=("p99_w", "mean"),
        bias_w=("bias_w", "mean"), coverage_80=("coverage_80", "mean"),
        width_80_w=("width_80_w", "mean"))
    aggregate.to_csv(out / "transition_metrics.csv", index=False)
    # The overall score is computed from all targets in each run, not from an
    # unweighted average of rise/drop/steady subgroup scores.
    original = pd.read_csv(out / "per_run_metrics.csv")
    overall = original.groupby(["run_id", "condition"], as_index=False).agg(
        crps_w=("crps_w", "mean"), mae_w=("mae_w", "mean"),
        p95_w=("p95_w", "mean"), p99_w=("p99_w", "mean"))
    overall_summary = overall.groupby("condition", as_index=False)[
        ["crps_w", "mae_w", "p95_w", "p99_w"]].mean()
    coverage = run.groupby(["run_id", "condition"]).apply(
        lambda x: pd.Series({
            "coverage_80": np.average(x.coverage_80, weights=x.n_targets),
            "width_80_w": np.average(x.width_80_w, weights=x.n_targets),
        }), include_groups=False).reset_index()
    coverage.groupby("condition", as_index=False)[["coverage_80", "width_80_w"]].mean().to_csv(
        out / "calibration_metrics.csv", index=False)
    overall_summary = overall_summary.merge(
        pd.read_csv(out / "calibration_metrics.csv"), on="condition", validate="one_to_one")
    overall_summary.to_csv(out / "summary_metrics.csv", index=False)
    # Reconcile against the runner's already saved run-balanced scores.
    prior = pd.read_csv(out / "paired_gain_ci.csv")
    for record in prior.itertuples():
        score = overall_summary.set_index("condition")
        if not np.isclose(record.history_crps_w, score.loc[BASE, "crps_w"]) or not np.isclose(
            record.metadata_crps_w, score.loc[record.condition, "crps_w"]
        ):
            raise AssertionError("Rebuilt CRPS differs from runner summary")
    report = ["# 5×3 grouped nested CV: 완료된 회고적 평가", "",
        "highres node 32개 run의 8 GPU 합산 전력, 완료된 200 ms bin에서 +1 s 전력을 예측했다. "
        "Outer 5-fold에서 run별 test 예측을 한 번씩 만들고 학습 표집 seed 17/42/2026을 평균했다. "
        "Inner 3-fold에서 공통 24개 후보를 q50으로 screening한 뒤 각 조건 상위 6개를 15 quantile로 선택했다. "
        "Controlled S2는 history-only의 후보와 quantile별 tree 수를 고정했다. 독립 선택 S2는 같은 후보 예산으로 별도 선택했다.", "",
        "| 조건 | CRPS (W) | MAE (W) | run 평균 p95 / p99 (W) | 80% coverage |",
        "|---|---:|---:|---:|---:|"]
    for r in overall_summary.itertuples():
        report.append(f"| {r.condition} | {r.crps_w:.2f} | {r.mae_w:.2f} | {r.p95_w:.1f} / {r.p99_w:.1f} | {r.coverage_80:.3f} |")
    report += ["", "100 W endpoint ΔP 기준의 전이별 결과(기준은 호환성용이며 계통 허용치가 아님):", "",
        "| 조건 | 구간 | run 수 | target 수 | CRPS (W) | MAE (W) | p95 / p99 (W) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in aggregate.itertuples():
        report.append(f"| {r.condition} | {r.transition} | {r.n_runs} | {r.n_targets} | {r.crps_w:.1f} | {r.mae_w:.1f} | {r.p95_w:.1f} / {r.p99_w:.1f} |")
    report += ["", "Paired run-cluster bootstrap 95% CI (저장 예측만 재표집):", ""]
    for r in prior.itertuples():
        report.append(f"- {r.condition}: history 대비 {r.gain_pct:+.2f}% "
                      f"({r.ci_low_pct:+.2f}% ~ {r.ci_high_pct:+.2f}%; 32 run).")
    report += ["", "원본 32개 trace는 네 수집일에 걸쳐 있으며 각 날짜와 GPU×workload 조합이 일대일로 결합된다(`../actual_20260926/highres_session_report_ko.md`). 이 CV는 같은 날짜 기록을 train/test 양쪽에 두므로 수집일 밖 일반화나 조합 효과를 확인하지 않는다. 같은 데이터의 이전 파일럿 결과도 이미 보았다. "
        "이 CI는 재학습·설정 선택 불확실성을 포함하지 않는다. 80% coverage가 nominal 값보다 낮으면 "
        "calibration 개선으로 해석하지 않는다. 날짜별 저장 예측의 기술적 민감도는 `day_sensitivity_ko.md`에 있다. GPU trace만 있으므로 시설 PCC 또는 장치 제어 성능은 입증하지 않는다.", "",
        "재계산 자료: `per_run_transition_metrics.csv`, `transition_metrics.csv`, "
        "`calibration_metrics.csv`, `summary_metrics.csv`, `selected_hyperparameters.csv`. "
        "Fold별 원본 예측과 검증은 `fold*/results/`에 있다.", ""]
    (out / "report_ko.md").write_text("\n".join(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT)
    summarize(parser.parse_args().out)
