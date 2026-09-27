"""Summarize the staged grouped search from stored outer predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def run(out: Path):
    out = Path(out)
    predictions = pd.read_parquet(out / "per_target_predictions.parquet")
    per = pd.read_csv(out / "per_run_metrics.csv")
    aggregate = pd.read_csv(out / "aggregate_metrics.csv")
    paired = pd.read_csv(out / "paired_gain_ci.csv")
    selection = json.loads((out / "selection.json").read_text())
    predictions["covered_80"] = ((predictions.y >= predictions["q_0.1"]) &
                                 (predictions.y <= predictions["q_0.9"]))
    predictions["width_80_w"] = predictions["q_0.9"] - predictions["q_0.1"]
    calibration = predictions.groupby(["run_id","seed","condition"]).agg(
        coverage_80=("covered_80","mean"),
        width_80_w=("width_80_w","mean")).reset_index()
    calibration.to_csv(out / "calibration_metrics.csv", index=False)
    predictions["transition"] = np.where(predictions.delta_w>=100,"rise",
        np.where(predictions.delta_w<=-100,"drop","steady"))
    transition = predictions.groupby(["run_id","seed","condition","transition"]).agg(
        n_targets=("y","size"), crps_w=("crps_w","mean"),
        mae_w=("absolute_error_w","mean"),
        p95_w=("absolute_error_w",lambda x:x.quantile(.95)),
        p99_w=("absolute_error_w",lambda x:x.quantile(.99))).reset_index()
    transition.to_csv(out / "transition_metrics.csv",index=False)
    rows = []
    for rec in aggregate.itertuples():
        cover = calibration.loc[calibration.condition==rec.condition]
        rows.append(dict(condition=rec.condition, n_runs=rec.n_runs, crps_w=rec.crps_w,
                         mae_w=rec.mae_w, p95_w=rec.p95_w,p99_w=rec.p99_w,
                         coverage_80=cover.coverage_80.mean(),
                         width_80_w=cover.width_80_w.mean()))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary_metrics.csv",index=False)
    horizon_ms = json.loads((out / "candidate_manifest.json").read_text()).get("horizon_ms", 1000)
    horizon_s = horizon_ms / 1000
    lines = [
        f"# Grouped inner 선택: 기존 +{horizon_s:g}초 holdout의 회고적 평가",
        "",
        f"highres node의 8 GPU 합산 전력, 200 ms 완료 bin, +{horizon_s:g} s target에 대해 22개 개발 run에서 3-fold grouped inner 선택을 실행했다. 24개 공통 후보를 q50·200 tree로 screening하고 feature 조건별 상위 6개를 15 quantile·최대 1000 tree·early stopping patience 50으로 평가했다. 10개 바깥 test run은 이전 파일럿에서 이미 평가된 동일 holdout이므로 **새로운 독립 확인 실험이 아니다**. 설정은 seed 2026의 inner 평가에서 정하고, outer fit은 seed 17/42/2026으로 run당 1000개 행을 표집했다.",
        "",
        "| 조건 | CRPS (W) | MAE (W) | p95 / p99 (W) | 80% coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {"history_only_tuned":"History tuned",
              "history_static_s2_controlled":"S2 controlled",
              "history_static_s2_tuned":"S2 independently tuned"}
    for rec in summary.itertuples():
        lines.append(f"| {labels[rec.condition]} | {rec.crps_w:.2f} | {rec.mae_w:.2f} | {rec.p95_w:.1f} / {rec.p99_w:.1f} | {rec.coverage_80:.3f} |")
    lines += ["", "Controlled S2는 history-only에서 선택한 후보와 quantile별 tree 수를 고정하고 metadata 열만 추가했다. Independently tuned S2는 같은 후보 수·inner split·예산에서 별도로 선택했다. 모든 조건은 seed별로 같은 학습 행과 run 가중치, test 시점을 사용했다.", ""]
    for rec in paired.itertuples():
        lines.append(f"- {labels[rec.condition]}의 history 대비 gain: {rec.gain_pct:+.2f}% (paired run bootstrap 95% CI {rec.ci_low_pct:+.2f}% ~ {rec.ci_high_pct:+.2f}%; test run 10개).")
    lines += ["", "선택된 후보와 inner CRPS:", ""]
    for condition, rec in selection.items():
        lines.append(f"- {condition}: {rec['candidate_id']}, inner run-balanced CRPS {rec['inner_run_balanced_crps_w']:.2f} W. Quantile별 실제 outer tree 수는 selected_hyperparameters.csv에 있다.")
    lines += ["", "후보 screening과 15-quantile inner 결과는 screening_metrics.csv, full15_inner_metrics.csv에 있다. Per-target 예측, run별 점수, calibration 및 전이 지표와 재계산 검증은 같은 폴더에 저장했다. Selection은 24개 후보 전체를 full15로 평가한 결과가 아니라 사전 고정한 단계적 예산 검색이다.", "", "원본 수집 session의 독립성은 확인되지 않았다. 기존 holdout 재사용, 작은 test run 수, seed별 학습 표집 변동을 감안해 이 결과로 배포 설정이나 metadata의 보편적 개선을 확정하지 않는다. 장치·PCC 계측 및 제어 반응은 별도 Phase 0이 필요하다.", "", "재현 명령:", "", f"    python trace/grid_stability_forecast/run_nested_horizon.py --source trace/grid_stability_forecast/results/grid_fixed_20260926/h{horizon_ms} --out {out} --horizon-ms {horizon_ms}" if horizon_ms != 1000 else "    python trace/grid_stability_forecast/nested_select.py", f"    python trace/grid_stability_forecast/verify_nested.py --source trace/grid_stability_forecast/results/grid_fixed_20260926/h{horizon_ms} --out {out}" if horizon_ms != 1000 else "    python trace/grid_stability_forecast/verify_nested.py", f"    python trace/grid_stability_forecast/summarize_nested.py --out {out}", ""]
    (out / "report_ko.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=HERE/"results/nested_select_20260926")
    print(run(parser.parse_args().out).to_string(index=False))
