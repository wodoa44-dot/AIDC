"""Choose onset alarm thresholds on held-out development runs only.

The final test probabilities are from train_onset_aligned.py's fixed history model.
No test outcome enters threshold selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd

from core import history_features
from evaluate_onset import match_one_to_one
from train import sample_indices
from train_onset_aligned import SOURCE, event_times, labels

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "results/onset_threshold_20260927"
PREDICTION = HERE / "results/onset_aligned_20260927/event_predictions.parquet"
THRESHOLDS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
DIRECTIONS = ("rise_event", "drop_event")


def score(scored: pd.DataFrame, raw: pd.DataFrame, threshold: float, direction: str) -> dict:
    n_events = n_alarms = n_matched = n_false = 0
    hours = 0.0
    for run_id, whole in raw.groupby("run_id", sort=False):
        events, bounds = event_times(whole)
        for segment, part in whole.groupby("segment_id", sort=False):
            start, end = bounds[segment]
            hours += max(0, end-start) / 3600
            truth = events[(segment, direction)]
            predictions = scored.loc[scored.run_id.eq(run_id) &
                scored.segment_id.eq(segment) & scored.event.eq(direction)].sort_values("issue_s")
            issue = predictions.issue_s.to_numpy(float)
            high = predictions.probability.to_numpy(float) >= threshold
            consecutive = np.r_[False, np.isclose(np.diff(issue), .2, atol=1e-5)]
            alarm = issue[high & ~(np.r_[False, high[:-1]] & consecutive)]
            alarm = alarm[(alarm >= start) & (alarm <= end-1.0)]
            pairs, missed, false = match_one_to_one(truth, alarm, 1.0)
            n_events += len(truth)
            n_alarms += len(alarm)
            n_matched += len(pairs)
            n_false += false
            if missed + len(pairs) != len(truth):
                raise AssertionError("Onset counts do not reconcile")
    return dict(threshold=threshold, direction=direction, n_onsets=n_events,
        n_alarms=n_alarms, n_matched=n_matched, n_missed=n_events-n_matched,
        n_false_alarms=n_false, observed_hours=hours,
        recall=n_matched/n_events if n_events else np.nan,
        precision=n_matched/n_alarms if n_alarms else np.nan,
        false_alarms_per_hour=n_false/hours if hours else np.nan)


def run(out: Path) -> None:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(SOURCE / "targets.parquet")
    raw = raw.loc[raw.horizon_ms.eq(1000)].reset_index(drop=True)
    frame, _ = labels(raw)
    split = pd.read_csv(SOURCE / "split_manifest.csv").set_index("group_id").split
    dev_runs = frame.loc[frame.group_id.map(split).eq("train"),
                         ["run_id", "gpu_model", "workload_type"]].drop_duplicates()
    validation_runs = []
    for _, cohort in dev_runs.groupby(["gpu_model", "workload_type"], sort=True):
        validation_runs.append(sorted(cohort.run_id)[0])
    training_runs = sorted(set(dev_runs.run_id)-set(validation_runs))
    if len(training_runs) != 18 or len(validation_runs) != 4:
        raise AssertionError("Unexpected development split")
    fit_pool = np.flatnonzero(frame.run_id.isin(training_runs))
    fit = sample_indices(frame, fit_pool)
    val = np.flatnonzero(frame.run_id.isin(validation_runs))
    counts = frame.iloc[fit].groupby("run_id").size()
    weights = frame.iloc[fit].run_id.map(1/counts).to_numpy()*len(counts)
    xfit = np.vstack([history_features(x) for x in frame.iloc[fit].history])
    xval = np.vstack([history_features(x) for x in frame.iloc[val].history])
    rows = []
    for direction in DIRECTIONS:
        truth = frame[direction+"_onset"].to_numpy(int)
        model = lgb.LGBMClassifier(objective="binary", n_estimators=100,
            num_leaves=15, learning_rate=.07, min_child_samples=30,
            random_state=42, n_jobs=1, verbosity=-1)
        model.fit(xfit, truth[fit], sample_weight=weights)
        part = frame.iloc[val][["run_id", "segment_id", "issue_s"]].copy()
        part["event"] = direction
        part["probability"] = model.predict_proba(xval)[:, 1]
        rows.append(part)
    val_pred = pd.concat(rows, ignore_index=True)
    val_raw = raw.loc[raw.run_id.isin(validation_runs)]
    val_grid = pd.DataFrame([score(val_pred, val_raw, t, d)
                             for d in DIRECTIONS for t in THRESHOLDS])
    val_grid.to_csv(out / "validation_threshold_grid.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for direction, part in val_grid.groupby("direction"):
        axes[0].plot(part.threshold, part.recall, marker="o", label=direction)
        axes[1].plot(part.threshold, part.false_alarms_per_hour, marker="o", label=direction)
    axes[0].set(xlabel="Probability threshold", ylabel="Onset recall", ylim=(0, 1))
    axes[1].set(xlabel="Probability threshold", ylabel="False alarms per hour")
    axes[1].axhline(100, color="0.3", linestyle="--", label="Scenario budget")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Inner validation alarm threshold trade-off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"validation_threshold_tradeoff.{ext}", dpi=250)
    plt.close(fig)
    choices = []
    for direction, candidates in val_grid.groupby("direction"):
        eligible = candidates.loc[candidates.false_alarms_per_hour <= 100]
        if len(eligible):
            chosen = eligible.sort_values(["recall", "false_alarms_per_hour", "threshold"],
                                          ascending=[False, True, False]).iloc[0]
            selection = "max recall under 100 false alarms/hour on inner validation"
        else:
            chosen = candidates.sort_values(["false_alarms_per_hour", "recall", "threshold"],
                                            ascending=[True, False, False]).iloc[0]
            selection = "minimum false alarms/hour; none met 100/h on inner validation"
        choices.append(dict(direction=direction, threshold=float(chosen.threshold),
                            validation_recall=float(chosen.recall),
                            validation_false_alarms_per_hour=float(chosen.false_alarms_per_hour),
                            rule=selection))
    selected = pd.DataFrame(choices)
    selected.to_csv(out / "selected_thresholds.csv", index=False)
    test_pred = pd.read_parquet(PREDICTION)
    test_pred = test_pred.loc[test_pred.condition.eq("history_only")].merge(
        raw[["run_id", "issue_s", "segment_id"]],
        on=["run_id", "issue_s"], validate="many_to_one")
    test_raw = raw.loc[raw.run_id.isin(test_pred.run_id.unique())]
    assessments = []
    for rec in selected.itertuples():
        for label, threshold in (("fixed_0.5", .5), ("inner_selected", rec.threshold)):
            assessments.append(dict(assessment=label,
                                    **score(test_pred, test_raw, threshold, rec.direction)))
    test_result = pd.DataFrame(assessments)
    test_result.to_csv(out / "test_threshold_comparison.csv", index=False)
    original = pd.read_csv(PREDICTION.parent / "onset_aggregate_metrics.csv")
    for direction in DIRECTIONS:
        reference = original.loc[(original.condition=="history_only") &
                                 (original.direction==direction)].iloc[0]
        actual = test_result.loc[(test_result.assessment=="fixed_0.5") &
                                 (test_result.direction==direction)].iloc[0]
        for column in ("n_onsets", "n_alarms", "n_matched", "n_false_alarms"):
            if int(reference[column]) != int(actual[column]):
                raise AssertionError(f"Baseline matcher mismatch: {direction} {column}")
    protocol = dict(threshold_grid=THRESHOLDS, validation_runs=validation_runs,
        training_runs=training_runs, test_runs=sorted(test_raw.run_id.unique()),
        false_alarm_budget_per_hour=100, classifier="same fixed history-only 100-tree configuration",
        final_test_probabilities="reused from train_onset_aligned.py; no test refit",
        assessment="reused outer holdout; development threshold selection only",
        warning="100 W onset and 100/h alarm budget are scenario values, not grid requirements")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    report = ["# Onset 경보 문턱값의 개발 run 내부 선택", "",
        "기존 22개 개발 run에서 GPU×workload별 1개씩 4개를 inner validation으로 고정하고, 나머지 18개로 history-only 분류기를 학습했다. 사전 고정한 확률 문턱값 0.3/0.5/0.7/0.8/0.9/0.95/0.98/0.99를 비교했다. 방향별 시간당 오경보 100건 이하에서 recall이 가장 큰 값을 선택했다. 만족하는 값이 없으면 오경보가 최소인 값을 택하도록 사전에 정했다. 10개 outer test run은 문턱값 선택에 사용하지 않았다.", "",
        selected.to_markdown(index=False,floatfmt=".3f"), "",
        "기존 22-run 학습 모델의 저장 test 확률에 고정 0.5와 선택된 문턱값을 적용했다. 분류기를 test 결과에 맞춰 재학습하지 않았다.", "",
        test_result.to_markdown(index=False,floatfmt=".3f"), "",
        "100 W onset 및 100회/시간 예산은 비교용 시나리오다. 원본 센서 업데이트/실제 계통 제어 요구치가 검증되지 않았으므로 운영 문턱값이 아니다. 같은 기존 holdout을 재사용했고, 집합일과 GPU×workload가 결합돼 있어 일반화 근거가 아니다.", "",
        "검증: 0.5 문턱값의 onsets/alarms/matches/false alarms가 기존 평가와 정확히 일치한다. 검증 문턱값 곡선은 `validation_threshold_tradeoff.png`에 있다. 재현: `python trace/grid_stability_forecast/tune_onset_threshold.py`.", ""]
    (out / "report_ko.md").write_text("\n".join(report))
    print(selected.to_string(index=False))
    print(test_result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=DEFAULT)
    run(parser.parse_args().out)
