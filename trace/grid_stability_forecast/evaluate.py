from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import write_json

HERE = Path(__file__).resolve().parent


def evaluate(out):
    out = Path(out)
    df = pd.read_parquet(out / "per_target_predictions.parquet")
    qcols = sorted((c for c in df if c.startswith("q_") and c[2:].replace(".", "", 1).isdigit()),
                   key=lambda c: float(c[2:]))
    for q in qcols:
        df[f"coverage_{q[2:]}"] = (df.y <= df[q]).astype(float)
    df["covered_80"] = ((df.y >= df["q_0.1"]) & (df.y <= df["q_0.9"])).astype(float)
    df["width_80_w"] = df["q_0.9"] - df["q_0.1"]
    df["bias_w"] = df.q50 - df.y
    df["squared_error"] = df.bias_w ** 2
    cols = ["condition", "horizon_ms", "run_id", "seed", "fold"]
    per = df.groupby(cols).agg(n_targets=("y", "size"), crps_w=("crps_w", "mean"),
        mae_w=("absolute_error_w", "mean"), mse_w2=("squared_error", "mean"),
        bias_w=("bias_w", "mean"), coverage_80=("covered_80", "mean"),
        width_80_w=("width_80_w", "mean"), p95_w=("absolute_error_w", lambda x: x.quantile(.95)),
        p99_w=("absolute_error_w", lambda x: x.quantile(.99))).reset_index()
    per["rmse_w"] = np.sqrt(per.mse_w2)
    per.to_csv(out / "per_run_metrics.csv", index=False)
    agg = per.groupby(["condition", "horizon_ms"]).agg(n_runs=("run_id", "nunique"),
        crps_w=("crps_w", "mean"), mae_w=("mae_w", "mean"), rmse_w=("rmse_w", "mean"),
        bias_w=("bias_w", "mean"), coverage_80=("coverage_80", "mean"),
        width_80_w=("width_80_w", "mean"), p95_w=("p95_w", "mean"), p99_w=("p99_w", "mean")).reset_index()
    agg.to_csv(out / "aggregate_metrics.csv", index=False)
    trans = df.groupby(["condition", "horizon_ms", "transition", "run_id"]).agg(
        n_targets=("y", "size"), crps_w=("crps_w", "mean"), mae_w=("absolute_error_w", "mean"),
        p95_w=("absolute_error_w", lambda x: x.quantile(.95)),
        p99_w=("absolute_error_w", lambda x: x.quantile(.99))).reset_index()
    trans.to_csv(out / "transition_metrics.csv", index=False)
    tail = df.groupby(["condition", "horizon_ms", "run_id"]).agg(
        error_p90_w=("absolute_error_w", lambda x: x.quantile(.9)),
        error_p95_w=("absolute_error_w", lambda x: x.quantile(.95)),
        error_p99_w=("absolute_error_w", lambda x: x.quantile(.99)),
        max_error_w=("absolute_error_w", "max"),
        worst_1pct_mean_w=("absolute_error_w", lambda x: x.nlargest(max(1, int(np.ceil(len(x)*.01)))).mean())).reset_index()
    tail.to_csv(out / "tail_metrics.csv", index=False)
    calibration = pd.concat([df.groupby(["condition", "horizon_ms", "run_id"])[f"coverage_{q[2:]}"].mean().reset_index(name="empirical_coverage").assign(nominal=float(q[2:])) for q in qcols])
    calibration.to_csv(out / "calibration_metrics.csv", index=False)
    pivot = per.pivot(index=["horizon_ms", "run_id"], columns="condition", values="crps_w").dropna()
    rng = np.random.default_rng(2026)
    baseline = pivot.history_only.to_numpy()
    paired_rows = []
    for condition in ("history_static_s1", "history_static_s2"):
        meta = pivot[condition].to_numpy()
        gains = 100 * (1 - meta / baseline)
        draws = rng.integers(0, len(gains), (2000, len(gains)))
        boot = 100 * (1 - meta[draws].mean(axis=1) / baseline[draws].mean(axis=1))
        paired_rows.append(dict(condition=condition, horizon_ms=int(pivot.index.get_level_values("horizon_ms")[0]), n_runs=len(gains),
            gain_of_run_balanced_scores_pct=100*(1-meta.mean()/baseline.mean()),
            mean_run_gain_pct=gains.mean(), ci_low_pct=np.quantile(boot, .025),
            ci_high_pct=np.quantile(boot, .975), bootstrap_replicates=2000,
            interpretation="stored-prediction cluster CI; excludes model selection and retraining uncertainty"))
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired_gain_ci.csv", index=False)
    write_json(out / "verification.json", {"same_test_rows": bool(df.groupby(["run_id", "issue_s", "horizon_ms"]).condition.nunique().eq(3).all()),
        "aggregate_matches_per_run": bool(np.isclose(agg.loc[agg.condition=="history_only", "crps_w"].iloc[0], per.loc[per.condition=="history_only", "crps_w"].mean())),
        "n_independent_test_runs": len(baseline), "score_grid": qcols})
    return agg, paired


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/actual_20260926"))
    evaluate(parser.parse_args().out)
