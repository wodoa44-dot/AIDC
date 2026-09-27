from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def tables(out):
    out = Path(out)
    dest = out / "tables"
    dest.mkdir(exist_ok=True)
    pd.read_csv(HERE / "control_requirements.csv").to_csv(dest / "T1_methods.csv", index=False)
    inventory = pd.read_csv(out / "data_inventory.csv")
    inventory.groupby("dataset", as_index=False).agg(canonical_runs=("run_id", "nunique"),
        source_rows=("source_rows", "sum"), analysis_rows=("analysis_rows", "sum"),
        source_grid_median_s=("source_grid_s", "median"),
        analysis_grid_s=("analysis_grid_s", "first"),
        verified_sensor_cadence_runs=("physical_sensor_cadence_s", "count"),
        conditions=("model_name", "nunique")).to_csv(dest / "T2_datasets.csv", index=False)
    pd.read_csv(HERE / "observability_matrix.csv").to_csv(dest / "T2_observability.csv", index=False)
    pd.read_csv(out / "current_model_audit.csv").to_csv(dest / "T3_model_audit.csv", index=False)
    pd.read_csv(out / "selected_hyperparameters.csv").to_csv(dest / "T3_pilot_parameters.csv", index=False)
    agg = pd.read_csv(out / "aggregate_metrics.csv")
    gains = pd.read_csv(out / "paired_gain_ci.csv")
    agg.merge(gains, on=["horizon_ms", "condition"], how="left").to_csv(dest / "T4_forecast.csv", index=False)
    event_path = out / "event_metrics.csv"
    if event_path.exists():
        pd.read_csv(event_path).to_csv(dest / "T4_event_windows.csv", index=False)
    per = pd.read_csv(out / "per_run_metrics.csv")
    wide = per.pivot(index=["horizon_ms", "run_id"], columns="condition", values="crps_w").dropna()
    wide["gain_pct"] = 100 * (1 - wide.history_static_s2 / wide.history_only)
    wide["s1_gain_pct"] = 100 * (1 - wide.history_static_s1 / wide.history_only)
    wide.reset_index().to_csv(dest / "T5_metadata_by_run.csv", index=False)
    perm_path = out / "metadata_permutation.csv"
    if perm_path.exists():
        pd.read_csv(perm_path).to_csv(dest / "T5_run_permutation.csv", index=False)
    pd.DataFrame([dict(status="not_run", assessment="individual static feature ablation",
                       reason="pilot covered S1 and S2 bundles only; no causal attribution"),
                  dict(status="completed", assessment="whole-bundle run-level metadata permutation negative control",
                       reason="five training-run permutations; feature-family-specific permutations not run")]).to_csv(
                           dest / "T5_ablation_status.csv", index=False)
    path = out / "control_metrics.csv"
    if path.exists():
        pd.read_csv(path).to_csv(dest / "T6_control_idealized.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results/actual_20260926"))
    tables(parser.parse_args().out)
