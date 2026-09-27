from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import lightgbm
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "trace/model_200ms_1s_2s"
sys.path.insert(0, str(MODEL))
from src.core import power_feature_names


def audit(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((MODEL / "configs/lite_20260920.json").read_text())
    resolved = json.loads((MODEL / "outputs/lite_full_20260920/effective_config.json").read_text())
    rows = []
    def add(item, requested, runtime, fitted, evidence, match=None):
        if match is None:
            match = requested == runtime == fitted
        rows.append(dict(item=item, requested=json.dumps(requested, default=str),
                         runtime_resolved=json.dumps(runtime, default=str), fitted=json.dumps(fitted, default=str),
                         evidence=str(evidence), match=match))
    for horizon in (1000, 2000):
        for condition in ("history_only", "history_meta"):
            path = MODEL / f"models/power-forecast-lite-20260920/full_h{horizon}_{condition}.joblib"
            if not path.exists():
                add(f"{horizon}/{condition}/artifact", "available", "unavailable", "unavailable", path, False)
                continue
            bundle = joblib.load(path)
            models = bundle["models"]
            params = bundle["parameters"]
            first = models[0].get_params()
            prefix = f"{horizon}/{condition}/"
            for key in ("n_estimators", "num_leaves", "learning_rate", "min_child_samples", "reg_alpha", "reg_lambda"):
                add(prefix + key, cfg.get(key, "candidate-selected"), resolved.get(key, "candidate-selected"),
                    first.get(key), path, first.get(key) in (cfg.get(key) if isinstance(cfg.get(key), list) else [cfg.get(key)]))
            for key in ("objective", "max_depth", "subsample", "subsample_freq", "colsample_bytree", "random_state", "n_jobs"):
                add(prefix + key, "code default", "src/core.py:fit_models", first.get(key), path, "verified")
            add(prefix + "library_version", "unavailable", lightgbm.__version__, lightgbm.__version__, path, "verified")
            add(prefix + "quantiles", cfg["quantiles"], resolved["quantiles"], bundle["quantiles"], path)
            add(prefix + "alpha_per_model", cfg["quantiles"], resolved["quantiles"],
                [m.get_params().get("alpha") for m in models], path)
            add(prefix + "actual_trees", "candidate-selected", "candidate-selected",
                [m.booster_.num_trees() for m in models], path, "verified")
            for key in ("history_ms", "resolution_ms"):
                add(prefix + key, cfg.get("histories_ms" if key == "history_ms" else key),
                    resolved.get("histories_ms" if key == "history_ms" else key), bundle.get(key), path, "verified")
            add(prefix + "feature_names", "src/core.py:power_feature_names", "src/core.py:fit_encoder",
                models[0].booster_.feature_name(), path, "verified")
            semantic_names = power_feature_names(bundle["resolution_ms"], bundle["history_ms"]) + bundle["encoder"]["names"]
            add(prefix + "semantic_feature_names", "code-derived", "code-derived", semantic_names,
                f"{path}; src/core.py:power_feature_names; encoder.names",
                len(semantic_names) == models[0].booster_.num_feature())
            add(prefix + "encoder", "training only", "src/core.py:fit_encoder", bundle["encoder"], path, "verified")
            add(prefix + "size_bytes", "unavailable", "unavailable", path.stat().st_size, path, "verified")
    latest_root = MODEL / "improvements/failure_aware_20260921_v1"
    latest_cfg = json.loads((latest_root / "config.json").read_text())
    for horizon in (1000, 2000):
        for condition in ("history_only", "history_meta"):
            path = latest_root / f"models/power-forecast-failure-aware-20260921-v1/full15_h{horizon}_{condition}.joblib"
            if not path.exists():
                add(f"latest/{horizon}/{condition}/artifact", "available", "unavailable", "unavailable", path, False)
                continue
            bundle = joblib.load(path)
            models = bundle["models"]
            first = models[0].get_params()
            prefix = f"latest/{horizon}/{condition}/"
            for key in ("objective", "n_estimators", "num_leaves", "max_depth", "learning_rate",
                        "min_child_samples", "reg_alpha", "reg_lambda", "colsample_bytree",
                        "subsample", "subsample_freq", "random_state", "n_jobs"):
                add(prefix + key, latest_cfg.get(key, "code/selection"), bundle["parameters"].get(key, "code default"),
                    first.get(key), path, "verified")
            add(prefix + "family", latest_cfg["families"], bundle.get("family"), bundle.get("family"), path, "verified")
            add(prefix + "quantiles", latest_cfg["quantiles"], bundle["quantiles"],
                [m.get_params().get("alpha") for m in models], path)
            add(prefix + "tree_counts", "selected", bundle["parameters"].get("n_estimators"),
                [m.booster_.num_trees() for m in models], path, "verified")
            add(prefix + "encoder", "training only", "src/core.py:fit_encoder", bundle["encoder"], path, "verified")
            semantic_names = power_feature_names(bundle["resolution_ms"], bundle["history_ms"]) + bundle["encoder"]["names"]
            add(prefix + "semantic_feature_names", "code-derived", "code-derived", semantic_names,
                f"{path}; src/core.py:power_feature_names; encoder.names",
                len(semantic_names) == models[0].booster_.num_feature())
            add(prefix + "history_ms", latest_cfg["histories_ms"], bundle["history_ms"], bundle["history_ms"], path, "verified")
            add(prefix + "resolution_ms", latest_cfg["resolution_ms"], bundle["resolution_ms"], bundle["resolution_ms"], path)
            add(prefix + "size_bytes", "unavailable", "unavailable", path.stat().st_size, path, "verified")
    for key in ("max_rows_per_run", "seeds", "outer_folds", "inner_folds", "workers", "model_threads"):
        add(key, cfg.get(key), resolved.get(key), "not stored in fitted model", MODEL / "train.py", cfg.get(key) == resolved.get(key))
    benchmark_path = MODEL / "outputs/lite_full_20260920/benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    for item in ("feature_construction_us", "prediction_only_six_models_us", "end_to_end_six_models_us",
                 "package_bytes", "maxrss_kb", "bin_collection_delay_ms"):
        add("historical_benchmark/" + item, "unavailable", "local warmed benchmark", benchmark[item],
            benchmark_path, "measured historical environment")
    for item, evidence in (("sampling_aggregation", "preprocessing/powerprep/forecast.py"),
                           ("target_start_end", "predictability_meta/corrected_experiment.py"),
                           ("forecast_issuance", "predictability_meta/corrected_experiment.py"),
                           ("timestamp_gap_handling", "predictability_meta/corrected_experiment.py"),
                           ("quantile_order_correction", "src/core.py:rearrange"),
                           ("inference_latency", "outputs/lite_full_20260920/benchmark.json"),
                           ("training_run_ids", "outputs/lite_full_20260920/development_runs.csv"),
                           ("split", "outputs/lite_full_20260920/corrected_folds.csv"),
                           ("sample_weights", "src/core.py:equal_run_weights")):
        add(item, "audit required", evidence, "artifact or code only; inspect referenced source", MODEL / evidence, "partial")
    pd.DataFrame(rows).to_csv(out / "current_model_audit.csv", index=False)
    (out / "environment.json").write_text(json.dumps({"python": platform.python_version(), "lightgbm": lightgbm.__version__,
        "pandas": pd.__version__, "argv": sys.argv}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent / "results/actual_20260926"))
    audit(parser.parse_args().out)
