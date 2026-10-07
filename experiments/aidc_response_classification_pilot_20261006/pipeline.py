"""Identity-checked checkpoints; unresolved label specification is a hard gate."""
import argparse
import datetime
import fcntl
import json
import os
import signal
import shutil
import sys
import time
import unittest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import lightgbm as lgb
import scipy
from common import ROOT, PROJECT, LEGACY, legacy, digest, atomic_json, parquet, KEY
from data import PREPARED, GROUPS, SELECTION, prepare, history_selection
from labeling import validate_spec, judge
from training import fit_predict, train_models
from evaluation import aggregate
from config import load_config
from reporting import report


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(specpath: Path) -> dict:
    """Hash canonical input/code/config paths independently of invocation cwd."""
    specpath = Path(specpath).resolve()
    paths = (
        [
            PREPARED,
            GROUPS,
            SELECTION,
            SELECTION.parent / "identity.json",
            LEGACY,
            PROJECT
            / "experiments/aidc_grid_metadata_20260929/src/aidc_exp/preprocessing.py",
            ROOT / "source_evidence.csv",
            specpath,
        ]
        + list(ROOT.glob("*.py"))
        + list((ROOT / "tests").glob("*.py"))
        + [ROOT / "configs/pilot.yaml"]
    )
    return {
        "files": {str(p): digest(p) for p in paths if p.exists()},
        "versions": {
            "lightgbm": lgb.__version__,
            "scipy": scipy.__version__,
            "python": sys.version,
        },
    }


def checkpoint(out, name, files):
    atomic_json(out / f"{name}_checkpoint.json", {f: digest(out / f) for f in files})


def verified_checkpoint(out, name):
    p = out / f"{name}_checkpoint.json"
    if not p.exists():
        return False
    saved = json.loads(p.read_text())
    if not all((out / f).exists() and digest(out / f) == h for f, h in saved.items()):
        raise RuntimeError(f"{name} checkpoint corruption")
    return True


def direction(path, current):
    diff = np.diff(np.r_[current, path[:5]])
    return (
        "mixed"
        if np.any(diff > 0) and np.any(diff < 0)
        else "rising"
        if np.any(diff > 0)
        else "falling"
        if np.any(diff < 0)
        else "flat"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/prepared_v1")
    ap.add_argument("--spec", type=Path, default=ROOT / "response_label_spec.yaml")
    ap.add_argument(
        "--mode",
        choices=["prepare", "preflight", "train", "report-only"],
        default="prepare",
    )
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / "pipeline.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Duplicate execution refused")
    start = time.monotonic()
    spec = load_config(a.spec)
    cfg = spec["experiment"]
    params = cfg["model_parameters"]
    ident = identity(a.spec)
    spec_hash = digest(a.spec)
    ready = validate_spec(spec)
    identfile = out / "identity.json"
    if identfile.exists():
        if json.loads(identfile.read_text()) != ident:
            raise RuntimeError(
                "Input/code/config/label identity changed: use a new output directory"
            )
        if a.mode != "report-only" and not a.resume:
            raise RuntimeError("Existing output requires --resume")
    else:
        if a.mode == "report-only":
            raise RuntimeError("report-only requires existing prepared output")
        atomic_json(identfile, ident)
    stage = "initialization"

    def check_stop():
        if (out / "stop_requested.json").exists():
            raise InterruptedError(f"stage={stage}: persisted stop request")

    def progress(state):
        if state not in ("interrupted", "failed"):
            check_stop()
        atomic_json(
            out / "progress.json",
            {
                "state": state,
                "stage": stage,
                "pid": os.getpid(),
                "updated_at": utc(),
                "elapsed_s": round(time.monotonic() - start, 2),
                "output": str(out),
            },
        )
        print(
            json.dumps(
                {
                    "stage": stage,
                    "state": state,
                    "at": utc(),
                    "elapsed_s": round(time.monotonic() - start, 2),
                    "output": str(out),
                }
            ),
            flush=True,
        )

    atomic_json(
        out / "process_identity.json",
        {
            "pid": os.getpid(),
            "start_time": utc(),
            "command_line": sys.argv,
            "stdin": "/dev/null for nohup launch",
            "mode": a.mode,
        },
    )

    def interrupt(signum, frame):
        raise InterruptedError(f"stage={stage} signal={signum}: stop requested")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        if a.mode == "report-only":
            for name in ["preparation", "labels"]:
                if not verified_checkpoint(out, name):
                    raise RuntimeError(f"stage=reporting: missing {name} checkpoint")
            if json.loads((out / "completion_status.json").read_text())[
                "all_completed"
            ]:
                aggregate(out)
            report(out)
            progress("report_completed")
            return 0
        (out / "response_label_spec.yaml").write_text(
            yaml.safe_dump(spec, allow_unicode=True, sort_keys=False)
        )
        shutil.copyfile(ROOT / "source_evidence.csv", out / "source_evidence.csv")
        history, provenance = history_selection(cfg)
        config = {
            "history_s": history,
            "history_provenance": provenance,
            "seed": cfg["seed"],
            "dt_s": spec["scenario"]["dt_s"],
            "features": {k: legacy.columns(history, k == "M2") for k in ["M1", "M2"]},
            "lightgbm_parameters": lgb.LGBMClassifier(**params).get_params(),
            "smoke_parameters": {**params, "n_estimators": cfg["smoke_trees"]},
            "fit_limits": {"main": 8, "smoke": 1},
            "class_prediction": "argmax",
            "unseen_rule": "training-only category map; -1 treated as missing",
            "class_weight": None,
            "sample_weight": "N/(R*nr) after U exclusion",
            "early_stopping": False,
            "threads": params["n_jobs"],
            "class_mapping": cfg["class_mapping"],
            "experiment": cfg,
            "label_spec_sha256": spec_hash,
            "resolved_versions": ident["versions"],
        }
        (out / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        )
        missing = []
        known = {
            "assessment.proxy_mapping": "GPU 8개 합산 W; 시설 PCC 실측 없음",
            "requirements.ramp_up_w_s": "과거 ramp_threshold_w_s=500 W/s, 기존 실험 가정",
            "requirements.ramp_down_w_s": "과거 ramp_threshold_w_s=500 W/s, 기존 실험 가정",
            "requirements.recovery_s": "과거 recovery_s=20 s, 기존 controller window 설계",
            "resources.U1": "과거 0.3×training P95 출력·0.02/2h·SoC 0.5/0.2/0.8·eta0.95·delay1bin",
            "resources.U2_additional": "과거 budget multiplier 후보는 자원 등급 정의가 아님",
        }
        for field in ready["missing_required_fields"] + ready["errors"]:
            missing.append(
                {
                    "field": field,
                    "why_required": "고정 요구조건·허용 자원·종료 조건이 없으면 최소 대응 수준을 계산할 수 없음",
                    "available_internal_value": known.get(
                        field, "이번 label에 적용할 확정값 없음"
                    ),
                    "additional_confirmation": "값·단위·출처·정확한 위치·적용 대상·한계와 이번 pilot 채택 여부",
                }
            )
        pd.DataFrame(
            missing,
            columns=[
                "field",
                "why_required",
                "available_internal_value",
                "additional_confirmation",
            ],
        ).to_csv(out / "missing_configuration.csv", index=False)
        from source_audit import audit_sources

        if not verified_checkpoint(out, "source_audit"):
            audit_sources(out)
            checkpoint(out, "source_audit", ["raw_source_validation.csv"])
        stage = "data_preparation"
        progress("running")
        future = (
            spec["scenario"]["future_bins"]
            + round(spec["requirements"]["recovery_s"] / spec["scenario"]["dt_s"])
            if ready["ready"]
            else spec["scenario"]["future_bins"]
        )
        if verified_checkpoint(out, "preparation"):
            train = pd.read_parquet(out / "train_anchors.parquet")
            ev = pd.read_parquet(out / "evaluation_anchors.parquet")
            inventory = pd.read_csv(out / "run_inventory.csv")
            splits = json.loads((out / "split_manifest.json").read_text())["folds"]
            mismatch = json.loads((out / "split_manifest.json").read_text())[
                "mismatches"
            ]
        else:
            train, ev, inventory, splits, mismatch = prepare(
                out, future, provisional=not ready["ready"], config=cfg
            )
            checkpoint(
                out,
                "preparation",
                [
                    "train_anchors.parquet",
                    "evaluation_anchors.parquet",
                    "continuous_features.parquet",
                    "run_inventory.csv",
                    "excluded_runs.csv",
                    "split_manifest.json",
                    "feature_schema.json",
                ],
            )
        union = (
            pd.concat([train, ev])
            .drop_duplicates(KEY)
            .sort_values(KEY)
            .reset_index(drop=True)
        )
        atomic_json(
            out / "preparation_validation.json",
            {
                "train_rows": len(train),
                "evaluation_rows": len(ev),
                "union_rows": len(union),
                "runs": inventory.run_id.nunique(),
                "date_mismatches": mismatch,
                "no_duplicate_keys": not union.duplicated(KEY).any(),
                "history_bins": 50,
                "future_bins": future,
                "feature_history_s": history,
                "provisional_anchors": not ready["ready"],
                "no_interpolation": True,
                "max_train_rows_per_run": int(train.groupby("run_id").size().max()),
                "max_eval_rows_per_run": int(ev.groupby("run_id").size().max()),
            },
        )
        stage = "label_configuration_validation"
        progress("running")
        if not verified_checkpoint(out, "labels"):
            results = []
            chunk_size = cfg["label_checkpoint_rows"]
            for chunk_start in range(0, len(union), chunk_size):
                chunk_name = f"label_chunk_{chunk_start:06d}"
                chunk_file = f"{chunk_name}.parquet"
                if verified_checkpoint(out, chunk_name):
                    results.extend(pd.read_parquet(out / chunk_file).to_dict("records"))
                    continue
                chunk_results = []
                for _, r in union.iloc[
                    chunk_start : chunk_start + chunk_size
                ].iterrows():
                    path = [r[f"future_w_{k}"] for k in range(1, future + 1)]
                    z = judge(path, r.current_w, spec, spec_hash)
                    chunk_results.append(
                        {
                            **r[
                                KEY
                                + [
                                    "sample_key",
                                    "collection_day",
                                    "issuance_s",
                                    "future_end_s",
                                ]
                            ].to_dict(),
                            **z,
                            "direction": direction(path, r.current_w),
                            "audit_only": not ready["ready"],
                        }
                    )
                parquet(out / chunk_file, pd.DataFrame(chunk_results))
                checkpoint(out, chunk_name, [chunk_file])
                results.extend(chunk_results)
                progress(f"labels {len(results)}/{len(union)}")
            labels = pd.DataFrame(results)
            parquet(out / "labels.parquet", labels)
            checkpoint(out, "labels", ["labels.parquet"])
        else:
            labels = pd.read_parquet(out / "labels.parquet")
        tests = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.discover(
                str(ROOT / "tests"), pattern="test_*.py"
            )
        )
        contradictions = labels.status.eq("nesting_contradiction").sum()
        validation = {
            **ready,
            "synthetic_judge_tests_passed": tests.wasSuccessful(),
            "synthetic_test_cases": tests.testsRun,
            "measured_label_validation_passed": ready["ready"]
            and contradictions == 0
            and not mismatch
            and tests.wasSuccessful(),
            "nesting_contradictions": int(contradictions),
            "R3_count": int(labels.label.eq("R3").sum()),
            "U_count": int(labels.label.eq("U").sum()),
            "U_fraction": float(labels.label.eq("U").mean()),
            "label_hash": spec_hash,
            "solver_options": spec["solver"],
            "rules_frozen_before_training": True,
            "class_boundary_tuning": False,
        }
        atomic_json(out / "label_validation.json", validation)
        blocked = not validation["measured_label_validation_passed"]
        status = {
            "all_completed": False,
            "blocked_stage": "label_configuration_validation" if blocked else None,
            "missing_required_fields": ready["missing_required_fields"],
            "configuration_errors": ready["errors"],
            "completed_stages": [
                "source_review",
                "data_preparation",
                "synthetic_unit_tests"
                if tests.wasSuccessful()
                else "synthetic_unit_tests_failed",
            ],
            "blocked_steps": [
                "measured_label_generation",
                "classifier_smoke",
                "nohup_training",
                "performance_evaluation",
            ]
            if blocked
            else [],
            "main_fits": 0,
            "smoke_fits": 0,
            "background_training_started": False,
            "label_spec_sha256": spec_hash,
            "finished_at": utc(),
        }
        atomic_json(out / "completion_status.json", status)
        if blocked:
            parquet(
                out / "predictions.parquet",
                pd.DataFrame(
                    columns=KEY
                    + [
                        "sample_key",
                        "scope",
                        "model",
                        "label",
                        "predicted_class",
                        "p0",
                        "p1",
                        "p2",
                        "p3",
                    ]
                ),
            )
            atomic_json(
                out / "evaluation_status.json",
                {"status": "not_run", "reason": "label_validation_failed"},
            )
            atomic_json(
                out / "artifact_availability.json",
                {
                    "labels.parquet": "U configuration audit only; no measured R labels",
                    "predictions.parquet": "empty: training blocked",
                    "per_run_metrics.csv": "empty: evaluation blocked",
                    "aggregate_metrics.csv": "empty: evaluation blocked",
                    "anchors": "provisional: recovery horizon unresolved",
                },
            )
            stage = "blocked"
            progress("blocked")
            report(out)
            return 2
        train = train.merge(
            labels[KEY + ["label", "direction"]], on=KEY, validate="one_to_one"
        )
        ev = ev.merge(
            labels[KEY + ["label", "direction"]], on=KEY, validate="one_to_one"
        )
        stage = "classifier_smoke"
        progress("running")
        if not verified_checkpoint(out, "smoke"):
            d = train[
                (train.collection_day != cfg["development_days"][0])
                & train.label.ne("U")
            ].head(cfg["smoke_rows"])
            v = ev[
                ev.collection_day.eq(cfg["development_days"][0]) & ev.label.ne("U")
            ].head(cfg["smoke_rows"])
            if d.empty or v.empty:
                raise RuntimeError("No determined smoke train/evaluation rows")
            prob, model, rec = fit_predict(d, v, history, False, smoke=True, config=cfg)
            assert (
                prob.shape == (len(v), 4)
                and np.isfinite(prob).all()
                and np.allclose(prob.sum(axis=1), 1)
            )
            atomic_json(
                out / "smoke.json",
                {
                    "passed": True,
                    "rows": len(d),
                    "trees": cfg["smoke_trees"],
                    "fit_count": int(model is not None),
                    "synthetic_training_rows": 0,
                    "record": rec,
                },
            )
            checkpoint(out, "smoke", ["smoke.json"])
        status["smoke_fits"] = json.loads((out / "smoke.json").read_text())["fit_count"]
        if a.mode in ["prepare", "preflight"]:
            status["blocked_stage"] = "nohup_training_not_started"
            status["completed_stages"] += [
                "measured_label_generation",
                "label_validation",
                "classifier_smoke",
            ]
            atomic_json(out / "completion_status.json", status)
            progress("preflight_passed")
            report(out)
            return 0
        stage = "training"
        progress("running")
        train_models(
            out, train, ev, splits, history, spec, spec_hash, stop_check=check_stop
        )
        stage = "aggregation"
        progress("running")
        aggregate(out)
        status.update(
            all_completed=False,
            blocked_stage=None,
            blocked_steps=[],
            main_fits=sum(
                x["fit_performed"]
                for x in json.loads((out / "fit_registry.json").read_text())
            ),
            background_training_started=True,
            completed_stages=status["completed_stages"]
            + [
                "measured_label_generation",
                "label_validation",
                "classifier_smoke",
                "training",
                "aggregation",
                "visualization",
                "report",
            ],
            finished_at=utc(),
        )
        atomic_json(out / "completion_status.json", status)
        stage = "reporting"
        progress("running")
        report(out, results_ready=True)
        status["all_completed"] = True
        atomic_json(out / "completion_status.json", status)
        progress("completed")
        return 0
    except (InterruptedError, KeyboardInterrupt) as exc:
        atomic_json(
            out / "completion_status.json",
            {
                "all_completed": False,
                "state": "interrupted",
                "blocked_stage": stage,
                "error": str(exc),
                "finished_at": utc(),
            },
        )
        progress("interrupted")
        return 130
    except Exception as exc:
        atomic_json(
            out / "completion_status.json",
            {
                "all_completed": False,
                "blocked_stage": stage,
                "missing_required_fields": ready["missing_required_fields"],
                "error": f"stage={stage} run_id={out.name}: {exc}",
                "finished_at": utc(),
            },
        )
        progress("failed")
        raise
    finally:
        atomic_json(
            out
            / (
                "report_only_timing.json"
                if a.mode == "report-only"
                else "execution_timing.json"
            ),
            {"wall_clock_s": time.monotonic() - start, "mode": a.mode},
        )


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:
        code = 1
        failed_out = (
            Path(sys.argv[sys.argv.index("--output") + 1])
            if "--output" in sys.argv
            else ROOT / "outputs/prepared_v1"
        )
        if failed_out.exists():
            state_file = failed_out / "completion_status.json"
            saved = json.loads(state_file.read_text()) if state_file.exists() else {}
            if not saved.get("error"):
                atomic_json(
                    state_file,
                    {
                        "all_completed": False,
                        "blocked_stage": "initialization",
                        "error": f"stage=initialization run_id={failed_out.name}: {exc}",
                        "finished_at": utc(),
                    },
                )
        import traceback

        traceback.print_exc()
    atomic_json(
        Path(sys.argv[sys.argv.index("--output") + 1]) / "exit_status.json"
        if "--output" in sys.argv
        else ROOT / "outputs/prepared_v1/exit_status.json",
        {
            "exit_code": code,
            "finished_at": utc(),
            "exit_code_2_meaning": "required label configuration unresolved; no training",
        },
    )
    sys.exit(code)
