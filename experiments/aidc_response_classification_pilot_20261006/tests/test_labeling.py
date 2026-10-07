import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
from common import ROOT, legacy
from labeling import (
    judge,
    validate_spec,
    feasibility,
    physical_check,
    calculate_base_power_w,
)
from training import assert_split
from config import load_config
from data import continuous
from evaluation import evaluate
from training import fit_predict


def synthetic_spec():
    # Test fixture only. These values never enter measured labels or training.
    s = yaml.safe_load((ROOT / "response_label_spec.yaml").read_text())
    s["assessment"]["proxy_mapping"] = "identity_GPU_sum_W"
    s["requirements"].update(
        ramp_up_w_s=0,
        ramp_down_w_s=0,
        source="synthetic_fixture_only",
        source_location="test_pilot.py",
        recovery_s=0,
        category="synthetic_test",
        recovery_ramp_enforced=True,
        envelope_required=False,
        tolerance_w=1e-6,
        tolerance_j=1e-6,
        tolerance_fraction=1e-8,
    )
    d = dict(
        id="basic",
        charge_w=100,
        discharge_w=100,
        capacity_j=10000,
        initial_soc=0.5,
        soc_min=0.1,
        soc_max=0.9,
        eta_charge=1,
        eta_discharge=1,
        delay_bins=0,
        slew_w_s=10000,
        throughput_budget_j=10000,
        terminal_soc_tolerance=0,
        source="synthetic_fixture_only",
        source_location="test_pilot.py",
        category="synthetic_test",
    )
    s["resources"].update(
        U1=[d],
        U2_additional=[{**d, "id": "additional"}],
        expansion_cost_or_workload_constraints="synthetic_no_workload_action",
    )
    s["scenario"].update(
        grid_conditions="synthetic_proxy", initial_pcc_rule="current_completed_bin_W"
    )
    return s


class LabelTests(unittest.TestCase):
    def test_unresolved_is_U(self):
        s = yaml.safe_load((ROOT / "response_label_spec.yaml").read_text())
        self.assertFalse(validate_spec(s)["ready"])
        self.assertEqual(judge([1000] * 5, 1000, s, "test")["label"], "U")

    def test_four_levels_and_nesting(self):
        s = synthetic_spec()
        self.assertTrue(validate_spec(s)["ready"])
        for amp, expected in [(0, "R0"), (100, "R1"), (200, "R2"), (400, "R3")]:
            result = judge(
                [1000 + amp, 1000 - amp, 1000, 1000, 1000], 1000, s, "fixture"
            )
            self.assertEqual(result["label"], expected, result)
            self.assertFalse(result["F0"] is True and result["F1"] is False)
            self.assertFalse(result["F1"] is True and result["F2"] is False)
        # Same endpoint, nonzero intermediate response requirement.
        self.assertNotEqual(
            judge([1100, 900, 1000, 1000, 1000], 1000, s, "fixture")["label"], "R0"
        )

    def test_requirement_tolerance_nesting(self):
        s = synthetic_spec()
        s["requirements"]["ramp_up_w_s"] = 500
        s["requirements"]["ramp_down_w_s"] = 500
        s["requirements"]["tolerance_w"] = 0.001
        z = judge([1100.0005, 1000, 1000, 1000, 1000], 1000, s, "fixture")
        self.assertEqual(z["label"], "R0")
        self.assertTrue(z["F1"])
        self.assertTrue(z["F2"])

    def test_soc_terminal_budget_and_delay_rechecked(self):
        s = synthetic_spec()
        s["resources"]["U1"][0]["delay_bins"] = 1
        z = judge([1100, 900, 1000, 1000, 1000], 1000, s, "fixture")
        self.assertEqual(z["label"], "R2")
        s = synthetic_spec()
        s["resources"]["U1"][0]["throughput_budget_j"] = 0
        z = judge([1100, 900, 1000, 1000, 1000], 1000, s, "fixture")
        self.assertEqual(z["label"], "R2")
        s = synthetic_spec()
        z = judge([1100, 1000, 1000, 1000, 1000], 1000, s, "fixture")
        self.assertEqual(z["label"], "R3")

    def test_missing_future_and_solver_timeout(self):
        s = synthetic_spec()
        self.assertEqual(judge([1000], 1000, s, "fixture")["label"], "U")
        with patch(
            "labeling.milp",
            return_value=SimpleNamespace(status=1, x=None, message="time limit"),
        ):
            self.assertIsNone(
                feasibility(
                    [1100, 900, 1000, 1000, 1000], 1000, s["resources"]["U1"], s
                )["F"]
            )

    def test_failed_physical_check_is_U(self):
        s = synthetic_spec()
        with patch(
            "labeling.milp",
            return_value=SimpleNamespace(status=0, x=np.zeros(15), message="optimal"),
        ):
            result = feasibility(
                [1100, 900, 1000, 1000, 1000], 1000, s["resources"]["U1"], s
            )
            self.assertIsNone(result["F"])
            self.assertEqual(result["status"], "physical_recheck_failed")

    def test_future_does_not_enter_features(self):
        g = pd.DataFrame(
            {
                "run_id": "test",
                "segment": 0,
                "bin": np.arange(100),
                "issuance_s": (np.arange(100) + 1) * 0.2,
                "current_w": np.arange(100, dtype=float),
            }
        )
        a = continuous(g, 5)
        g.loc[60:, "current_w"] = 9999
        b = continuous(g, 5)
        pd.testing.assert_frame_equal(
            a.loc[:59, legacy.columns(5)], b.loc[:59, legacy.columns(5)]
        )
        self.assertEqual(a.loc[49, "future_w_5"], 54)
        self.assertTrue(a.loc[49, "valid_anchor"])
        self.assertFalse(a.loc[48, "valid_anchor"])

    def test_gap_and_missing_break_eligibility(self):
        g = pd.DataFrame(
            {
                "run_id": "test",
                "segment": 0,
                "bin": np.arange(100),
                "issuance_s": (np.arange(100) + 1) * 0.2,
                "current_w": 1.0,
            }
        )
        g.loc[50, "current_w"] = np.nan
        d = continuous(g, 5)
        self.assertFalse(d.loc[49, "valid_anchor"])
        self.assertFalse(d.loc[60, "valid_anchor"])
        g = g.drop(50)
        d = continuous(g, 5)
        self.assertEqual(d.valid_anchor.sum(), 0)

    def test_run_balanced_metrics_and_absent_recall(self):
        g = pd.DataFrame(
            {
                "run_id": ["a", "a", "a", "b"],
                "label": ["R0", "R0", "R0", "R2"],
                "predicted_class": [0, 0, 0, 0],
                "p0": 1.0,
                "p1": 0.0,
                "p2": 0.0,
                "p3": 0.0,
            }
        )
        pooled, _, _ = evaluate(g, False)
        balanced, rows, cm = evaluate(g, True)
        self.assertEqual(pooled["ordinal_mae"], 0.5)
        self.assertEqual(balanced["ordinal_mae"], 1)
        self.assertIsNone(rows[3]["recall"])
        self.assertAlmostEqual(cm.sum(), 1)

    def test_missing_class_coordinates_without_training(self):
        # A mocked API contract check: no actual classifier fit or performance rows.
        train = pd.DataFrame({c: [1.0, 2.0] for c in legacy.columns(5)})
        train["run_id"] = ["a", "b"]
        train["label"] = ["R0", "R2"]
        dummy = SimpleNamespace(
            booster_=SimpleNamespace(
                predict=lambda x: np.array([[0.2, 0.8], [0.7, 0.3]])
            ),
            fit=lambda *a, **k: None,
            get_params=lambda: {"num_class": 2},
        )
        with patch("training.lgb.LGBMClassifier", return_value=dummy) as constructor:
            prob, _, rec = fit_predict(train, train, 5, False)
        self.assertEqual(constructor.call_args.kwargs["num_class"], 2)
        self.assertEqual(rec["missing_classes"], [1, 3])
        np.testing.assert_array_equal(prob[:, [1, 3]], 0.0)
        np.testing.assert_allclose(prob[:, [0, 2]], [[0.2, 0.8], [0.7, 0.3]])


class CoreContractTests(unittest.TestCase):
    def test_watt_seconds_to_joules_and_base(self):
        spec = synthetic_spec()
        device = spec["resources"]["U1"][0]
        device["throughput_budget_j"] = 19.0
        charge = np.array([[100.0, 0.0, 0.0, 0.0, 0.0]])
        discharge = np.zeros_like(charge)
        check = physical_check(
            [900.0, 1000.0, 1000.0, 1000.0, 1000.0],
            1000.0,
            [device],
            charge,
            discharge,
            spec,
        )
        self.assertAlmostEqual(check["max_violation_j"], 1.0)  # 100 W × 0.2 s = 20 J.
        self.assertEqual(calculate_base_power_w(1234.0), 1234.0)
        with self.assertRaises(ValueError):
            calculate_base_power_w(float("nan"))
        self.assertEqual(
            judge([1000.0, np.nan, 1000.0, 1000.0, 1000.0], 1000.0, spec, "test")[
                "status"
            ],
            "data_error",
        )

    def test_fold_disjointness(self):
        train = pd.DataFrame(
            {"run_id": ["a"], "collection_day": ["day_a"], "sample_key": ["a:1"]}
        )
        ev = pd.DataFrame(
            {"run_id": ["b"], "collection_day": ["day_b"], "sample_key": ["b:1"]}
        )
        assert_split(train, ev, 0)
        for column in train:
            bad = ev.copy()
            bad[column] = train[column]
            with self.assertRaisesRegex(ValueError, "fold=0"):
                assert_split(train, bad, 0)

    def test_training_only_category_map(self):
        train = pd.DataFrame({"hardware": ["known"], "current_w": [1.0]})
        ev = pd.DataFrame({"hardware": ["future_only"], "current_w": [2.0]})
        _, encoded, mapping, _ = legacy.encode(train, ev, ["hardware", "current_w"])
        self.assertNotIn("future_only", mapping["hardware"])
        self.assertEqual(encoded.hardware.iloc[0], -1.0)

    def test_module_imports_do_not_create_outputs_or_fit(self):
        import os
        import subprocess
        import tempfile

        modules = [
            "config",
            "data",
            "features",
            "labeling",
            "training",
            "evaluation",
            "plotting",
            "reporting",
            "pipeline",
            "manage",
            "labels",
            "prepare",
            "modeling",
        ]
        with tempfile.TemporaryDirectory() as directory:
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
                "MPLCONFIGDIR": "/tmp/aidc_quality_matplotlib_cache",
            }
            code = (
                "from unittest.mock import patch; import lightgbm; "
                + "\nwith patch.object(lightgbm.LGBMClassifier, 'fit', side_effect=AssertionError('import fit')):\n"
                + "\n".join(
                    "    __import__(" + repr(module) + ")" for module in modules
                )
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_identity_uses_canonical_spec_path(self):
        from pipeline import identity
        import os

        spec = ROOT / "configs/pilot.yaml"
        self.assertEqual(identity(spec), identity(Path(os.path.relpath(spec))))

    def test_pilot_config_has_fixed_assumptions(self):
        spec = load_config()
        self.assertTrue(validate_spec(spec)["ready"])
        self.assertEqual(spec["status"], "hypothetical_engineering_pilot")
        self.assertNotIn(
            spec["experiment"]["holdout_day"], spec["experiment"]["development_days"]
        )


if __name__ == "__main__":
    unittest.main()
