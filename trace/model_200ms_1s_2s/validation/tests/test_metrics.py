import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import (assert_unique_and_paired, change_labels, error_values,
                     failure_episodes, interval_metrics, match_full_compact,
                     pinball_crps)


def test_error_sign_and_persistence_alignment():
    observed = np.array([100.0, 100.0])
    predicted = np.array([80.0, 125.0])
    np.testing.assert_array_equal(error_values(predicted, observed), [-20.0, 25.0])
    issue = np.array([90.0, 120.0])
    np.testing.assert_array_equal(error_values(issue, observed), [-10.0, 20.0])


def test_quantile_metrics_and_interval_score():
    y = np.array([0.0, 2.0])
    pred = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.5]])
    q = np.array([0.1, 0.5, 0.9])
    assert np.all(pinball_crps(y, pred, q) >= 0)
    got = interval_metrics(y, pred[:, 0], pred[:, 2])
    assert got["coverage"] == 0.5
    assert got["upper_violation_rate"] == 0.5
    assert got["interval_score_mean_w"] > 0


def test_episode_segmentation_gap_and_censoring():
    f = pd.DataFrame({"assessment": "a", "run_id": "r", "segment_key": "s",
                      "horizon_ms": 1000, "condition": "m", "representation": "c", "seed": 42,
                      "issue_s": [0.0, 0.2, 0.4, 1.0, 1.2],
                      "error_w": [600.0, 700.0, 0.0, -800.0, -900.0]})
    got = failure_episodes(f, 500.0)
    assert got.n_slots.tolist() == [2, 2]
    assert got.duration_s.tolist() == [0.4, 0.4]
    assert got.left_censored.tolist() == [True, True]
    assert got.right_censored.tolist() == [False, True]


def test_pairing_duplicates_and_change_boundaries():
    f = pd.DataFrame({"k": [1, 1, 2, 2], "condition": ["a", "b", "a", "b"]})
    assert assert_unique_and_paired(f, ["k"])["passed"]
    assert not assert_unique_and_paired(pd.concat([f, f.iloc[[0]]]), ["k"])["passed"]
    assert change_labels([-100, -99.9, 100]).tolist() == ["falling", "stable", "rising"]


def test_full_compact_matching():
    full = np.arange(30.0).reshape(2, 15)
    compact = full[:, [3, 7, 11]].copy()
    assert match_full_compact(full, compact)["passed"]
    compact[1, 1] += 0.01
    got = match_full_compact(full, compact)
    assert not got["passed"]
    assert np.isclose(got["max_abs_difference_w"], 0.01)
