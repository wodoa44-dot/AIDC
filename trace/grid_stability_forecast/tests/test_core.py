import unittest

import numpy as np
import pandas as pd

from core import crps_quantile, replay_filler, target_rows


class CoreTests(unittest.TestCase):
    def test_crps_degenerate_forecast_equals_absolute_error(self):
        q = np.array([.1, .5, .9])
        p = np.full((2, 3), 10.)
        np.testing.assert_allclose(crps_quantile([10, 13], p, q), [0, 3])

    def test_nonuniform_quantile_quadrature(self):
        self.assertAlmostEqual(crps_quantile([.5], [[0., 1.]], [.1, .9])[0], .09)
        with self.assertRaises(ValueError):
            crps_quantile([.5], [[1., 0.]], [.1, .9])

    def test_target_is_future_completed_bin_and_gap_resets(self):
        n = 34
        t = np.arange(1, n+1) * .2
        f = pd.DataFrame(dict(run_id="r", segment_id="s", is_valid=True, available_s=t,
                              bin_start_s=t-.2, power_w=np.arange(n, dtype=float)))
        rows = target_rows(f, horizons=(5,), history=25)
        self.assertAlmostEqual(rows.iloc[0].issue_s, 5.)
        self.assertAlmostEqual(rows.iloc[0].target_start_s, 5.8)
        self.assertAlmostEqual(rows.iloc[0].target_end_s, 6.)
        self.assertEqual(rows.iloc[0].delta_w, 5)
        f.loc[27, "available_s"] += .1
        self.assertTrue(target_rows(f, horizons=(5,), history=25).empty)

    def test_delay_is_causal_and_capacity_limited(self):
        power = [1000., 1000., 0., 0., 0.]
        grid, filler = replay_filler(power, power, capacity=150., delay_bins=1)
        self.assertEqual(filler[2], 0.)
        self.assertTrue(np.all(filler <= 150.))
        self.assertTrue(np.all(np.abs(np.diff(filler)) <= 40. + 1e-8))
        self.assertTrue(np.all(grid >= power))


if __name__ == "__main__":
    unittest.main()
