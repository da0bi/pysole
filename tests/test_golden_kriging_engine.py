"""Golden regression: the native production kriging engine must reproduce its pre-refactor output.

Reference data: ``tests/data/golden_kriging_engine.npz`` (generated with ``tests/golden_fixture.py`` before the
engine refactor). Covers Ordinary + all 23 explicit drift models; kriging estimate and variance.
"""
import os
import unittest

import numpy as np

from pysole.interpolation import kriging_interpolation

from tests.golden_fixture import VARIOGRAM_PARAMS, build_fixture, cases

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "golden_kriging_engine.npz")


class TestGoldenKrigingEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = np.load(GOLDEN)
        cls.fx = build_fixture()

    def _run(self, method, terms):
        fx = self.fx
        return kriging_interpolation(
            fx["pts"], fx["geometry"], method=method, variogram_model="spherical",
            variogram_params=VARIOGRAM_PARAMS, dem_grid=fx["dem"], opt_slope_grid=fx["slope_rad"],
            drift_terms=terms, outline_mask=None, include_zero_boundary_condition=False,
            n_cores=-1, engine="native", show_progress=False, return_variance=True,
        )

    def test_case_list_is_complete(self):
        names = [c[0] for c in cases()]
        self.assertEqual(len(names), 24)
        self.assertEqual(len(set(names)), 24)
        for n in names:
            self.assertIn(f"{n}/z", self.gold.files)
            self.assertIn(f"{n}/var", self.gold.files)

    def test_all_cases_match_golden_output(self):
        for name, method, terms in cases():
            with self.subTest(case=name):
                res = self._run(method, terms)
                np.testing.assert_allclose(res.bedrock_grid, self.gold[f"{name}/z"], rtol=1e-8, atol=1e-8)
                np.testing.assert_allclose(res.variance_grid, self.gold[f"{name}/var"], rtol=1e-7, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
