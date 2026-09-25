"""
Unit tests for variogram calculation and BSS surface slope optimization using Wurtenkees Glacier dataset.
"""

import unittest
import os
import numpy as np
from pysole.raster import load_dem, load_outline, GridGeometry
from pysole.variogram import calculate_variogram, fit_variogram_model, optimize_bss_variance, OptimizationResult


class TestVariogramWuk(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.base_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.base_dir, "wuk_outline_clean.csv")
        self.wuk_survey = os.path.join(self.base_dir, "wuk_survey_clean.csv")

    def test_variogram_calculation_and_fitting(self):
        survey = np.loadtxt(self.wuk_survey, delimiter=",", skiprows=1)
        coords = np.column_stack((survey[:, 0], survey[:, 1]))
        var_res = calculate_variogram(coords, survey[:, 3], nrbins=15)

        self.assertIn("distance", var_res)
        self.assertIn("val", var_res)

        a_range, sill, nugget, curve = fit_variogram_model(var_res["distance"], var_res["val"], model_type="spherical")
        self.assertGreater(a_range, 0)
        self.assertGreater(sill, 0)

    def test_optimize_bss_variance_wuk(self):
        dem, meta = load_dem(self.wuk_dem)
        outline_mask = load_outline(self.wuk_outline, dem, meta)
        geometry = GridGeometry.create(dem.shape, dx=meta["dx"], dy=meta["dy"], bounds=meta["bounds"])
        survey = np.loadtxt(self.wuk_survey, delimiter=",", skiprows=1)

        opt_res = optimize_bss_variance(
            dem=dem,
            survey_points=survey,
            geometry=geometry,
            kc_max=10.0,
            kc_min=0.01,
            d_kc=4.0,
            plots_dir=None,
            prefix="01_",
            stage_name="stage1",
            interactive=False,
        )

        self.assertIsInstance(opt_res, OptimizationResult)
        self.assertGreater(opt_res.optimal_kc, 0)
        self.assertEqual(opt_res.optimal_slope_grid.shape, dem.shape)
        self.assertGreater(len(opt_res.all_smoothed_slopes), 0)

    def test_n_cores_parallelization_wuk(self):
        dem, meta = load_dem(self.wuk_dem)
        geometry = GridGeometry.create(dem.shape, dx=meta["dx"], dy=meta["dy"], bounds=meta["bounds"])
        survey = np.loadtxt(self.wuk_survey, delimiter=",", skiprows=1)

        res_single = optimize_bss_variance(
            dem=dem, survey_points=survey, geometry=geometry, d_kc=4.0, n_cores=1
        )
        res_multi = optimize_bss_variance(
            dem=dem, survey_points=survey, geometry=geometry, d_kc=4.0, n_cores=-1
        )

        self.assertAlmostEqual(res_single.optimal_kc, res_multi.optimal_kc, places=6)
        np.testing.assert_allclose(res_single.optimal_slope_grid, res_multi.optimal_slope_grid)

    def test_variogram_override_warning(self):
        survey = np.loadtxt(self.wuk_survey, delimiter=",", skiprows=1)
        coords = np.column_stack((survey[:, 0], survey[:, 1]))
        # Small subset of 20 points -> N_pairs = 190 -> max_bins = 190//30 = 6 bins
        sub_coords = coords[:20]
        sub_vals = survey[:20, 3]

        # Requesting nrbins=50 should log warning about violating 30 pairs/bin threshold but honor override
        with self.assertLogs("pysole", level="WARNING") as cm:
            var_res = calculate_variogram(sub_coords, sub_vals, nrbins=50)

        self.assertTrue(any("violating recommended minimum threshold of 30 pairs/bin" in msg for msg in cm.output))
        self.assertIn("distance", var_res)


if __name__ == "__main__":
    unittest.main()
