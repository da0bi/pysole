"""
Unit tests for variogram calculation and BSS surface slope optimization using Wurtenkees Glacier dataset.
"""

import unittest
import os
import numpy as np
from pysole.raster import load_dem, load_outline, ensure_spatial_coords
from pysole.variogram import calculate_variogram, fit_variogram_model, optimize_bss_variance


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
        x_coords, y_coords, bounds = ensure_spatial_coords(dem.shape, dx=meta["dx"], dy=meta["dy"], bounds=meta["bounds"])
        survey = np.loadtxt(self.wuk_survey, delimiter=",", skiprows=1)

        opt_kc, opt_slope, min_var, all_slopes = optimize_bss_variance(
            dem=dem,
            survey_points=survey,
            dx=meta["dx"],
            dy=meta["dy"],
            x_coords=x_coords,
            y_coords=y_coords,
            bounds=bounds,
            kc_max=10.0,
            kc_min=0.01,
            d_kc=0.1,
            num_steps=3,
            plots_dir=None,
            prefix="01_",
            stage_name="stage1",
            interactive=False,
        )

        self.assertGreater(opt_kc, 0)
        self.assertEqual(opt_slope.shape, dem.shape)
        self.assertGreater(len(all_slopes), 0)


if __name__ == "__main__":
    unittest.main()
