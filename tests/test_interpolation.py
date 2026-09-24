"""
Unit tests for Kriging interpolation approaches using Wurtenkees Glacier dataset.
"""

import unittest
import os
import numpy as np
from pysole.raster import load_dem, load_outline, GridGeometry
from pysole.interpolation import kriging_interpolation, KrigingResult


class TestInterpolationWuk(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.base_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.base_dir, "wuk_outline_clean.csv")
        self.wuk_survey = os.path.join(self.base_dir, "wuk_survey_clean.csv")

        self.dem, self.meta = load_dem(self.wuk_dem)
        self.outline_mask = load_outline(self.wuk_outline, self.dem, self.meta)
        self.geometry = GridGeometry.create(
            self.dem.shape, dx=self.meta["dx"], dy=self.meta["dy"], bounds=self.meta["bounds"]
        )
        self.survey = np.loadtxt(self.wuk_survey, delimiter=",", skiprows=1)
        # Sample points [X, Y, Value]
        self.sample_pts = np.column_stack((self.survey[:, 0], self.survey[:, 1], self.survey[:, 3]))

    def test_universal_kriging_wuk(self):
        krig_res = kriging_interpolation(
            sample_points=self.sample_pts,
            geometry=self.geometry,
            method="universal",
            variogram_model="spherical",
            dem_grid=self.dem,
            outline_mask=self.outline_mask,
        )
        self.assertIsInstance(krig_res, KrigingResult)
        z, var = krig_res.bedrock_grid, krig_res.variance_grid
        self.assertEqual(z.shape, self.dem.shape)
        self.assertEqual(var.shape, self.dem.shape)
        self.assertFalse(np.isnan(z[self.outline_mask]).any())

    def test_ordinary_kriging_wuk(self):
        krig_res = kriging_interpolation(
            sample_points=self.sample_pts,
            geometry=self.geometry,
            method="ordinary",
            variogram_model="spherical",
            outline_mask=self.outline_mask,
        )
        self.assertIsInstance(krig_res, KrigingResult)
        z, var = krig_res.bedrock_grid, krig_res.variance_grid
        self.assertEqual(z.shape, self.dem.shape)
        self.assertEqual(var.shape, self.dem.shape)
        self.assertFalse(np.isnan(z[self.outline_mask]).any())

    def test_zero_boundary_condition_wuk(self):
        krig_res = kriging_interpolation(
            sample_points=self.sample_pts,
            geometry=self.geometry,
            method="universal",
            variogram_model="spherical",
            dem_grid=self.dem,
            outline_mask=self.outline_mask,
            include_zero_boundary_condition=True,
        )
        self.assertIsInstance(krig_res, KrigingResult)
        z = krig_res.bedrock_grid
        self.assertEqual(z.shape, self.dem.shape)
    def test_random_forest_hole_filling_n_cores(self):
        from pysole.interpolation import random_forest_hole_filling

        bedrock_mock = self.dem.copy() - 50.0
        # Create a hole where thickness <= 0.1 m inside boundary mask
        bedrock_mock[self.outline_mask] = self.dem[self.outline_mask] - 50.0
        hole_indices = np.where(self.outline_mask)
        if len(hole_indices[0]) > 20:
            sub_r = hole_indices[0][:10]
            sub_c = hole_indices[1][:10]
            bedrock_mock[sub_r, sub_c] = self.dem[sub_r, sub_c]

        filled = random_forest_hole_filling(
            dem=self.dem,
            bedrock_grid=bedrock_mock,
            boundary_mask=self.outline_mask,
            geometry=self.geometry,
            n_cores=2,
        )
        self.assertEqual(filled.shape, self.dem.shape)
        self.assertFalse(np.isnan(filled).any())

    def test_built_in_kriging_flag(self):
        with self.assertLogs("pysole", level="INFO") as cm:
            krig_res = kriging_interpolation(
                sample_points=self.sample_pts,
                geometry=self.geometry,
                method="universal",
                variogram_model="spherical",
                dem_grid=self.dem,
                outline_mask=self.outline_mask,
                built_in_kriging=False,
            )
        self.assertTrue(any("Built-in native Kriging engine active" in log_msg for log_msg in cm.output))
        self.assertIsInstance(krig_res, KrigingResult)
        self.assertEqual(krig_res.bedrock_grid.shape, self.dem.shape)


if __name__ == "__main__":
    unittest.main()

