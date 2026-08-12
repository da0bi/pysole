"""
Unit tests for Kriging interpolation approaches using Wurtenkees Glacier dataset.
"""

import unittest
import os
import numpy as np
from pysole.raster import load_dem, load_outline, ensure_spatial_coords
from pysole.interpolation import kriging_interpolation


class TestInterpolationWuk(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.base_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.base_dir, "wuk_outline_clean.csv")
        self.wuk_survey = os.path.join(self.base_dir, "wuk_survey_clean.csv")

        self.dem, self.meta = load_dem(self.wuk_dem)
        self.outline_mask = load_outline(self.wuk_outline, self.dem, self.meta)
        self.x_coords, self.y_coords, self.bounds = ensure_spatial_coords(
            self.dem.shape, dx=self.meta["dx"], dy=self.meta["dy"], bounds=self.meta["bounds"]
        )
        self.survey = np.loadtxt(self.wuk_survey, delimiter=",", skiprows=1)
        # Sample points [X, Y, Value]
        self.sample_pts = np.column_stack((self.survey[:, 0], self.survey[:, 1], self.survey[:, 3]))

    def test_universal_kriging_wuk(self):
        z, var = kriging_interpolation(
            sample_points=self.sample_pts,
            target_grid_shape=self.dem.shape,
            bounds=self.bounds,
            method="universal",
            variogram_model="spherical",
            dem_grid=self.dem,
            outline_mask=self.outline_mask,
        )
        self.assertEqual(z.shape, self.dem.shape)
        self.assertEqual(var.shape, self.dem.shape)
        self.assertFalse(np.isnan(z[self.outline_mask]).any())

    def test_ordinary_kriging_wuk(self):
        z, var = kriging_interpolation(
            sample_points=self.sample_pts,
            target_grid_shape=self.dem.shape,
            bounds=self.bounds,
            method="ordinary",
            variogram_model="spherical",
            outline_mask=self.outline_mask,
        )
        self.assertEqual(z.shape, self.dem.shape)
        self.assertEqual(var.shape, self.dem.shape)
        self.assertFalse(np.isnan(z[self.outline_mask]).any())

    def test_zero_boundary_condition_wuk(self):
        z, var = kriging_interpolation(
            sample_points=self.sample_pts,
            target_grid_shape=self.dem.shape,
            bounds=self.bounds,
            method="universal",
            variogram_model="spherical",
            dem_grid=self.dem,
            outline_mask=self.outline_mask,
            include_zero_boundary_condition=True,
        )
        self.assertEqual(z.shape, self.dem.shape)
        self.assertFalse(np.isnan(z[self.outline_mask]).any())


if __name__ == "__main__":
    unittest.main()
