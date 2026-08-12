"""
Unit tests for surface gradient computation, FFT Gaussian smoothing, and margin blending using Wurtenkees Glacier dataset.
"""

import unittest
import os
import numpy as np
from pysole.raster import load_dem, load_outline, ensure_spatial_coords
from pysole.smoothing import compute_gradients, fft_gaussian_smooth
from pysole.solver import blend_margin_topography


class TestSmoothingWuk(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.base_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.base_dir, "wuk_outline_clean.csv")

        self.dem, self.meta = load_dem(self.wuk_dem)
        self.outline_mask = load_outline(self.wuk_outline, self.dem, self.meta)
        self.x_coords, self.y_coords, self.bounds = ensure_spatial_coords(
            self.dem.shape, dx=self.meta["dx"], dy=self.meta["dy"], bounds=self.meta["bounds"]
        )

    def test_compute_gradients_wuk(self):
        grads = compute_gradients(self.dem, dx=self.meta["dx"], dy=self.meta["dy"])
        self.assertIn("slope_x", grads)
        self.assertIn("slope_y", grads)
        self.assertIn("slope_rad", grads)
        self.assertEqual(grads["slope_rad"].shape, self.dem.shape)
        self.assertTrue(np.all(grads["slope_rad"] >= 0))

    def test_fft_gaussian_smooth_wuk(self):
        smoothed, k_grid, k_max = fft_gaussian_smooth(self.dem, dx=self.meta["dx"], dy=self.meta["dy"], kc=0.1)
        self.assertEqual(smoothed.shape, self.dem.shape)
        self.assertFalse(np.isnan(smoothed).any())

    def test_blend_margin_topography_wuk(self):
        thick = np.where(self.outline_mask, 50.0, 0.0)
        bedrock_in = self.dem - thick

        blended = blend_margin_topography(
            dem=self.dem,
            bedrock_input=bedrock_in,
            boundary_mask=self.outline_mask,
            dx=self.meta["dx"],
            dy=self.meta["dy"],
            x_coords=self.x_coords,
            y_coords=self.y_coords,
            min_gap_dist=30.0,
        )

        self.assertEqual(blended.shape, self.dem.shape)
        self.assertTrue(np.all(np.isfinite(blended)))


if __name__ == "__main__":
    unittest.main()
