"""
Unit tests for DEM spatial metadata loading, outline rasterization, and CRS alignment using Wurtenkees Glacier dataset.
"""

import unittest
import os
import numpy as np
from pysole.raster import load_dem, resample_dem, load_outline, check_crs_alignment


class TestRasterWuk(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.base_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.base_dir, "wuk_outline_clean.csv")

    def test_load_dem_wuk(self):
        grid, meta = load_dem(self.wuk_dem)
        self.assertEqual(grid.shape, (179, 213))
        self.assertEqual(meta["dx"], 5.0)
        self.assertEqual(meta["dy"], 5.0)
        self.assertFalse(np.isnan(grid).all())

    def test_resample_dem_wuk(self):
        grid, meta = load_dem(self.wuk_dem)
        resampled, bounds = resample_dem(
            grid=grid,
            native_dx=meta["dx"],
            native_dy=meta["dy"],
            target_dx=10.0,
            target_dy=10.0,
            bounds=meta["bounds"],
        )
        self.assertEqual(resampled.shape, (90, 106))
        self.assertIsNotNone(bounds)

    def test_load_outline_wuk(self):
        grid, meta = load_dem(self.wuk_dem)
        mask = load_outline(self.wuk_outline, grid, meta)
        self.assertEqual(mask.shape, grid.shape)
        self.assertTrue(mask.dtype == bool)
        self.assertGreater(np.sum(mask), 0)

    def test_crs_alignment_check(self):
        with self.assertRaises(ValueError):
            check_crs_alignment("EPSG:32632", "EPSG:4326")
        check_crs_alignment("EPSG:32632", "EPSG:32632")


if __name__ == "__main__":
    unittest.main()
