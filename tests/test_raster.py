"""
Unit tests for DEM spatial metadata loading, outline rasterization, and CRS alignment using Wurtenkees Glacier dataset.
"""

import unittest
import os
import numpy as np
from pysole.raster import load_dem, resample_dem, load_outline, check_crs_alignment, load_survey_points, BedrockMap


class TestRasterWuk(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.data_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.data_dir, "wuk_outline_clean.csv")

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

    def test_load_survey_points_deduplication(self):
        # Input with metric coordinates rounding to same 1 decimal place (1000.0, 2000.0)
        raw_pts = np.array([
            [1000.01, 2000.04, 100.0, 50.0],
            [1000.04, 2000.02, 100.0, 60.0],  # Duplicate at 0.1m resolution
            [1005.00, 2005.00, 105.0, 70.0],
        ])
        consolidated = load_survey_points(raw_pts)
        self.assertEqual(len(consolidated), 2)
        # Check that value for (1000.0, 2000.0) is averaged (55.0)
        idx = np.where((np.round(consolidated[:, 0], 1) == 1000.0) & (np.round(consolidated[:, 1], 1) == 2000.0))[0][0]
        self.assertAlmostEqual(consolidated[idx, 3], 55.0)

    def test_load_dem_raw_array_fallback(self):
        raw_grid = np.zeros((20, 30), dtype=np.float64)
        grid, meta = load_dem(raw_grid, dx=2.0, dy=2.0)
        self.assertEqual(grid.shape, (20, 30))
        self.assertEqual(meta["bounds"], (0.0, 0.0, 60.0, 40.0))

    def test_bedrock_map_save_without_crs(self):
        raw_grid = np.zeros((10, 10), dtype=np.float64)
        bmap = BedrockMap(grid=raw_grid, bounds=(0.0, 0.0, 10.0, 10.0), crs=None)
        test_out = "test_output_no_crs.tif"
        try:
            res = bmap.save(test_out)
            self.assertTrue(os.path.exists(test_out))
        finally:
            if os.path.exists(test_out):
                os.remove(test_out)

    def test_bedrock_map_save_directory_path(self):
        raw_grid = np.zeros((5, 5), dtype=np.float64)
        bmap = BedrockMap(grid=raw_grid, bounds=(0.0, 0.0, 5.0, 5.0), crs=None)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            res = bmap.save(tmpdir)
            expected = os.path.join(tmpdir, "final_bedrock.tif")
            self.assertEqual(res, expected)
            self.assertTrue(os.path.exists(expected))


    def test_projected_metric_crs_check(self):
        from pysole.raster import check_projected_metric_crs
        # EPSG:4326 (WGS84 Lat/Lon degrees) must raise ValueError
        with self.assertRaises(ValueError):
            check_projected_metric_crs(crs="EPSG:4326")

        # Projected EPSG:32632 (UTM Zone 32N meters) must pass cleanly
        check_projected_metric_crs(crs="EPSG:32632")

        # Lat/Lon degree coordinates in range [-180, 180] x [-90, 90] must raise ValueError
        lat_lon_pts = np.array([
            [13.1234, 47.5678, 100.0],
            [13.1250, 47.5690, 110.0],
        ])
        with self.assertRaises(ValueError):
            check_projected_metric_crs(coords=lat_lon_pts)

        # Geographic bounds in lat/lon degrees must raise ValueError
        with self.assertRaises(ValueError):
            check_projected_metric_crs(bounds=(13.0, 47.0, 14.0, 48.0))

    def test_load_dem_csv_matrix_and_origin(self):
        import tempfile
        matrix = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            np.savetxt(f.name, matrix, delimiter=",")
            tmp_path = f.name
        try:
            grid, meta = load_dem(tmp_path, dx=10.0, dy=10.0, origin=(500000.0, 5200000.0), crs="EPSG:32632")
            self.assertEqual(grid.shape, (2, 2))
            self.assertEqual(meta["dx"], 10.0)
            self.assertEqual(meta["dy"], 10.0)
            self.assertEqual(meta["bounds"], (500000.0, 5200000.0, 500020.0, 5200020.0))
            self.assertEqual(meta["crs"], "EPSG:32632")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_dem_csv_3column_xyz(self):
        import tempfile
        import pandas as pd
        df = pd.DataFrame({
            "X": [100.0, 110.0, 100.0, 110.0],
            "Y": [500.0, 500.0, 510.0, 510.0],
            "Z": [10.0, 20.0, 30.0, 40.0],
        })
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f.name, index=False)
            tmp_path = f.name
        try:
            grid, meta = load_dem(tmp_path)
            self.assertEqual(grid.shape, (2, 2))
            self.assertEqual(meta["dx"], 10.0)
            self.assertEqual(meta["dy"], 10.0)
            self.assertEqual(meta["bounds"], (95.0, 495.0, 115.0, 515.0))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_outline_none_with_nans(self):
        dem = np.array([
            [np.nan, 100.0, 105.0, np.nan],
            [110.0, np.nan, 115.0, 120.0],
            [np.nan, 125.0, 130.0, np.nan],
        ], dtype=np.float64)
        meta = {"dx": 10.0, "dy": 10.0, "bounds": (0.0, 0.0, 40.0, 30.0)}
        mask = load_outline(None, dem, meta)
        self.assertEqual(mask.shape, dem.shape)
        self.assertEqual(np.sum(mask), 7)
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[1, 1])
        self.assertTrue(mask[0, 1])

    def test_load_outline_none_without_nans(self):
        dem = np.array([
            [100.0, 105.0],
            [110.0, 115.0],
        ], dtype=np.float64)
        meta = {"dx": 10.0, "dy": 10.0, "bounds": (0.0, 0.0, 20.0, 20.0)}
        mask = load_outline(None, dem, meta)
        self.assertEqual(mask.shape, dem.shape)
        self.assertTrue(np.all(mask))


if __name__ == "__main__":
    unittest.main()
