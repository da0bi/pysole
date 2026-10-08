"""
Unit tests for optional PyKrige engine integration and Regression Kriging mode.
"""

import unittest
import numpy as np
import pysole
from pysole.raster import GridGeometry
from pysole.interpolation import kriging_interpolation, KrigingResult


class TestPyKrigeIntegration(unittest.TestCase):
    def setUp(self):
        self.geometry = GridGeometry.create(shape=(20, 20), dx=10.0, dy=10.0)
        xx, yy = self.geometry.meshgrid
        self.dem = 1000.0 + 0.1 * xx + 0.2 * yy

        np.random.seed(42)
        x_pts = np.random.uniform(10.0, 180.0, 15)
        y_pts = np.random.uniform(10.0, 180.0, 15)
        z_pts = np.random.uniform(50.0, 150.0, 15)
        self.sample_pts = np.column_stack((x_pts, y_pts, z_pts))

    def test_native_engine_default(self):
        krig_res = kriging_interpolation(
            sample_points=self.sample_pts,
            geometry=self.geometry,
            method="universal",
            engine="native",
        )
        self.assertIsInstance(krig_res, KrigingResult)
        self.assertEqual(krig_res.bedrock_grid.shape, (20, 20))

    def test_pykrige_engine_and_regression(self):
        try:
            import pykrige
            has_pykrige = True
        except ImportError:
            has_pykrige = False

        if has_pykrige:
            # Test PyKrige Universal Kriging engine
            res_uk = kriging_interpolation(
                sample_points=self.sample_pts,
                geometry=self.geometry,
                method="universal",
                engine="pykrige",
            )
            self.assertIsInstance(res_uk, KrigingResult)
            self.assertEqual(res_uk.bedrock_grid.shape, (20, 20))

            # Test Regression Kriging
            res_rk = kriging_interpolation(
                sample_points=self.sample_pts,
                geometry=self.geometry,
                method="regression",
            )
            self.assertIsInstance(res_rk, KrigingResult)
            self.assertEqual(res_rk.bedrock_grid.shape, (20, 20))
        else:
            with self.assertRaises(ImportError):
                kriging_interpolation(
                    sample_points=self.sample_pts,
                    geometry=self.geometry,
                    method="regression",
                )


    def test_native_multi_drift_combined(self):
        # Test combining raster elevation drift (z_dem) + quadratic spatial coordinate drift in native engine
        res_multi = kriging_interpolation(
            sample_points=self.sample_pts,
            geometry=self.geometry,
            method="universal",
            engine="native",
            dem_grid=self.dem,
            drift_terms=["z_dem", "quadratic_xy"],
        )
        self.assertIsInstance(res_multi, KrigingResult)
        self.assertEqual(res_multi.bedrock_grid.shape, (20, 20))
        self.assertFalse(np.isnan(res_multi.bedrock_grid).all())


if __name__ == "__main__":
    unittest.main()
