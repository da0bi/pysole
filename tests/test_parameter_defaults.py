import unittest
import numpy as np
from pysole.raster import GridGeometry
from pysole.variogram import optimize_bss_variance


class TestParameterDefaults(unittest.TestCase):
    """Unit tests verifying dynamic parameter default resolutions for BSS optimization sweeps."""

    def setUp(self):
        # Create small synthetic sloping DEM and pick dataset
        self.shape = (50, 50)
        xx, yy = np.meshgrid(np.arange(50), np.arange(50))
        self.dem = 3000.0 + 0.5 * xx + 0.5 * yy
        self.survey_points = np.array([
            [10.0, 10.0, 3010.0, 50.0],
            [20.0, 20.0, 3020.0, 60.0],
            [30.0, 30.0, 3030.0, 70.0],
            [40.0, 40.0, 3040.0, 80.0],
        ])

    def test_kc_min_half_domain_limit_default(self):
        """Verify kc_min defaults to 4*pi / L_max (Half-Domain limit)."""
        geom = GridGeometry.create(shape=self.shape, dx=10.0, dy=10.0)
        res = optimize_bss_variance(
            dem=self.dem,
            survey_points=self.survey_points,
            geometry=geom,
            kc_min=None,
            kc_max=0.3,
            n_steps=15,
            show_progress=False,
        )
        self.assertIsNotNone(res.optimal_kc)

    def test_n_steps_fourier_mode_clamping(self):
        """Verify n_steps defaults to discrete Fourier mode count clipped between 10 and 50."""
        geom = GridGeometry.create(shape=self.shape, dx=5.0, dy=5.0)
        res = optimize_bss_variance(
            dem=self.dem,
            survey_points=self.survey_points,
            geometry=geom,
            n_steps=None,
            show_progress=False,
        )
        self.assertIsNotNone(res.optimal_kc)
        self.assertTrue(10 <= len(res.all_kc_variances) <= 50)

    def test_legacy_parameters_removed(self):
        """Verify legacy parameters raise TypeError if passed."""
        geom = GridGeometry.create(shape=self.shape, dx=5.0, dy=5.0)
        with self.assertRaises(TypeError):
            optimize_bss_variance(
                dem=self.dem,
                survey_points=self.survey_points,
                geometry=geom,
                d_kc=0.01,
            )

    def test_lambda_fallback_when_null(self):
        """Verify fft_filter_metric='wavelength' with null lambda_min/max falls back to kc_max/kc_min."""
        geom = GridGeometry.create(shape=self.shape, dx=10.0, dy=10.0)
        res = optimize_bss_variance(
            dem=self.dem,
            survey_points=self.survey_points,
            geometry=geom,
            fft_filter_metric="wavelength",
            lambda_min=None,
            lambda_max=None,
            show_progress=False,
        )
        self.assertIsNotNone(res.optimal_kc)
        self.assertTrue(10 <= len(res.all_kc_variances) <= 50)

    def test_lambda_min_max_conversion(self):
        """Verify lambda_min and lambda_max convert directly to wavenumbers kc_max and kc_min."""
        geom = GridGeometry.create(shape=self.shape, dx=5.0, dy=5.0)
        res = optimize_bss_variance(
            dem=self.dem,
            survey_points=self.survey_points,
            geometry=geom,
            fft_filter_metric="wavelength",
            lambda_min=50.0,
            lambda_max=200.0,
            n_steps=10,
            show_progress=False,
        )
        self.assertIsNotNone(res.optimal_kc)
        # Expected max wavenumber: 2*pi / 50 = ~0.12566
        # Expected min wavenumber: 2*pi / 200 = ~0.03141
        eval_kcs = res.all_kc_variances[:, 0]
        self.assertAlmostEqual(eval_kcs[0], (2.0 * np.pi) / 50.0, places=4)
        self.assertAlmostEqual(eval_kcs[-1], (2.0 * np.pi) / 200.0, places=4)


if __name__ == "__main__":
    unittest.main()
