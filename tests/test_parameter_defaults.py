import unittest
import numpy as np
from pysole.raster import GridGeometry
from pysole.variogram import optimize_bss_variance


def _make_inputs(n: int):
    """Sloping n x n synthetic DEM (cell units) and a 4-pick survey well inside the domain."""
    xx, yy = np.meshgrid(np.arange(n), np.arange(n))
    dem = 3000.0 + 0.5 * xx + 0.5 * yy
    pts = np.array([
        [10.0, 10.0, 3010.0, 50.0],
        [20.0, 20.0, 3020.0, 60.0],
        [30.0, 30.0, 3030.0, 70.0],
        [40.0, 40.0, 3040.0, 80.0],
    ])
    return dem, pts


def _expected_n_steps(n: int, dx: float, kc_min: float | None = None, kc_max: float | None = None) -> int:
    """n_steps = clip(floor((kc_max - kc_min) * L_max / (2 pi)), 10, 50) with Half-Domain kc_min = 4 pi / L_max."""
    l_max = n * dx
    kc_max = np.pi / dx if kc_max is None else kc_max
    kc_min = 4.0 * np.pi / l_max if kc_min is None else kc_min
    return int(np.clip(np.floor((kc_max - kc_min) * l_max / (2.0 * np.pi)), 10, 50))


class TestParameterDefaults(unittest.TestCase):
    """Unit tests verifying dynamic parameter default resolutions for BSS optimization sweeps."""

    def setUp(self):
        self.shape = (50, 50)
        self.dem, self.survey_points = _make_inputs(50)

    def test_kc_min_half_domain_limit_default(self):
        """Verify kc_min defaults to 4*pi / L_max (Half-Domain limit) with L_max = 50 * 10 m = 500 m."""
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
        kcs = res.all_kc_variances[:, 0]
        kc_min_expected = 4.0 * np.pi / 500.0  # 0.025133 rad/m
        self.assertEqual(len(kcs), 15)
        self.assertAlmostEqual(kcs[0], 0.3, places=12)
        self.assertAlmostEqual(kcs[-1], kc_min_expected, places=12)
        # geometric (log-uniform) spacing between the two bounds
        np.testing.assert_allclose(kcs, np.geomspace(0.3, kc_min_expected, 15), rtol=1e-12)
        self.assertTrue(kc_min_expected <= res.optimal_kc <= 0.3)

    def test_n_steps_fourier_mode_clamping(self):
        """Verify n_steps defaults to the discrete Fourier mode count clipped between 10 and 50 (exact counts)."""
        # (a) unclipped: N = 51 -> floor(N/2 - 2) = 23 modes (odd N avoids the exact-integer floating-point tie)
        dem51, pts = _make_inputs(51)
        geom = GridGeometry.create(shape=(51, 51), dx=5.0, dy=5.0)
        res = optimize_bss_variance(dem=dem51, survey_points=pts, geometry=geom, n_steps=None, show_progress=False)
        self.assertEqual(_expected_n_steps(51, 5.0), 23)
        self.assertEqual(len(res.all_kc_variances), 23)

        # (b) clamped from below: N = 20 -> 8 modes -> 10 steps
        dem20, pts = _make_inputs(20)
        geom = GridGeometry.create(shape=(20, 20), dx=5.0, dy=5.0)
        res = optimize_bss_variance(dem=dem20, survey_points=pts, geometry=geom, n_steps=None, show_progress=False)
        self.assertEqual(_expected_n_steps(20, 5.0), 10)
        self.assertEqual(len(res.all_kc_variances), 10)

        # (c) clamped from above: N = 200 -> 98 modes -> 50 steps
        dem200, pts = _make_inputs(200)
        geom = GridGeometry.create(shape=(200, 200), dx=1.0, dy=1.0)
        res = optimize_bss_variance(dem=dem200, survey_points=pts, geometry=geom, n_steps=None, show_progress=False)
        self.assertEqual(_expected_n_steps(200, 1.0), 50)
        self.assertEqual(len(res.all_kc_variances), 50)

        # (d) explicit n_steps are honoured but floored at 3
        geom = GridGeometry.create(shape=self.shape, dx=5.0, dy=5.0)
        res = optimize_bss_variance(dem=self.dem, survey_points=self.survey_points, geometry=geom, n_steps=1, show_progress=False)
        self.assertEqual(len(res.all_kc_variances), 3)

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
        """Verify fft_filter_metric='wavelength' with null lambda_min/max falls back to k_Nyquist / Half-Domain kc_min."""
        dem51, pts = _make_inputs(51)
        geom = GridGeometry.create(shape=(51, 51), dx=10.0, dy=10.0)
        res = optimize_bss_variance(
            dem=dem51,
            survey_points=pts,
            geometry=geom,
            fft_filter_metric="wavelength",
            lambda_min=None,
            lambda_max=None,
            show_progress=False,
        )
        kcs = res.all_kc_variances[:, 0]
        kc_max_expected = np.pi / 10.0                # k_Nyquist
        kc_min_expected = 4.0 * np.pi / (51 * 10.0)   # Half-Domain limit
        self.assertEqual(len(kcs), _expected_n_steps(51, 10.0))
        self.assertEqual(len(kcs), 23)
        self.assertAlmostEqual(kcs[0], kc_max_expected, places=12)
        self.assertAlmostEqual(kcs[-1], kc_min_expected, places=12)

    def test_lambda_min_max_conversion(self):
        """Verify lambda_min and lambda_max convert directly to wavenumbers kc_max = 2pi/lambda_min, kc_min = 2pi/lambda_max."""
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
        eval_kcs = res.all_kc_variances[:, 0]
        self.assertEqual(len(eval_kcs), 10)
        self.assertAlmostEqual(eval_kcs[0], (2.0 * np.pi) / 50.0, places=12)
        self.assertAlmostEqual(eval_kcs[-1], (2.0 * np.pi) / 200.0, places=12)
        self.assertTrue((2.0 * np.pi) / 200.0 <= res.optimal_kc <= (2.0 * np.pi) / 50.0)

    def test_lambda_min_below_nyquist_is_clamped(self):
        """lambda_min below the Nyquist wavelength (2*dx) clamps kc_max to k_Nyquist = pi/dx."""
        geom = GridGeometry.create(shape=self.shape, dx=5.0, dy=5.0)
        with self.assertLogs("pysole", level="WARNING"):
            res = optimize_bss_variance(
                dem=self.dem,
                survey_points=self.survey_points,
                geometry=geom,
                fft_filter_metric="wavelength",
                lambda_min=2.0,   # < 10 m Nyquist wavelength
                lambda_max=200.0,
                n_steps=5,
                show_progress=False,
            )
        self.assertAlmostEqual(res.all_kc_variances[0, 0], np.pi / 5.0, places=12)


if __name__ == "__main__":
    unittest.main()
