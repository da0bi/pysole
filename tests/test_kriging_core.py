"""Tests of the shared kriging core (:mod:`pysole.kriging_core`) and the BLAS thread policy."""
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np
from threadpoolctl import threadpool_info

from pysole import interpolation
from pysole.kriging_core import KrigingSystem, polynomial_drift_columns, resolve_variogram_params
from pysole.memory import blas_thread_limit, physical_core_count


def _spherical(h, a, c, n):
    """Independent spherical semivariogram (zero at h == 0)."""
    h = np.asarray(h, dtype=float)
    g = np.where(h < a, n + c * (1.5 * h / a - 0.5 * (h / a) ** 3), n + c)
    return np.where(h == 0, 0.0, g)


def _dense_reference(x, y, z, cols, rows, xp, yp, params):
    """Straightforward dense solve of the augmented system for every prediction point (independent of the core)."""
    a, c, n = params
    npts = len(z)
    p = cols.shape[1]
    d_pp = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    amat = np.zeros((npts + 1 + p, npts + 1 + p))
    amat[:npts, :npts] = _spherical(d_pp, a, c, n) + 1e-6 * c * np.eye(npts)
    amat[:npts, npts] = amat[npts, :npts] = 1.0
    for j in range(p):
        amat[:npts, npts + 1 + j] = amat[npts + 1 + j, :npts] = cols[:, j]
    est, var = [], []
    for k in range(len(xp)):
        rhs = np.zeros(npts + 1 + p)
        rhs[:npts] = _spherical(np.hypot(x - xp[k], y - yp[k]), a, c, n)
        rhs[npts] = 1.0
        for j in range(p):
            rhs[npts + 1 + j] = rows[j, k]
        sol = np.linalg.solve(amat, rhs)
        est.append(sol[:npts] @ z)
        var.append(max(sol @ rhs, 0.0))
    return np.array(est), np.array(var)


class TestKrigingSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.n = 60
        cls.x = rng.uniform(0, 1000, cls.n)
        cls.y = rng.uniform(0, 800, cls.n)
        cls.z = 50 + 0.02 * cls.x - 0.01 * cls.y + rng.normal(0, 3, cls.n)
        cls.xp = rng.uniform(0, 1000, 25)
        cls.yp = rng.uniform(0, 800, 25)
        cls.params = (400.0, 12.0, 0.5)

    def _drift(self, quadratic):
        xn, yn = (self.x - 500) / 1000, (self.y - 400) / 800
        xpn, ypn = (self.xp - 500) / 1000, (self.yp - 400) / 800
        cols = np.column_stack(polynomial_drift_columns(xn, yn, quadratic))
        rows = np.vstack(polynomial_drift_columns(xpn, ypn, quadratic))
        return cols, rows

    def test_ordinary_matches_dense_solve(self):
        system = KrigingSystem(self.x, self.y, self.z, None, "spherical", self.params)
        est, var = system.predict_points(self.xp, self.yp, None, return_variance=True)
        ref_est, ref_var = _dense_reference(
            self.x, self.y, self.z, np.zeros((self.n, 0)), np.zeros((0, 25)), self.xp, self.yp, self.params
        )
        np.testing.assert_allclose(est, ref_est, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(var, ref_var, rtol=1e-8, atol=1e-9)
        self.assertEqual(system.n_drift, 1)

    def test_linear_and_quadratic_match_dense_solve(self):
        for quadratic, n_cols in ((False, 2), (True, 5)):
            cols, rows = self._drift(quadratic)
            self.assertEqual(cols.shape[1], n_cols)
            system = KrigingSystem(self.x, self.y, self.z, cols, "spherical", self.params)
            est, var = system.predict_points(self.xp, self.yp, rows, return_variance=True)
            ref_est, ref_var = _dense_reference(self.x, self.y, self.z, cols, rows, self.xp, self.yp, self.params)
            np.testing.assert_allclose(est, ref_est, rtol=1e-8, atol=1e-8, err_msg=f"quadratic={quadratic}")
            np.testing.assert_allclose(var, ref_var, rtol=1e-6, atol=1e-8, err_msg=f"quadratic={quadratic}")
            self.assertEqual(system.n_drift, 1 + n_cols)

    def test_regularisation_is_fraction_of_sill(self):
        system = KrigingSystem(self.x, self.y, self.z, None, "spherical", self.params)
        self.assertAlmostEqual(system.reg_value, 1e-6 * self.params[1])
        self.assertEqual((system.range_a, system.sill, system.nugget), self.params)

    def test_variogram_is_zero_at_zero_lag_and_estimate_is_exact_at_data(self):
        system = KrigingSystem(self.x, self.y, self.z, None, "spherical", (400.0, 12.0, 0.0))
        self.assertEqual(float(system.variogram(np.array([0.0]))[0]), 0.0)
        est, _ = system.predict_points(self.x[:5], self.y[:5], None)
        np.testing.assert_allclose(est, self.z[:5], atol=1e-3)

    def test_variance_flag_does_not_change_the_estimate(self):
        cols, rows = self._drift(False)
        system = KrigingSystem(self.x, self.y, self.z, cols, "spherical", self.params)
        est_a, var_a = system.predict_points(self.xp, self.yp, rows, return_variance=False)
        est_b, var_b = system.predict_points(self.xp, self.yp, rows, return_variance=True)
        self.assertIsNone(var_a)
        np.testing.assert_array_equal(est_a, est_b)
        self.assertTrue(np.all(var_b >= 0))

    def test_predict_points_is_thread_safe(self):
        cols, rows = self._drift(True)
        system = KrigingSystem(self.x, self.y, self.z, cols, "spherical", self.params)
        system.factorize()
        system.inverse()
        system.release()
        serial = [system.predict_points(self.xp[i : i + 5], self.yp[i : i + 5], rows[:, i : i + 5], True) for i in range(0, 25, 5)]
        with ThreadPoolExecutor(max_workers=5) as pool:
            threaded = list(
                pool.map(lambda i: system.predict_points(self.xp[i : i + 5], self.yp[i : i + 5], rows[:, i : i + 5], True), range(0, 25, 5))
            )
        for (e1, v1), (e2, v2) in zip(serial, threaded):
            np.testing.assert_array_equal(e1, e2)
            np.testing.assert_array_equal(v1, v2)

    def test_inverse_is_a_true_inverse_and_cached(self):
        cols, _ = self._drift(False)
        system = KrigingSystem(self.x, self.y, self.z, cols, "spherical", self.params)
        system.factorize()
        inv = system.inverse()
        self.assertIs(inv, system.inverse())
        np.testing.assert_allclose(system._matrix @ inv, np.eye(self.n + 3), atol=1e-6)

    def test_release_drops_matrix_but_keeps_prediction(self):
        system = KrigingSystem(self.x, self.y, self.z, None, "spherical", self.params)
        est_before, _ = system.predict_points(self.xp, self.yp, None)
        system.release()
        est_after, _ = system.predict_points(self.xp, self.yp, None)
        np.testing.assert_array_equal(est_before, est_after)
        with self.assertRaises(RuntimeError):
            system.factorize()

    def test_wrong_drift_shapes_raise(self):
        with self.assertRaises(ValueError):
            KrigingSystem(self.x, self.y, self.z, np.zeros((self.n + 1, 2)))
        cols, _ = self._drift(False)
        system = KrigingSystem(self.x, self.y, self.z, cols, "spherical", self.params)
        with self.assertRaises(ValueError):
            system.predict_points(self.xp, self.yp, None)

    def test_resolve_variogram_params(self):
        vals = np.array([1.0, 3.0, 5.0])
        self.assertEqual(resolve_variogram_params(None, vals, 100.0), (60.0, float(np.var(vals)), 0.0))
        self.assertEqual(resolve_variogram_params(None, np.ones(3), 1.0), (1.0, 1.0, 0.0))
        self.assertEqual(resolve_variogram_params({"range": 9.0}, vals, 100.0), (9.0, float(np.var(vals)), 0.0))
        self.assertEqual(resolve_variogram_params((1.0, 2.0, 3.0), vals, 100.0), (1.0, 2.0, 3.0))

    def test_polynomial_columns(self):
        x, y = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        lin = polynomial_drift_columns(x, y, False)
        quad = polynomial_drift_columns(x, y, True)
        self.assertEqual(len(lin), 2)
        self.assertEqual(len(quad), 5)
        np.testing.assert_array_equal(quad[4], x * y)


def _blas_threads():
    info = [lib["num_threads"] for lib in threadpool_info() if lib.get("user_api") == "blas"]
    return info


class TestBlasThreadPolicy(unittest.TestCase):
    def setUp(self):
        np.dot(np.ones((4, 4)), np.ones((4, 4)))  # make sure the BLAS library is loaded
        if not _blas_threads():
            self.skipTest("no BLAS library detected by threadpoolctl")

    def test_physical_core_count_positive(self):
        self.assertGreaterEqual(physical_core_count(), 1)

    def test_cap_applied_and_restored(self):
        before = _blas_threads()
        with blas_thread_limit(1) as applied:
            self.assertEqual(applied, 1)
            self.assertTrue(all(t == 1 for t in _blas_threads()))
        self.assertEqual(_blas_threads(), before)

    def test_default_cap_is_physical_cores(self):
        with blas_thread_limit() as applied:
            self.assertEqual(applied, physical_core_count())
            self.assertTrue(all(t <= physical_core_count() for t in _blas_threads()))

    def test_restored_after_exception(self):
        before = _blas_threads()
        with self.assertRaises(RuntimeError):
            with blas_thread_limit(1):
                raise RuntimeError("boom")
        self.assertEqual(_blas_threads(), before)

    def test_engine_runs_chunks_with_one_blas_thread_and_restores(self):
        rng = np.random.default_rng(3)
        pts = np.column_stack((rng.uniform(0, 1000, 40), rng.uniform(0, 1000, 40), rng.normal(100, 5, 40)))
        xs, ys = np.linspace(0, 1000, 50), np.linspace(0, 1000, 40)  # 2000 cells = 4 chunks of 500 (chunk size patched)
        calls = []
        real = interpolation.blas_thread_limit

        def spy(limit=None):
            calls.append(limit)
            return real(limit)

        before = _blas_threads()
        with mock.patch.object(interpolation, "blas_thread_limit", spy), mock.patch.object(
            interpolation, "kriging_chunk_size", lambda n: 500
        ):
            z_par, v_par = interpolation.built_in_kriging_interpolation(
                pts, xs, ys, method="ordinary", n_cores=4, show_progress=False
            )
            z_ser, v_ser = interpolation.built_in_kriging_interpolation(
                pts, xs, ys, method="ordinary", n_cores=1, show_progress=False
            )
        self.assertIn(1, calls)  # chunk threads
        self.assertIn(None, calls)  # factorisation / serial path
        self.assertEqual(_blas_threads(), before)
        np.testing.assert_allclose(z_par, z_ser, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(v_par, v_ser, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
