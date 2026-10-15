"""
Tests of the PyKrige engines (v0.5.0, Phase 5): Universal Kriging with every drift term and Regression Kriging
with the drift terms as Random Forest covariates. Skipped if PyKrige is not installed.
"""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from pysole.drift import build_drift_suite
from pysole.interpolation import kriging_interpolation, pykrige_kriging_interpolation, standardise_drift_grid
from pysole.kriging_core import polynomial_drift_columns
from tests.test_drift_covariates import make_asymmetric_fixture, make_points

HAS_PYKRIGE = importlib.util.find_spec("pykrige") is not None
PARAMS = {"range": 150.0, "sill": 4.0, "nugget": 0.05}


class PyKrigeFixture(unittest.TestCase):
    def setUp(self):
        self.geom, self.dem, self.slope, self.curv, self.grids = make_asymmetric_fixture()
        x, y, z = make_points(self.geom, self.dem, n=90)
        self.pts = np.column_stack((x, y, z))

    def run_engine(self, engine, method="universal", terms=None, model="spherical", params=PARAMS, **kwargs):
        terms = list(terms or [])
        return kriging_interpolation(
            sample_points=self.pts, geometry=self.geom, method=method, variogram_model=model, variogram_params=params,
            dem_grid=self.dem, opt_slope_grid=self.slope, drift_terms=terms, include_zero_boundary_condition=False,
            n_cores=1, show_progress=False, engine=engine, return_variance=True,
            external_drift_grid={"curvature_dem": self.curv} if "curvature_dem" in terms else None, **kwargs,
        )


@unittest.skipUnless(HAS_PYKRIGE, "PyKrige is not installed")
class TestPyKrigeUniversalMatchesNative(PyKrigeFixture):
    def assert_match(self, terms, model="spherical", method="universal", tol=1e-4):
        native = self.run_engine("native", method, terms, model)
        pykrige = self.run_engine("pykrige", method, terms, model)
        scale = float(np.ptp(native.bedrock_grid))
        self.assertLess(float(np.max(np.abs(native.bedrock_grid - pykrige.bedrock_grid))) / scale, tol)
        return native, pykrige

    def test_ordinary_and_all_23_drift_models(self):
        for terms in build_drift_suite(None):
            with self.subTest(model=terms or "ordinary"):
                self.assert_match(terms, method="universal" if terms else "ordinary")

    def test_variance_matches_for_a_multi_drift_model(self):
        native, pykrige = self.assert_match(["sia", "z_dem", "quadratic_xy"])
        # PyKrige's own kriging variance is ~1 % below the native one, which an independent dense solve
        # of the kriging system reproduces exactly: the native variance is the reference, PyKrige's only
        # has to agree closely and rank the grid alike.
        nv, pv = native.variance_grid, pykrige.variance_grid
        self.assertLess(float(np.max(np.abs(nv - pv))) / float(np.ptp(nv)), 3e-2)
        self.assertGreater(float(np.corrcoef(nv.ravel(), pv.ravel())[0, 1]), 0.999)

    def test_exponential_and_gaussian_conventions(self):
        for model in ("exponential", "gaussian"):
            for terms, method in (([], "ordinary"), (["z_dem", "linear_xy"], "universal")):
                with self.subTest(model=model, terms=terms):
                    # the near-singular Gaussian system is regularised natively (1e-6 * sill)
                    self.assert_match(terms, model=model, method=method, tol=1e-3 if model == "gaussian" else 1e-4)

    def test_descending_grid_axes(self):
        geom = self.geom
        flipped = type(geom)(
            shape=geom.shape, dx=geom.dx, dy=geom.dy, x_coords=geom.x_coords, y_coords=geom.y_coords[::-1],
            bounds=geom.bounds,
        )
        dem, slope = self.dem[::-1], self.slope[::-1]
        kwargs = dict(
            sample_points=self.pts, method="universal", variogram_model="spherical", variogram_params=PARAMS,
            dem_grid=dem, opt_slope_grid=slope, drift_terms=["sia", "z_dem", "quadratic_xy"],
            include_zero_boundary_condition=False, n_cores=1, show_progress=False,
        )
        native = kriging_interpolation(geometry=flipped, engine="native", **kwargs).bedrock_grid
        pykrige = kriging_interpolation(geometry=flipped, engine="pykrige", **kwargs).bedrock_grid
        self.assertLess(float(np.max(np.abs(native - pykrige))) / float(np.ptp(native)), 1e-4)

    def test_without_variogram_parameters_pykrige_fits_its_own(self):
        result = self.run_engine("pykrige", "universal", ["z_dem", "curvature_dem"], params=None)
        self.assertTrue(np.isfinite(result.bedrock_grid).all())

    def test_linear_model_runs_with_pykrige_parameters_ignored(self):
        result = self.run_engine("pykrige", "universal", ["sia"], model="linear")
        self.assertTrue(np.isfinite(result.bedrock_grid).all())

    def test_every_listed_term_changes_the_result(self):
        reference = self.run_engine("pykrige", "universal", ["z_dem"]).bedrock_grid
        for terms in (["z_dem", "sia"], ["z_dem", "curvature_dem"], ["z_dem", "linear_xy"], ["z_dem", "quadratic_xy"]):
            with self.subTest(terms=terms):
                other = self.run_engine("pykrige", "universal", terms).bedrock_grid
                self.assertGreater(float(np.max(np.abs(other - reference))), 1e-6)

    def test_missing_grid_and_empty_models_raise(self):
        with self.assertRaises(ValueError):
            kriging_interpolation(
                sample_points=self.pts, geometry=self.geom, method="universal", drift_terms=["z_dem"],
                engine="pykrige", show_progress=False, include_zero_boundary_condition=False,
            )
        x_coords, y_coords = self.geom.x_coords, self.geom.y_coords
        with self.assertRaises(ValueError):
            pykrige_kriging_interpolation(self.pts, x_coords, y_coords, method="universal", drift_terms=[])
        flat = {"z_dem": np.full(self.geom.shape, 5.0)}
        with self.assertLogs("pysole", level="WARNING") as cm, self.assertRaises(ValueError):
            pykrige_kriging_interpolation(self.pts, x_coords, y_coords, "universal", external_drift_grid=flat)
        self.assertTrue(any("flat" in m for m in cm.output))


@unittest.skipUnless(HAS_PYKRIGE, "PyKrige is not installed")
class TestPyKrigeRegressionCovariates(PyKrigeFixture):
    def captured_covariates(self, terms):
        from pykrige.rk import RegressionKriging

        seen = {}
        original = RegressionKriging.fit

        def spy(rk, p, x, y, *args, **kwargs):
            seen["p"] = np.array(p)
            return original(rk, p, x, y, *args, **kwargs)

        with patch.object(RegressionKriging, "fit", spy):
            result = self.run_engine("native", "regression", terms)
        return seen["p"], result

    def test_no_terms_use_the_coordinates(self):
        covariates, result = self.captured_covariates([])
        np.testing.assert_allclose(covariates, self.pts[:, :2])
        self.assertEqual(result.bedrock_grid.shape, self.geom.shape)
        self.assertTrue(np.isnan(result.variance_grid).all())

    def test_terms_are_the_covariates(self):
        x_mean, x_scale = float(np.mean(self.geom.x_coords)), float(np.ptp(self.geom.x_coords))
        y_mean, y_scale = float(np.mean(self.geom.y_coords)), float(np.ptp(self.geom.y_coords))
        poly = lambda quad: np.column_stack(  # noqa: E731
            polynomial_drift_columns((self.pts[:, 0] - x_mean) / x_scale, (self.pts[:, 1] - y_mean) / y_scale, quad)
        )
        cases = {
            ("linear_xy",): 2, ("quadratic_xy",): 5, ("sia", "z_dem"): 2, ("sia", "curvature_dem", "quadratic_xy"): 7,
        }
        for terms, n_columns in cases.items():
            with self.subTest(terms=terms):
                covariates, _ = self.captured_covariates(list(terms))
                self.assertEqual(covariates.shape, (len(self.pts), n_columns))
        covariates, _ = self.captured_covariates(["quadratic_xy"])
        np.testing.assert_allclose(covariates, poly(True), atol=1e-12)
        covariates, _ = self.captured_covariates(["z_dem"])
        std = standardise_drift_grid("z_dem", self.grids["z_dem"])
        from pysole.interpolation import sample_drift_grid

        np.testing.assert_allclose(
            covariates[:, 0], sample_drift_grid("z_dem", std, self.geom.x_coords, self.geom.y_coords, self.pts[:, 0], self.pts[:, 1])
        )

    def test_terms_change_the_prediction_and_nothing_is_dropped(self):
        base = self.run_engine("native", "regression", []).bedrock_grid
        for terms in (["quadratic_xy"], ["sia"], ["z_dem", "quadratic_xy"]):
            with self.subTest(terms=terms):
                other = self.run_engine("native", "regression", terms).bedrock_grid
                self.assertTrue(np.isfinite(other).all())
                self.assertGreater(float(np.max(np.abs(other - base))), 1e-9)
        # A random forest is invariant to monotone rescaling of x and y: 'linear_xy' (x, y) is exactly
        # the no-term default (x, y), not a dropped term.
        linear = self.run_engine("native", "regression", ["linear_xy"]).bedrock_grid
        np.testing.assert_allclose(linear, base)

    def test_every_default_suite_model_runs(self):
        for terms in build_drift_suite(None):
            with self.subTest(model=terms or "ordinary"):
                self.assertTrue(np.isfinite(self.run_engine("pykrige", "regression", terms).bedrock_grid).all())

    def test_missing_raster_grid_raises(self):
        with self.assertRaises(ValueError):
            kriging_interpolation(
                sample_points=self.pts, geometry=self.geom, method="regression", drift_terms=["sia"],
                show_progress=False, include_zero_boundary_condition=False,
            )

    def test_regression_default_null_uses_all_primitives(self):
        res = kriging_interpolation(
            sample_points=self.pts, geometry=self.geom, method="regression",
            dem_grid=self.grids["z_dem"], opt_slope_grid=self.grids["sia"],
            show_progress=False, include_zero_boundary_condition=False,
        )
        self.assertTrue(np.isfinite(res.bedrock_grid).all())

    def test_regression_default_null_without_dem_falls_back_to_coordinates(self):
        with self.assertLogs("pysole", level="WARNING") as cm:
            res = kriging_interpolation(
                sample_points=self.pts, geometry=self.geom, method="regression",
                show_progress=False, include_zero_boundary_condition=False,
            )
        self.assertTrue(np.isfinite(res.bedrock_grid).all())
        self.assertTrue(any("DEM grid is missing for Regression Kriging" in msg for msg in cm.output))


if __name__ == "__main__":
    unittest.main()
