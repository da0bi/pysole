"""
Regression tests for the drift covariates used by the Drift Analyzer (v0.5.0, Phase 1).

The analyzer must sample every raster covariate exactly like the production engine: bottom-up grid
orientation (row 0 = minimum Y), grid-level standardisation, k_c-smoothed curvature and the polynomial
columns of the native engine (``quadratic_xy`` = x, y, x^2, y^2, xy).
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter

from pysole.drift import RASTER_DRIFT_TERMS
from pysole.interpolation import (
    DriftBasis,
    KrigingResult,
    build_drift_grids,
    kriging_interpolation,
)
from pysole.kriging_core import KrigingSystem
from pysole.raster import GridGeometry, load_dem
from pysole.smoothing import compute_gradients, compute_surface_curvature
from pysole.solver import Solver

SHAPE = (24, 32)
DX, DY = 10.0, 15.0
BOUNDS = (1000.0, 5000.0, 1000.0 + SHAPE[1] * DX, 5000.0 + SHAPE[0] * DY)
SLOPE_FLOOR_DEG = 5.0
WUK_DEM = Path(__file__).resolve().parent.parent / "examples" / "wuk" / "input_data" / "dgm_unt_wuk.tif"


def make_asymmetric_fixture():
    """Bottom-up grid whose DEM is deliberately not symmetric in Y (a vertical flip changes every value)."""
    geom = GridGeometry.create(SHAPE, dx=DX, dy=DY, bounds=BOUNDS)
    xx, yy = geom.meshgrid
    ex, ey = xx - BOUNDS[0], yy - BOUNDS[1]
    dem = 2000.0 + 0.25 * ex + 1.7 * ey + 40.0 * np.sin(ex / 70.0) * np.cos(ey / 90.0) + 15.0 * (ey / 360.0) ** 2
    slope = compute_gradients(dem, dx=DX, dy=DY)["slope_rad"]
    curvature = compute_surface_curvature(gaussian_filter(dem, 1.5), dx=DX, dy=DY)
    mb = -1.2 + 0.004 * (dem - 2000.0) + 0.3 * np.sin(ex / 80.0) * np.cos(ey / 100.0)
    vel = 5.0 + 18.0 * np.exp(-((ex - 160.0) ** 2 + (ey - 180.0) ** 2) / (140.0 ** 2)) + 0.02 * ey
    grids = build_drift_grids(
        RASTER_DRIFT_TERMS, geometry=geom, dem_grid=dem, opt_slope_grid=slope,
        slope_floor_deg=SLOPE_FLOOR_DEG, mass_balance_grid=mb, velocity_grid=vel,
        provided={"curvature_dem": curvature},
    )
    return geom, dem, slope, curvature, grids


def make_points(geom, dem, n=70, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(BOUNDS[0] + 25.0, BOUNDS[2] - 25.0, n)
    y = rng.uniform(BOUNDS[1] + 25.0, BOUNDS[3] - 25.0, n)
    z = 0.02 * (x - BOUNDS[0]) - 0.01 * (y - BOUNDS[1]) + 3.0 * np.sin(x / 60.0) + rng.normal(0.0, 0.05, n)
    return x, y, z


class TestDriftBasisOrientation(unittest.TestCase):
    def setUp(self):
        self.geom, self.dem, self.slope, self.curv, self.grids = make_asymmetric_fixture()

    def test_coords_to_grid_indices_removed(self):
        """The top-down (GDAL) indexer mirrored PySole's bottom-up grids and is deleted."""
        self.assertFalse(hasattr(GridGeometry, "coords_to_grid_indices"))

    def test_raster_columns_reproduce_grid_at_cell_centres(self):
        """Evaluating at every cell centre returns the standardised grid itself (no vertical mirroring)."""
        xx, yy = self.geom.meshgrid
        for term in RASTER_DRIFT_TERMS:
            with self.subTest(term=term):
                basis = DriftBasis([term], xx.ravel(), yy.ravel(), geometry=self.geom, drift_grids=self.grids)
                col = basis.evaluate(xx.ravel(), yy.ravel())[:, 0].reshape(SHAPE)
                grid = self.grids[term]
                expected = (grid - np.nanmean(grid)) / np.nanstd(grid)
                np.testing.assert_allclose(col, expected, atol=1e-9)
                self.assertGreater(float(np.max(np.abs(expected - expected[::-1, :]))), 1e-3)  # fixture is Y-asymmetric

    def test_raster_columns_match_production_sampling(self):
        """Off-node samples: same RegularGridInterpolator on (y, x) of the standardised grid as production."""
        from scipy.interpolate import RegularGridInterpolator

        x, y, _ = make_points(self.geom, self.dem)
        for term in RASTER_DRIFT_TERMS:
            with self.subTest(term=term):
                grid = self.grids[term]
                norm = (grid - np.nanmean(grid)) / np.nanstd(grid)
                reference = RegularGridInterpolator(
                    (self.geom.y_coords, self.geom.x_coords), norm, bounds_error=False, fill_value=0.0
                )(np.column_stack((y, x)))
                basis = DriftBasis([term], x, y, geometry=self.geom, drift_grids=self.grids)
                np.testing.assert_allclose(basis.evaluate(x, y)[:, 0], reference, atol=1e-9)

    @unittest.skipUnless(WUK_DEM.exists(), "WUK example DEM not available")
    def test_real_dem_cell_centres_match_geotiff(self):
        """Real north-up GeoTIFF: after load_dem the sampled z_dem equals the raster value at the same coordinates."""
        import rasterio
        from rasterio.transform import xy as transform_xy

        grid, meta = load_dem(str(WUK_DEM))
        geom = GridGeometry.create(grid.shape, dx=meta["dx"], dy=meta["dy"], bounds=meta["bounds"])
        with rasterio.open(WUK_DEM) as src:
            raw = np.array(src.read(1), dtype=np.float64)
            if src.nodata is not None:
                raw[raw == src.nodata] = np.nan
            transform = src.transform

        finite = np.isfinite(raw)
        interior = finite.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                interior &= np.roll(np.roll(finite, dr, axis=0), dc, axis=1)
        interior[[0, -1], :] = False
        interior[:, [0, -1]] = False
        rows, cols = np.nonzero(interior)
        pick = np.random.default_rng(3).choice(len(rows), size=40, replace=False)
        rows, cols = rows[pick], cols[pick]
        xs, ys = transform_xy(transform, rows, cols, offset="center")
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

        basis = DriftBasis(["z_dem"], xs, ys, geometry=geom, drift_grids={"z_dem": grid})
        sampled = basis.evaluate(xs, ys)[:, 0] * np.nanstd(grid) + np.nanmean(grid)
        np.testing.assert_allclose(sampled, raw[rows, cols], atol=1e-6)


class TestDriftBasisColumns(unittest.TestCase):
    def setUp(self):
        self.geom, self.dem, self.slope, self.curv, self.grids = make_asymmetric_fixture()
        self.x, self.y, _ = make_points(self.geom, self.dem)

    def basis(self, terms, **kwargs):
        return DriftBasis(terms, self.x, self.y, geometry=self.geom, drift_grids=self.grids, **kwargs)

    def test_quadratic_xy_has_five_columns_and_is_not_duplicated_with_linear(self):
        for terms, n_expected in ((["linear_xy"], 2), (["quadratic_xy"], 5), (["linear_xy", "quadratic_xy"], 5)):
            with self.subTest(terms=terms):
                basis = self.basis(terms)
                self.assertEqual(basis.n_columns, n_expected)
                self.assertEqual(basis.evaluate(self.x, self.y).shape, (len(self.x), n_expected))
        self.assertEqual(self.basis(["quadratic_xy"]).column_names, ["x", "y", "x^2", "y^2", "x*y"])

    def test_polynomial_columns_use_grid_coordinates(self):
        """Centring and range scaling come from the grid coordinates, as in the native engine."""
        basis = self.basis(["quadratic_xy"])
        xn = (self.x - self.geom.x_coords.mean()) / np.ptp(self.geom.x_coords)
        yn = (self.y - self.geom.y_coords.mean()) / np.ptp(self.geom.y_coords)
        expected = np.column_stack((xn, yn, xn ** 2, yn ** 2, xn * yn))
        np.testing.assert_allclose(basis.evaluate(self.x, self.y), expected, atol=1e-12)

    def test_polynomial_columns_without_geometry_use_sample_extent(self):
        """Conditioning of UTM-sized coordinates when no grid is available."""
        X = np.linspace(400000.0, 402000.0, 20)
        Y = np.linspace(5000000.0, 5002000.0, 20)
        XX, YY = np.meshgrid(X, Y)
        basis = DriftBasis(["quadratic_xy"], XX.ravel(), YY.ravel())
        self.assertLess(np.linalg.cond(basis.evaluate(XX.ravel(), YY.ravel())), 500.0)

    def test_missing_grid_raises(self):
        for term in RASTER_DRIFT_TERMS:
            with self.subTest(term=term):
                with self.assertRaises(ValueError) as ctx:
                    DriftBasis([term], self.x, self.y, geometry=self.geom, drift_grids={})
                self.assertIn(term, str(ctx.exception))
                with self.assertRaises(ValueError):
                    DriftBasis([term], self.x, self.y)
                with self.assertRaises(ValueError):
                    build_drift_grids([term], geometry=self.geom)

    def test_nan_covariates_are_imputed_with_warning(self):
        dem = self.dem.copy()
        px, py = float(self.x[0]), float(self.y[0])
        r = int(np.argmin(np.abs(self.geom.y_coords - py)))
        c = int(np.argmin(np.abs(self.geom.x_coords - px)))
        dem[r - 1:r + 2, c - 1:c + 2] = np.nan
        basis = DriftBasis(["z_dem"], self.x, self.y, geometry=self.geom, drift_grids={"z_dem": dem})
        with self.assertLogs("pysole", level="WARNING") as cm:
            values = basis.evaluate(self.x, self.y)
        self.assertTrue(any("imputing" in m for m in cm.output))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertEqual(values[0, 0], 0.0)

    def test_flat_raster_is_skipped_with_warning(self):
        with self.assertLogs("pysole", level="WARNING") as cm:
            basis = DriftBasis(["z_dem"], self.x, self.y, geometry=self.geom, drift_grids={"z_dem": np.full(SHAPE, 5.0)})
        self.assertTrue(any("zero spatial variance" in m for m in cm.output))
        self.assertEqual(basis.n_columns, 0)
        self.assertEqual(basis.evaluate(self.x, self.y).shape, (len(self.x), 0))

    def test_build_drift_grids_formulas(self):
        mb = self.grids["mass_balance"]
        vel = self.grids["velocity"]
        grids = build_drift_grids(
            ["sia", "z_dem", "curvature_dem", "mass_balance", "velocity", "velocity_sia"],
            geometry=self.geom, dem_grid=self.dem,
            opt_slope_grid=self.slope, slope_floor_deg=SLOPE_FLOOR_DEG,
            mass_balance_grid=mb, velocity_grid=vel,
        )
        np.testing.assert_array_equal(grids["z_dem"], self.dem)
        np.testing.assert_array_equal(grids["mass_balance"], mb)
        np.testing.assert_allclose(grids["velocity"], np.maximum(vel, 0.0), rtol=1e-14)
        min_sin = np.sin(np.radians(SLOPE_FLOOR_DEG))
        sin_clamped = np.maximum(np.sin(self.slope), min_sin)
        np.testing.assert_allclose(grids["sia"], 1.0 / sin_clamped, rtol=1e-14)
        np.testing.assert_allclose(
            grids["velocity_sia"], (np.maximum(vel, 0.0) ** 0.25) / (sin_clamped ** 0.75), rtol=1e-14
        )
        np.testing.assert_allclose(
            grids["curvature_dem"], compute_surface_curvature(self.dem, dx=DX, dy=DY), rtol=1e-14
        )
        provided = np.full(SHAPE, 1.0)
        kept = build_drift_grids(["curvature_dem"], geometry=self.geom, dem_grid=self.dem, provided={"curvature_dem": provided})
        self.assertIs(kept["curvature_dem"], provided)
        self.assertEqual(build_drift_grids(["linear_xy", "quadratic_xy"], geometry=self.geom), {})


class TestAnalyzerMatchesProductionEngine(unittest.TestCase):
    """KrigingSystem + DriftBasis (analyzer) must reproduce the production engine for every primitive."""

    VARIOGRAM = (150.0, 4.0, 0.0)  # range, sill, nugget

    def setUp(self):
        self.geom, self.dem, self.slope, self.curv, self.grids = make_asymmetric_fixture()
        self.x, self.y, self.z = make_points(self.geom, self.dem)
        self.xx, self.yy = self.geom.meshgrid

    def production(self, terms):
        pts = np.column_stack((self.x, self.y, self.z))
        return kriging_interpolation(
            sample_points=pts, geometry=self.geom, method="universal", variogram_model="spherical",
            variogram_params=self.VARIOGRAM, dem_grid=self.dem, opt_slope_grid=self.slope, drift_terms=list(terms),
            include_zero_boundary_condition=False, n_cores=1, show_progress=False, return_variance=False,
            slope_floor_deg=SLOPE_FLOOR_DEG,
            mass_balance_grid=self.grids.get("mass_balance"),
            velocity_grid=self.grids.get("velocity"),
            external_drift_grid={k: self.grids[k] for k in terms if k in self.grids} or None,
        ).bedrock_grid

    def analyzer(self, terms):
        basis = DriftBasis(list(terms), self.x, self.y, geometry=self.geom, drift_grids=self.grids)
        system = KrigingSystem(
            self.x, self.y, self.z, basis.evaluate(self.x, self.y), "spherical", self.VARIOGRAM,
        )
        rows = basis.evaluate(self.xx.ravel(), self.yy.ravel()).T
        estimate, _ = system.predict_points(self.xx.ravel(), self.yy.ravel(), rows if rows.shape[0] else None)
        return estimate.reshape(SHAPE)

    def test_every_primitive_and_combination(self):
        cases = [
            ["sia"], ["z_dem"], ["curvature_dem"], ["mass_balance"], ["velocity"], ["velocity_sia"],
            ["linear_xy"], ["quadratic_xy"],
            ["sia", "z_dem"], ["z_dem", "curvature_dem", "linear_xy"], ["sia", "curvature_dem", "quadratic_xy"],
            ["z_dem", "mass_balance", "velocity_sia", "linear_xy"],
            ["sia", "mass_balance", "velocity", "quadratic_xy"],
        ]
        for terms in cases:
            with self.subTest(terms=terms):
                prod, ana = self.production(terms), self.analyzer(terms)
                scale = float(np.ptp(prod))
                self.assertLess(float(np.max(np.abs(prod - ana))) / scale, 1e-9)


class TestSolverSharesDriftGrids(unittest.TestCase):
    """The Solver hands the same covariate grids to the analyzer and to the production run."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        geom, self.dem, self.slope, _, self.grids = make_asymmetric_fixture()
        self.geom = geom
        x, y, z = make_points(geom, self.dem)
        self.solver = Solver(
            dem=self.dem, bounds=geom.bounds, survey_data_type="depth",
            mass_balance=self.grids["mass_balance"], velocity=self.grids["velocity"],
            post_kriging_method="universal", post_drift_analyzer=True,
            output_dir=os.path.join(self._tmp.name, "out"),
        )
        self.solver.opt_slope = self.slope
        self.solver.opt_kc = 0.03
        pts = np.column_stack((x, y, z, np.zeros_like(x)))
        self.solver.survey_points = pts.copy()
        self.solver.migrated_points = pts.copy()

    def test_curvature_is_the_smoothed_curvature_at_opt_kc(self):
        grids = self.solver.get_drift_grids(["curvature_dem", "velocity", "mass_balance"])
        np.testing.assert_array_equal(grids["curvature_dem"], self.solver.get_smoothed_curvature(0.03))
        np.testing.assert_array_equal(grids["velocity"], self.solver.get_smoothed_velocity(0.03))
        np.testing.assert_array_equal(grids["mass_balance"], self.grids["mass_balance"])
        raw = compute_surface_curvature(self.dem, dx=DX, dy=DY)
        self.assertGreater(float(np.max(np.abs(grids["curvature_dem"] - raw))), 1e-9)

    def test_analyzer_and_production_receive_identical_grids(self):
        fake = KrigingResult(bedrock_grid=np.zeros(SHAPE), variance_grid=np.zeros(SHAPE))
        selected = SimpleNamespace(terms=["curvature_dem", "sia", "mass_balance", "velocity"], aicc=0.0)
        with patch("pysole.solver.DriftAnalyzer") as mock_analyzer, \
                patch("pysole.solver.kriging_interpolation", return_value=fake) as mock_kriging:
            mock_analyzer.return_value.run_diagnostics.return_value = [selected]
            self.solver._execute_kriging_pass(
                target_type="D", points=self.solver.migrated_points, krig_method="universal",
                var_model="spherical", zero_boundary=False,
                pass_name="Pass 2 - Constructing Depth Grid",
            )
        analyzer_kwargs = mock_analyzer.return_value.run_diagnostics.call_args.kwargs
        production_grids = mock_kriging.call_args.kwargs["external_drift_grid"]
        self.assertIs(analyzer_kwargs["geometry"], self.solver.geometry)
        self.assertEqual(set(analyzer_kwargs["drift_grids"]), set(RASTER_DRIFT_TERMS))
        self.assertEqual(set(production_grids), {"curvature_dem", "sia", "mass_balance", "velocity"})
        for term, grid in production_grids.items():
            np.testing.assert_array_equal(grid, analyzer_kwargs["drift_grids"][term])


if __name__ == "__main__":
    unittest.main()
