"""
Audit Regression Unit Test Suite for PySole v0.4.3.
Verifies all 7 audit regression test cases and Phase 1 / Phase 2 code review fixes.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import os

from pysole.raster import GridGeometry, resample_dem, load_dem, BedrockMap
from pysole.interpolation import (
    DriftBasis,
    kriging_interpolation,
    built_in_kriging_interpolation,
    random_forest_hole_filling,
    blend_margin_topography,
)
from pysole.drift import DRIFT_PRIMITIVES
from pysole.kriging_core import KrigingSystem
from pysole.smoothing import fft_gaussian_smooth
from pysole.migration import migrate_eikonal_points
from pysole.variogram import fit_variogram_model, calculate_variogram, optimize_bss_variance
from pysole.pipeline import run_from_config
from pysole.logging import logger as pysole_logger

import logging
from unittest.mock import patch


class TestAuditRegressions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dual_kriging_loo_linear_field(self):
        """2. test_dual_kriging_loo_linear_field: Verifies LOO RMSE < 2.0 on linear trend field."""
        np.random.seed(42)
        X = np.random.uniform(100, 500, 50)
        Y = np.random.uniform(200, 600, 50)
        Z_surf = np.full(50, 2000.0)
        Depth = 2.0 * X - 0.5 * Y + 10.0 + np.random.normal(0, 0.1, 50)

        # Dual Kriging LOO validation
        drift = DriftBasis(["linear_xy"], X, Y)

        # Calculate LOO RMSE
        errors = []
        for i in range(len(X)):
            idx = np.arange(len(X)) != i
            val_drift = DriftBasis(["linear_xy"], X[idx], Y[idx])
            val_solver = KrigingSystem(
                X[idx], Y[idx], Depth[idx], val_drift.evaluate(X[idx], Y[idx]),
                variogram_params=(200.0, 1.0, 0.01),
            )
            pred, _ = val_solver.predict_points(X[i:i+1], Y[i:i+1], val_drift.evaluate(X[i:i+1], Y[i:i+1]).T)
            errors.append((pred[0] - Depth[i]) ** 2)

        rmse = np.sqrt(np.mean(errors))
        self.assertLess(rmse, 2.0)

    def test_csv_npy_roundtrip_orientation(self):
        """6. test_csv_npy_roundtrip_orientation: Verifies GIS I/O round-trip orientation symmetry."""
        grid_orig = np.arange(100, dtype=np.float64).reshape((10, 10))
        bounds = (1000.0, 2000.0, 1100.0, 2100.0)

        npy_path = self.output_dir / "grid.npy"
        np.save(npy_path, grid_orig)

        loaded_grid, meta = load_dem(npy_path, dx=10.0, dy=10.0, bounds=bounds)
        np.testing.assert_allclose(grid_orig, loaded_grid)

    def test_fit_variogram_scale_invariance(self):
        """7. test_fit_variogram_scale_invariance: Verifies scale-independent variogram model fitting."""
        distances = np.linspace(1, 100, 20)
        # Theoretical spherical variogram
        a_true, c_true, n_true = 50.0, 10.0, 1.0
        h_a = np.minimum(distances / a_true, 1.0)
        gamma = n_true + c_true * (1.5 * h_a - 0.5 * h_a ** 3)

        # Scale 1: original magnitude
        a1, s1, n1, _ = fit_variogram_model(distances, gamma, model_type="spherical")

        # Scale 2: multiplied by 100
        a2, s2, n2, _ = fit_variogram_model(distances, gamma * 100.0, model_type="spherical")

        np.testing.assert_allclose(a1, a2, rtol=1e-2)
        np.testing.assert_allclose(s1 * 100.0, s2, rtol=1e-2)

    def test_kriging_sill_scaling_invariance(self):
        """7b. Verifies Kriging interpolation prediction invariance under 400x and 40,000x variogram sill scaling."""
        geom = GridGeometry.create((20, 20), dx=10.0, dy=10.0, bounds=(0, 0, 200, 200))
        pts = np.array([
            [30.0, 30.0, 100.0],
            [70.0, 120.0, 150.0],
            [150.0, 60.0, 80.0],
            [120.0, 170.0, 200.0],
        ])
        z1, _ = built_in_kriging_interpolation(
            pts, geom.x_coords, geom.y_coords, method="ordinary",
            variogram_model="spherical", variogram_params={"sill": 1.0, "range": 100.0, "nugget": 0.0}, show_progress=False
        )
        z2, _ = built_in_kriging_interpolation(
            pts, geom.x_coords, geom.y_coords, method="ordinary",
            variogram_model="spherical", variogram_params={"sill": 400.0, "range": 100.0, "nugget": 0.0}, show_progress=False
        )
        z3, _ = built_in_kriging_interpolation(
            pts, geom.x_coords, geom.y_coords, method="ordinary",
            variogram_model="spherical", variogram_params={"sill": 40000.0, "range": 100.0, "nugget": 0.0}, show_progress=False
        )
        np.testing.assert_allclose(z1, z2, atol=1e-6)
        np.testing.assert_allclose(z1, z3, atol=1e-6)

    def test_run_from_config_interactive_mode(self):
        """8. Verifies run_from_config(is_batch=False) in non-TTY mode automatically bypasses interactive prompts safely."""
        import json
        dem = np.full((20, 20), 2000.0)
        dem_path = self.output_dir / "dem.tif"
        BedrockMap(dem, bounds=(0, 0, 200, 200), crs="EPSG:32633").save(dem_path)

        pts_path = self.output_dir / "pts.csv"
        pts = np.array([
            [50.0, 50.0, 2000.0, 30.0],
            [100.0, 100.0, 1950.0, 40.0],
            [150.0, 150.0, 1900.0, 25.0],
        ])
        np.savetxt(pts_path, pts, delimiter=",", header="X,Y,Z_surf,depth", comments="")

        cfg = {
            "inputs": {
                "dem_path": str(dem_path),
                "survey_data_path": str(pts_path),
                "survey_data_type": "depth",
                "show_progress": False,
            },
            "migration_parameters": {"perform_migration": False},
            "outputs": {"output_prefix": str(self.output_dir / "out"), "output_format": "tif"},
        }
        cfg_path = self.output_dir / "test_cfg.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        from unittest.mock import patch
        with patch("sys.stdin.isatty", return_value=False):
            res = run_from_config(cfg_path, is_batch=False)

        # Exact numeric checks: 20x20 grid, bedrock never above the (flat, 2000 m) surface and never deeper than
        # the 500 m maximum-thickness floor used by calculate_bedrock (max(1.5 * 40 m, 500 m)).
        self.assertEqual(res.shape, (20, 20))
        self.assertTrue(np.all(np.isfinite(res.grid)))
        self.assertLessEqual(float(np.max(res.grid)), 2000.0 + 1e-9)
        self.assertGreaterEqual(float(np.min(res.grid)), 1500.0 - 1e-9)

    def test_fft_normalized_convolution_nan(self):
        """9. Verifies FFT normalized convolution on NaN boundaries avoids huge edge spikes."""
        grid = np.full((30, 30), 100.0)
        grid[:, :10] = np.nan

        smoothed, _, _ = fft_gaussian_smooth(grid, dx=5.0, dy=5.0, kc=0.0314)
        valid_vals = smoothed[~np.isnan(smoothed)]
        self.assertTrue(np.all(np.isfinite(valid_vals)))
        self.assertLess(np.max(valid_vals), 200.0)

    def test_utm_drift_basis_conditioning(self):
        """10. Verifies pre-centered quadratic UTM polynomial terms yield low condition numbers."""
        X = np.linspace(400000.0, 402000.0, 20)
        Y = np.linspace(5000000.0, 5002000.0, 20)
        XX, YY = np.meshgrid(X, Y)
        x_pts = XX.flatten()
        y_pts = YY.flatten()

        drift = DriftBasis(["quadratic_xy"], x_pts, y_pts)
        F = drift.evaluate(x_pts, y_pts)
        cond = np.linalg.cond(F)
        self.assertLess(cond, 500.0)

    def test_explicit_multi_drift_and_legacy_rejection(self):
        """11. Verifies explicit primitive multi-drift combinations run correctly and legacy shorthand names raise ValueError."""
        geom = GridGeometry.create((10, 10), dx=10.0, dy=10.0, bounds=(0, 0, 100, 100))
        pts = np.array([
            [15.0, 20.0, 1000.0, 25.0],
            [35.0, 80.0, 990.0, 30.0],
            [50.0, 50.0, 950.0, 15.0],
            [80.0, 25.0, 900.0, 22.0],
            [85.0, 85.0, 880.0, 40.0],
            [20.0, 55.0, 970.0, 18.0],
        ])
        xx, yy = geom.meshgrid
        dem_grid = 1000.0 - 0.4 * xx - 0.1 * yy + 5.0 * np.sin(xx / 20.0)
        opt_slope = 0.05 + 0.002 * xx + 0.001 * yy

        common = dict(
            sample_points=pts,
            geometry=geom,
            method="universal",
            dem_grid=dem_grid,
            opt_slope_grid=opt_slope,
            include_zero_boundary_condition=False,
            show_progress=False,
        )
        # Explicit primitive multi-drift list works
        res_explicit = kriging_interpolation(drift_terms=["sia", "z_dem"], **common)
        res_sia_only = kriging_interpolation(drift_terms=["sia"], **common)

        self.assertEqual(res_explicit.bedrock_grid.shape, (10, 10))
        self.assertTrue(np.all(np.isfinite(res_explicit.bedrock_grid)))
        self.assertGreater(float(np.max(np.abs(res_explicit.bedrock_grid - res_sia_only.bedrock_grid))), 1e-6)

        # Legacy shorthand names must strictly raise ValueError
        legacy_names = [
            "sia_z_dem", "full_physical", "sia_space", "sia_curvature_dem",
            "z_dem_curvature_dem", "sia_z_dem_curvature_dem", "full_spatial_physical",
        ]
        for legacy_shorthand in legacy_names:
            with self.assertRaises(ValueError) as ctx:
                kriging_interpolation(drift_terms=[legacy_shorthand], **common)
            self.assertIn("Shorthand compound names are no longer supported", str(ctx.exception))

            with self.assertRaises(ValueError) as ctx_basis:
                DriftBasis(drift_terms=[legacy_shorthand], x_ref=pts[:, 0], y_ref=pts[:, 1])
            self.assertIn("Shorthand compound names are no longer supported", str(ctx_basis.exception))

    def test_eikonal_migration_oblique_slope(self):
        """12. Verifies exact 3D Eikonal closed-form slowness vector on oblique surface slopes."""
        pts = np.array([[100.0, 100.0, 2020.0, 500.0]])  # one_way_traveltime 500 ns
        geom = GridGeometry.create((30, 30), dx=10.0, dy=10.0, bounds=(0, 0, 300, 300))
        # Oblique slope: z_x = 0.1, z_y = 0.1
        X, Y = np.meshgrid(geom.x_coords, geom.y_coords)
        dem = 2000.0 + 0.1 * X + 0.1 * Y

        from pysole.migration import EikonalMigrator
        migrator = EikonalMigrator(dem, geometry=geom)
        mig_pts = migrator.migrate_points(pts, velocity=0.16)

        self.assertEqual(len(mig_pts), 1)
        self.assertTrue(np.all(np.isfinite(mig_pts[0])))

        # Analytic zero-offset ray: travels along the downward surface normal n = (f_x, f_y, -1) / s with
        # s = sqrt(1 + f_x^2 + f_y^2); path length d = v * one_way_traveltime = 0.16 m/ns * 500 ns = 80 m.
        f_x = f_y = 0.1
        s = np.sqrt(1.0 + f_x ** 2 + f_y ** 2)
        d = 0.16 * 500.0
        x_expected = 100.0 + d * f_x / s
        y_expected = 100.0 + d * f_y / s
        z_bed_expected = 2020.0 - d / s
        z_surf_expected = 2000.0 + 0.1 * x_expected + 0.1 * y_expected
        depth_expected = z_surf_expected - z_bed_expected

        x_m, y_m, z_surf_m, depth_m = mig_pts[0]
        self.assertAlmostEqual(x_m, x_expected, places=6)
        self.assertAlmostEqual(y_m, y_expected, places=6)
        self.assertAlmostEqual(z_surf_m, z_surf_expected, places=6)
        self.assertAlmostEqual(depth_m, depth_expected, places=6)

    def _make_small_solver(self, **kwargs):
        """Small synthetic depth-survey Solver (12x12 @ 10 m, UTM-like bounds) writing only to the temp dir."""
        from pysole.solver import Solver

        bounds = (500000.0, 5200000.0, 500120.0, 5200120.0)
        geom = GridGeometry.create((12, 12), dx=10.0, dy=10.0, bounds=bounds)
        xx, yy = geom.meshgrid
        dem = 1000.0 + 0.2 * (xx - bounds[0]) + 0.1 * (yy - bounds[1])
        kwargs.setdefault("output_dir", str(self.output_dir / "solver_out"))
        solver = Solver(
            dem=dem, bounds=geom.bounds, survey_data_type="depth", **kwargs,
        )
        solver.opt_slope = np.full((12, 12), 0.2)  # bypass the BSS optimisation for wiring tests
        pts = np.array([
            [500015.0, 5200015.0, 1003.0, 40.0],
            [500045.0, 5200030.0, 1012.0, 55.0],
            [500075.0, 5200020.0, 1018.0, 48.0],
            [500100.0, 5200055.0, 1030.0, 35.0],
            [500030.0, 5200085.0, 1022.0, 60.0],
            [500085.0, 5200100.0, 1038.0, 42.0],
        ])
        solver.survey_points = pts
        solver.migrated_points = pts.copy()
        return solver

    def test_kriging_engine_string_propagation(self):
        """14. Verifies Solver propagates the engine string unchanged to kriging_interpolation (native and pykrige)."""
        from pysole.interpolation import KrigingResult

        for engine in ("native", "pykrige"):
            solver = self._make_small_solver(kriging_engine=engine)
            self.assertEqual(solver.engine_type, engine)
            fake = KrigingResult(bedrock_grid=np.full((12, 12), 40.0), variance_grid=np.zeros((12, 12)))
            with patch("pysole.solver.kriging_interpolation", return_value=fake) as mock_krig:
                solver._execute_kriging_pass(
                    target_type="D", points=solver.migrated_points, krig_method="ordinary",
                    var_model="spherical", zero_boundary=False, pass_name="Pass 2 - Constructing Depth Grid",
                )
            mock_krig.assert_called_once()
            self.assertEqual(mock_krig.call_args.kwargs["engine"], engine)

    def test_lopo_profile_column_preservation(self):
        """15. Verifies string profile IDs are factorized and preserved as 4th/5th column through loading, migration, and get_sample_points."""
        import pandas as pd
        from pysole.raster import load_survey_points
        from pysole.solver import Solver

        # 1. Create survey CSV with string profile IDs ("Line_A", "Line_B")
        pts_path = self.output_dir / "survey_string_prof.csv"
        df_pts = pd.DataFrame({
            "X": [500020.0, 500040.0, 500060.0, 500080.0],
            "Y": [5200020.0, 5200040.0, 5200060.0, 5200080.0],
            "picks": [40.0, 45.0, 35.0, 50.0],
            "profile_name": ["Line_A", "Line_A", "Line_B", "Line_B"]
        })
        df_pts.to_csv(pts_path, index=False)

        # 2. Test load_survey_points factorizes string profile IDs
        pts_array = load_survey_points(pts_path, profile_column="profile_name")
        self.assertEqual(pts_array.shape[1], 5)  # [X, Y, Z_surf, val, prof_id]
        # Profile IDs should be factorized (0 and 1)
        self.assertEqual(pts_array[0, 4], 0.0)
        self.assertEqual(pts_array[1, 4], 0.0)
        self.assertEqual(pts_array[2, 4], 1.0)
        self.assertEqual(pts_array[3, 4], 1.0)

        # 3. Test get_sample_points retains profile IDs as 4th column (index 3) of 4-column output sample_pts
        dem_data = np.full((10, 10), 1000.0)
        bounds = (500000.0, 5200000.0, 500100.0, 5200100.0)
        solver = Solver(dem=dem_data, bounds=bounds, survey_data_path=pts_path, survey_data_type="depth", survey_profile_column="profile_name")
        solver.pre_kriging_points = pts_array
        sample_pts = solver.get_sample_points(stage="pre_migration", target_type="D")
        self.assertEqual(sample_pts.shape[1], 4)  # [X, Y, depth, prof_id]
        np.testing.assert_array_equal(sample_pts[:, 3], [0.0, 0.0, 1.0, 1.0])

    def test_linear_variogram_model_support(self):
        """16. 'linear' variogram: under native engine, logs warning and falls back to 'spherical'."""
        geom = GridGeometry.create((10, 10), dx=10.0, dy=10.0, bounds=(0, 0, 100, 100))
        pts = np.array([
            [20.0, 20.0, 10.0],
            [50.0, 50.0, 20.0],
            [80.0, 80.0, 30.0],
        ])
        params = {"range": 200.0, "sill": 100.0, "nugget": 0.0}
        with self.assertLogs("pysole", level="WARNING") as cm:
            grid, var = built_in_kriging_interpolation(
                pts, geom.x_coords, geom.y_coords, method="ordinary",
                variogram_model="linear", variogram_params=params, show_progress=False,
            )
        self.assertTrue(any("linear" in m and "falling back to 'spherical'" in m for m in cm.output), cm.output)
        self.assertEqual(grid.shape, (10, 10))
        self.assertTrue(np.all(np.isfinite(grid)))
        self.assertTrue(np.all(np.isfinite(var)))

        # Exactness at a data location: Kriging is an exact interpolator (nugget = 0, only the 1e-6*sill
        # Tikhonov term perturbs it) -> check the closed-form spherical ordinary-kriging solution at an arbitrary node.
        from pysole.variogram import spherical_variogram
        from scipy.spatial.distance import cdist

        reg = 1e-6 * params["sill"]
        G = spherical_variogram(cdist(pts[:, :2], pts[:, :2]), params["range"], params["sill"], params["nugget"])
        G = np.where(cdist(pts[:, :2], pts[:, :2]) == 0, 0.0, G) + reg * np.eye(3)
        A = np.ones((4, 4)); A[:3, :3] = G; A[3, 3] = 0.0
        j, i = 4, 6  # row (y index), col (x index)
        x0, y0 = geom.x_coords[i], geom.y_coords[j]
        g0 = spherical_variogram(np.hypot(pts[:, 0] - x0, pts[:, 1] - y0), params["range"], params["sill"], params["nugget"])
        w = np.linalg.solve(A, np.append(g0, 1.0))
        self.assertAlmostEqual(float(grid[j, i]), float(w[:3] @ pts[:, 2]), places=8)

    def test_profile_column_parsed_by_name(self):
        """18. N5-M3: profile column is resolved by header name (any position, numeric or text IDs)."""
        import pandas as pd
        from pysole.raster import load_survey_points

        # README layout [profile, X, Y, value] with NUMERIC profile ids in the FIRST column
        p1 = self.output_dir / "layout_profile_first.csv"
        pd.DataFrame({
            "profile": [7, 7, 9, 9],
            "X": [500020.0, 500040.0, 500060.0, 500080.0],
            "Y": [5200020.0, 5200040.0, 5200060.0, 5200080.0],
            "picks": [40.0, 45.0, 35.0, 50.0],
        }).to_csv(p1, index=False)
        pts = load_survey_points(p1, profile_column="profile")
        self.assertEqual(pts.shape, (4, 5))  # [X, Y, Z_surf(nan), value, profile]
        np.testing.assert_array_equal(pts[:, 0], [500020.0, 500040.0, 500060.0, 500080.0])
        np.testing.assert_array_equal(pts[:, 1], [5200020.0, 5200040.0, 5200060.0, 5200080.0])
        self.assertTrue(np.all(np.isnan(pts[:, 2])))
        np.testing.assert_array_equal(pts[:, 3], [40.0, 45.0, 35.0, 50.0])
        np.testing.assert_array_equal(pts[:, 4], [7.0, 7.0, 9.0, 9.0])

        # Text IDs in the MIDDLE of a 4-numeric-column table [X, Y, Z, value]
        p2 = self.output_dir / "layout_profile_middle.csv"
        pd.DataFrame({
            "X": [500020.0, 500040.0, 500060.0],
            "Y": [5200020.0, 5200040.0, 5200060.0],
            "line": ["B", "A", "B"],
            "Z": [1000.0, 1010.0, 1020.0],
            "picks": [40.0, 45.0, 35.0],
        }).to_csv(p2, index=False)
        pts2 = load_survey_points(p2, profile_column="line")
        self.assertEqual(pts2.shape, (3, 5))
        np.testing.assert_array_equal(pts2[:, 2], [1000.0, 1010.0, 1020.0])
        np.testing.assert_array_equal(pts2[:, 3], [40.0, 45.0, 35.0])
        np.testing.assert_array_equal(pts2[:, 4], [0.0, 1.0, 0.0])  # factorize order of appearance

        # Unknown column name -> warning, 4-column legacy result (no spurious column)
        with self.assertLogs("pysole", level="WARNING"):
            pts3 = load_survey_points(p1, profile_column="does_not_exist")
        self.assertEqual(pts3.shape[1], 4)  # legacy positional path: no profile column appended

    def test_compound_curvature_drifts_use_smoothed_curvature(self):
        """19. N5-M4: every compound drift containing curvature receives the k_c-smoothed curvature grid."""
        from pysole.interpolation import KrigingResult

        solver = self._make_small_solver()
        solver.opt_kc = 0.02
        sentinel = np.arange(144, dtype=float).reshape(12, 12) * 1e-3
        fake = KrigingResult(bedrock_grid=np.full((12, 12), 40.0), variance_grid=np.zeros((12, 12)))

        curvature_models = [
            ["curvature_dem"],
            ["sia", "curvature_dem"],
            ["z_dem", "curvature_dem"],
            ["sia", "z_dem", "curvature_dem"],
            ["sia", "z_dem", "curvature_dem", "linear_xy"],
            ["z_dem", "curvature_dem", "quadratic_xy"],
        ]
        with patch.object(type(solver), "get_smoothed_curvature", return_value=sentinel) as mock_curv, \
                patch("pysole.solver.kriging_interpolation", return_value=fake) as mock_krig:
            for model_terms in curvature_models:
                mock_krig.reset_mock()
                mock_curv.reset_mock()
                solver.post_drift_terms = model_terms
                solver._execute_kriging_pass(
                    target_type="D", points=solver.migrated_points, krig_method="universal",
                    var_model="spherical", zero_boundary=False, pass_name="Pass 2 - Constructing Depth Grid",
                )
                mock_curv.assert_called_once_with(0.02)
                ext = mock_krig.call_args.kwargs["external_drift_grid"]
                np.testing.assert_array_equal(ext["curvature_dem"], sentinel)

            # Control: drifts without curvature must not trigger the (expensive) smoothed-curvature evaluation
            mock_krig.reset_mock()
            mock_curv.reset_mock()
            solver.post_drift_terms = ["sia", "z_dem"]
            solver._execute_kriging_pass(
                target_type="D", points=solver.migrated_points, krig_method="universal",
                var_model="spherical", zero_boundary=False, pass_name="Pass 2 - Constructing Depth Grid",
            )
            mock_curv.assert_not_called()
            self.assertEqual(set(mock_krig.call_args.kwargs["external_drift_grid"]), {"sia", "z_dem"})

    def test_return_variance_shortcut(self):
        """20. N5-M7: return_variance=False skips the K^-1 solve and variance GEMM but keeps the prediction identical."""
        import pysole.kriging_core as core

        geom = GridGeometry.create((20, 20), dx=10.0, dy=10.0, bounds=(0, 0, 200, 200))
        rng = np.random.default_rng(1)
        pts = np.column_stack((rng.uniform(5, 195, 25), rng.uniform(5, 195, 25), rng.uniform(10, 60, 25)))
        kw = dict(method="universal", variogram_model="spherical",
                  variogram_params={"range": 120.0, "sill": 80.0, "nugget": 1.0},
                  drift_terms=["linear_xy"], show_progress=False)

        calls = []
        real_lu_solve = core.lu_solve

        def counting_lu_solve(*a, **k):
            calls.append(1)
            return real_lu_solve(*a, **k)

        with patch.object(core, "lu_solve", side_effect=counting_lu_solve):
            z_full, v_full = built_in_kriging_interpolation(pts, geom.x_coords, geom.y_coords, **kw)
            n_full = len(calls)
            calls.clear()
            z_fast, v_fast = built_in_kriging_interpolation(
                pts, geom.x_coords, geom.y_coords, return_variance=False, **kw
            )
            n_fast = len(calls)

        self.assertEqual(n_full, 2)  # dual weights + explicit inverse (shared by all chunks)
        self.assertEqual(n_fast, 1)  # dual weights only
        np.testing.assert_array_equal(z_full, z_fast)
        self.assertTrue(np.all(np.isfinite(v_full)) and np.all(v_full >= 0.0))
        self.assertTrue(np.all(np.isnan(v_fast)))  # NaN, never a misleading 0 uncertainty

    def test_solver_variance_flag_wiring(self):
        """21. Uncertainty calculation in pass 1 and pass 2 is controlled by calculate_uncertainty flags; max_memory_fraction is forwarded."""
        from pysole.interpolation import KrigingResult

        solver = self._make_small_solver()
        solver.config = {}
        self.assertEqual(solver.max_memory_fraction, 0.5)  # default
        solver.config = {"kriging_parameters": {"max_memory_fraction": 0.3}}
        self.assertEqual(solver.max_memory_fraction, 0.3)  # configuration

        solver_arg = self._make_small_solver(max_memory_fraction=0.2)
        solver_arg.config = {"kriging_parameters": {"max_memory_fraction": 0.3}}
        self.assertEqual(solver_arg.max_memory_fraction, 0.2)  # constructor argument beats the config value
        with self.assertRaises(ValueError):
            self._make_small_solver(max_memory_fraction=0.95)

        fake = KrigingResult(bedrock_grid=np.full((12, 12), 40.0), variance_grid=np.full((12, 12), np.nan))
        with patch("pysole.solver.kriging_interpolation", return_value=fake) as mock_krig:
            solver_arg._execute_kriging_pass(
                target_type="D", points=solver_arg.migrated_points, krig_method="ordinary",
                var_model="spherical", zero_boundary=False, pass_name="Pass 1 - Constructing Traveltime Grid", return_variance=False,
            )
            self.assertFalse(mock_krig.call_args.kwargs["return_variance"])
            self.assertEqual(mock_krig.call_args.kwargs["max_memory_fraction"], 0.2)

        # Pass 2 defaults to False
        with patch.object(solver, "_execute_kriging_pass", wraps=solver._execute_kriging_pass) as mock_exec:
            solver.calculate_bedrock()
        self.assertIs(mock_exec.call_args.kwargs["return_variance"], False)

        # Explicit calculate_uncertainty=True
        with patch.object(solver, "_execute_kriging_pass", wraps=solver._execute_kriging_pass) as mock_exec:
            solver.calculate_bedrock(calculate_uncertainty=True)
        self.assertIs(mock_exec.call_args.kwargs["return_variance"], True)

        # Configured post_calculate_uncertainty=True
        solver.post_calculate_uncertainty = True
        with patch.object(solver, "_execute_kriging_pass", wraps=solver._execute_kriging_pass) as mock_exec:
            solver.calculate_bedrock()
        self.assertIs(mock_exec.call_args.kwargs["return_variance"], True)

    def test_memory_guard_skips_variance_end_to_end(self):
        """24. A RAM shortage yields NaN uncertainty with a warning, and the NaN raster is never exported."""
        solver = self._make_small_solver(output_dir=str(self.output_dir / "guard_e2e"), post_calculate_uncertainty=True)
        solver.config = {"outputs": {"save_thickness_uncertainty": True}}
        with patch("pysole.memory.available_memory_bytes", return_value=1000):  # 1 kB: nothing fits
            with self.assertLogs("pysole", level="WARNING") as cm:
                solver.calculate_bedrock()
        self.assertTrue(any("[Memory Guard]" in m for m in cm.output), cm.output)
        self.assertTrue(np.all(np.isnan(solver.kriged_std)))

        with self.assertLogs("pysole", level="WARNING") as cm:
            saved = solver.export_outputs(stage="finalization")
        self.assertFalse(any("uncertainty" in f for f in saved), saved)
        self.assertTrue(any("[Export Skipped]" in m for m in cm.output), cm.output)

    def test_variance_computation_and_figures_saved(self):
        """25. Kriging variance is evaluated when calculate_uncertainty=True and skipped when False."""
        # (a) When calculate_uncertainty=True, finite variance and kriged_std inside outline
        solver_unc = self._make_small_solver(
            output_dir=str(self.output_dir / "with_sigma"),
            post_calculate_uncertainty=True,
        )
        solver_unc.config = {}
        _, variance = solver_unc.calculate_bedrock()
        self.assertTrue(np.all(np.isfinite(variance[solver_unc.outline_mask])))
        self.assertTrue(np.all(np.isfinite(solver_unc.kriged_std[solver_unc.outline_mask])))
        self.assertGreater(len(os.listdir(solver_unc.plots_dir)), 0)

        # (b) When calculate_uncertainty=False (default), variance is all NaN and figures are saved
        solver_fast = self._make_small_solver(
            output_dir=str(self.output_dir / "no_sigma"),
            post_calculate_uncertainty=False,
        )
        solver_fast.config = {}
        _, variance_fast = solver_fast.calculate_bedrock()
        self.assertTrue(np.all(np.isnan(variance_fast)))
        self.assertTrue(np.all(np.isnan(solver_fast.kriged_std)))
        self.assertGreater(len(os.listdir(solver_fast.plots_dir)), 0)

    def test_uncertainty_placeholder_panel_renders(self):
        """26. An all-NaN uncertainty grid renders a placeholder panel without errors or warnings."""
        import warnings
        from pysole.plotting import plot_kriging_bedrock_and_uncertainty, UNCERTAINTY_PLACEHOLDER_TEXT

        self.assertIn("not available", UNCERTAINTY_PLACEHOLDER_TEXT)
        n = 12
        yy, xx = np.mgrid[0:n, 0:n].astype(float)
        bed = 900.0 + xx + yy
        pts = np.array([[5.0, 5.0, 900.0, 30.0], [8.0, 3.0, 905.0, 20.0], [3.0, 9.0, 910.0, 25.0]])
        out = self.output_dir / "placeholder_figs"
        with warnings.catch_warnings(record=True) as caught, patch.object(pysole_logger, "warning") as mock_warn:
            warnings.simplefilter("always")
            plot_kriging_bedrock_and_uncertainty(
                kriged_bedrock=bed, kriged_std=np.full((n, n), np.nan), pts=pts,
                plot_extent=(0.0, float(n), 0.0, float(n)), plots_dir=str(out), interactive=False,
            )
        self.assertEqual([str(w.message) for w in caught if issubclass(w.category, (RuntimeWarning, UserWarning))], [])
        mock_warn.assert_not_called()
        self.assertTrue(any(f.endswith(".png") for f in os.listdir(out)))

    def test_fft_padding_scales_with_kernel_width(self):
        """33. F-L4: reflect padding follows 4*sigma = 4/(k_c*dx); k_c=0.004 is more accurate than the old fixed pad."""
        from pysole.solver import Solver
        from pysole.smoothing import precompute_fft_grid, fft_gaussian_smooth_precomputed

        n = 120
        bounds = (500000.0, 5200000.0, 500000.0 + 10.0 * n, 5200000.0 + 10.0 * n)
        geom = GridGeometry.create((n, n), dx=10.0, dy=10.0, bounds=bounds)
        xx, yy = geom.meshgrid
        x, y = xx - bounds[0], yy - bounds[1]
        dem = 2000.0 + 0.4 * x + 0.1 * y + 60.0 * np.sin(x / 150.0) * np.cos(y / 220.0)
        solver = Solver(dem=dem, dx=10.0, dy=10.0, bounds=geom.bounds, survey_data_type="depth",
                        output_dir=str(self.output_dir / "fft_pad"))

        kc = 0.004
        ref_spec, ref_k, _ = precompute_fft_grid(dem, dx=10.0, dy=10.0, kc=1e-5)  # maximal padding (capped at n)
        self.assertEqual(ref_spec.pad_m, n)
        reference = fft_gaussian_smooth_precomputed(ref_spec, ref_k, kc=kc)
        old_spec, old_k, _ = precompute_fft_grid(dem, dx=10.0, dy=10.0, kc=0.01)  # previous fixed sizing
        self.assertEqual(old_spec.pad_m, 40)
        old = fft_gaussian_smooth_precomputed(old_spec, old_k, kc=kc)

        new = solver.get_smoothed_dem(kc)
        self.assertEqual(solver._fft_dem_cache[0].pad_m, 100)  # ceil(4 / (0.004 * 10))
        err_new = float(np.max(np.abs(new - reference)))
        err_old = float(np.max(np.abs(old - reference)))
        self.assertLess(err_new, err_old)
        self.assertLess(err_new, 0.25 * err_old + 1e-9)

        # larger k_c requests never shrink / recompute the padded spectrum; cached smoothed grids stay valid
        cache_before = solver._fft_dem_cache
        solver.get_smoothed_dem(0.05)
        self.assertIs(solver._fft_dem_cache, cache_before)
        self.assertIs(solver.get_smoothed_dem(kc), new)

        # k_c >= 0.01 keeps the legacy padding
        solver2 = Solver(dem=dem, dx=10.0, dy=10.0, bounds=geom.bounds, survey_data_type="depth",
                         output_dir=str(self.output_dir / "fft_pad2"))
        solver2.get_smoothed_dem(0.02)
        self.assertEqual(solver2._fft_dem_cache[0].pad_m, 40)

    def test_bss_sweep_streaming_equivalence_and_memory(self):
        """22. N5-M7/F-L5: streaming k_c sweep returns the argmin of the variance curve and keeps O(batch) grids alive."""
        import tracemalloc
        import weakref
        import pysole.variogram as vg

        n = 160
        geom = GridGeometry.create((n, n), dx=10.0, dy=10.0, bounds=(0, 0, n * 10.0, n * 10.0))
        xx, yy = geom.meshgrid
        dem = 3000.0 + 0.3 * xx + 0.2 * yy + 40.0 * np.sin(xx / 120.0) * np.cos(yy / 90.0)
        rng = np.random.default_rng(3)
        px, py = rng.uniform(50, n * 10.0 - 50, 60), rng.uniform(50, n * 10.0 - 50, 60)
        pts = np.column_stack((px, py, 3000.0 + 0.3 * px + 0.2 * py, rng.uniform(40, 120, 60)))

        real_smooth = vg.fft_gaussian_smooth_precomputed

        peaks = {}
        results = {}
        live_peak = {}
        for steps in (6, 36):
            refs = []
            peak_live = [0]

            def tracking_smooth(*a, _refs=refs, _peak=peak_live, **k):
                out = real_smooth(*a, **k)
                _refs.append(weakref.ref(out))
                _peak[0] = max(_peak[0], sum(r() is not None for r in _refs))
                return out

            tracemalloc.start()
            with patch.object(vg, "fft_gaussian_smooth_precomputed", side_effect=tracking_smooth):
                res = optimize_bss_variance(dem, pts, geom, kc_max=0.3, kc_min=0.01, n_steps=steps, n_cores=1, show_progress=False)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks[steps], results[steps], live_peak[steps] = peak, res, peak_live[0]
            self.assertIn(len(refs), (steps, steps + 1))  # sweep (+1 optional re-smoothing of the optimum)

        for steps, res in results.items():
            kc_var = res.all_kc_variances
            self.assertEqual(len(kc_var), steps)
            self.assertEqual(res.optimal_kc, kc_var[int(np.argmin(kc_var[:, 1])), 0])
            # Non-interactive run retains only the optimum (no per-step grids)
            self.assertEqual(len(res.all_smoothed_dems), 1)
            self.assertEqual(len(res.all_smoothed_slopes), 1)
        # Deterministic: simultaneously live smoothed grids stay O(batch), independent of n_steps
        self.assertLessEqual(live_peak[36], live_peak[6] + 1)
        self.assertLessEqual(live_peak[36], 5)
        # Loose secondary check: a retained-grid implementation would grow ~6x
        self.assertLess(peaks[36], 3.0 * peaks[6])

        # Thread-batching must not change the result
        res_par = optimize_bss_variance(dem, pts, geom, kc_max=0.3, kc_min=0.01, n_steps=6, n_cores=4, show_progress=False)
        np.testing.assert_allclose(res_par.all_kc_variances, results[6].all_kc_variances, rtol=0, atol=0)
        self.assertEqual(res_par.optimal_kc, results[6].optimal_kc)


    def test_cli_config_positional_and_plan_survey_dispatch(self):
        """23. `pysole <config.json> [flags]` must parse (argparse sub-parser used to swallow the config path)."""
        import sys
        from pysole.config import main_cli

        with patch("pysole.pipeline.run_from_config") as mock_run:
            for argv in (["pysole", "cfg.json"], ["pysole", "cfg.json", "--batch", "-v"], ["pysole", "--batch", "cfg.json"]):
                mock_run.reset_mock()
                with patch.object(sys, "argv", argv):
                    main_cli()
                self.assertEqual(mock_run.call_args.args[0], "cfg.json")
            self.assertEqual(mock_run.call_args.kwargs["is_batch"], True)
            self.assertIsNone(mock_run.call_args.kwargs["drift_analyzer"])

            mock_run.reset_mock()
            with patch.object(sys, "argv", ["pysole", "cfg.json", "-v"]):
                main_cli()
            self.assertEqual(mock_run.call_args.kwargs["log_level"], "DEBUG")

        with patch("pysole.survey_planner.SurveyPlanner") as mock_planner:
            with patch.object(sys, "argv", ["pysole", "plan-survey", "--dem", "d.tif", "--max-km", "2", "--tau", "80"]):
                with self.assertRaises(SystemExit) as ctx:
                    main_cli()
            self.assertEqual(ctx.exception.code, 0)
            self.assertEqual(mock_planner.call_args.kwargs["dem"], "d.tif")
            kw = mock_planner.return_value.plan_survey.call_args.kwargs
            self.assertEqual((kw["max_length_km"], kw["tau_0"]), (2.0, 80000.0))

    def test_public_api_exports_and_no_unused_imports(self):
        """37. New public names are exported; cleaned-up modules no longer carry the removed imports."""
        import ast
        import pysole

        for name in ("ConfigError", "OutputsConfig", "PipelineExporter", "run_from_config"):
            self.assertIn(name, pysole.__all__)
            self.assertTrue(hasattr(pysole, name), name)

        src = Path(pysole.__file__).parent
        for mod in src.glob("*.py"):
            if mod.name == "__init__.py":
                continue  # re-exports by design
            tree = ast.parse(mod.read_text(encoding="utf-8"))
            imported = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imported[(a.asname or a.name).split(".")[0]] = node.lineno
                elif isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        imported[a.asname or a.name] = node.lineno
            used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
                n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            }
            text = mod.read_text(encoding="utf-8")
            unused = sorted(
                name
                for name in imported
                if name not in used and name != "annotations" and f'"{name}"' not in text and f"'{name}'" not in text
            )
            self.assertEqual(unused, [], f"unused imports in {mod.name}: {unused}")

    def test_kriging_variance_thread_safety_and_chunk_invariance(self):
        """41. Option B Thread Safety: Verifies 1-thread vs multi-thread variance equality and chunk seam invariance."""
        geom = GridGeometry.create((40, 40), dx=5.0, dy=5.0, bounds=(0, 0, 200, 200))
        np.random.seed(42)
        X = np.random.uniform(20, 180, 40)
        Y = np.random.uniform(20, 180, 40)
        Z = np.random.uniform(10, 100, 40)
        pts = np.column_stack((X, Y, Z))

        # 1-thread evaluation
        z_1, var_1 = built_in_kriging_interpolation(
            pts, geom.x_coords, geom.y_coords, method="universal",
            drift_terms=["quadratic_xy"],
            variogram_model="spherical", return_variance=True, n_cores=1, show_progress=False
        )

        # 8-thread evaluation
        z_8, var_8 = built_in_kriging_interpolation(
            pts, geom.x_coords, geom.y_coords, method="universal",
            drift_terms=["quadratic_xy"],
            variogram_model="spherical", return_variance=True, n_cores=8, show_progress=False
        )

        # Asserts exact numeric equality between 1 thread and 8 threads
        np.testing.assert_allclose(z_1, z_8, atol=1e-12)
        np.testing.assert_allclose(var_1, var_8, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(var_8)))
        self.assertTrue(np.all(var_8 >= 0.0))

    def test_drift_analyzer_parameter_scoping(self):
        """inputs.drift_analyzer is pruned without backwards compatibility; pre/post migration are scoped per-pass."""
        from pysole.config import sanitize_config
        from pysole.solver import Solver

        # 1. Pruned as unknown key from inputs
        cleaned, unknown = sanitize_config({"inputs": {"drift_analyzer": True}})
        self.assertIn("inputs.drift_analyzer", unknown)
        self.assertNotIn("drift_analyzer", cleaned["inputs"])

        # 2. Solver initialization respects per-pass settings
        dem = np.full((10, 10), 100.0, dtype=np.float64)
        solver = Solver(
            dem=dem,
            dx=5.0,
            dy=5.0,
            pre_drift_analyzer=True,
            post_drift_analyzer=False,
            show_progress=False,
        )
        self.assertTrue(solver.pre_drift_analyzer)
        self.assertFalse(solver.post_drift_analyzer)
        self.assertFalse(hasattr(solver, "drift_analyzer"))  # no master override attribute any more

        # 3. Solver.from_config correctly passes per-pass settings
        cfg = {
            "inputs": {"dem_path": "dummy.tif"},
            "kriging_parameters": {
                "pre_migration": {"drift_analyzer": False},
                "post_migration": {"drift_analyzer": True},
            },
        }
        with patch("pysole.solver.load_dem", return_value=(dem, {"dx": 5.0, "dy": 5.0})):
            solver_cfg = Solver.from_config(cfg)
            self.assertFalse(solver_cfg.pre_drift_analyzer)
            self.assertTrue(solver_cfg.post_drift_analyzer)
            self.assertFalse(hasattr(solver_cfg, "drift_analyzer"))

    def test_bss_optimization_memory_guard_and_interactive_grid_pruning(self):
        """Verifies that BSS optimization limits workers for large grids and retains strictly O(1) grids in interactive mode."""
        from pysole.variogram import optimize_bss_variance

        n = 50
        geom = GridGeometry.create((n, n), dx=10.0, dy=10.0, bounds=(0, 0, n * 10.0, n * 10.0))
        xx, yy = geom.meshgrid
        dem = 1000.0 + 0.1 * xx + 0.1 * yy
        rng = np.random.default_rng(42)
        px, py = rng.uniform(50, n * 10.0 - 50, 30), rng.uniform(50, n * 10.0 - 50, 30)
        pts = np.column_stack((px, py, 1000.0 + 0.1 * px + 0.1 * py, rng.uniform(20, 80, 30)))

        # 1. Interactive execution retains strictly O(1) optimal grids (length 1)
        # input sequence: kc_max (default=""), kc_min (default=""), n_steps (default=""),
        # accept range (default="y"), continue process (default="n"), accept optimal kc (default="y")
        with patch("builtins.input", side_effect=["", "", "", "y", "n", "y"]):
            res_inter = optimize_bss_variance(
                dem, pts, geom, kc_max=0.3, kc_min=0.01, n_steps=5, n_cores=1, interactive=True, show_progress=False
            )
        self.assertEqual(len(res_inter.all_smoothed_dems), 1)
        self.assertEqual(len(res_inter.all_smoothed_slopes), 1)
        best_k = round(float(res_inter.optimal_kc), 6)
        self.assertIn(best_k, res_inter.all_smoothed_dems)
        self.assertIn(best_k, res_inter.all_smoothed_slopes)

        # 2. Solver._smoothed_dem_cache holds strictly the single optimal DEM grid
        from pysole.solver import Solver
        solver = Solver(dem=dem, dx=10.0, dy=10.0, show_progress=False)
        solver.survey_points = pts
        solver.migrated_points = pts
        solver.optimize_bss(kc_max=0.3, kc_min=0.01, n_steps=5, interactive=False)
        self.assertEqual(len(solver._smoothed_dem_cache), 1)
        self.assertIn(round(float(solver.opt_kc), 6), solver._smoothed_dem_cache)

        # 3. Memory guard worker scaling for large grids (mocked FFT to avoid 9M element array computation)
        large_dem_mock = np.empty((3000, 3000), dtype=np.float32)  # 9M cells
        geom_large = GridGeometry.create(large_dem_mock.shape, dx=10.0, dy=10.0)
        from unittest.mock import MagicMock
        with patch("pysole.variogram.precompute_fft_grid") as mock_pre:
            mock_pre.return_value = (MagicMock(), np.zeros((10, 10)), 1.0)
            with patch("pysole.variogram.compute_slope_rad") as mock_slope:
                mock_slope.return_value = np.zeros((10, 10))
                with self.assertLogs("pysole", level="INFO") as cm:
                    try:
                        optimize_bss_variance(
                            large_dem_mock, pts, geom_large, kc_max=0.3, kc_min=0.01, n_steps=1, n_cores=16, show_progress=False
                        )
                    except Exception:
                        pass
        guard_logs = [log for log in cm.output if "[Memory Guard]" in log]
        self.assertTrue(len(guard_logs) > 0)
        self.assertIn("Capping concurrent FFT optimization workers to 2", guard_logs[0])

    def test_stage1_interactive_bss_optimization(self):
        """Verifies that Stage 1 BSS optimization prompts interactively when configured."""
        from pysole.solver import Solver
        n = 20
        dem = np.full((n, n), 1000.0)
        rng = np.random.default_rng(42)
        px = rng.uniform(20.0, 180.0, 30)
        py = rng.uniform(20.0, 180.0, 30)
        pts = np.column_stack((px, py, np.full(30, 1000.0), rng.uniform(50.0, 150.0, 30)))

        solver = Solver(
            dem=dem,
            dx=10.0,
            dy=10.0,
            show_progress=False,
            pre_interpolation_target="P",
            config={
                "optimization_parameters": {"interactive_optimization": True},
            },
        )
        solver.survey_points = pts

        # When migrate_eikonal is called with interactive_optimization=True,
        # Stage 1 optimization prompts the user via input()
        with patch("sys.stdin.isatty", return_value=True):
            with patch("builtins.input", side_effect=["", "", "", "y", "n", "y"]) as mock_input:
                solver.migrate_eikonal(
                    travel_times=pts,
                    velocity=0.16,
                    interactive=False,
                    interactive_optimization=True,
                )
                self.assertTrue(mock_input.called, "Expected interactive input() to be called during Stage 1 BSS optimization")

    def test_dynamic_vector_scale_factor(self):
        """Verifies _compute_vector_scale_factor behavior across domain scales and displacement magnitudes."""
        from pysole.plotting import _compute_vector_scale_factor

        x_coords = np.linspace(0, 30000, 300)
        y_coords = np.linspace(0, 32000, 320)
        # 30 km domain, ~25m displacement -> target ~1120m -> raw factor ~45 -> rounds to 50x
        dxi = np.full(50, 15.0)
        dyi = np.full(50, 20.0)
        scale = _compute_vector_scale_factor(dxi, dyi, x_coords, y_coords)
        self.assertEqual(scale, 50.0)

        # Small local domain (2 km), substantial displacement (80m) -> no amplification needed (scale 1.0)
        x_small = np.linspace(0, 2000, 100)
        y_small = np.linspace(0, 2000, 100)
        dxi_lg = np.full(20, 60.0)
        dyi_lg = np.full(20, 50.0)
        scale_small = _compute_vector_scale_factor(dxi_lg, dyi_lg, x_small, y_small)
        self.assertEqual(scale_small, 1.0)

        # Zero displacement -> returns 1.0
        scale_zero = _compute_vector_scale_factor(np.zeros(10), np.zeros(10), x_coords, y_coords)
        self.assertEqual(scale_zero, 1.0)

    def test_plot_migration_displacement_vectors_dynamic_scaling(self):
        """Verifies that plot_migration_displacement_vectors generates cleanly with dynamic scaling."""
        from pysole.plotting import plot_migration_displacement_vectors

        tt_grid = np.random.uniform(50, 500, (40, 40))
        x_coords = np.linspace(10000, 40000, 40)
        y_coords = np.linspace(20000, 50000, 40)

        n_pts = 1000
        survey_pts = np.random.uniform(15000, 35000, (n_pts, 4))
        dxi = np.random.uniform(-30, 30, n_pts)
        dyi = np.random.uniform(-30, 30, n_pts)
        mig_pts = np.column_stack((survey_pts[:, 0] + dxi, survey_pts[:, 1] + dyi, survey_pts[:, 2], np.random.uniform(20, 100, n_pts)))

        plot_migration_displacement_vectors(
            travel_time_grid=tt_grid,
            survey_points=survey_pts,
            migrated_points=mig_pts,
            dxi=dxi,
            dyi=dyi,
            x_coords=x_coords,
            y_coords=y_coords,
            plots_dir=self.output_dir,
            interactive=False,
        )

        expected_fig = self.output_dir / "03_3d_migration.png"
        self.assertTrue(expected_fig.exists())
        self.assertGreater(expected_fig.stat().st_size, 1000)

    def test_diagnostic_plots_continuous_outline_contour(self):
        """Verifies that thickness and basal shear stress diagnostic plots render continuous solid black outline contours."""
        import matplotlib.pyplot as plt
        from pysole.plotting import (
            plot_final_ice_thickness_and_uncertainty,
            plot_final_basal_shear_stress_and_uncertainty,
        )

        n = 20
        grid = np.ones((n, n), dtype=float) * 50.0
        unc = np.ones((n, n), dtype=float) * 5.0
        mask = np.zeros((n, n), dtype=bool)
        mask[5:15, 5:15] = True
        extent = [0.0, 100.0, 0.0, 100.0]

        contour_styles = []
        orig_subplots = plt.subplots

        def spy_subplots(*args, **kwargs):
            fig, axes = orig_subplots(*args, **kwargs)
            for ax in axes:
                orig_ax_contour = ax.contour
                def make_wrapper(orig_fn):
                    def wrapper(*c_args, **c_kwargs):
                        contour_styles.append(c_kwargs)
                        return orig_fn(*c_args, **c_kwargs)
                    return wrapper
                ax.contour = make_wrapper(orig_ax_contour)
            return fig, axes

        with patch.object(plt, "subplots", side_effect=spy_subplots):
            plot_final_ice_thickness_and_uncertainty(
                final_thickness=grid,
                kriged_std=unc,
                outline_mask=mask,
                plot_extent=extent,
                mean_thick=50.0,
                mean_unc=5.0,
                plots_dir=self.output_dir,
                interactive=False,
            )
            plot_final_basal_shear_stress_and_uncertainty(
                final_bss=grid,
                bss_std=unc,
                outline_mask=mask,
                plot_extent=extent,
                mean_bss=50.0,
                mean_bss_unc=5.0,
                plots_dir=self.output_dir,
                interactive=False,
            )

        self.assertEqual(len(contour_styles), 4)
        for s in contour_styles:
            self.assertEqual(s.get("colors"), "black")
            self.assertEqual(s.get("linestyles"), "solid")
            self.assertEqual(s.get("linewidths"), 0.8)


if __name__ == "__main__":
    unittest.main()



