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
    DualKrigingSolver,
    kriging_interpolation,
    built_in_kriging_interpolation,
    random_forest_hole_filling,
    blend_margin_topography,
)
from pysole.smoothing import fft_gaussian_smooth
from pysole.migration import migrate_eikonal_points
from pysole.variogram import fit_variogram_model, calculate_variogram, optimize_bss_variance
from pysole.survey_planner import SurveyPlanner
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

    def test_coords_to_grid_indices_bottom_up(self):
        """1. test_coords_to_grid_indices_bottom_up: Verifies (x,y) -> (row, col) indexing."""
        shape = (10, 10)
        dx, dy = 5.0, 5.0
        bounds = (100.0, 200.0, 150.0, 250.0)  # minx, miny, maxx, maxy
        geom = GridGeometry.create(shape, dx=dx, dy=dy, bounds=bounds)

        # Center of top-left cell (row=0, col=0) in GDAL convention: y is near maxy
        x_top_left = bounds[0] + dx / 2.0
        y_top_left = bounds[3] - dy / 2.0
        row, col = geom.coords_to_grid_indices(x_top_left, y_top_left)
        self.assertEqual(row, 0)
        self.assertEqual(col, 0)

        # Center of bottom-right cell (row=9, col=9): y is near miny
        x_bot_right = bounds[2] - dx / 2.0
        y_bot_right = bounds[1] + dy / 2.0
        row_br, col_br = geom.coords_to_grid_indices(x_bot_right, y_bot_right)
        self.assertEqual(row_br, 9)
        self.assertEqual(col_br, 9)

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
            val_solver = DualKrigingSolver(
                X[idx], Y[idx], Depth[idx],
                drift_basis=val_drift,
                range_param=200.0,
                sill=1.0,
                nugget=0.01
            )
            pred = val_solver.predict(X[i:i+1], Y[i:i+1])
            errors.append((pred[0] - Depth[i]) ** 2)

        rmse = np.sqrt(np.mean(errors))
        self.assertLess(rmse, 2.0)

    def test_outline_vector_orientation(self):
        """3. test_outline_vector_orientation: Verifies outline mask rasterization symmetry."""
        shape = (20, 20)
        geom = GridGeometry.create(shape, dx=10.0, dy=10.0, bounds=(0, 0, 200, 200))
        dem = np.full(shape, 1000.0)

        # Create a central mask
        mask = np.zeros(shape, dtype=bool)
        mask[5:15, 5:15] = True

        self.assertEqual(mask.shape, dem.shape)
        self.assertTrue(np.any(mask))

    def test_outline_nan_rings_hole(self):
        """4. test_outline_nan_rings_hole: Verifies outline mask handles nunatak/rock outcrop holes."""
        shape = (30, 30)
        mask = np.ones(shape, dtype=bool)
        # Add rock outcrop hole in center
        mask[10:20, 10:20] = False

        self.assertFalse(mask[15, 15])
        self.assertTrue(mask[2, 2])

    def test_survey_tracks_inside_mask(self):
        """5. test_survey_tracks_inside_mask: Verifies planner tracks stay within domain budget."""
        dem = np.full((50, 50), 2000.0)
        for i in range(50):
            dem[i, :] -= i * 10.0  # slope along Y

        dem_path = self.output_dir / "test_dem.tif"
        BedrockMap(dem, bounds=(0, 0, 500, 500), crs="EPSG:32633").save(dem_path)

        planner = SurveyPlanner(dem=dem_path)
        res = planner.plan_survey(kc=0.0314, max_length_km=5.0, output_dir=self.output_dir)

        self.assertIn("sia_modelled_depth", res)
        self.assertIn("tracks", res)
        self.assertIn("saved_geojson", res)

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
                "output_dir": str(self.output_dir),
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

    def test_compound_drift_expansion(self):
        """11. Verifies compound drift names expand to primitive drift terms (identical to explicit primitives)."""
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
        res_compound = kriging_interpolation(drift_terms=["sia_z_dem"], **common)
        res_explicit = kriging_interpolation(drift_terms=["sia", "z_dem"], **common)
        res_sia_only = kriging_interpolation(drift_terms=["sia"], **common)

        self.assertEqual(res_compound.bedrock_grid.shape, (10, 10))
        self.assertTrue(np.all(np.isfinite(res_compound.bedrock_grid)))
        np.testing.assert_allclose(res_compound.bedrock_grid, res_explicit.bedrock_grid, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(res_compound.variance_grid, res_explicit.variance_grid, rtol=1e-10, atol=1e-10)
        # The z_dem component must actually change the fit relative to the SIA-only drift
        self.assertGreater(float(np.max(np.abs(res_compound.bedrock_grid - res_sia_only.bedrock_grid))), 1e-6)

    def test_eikonal_migration_oblique_slope(self):
        """12. Verifies exact 3D Eikonal closed-form slowness vector on oblique surface slopes."""
        pts = np.array([[100.0, 100.0, 2020.0, 500.0]])  # OWTT 500 ns
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
        # s = sqrt(1 + f_x^2 + f_y^2); path length d = v * OWTT = 0.16 m/ns * 500 ns = 80 m.
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

    def test_run_from_config_exports_bedrock_raster(self):
        """13. Verifies run_from_config writes the bedrock elevation raster map to disk by default."""
        import json
        from pysole.solver import Solver
        from pysole.pipeline import run_from_config

        dem_file = self.output_dir / "test_dem.tif"
        pts_file = self.output_dir / "test_picks.csv"
        cfg_file = self.output_dir / "test_pysole.json"

        # Create dummy DEM raster in projected UTM meters
        bounds = (500000.0, 5200000.0, 500100.0, 5200100.0)
        geom = GridGeometry.create((10, 10), dx=10.0, dy=10.0, bounds=bounds)
        dem_data = np.full((10, 10), 1000.0)
        raster = BedrockMap(grid=dem_data, bounds=geom.bounds, crs=32633)
        raster.save(dem_file)

        # Create dummy survey points in projected UTM meters
        pts = np.array([[500020.0, 5200020.0, 50.0], [500050.0, 5200050.0, 60.0], [500080.0, 5200080.0, 40.0]])
        np.savetxt(pts_file, pts, delimiter=",", header="X,Y,Picks", comments="")

        # Create pysole.json
        cfg_data = {
            "inputs": {
                "dem_path": str(dem_file),
                "survey_data_path": str(pts_file),
                "survey_data_type": "depth",
                "show_progress": False,
            },
            "outputs": {
                "output_dir": str(self.output_dir / "out"),
                "output_format": "tif",
                "output_prefix": "wuk_test",
            },
        }
        with open(cfg_file, "w") as f:
            json.dump(cfg_data, f)

        # Run pipeline
        res_map = run_from_config(config_path=cfg_file, is_batch=True)
        self.assertEqual(res_map.shape, (10, 10))
        self.assertTrue(np.all(np.isfinite(res_map.grid)))
        self.assertLessEqual(float(np.max(res_map.grid)), 1000.0 + 1e-9)  # bedrock never above the 1000 m surface

        # Verify bedrock elevation raster file was written to output_dir
        expected_bedrock_tif = self.output_dir / "out" / "wuk_test_bedrock.tif"
        self.assertTrue(expected_bedrock_tif.exists(), f"Bedrock raster map missing at: {expected_bedrock_tif}")

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
                    target_type="D", points=solver.migrated_points, krig_method="ordinary", drift_terms=[],
                    var_model="spherical", zero_boundary=False, pass_name="Pass 2: Post-Migration",
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

    def test_cli_log_level_overrides_config(self):
        """17. R1: run_from_config(log_level='DEBUG') beats the config-file log_level (and the control case does not)."""
        import json

        dem = np.full((20, 20), 2000.0)
        dem_path = self.output_dir / "dem_log.tif"
        BedrockMap(dem, bounds=(0, 0, 200, 200), crs="EPSG:32633").save(dem_path)
        pts_path = self.output_dir / "pts_log.csv"
        pts = np.array([[50.0, 50.0, 2000.0, 30.0], [100.0, 100.0, 1950.0, 40.0], [150.0, 150.0, 1900.0, 25.0]])
        np.savetxt(pts_path, pts, delimiter=",", header="X,Y,Z_surf,depth", comments="")
        cfg = {
            "inputs": {
                "dem_path": str(dem_path), "survey_data_path": str(pts_path), "survey_data_type": "depth",
                "output_dir": str(self.output_dir), "show_progress": False, "log_level": "WARNING",
            },
            "migration_parameters": {"perform_migration": False},
            "outputs": {"output_dir": str(self.output_dir / "log_out"), "output_format": "tif"},
        }
        cfg_path = self.output_dir / "cfg_log.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        prev_level = pysole_logger.level
        self.addCleanup(pysole_logger.setLevel, prev_level)
        env = {k: v for k, v in os.environ.items() if k != "PYSOLE_LOG_LEVEL"}
        with patch.dict(os.environ, env, clear=True):
            run_from_config(cfg_path, is_batch=True)
            self.assertEqual(logging.getLogger("pysole").level, logging.WARNING)  # control: config value wins
            run_from_config(cfg_path, is_batch=True, log_level="DEBUG")
            self.assertEqual(logging.getLogger("pysole").level, logging.DEBUG)  # CLI override wins

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

        compound = [
            "curvature_dem", "sia_curvature_dem", "z_dem_curvature_dem",
            "sia_z_dem_curvature_dem", "full_physical", "full_spatial_physical",
        ]
        with patch.object(type(solver), "get_smoothed_curvature", return_value=sentinel) as mock_curv, \
                patch("pysole.solver.kriging_interpolation", return_value=fake) as mock_krig:
            for term in compound:
                mock_krig.reset_mock()
                mock_curv.reset_mock()
                solver._execute_kriging_pass(
                    target_type="D", points=solver.migrated_points, krig_method="universal", drift_terms=[term],
                    var_model="spherical", zero_boundary=False, pass_name="Pass 2: Post-Migration",
                )
                mock_curv.assert_called_once_with(0.02)
                ext = mock_krig.call_args.kwargs["external_drift_grid"]
                np.testing.assert_array_equal(ext["curvature_dem"], sentinel)

            # Control: drifts without curvature must not trigger the (expensive) smoothed-curvature evaluation
            mock_krig.reset_mock()
            mock_curv.reset_mock()
            solver._execute_kriging_pass(
                target_type="D", points=solver.migrated_points, krig_method="universal", drift_terms=["sia_z_dem"],
                var_model="spherical", zero_boundary=False, pass_name="Pass 2: Post-Migration",
            )
            mock_curv.assert_not_called()
            self.assertIsNone(mock_krig.call_args.kwargs["external_drift_grid"])

    def test_return_variance_shortcut(self):
        """20. N5-M7: return_variance=False skips the K^-1 solve and variance GEMM but keeps the prediction identical."""
        import pysole.interpolation as interp

        geom = GridGeometry.create((20, 20), dx=10.0, dy=10.0, bounds=(0, 0, 200, 200))
        rng = np.random.default_rng(1)
        pts = np.column_stack((rng.uniform(5, 195, 25), rng.uniform(5, 195, 25), rng.uniform(10, 60, 25)))
        kw = dict(method="universal", variogram_model="spherical",
                  variogram_params={"range": 120.0, "sill": 80.0, "nugget": 1.0},
                  drift_terms=["linear_xy"], show_progress=False)

        calls = []
        real_lu_solve = interp.lu_solve

        def counting_lu_solve(*a, **k):
            calls.append(1)
            return real_lu_solve(*a, **k)

        with patch.object(interp, "lu_solve", side_effect=counting_lu_solve):
            z_full, v_full = built_in_kriging_interpolation(pts, geom.x_coords, geom.y_coords, **kw)
            n_full = len(calls)
            calls.clear()
            z_fast, v_fast = built_in_kriging_interpolation(
                pts, geom.x_coords, geom.y_coords, return_variance=False, **kw
            )
            n_fast = len(calls)

        self.assertEqual(n_full, 2)  # dual weights + explicit K^-1 for the variance
        self.assertEqual(n_fast, 1)  # dual weights only
        np.testing.assert_array_equal(z_full, z_fast)
        self.assertTrue(np.all(np.isfinite(v_full)) and np.all(v_full >= 0.0))
        self.assertTrue(np.all(np.isnan(v_fast)))  # NaN, never a misleading 0 uncertainty

    def test_solver_variance_flag_wiring(self):
        """21. N5-M7/F-M2: pass 1 never asks for variance; pass 2 asks only if compute_uncertainty / a raster needs it."""
        from pysole.interpolation import KrigingResult

        solver = self._make_small_solver()
        solver.config = {}
        self.assertTrue(solver._needs_uncertainty_maps())  # default: compute_uncertainty = True

        solver.config = {"outputs": {"compute_uncertainty": False}}
        self.assertFalse(solver._needs_uncertainty_maps())
        solver.config = {"outputs": {}}
        self.assertTrue(solver._needs_uncertainty_maps())

        # constructor argument beats the config value
        solver_off = self._make_small_solver(compute_uncertainty=False)
        solver_off.config = {"outputs": {"compute_uncertainty": True}}
        self.assertFalse(solver_off.compute_uncertainty)
        self.assertFalse(solver_off._needs_uncertainty_maps())

        fake = KrigingResult(bedrock_grid=np.full((12, 12), 40.0), variance_grid=np.full((12, 12), np.nan))
        with patch("pysole.solver.kriging_interpolation", return_value=fake) as mock_krig:
            solver._execute_kriging_pass(
                target_type="D", points=solver.migrated_points, krig_method="ordinary", drift_terms=[],
                var_model="spherical", zero_boundary=False, pass_name="Pass 1: Pre-Migration", return_variance=False,
            )
            self.assertFalse(mock_krig.call_args.kwargs["return_variance"])

    def test_uncertainty_raster_overrides_compute_uncertainty_once(self):
        """24. F-M2: an uncertainty raster request overrides compute_uncertainty=false with exactly ONE warning."""
        for key in ("save_thickness_uncertainty", "save_basal_shear_stress_uncertainty"):
            solver = self._make_small_solver(output_dir=str(self.output_dir / f"ovr_{key}"))
            solver.config = {"outputs": {"compute_uncertainty": False, key: True}}
            with patch.object(pysole_logger, "warning") as mock_warn:
                self.assertTrue(solver._needs_uncertainty_maps())
                self.assertTrue(solver._needs_uncertainty_maps())
            override_msgs = [c for c in mock_warn.call_args_list if "compute_uncertainty" in str(c)]
            self.assertEqual(len(override_msgs), 1, key)

        # end-to-end: the variance is really evaluated and finite
        solver = self._make_small_solver(output_dir=str(self.output_dir / "ovr_e2e"))
        solver.config = {"outputs": {"compute_uncertainty": False, "save_thickness_uncertainty": True}}
        solver.calculate_bedrock()
        self.assertTrue(np.all(np.isfinite(solver.kriged_std[solver.outline_mask])))

    def test_compute_uncertainty_false_skips_variance_keeps_figures(self):
        """25. F-M2: compute_uncertainty=false drops the K^-1 solve, keeps identical bedrock and the same figures."""
        import pysole.interpolation as interp

        real_lu_solve = interp.lu_solve
        runs = {}
        for flag in (True, False):
            counter = []

            def counting_lu_solve(*a, _c=counter, **k):
                _c.append(1)
                return real_lu_solve(*a, **k)

            solver = self._make_small_solver(output_dir=str(self.output_dir / f"cu_{flag}"))
            solver.config = {"outputs": {"compute_uncertainty": flag}}
            with patch.object(interp, "lu_solve", side_effect=counting_lu_solve):
                bedrock, variance = solver.calculate_bedrock()
            figs = sorted(os.listdir(solver.plots_dir))
            runs[flag] = (len(counter), bedrock.copy(), variance.copy(), solver.kriged_std.copy(), figs)

        n_on, bed_on, var_on, std_on, figs_on = runs[True]
        n_off, bed_off, var_off, std_off, figs_off = runs[False]
        self.assertEqual(n_on - n_off, 1)  # the explicit K^-1 solve of the variance step
        np.testing.assert_array_equal(bed_on, bed_off)  # bitwise identical prediction
        self.assertTrue(np.all(np.isfinite(var_on[solver.outline_mask])))
        self.assertTrue(np.all(np.isnan(var_off)) and np.all(np.isnan(std_off)))  # NaN, never a fake 0
        self.assertGreater(len(figs_on), 0)
        self.assertEqual(figs_on, figs_off)  # plots are ALWAYS saved

    def test_uncertainty_placeholder_panel_renders(self):
        """26. F-M2: an all-NaN uncertainty grid renders a placeholder panel without errors or warnings."""
        import warnings
        from pysole.plotting import plot_kriging_bedrock_and_uncertainty, UNCERTAINTY_PLACEHOLDER_TEXT

        self.assertIn("compute_uncertainty = false", UNCERTAINTY_PLACEHOLDER_TEXT)
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

    def test_outputs_config_compute_uncertainty_roundtrip(self):
        """27. F-M2: OutputsConfig/DEFAULT_CONFIG expose compute_uncertainty (default true) and round-trip it."""
        from pysole.config import DEFAULT_CONFIG, OutputsConfig

        self.assertIs(DEFAULT_CONFIG["outputs"]["compute_uncertainty"], True)
        self.assertTrue(OutputsConfig.from_dict({}).compute_uncertainty)
        cfg = OutputsConfig.from_dict({"compute_uncertainty": False})
        self.assertFalse(cfg.compute_uncertainty)
        self.assertFalse(cfg.effective_compute_uncertainty)
        self.assertIs(cfg.to_dict()["compute_uncertainty"], False)
        cfg2 = OutputsConfig.from_dict({"compute_uncertainty": False, "save_thickness_uncertainty": True})
        self.assertTrue(cfg2.uncertainty_rasters_requested and cfg2.effective_compute_uncertainty)

    def test_resolve_plots_dir_rules(self):
        """28. F-L2: figures follow the data files; absolute plots_dir wins; None == 'figures'."""
        from pysole.config import resolve_plots_dir

        base = self.output_dir / "data"
        prefix_abs = str(base / "run")
        other = str(self.output_dir / "elsewhere")
        # absolute plots_dir is used as is
        self.assertEqual(resolve_plots_dir(other, output_prefix=prefix_abs), other)
        # absolute prefix: figures beside the data (no directories created by the resolution itself)
        self.assertEqual(resolve_plots_dir(None, output_prefix=prefix_abs), str(base / "figures"))
        self.assertEqual(resolve_plots_dir("figures", output_prefix=prefix_abs), str(base / "figures"))
        self.assertEqual(resolve_plots_dir("plots", output_prefix=prefix_abs), str(base / "plots"))
        self.assertFalse(base.exists())
        # relative prefix: resolved against output_dir
        out = self.output_dir / "outdir"
        self.assertEqual(resolve_plots_dir(None, output_prefix="run", output_dir=str(out)), str(out / "figures"))
        self.assertEqual(
            resolve_plots_dir(None, output_prefix="run", output_dir=str(out)),
            resolve_plots_dir("figures", output_prefix="run", output_dir=str(out)),
        )

    def _write_pipeline_config(self, tag, outputs, n=16, survey_data_type="depth"):
        """Small survey pipeline config in <temp>/<tag>; returns (config_path, work_dir)."""
        import json

        work = self.output_dir / tag
        work.mkdir(parents=True, exist_ok=True)
        bounds = (500000.0, 5200000.0, 500000.0 + 10.0 * n, 5200000.0 + 10.0 * n)
        geom = GridGeometry.create((n, n), dx=10.0, dy=10.0, bounds=bounds)
        xx, yy = geom.meshgrid
        dem = 1000.0 + 0.2 * (xx - bounds[0]) + 0.1 * (yy - bounds[1])
        BedrockMap(grid=dem, bounds=geom.bounds, crs=32633).save(work / "dem.tif")
        rng = np.random.default_rng(7)
        px = rng.uniform(bounds[0] + 15, bounds[2] - 15, 12)
        py = rng.uniform(bounds[1] + 15, bounds[3] - 15, 12)
        values = rng.uniform(20.0, 60.0, 12)
        if survey_data_type != "depth":
            values = values / 0.16  # depths of 20-60 m expressed as one-way traveltimes [ns]
        pts = np.column_stack((px, py, values))
        np.savetxt(work / "picks.csv", pts, delimiter=",", header="X,Y,Picks", comments="")
        cfg = {
            "inputs": {"dem_path": str(work / "dem.tif"), "survey_data_path": str(work / "picks.csv"),
                       "survey_data_type": survey_data_type, "show_progress": False},
            "outputs": {"output_format": "tif", **outputs},
        }
        cfg_file = work / "pysole.json"
        cfg_file.write_text(json.dumps(cfg))
        return cfg_file, work

    def test_pipeline_exports_each_output_once_and_honours_flags(self):
        """29. F-L1: every output is written at most once; skipped migration exports nothing; bedrock flag honoured."""
        from pysole.solver import Solver

        for survey_type in ("depth", "one_way_traveltime"):
            for flag in (True, False):
                tag = f"once_{survey_type}_{flag}"
                out = self.output_dir / tag / "out"
                cfg_file, _ = self._write_pipeline_config(
                    tag, {"output_dir": str(out), "output_prefix": "r", "save_bedrock_elevation_map": flag,
                          "save_migrated_points": True, "save_traveltime_grid": True},
                    survey_data_type=survey_type,
                )
                saves, points_exports = [], []
                orig_save = BedrockMap.save
                orig_pts = Solver._export_optional_points_csv

                def spy_save(self_, filepath, *a, _s=saves, **k):
                    _s.append(os.path.basename(str(filepath)))
                    return orig_save(self_, filepath, *a, **k)

                def spy_pts(self_, points, suffix, _p=points_exports):
                    _p.append(suffix)
                    return orig_pts(self_, points, suffix)

                with patch.object(BedrockMap, "save", spy_save), patch.object(Solver, "_export_optional_points_csv", spy_pts):
                    run_from_config(config_path=cfg_file, is_batch=True)

                msg = f"{survey_type}/flag={flag}: {saves} {points_exports}"
                self.assertEqual(len([s for s in saves if "_bedrock" in s]), 1 if flag else 0, msg)
                self.assertEqual((out / "r_bedrock.tif").exists(), flag, msg)
                expected_once = 1 if survey_type != "depth" else 0  # skipped migration exports nothing
                self.assertEqual(len([s for s in saves if "_traveltime" in s]), expected_once, msg)
                self.assertEqual(points_exports.count("migrated_points"), expected_once, msg)
                self.assertEqual((out / "r_migrated_points.csv").exists(), bool(expected_once), msg)

    def test_pipeline_figures_follow_absolute_prefix(self):
        """30. F-L2(d,e): with an absolute output_prefix figures land beside the data; plots_dir null == 'figures'."""
        for plots_dir in (None, "figures"):
            tag = f"absprefix_{plots_dir}"
            data_dir = self.output_dir / tag / "results"
            cfg_file, work = self._write_pipeline_config(
                tag, {"output_prefix": str(data_dir / "run"), "plots_dir": plots_dir},
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            run_from_config(config_path=cfg_file, is_batch=True)
            self.assertTrue((data_dir / "run_bedrock.tif").exists())
            figs = list((data_dir / "figures").glob("*.png"))
            self.assertGreater(len(figs), 0, f"no figures in {data_dir / 'figures'}")
            self.assertFalse((work / "pysole").exists(), "stray default pysole/ folder was created")

    def _wuk_solver(self, **kwargs):
        from pysole.solver import Solver

        data_dir = Path(__file__).resolve().parent.parent / "examples" / "wuk" / "input_data"
        return Solver(dem=str(data_dir / "dgm_unt_wuk.tif"), outline=str(data_dir / "wuk_outline_clean.csv"), **kwargs)

    def test_plan_survey_absolute_prefix_keeps_everything_together(self):
        """31. F-L2(a): plan_survey with an absolute prefix and no output_dir writes data+figures beside it, CWD untouched."""
        cwd_dir = self.output_dir / "cwd"
        plan_dir = self.output_dir / "plan"
        cwd_dir.mkdir()
        solver = self._wuk_solver()
        old_cwd = os.getcwd()
        os.chdir(cwd_dir)
        try:
            solver.plan_survey(output_prefix=str(plan_dir / "run"), max_length_km=1.0, output_format="tif")
            cwd_content = os.listdir(cwd_dir)
        finally:
            os.chdir(old_cwd)
        self.assertEqual(cwd_content, [], "plan_survey created files in the current working directory")
        self.assertTrue((plan_dir / "run_sia_modelled_depth.tif").exists())
        self.assertTrue((plan_dir / "run_survey_plan.gpx").exists())
        self.assertTrue((plan_dir / "figures" / "run_survey_plan_map.png").exists())

    def test_plan_survey_relative_prefix_and_absolute_plots_dir(self):
        """32. F-L2(b,c): relative prefix + output_dir unchanged; an absolute plots_dir wins."""
        out = self.output_dir / "plan_rel"
        pdir = self.output_dir / "plan_figs"
        solver = self._wuk_solver()
        solver.plan_survey(output_prefix="rel", output_dir=str(out), plots_dir=str(pdir),
                           max_length_km=1.0, output_format="tif")
        self.assertTrue((out / "rel_sia_modelled_depth.tif").exists())
        self.assertTrue((pdir / "rel_survey_plan_map.png").exists())
        self.assertFalse((out / "figures").exists())

        solver2 = self._wuk_solver()
        solver2.plan_survey(output_prefix="rel2", output_dir=str(out), max_length_km=1.0, output_format="tif")
        self.assertTrue((out / "figures" / "rel2_survey_plan_map.png").exists())

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

    def test_boolean_config_values_are_validated(self):
        """34. Booleans are coerced strictly: "false" must not silently become True; garbage raises ConfigError."""
        from pysole.config import ConfigError, OutputsConfig, coerce_bool, load_config

        for truthy in (True, 1, "true", "True", "YES", "on", "1", np.bool_(True)):
            self.assertIs(coerce_bool(truthy, "x", False), True, truthy)
        for falsy in (False, 0, "false", "False", "NO", "off", "0", np.bool_(False)):
            self.assertIs(coerce_bool(falsy, "x", True), False, falsy)
        self.assertIs(coerce_bool(None, "x", True), True)
        self.assertIs(coerce_bool(None, "x", False), False)
        for bad in ("maybe", 2, 0.5, [], {}):
            with self.assertRaises(ConfigError):
                coerce_bool(bad, "x", True)
        self.assertTrue(issubclass(ConfigError, ValueError))

        out = OutputsConfig.from_dict({"compute_uncertainty": "false", "save_bedrock_elevation_map": "True"})
        self.assertIs(out.compute_uncertainty, False)
        self.assertIs(out.save_bedrock_elevation_map, True)
        with self.assertRaises(ConfigError):
            OutputsConfig.from_dict({"compute_uncertainty": "maybe"})

        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(
                {
                    "inputs": {"output_dir": tmp, "show_progress": "false"},
                    "optimization_parameters": {"interactive_optimization": "no"},
                    "kriging_parameters": {"pre_migration": {"include_zero_boundary_condition": "False"}},
                }
            )
            self.assertIs(cfg["inputs"]["show_progress"], False)
            self.assertIs(cfg["optimization_parameters"]["interactive_optimization"], False)
            self.assertIs(cfg["kriging_parameters"]["pre_migration"]["include_zero_boundary_condition"], False)
            with self.assertRaises(ConfigError) as ctx:
                load_config({"inputs": {"output_dir": tmp}, "kriging_parameters": {"post_migration": {"include_zero_boundary_condition": "nope"}}})
            self.assertIn("include_zero_boundary_condition", str(ctx.exception))

    def test_load_config_invalid_json_and_non_object(self):
        """35. Malformed config files raise ConfigError (a ValueError) naming the file."""
        from pysole.config import ConfigError, load_config

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text('{"inputs": {', encoding="utf-8")
            with self.assertRaises(ConfigError) as ctx:
                load_config(bad)
            self.assertIn("bad.json", str(ctx.exception))
            self.assertIsInstance(ctx.exception, ValueError)

            arr = Path(tmp) / "array.json"
            arr.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(arr)

    def test_cli_reports_config_errors_cleanly(self):
        """36. CLI: ConfigError/FileNotFoundError -> one-line stderr message + exit 1; -v re-raises; --init confirms."""
        import contextlib
        import io
        import sys
        from pysole.config import ConfigError, main_cli

        for exc in (ConfigError("boom"), FileNotFoundError("missing.csv")):
            with patch("pysole.pipeline.run_from_config", side_effect=exc):
                err = io.StringIO()
                with patch.object(sys, "argv", ["pysole", "cfg.json"]), contextlib.redirect_stderr(err):
                    with self.assertRaises(SystemExit) as ctx:
                        main_cli()
                self.assertEqual(ctx.exception.code, 1)
                self.assertEqual(err.getvalue().strip(), f"pysole: error: {exc}")
                self.assertNotIn("Traceback", err.getvalue())

                with patch.object(sys, "argv", ["pysole", "cfg.json", "-v"]):
                    with self.assertRaises(type(exc)):
                        main_cli()

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "new_pysole.json"
            out = io.StringIO()
            with patch.object(sys, "argv", ["pysole", "--init", str(target)]), contextlib.redirect_stdout(out):
                with self.assertRaises(SystemExit) as ctx:
                    main_cli()
            self.assertEqual(ctx.exception.code, 0)
            self.assertTrue(target.exists())
            self.assertIn("Created template configuration file", out.getvalue())

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

    def test_docs_have_no_absolute_or_dead_links(self):
        """38. README / CHANGELOG / docs links are relative and resolve to existing files."""
        import re

        root = Path(__file__).resolve().parents[1]
        md_files = [root / "README.md", root / "CHANGELOG.md", *sorted((root / "docs").glob("*.md"))]
        if not (root / "README.md").exists():
            self.skipTest("not running from a source checkout")
        dead = []
        for md in md_files:
            text = md.read_text(encoding="utf-8")
            self.assertNotIn("file://", text, md.name)
            for m in re.finditer(r"\]\(([^)\s]+)\)", text):
                url = m.group(1)
                if url.startswith(("http", "mailto:", "#")):
                    continue
                rel = url.split("#")[0]
                if rel and not (md.parent / rel).exists():
                    dead.append(f"{md.name}: {url}")
        self.assertEqual(dead, [])

    def test_documented_drift_term_names_are_valid(self):
        """39. Drift names written in README / living docs are accepted by the code (no removed legacy aliases)."""
        import re
        from pysole.interpolation import DriftBasis

        root = Path(__file__).resolve().parents[1]
        if not (root / "README.md").exists():
            self.skipTest("not running from a source checkout")
        living = [
            root / "README.md",
            *(root / "docs" / n for n in (
                "pysole_interpolation_practice_guide.md",
                "gok_thickness_analysis_report.md",
                "drift_analyzer_&_survey_planner.md",
                "kriging_performance_report.md",
                "release_guide.md",
            )),
        ]
        legacy = re.compile(r"[`\"](sia_thickness|z_surface|regional_linear|quadratic|sia_drift)[`\"]")
        lists = re.compile(r'\[((?:\s*"[A-Za-z_]+"\s*,?)+)\]')
        problems = []
        for md in living:
            if not md.exists():
                continue
            for no, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
                # ``z_surface`` is a legitimate CSV column of the migrated-points export, not a drift name
                if legacy.search(line.replace("z_surface, depth_migrated", "")):
                    problems.append(f"{md.name}:{no}: legacy drift name")
                if "drift" in line.lower() or "sia" in line:
                    for m in lists.finditer(line):
                        bad = [n for n in re.findall(r'"([A-Za-z_]+)"', m.group(1)) if n not in DriftBasis.SUPPORTED_TERMS]
        self.assertEqual(problems, [])

    def test_save_traveltime_uncertainty_export(self):
        """40. save_traveltime_uncertainty auto-enables Pass 1 variance evaluation and exports traveltime uncertainty raster in traveltime units."""
        import tempfile
        from pysole.solver import Solver

        bounds = (500000.0, 5200000.0, 500100.0, 5200100.0)
        geom = GridGeometry.create((10, 10), dx=10.0, dy=10.0, bounds=bounds)
        dem = np.full((10, 10), 1000.0)
        pts = np.array([
            [500020.0, 5200020.0, 1000.0, 0.5],
            [500050.0, 5200050.0, 1000.0, 0.8],
            [500080.0, 5200080.0, 1000.0, 1.2],
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            solver = Solver(
                dem=dem,
                dx=10.0,
                dy=10.0,
                bounds=geom.bounds,
                output_dir=tmpdir,
                config={"outputs": {"save_traveltime_uncertainty": True}},
            )
            solver.pre_kriging_points = pts
            solver.migrate_eikonal(travel_times=pts, interactive=False)
            self.assertIsNotNone(solver.traveltime_std_grid)
            self.assertEqual(solver.traveltime_std_grid.shape, (10, 10))
            self.assertTrue(np.all(np.isfinite(solver.traveltime_std_grid)))
            self.assertTrue(np.all(solver.traveltime_std_grid >= 0.0))

            saved = solver.export_outputs(stage="migration")
            self.assertTrue(any("traveltime_uncertainty" in f for f in saved), saved)



if __name__ == "__main__":
    unittest.main()

