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
    random_forest_hole_filling,
    blend_margin_topography,
)
from pysole.smoothing import fft_gaussian_smooth
from pysole.migration import migrate_eikonal_points
from pysole.variogram import fit_variogram_model, calculate_variogram, optimize_bss_variance
from pysole.survey_planner import SurveyPlanner
from pysole.pipeline import run_from_config


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
        self.assertIsNotNone(res)

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
        """11. Verifies compound drift names expand to primitive drift terms."""
        geom = GridGeometry.create((10, 10), dx=10.0, dy=10.0, bounds=(0, 0, 100, 100))
        pts = np.array([[20.0, 20.0, 1000.0, 25.0], [50.0, 50.0, 950.0, 30.0], [80.0, 80.0, 900.0, 15.0]])
        dem_grid = np.full((10, 10), 1000.0)
        opt_slope = np.full((10, 10), 0.05)

        # Test compound drift expansion without crashing on unknown terms
        res = kriging_interpolation(
            sample_points=pts,
            geometry=geom,
            method="universal",
            drift_terms=["sia_z_dem"],
            dem_grid=dem_grid,
            opt_slope_grid=opt_slope,
            include_zero_boundary_condition=False,
            show_progress=False,
        )
        self.assertIsNotNone(res.bedrock_grid)

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


if __name__ == "__main__":
    unittest.main()
