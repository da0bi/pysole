"""
Unit tests for Surface Curvature drift model, Unified DEM Smoothing Architecture,
and raw DEM bedrock embedding preservation.
"""

import unittest
import numpy as np
from pysole.smoothing import compute_surface_curvature, compute_gradients, fft_gaussian_smooth
from pysole.raster import GridGeometry
from pysole.interpolation import kriging_interpolation, built_in_kriging_interpolation
from pysole.solver import Solver


class TestSurfaceCurvatureAndBedrockEmbedding(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._out_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._out_dir.cleanup)
        self.dx = 10.0
        self.dy = 10.0
        self.nx = 30
        self.ny = 30
        self.shape = (self.ny, self.nx)
        self.geometry = GridGeometry.create(
            shape=self.shape,
            dx=self.dx,
            dy=self.dy,
            bounds=(0.0, 0.0, self.nx * self.dx, self.ny * self.dy),
        )

        # Create synthetic paraboloid bowl: Z(x, y) = 0.01 * (x^2 + y^2) + 2000.0
        xx, yy = self.geometry.meshgrid
        self.dem = 0.01 * (xx**2 + yy**2) + 2000.0

        # Active glacier outline mask (circular region in center)
        cx, cy = 150.0, 150.0
        dist_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        self.outline_mask = dist_center <= 100.0

    def test_compute_surface_curvature_paraboloid(self):
        """Verifies 2nd-order central difference Laplacian curvature on synthetic paraboloid Z = A*(x^2 + y^2)."""
        # Analytical d2Z/dx2 = 0.02, d2Z/dy2 = 0.02 -> Laplacian kappa = 0.04
        curv = compute_surface_curvature(self.dem, dx=self.dx, dy=self.dy)

        # Check interior points away from boundaries
        interior_curv = curv[2:-2, 2:-2]
        np.testing.assert_allclose(interior_curv, 0.04, rtol=1e-3, atol=1e-3)

    def test_compute_gradients_includes_curvature(self):
        """Verifies that compute_gradients returns curvature in output dictionary."""
        grads = compute_gradients(self.dem, dx=self.dx, dy=self.dy)
        self.assertIn("curvature", grads)
        self.assertEqual(grads["curvature"].shape, self.shape)

    def test_native_kriging_with_surface_curvature_drift(self):
        """Verifies Universal Kriging interpolation with 'curvature_dem' drift."""
        # Create synthetic sample points inside glacier
        xx, yy = self.geometry.meshgrid
        pts_x = xx[self.outline_mask][::5]
        pts_y = yy[self.outline_mask][::5]
        pts_t = np.ones(len(pts_x)) * 50.0  # 50m ice thickness
        sample_pts = np.column_stack((pts_x, pts_y, pts_t))

        curv_grid = compute_surface_curvature(self.dem, dx=self.dx, dy=self.dy)
        ext_drifts = {"curvature_dem": curv_grid}

        z_b, v_b = built_in_kriging_interpolation(
            sample_points=sample_pts,
            x_coords=self.geometry.x_coords,
            y_coords=self.geometry.y_coords,
            method="universal",
            external_drift_grid=ext_drifts,
            drift_terms=["curvature_dem", "quadratic_xy"],
        )

        self.assertEqual(z_b.shape, self.shape)
        self.assertEqual(v_b.shape, self.shape)
        self.assertFalse(np.isnan(z_b).any())

    def test_kriging_interpolation_multi_drift(self):
        """Verifies kriging_interpolation with combined z_dem, curvature_dem, and quadratic_xy drifts."""
        xx, yy = self.geometry.meshgrid
        pts_x = xx[self.outline_mask][::4]
        pts_y = yy[self.outline_mask][::4]
        pts_t = np.ones(len(pts_x)) * 60.0
        sample_pts = np.column_stack((pts_x, pts_y, pts_t))

        krig_res = kriging_interpolation(
            sample_points=sample_pts,
            geometry=self.geometry,
            method="universal",
            dem_grid=self.dem,
            drift_terms=["z_dem", "curvature_dem", "quadratic_xy"],
            outline_mask=self.outline_mask,
            include_zero_boundary_condition=True,
        )

        self.assertEqual(krig_res.bedrock_grid.shape, self.shape)
        self.assertFalse(np.isnan(krig_res.bedrock_grid).any())

    def test_raw_dem_bedrock_embedding_preservation(self):
        """Verifies that bedrock DEM outside glacier outline equals raw DEM down to exact floating-point identity."""
        xx, yy = self.geometry.meshgrid
        pts_x = xx[self.outline_mask][::5]
        pts_y = yy[self.outline_mask][::5]
        pts_t = np.ones(len(pts_x)) * 40.0
        sample_pts = np.column_stack((pts_x, pts_y, pts_t))

        solver = Solver(
            dem=self.dem,
            outline=self.outline_mask,
            dx=self.dx,
            dy=self.dy,
            post_drift_terms=["z_dem", "curvature_dem", "quadratic_xy"],
            perform_migration=False,
            survey_data_type="depth",
            output_dir=self._out_dir.name,
        )

        solver.survey_points = sample_pts
        solver.migrated_points = sample_pts.copy()

        bedrock_map = solver.finalize_topography(
            interactive=False,
            plotit=False,
            random_forest_gap_filling=False,
            apply_margin_blend=True,
            min_gap_dist=20.0,
        )

        # Outside glacier boundary, final bedrock elevation MUST equal raw DEM exactly
        outside_mask = ~self.outline_mask
        raw_outside = self.dem[outside_mask]
        blended_outside = solver.blended_bedrock[outside_mask]

        np.testing.assert_array_equal(blended_outside, raw_outside)
        self.assertTrue(np.all(solver.final_thickness[outside_mask] == 0.0))


if __name__ == "__main__":
    unittest.main()
