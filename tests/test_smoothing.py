"""
Unit tests for surface gradient computation, FFT Gaussian smoothing, and margin blending using Wurtenkees Glacier dataset.
"""

import unittest
import os
import numpy as np
from pysole.smoothing import compute_gradients, fft_gaussian_smooth, compute_surface_curvature
from pysole.raster import load_dem, load_outline, ensure_spatial_coords, GridGeometry
from pysole.solver import blend_margin_topography, Solver


class TestSmoothingWuk(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.data_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.data_dir, "wuk_outline_clean.csv")

        self.dem, self.meta = load_dem(self.wuk_dem)
        self.outline_mask = load_outline(self.wuk_outline, self.dem, self.meta)
        self.x_coords, self.y_coords, self.bounds = ensure_spatial_coords(
            self.dem.shape, dx=self.meta["dx"], dy=self.meta["dy"], bounds=self.meta["bounds"]
        )

    def test_compute_gradients_wuk(self):
        grads = compute_gradients(self.dem, dx=self.meta["dx"], dy=self.meta["dy"])
        self.assertIn("slope_x", grads)
        self.assertIn("slope_y", grads)
        self.assertIn("slope_rad", grads)
        self.assertEqual(grads["slope_rad"].shape, self.dem.shape)
        self.assertTrue(np.all(grads["slope_rad"] >= 0))

    def test_fft_gaussian_smooth_wuk(self):
        smoothed, k_grid, k_max = fft_gaussian_smooth(self.dem, dx=self.meta["dx"], dy=self.meta["dy"], kc=0.1)
        self.assertEqual(smoothed.shape, self.dem.shape)
        self.assertFalse(np.isnan(smoothed).any())

    def test_blend_margin_topography_wuk(self):
        thick = np.where(self.outline_mask, 50.0, 0.0)
        bedrock_in = self.dem - thick
        geometry = GridGeometry.create(self.dem.shape, dx=self.meta["dx"], dy=self.meta["dy"], bounds=self.meta["bounds"])

        blended = blend_margin_topography(
            dem=self.dem,
            bedrock_input=bedrock_in,
            boundary_mask=self.outline_mask,
            geometry=geometry,
            min_gap_dist=30.0,
        )

        self.assertEqual(blended.shape, self.dem.shape)
        self.assertTrue(np.all(np.isfinite(blended)))


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

        bedrock_map = solver.finalize_bedrock(
            interactive=False,
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
