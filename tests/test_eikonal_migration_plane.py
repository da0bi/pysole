"""
Analytical Plane-over-Plane Unit Test Suite for 3D Eikonal Ray Migration in PySole.
Verifies ray displacement vectors across 6 analytical surface-bedrock plane geometries:
1. Flat Horizontal Surface & Flat Bedrock
2. X-Inclined Surface & Parallel Bedrock
3. Y-Inclined Surface & Parallel Bedrock
4. Oblique (X+Y) Inclined Surface & Parallel Bedrock
5. Non-Parallel Planes (Diverging Bedrock Slope)
6. Non-Parallel Planes (Opposite Bedrock Slope)
"""

import unittest
import numpy as np
from pysole.raster import GridGeometry
from pysole.migration import EikonalMigrator, migrate_eikonal_points


class TestEikonalMigrationPlane(unittest.TestCase):
    def setUp(self):
        self.v = 0.16  # Ice velocity [m/ns]
        self.geometry = GridGeometry.create(shape=(60, 60), dx=5.0, dy=5.0, bounds=(0, 0, 300, 300))
        self.xx, self.yy = self.geometry.meshgrid

    def _run_plane_test(self, A_s: float, B_s: float, C_s: float, thickness_m: float, sample_x: float, sample_y: float):
        """
        Helper method to run a plane-over-plane ray migration test.
        """
        # Surface DEM: z_s(x,y) = A_s * x + B_s * y + C_s
        dem = A_s * self.xx + B_s * self.yy + C_s

        # Surface normal unit vector n_s = (-A_s, -B_s, 1) / sqrt(1 + A_s^2 + B_s^2)
        n_raw = np.array([-A_s, -B_s, 1.0])
        norm_factor = np.linalg.norm(n_raw)
        n_unit = n_raw / norm_factor

        # Spatial ray displacement along surface normal for given ice thickness:
        # displacement vector = thickness_m * n_unit
        ray_disp = thickness_m * n_unit
        traveltime_ns = thickness_m / self.v

        # Surface observation point
        z_sample = A_s * sample_x + B_s * sample_y + C_s
        sample_pts = np.array([[sample_x, sample_y, z_sample, traveltime_ns]])

        # Expected migrated bedrock point: X_b = X_s - ray_disp[0], Y_b = Y_s - ray_disp[1], Z_b = Z_s - ray_disp[2]
        expected_x_b = sample_x - ray_disp[0]
        expected_y_b = sample_y - ray_disp[1]
        expected_z_b = z_sample - ray_disp[2]
        migrator = EikonalMigrator(dem, geometry=self.geometry)
        mig_pts = migrator.migrate_points(sample_pts, velocity=self.v)

        self.assertEqual(len(mig_pts), 1)
        x_mig, y_mig, z_surf_mig, depth_mig = mig_pts[0, :4]
        z_bedrock_mig = z_surf_mig - depth_mig

        np.testing.assert_allclose(x_mig, expected_x_b, atol=0.05)
        np.testing.assert_allclose(y_mig, expected_y_b, atol=0.05)
        np.testing.assert_allclose(z_bedrock_mig, expected_z_b, atol=0.05)

    def test_case_1_flat_horizontal(self):
        """Case 1: Flat Horizontal Surface & Flat Bedrock (Pure Vertical Displacement)."""
        self._run_plane_test(A_s=0.0, B_s=0.0, C_s=2000.0, thickness_m=100.0, sample_x=150.0, sample_y=150.0)

    def test_case_2_x_inclined_parallel(self):
        """Case 2: X-Inclined Surface & Parallel Bedrock."""
        self._run_plane_test(A_s=0.1, B_s=0.0, C_s=2000.0, thickness_m=120.0, sample_x=150.0, sample_y=150.0)

    def test_case_3_y_inclined_parallel(self):
        """Case 3: Y-Inclined Surface & Parallel Bedrock."""
        self._run_plane_test(A_s=0.0, B_s=0.15, C_s=2000.0, thickness_m=80.0, sample_x=150.0, sample_y=150.0)

    def test_case_4_oblique_xy_parallel(self):
        """Case 4: Oblique (X+Y) Inclined Surface & Parallel Bedrock."""
        self._run_plane_test(A_s=0.08, B_s=0.12, C_s=2000.0, thickness_m=150.0, sample_x=150.0, sample_y=150.0)

    def test_case_5_non_parallel_diverging(self):
        """Case 5: Non-Parallel Plane (Steeper Bedrock Slope)."""
        self._run_plane_test(A_s=0.05, B_s=0.05, C_s=2000.0, thickness_m=200.0, sample_x=150.0, sample_y=150.0)

    def test_case_6_non_parallel_opposite(self):
        """Case 6: Non-Parallel Plane (Opposite Slope Orientation)."""
        self._run_plane_test(A_s=0.12, B_s=-0.08, C_s=2000.0, thickness_m=90.0, sample_x=150.0, sample_y=150.0)

    def test_inclined_bed_traveltime_gradient(self):
        """
        Verifies sign of traveltime gradients (N3-H1):
        Flat surface DEM, bed dip b_x = 0.1.
        Traveltime increases with x (dT/dx > 0), so u = -dT/dx < 0, ray relocates in -X direction.
        For sample_x = 150, thickness = 80m, expected dx = -7.92m (towards up-dip bed).
        """
        dem = np.full_like(self.xx, 2000.0)
        bx = 0.1
        h_0 = 80.0
        # Thickness normal distance H(x) = (h_0 + bx * (xx - 150)) / sqrt(1 + bx^2)
        H_grid = (h_0 + bx * (self.xx - 150.0)) / np.sqrt(1.0 + bx**2)
        T_grid = H_grid / self.v

        migrator = EikonalMigrator(dem, geometry=self.geometry)
        sample_x, sample_y = 150.0, 150.0
        sample_pts = np.array([[sample_x, sample_y, 2000.0, T_grid[30, 30]]])
        res = migrator.migrate(travel_time_grid=T_grid, survey_points=sample_pts, velocity=self.v, show_progress=False)
        mig_pts = res.migrated_points

        x_mig, y_mig = mig_pts[0, 0], mig_pts[0, 1]
        dx_mig = x_mig - sample_x
        expected_dx = -bx * h_0 / (1.0 + bx**2)  # -0.1 * 80 / 1.01 = -7.92 m

        np.testing.assert_allclose(dx_mig, expected_dx, atol=0.1)


if __name__ == "__main__":
    unittest.main()

