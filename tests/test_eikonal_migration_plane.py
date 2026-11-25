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

    def test_ray_metrics_horizontal_ds_azimuth_and_ray_angle(self):
        """Verifies horizontal_ds, cardinal azimuth bearings (0° N, 90° E, 180° S, 270° W), and ray_angle."""
        dem = np.full_like(self.xx, 2000.0)
        h_0 = 80.0
        dip_deg = 20.0
        b = np.tan(np.radians(dip_deg))

        # Test all 4 cardinal shift directions by tilting the bed along -Y (North shift, 0°),
        # -X (East shift, 90°), +Y (South shift, 180°), and +X (West shift, 270°)
        cases = [
            (0.0, -b, 0.0),    # dT/dy < 0 -> shift +Y (North = 0°)
            (-b, 0.0, 90.0),   # dT/dx < 0 -> shift +X (East = 90°)
            (0.0, b, 180.0),   # dT/dy > 0 -> shift -Y (South = 180°)
            (b, 0.0, 270.0),   # dT/dx > 0 -> shift -X (West = 270°)
        ]
        migrator = EikonalMigrator(dem, geometry=self.geometry)
        sample_x, sample_y = 150.0, 150.0

        for bx, by, expected_az in cases:
            H_grid = (h_0 + bx * (self.xx - sample_x) + by * (self.yy - sample_y)) / np.sqrt(1.0 + bx**2 + by**2)
            T_grid = H_grid / self.v
            t_0 = (h_0 / np.sqrt(1.0 + bx**2 + by**2)) / self.v
            sample_pts = np.array([[sample_x, sample_y, 2000.0, t_0]])

            res = migrator.migrate(travel_time_grid=T_grid, survey_points=sample_pts, velocity=self.v, show_progress=False)
            self.assertEqual(res.all_migrated_points.shape, (1, 7))
            expected_ds = (h_0 / np.sqrt(1.0 + bx**2 + by**2)) * np.sin(np.radians(dip_deg))
            np.testing.assert_allclose(res.all_migrated_points[0, 4], expected_ds, atol=0.1)
            np.testing.assert_allclose(res.all_migrated_points[0, 5], expected_az, atol=0.1)
            np.testing.assert_allclose(res.all_migrated_points[0, 6], dip_deg, atol=0.1)

    def test_max_ray_angle_discards_high_angle_picks_and_keeps_all_in_export_table(self):
        """Verifies max_ray_angle_deg discards steep picks for Pass 2, logs discard count, and keeps all picks in all_migrated_points."""
        import tempfile
        import os

        dem = np.full_like(self.xx, 2000.0)
        # Left half of grid has gentle dip (10°), right half has steep dip (65°)
        b_gentle = np.tan(np.radians(10.0))
        b_steep = np.tan(np.radians(65.0))
        H_gentle = (80.0 + b_gentle * (self.xx - 75.0)) / np.sqrt(1.0 + b_gentle**2)
        H_steep = (80.0 + b_steep * (self.xx - 225.0)) / np.sqrt(1.0 + b_steep**2)
        T_grid = np.where(self.xx < 150.0, H_gentle, H_steep) / self.v

        # Two survey picks with profile IDs (column 4): pick 0 in gentle zone (10°), pick 1 in steep zone (65°)
        t_gentle = (80.0 / np.sqrt(1.0 + b_gentle**2)) / self.v
        t_steep = (80.0 / np.sqrt(1.0 + b_steep**2)) / self.v
        sample_pts = np.array([
            [75.0, 150.0, 2000.0, t_gentle, 1.0],
            [225.0, 150.0, 2000.0, t_steep, 2.0],
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            for viz_mode in ("arrows", "grad_traveltime_grid"):
                with self.assertLogs("pysole", level="INFO") as cm:
                    res = migrate_eikonal_points(
                        dem=dem,
                        travel_time_grid=T_grid,
                        survey_points=sample_pts,
                        geometry=self.geometry,
                        velocity=self.v,
                        max_ray_angle_deg=60.0,
                        plots_dir=tmpdir,
                        show_progress=False,
                        horizontal_displacement_visualization=viz_mode,
                    )
                self.assertTrue(
                    any("Discarded 1 of 2 survey picks exceeding max_ray_angle_deg = 60.0°" in msg for msg in cm.output),
                    f"Expected discard log message not found in {cm.output}",
                )
                # Retained migrated_points has 1 pick and 5 columns (with profile_id at index 4)
                self.assertEqual(res.migrated_points.shape, (1, 5))
                self.assertEqual(res.migrated_points[0, 4], 1.0)
                # Export table all_migrated_points retains ALL 2 picks and 8 columns (with profile_id at index 7)
                self.assertEqual(res.all_migrated_points.shape, (2, 8))
                self.assertEqual(res.all_migrated_points[0, 7], 1.0)
                self.assertEqual(res.all_migrated_points[1, 7], 2.0)
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "03_3d_migration.png")))


if __name__ == "__main__":
    unittest.main()

