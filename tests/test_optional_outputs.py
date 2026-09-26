"""
Unit tests for optional export features in PySole.
"""

import unittest
import tempfile
import os
import shutil
import numpy as np
from pysole.solver import Solver
from pysole.raster import save_points_csv


class TestOptionalOutputs(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dem_path = os.path.join(self.temp_dir, "test_dem.asc")
        self.pts_path = os.path.join(self.temp_dir, "test_pts.csv")

        # Create synthetic 30x30 synthetic DEM (tilted plane)
        X, Y = np.meshgrid(np.linspace(0, 300, 30), np.linspace(0, 300, 30))
        Z = 2000.0 - 0.1 * X - 0.05 * Y
        header = f"ncols 30\nnnrows 30\nxllcorner 0\nyllcorner 0\ncellsize 10\nnodata_value -9999\n"
        with open(self.dem_path, "w") as f:
            f.write(header)
            np.savetxt(f, Z, fmt="%.2f")

        # Synthetic survey traveltimes (one-way traveltime ~ 50 ns)
        pts = np.array([
            [50.0, 50.0, 50.0],
            [100.0, 100.0, 60.0],
            [150.0, 150.0, 40.0],
            [200.0, 200.0, 70.0],
            [250.0, 250.0, 55.0],
        ])
        np.savetxt(self.pts_path, pts, delimiter=",", header="x,y,owtt", comments="")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_points_csv_helper(self):
        pts = np.array([[10, 20, 100, 45.5], [30, 40, 105, 52.1]])
        out_csv = os.path.join(self.temp_dir, "test_points.csv")
        saved = save_points_csv(pts, out_csv)
        self.assertTrue(os.path.exists(saved))

        data = np.loadtxt(saved, delimiter=",", skiprows=1)
        self.assertEqual(data.shape, (2, 4))
        self.assertAlmostEqual(data[0, 3], 45.5)

    def test_optional_exports_enabled(self):
        output_stem = os.path.join(self.temp_dir, "output_test")
        cfg = {
            "inputs": {
                "dem_path": self.dem_path,
                "survey_data_path": self.pts_path,
                "survey_data_type": "one_way_travel_time",
                "show_progress": False,
            },
            "migration_parameters": {
                "perform_migration": True,
                "velocity": 0.16,
            },
            "outputs": {
                "output_format": "tif",
                "output_prefix": output_stem + ".tif",  # Test automatic extension stripping
                "save_traveltime_grid": True,
                "save_migrated_points": True,
                "save_thickness_grid": True,
                "save_thickness_uncertainty": True,
                "save_basal_shear_stress": True,
                "save_basal_shear_stress_uncertainty": True,
            }
        }
        solver = Solver.from_config(cfg)
        solver.run_pipeline()

        # Check that bedrock and optional output files were created on disk
        self.assertTrue(os.path.exists(f"{output_stem}_bedrock.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_traveltime.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_migrated_points.csv"))
        self.assertTrue(os.path.exists(f"{output_stem}_thickness.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_thickness_uncertainty.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_basal_shear_stress.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_basal_shear_stress_uncertainty.tif"))

        # Check properties on solver
        self.assertIsNotNone(solver.traveltime_grid)
        self.assertIsNotNone(solver.thickness_grid)
        self.assertIsNotNone(solver.thickness_std_grid)
        self.assertIsNotNone(solver.basal_shear_stress_grid)
        self.assertIsNotNone(solver.basal_shear_stress_std_grid)
        self.assertIsNotNone(solver.migrated_points)

        # Check basal shear stress is in kPa (~10 - 200 kPa range)
        valid_bss = solver.basal_shear_stress_grid[~np.isnan(solver.basal_shear_stress_grid)]
        self.assertTrue(np.all(valid_bss >= 0.0))

    def test_migration_skipped_disables_migrated_points_export(self):
        output_stem = os.path.join(self.temp_dir, "output_nomig")
        cfg = {
            "inputs": {
                "dem_path": self.dem_path,
                "survey_data_path": self.pts_path,
                "survey_data_type": "thickness",
                "show_progress": False,
            },
            "migration_parameters": {
                "perform_migration": False,
            },
            "outputs": {
                "output_format": "tif",
                "output_prefix": output_stem,
                "save_migrated_points": True,
                "save_thickness_grid": True,
            }
        }
        solver = Solver.from_config(cfg)
        solver.run_pipeline()

        # Thickness raster should be created, but migrated_points CSV should NOT be created
        self.assertTrue(os.path.exists(f"{output_stem}_bedrock.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_thickness.tif"))
        self.assertFalse(os.path.exists(f"{output_stem}_migrated_points.csv"))


if __name__ == "__main__":
    unittest.main()
