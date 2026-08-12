"""
Unit tests for 3D Eikonal Ray Migration using Wurtenkees Glacier dataset.
"""

import unittest
import os
import numpy as np
from pysole.raster import load_dem, load_outline, ensure_spatial_coords
from pysole.migration import migrate_eikonal_points


class TestMigrationWuk(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.base_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.base_dir, "wuk_outline_clean.csv")
        self.wuk_survey = os.path.join(self.base_dir, "wuk_survey_clean.csv")

    def test_migrate_eikonal_points_wuk(self):
        dem, meta = load_dem(self.wuk_dem)
        outline_mask = load_outline(self.wuk_outline, dem, meta)
        x_coords, y_coords, bounds = ensure_spatial_coords(dem.shape, dx=meta["dx"], dy=meta["dy"], bounds=meta["bounds"])

        # Load clean numeric survey picks [X, Y, Z_surf, OWTT_ns]
        survey = np.loadtxt(self.wuk_survey, delimiter=",", skiprows=1)

        # Create traveltime grid for 3D ray migration
        tt_grid = np.zeros_like(dem)

        migrated = migrate_eikonal_points(
            dem=dem,
            travel_time_grid=tt_grid,
            survey_points=survey,
            velocity=0.16,
            dx=meta["dx"],
            dy=meta["dy"],
            x_coords=x_coords,
            y_coords=y_coords,
            outline_mask=outline_mask,
            bounds=bounds,
            plots_dir=None,
            interactive=False,
        )

        self.assertEqual(migrated.shape[0], survey.shape[0])
        self.assertEqual(migrated.shape[1], 4)
        self.assertTrue(np.all(migrated[:, 3] >= 0))


if __name__ == "__main__":
    unittest.main()
