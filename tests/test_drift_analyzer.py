"""
Unit tests for Universal Kriging Drift Analyzer module.
"""

import unittest
import tempfile
import pathlib
import os
import numpy as np
import pandas as pd
import pysole
from pysole.drift_analyzer import DriftAnalyzer


class TestDriftAnalyzer(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.data_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.data_dir, "wuk_outline_clean.csv")
        self.wuk_survey = os.path.join(self.data_dir, "wuk_survey_clean.csv")

    def test_drift_analyzer_initialization_and_evaluation(self):
        model = pysole.Solver(
            dem=self.wuk_dem,
            outline=self.wuk_outline,
            perform_migration=True,
        )
        mig_pts = model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
        model.optimize_bss(kc_min=0.01, kc_max=10.0, d_kc=2.0)

        analyzer = DriftAnalyzer(
            mode="post_migration",
            target_name="depth",
            interpolation_target="D",
            include_zero_boundary=True,
            survey_profile_column=None,
        )

        opt_alpha = np.degrees(model.opt_slope) if model.opt_slope is not None else np.zeros_like(model.dem_grid)

        results = analyzer.run_diagnostics(
            x_pts=mig_pts[:, 0],
            y_pts=mig_pts[:, 1],
            z_values=mig_pts[:, 2],
            dem_grid=model.dem_grid,
            dx=model.dx,
            dy=model.dy,
            bounds=model.bounds,
            alpha_opt_deg=opt_alpha,
            variogram_model="spherical",
            variogram_params={"range": 100.0, "sill": 1.0, "nugget": 0.0},
            interactive=False,
        )

        self.assertGreater(len(results), 0)
        self.assertIsNotNone(results[0].name)

    def test_lopo_cv_with_profile_col(self):
        model = pysole.Solver(
            dem=self.wuk_dem,
            outline=self.wuk_outline,
            perform_migration=True,
        )
        mig_pts = model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
        profile_ids = np.random.choice([1, 2, 3, 4], size=len(mig_pts))

        analyzer = DriftAnalyzer(
            mode="post_migration",
            target_name="depth",
            interpolation_target="D",
            include_zero_boundary=True,
            survey_profile_column="profile_id",
        )

        opt_alpha = np.degrees(model.opt_slope) if model.opt_slope is not None else np.zeros_like(model.dem_grid)

        results = analyzer.run_diagnostics(
            x_pts=mig_pts[:, 0],
            y_pts=mig_pts[:, 1],
            z_values=mig_pts[:, 2],
            dem_grid=model.dem_grid,
            dx=model.dx,
            dy=model.dy,
            bounds=model.bounds,
            alpha_opt_deg=opt_alpha,
            variogram_model="spherical",
            variogram_params={"range": 100.0, "sill": 1.0, "nugget": 0.0},
            profile_data=profile_ids,
            interactive=False,
        )

        self.assertGreater(len(results), 0)

    def test_solver_recommend_drift_model(self):
        model = pysole.Solver(
            dem=self.wuk_dem,
            outline=self.wuk_outline,
            perform_migration=True,
        )
        model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
        model.optimize_bss(kc_min=0.01, kc_max=10.0, d_kc=2.0)

        # Test single-computation caching via get_sample_points
        pts1 = model.get_sample_points("post_migration", "P")
        pts2 = model.get_sample_points("post_migration", "P")
        self.assertIs(pts1, pts2)  # Identical cached object

        recommended = model.recommend_drift_model(stage="post_migration", interactive=False)
        self.assertIsInstance(recommended, list)
        self.assertGreater(len(recommended), 0)

        # Verify sia slope drift receives is_penalized=True on Product P target
        sia_res = next((r for r in recommended if r.name == "sia"), None)
        if sia_res is not None:
            self.assertTrue(sia_res.is_penalized)


if __name__ == "__main__":
    unittest.main()
