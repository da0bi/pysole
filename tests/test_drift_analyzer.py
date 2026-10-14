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
from pysole.drift import RASTER_DRIFT_TERMS, build_drift_suite


class TestDriftAnalyzer(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.data_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.data_dir, "wuk_outline_clean.csv")
        self.wuk_survey = os.path.join(self.data_dir, "wuk_survey_clean.csv")

        # Keep logs / figures / rasters out of the repository tree
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = os.path.join(self._tmp.name, "out")

    def test_drift_analyzer_initialization_and_evaluation(self):
        model = pysole.Solver(
            dem=self.wuk_dem,
            outline=self.wuk_outline,
            perform_migration=True,
            output_dir=self.output_dir,
        )
        mig_pts = model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
        model.optimize_bss(kc_min=0.01, kc_max=0.3, n_steps=5)

        analyzer = DriftAnalyzer(
            mode="post_migration",
            target_name="depth",
            interpolation_target="D",
            include_zero_boundary=True,
            survey_profile_column=None,
            show_progress=False,
        )

        results = analyzer.run_diagnostics(
            x_pts=mig_pts[:, 0],
            y_pts=mig_pts[:, 1],
            z_values=mig_pts[:, 2],
            geometry=model.geometry,
            drift_grids=model.get_drift_grids(RASTER_DRIFT_TERMS),
            candidates=build_drift_suite(None),
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
            output_dir=self.output_dir,
        )
        mig_pts = model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
        profile_ids = np.random.choice([1, 2, 3, 4], size=len(mig_pts))

        analyzer = DriftAnalyzer(
            mode="post_migration",
            target_name="depth",
            interpolation_target="D",
            include_zero_boundary=True,
            survey_profile_column="profile_id",
            show_progress=False,
        )

        results = analyzer.run_diagnostics(
            x_pts=mig_pts[:, 0],
            y_pts=mig_pts[:, 1],
            z_values=mig_pts[:, 2],
            geometry=model.geometry,
            drift_grids=model.get_drift_grids(RASTER_DRIFT_TERMS),
            candidates=build_drift_suite(None),
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
            output_dir=self.output_dir,
        )
        model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
        model.optimize_bss(kc_min=0.01, kc_max=0.3, n_steps=5)

        # Test single-computation caching via get_sample_points
        pts1 = model.get_sample_points("post_migration", "P")
        pts2 = model.get_sample_points("post_migration", "P")
        self.assertIs(pts1, pts2)  # Identical cached object

        recommended = model.recommend_drift_model(stage="post_migration", interactive=False)
        self.assertIsInstance(recommended, list)
        self.assertGreater(len(recommended), 0)

        # sia on the Product P target carries a warning but is ranked normally (not penalised)
        sia_res = next(r for r in recommended if r.name == "sia")
        self.assertFalse(sia_res.is_penalized)
        self.assertIsNotNone(sia_res.warning_msg)

        # The ranking exposes its deltas: the best non-penalised model has dAICc == 0, the winner dRMSE% == 0
        self.assertEqual(recommended[0].delta_rmse_pct, 0.0)
        best = [r for r in recommended if not r.is_penalized]
        self.assertEqual(min(r.delta_aicc for r in best), 0.0)


if __name__ == "__main__":
    unittest.main()
