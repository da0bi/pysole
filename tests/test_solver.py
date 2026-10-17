"""
Integration test for full Solver workflow using Wurtenkees Glacier dataset.
"""

import unittest
from unittest.mock import patch
import tempfile
import pathlib
import os
import numpy as np
import pysole


class TestSolverWuk(unittest.TestCase):
    _shared_migrated_pts = None

    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.data_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.data_dir, "wuk_outline_clean.csv")
        self.wuk_survey = os.path.join(self.data_dir, "wuk_survey_clean.csv")

        # All outputs (logs, figures, rasters) must go to a throw-away directory, never to the repo / examples.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = os.path.join(self._tmp.name, "out")

    def _get_shared_migrated_points(self):
        if TestSolverWuk._shared_migrated_pts is None:
            m = pysole.Solver(dem=self.wuk_dem, outline=self.wuk_outline, perform_migration=True, output_dir=self.output_dir)
            TestSolverWuk._shared_migrated_pts = m.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16).copy()
        return TestSolverWuk._shared_migrated_pts.copy()

    def test_full_solver_workflow_wuk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)

            model = pysole.Solver(dem=self.wuk_dem, outline=self.wuk_outline, perform_migration=True, output_dir=self.output_dir)
            self.assertEqual(model.dem_grid.shape, (179, 213))
            self.assertEqual(model.dx, 5.0)
            self.assertEqual(model.dy, 5.0)

            mig_pts = model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
            TestSolverWuk._shared_migrated_pts = mig_pts.copy()
            self.assertGreater(mig_pts.shape[0], 900)

            opt_kc = model.optimize_bss(kc_max=0.3, kc_min=0.01, n_steps=5)
            self.assertGreater(opt_kc, 0)

            bedrock_map = model.finalize_bedrock(
                interactive=False,
                random_forest_gap_filling=False,
                apply_margin_blend=True,
                min_gap_dist=30.0,
                smooth_bedrock=True,
                smoothing_method="gaussian",
                smoothing_sigma=1.5,
            )
            self.assertIsInstance(bedrock_map, pysole.BedrockMap)
            self.assertEqual(bedrock_map.shape, (179, 213))

            out_file = str(tmp_path / "final_bedrock.csv")
            bedrock_map.save(out_file)
            self.assertTrue(os.path.exists(out_file))

    def test_sia_kriging_drift_wuk(self):
        model = pysole.Solver(
            dem=self.wuk_dem,
            outline=self.wuk_outline,
            post_kriging_method="universal",
            post_drift_terms=["sia"],
            perform_migration=True,
            output_dir=self.output_dir,
        )
        model.migrated_points = self._get_shared_migrated_points()
        model.optimize_bss(kc_min=0.01, kc_max=0.3, n_steps=5)
        model.calculate_bedrock()

        self.assertIsNotNone(model.kriged_bedrock)
        self.assertEqual(model.kriged_bedrock.shape, (179, 213))
        self.assertTrue(np.all(np.isfinite(model.kriged_bedrock)))

    def test_zero_boundary_condition_wuk(self):
        model = pysole.Solver(
            dem=self.wuk_dem,
            outline=self.wuk_outline,
            pre_zero_boundary=True,
            post_zero_boundary=True,
            perform_migration=True,
            output_dir=self.output_dir,
        )
        model.migrated_points = self._get_shared_migrated_points()
        model.optimize_bss(kc_min=0.01, kc_max=0.3, n_steps=5)
        bedrock_map = model.finalize_bedrock(interactive=False, random_forest_gap_filling=False, apply_margin_blend=False)

        self.assertIsInstance(bedrock_map, pysole.BedrockMap)
        self.assertTrue(np.all(np.isfinite(model.kriged_bedrock)))

    def test_bedrock_dem_smoothing(self):
        dem = np.full((30, 30), 2000.0, dtype=np.float64)
        dem[10:20, 10:20] += 50.0
        model = pysole.Solver(dem=dem, dx=10.0, dy=10.0, bounds=(0, 0, 300, 300), output_dir=self.output_dir)

        g_smoothed = model.smooth_bedrock_dem(dem, method="gaussian", sigma=1.5)
        self.assertEqual(g_smoothed.shape, dem.shape)
        self.assertFalse(np.isnan(g_smoothed).any())

        m_smoothed = model.smooth_bedrock_dem(dem, method="median", kernel_size=3)
        self.assertEqual(m_smoothed.shape, dem.shape)
        self.assertFalse(np.isnan(m_smoothed).any())

    def test_direct_interpolation_targets_wuk(self):
        model = pysole.Solver(
            dem=self.wuk_dem,
            outline=self.wuk_outline,
            pre_interpolation_target="T",
            post_interpolation_target="D",
            perform_migration=True,
            output_dir=self.output_dir,
        )
        self.assertEqual(model.pre_interpolation_target, "T")
        self.assertEqual(model.post_interpolation_target, "D")

        model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
        self.assertIsNotNone(model.traveltime_grid)

        model.calculate_bedrock()
        self.assertIsNotNone(model.kriged_bedrock)
        self.assertEqual(model.kriged_bedrock.shape, (179, 213))
        self.assertTrue(np.all(np.isfinite(model.kriged_bedrock)))

    def test_solver_summary_and_results(self):
        model = pysole.Solver(dem=self.wuk_dem, outline=self.wuk_outline, output_dir=self.output_dir)
        summary_txt = model.summary()
        self.assertIn("PYSOLE SOLVER EXECUTION SUMMARY", summary_txt)
        res_dict = model.results
        self.assertIsInstance(res_dict, dict)
        self.assertIn("bedrock_map", res_dict)
        self.assertIn("thickness_grid", res_dict)

    def test_missing_dem_path_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            pysole.Solver.from_config({"inputs": {"dem_path": None}, "outputs": {"output_dir": self.output_dir}})
        self.assertIn("Missing required 'dem_path'", str(ctx.exception))

    @patch("sys.stdin.isatty", return_value=True)
    def test_interactive_bedrock_finalization_defaults(self, mock_isatty):
        dem = np.full((20, 20), 2000.0, dtype=np.float64)
        model = pysole.Solver(dem=dem, dx=10.0, dy=10.0, bounds=(0, 0, 200, 200), output_dir=self.output_dir)
        model.kriged_thickness = np.full((20, 20), 50.0)
        model.opt_slope = np.full((20, 20), 0.1)
        model.plots_dir = self.output_dir

        with patch("builtins.input", side_effect=["", "", "", "y"]):
            bedrock_map = model.finalize_bedrock(interactive=True)

        self.assertIsInstance(bedrock_map, pysole.BedrockMap)
        fig07 = os.path.join(self.output_dir, "07_final_embedded_bedrock.png")
        self.assertTrue(os.path.exists(fig07), "Figure 07 should be refreshed and saved to disk")

    @patch("sys.stdin.isatty", return_value=True)
    def test_interactive_bedrock_finalization_readjust_loop(self, mock_isatty):
        dem = np.full((20, 20), 2000.0, dtype=np.float64)
        model = pysole.Solver(dem=dem, dx=10.0, dy=10.0, bounds=(0, 0, 200, 200), output_dir=self.output_dir)
        model.kriged_thickness = np.full((20, 20), 50.0)
        model.opt_slope = np.full((20, 20), 0.1)
        model.plots_dir = self.output_dir

        # Pass 1: "y" -> smoothing, "gaussian", "2.0", "n" -> RF, "y" -> margin, "30.0", "r" -> re-adjust!
        # Pass 2: "n" -> smoothing, "n" -> RF, "n" -> margin, "y" -> accept!
        inputs = [
            "y", "gaussian", "2.0", "n", "y", "30.0", "r",
            "n", "n", "n", "y"
        ]
        with patch("builtins.input", side_effect=inputs):
            bedrock_map = model.finalize_bedrock(interactive=True)

        self.assertIsInstance(bedrock_map, pysole.BedrockMap)
        fig07 = os.path.join(self.output_dir, "07_final_embedded_bedrock.png")
        self.assertTrue(os.path.exists(fig07), "Figure 07 should be updated and saved to disk")

    def test_non_interactive_bedrock_finalization_never_prompts(self):
        dem = np.full((20, 20), 2000.0, dtype=np.float64)
        model = pysole.Solver(dem=dem, dx=10.0, dy=10.0, bounds=(0, 0, 200, 200), output_dir=self.output_dir)
        model.kriged_thickness = np.full((20, 20), 50.0)
        model.opt_slope = np.full((20, 20), 0.1)
        model.plots_dir = self.output_dir

        def fail_if_called(*args, **kwargs):
            raise AssertionError("input() should never be called in non-interactive mode")

        with patch("builtins.input", side_effect=fail_if_called):
            bedrock_map = model.finalize_bedrock(interactive=False)

        self.assertIsInstance(bedrock_map, pysole.BedrockMap)

    def test_positive_histogram_and_domain_means(self):
        dem = np.full((10, 10), 2000.0, dtype=np.float64)
        model = pysole.Solver(dem=dem, dx=10.0, dy=10.0, bounds=(0, 0, 100, 100), output_dir=self.output_dir)
        # 50 cells with thickness 100.0 m, 50 cells with thickness 0.0 m (e.g. margin/outcrops)
        thick = np.full((10, 10), 0.0, dtype=np.float64)
        thick[:5, :] = 100.0
        model.kriged_thickness = thick
        model.opt_slope = np.full((10, 10), 0.1)
        model.plots_dir = self.output_dir

        model.finalize_bedrock(interactive=False, apply_margin_blend=False)

        # Domain mean thickness must be 100.0 m (excluding 0.0 cells)
        # If zeros were included, it would be diluted to 50.0 m.
        summary_txt = model.summary()
        self.assertIn("Ice Thickness Range  : min=100.0 m, mean=100.0 m, max=100.0 m", summary_txt)

        # Basal Shear Stress mean must also be computed strictly over positive cells
        expected_bss = (model.ice_density * model.g * 100.0 * np.sin(0.1)) / 1000.0
        self.assertIn(f"mean={expected_bss:.1f} kPa", summary_txt)


if __name__ == "__main__":
    unittest.main()
