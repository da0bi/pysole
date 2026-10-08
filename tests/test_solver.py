"""
Integration test for full Solver workflow using Wurtenkees Glacier dataset.
"""

import unittest
import tempfile
import pathlib
import os
import numpy as np
import pysole


class TestSolverWuk(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.data_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.data_dir, "wuk_outline_clean.csv")
        self.wuk_survey = os.path.join(self.data_dir, "wuk_survey_clean.csv")

        # All outputs (logs, figures, rasters) must go to a throw-away directory, never to the repo / examples.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = os.path.join(self._tmp.name, "out")

    def test_full_solver_workflow_wuk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)

            model = pysole.Solver(dem=self.wuk_dem, outline=self.wuk_outline, perform_migration=True, output_dir=self.output_dir)
            self.assertEqual(model.dem_grid.shape, (179, 213))
            self.assertEqual(model.dx, 5.0)
            self.assertEqual(model.dy, 5.0)

            mig_pts = model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
            self.assertGreater(mig_pts.shape[0], 900)

            opt_kc = model.optimize_bss(kc_max=0.3, kc_min=0.01, n_steps=5)
            self.assertGreater(opt_kc, 0)

            bedrock_map = model.finalize_topography(
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
        model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
        model.optimize_bss(kc_min=0.01, kc_max=0.3, n_steps=5)
        model.interpolate_kriging()

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
        model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
        model.optimize_bss(kc_min=0.01, kc_max=0.3, n_steps=5)
        bedrock_map = model.finalize_topography(interactive=False, random_forest_gap_filling=False, apply_margin_blend=False)

        self.assertIsInstance(bedrock_map, pysole.BedrockMap)
        self.assertTrue(np.all(np.isfinite(model.kriged_bedrock)))

    def test_bedrock_dem_smoothing_wuk(self):
        dem, meta = pysole.load_dem(self.wuk_dem)
        model = pysole.Solver(dem=self.wuk_dem, outline=self.wuk_outline, output_dir=self.output_dir)

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

        model.interpolate_kriging()
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


if __name__ == "__main__":
    unittest.main()
