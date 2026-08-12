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
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.base_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.base_dir, "wuk_outline_clean.csv")
        self.wuk_survey = os.path.join(self.base_dir, "wuk_survey_clean.csv")

    def test_full_solver_workflow_wuk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)

            model = pysole.Solver(dem=self.wuk_dem, outline=self.wuk_outline, perform_migration=True)
            self.assertEqual(model.dem_grid.shape, (179, 213))
            self.assertEqual(model.dx, 5.0)
            self.assertEqual(model.dy, 5.0)

            mig_pts = model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
            self.assertGreater(mig_pts.shape[0], 900)

            opt_kc = model.optimize_bss(kc_max=10.0, kc_min=0.01, d_kc=0.1)
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

    def test_sia_thickness_kriging_drift_wuk(self):
        model = pysole.Solver(
            dem=self.wuk_dem,
            outline=self.wuk_outline,
            post_kriging_method="universal",
            post_drift_terms=["sia_thickness"],
            perform_migration=True,
        )
        model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
        model.optimize_bss(kc_min=0.01, kc_max=10.0, d_kc=0.1)
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
        )
        model.migrate_eikonal(travel_times=self.wuk_survey, velocity=0.16)
        model.optimize_bss(kc_min=0.01, kc_max=10.0, d_kc=0.1)
        bedrock_map = model.finalize_topography(interactive=False, random_forest_gap_filling=False, apply_margin_blend=False)

        self.assertIsInstance(bedrock_map, pysole.BedrockMap)
        self.assertTrue(np.all(np.isfinite(model.kriged_bedrock)))

    def test_bedrock_dem_smoothing_wuk(self):
        dem, meta = pysole.load_dem(self.wuk_dem)
        model = pysole.Solver(dem=self.wuk_dem, outline=self.wuk_outline)

        g_smoothed = model.smooth_bedrock_dem(dem, method="gaussian", sigma=1.5)
        self.assertEqual(g_smoothed.shape, dem.shape)
        self.assertFalse(np.isnan(g_smoothed).any())

        m_smoothed = model.smooth_bedrock_dem(dem, method="median", kernel_size=3)
        self.assertEqual(m_smoothed.shape, dem.shape)
        self.assertFalse(np.isnan(m_smoothed).any())

        fft_smoothed = model.smooth_bedrock_dem(dem, method="fft_lowpass", kc_cutoff=0.1)
        self.assertEqual(fft_smoothed.shape, dem.shape)
        self.assertFalse(np.isnan(fft_smoothed).any())


if __name__ == "__main__":
    unittest.main()
