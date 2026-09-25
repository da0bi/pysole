"""
Unit tests for pysole.json configuration parsing and run_from_config workflow using Wurtenkees Glacier dataset.
"""

import unittest
import tempfile
import pathlib
import os
import numpy as np
import pysole


class TestConfigWorkflowWuk(unittest.TestCase):
    def test_run_from_config_wuk(self):
        wuk_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/pysole_wuk.json"))

        # Execute complete workflow from Wurtenkees Glacier config
        solver = pysole.Solver.from_config(wuk_config)
        self.assertEqual(solver.dem_grid.shape, (179, 213))
        self.assertEqual(solver.dx, 5.0)
        self.assertEqual(solver.dy, 5.0)
        self.assertEqual(solver.n_cores, -1)
        self.assertTrue(solver.built_in_kriging)

        gok_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/gok/pysole_gok.json"))
        solver_gok = pysole.Solver.from_config(gok_config)
        self.assertEqual(solver_gok.n_cores, -1)
        self.assertTrue(solver_gok.built_in_kriging)

        bedrock_map = pysole.run_from_config(wuk_config)
        self.assertIsInstance(bedrock_map, pysole.BedrockMap)
        self.assertEqual(bedrock_map.shape, (179, 213))
        self.assertTrue(np.all(np.isfinite(bedrock_map.grid)))

    def test_save_multi_format_all(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            grid = np.ones((10, 10)) * 500.0
            bedrock_map = pysole.BedrockMap(grid=grid, bounds=(0.0, 0.0, 50.0, 50.0))

            base_file = str(tmp_path / "test_export.tif")
            saved = bedrock_map.save(base_file, formats="all")

            self.assertIsInstance(saved, list)
            self.assertEqual(len(saved), 4)

            expected_exts = [".tif", ".asc", ".csv", ".npy"]
            for ext in expected_exts:
                expected_file = str(tmp_path / f"test_export{ext}")
                self.assertTrue(os.path.exists(expected_file), f"Missing exported file: {expected_file}")

    def test_null_input_paths_default(self):
        cfg_inputs = pysole.config.DEFAULT_CONFIG["inputs"]
        self.assertIsNone(cfg_inputs["dem_path"])
        self.assertIsNone(cfg_inputs["outline_path"])
        self.assertIsNone(cfg_inputs["survey_data_path"])
        self.assertIsNone(cfg_inputs["base_dir"])

        cfg_outputs = pysole.config.DEFAULT_CONFIG["outputs"]
        self.assertEqual(cfg_outputs["output_name"], "final_bedrock")
        self.assertEqual(cfg_outputs["output_format"], "tif")
        self.assertEqual(cfg_outputs["plots_dir"], "figures")

    def test_base_dir_resolution(self):
        dem = np.ones((10, 10))
        # 1. Fallback to survey_data_path directory when base_dir is None (plots_dir defaults to figures)
        solver_fallback = pysole.Solver(dem=dem, survey_data_path="/tmp/test_workspace/survey.csv")
        self.assertEqual(solver_fallback.effective_base_dir, "/tmp/test_workspace")
        self.assertEqual(solver_fallback.plots_dir, "/tmp/test_workspace/figures")
        self.assertEqual(solver_fallback.resolve_path("final_bedrock"), "/tmp/test_workspace/final_bedrock")

        # 2. Explicit base_dir overrides survey_data_path directory
        solver_explicit = pysole.Solver(
            dem=dem,
            base_dir="/custom/output_dir",
            survey_data_path="/tmp/test_workspace/survey.csv",
            plots_dir="figures",
        )
        self.assertEqual(solver_explicit.effective_base_dir, "/custom/output_dir")
        self.assertEqual(solver_explicit.plots_dir, "/custom/output_dir/figures")
        self.assertEqual(solver_explicit.resolve_path("final_bedrock"), "/custom/output_dir/final_bedrock")

        # 3. Absolute path overrides base_dir
        self.assertEqual(solver_explicit.resolve_path("/abs/path/bedrock"), "/abs/path/bedrock")


if __name__ == "__main__":
    unittest.main()
