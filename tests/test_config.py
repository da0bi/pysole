"""
Unit tests for pysole.json configuration parsing and run_from_config workflow using Wurtenkees Glacier dataset.
"""

import json
import unittest
import tempfile
import pathlib
import os
import numpy as np
import pysole


class TestConfigWorkflowWuk(unittest.TestCase):
    @staticmethod
    def _config_with_temp_outputs(cfg_path: str, out_dir: str) -> str:
        """Copies an example config into out_dir with absolute input paths and outputs.output_dir = out_dir."""
        cfg_p = pathlib.Path(cfg_path)
        with open(cfg_p, "r") as f:
            cfg = json.load(f)
        for key in ("dem_path", "outline_path", "survey_data_path"):
            val = cfg["inputs"].get(key)
            if val:
                cfg["inputs"][key] = str((cfg_p.parent / val).resolve())
        cfg.setdefault("outputs", {})["output_dir"] = str(out_dir)
        new_path = pathlib.Path(out_dir) / cfg_p.name
        with open(new_path, "w") as f:
            json.dump(cfg, f)
        return str(new_path)

    def test_run_from_config_wuk(self):
        wuk_config_orig = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/pysole_wuk.json"))
        gok_config_orig = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/gok/pysole_gok.json"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            wuk_config = self._config_with_temp_outputs(wuk_config_orig, tmp_dir)

            # Execute complete workflow from Wurtenkees Glacier config
            solver = pysole.Solver.from_config(wuk_config)
            self.assertEqual(solver.dem_grid.shape, (179, 213))
            self.assertEqual(solver.dx, 5.0)
            self.assertEqual(solver.dy, 5.0)
            self.assertEqual(solver.n_cores, -1)

            gok_dir = os.path.join(tmp_dir, "gok")
            os.makedirs(gok_dir)
            gok_config = self._config_with_temp_outputs(gok_config_orig, gok_dir)
            solver_gok = pysole.Solver.from_config(gok_config)
            self.assertEqual(solver_gok.n_cores, -1)

            bedrock_map = pysole.run_from_config(wuk_config, is_batch=True)
            self.assertIsInstance(bedrock_map, pysole.BedrockMap)
            self.assertEqual(bedrock_map.shape, (179, 213))
            self.assertTrue(np.all(np.isfinite(bedrock_map.grid)))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "wuk_final_bedrock.tif")))

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

        cfg_outputs = pysole.config.DEFAULT_CONFIG["outputs"]
        self.assertIsNone(cfg_outputs["output_dir"])
        self.assertEqual(cfg_outputs["output_prefix"], "final")
        self.assertEqual(cfg_outputs["output_format"], "tif")
        self.assertEqual(cfg_outputs["plots_dir"], "figures")

    def test_output_dir_resolution(self):
        dem = np.ones((10, 10))
        # 1. Fallback to survey_data_path directory when output_dir is None (plots_dir defaults to figures)
        solver_fallback = pysole.Solver(dem=dem, survey_data_path="/tmp/test_workspace/survey.csv")
        self.assertEqual(solver_fallback.effective_output_dir, "/tmp/test_workspace/pysole")
        self.assertEqual(solver_fallback.plots_dir, "/tmp/test_workspace/pysole/figures")
        self.assertEqual(solver_fallback.resolve_path("final_bedrock"), "/tmp/test_workspace/pysole/final_bedrock")

        # 2. Explicit output_dir overrides survey_data_path directory
        with tempfile.TemporaryDirectory() as custom_tmp:
            solver_explicit = pysole.Solver(
                dem=dem,
                output_dir=custom_tmp,
                survey_data_path="/tmp/test_workspace/survey.csv",
                plots_dir="figures",
            )
            self.assertEqual(solver_explicit.effective_output_dir, custom_tmp)
            self.assertEqual(solver_explicit.plots_dir, os.path.join(custom_tmp, "figures"))
            self.assertEqual(solver_explicit.resolve_path("final_bedrock"), os.path.join(custom_tmp, "final_bedrock"))

        # 3. Absolute path overrides output_dir
        self.assertEqual(solver_fallback.resolve_path("/abs/path/bedrock"), "/abs/path/bedrock")

    def test_outputs_config_utility_methods(self):
        cfg = pysole.OutputsConfig(
            output_format="tif",
            output_prefix="custom_prefix",
            save_thickness_grid=True,
            save_basal_shear_stress=True,
        )
        active = cfg.active_exports()
        self.assertTrue(active["save_thickness_grid"])
        self.assertTrue(active["save_basal_shear_stress"])
        self.assertFalse(active["save_traveltime_grid"])
        self.assertFalse(active["save_migrated_points"])

        d = cfg.to_dict()
        self.assertEqual(d["output_format"], "tif")
        self.assertEqual(d["output_prefix"], "custom_prefix")
        self.assertTrue(d["save_thickness_grid"])
        self.assertFalse(d["save_migrated_points"])


if __name__ == "__main__":
    unittest.main()
