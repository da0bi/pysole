"""
Unit tests for Unprobed Glacier Survey Planner module.
"""

import unittest
import tempfile
import pathlib
import os
import json
import numpy as np
import pysole
from pysole.survey_planner import SurveyPlanner


class TestSurveyPlanner(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/wuk/input_data"))
        self.wuk_dem = os.path.join(self.data_dir, "dgm_unt_wuk.tif")
        self.wuk_outline = os.path.join(self.data_dir, "wuk_outline_clean.csv")

    def test_survey_planner_direct_execution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)

            model = pysole.Solver(
                dem=self.wuk_dem,
                outline=self.wuk_outline,
            )

            planner = SurveyPlanner(
                dem=model.dem_grid,
                outline=model.outline_mask,
                dx=model.dx,
                dy=model.dy,
                bounds=model.bounds,
            )
            sia_grid = planner.compute_synthetic_sia_depth(kc=3.0, tau_0=100e3)
            self.assertEqual(sia_grid.shape, model.dem_grid.shape)
            self.assertTrue(np.all(np.isfinite(sia_grid)))

            output_prefix = str(tmp_path / "wuk_test")
            plots_dir = str(tmp_path / "figures")
            res = planner.plan_survey(
                kc=3.0,
                tau_0=100e3,
                max_length_km=10.0,
                output_prefix=output_prefix,
                output_dir=str(tmp_path),
                plots_dir=plots_dir,
                output_format="tif",
            )

            self.assertIn("tracks", res)
            self.assertIn("sia_modelled_depth", res)

            # Check exported files match exact specified output paths
            expected_tif = f"{output_prefix}_sia_modelled_depth.tif"
            expected_gpx = f"{output_prefix}_survey_plan.gpx"
            expected_geojson = f"{output_prefix}_survey_plan.geojson"
            expected_png = os.path.join(plots_dir, "wuk_test_survey_plan_map.png")

            self.assertTrue(os.path.exists(expected_tif), f"Missing TIF: {expected_tif}")
            self.assertTrue(os.path.exists(expected_gpx), f"Missing GPX: {expected_gpx}")
            self.assertTrue(os.path.exists(expected_geojson), f"Missing GeoJSON: {expected_geojson}")
            self.assertTrue(os.path.exists(expected_png), f"Missing PNG: {expected_png}")

    def test_solver_plan_survey_integration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            out_prefix = str(tmp_path / "test_run")

            model = pysole.Solver(
                dem=self.wuk_dem,
                outline=self.wuk_outline,
            )

            res = model.plan_survey(output_prefix=out_prefix, output_format="tif")
            self.assertIn("tracks", res)
            self.assertTrue(os.path.exists(f"{out_prefix}_sia_modelled_depth.tif"))

    def test_automatic_survey_planner_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "inputs": {
                    "dem_path": self.wuk_dem,
                    "outline_path": self.wuk_outline,
                    "survey_data_path": None,
                },
                "outputs": {
                    "output_dir": tmp_dir,
                    "output_prefix": "unprobed_wuk",
                    "output_format": "tif",
                    "plots_dir": "figures",
                },
            }
            config_file = os.path.join(tmp_dir, "test_config.json")
            with open(config_file, "w") as f:
                json.dump(config, f)

            solver = pysole.Solver.from_config(config_file)
            res = solver.run_pipeline()
            self.assertIsInstance(res, pysole.BedrockMap)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "unprobed_wuk_sia_modelled_depth.tif")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "unprobed_wuk_survey_plan.gpx")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "unprobed_wuk_survey_plan.geojson")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "figures", "unprobed_wuk_survey_plan_map.png")))


if __name__ == "__main__":
    unittest.main()
