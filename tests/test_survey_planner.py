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
                output_dir=str(tmp_path),
            )

            res = model.plan_survey(output_prefix=out_prefix, output_format="tif", interactive=False)
            self.assertIn("tracks", res)
            self.assertTrue(os.path.exists(f"{out_prefix}_sia_modelled_depth.tif"))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "figures", "test_run_survey_plan_map.png")))

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
            res = solver.run_pipeline(interactive=False)
            self.assertIsInstance(res, pysole.BedrockMap)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "unprobed_wuk_sia_modelled_depth.tif")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "unprobed_wuk_survey_plan.gpx")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "unprobed_wuk_survey_plan.geojson")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "figures", "unprobed_wuk_survey_plan_map.png")))

    def test_interactive_survey_planner_defaults(self):
        """Interactive survey planner prompts for tau_0, kc, and max_km; empty input adopts defaults."""
        from unittest.mock import patch
        import sys

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = pysole.Solver(dem=self.wuk_dem, outline=self.wuk_outline, output_dir=tmp_dir)
            planner = SurveyPlanner(
                dem=model.dem_grid, outline=model.outline_mask, dx=model.dx, dy=model.dy, bounds=model.bounds
            )
            # Empty strings simulate pressing Enter for tau_0, kc, and max_km
            with patch("builtins.input", side_effect=["", "", ""]) as mock_input, \
                 patch.object(sys.stdin, "isatty", return_value=True):
                res = planner.plan_survey(output_dir=tmp_dir, interactive=True)
                self.assertEqual(mock_input.call_count, 3)
                self.assertIn("tracks", res)
                self.assertIn("sia_modelled_depth", res)

    def test_interactive_survey_planner_custom_values(self):
        """Interactive survey planner applies custom user-entered values."""
        from unittest.mock import patch
        import sys

        with tempfile.TemporaryDirectory() as tmp_dir:
            planner = SurveyPlanner(dem=self.wuk_dem, outline=self.wuk_outline)
            # User enters 120 kPa for tau, 0.05 for kc, and 8.0 for max_km
            with patch("builtins.input", side_effect=["120", "0.05", "8.0"]) as mock_input, \
                 patch.object(sys.stdin, "isatty", return_value=True):
                res = planner.plan_survey(output_dir=tmp_dir, interactive=True)
                self.assertEqual(mock_input.call_count, 3)
                self.assertIn("tracks", res)

    def test_interactive_survey_planner_partial_inputs(self):
        """Interactive survey planner prompts only for parameters not passed programmatically."""
        from unittest.mock import patch
        import sys

        with tempfile.TemporaryDirectory() as tmp_dir:
            planner = SurveyPlanner(dem=self.wuk_dem, outline=self.wuk_outline)
            # tau_0 is passed programmatically (80 kPa = 80,000 Pa), so input is only called for kc and max_km
            with patch("builtins.input", side_effect=["0.04", "3.5"]) as mock_input, \
                 patch.object(sys.stdin, "isatty", return_value=True):
                res = planner.plan_survey(tau_0=80e3, output_dir=tmp_dir, interactive=True)
                self.assertEqual(mock_input.call_count, 2)
                self.assertIn("tracks", res)

    def test_interactive_survey_planner_retry_on_invalid_input(self):
        """Interactive survey planner warns and re-prompts on non-numeric or non-positive input."""
        from unittest.mock import patch
        import sys

        with tempfile.TemporaryDirectory() as tmp_dir:
            planner = SurveyPlanner(dem=self.wuk_dem, outline=self.wuk_outline)
            # First prompt (tau_0) gets "invalid", then "-50", then "100". Next two accept defaults ("").
            with patch("builtins.input", side_effect=["invalid", "-50", "100", "", ""]) as mock_input, \
                 patch.object(sys.stdin, "isatty", return_value=True):
                res = planner.plan_survey(output_dir=tmp_dir, interactive=True)
                self.assertEqual(mock_input.call_count, 5)
                self.assertIn("tracks", res)

    def test_non_interactive_and_batch_safety(self):
        """When interactive=False, sys.stdin is not a tty, or in batch mode, input is never called."""
        from unittest.mock import patch
        import sys

        with tempfile.TemporaryDirectory() as tmp_dir:
            planner = SurveyPlanner(dem=self.wuk_dem, outline=self.wuk_outline)
            with patch("builtins.input", side_effect=AssertionError("input() must not be called")) as mock_input:
                # 1. interactive=False explicitly
                res1 = planner.plan_survey(output_dir=tmp_dir, interactive=False)
                self.assertEqual(mock_input.call_count, 0)
                self.assertIn("tracks", res1)

                # 2. interactive=True but not a tty
                with patch.object(sys.stdin, "isatty", return_value=False):
                    res2 = planner.plan_survey(output_dir=tmp_dir, interactive=True)
                    self.assertEqual(mock_input.call_count, 0)
                    self.assertIn("tracks", res2)

    def test_cli_plan_survey_interactive_and_batch(self):
        """CLI pysole plan-survey respects CLI arguments and interactive/batch flags."""
        from unittest.mock import patch
        import sys
        from pysole.config import main_cli

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. User passes all parameters on CLI: no interactive prompt needed
            with patch("builtins.input", side_effect=AssertionError("input() should not be called")):
                with patch.object(
                    sys,
                    "argv",
                    ["pysole", "plan-survey", "--dem", self.wuk_dem, "--outline", self.wuk_outline,
                     "--tau", "120", "--kc", "0.04", "--max-km", "4.0", "--out-dir", tmp_dir],
                ):
                    with self.assertRaises(SystemExit) as ctx:
                        main_cli()
                    self.assertEqual(ctx.exception.code, 0)

            # 2. User omits parameters on CLI and runs in interactive terminal: prompts for omitted parameters
            with patch("builtins.input", side_effect=["110", "0.035", "6.0"]) as mock_input, \
                 patch.object(sys.stdin, "isatty", return_value=True):
                with patch.object(
                    sys,
                    "argv",
                    ["pysole", "plan-survey", "--dem", self.wuk_dem, "--outline", self.wuk_outline, "--out-dir", tmp_dir],
                ):
                    with self.assertRaises(SystemExit) as ctx:
                        main_cli()
                    self.assertEqual(ctx.exception.code, 0)
                    self.assertEqual(mock_input.call_count, 3)

            # 3. User omits parameters on CLI but runs with --batch: no input called, defaults used
            with patch("builtins.input", side_effect=AssertionError("input() should not be called")):
                with patch.object(sys.stdin, "isatty", return_value=True):
                    with patch.object(
                        sys,
                        "argv",
                        ["pysole", "plan-survey", "--dem", self.wuk_dem, "--outline", self.wuk_outline,
                         "--batch", "--out-dir", tmp_dir],
                    ):
                        with self.assertRaises(SystemExit) as ctx:
                            main_cli()
                        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

