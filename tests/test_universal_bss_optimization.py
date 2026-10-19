"""Unit tests for universal BSS slope optimization and architectural decoupling across interpolation targets."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from pysole.raster import GridGeometry
from pysole.solver import Solver
from pysole.variogram import OptimizationResult


class TestUniversalBSSOptimization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        bounds = (500000.0, 5200000.0, 500120.0, 5200120.0)
        self.geom = GridGeometry.create((12, 12), dx=10.0, dy=10.0, bounds=bounds)
        xx, yy = self.geom.meshgrid
        self.dem = 1000.0 + 0.2 * (xx - bounds[0]) + 0.1 * (yy - bounds[1])

        self.pts = np.array([
            [500015.0, 5200015.0, 1003.0, 40.0],
            [500045.0, 5200030.0, 1012.0, 55.0],
            [500075.0, 5200020.0, 1018.0, 48.0],
            [500100.0, 5200055.0, 1030.0, 35.0],
            [500030.0, 5200085.0, 1022.0, 60.0],
            [500085.0, 5200100.0, 1038.0, 42.0],
        ])

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_solver(self, pre_target="T", post_target="D", **kwargs):
        solver = Solver(
            dem=self.dem,
            bounds=self.geom.bounds,
            survey_data_type="one_way_traveltime",
            pre_interpolation_target=pre_target,
            post_interpolation_target=post_target,
            output_dir=str(self.output_dir / "out"),
            plots_dir=str(self.output_dir / "plots"),
            **kwargs,
        )
        solver.survey_points = self.pts.copy()
        solver.migrated_points = self.pts.copy()
        return solver

    def test_get_sample_points_has_no_side_effects(self):
        """get_sample_points must not trigger optimize_bss or mutate solver optimization state."""
        solver = self._make_solver(pre_target="T", post_target="D")
        self.assertIsNone(solver.opt_slope)
        self.assertIsNone(solver.pre_opt_res)

        pts = solver.get_sample_points("pre_migration", "T")
        self.assertEqual(pts.shape, (6, 3))
        self.assertIsNone(solver.opt_slope)
        self.assertIsNone(solver.pre_opt_res)

    def test_optimize_bss_stage_selection_and_tracking(self):
        """optimize_bss records stage results into pre_opt_res and post_opt_res cleanly."""
        solver = self._make_solver(pre_target="T", post_target="D")

        fake_res = OptimizationResult(
            optimal_kc=0.025,
            optimal_slope_grid=np.full((12, 12), 0.15),
            all_kc_variances=np.array([[0.025, 10.0]]),
            all_smoothed_slopes={0.025: np.full((12, 12), 0.15)},
            optimal_dem_grid=self.dem,
            all_smoothed_dems={0.025: self.dem},
            opt_variogram_params={"range": 350.0, "sill": 15.0, "nugget": 0.5},
            dx=10.0,
            dy=10.0,
        )

        with patch.object(solver.bss_optimizer, "optimize", return_value=fake_res) as mock_opt:
            kc = solver.optimize_bss(stage="pre_migration")
            self.assertEqual(kc, 0.025)
            self.assertIsNotNone(solver.pre_opt_res)
            self.assertIsNone(solver.post_opt_res)
            self.assertEqual(solver.opt_kc, 0.025)
            self.assertEqual(mock_opt.call_args.kwargs["prefix"], "01_")

            kc2 = solver.optimize_bss(stage="post_migration")
            self.assertEqual(kc2, 0.025)
            self.assertIsNotNone(solver.post_opt_res)
            self.assertEqual(mock_opt.call_args.kwargs["prefix"], "03_")

    def test_ensure_optimized_triggers_for_direct_targets_in_pass1_and_pass2(self):
        """_execute_kriging_pass triggers BSS optimization for target 'T' (Pass 1) and 'D' (Pass 2)."""
        solver = self._make_solver(pre_target="T", post_target="D")

        fake_opt_res1 = OptimizationResult(
            optimal_kc=0.03,
            optimal_slope_grid=np.full((12, 12), 0.2),
            all_kc_variances=np.array([[0.03, 12.0]]),
            all_smoothed_slopes={0.03: np.full((12, 12), 0.2)},
            optimal_dem_grid=self.dem,
            all_smoothed_dems={0.03: self.dem},
            opt_variogram_params={"range": 300.0, "sill": 20.0, "nugget": 0.0},
            dx=10.0,
            dy=10.0,
        )

        with patch.object(solver.bss_optimizer, "optimize", return_value=fake_opt_res1) as mock_opt, \
             patch("pysole.solver.kriging_interpolation") as mock_krig:
            from pysole.interpolation import KrigingResult
            mock_krig.return_value = KrigingResult(bedrock_grid=np.full((12, 12), 40.0), variance_grid=np.zeros((12, 12)))

            # Pass 1 with direct traveltime target 'T'
            solver._execute_kriging_pass(
                target_type="T",
                points=solver.survey_points,
                krig_method="ordinary",
                var_model="spherical",
                zero_boundary=False,
                pass_name="Pass 1 - Constructing Traveltime Grid",
            )
            mock_opt.assert_called_once()
            self.assertEqual(mock_opt.call_args.kwargs["prefix"], "01_")
            self.assertIsNotNone(solver.pre_opt_res)

            # Check that Kriging received the fitted spatial range (300.0 m) and scaled sill (var of values)
            krig_kwargs = mock_krig.call_args.kwargs
            var_params = krig_kwargs["variogram_params"]
            self.assertEqual(var_params["range"], 300.0)
            self.assertAlmostEqual(var_params["sill"], float(np.var(solver.survey_points[:, 3])), places=5)
            self.assertEqual(var_params["nugget"], 0.0)

            # Pass 2 with direct depth target 'D'
            mock_opt.reset_mock()
            fake_opt_res2 = OptimizationResult(
                optimal_kc=0.028,
                optimal_slope_grid=np.full((12, 12), 0.18),
                all_kc_variances=np.array([[0.028, 8.0]]),
                all_smoothed_slopes={0.028: np.full((12, 12), 0.18)},
                optimal_dem_grid=self.dem,
                all_smoothed_dems={0.028: self.dem},
                opt_variogram_params={"range": 320.0, "sill": 16.0, "nugget": 0.0},
                dx=10.0,
                dy=10.0,
            )
            mock_opt.return_value = fake_opt_res2

            solver._execute_kriging_pass(
                target_type="D",
                points=solver.migrated_points,
                krig_method="ordinary",
                var_model="spherical",
                zero_boundary=False,
                pass_name="Pass 2 - Constructing Depth Grid",
            )
            mock_opt.assert_called_once()
            self.assertEqual(mock_opt.call_args.kwargs["prefix"], "03_")
            self.assertIsNotNone(solver.post_opt_res)

    def test_optimize_bss_auto_loads_survey_points_from_path(self):
        """optimize_bss auto-loads survey points from survey_data_path when survey_points is None."""
        csv_file = self.output_dir / "test_survey.csv"
        np.savetxt(
            csv_file,
            self.pts,
            delimiter=",",
            header="X,Y,Z,T",
            comments="",
        )

        solver = Solver(
            dem=self.dem,
            bounds=self.geom.bounds,
            survey_data_type="one_way_traveltime",
            survey_data_path=str(csv_file),
            pre_interpolation_target="T",
            post_interpolation_target="D",
            output_dir=str(self.output_dir / "out"),
            plots_dir=str(self.output_dir / "plots"),
        )
        self.assertIsNone(solver.survey_points)

        fake_res = OptimizationResult(
            optimal_kc=0.02,
            optimal_slope_grid=np.full((12, 12), 0.1),
            all_kc_variances=np.array([[0.02, 5.0]]),
            all_smoothed_slopes={0.02: np.full((12, 12), 0.1)},
            optimal_dem_grid=self.dem,
            all_smoothed_dems={0.02: self.dem},
            opt_variogram_params={"range": 250.0, "sill": 10.0, "nugget": 0.0},
            dx=10.0,
            dy=10.0,
        )

        with patch.object(solver.bss_optimizer, "optimize", return_value=fake_res) as mock_opt:
            kc = solver.optimize_bss(stage="pre_migration")
            self.assertEqual(kc, 0.02)
            self.assertIsNotNone(solver.survey_points)
            self.assertEqual(len(solver.survey_points), len(self.pts))
            # Verify optimize received the loaded survey points, not a synthetic DEM meshgrid
            passed_pts = mock_opt.call_args.kwargs["survey_points"]
            self.assertEqual(passed_pts.shape, (len(self.pts), 4))
            np.testing.assert_allclose(passed_pts[:, :2], self.pts[:, :2])

    def test_optimize_bss_raises_when_no_points_available(self):
        """optimize_bss raises ValueError instead of generating a synthetic DEM meshgrid when no points exist."""
        solver = Solver(
            dem=self.dem,
            bounds=self.geom.bounds,
            survey_data_type="one_way_traveltime",
            output_dir=str(self.output_dir / "out"),
            plots_dir=str(self.output_dir / "plots"),
        )
        solver.survey_points = None
        solver.migrated_points = None
        solver.survey_data_path = None

        with self.assertRaises(ValueError) as ctx:
            solver.optimize_bss(stage="pre_migration")
        self.assertIn("Cannot run BSS variogram optimization or Kriging, because no survey data was configured", str(ctx.exception))
        self.assertIn("Call solver.plan_survey()", str(ctx.exception))

    def test_pipeline_unprobed_empty_string_triggers_survey_planner(self):
        """pipeline.run_from_config dispatches survey planner when survey_data_path is empty string or None."""
        from pysole.pipeline import run_from_config
        import json

        dem_file = self.output_dir / "dem.npy"
        np.save(dem_file, self.dem)

        cfg = {
            "inputs": {
                "dem_path": str(dem_file),
                "survey_data_path": "",  # Empty string should trigger survey planner
            },
            "spatial_parameters": {
                "bounds": list(self.geom.bounds),
                "dx": 10.0,
                "dy": 10.0,
            },
            "outputs": {
                "output_dir": str(self.output_dir / "unprobed_test"),
                "output_prefix": "unprobed",
            },
        }
        cfg_file = self.output_dir / "unprobed_cfg.json"
        with open(cfg_file, "w") as f:
            json.dump(cfg, f)

        with patch("pysole.solver.Solver.plan_survey") as mock_plan:
            mock_plan.return_value = {"sia_modelled_depth": np.zeros((12, 12))}
            res = run_from_config(cfg_file, is_batch=True)
            mock_plan.assert_called_once()
            self.assertEqual(res.name, "SIA Modelled Ice Thickness")


if __name__ == "__main__":
    unittest.main()
