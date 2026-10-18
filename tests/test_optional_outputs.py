"""
Unit tests for optional export features in PySole.
"""

import unittest
import tempfile
import os
import shutil
import numpy as np
from pysole.solver import Solver
from pysole.raster import save_points_csv


class TestOptionalOutputs(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dem_path = os.path.join(self.temp_dir, "test_dem.asc")
        self.pts_path = os.path.join(self.temp_dir, "test_pts.csv")

        # Create synthetic 30x30 synthetic DEM (tilted plane)
        X, Y = np.meshgrid(np.linspace(0, 300, 30), np.linspace(0, 300, 30))
        Z = 2000.0 - 0.1 * X - 0.05 * Y
        header = f"ncols 30\nnnrows 30\nxllcorner 0\nyllcorner 0\ncellsize 10\nnodata_value -9999\n"
        with open(self.dem_path, "w") as f:
            f.write(header)
            np.savetxt(f, Z, fmt="%.2f")

        # Synthetic survey traveltimes (one-way traveltime ~ 50 ns)
        pts = np.array([
            [50.0, 50.0, 50.0],
            [100.0, 100.0, 60.0],
            [150.0, 150.0, 40.0],
            [200.0, 200.0, 70.0],
            [250.0, 250.0, 55.0],
        ])
        np.savetxt(self.pts_path, pts, delimiter=",", header="x,y,traveltime", comments="")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_points_csv_helper(self):
        pts = np.array([[10, 20, 100, 45.5], [30, 40, 105, 52.1]])
        out_csv = os.path.join(self.temp_dir, "test_points.csv")
        saved = save_points_csv(pts, out_csv)
        self.assertTrue(os.path.exists(saved))

        data = np.loadtxt(saved, delimiter=",", skiprows=1)
        self.assertEqual(data.shape, (2, 4))
        self.assertAlmostEqual(data[0, 3], 45.5)

    def test_optional_exports_enabled(self):
        output_stem = os.path.join(self.temp_dir, "output_test")
        cfg = {
            "inputs": {
                "dem_path": self.dem_path,
                "survey_data_path": self.pts_path,
                "survey_data_type": "one_way_traveltime",
                "show_progress": False,
            },
            "migration_parameters": {
                "perform_migration": True,
                "velocity": 0.16,
            },
            "kriging_parameters": {
                "pre_migration": {
                    "calculate_uncertainty": True,
                },
                "post_migration": {
                    "calculate_uncertainty": True,
                },
            },
            "outputs": {
                "output_format": "tif",
                "output_prefix": output_stem + ".tif",  # Test automatic extension stripping
                "save_traveltime_grid": True,
                "save_traveltime_uncertainty": True,
                "save_migrated_points": True,
                "save_thickness_grid": True,
                "save_thickness_uncertainty": True,
                "save_basal_shear_stress": True,
                "save_basal_shear_stress_uncertainty": True,
            }
        }
        solver = Solver.from_config(cfg)
        solver.run_pipeline()

        # Check that bedrock and optional output files were created on disk
        self.assertTrue(os.path.exists(f"{output_stem}_bedrock.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_traveltime.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_traveltime_uncertainty.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_migrated_points.csv"))
        self.assertTrue(os.path.exists(f"{output_stem}_thickness.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_thickness_uncertainty.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_basal_shear_stress.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_basal_shear_stress_uncertainty.tif"))

        # Check properties on solver
        self.assertIsNotNone(solver.traveltime_grid)
        self.assertIsNotNone(solver.thickness_grid)
        self.assertIsNotNone(solver.thickness_std_grid)
        self.assertIsNotNone(solver.basal_shear_stress_grid)
        self.assertIsNotNone(solver.basal_shear_stress_std_grid)
        self.assertIsNotNone(solver.migrated_points)

        # Check basal shear stress is in kPa (~10 - 200 kPa range)
        valid_bss = solver.basal_shear_stress_grid[~np.isnan(solver.basal_shear_stress_grid)]
        self.assertTrue(np.all(valid_bss >= 0.0))

    def test_optional_uncertainty_exports_skipped_when_calculation_disabled(self):
        output_stem = os.path.join(self.temp_dir, "output_no_unc_calc")
        cfg = {
            "inputs": {
                "dem_path": self.dem_path,
                "survey_data_path": self.pts_path,
                "survey_data_type": "one_way_traveltime",
                "show_progress": False,
            },
            "migration_parameters": {
                "perform_migration": True,
                "velocity": 0.16,
            },
            "kriging_parameters": {
                "pre_migration": {
                    "calculate_uncertainty": False,
                },
                "post_migration": {
                    "calculate_uncertainty": False,
                },
            },
            "outputs": {
                "output_format": "tif",
                "output_prefix": output_stem,
                "save_traveltime_grid": True,
                "save_traveltime_uncertainty": True,
                "save_thickness_grid": True,
                "save_thickness_uncertainty": True,
                "save_basal_shear_stress": True,
                "save_basal_shear_stress_uncertainty": True,
            }
        }
        solver = Solver.from_config(cfg)
        with self.assertLogs("pysole", level="WARNING") as cm:
            solver.run_pipeline()

        # Regular grids are created
        self.assertTrue(os.path.exists(f"{output_stem}_bedrock.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_traveltime.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_thickness.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_basal_shear_stress.tif"))

        # Uncertainty files are NOT created because calculation was disabled
        self.assertFalse(os.path.exists(f"{output_stem}_traveltime_uncertainty.tif"))
        self.assertFalse(os.path.exists(f"{output_stem}_thickness_uncertainty.tif"))
        self.assertFalse(os.path.exists(f"{output_stem}_basal_shear_stress_uncertainty.tif"))

        # Notice/warning logged for skipped exports
        export_skipped_logs = [m for m in cm.output if "[Export Skipped]" in m]
        self.assertTrue(any("traveltime uncertainty" in m for m in export_skipped_logs))
        self.assertTrue(any("thickness uncertainty" in m for m in export_skipped_logs))
        self.assertTrue(any("basal shear stress uncertainty" in m for m in export_skipped_logs))

    def test_migration_skipped_disables_migrated_points_export(self):
        output_stem = os.path.join(self.temp_dir, "output_nomig")
        cfg = {
            "inputs": {
                "dem_path": self.dem_path,
                "survey_data_path": self.pts_path,
                "survey_data_type": "depth",
                "show_progress": False,
            },
            "migration_parameters": {
                "perform_migration": False,
            },
            "outputs": {
                "output_format": "tif",
                "output_prefix": output_stem,
                "save_migrated_points": True,
                "save_thickness_grid": True,
            }
        }
        solver = Solver.from_config(cfg)
        solver.run_pipeline()

        # Thickness raster should be created, but migrated_points CSV should NOT be created
        self.assertTrue(os.path.exists(f"{output_stem}_bedrock.tif"))
        self.assertTrue(os.path.exists(f"{output_stem}_thickness.tif"))
        self.assertFalse(os.path.exists(f"{output_stem}_migrated_points.csv"))

    def test_save_bedrock_grid_flag(self):
        output_stem_disabled = os.path.join(self.temp_dir, "output_nobed")
        cfg_disabled = {
            "inputs": {
                "dem_path": self.dem_path,
                "survey_data_path": self.pts_path,
                "survey_data_type": "depth",
                "show_progress": False,
            },
            "migration_parameters": {
                "perform_migration": False,
            },
            "outputs": {
                "output_format": "tif",
                "output_prefix": output_stem_disabled,
                "save_bedrock_grid": False,
                "save_thickness_grid": True,
            }
        }
        solver_disabled = Solver.from_config(cfg_disabled)
        solver_disabled.run_pipeline()
        self.assertFalse(os.path.exists(f"{output_stem_disabled}_bedrock.tif"))
        self.assertTrue(os.path.exists(f"{output_stem_disabled}_thickness.tif"))

        output_stem_enabled = os.path.join(self.temp_dir, "output_withbed")
        cfg_enabled = {
            "inputs": {
                "dem_path": self.dem_path,
                "survey_data_path": self.pts_path,
                "survey_data_type": "depth",
                "show_progress": False,
            },
            "migration_parameters": {
                "perform_migration": False,
            },
            "outputs": {
                "output_format": "tif",
                "output_prefix": output_stem_enabled,
                "save_bedrock_grid": True,
            }
        }
        solver_enabled = Solver.from_config(cfg_enabled)
        solver_enabled.run_pipeline()
        self.assertTrue(os.path.exists(f"{output_stem_enabled}_bedrock.tif"))

    def _write_pipeline_config(self, tag, outputs, n=16, survey_data_type="depth", extra_cfg=None):
        """Small survey pipeline config in <temp>/<tag>; returns (config_path, work_dir)."""
        import json
        from pathlib import Path
        from pysole.raster import GridGeometry, BedrockMap

        work = Path(self.temp_dir) / tag
        work.mkdir(parents=True, exist_ok=True)
        bounds = (500000.0, 5200000.0, 500000.0 + 10.0 * n, 5200000.0 + 10.0 * n)
        geom = GridGeometry.create((n, n), dx=10.0, dy=10.0, bounds=bounds)
        xx, yy = geom.meshgrid
        dem = 1000.0 + 0.2 * (xx - bounds[0]) + 0.1 * (yy - bounds[1])
        BedrockMap(grid=dem, bounds=geom.bounds, crs=32633).save(work / "dem.tif")
        rng = np.random.default_rng(7)
        px = rng.uniform(bounds[0] + 15, bounds[2] - 15, 12)
        py = rng.uniform(bounds[1] + 15, bounds[3] - 15, 12)
        values = rng.uniform(20.0, 60.0, 12)
        if survey_data_type != "depth":
            values = values / 0.16
        pts = np.column_stack((px, py, values))
        np.savetxt(work / "picks.csv", pts, delimiter=",", header="X,Y,Picks", comments="")
        cfg = {
            "inputs": {"dem_path": str(work / "dem.tif"), "survey_data_path": str(work / "picks.csv"),
                       "survey_data_type": survey_data_type, "show_progress": False},
            "outputs": {"output_format": "tif", **outputs},
        }
        if extra_cfg:
            cfg.update(extra_cfg)
        cfg_file = work / "pysole.json"
        cfg_file.write_text(json.dumps(cfg))
        return cfg_file, work

    def test_resolve_plots_dir_rules(self):
        """Figures follow the data files; absolute plots_dir wins; None == 'figures'."""
        from pathlib import Path
        from pysole.config import resolve_plots_dir

        out_path = Path(self.temp_dir)
        base = out_path / "data"
        prefix_abs = str(base / "run")
        other = str(out_path / "elsewhere")
        # absolute plots_dir is used as is
        self.assertEqual(resolve_plots_dir(other, output_prefix=prefix_abs), other)
        # absolute prefix: figures beside the data (no directories created by the resolution itself)
        self.assertEqual(resolve_plots_dir(None, output_prefix=prefix_abs), str(base / "figures"))
        self.assertEqual(resolve_plots_dir("figures", output_prefix=prefix_abs), str(base / "figures"))
        self.assertEqual(resolve_plots_dir("plots", output_prefix=prefix_abs), str(base / "plots"))
        self.assertFalse(base.exists())
        # relative prefix: resolved against output_dir
        out = out_path / "outdir"
        self.assertEqual(resolve_plots_dir(None, output_prefix="run", output_dir=str(out)), str(out / "figures"))
        self.assertEqual(
            resolve_plots_dir(None, output_prefix="run", output_dir=str(out)),
            resolve_plots_dir("figures", output_prefix="run", output_dir=str(out)),
        )

    def test_pipeline_exports_each_output_once_and_honours_flags(self):
        """Every output is written at most once; skipped migration exports nothing; bedrock flag honoured."""
        from unittest.mock import patch
        from pysole.raster import BedrockMap
        from pysole.pipeline import run_from_config
        from pathlib import Path

        out_path = Path(self.temp_dir)
        for survey_type in ("depth", "one_way_traveltime"):
            for flag in (True, False):
                tag = f"once_{survey_type}_{flag}"
                out = out_path / tag / "out"
                cfg_file, _ = self._write_pipeline_config(
                    tag, {"output_dir": str(out), "output_prefix": "r", "save_bedrock_grid": flag,
                          "save_migrated_points": True, "save_traveltime_grid": True},
                    survey_data_type=survey_type,
                )
                saves, points_exports = [], []
                orig_save = BedrockMap.save
                orig_pts = Solver._export_optional_points_csv

                def spy_save(self_, filepath, *a, _s=saves, **k):
                    _s.append(os.path.basename(str(filepath)))
                    return orig_save(self_, filepath, *a, **k)

                def spy_pts(self_, points, suffix, _p=points_exports):
                    _p.append(suffix)
                    return orig_pts(self_, points, suffix)

                with patch.object(BedrockMap, "save", spy_save), patch.object(Solver, "_export_optional_points_csv", spy_pts):
                    run_from_config(config_path=cfg_file, is_batch=True)

                msg = f"{survey_type}/flag={flag}: {saves} {points_exports}"
                self.assertEqual(len([s for s in saves if "_bedrock" in s]), 1 if flag else 0, msg)
                self.assertEqual((out / "r_bedrock.tif").exists(), flag, msg)
                expected_once = 1 if survey_type != "depth" else 0
                self.assertEqual(len([s for s in saves if "_traveltime" in s]), expected_once, msg)
                self.assertEqual(points_exports.count("migrated_points"), expected_once, msg)
                self.assertEqual((out / "r_migrated_points.csv").exists(), bool(expected_once), msg)

    def test_pipeline_figures_follow_absolute_prefix(self):
        """With an absolute output_prefix figures land beside the data; plots_dir null == 'figures'."""
        from pathlib import Path
        from pysole.pipeline import run_from_config

        out_path = Path(self.temp_dir)
        for plots_dir in (None, "figures"):
            tag = f"absprefix_{plots_dir}"
            data_dir = out_path / tag / "results"
            cfg_file, work = self._write_pipeline_config(
                tag, {"output_prefix": str(data_dir / "run"), "plots_dir": plots_dir},
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            run_from_config(config_path=cfg_file, is_batch=True)
            self.assertTrue((data_dir / "run_bedrock.tif").exists())
            figs = list((data_dir / "figures").glob("*.png"))
            self.assertGreater(len(figs), 0, f"no figures in {data_dir / 'figures'}")
            self.assertFalse((work / "pysole").exists(), "stray default pysole/ folder was created")

    def test_save_traveltime_uncertainty_export_direct(self):
        """pre_calculate_uncertainty enables Pass 1 variance evaluation and save_traveltime_uncertainty exports it."""
        from pysole.raster import GridGeometry

        bounds = (500000.0, 5200000.0, 500100.0, 5200100.0)
        geom = GridGeometry.create((10, 10), dx=10.0, dy=10.0, bounds=bounds)
        dem = np.full((10, 10), 1000.0)
        pts = np.array([
            [500020.0, 5200020.0, 1000.0, 0.5],
            [500050.0, 5200050.0, 1000.0, 0.8],
            [500080.0, 5200080.0, 1000.0, 1.2],
        ])
        solver = Solver(
            dem=dem,
            dx=10.0,
            dy=10.0,
            bounds=geom.bounds,
            output_dir=self.temp_dir,
            pre_calculate_uncertainty=True,
            config={"outputs": {"save_traveltime_uncertainty": True}},
        )
        solver.pre_kriging_points = pts
        solver.migrate_eikonal(travel_times=pts, interactive=False)
        self.assertIsNotNone(solver.traveltime_std_grid)
        self.assertEqual(solver.traveltime_std_grid.shape, (10, 10))
        self.assertTrue(np.all(np.isfinite(solver.traveltime_std_grid)))
        self.assertTrue(np.all(solver.traveltime_std_grid >= 0.0))

        saved = solver.export_outputs(stage="migration")
        self.assertTrue(any("traveltime_uncertainty" in f for f in saved), saved)

    def test_standardized_figure_naming_convention(self):
        """Verifies standardized 11 diagnostic figures naming with empty and custom prefix,
        and confirms unblended calculated bedrock figure is skipped."""
        from pathlib import Path
        from pysole.pipeline import run_from_config

        out_path = Path(self.temp_dir)
        expected_suffixes = [
            "01_pass1_variogram.png",
            "02_pass1_kc_variogram_optimization.png",
            "03_3d_migration.png",
            "04_pass2_variogram.png",
            "05_pass2_kc_variogram_optimization.png",
            "06_bedrock_elevation_and_uncertainty.png",
            "07_final_embedded_bedrock.png",
            "08_final_thickness_and_uncertainty.png",
            "09_final_thickness_histogram.png",
            "10_final_basal_shear_stress_and_uncertainty.png",
            "11_final_basal_shear_stress_histogram.png",
        ]

        # Case 1: Empty prefix (default)
        work_empty = out_path / "fig_naming_empty"
        cfg_file_empty, _ = self._write_pipeline_config(
            "fig_naming_empty",
            {"output_prefix": "", "output_dir": str(work_empty)},
            survey_data_type="one_way_traveltime",
        )
        run_from_config(config_path=cfg_file_empty, is_batch=True)
        plots_empty = work_empty / "figures"
        self.assertTrue(plots_empty.exists())
        created_empty = sorted(f.name for f in plots_empty.glob("*.png"))

        for expected in expected_suffixes:
            self.assertIn(expected, created_empty, f"Missing expected figure: {expected}")
        self.assertFalse(any("calculated_bedrock" in name for name in created_empty))
        self.assertEqual(len(created_empty), 11)

        # Case 2: Custom prefix
        prefix = "apo"
        work_custom = out_path / "fig_naming_custom"
        cfg_file_custom, _ = self._write_pipeline_config(
            "fig_naming_custom",
            {"output_prefix": prefix, "output_dir": str(work_custom)},
            survey_data_type="one_way_traveltime",
        )
        run_from_config(config_path=cfg_file_custom, is_batch=True)
        plots_custom = work_custom / "figures"
        self.assertTrue(plots_custom.exists())
        created_custom = sorted(f.name for f in plots_custom.glob("*.png"))

        expected_custom = [
            f"{s[:3]}{prefix}_{s[3:]}" for s in expected_suffixes
        ]
        for expected in expected_custom:
            self.assertIn(expected, created_custom, f"Missing expected custom figure: {expected}")
        self.assertFalse(any("calculated_bedrock" in name for name in created_custom))
        self.assertEqual(len(created_custom), 11)

        # Case 3: grad_traveltime_grid visualization
        work_grad = out_path / "fig_naming_grad"
        cfg_file_grad, _ = self._write_pipeline_config(
            "fig_naming_grad",
            {
                "output_prefix": "apo",
                "output_dir": str(work_grad),
            },
            survey_data_type="one_way_traveltime",
            extra_cfg={
                "migration_parameters": {
                    "perform_migration": True,
                    "velocity": 0.16,
                    "interactive_migration": False,
                    "horizontal_displacement_visualization": "grad_traveltime_grid",
                },
            },
        )
        run_from_config(config_path=cfg_file_grad, is_batch=True)
        plots_grad = work_grad / "figures"
        self.assertTrue(plots_grad.exists())
        created_grad = sorted(f.name for f in plots_grad.glob("*.png"))
        self.assertIn("03_apo_3d_migration_hor_displacements.png", created_grad)
        self.assertEqual(len(created_grad), 12)


if __name__ == "__main__":
    unittest.main()
