"""
Regression tests for the v0.4.4 code review findings (F1-F7):
slope handling (F1/F6), native "linear" variogram handling (F4), variance propagation (F3),
traveltime export warnings (F7), the memory guard / always-on uncertainty (F2), warn-and-prune of
unknown configuration keys, and the README API examples (F5).
"""

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pysole.config import ConfigError, DEFAULT_CONFIG, OutputsConfig, load_config, sanitize_config
from pysole.interpolation import DualKrigingSolver, KrigingResult, kriging_interpolation
from pysole.memory import (
    estimate_native_kriging_bytes,
    kriging_chunk_size,
    plan_native_kriging,
)
from pysole.raster import GridGeometry
from pysole.solver import Solver
from pysole.variogram import evaluate_variogram_model, resolve_native_variogram_model, spherical_variogram

BOUNDS = (500000.0, 5200000.0, 500120.0, 5200120.0)
SURVEY = np.array([
    [500015.0, 5200015.0, 1003.0, 40.0],
    [500045.0, 5200030.0, 1012.0, 55.0],
    [500075.0, 5200020.0, 1018.0, 48.0],
    [500100.0, 5200055.0, 1030.0, 35.0],
    [500030.0, 5200085.0, 1022.0, 60.0],
    [500085.0, 5200100.0, 1038.0, 42.0],
])


class ReviewTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.addCleanup(self.temp_dir.cleanup)

    def make_solver(self, **kwargs) -> Solver:
        """Small synthetic depth-survey Solver (12x12 @ 10 m) writing only to the temp dir."""
        geom = GridGeometry.create((12, 12), dx=10.0, dy=10.0, bounds=BOUNDS)
        xx, yy = geom.meshgrid
        dem = 1000.0 + 0.2 * (xx - BOUNDS[0]) + 0.1 * (yy - BOUNDS[1])
        kwargs.setdefault("output_dir", str(self.output_dir / "solver_out"))
        solver = Solver(dem=dem, bounds=geom.bounds, survey_data_type="depth", **kwargs)
        solver.opt_slope = np.full((12, 12), 0.2)
        solver.survey_points = SURVEY.copy()
        solver.migrated_points = SURVEY.copy()
        return solver


class TestSlopeHandling(ReviewTestCase):
    def test_bss_uses_unfloored_slope(self):
        """F1: the basal shear stress uses sin(alpha_opt) without the slope floor (as in v0.4.4)."""
        solver = self.make_solver()
        solver.opt_slope = np.full((12, 12), 0.02)  # ~1.1 deg, far below the 5 deg slope floor
        solver.calculate_bedrock()
        solver.finalize_bedrock()
        mask = solver.outline_mask
        expected = solver.ice_density * solver.g * solver.final_thickness * np.sin(0.02) / 1000.0
        np.testing.assert_allclose(solver.final_bss[mask], expected[mask], rtol=1e-12)
        expected_std = solver.ice_density * solver.g * solver.kriged_std * np.sin(0.02) / 1000.0
        np.testing.assert_allclose(solver.bss_std[mask], expected_std[mask], rtol=1e-12)
        # the P <-> D/T conversions keep the floor
        self.assertAlmostEqual(float(solver.safe_slope_sin.min()), float(np.sin(np.radians(5.0))))
        self.assertAlmostEqual(float(solver.slope_sin.max()), float(np.sin(0.02)))

    def test_safe_slope_sin_is_cached_and_invalidated(self):
        """F6: safe_slope_sin is cached; a new slope grid or a new floor invalidates the cache."""
        solver = self.make_solver()
        first = solver.safe_slope_sin
        self.assertIs(solver.safe_slope_sin, first)
        solver.opt_slope = np.full((12, 12), 0.5)
        second = solver.safe_slope_sin
        self.assertIsNot(second, first)
        np.testing.assert_allclose(second, np.sin(0.5))
        solver.config = {"optimization_parameters": {"slope_floor_deg": 40.0}}
        third = solver.safe_slope_sin
        self.assertIsNot(third, second)
        np.testing.assert_allclose(third, np.sin(np.radians(40.0)))


class TestLinearVariogramHandling(ReviewTestCase):
    def test_resolver(self):
        self.assertEqual(resolve_native_variogram_model(" Exponential "), "exponential")
        with self.assertLogs("pysole", level="WARNING") as cm:
            self.assertEqual(resolve_native_variogram_model("linear"), "spherical")
        self.assertEqual(len(cm.output), 1)
        with self.assertNoLogs("pysole", level="WARNING"):
            self.assertEqual(resolve_native_variogram_model("linear", warn=False), "spherical")

    def test_linear_variogram_function_is_removed(self):
        import pysole.variogram as vg

        self.assertFalse(hasattr(vg, "linear_variogram"))
        h = np.linspace(0.0, 300.0, 7)
        np.testing.assert_allclose(
            evaluate_variogram_model(h, "linear", 200.0, 1.0, 0.1), spherical_variogram(h, 200.0, 1.0, 0.1)
        )

    def test_solver_resolves_linear_once_for_native_engine(self):
        with self.assertLogs("pysole", level="WARNING") as cm:
            solver = self.make_solver(pre_variogram_model="linear", post_variogram_model="linear")
        self.assertEqual(sum("'linear' variogram" in m for m in cm.output), 1, cm.output)
        self.assertEqual(solver.pre_variogram_model, "spherical")
        self.assertEqual(solver.post_variogram_model, "spherical")

    def test_pykrige_engine_and_regression_keep_linear(self):
        with patch("pysole.variogram.logger") as mock_logger:
            solver = self.make_solver(kriging_engine="pykrige", pre_variogram_model="linear")
        mock_logger.warning.assert_not_called()
        self.assertEqual(solver.pre_variogram_model, "linear")
        native = self.make_solver()
        self.assertEqual(native._resolve_variogram_for_engine("linear", "regression"), "linear")
        self.assertEqual(native._resolve_variogram_for_engine("linear", "ordinary", warn=False), "spherical")

    def test_dual_kriging_solver_never_sees_linear(self):
        rng = np.random.default_rng(3)
        x, y, z = rng.uniform(0, 100, 12), rng.uniform(0, 100, 12), rng.normal(10, 1, 12)
        with self.assertLogs("pysole", level="WARNING"):
            dual = DualKrigingSolver(x, y, z, variogram_model="linear", sill=1.0, range_param=60.0)
        self.assertEqual(dual.variogram_model, "spherical")

    def test_drift_analyzer_receives_spherical_model(self):
        """F4: the drift analyzer is native-only, so even engine='pykrige' + 'linear' hands it 'spherical'."""
        solver = self.make_solver(kriging_engine="pykrige", post_variogram_model="linear")
        solver.drift_analyzer = True
        fake = KrigingResult(bedrock_grid=np.full((12, 12), 40.0), variance_grid=np.zeros((12, 12)))
        with patch("pysole.solver.DriftAnalyzer") as mock_analyzer, \
                patch("pysole.solver.kriging_interpolation", return_value=fake):
            mock_analyzer.return_value.run_diagnostics.return_value = []
            solver._execute_kriging_pass(
                target_type="D", points=solver.migrated_points, krig_method="universal", drift_terms=["sia"],
                var_model="linear", zero_boundary=False, pass_name="Pass 2: Post-Migration",
            )
        self.assertEqual(mock_analyzer.return_value.run_diagnostics.call_args.kwargs["variogram_model"], "spherical")


class TestVariancePropagation(ReviewTestCase):
    def _direct(self, solver, target):
        sample_pts = solver.get_sample_points("pre_migration", target)
        return kriging_interpolation(
            sample_points=sample_pts, geometry=solver.geometry, method="ordinary", variogram_model="spherical",
            dem_grid=solver.dem_grid, opt_slope_grid=solver.opt_slope, drift_terms=[],
            outline_mask=solver.outline_mask, include_zero_boundary_condition=False, n_cores=1,
            engine="native", show_progress=False, return_variance=True,
        )

    def test_product_target_variance_propagation(self):
        """F3: for target P, sigma_T^2 = sigma_P^2 / sin^2(alpha) with the floored slope (sloped, non-uniform alpha)."""
        solver = self.make_solver(n_cores=1)
        solver.opt_slope = 0.04 + 0.01 * np.arange(12)[None, :] * np.ones((12, 1))  # 0.04 .. 0.15 rad, floor engaged
        direct = self._direct(solver, "P")
        res = solver._execute_kriging_pass(
            target_type="P", points=solver.survey_points, krig_method="ordinary", drift_terms=[],
            var_model="spherical", zero_boundary=False, pass_name="Pass 1: Pre-Migration", return_variance=True,
        )
        safe = np.maximum(np.sin(solver.opt_slope), np.sin(np.radians(solver.slope_floor_deg)))
        self.assertTrue(np.any(np.sin(solver.opt_slope) < np.sin(np.radians(solver.slope_floor_deg))))
        np.testing.assert_allclose(res.bedrock_grid, direct.bedrock_grid / safe, rtol=1e-10)
        np.testing.assert_allclose(res.variance_grid, direct.variance_grid / safe**2, rtol=1e-10)
        self.assertTrue(np.all(np.isfinite(res.variance_grid)) and np.all(res.variance_grid >= 0.0))

    def test_direct_target_variance_passthrough(self):
        """F3: for targets T / D the variance is passed through unchanged."""
        solver = self.make_solver(n_cores=1)
        direct = self._direct(solver, "D")
        res = solver._execute_kriging_pass(
            target_type="D", points=solver.survey_points, krig_method="ordinary", drift_terms=[],
            var_model="spherical", zero_boundary=False, pass_name="Pass 1: Pre-Migration", return_variance=True,
        )
        np.testing.assert_allclose(res.bedrock_grid, direct.bedrock_grid, rtol=1e-10)
        np.testing.assert_allclose(res.variance_grid, direct.variance_grid, rtol=1e-10)

    def test_lu_solve_variance_matches_dense_inverse(self):
        """The per-chunk LU variance equals the textbook formula with the explicit inverse (1e-8)."""
        from scipy.spatial.distance import cdist
        from pysole.interpolation import built_in_kriging_interpolation

        rng = np.random.default_rng(0)
        n = 60
        pts = np.column_stack([rng.uniform(0, 1000, n), rng.uniform(0, 1000, n), rng.normal(10, 2, n)])
        xc, yc = np.linspace(0, 1000, 25), np.linspace(0, 1000, 20)
        z, v = built_in_kriging_interpolation(
            pts, xc, yc, method="ordinary", variogram_params=(300.0, 1.0, 0.0), show_progress=False, n_cores=3,
        )
        d = cdist(pts[:, :2], pts[:, :2])
        gamma = np.where(d == 0, 0.0, spherical_variogram(d, 300.0, 1.0, 0.0))
        k_aug = np.zeros((n + 1, n + 1))
        k_aug[:n, :n] = gamma + np.eye(n) * 1e-6
        k_aug[:n, n] = k_aug[n, :n] = 1.0
        k_inv = np.linalg.inv(k_aug)
        xx, yy = np.meshgrid(xc, yc)
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
        rhs = np.vstack([spherical_variogram(cdist(pts[:, :2], grid_pts), 300.0, 1.0, 0.0), np.ones((1, len(grid_pts)))])
        w = k_inv @ rhs
        z_ref = (w[:n] * pts[:, 2:3]).sum(0)
        v_ref = np.maximum((w[:n] * rhs[:n]).sum(0) + w[n], 0.0)
        np.testing.assert_allclose(z.ravel(), z_ref, atol=1e-8)
        np.testing.assert_allclose(v.ravel(), v_ref, atol=1e-8)


class TestTraveltimeExportWarning(ReviewTestCase):
    def test_warning_when_migration_is_skipped(self):
        """F7: traveltime exports requested without a migration produce one warning naming both flags."""
        for reason in ("depth", "no_migration"):
            if reason == "depth":
                solver = self.make_solver(config={"outputs": {"save_traveltime_grid": True, "save_traveltime_uncertainty": True}})
            else:
                geom = GridGeometry.create((12, 12), dx=10.0, dy=10.0, bounds=BOUNDS)
                solver = Solver(
                    dem=np.full((12, 12), 1000.0), bounds=geom.bounds, perform_migration=False,
                    output_dir=str(self.output_dir / "nomig"),
                    config={"outputs": {"save_traveltime_grid": True, "save_traveltime_uncertainty": True}},
                )
            pts = SURVEY[:, [0, 1, 2, 3]].copy()
            pts[:, 3] = [0.3, 0.4, 0.35, 0.2, 0.5, 0.25]
            with self.assertLogs("pysole", level="WARNING") as cm:
                solver.migrate_eikonal(travel_times=pts)
            msgs = [m for m in cm.output if "save_traveltime_grid" in m]
            self.assertEqual(len(msgs), 1, cm.output)
            self.assertIn("save_traveltime_uncertainty", msgs[0])

    def test_no_warning_without_traveltime_flags(self):
        solver = self.make_solver()
        with self.assertNoLogs("pysole", level="WARNING"):
            solver.migrate_eikonal(travel_times=SURVEY.copy())


class TestMemoryGuard(unittest.TestCase):
    N, ND, GRID = 6000, 1, 1_000_000

    def est(self, threads, var=True):
        return estimate_native_kriging_bytes(self.N, self.ND, threads, var, self.GRID)

    def plan(self, budget_bytes, threads=8, variance=True, fraction=0.5, n=None):
        with patch("pysole.memory.available_memory_bytes", return_value=int(budget_bytes / fraction)):
            return plan_native_kriging(n or self.N, self.ND, threads, variance, fraction, self.GRID)

    def test_chunk_size_formula(self):
        self.assertEqual(kriging_chunk_size(1), 10000)
        self.assertEqual(kriging_chunk_size(3000), 5_000_000 // 3000)
        self.assertEqual(kriging_chunk_size(50_000), 500)

    def test_estimate_scales_with_threads_and_variance(self):
        self.assertLess(self.est(1), self.est(8))
        self.assertLessEqual(self.est(4, False), self.est(4, True))
        # threads beyond the number of chunks add nothing
        self.assertEqual(
            estimate_native_kriging_bytes(100, 1, 64, True, 500),
            estimate_native_kriging_bytes(100, 1, 1, True, 500),
        )

    def test_fits_as_requested(self):
        plan = self.plan(self.est(8) * 2)
        self.assertEqual((plan.n_threads, plan.return_variance, plan.message), (8, True, ""))

    def test_threads_are_reduced_first(self):
        # beyond ~3 threads the per-thread working sets exceed the (thread-independent) assembly peak
        budget = 0.5 * (self.est(4) + self.est(5))
        with self.assertLogs("pysole", level="INFO") as cm:
            plan = self.plan(budget)
        self.assertEqual(plan.n_threads, 4)
        self.assertTrue(plan.return_variance)
        self.assertTrue(any("Reducing worker threads 8 -> 4" in m for m in cm.output), cm.output)

    def test_variance_skipped_only_if_single_thread_does_not_fit(self):
        n = 200  # small N: the per-thread working set (not the N x N matrices) dominates
        e = lambda t, v: estimate_native_kriging_bytes(n, 1, t, v, self.GRID)  # noqa: E731
        self.assertLess(e(1, False), e(1, True))
        budget = 0.5 * (e(1, False) + e(1, True))
        with self.assertLogs("pysole", level="WARNING") as cm:
            plan = self.plan(budget, threads=4, n=n)
        self.assertFalse(plan.return_variance)
        self.assertTrue(any("Kriging variance skipped" in m for m in cm.output), cm.output)

    def test_nothing_fits_proceeds_single_threaded_with_warning(self):
        with self.assertLogs("pysole", level="WARNING") as cm:
            plan = self.plan(1000)
        self.assertEqual((plan.n_threads, plan.return_variance), (1, False))
        self.assertTrue(any("[Memory Guard]" in m for m in cm.output), cm.output)

    def test_unknown_ram_disables_guard(self):
        with patch("pysole.memory.available_memory_bytes", return_value=None):
            plan = plan_native_kriging(self.N, self.ND, 8, True, 0.5, self.GRID)
        self.assertEqual((plan.n_threads, plan.return_variance), (8, True))

    def test_engine_validates_fraction(self):
        from pysole.interpolation import built_in_kriging_interpolation

        pts = np.array([[1.0, 1.0, 1.0], [5.0, 5.0, 2.0], [9.0, 2.0, 3.0]])
        for bad in (0.0, -1.0, 0.95):
            with self.assertRaises(ValueError):
                built_in_kriging_interpolation(pts, np.arange(3.0), np.arange(3.0), max_memory_fraction=bad)

    def test_available_memory_is_positive(self):
        from pysole.memory import available_memory_bytes

        avail = available_memory_bytes()
        self.assertTrue(avail is None or avail > 0)

    def test_regression_variance_is_nan_not_zero(self):
        pytest = __import__("importlib").util.find_spec("pykrige")
        if pytest is None:
            self.skipTest("pykrige not installed")
        from pysole.interpolation import pykrige_kriging_interpolation

        rng = np.random.default_rng(1)
        pts = np.column_stack([rng.uniform(0, 100, 30), rng.uniform(0, 100, 30), rng.normal(5, 1, 30)])
        _, var = pykrige_kriging_interpolation(pts, np.linspace(0, 100, 8), np.linspace(0, 100, 8), method="regression")
        self.assertTrue(np.all(np.isnan(var)))


class TestConfigKeyHandling(unittest.TestCase):
    def test_unknown_keys_are_pruned_with_one_warning(self):
        user = {
            "outputs": {"compute_uncertainty": False, "save_thickness_grid": True, "save_thikness_uncertainty": True},
            "bogus_section": {"x": 1},
            "inputs": {"show_progress": False},
        }
        cleaned, unknown = sanitize_config(user)
        self.assertEqual(cleaned["outputs"], {"save_thickness_grid": True})
        self.assertNotIn("bogus_section", cleaned)
        self.assertEqual(cleaned["inputs"], {"show_progress": False})
        joined = " ".join(unknown)
        for path in ("outputs.compute_uncertainty", "outputs.save_thikness_uncertainty", "bogus_section"):
            self.assertIn(path, joined)
        self.assertIn("did you mean 'outputs.save_thickness_uncertainty'", joined)
        self.assertEqual(len(unknown), 3)
        self.assertIn("compute_uncertainty", user["outputs"])  # the caller's dict is not modified

    def test_valid_config_is_returned_unchanged(self):
        user = {"outputs": {"save_thickness_grid": True}, "kriging_parameters": {"pre_migration": {"method": "ordinary"}}}
        cleaned, unknown = sanitize_config(user)
        self.assertIs(cleaned, user)
        self.assertEqual(unknown, [])
        self.assertEqual(sanitize_config(None), (None, []))

    def test_section_must_be_an_object(self):
        with self.assertRaises(ConfigError):
            sanitize_config({"outputs": 5})
        with self.assertRaises(ConfigError):
            sanitize_config({"kriging_parameters": {"pre_migration": "ordinary"}})

    def test_load_config_warns_once_and_continues(self):
        with patch("pysole.config.logger") as mock_logger:  # setup_logging() inside load_config resets handlers
            cfg = load_config({"outputs": {"compute_uncertainty": True}, "inputs": {"n_cores": 2}})
        msgs = [str(c) for c in mock_logger.warning.call_args_list if "unknown configuration key" in str(c)]
        self.assertEqual(len(msgs), 1, mock_logger.warning.call_args_list)
        self.assertIn("outputs.compute_uncertainty", msgs[0])
        self.assertNotIn("compute_uncertainty", cfg["outputs"])
        self.assertEqual(cfg["inputs"]["n_cores"], 2)

    def test_solver_prunes_unknown_keys(self):
        geom = GridGeometry.create((6, 6), dx=10.0, dy=10.0, bounds=(0.0, 0.0, 60.0, 60.0))
        with tempfile.TemporaryDirectory() as tmp, self.assertLogs("pysole", level="WARNING") as cm:
            solver = Solver(
                dem=np.full((6, 6), 100.0), bounds=geom.bounds, output_dir=tmp,
                config={"outputs": {"compute_uncertainty": False, "save_thickness_grid": True}},
            )
        self.assertEqual(solver.config["outputs"], {"save_thickness_grid": True})
        self.assertTrue(any("outputs.compute_uncertainty" in m for m in cm.output), cm.output)

    def test_outputs_config_warns_on_unknown_key(self):
        with self.assertLogs("pysole", level="WARNING") as cm:
            cfg = OutputsConfig.from_dict({"compute_uncertainty": False, "save_thickness_grid": True})
        self.assertTrue(cfg.save_thickness_grid)
        self.assertTrue(any("outputs.compute_uncertainty" in m for m in cm.output), cm.output)

    def test_shipped_configs_are_clean(self):
        root = Path(__file__).resolve().parents[1]
        configs = [root / "pysole.json", root / "examples" / "gok" / "pysole_gok.json", root / "examples" / "wuk" / "pysole_wuk.json"]
        if not (root / "pysole.json").exists():
            self.skipTest("not running from a source checkout")
        import json

        for path in configs:
            if not path.exists():
                continue
            _, unknown = sanitize_config(json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(unknown, [], path.name)

    def test_default_config_is_its_own_schema(self):
        cleaned, unknown = sanitize_config(DEFAULT_CONFIG)
        self.assertEqual(unknown, [])


class TestReadmeApiExamples(unittest.TestCase):
    def test_readme_solver_calls_exist(self):
        """F5: every ``model.<method>(...)`` / ``solver.<method>(...)`` call in README Python blocks exists on Solver."""
        root = Path(__file__).resolve().parents[1]
        readme = root / "README.md"
        if not readme.exists():
            self.skipTest("not running from a source checkout")
        text = readme.read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
        missing = []
        for block in blocks:
            for match in re.finditer(r"\b(?:model|solver)\.([A-Za-z_]\w*)\s*\(", block):
                name = match.group(1)
                if not hasattr(Solver, name):
                    missing.append(name)
        self.assertEqual(sorted(set(missing)), [])

    def test_readme_has_no_removed_option(self):
        root = Path(__file__).resolve().parents[1]
        readme = root / "README.md"
        if not readme.exists():
            self.skipTest("not running from a source checkout")
        self.assertNotIn("compute_uncertainty", readme.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
