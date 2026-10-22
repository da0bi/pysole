"""
Tests of the exact Drift Analyzer evaluation engine (v0.5.0, Phase 4).

The analyzer must reproduce an independent per-fold refit of the production kriging system, expose the
ΔAICc / ΔRMSE % columns, warn about (not penalise) the SIA drift on the product target, parse the interactive
answers strictly, and degrade deterministically when the memory plan asks for a subset.
"""

import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pysole.drift import DRIFT_PRIMITIVES, build_drift_suite, slugify_drift_name
from pysole.drift_analyzer import (
    FAILED_AICC,
    CandidateDriftResult,
    DriftAnalyzer,
    DriftComparisonRecord,
    add_ranking_deltas,
    farthest_point_indices,
    format_drift_comparison_table,
    format_ranking_table,
    parse_model_selection,
    resolve_rmse_unit,
    save_drift_comparison_files,
)
from pysole.interpolation import DriftBasis
from pysole.kriging_core import KrigingSystem
from pysole.memory import CV_MIN_EVALUATION_POINTS, CvMemoryPlan, plan_cv_memory
from pysole.solver import Solver
from tests.test_drift_covariates import make_asymmetric_fixture, make_points

VARIOGRAM = {"range": 150.0, "sill": 4.0, "nugget": 0.05}
CANDIDATES = [
    [], ["sia"], ["z_dem"], ["curvature_dem"], ["linear_xy"], ["quadratic_xy"],
    ["sia", "z_dem"], ["z_dem", "curvature_dem", "linear_xy"],
]


def make_result(name, terms, rmse, aicc, penalized=False, warning=None):
    return CandidateDriftResult(
        name=name, terms=terms, n_terms=len(terms), vif_max=1.0, is_multicollinear=False, cv_rmse=rmse, cv_mae=rmse,
        cv_r2=0.5, aicc=aicc, is_penalized=penalized, warning_msg=warning,
    )


def refit_residuals(x, y, z, cols, blocks, targets, params):
    """Independent leave-block-out residuals: a fresh kriging system per block (no update formulas)."""
    out = []
    for block, target in zip(blocks, targets):
        keep = np.setdiff1d(np.arange(len(z)), block)
        system = KrigingSystem(x[keep], y[keep], z[keep], cols[keep] if cols.shape[1] else None, "spherical", params)
        rows = cols[target].T if cols.shape[1] else None
        estimate, _ = system.predict_points(x[target], y[target], rows)
        out.append(z[target] - estimate)
    return np.concatenate(out)


class AnalyzerFixture(unittest.TestCase):
    def setUp(self):
        self.geom, self.dem, self.slope, self.curv, self.grids = make_asymmetric_fixture()
        self.x, self.y, self.z = make_points(self.geom, self.dem, n=60)

    def run_cv(self, candidates=CANDIDATES, target="D", **kwargs):
        analyzer = DriftAnalyzer(interpolation_target=target, show_progress=False)
        return analyzer.evaluate_spatial_cv(
            self.x, self.y, self.z, self.geom, self.grids, candidates, "spherical", dict(VARIOGRAM), **kwargs
        )


class TestMatchesIndependentRefit(AnalyzerFixture):
    def check(self, blocks, targets, **kwargs):
        results = {r.name: r for r in self.run_cv(**kwargs)}
        for terms in CANDIDATES:
            basis = DriftBasis(terms, self.x, self.y, geometry=self.geom, drift_grids=self.grids)
            cols = basis.evaluate(self.x, self.y)
            residual = refit_residuals(
                self.x, self.y, self.z, cols, blocks, targets, (150.0, 4.0, 0.05)
            )
            name = "ordinary" if not terms else " + ".join(terms)
            got = next(r for r in results.values() if r.terms == terms)
            with self.subTest(model=name):
                self.assertAlmostEqual(got.cv_rmse, float(np.sqrt(np.mean(residual**2))), delta=1e-7 * got.cv_rmse)
                self.assertAlmostEqual(got.cv_mae, float(np.mean(np.abs(residual))), delta=1e-7 * got.cv_rmse)

    def test_buffer_blocks_every_point_evaluated(self):
        radius = 75.0
        xy = np.column_stack((self.x, self.y))
        d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        blocks = [np.flatnonzero(d[i] <= radius) for i in range(len(self.z))]
        targets = [np.array([i]) for i in range(len(self.z))]
        self.check(blocks, targets, cv_mode="buffer", buffer_radius_m=radius)

    def test_leave_one_profile_out(self):
        profiles = np.arange(len(self.z)) % 5
        blocks = [np.flatnonzero(profiles == p) for p in range(5)]
        self.check(blocks, blocks, cv_mode="lopo", profile_ids=profiles)

    def test_ordinary_candidate_is_reported(self):
        names = [r.name for r in self.run_cv()]
        self.assertIn("ordinary", names)
        ordinary = next(r for r in self.run_cv() if r.terms == [])
        self.assertEqual(ordinary.n_terms, 0)

    def test_full_default_suite_is_ranked(self):
        results = self.run_cv(candidates=build_drift_suite(None), cv_mode="buffer", buffer_radius_m=60.0)
        self.assertEqual(len(results), 24)
        keys = [(r.is_penalized, r.aicc, r.cv_rmse) for r in results]
        self.assertEqual(keys, sorted(keys))


class TestProductTargetWarning(AnalyzerFixture):
    def test_one_warning_per_sia_model_and_no_penalty(self):
        suite = [[], ["sia"], ["sia", "z_dem"], ["z_dem"]]
        with self.assertLogs("pysole", level="WARNING") as cm:
            results = self.run_cv(candidates=suite, target="P", buffer_radius_m=60.0)
        warned = [m for m in cm.output if "'sia' drift on the BSS product P target" in m]
        self.assertEqual(len(warned), 2, cm.output)
        by_name = {r.name: r for r in results}
        for name in ("sia", "sia + z_dem"):
            self.assertIsNotNone(by_name[name].warning_msg)
            self.assertFalse(by_name[name].is_penalized)
        self.assertIsNone(by_name["z_dem"].warning_msg)

    def test_no_warning_on_direct_targets(self):
        with self.assertNoLogs("pysole", level="WARNING"):
            results = self.run_cv(candidates=[[], ["sia"]], target="D", buffer_radius_m=60.0)
        self.assertTrue(all(r.warning_msg is None for r in results))


class TestFailedCandidates(AnalyzerFixture):
    def test_too_few_points_is_penalised_and_ranked_last(self):
        self.x, self.y, self.z = self.x[:6], self.y[:6], self.z[:6]
        results = self.run_cv(candidates=[[], ["quadratic_xy"]], buffer_radius_m=1e-3)
        self.assertEqual([r.name for r in results], ["ordinary", "quadratic_xy"])
        failed = results[1]
        self.assertTrue(failed.is_penalized)
        self.assertEqual(failed.aicc, FAILED_AICC)
        self.assertTrue(np.isnan(failed.delta_aicc) and np.isnan(failed.delta_rmse_pct))
        self.assertIn("singularity", failed.warning_msg)

    def test_all_failed_is_reported(self):
        self.x, self.y, self.z = self.x[:6], self.y[:6], self.z[:6]
        with self.assertLogs("pysole", level="WARNING") as cm:
            results = self.run_cv(candidates=[["quadratic_xy"]], buffer_radius_m=1e-3)
        self.assertTrue(results[0].is_penalized)
        self.assertTrue(any("No candidate could be cross-validated" in m for m in cm.output))


class TestRankingDeltas(unittest.TestCase):
    def test_deltas_reference_the_best_models(self):
        results = [
            make_result("a", ["sia"], 2.0, 100.0),
            make_result("b", ["z_dem"], 2.5, 104.0),
            make_result("c", ["linear_xy"], 1.0, 90.0, penalized=True),
            make_result("d", ["quadratic_xy"], 9999.0, FAILED_AICC, penalized=True),
        ]
        add_ranking_deltas(results)
        self.assertEqual([r.delta_aicc for r in results[:2]], [0.0, 4.0])
        self.assertEqual(results[0].delta_rmse_pct, 0.0)
        self.assertAlmostEqual(results[1].delta_rmse_pct, 25.0)
        self.assertAlmostEqual(results[2].delta_aicc, -10.0)  # penalised models are not the reference
        self.assertTrue(np.isnan(results[3].delta_aicc) and np.isnan(results[3].delta_rmse_pct))

    def test_all_penalised_uses_all_usable_models(self):
        results = [make_result("a", ["sia"], 2.0, 100.0, True), make_result("b", ["z_dem"], 2.0, 103.0, True)]
        add_ranking_deltas(results)
        self.assertEqual([r.delta_aicc for r in results], [0.0, 3.0])

    def test_empty(self):
        add_ranking_deltas([])

    def test_table_columns_and_status(self):
        results = [
            make_result("sia", ["sia"], 2.0, 100.0, warning="margin artifact"),
            make_result("z_dem", ["z_dem"], 2.5, 104.0),
            make_result("bad", ["quadratic_xy"], 9999.0, FAILED_AICC, penalized=True),
        ]
        add_ranking_deltas(results)
        table = format_ranking_table(results, "TITLE")
        for header in ("dRMSE %", "dAICc", "AICc", "VIF", "Status", "TITLE"):
            self.assertIn(header, table)
        rows = {line.split()[1]: line for line in table.splitlines() if line[:1].isdigit()}
        self.assertIn("WARNING", rows["sia"])
        self.assertIn("VALID", rows["z_dem"])
        self.assertIn("PENALIZED", rows["bad"])
        self.assertIn("n/a", rows["bad"])
        self.assertIn("WARNING: margin artifact", table)

    def test_table_custom_units_and_large_aicc(self):
        """Large AICc (> 100,000) does not get falsely marked as penalized or n/a."""
        results = [
            make_result("sia", ["sia"], 336.62, 103808.0),
            make_result("ordinary", [], 404.23, 107000.0),
            make_result("bad", ["quadratic_xy"], 9999.0, FAILED_AICC, penalized=True),
        ]
        add_ranking_deltas(results)
        table = format_ranking_table(results, "TITLE", rmse_unit="ns")
        self.assertIn("RMSE (ns)", table)
        rows = {line.split()[1]: line for line in table.splitlines() if line[:1].isdigit()}
        self.assertIn("VALID", rows["sia"])
        self.assertIn("103808.0", rows["sia"])
        self.assertIn("336.62", rows["sia"])
        self.assertIn("0.00", rows["sia"])  # dRMSE %
        self.assertIn("PENALIZED", rows["bad"])

    def test_resolve_rmse_unit(self):
        # Pass 1 with traveltime (radar velocity)
        self.assertEqual(resolve_rmse_unit(mode="pass_1", interpolation_target="T", survey_data_type="one_way_traveltime", velocity=0.16), "ns")
        self.assertEqual(resolve_rmse_unit(mode="pass_1", interpolation_target="P", survey_data_type="one_way_traveltime", velocity=0.16), "ns x rad")
        # Pass 1 with seismic velocity
        self.assertEqual(resolve_rmse_unit(mode="pass_1", interpolation_target="T", survey_data_type="one_way_traveltime", velocity=3800.0), "ms")
        self.assertEqual(resolve_rmse_unit(mode="pass_1", interpolation_target="P", survey_data_type="one_way_traveltime", velocity=3800.0), "ms x rad")
        # Pass 1 with unknown velocity fallback
        self.assertEqual(resolve_rmse_unit(mode="pass_1", interpolation_target="T", survey_data_type="one_way_traveltime", velocity=None), "ns")
        # Pass 1 with depth survey data
        self.assertEqual(resolve_rmse_unit(mode="pass_1", interpolation_target="T", survey_data_type="depth"), "m")
        self.assertEqual(resolve_rmse_unit(mode="pass_1", interpolation_target="P", survey_data_type="depth"), "m x rad")
        # Pass 2 with depth and product targets
        self.assertEqual(resolve_rmse_unit(mode="pass_2", interpolation_target="D"), "m")
        self.assertEqual(resolve_rmse_unit(mode="pass_2", interpolation_target="P"), "m x rad")



class TestParseModelSelection(unittest.TestCase):
    def setUp(self):
        self.results = [
            make_result("sia + z_dem", ["sia", "z_dem"], 1.0, 10.0),
            make_result("ordinary", [], 1.1, 11.0),
            make_result("linear_xy", ["linear_xy"], 1.2, 12.0),
        ]

    def parse(self, text):
        return parse_model_selection(text, self.results)

    def test_empty_keeps_the_recommendation(self):
        for text in ("", "   ", "#"):
            self.assertEqual(self.parse(text), ([self.results[0]], None))

    def test_row_numbers(self):
        for text, row in (("1", 0), ("3", 2), (" #2 ", 1)):
            chosen, error = self.parse(text)
            self.assertIsNone(error)
            self.assertEqual(chosen, [self.results[row]])

    def test_row_out_of_range(self):
        for text in ("0", "4", "99"):
            chosen, error = self.parse(text)
            self.assertIsNone(chosen)
            self.assertIn("1 to 3", error)

    def test_names_and_term_lists(self):
        for text in ("sia + z_dem", "z_dem, sia", "['sia', 'z_dem']", '("z_dem" + "sia")', "[sia,z_dem]"):
            with self.subTest(text=text):
                chosen, error = self.parse(text)
                self.assertIsNone(error)
                self.assertEqual(chosen, [self.results[0]])
        chosen, error = self.parse("linear_xy")
        self.assertIsNone(error)
        self.assertEqual(chosen, [self.results[2]])

    def test_ordinary(self):
        for text in ("ordinary", "Ordinary", "[]", "()"):
            with self.subTest(text=text):
                chosen, error = self.parse(text)
                self.assertIsNone(error)
                self.assertEqual(chosen, [self.results[1]])

    def test_ranges_and_multi_selection(self):
        # Range 1-3
        chosen, err = self.parse("1-3")
        self.assertIsNone(err)
        self.assertEqual(chosen, self.results)

        # Range 1-2
        chosen, err = self.parse("1-2")
        self.assertIsNone(err)
        self.assertEqual(chosen, [self.results[0], self.results[1]])

        # Inverted range returns clear validation error
        chosen, err = self.parse("2-1")
        self.assertIsNone(chosen)
        self.assertIn("must be <= end row", err)

        # Comma list
        chosen, err = self.parse("1, 3")
        self.assertIsNone(err)
        self.assertEqual(chosen, [self.results[0], self.results[2]])

        # Mixed range and list with duplicates preserves order of appearance
        chosen, err = self.parse("3, 1-2, 3")
        self.assertIsNone(err)
        self.assertEqual(chosen, [self.results[2], self.results[0], self.results[1]])

    def test_range_out_of_bounds(self):
        chosen, err = self.parse("1-5")
        self.assertIsNone(chosen)
        self.assertIn("out of bounds", err)

    def test_unknown_term(self):
        chosen, error = self.parse("sia + banana")
        self.assertIsNone(chosen)
        self.assertIn("banana", error)
        for term in DRIFT_PRIMITIVES:
            self.assertIn(term, error)

    def test_valid_terms_that_were_not_evaluated(self):
        chosen, error = self.parse("curvature_dem")
        self.assertIsNone(chosen)
        self.assertIn("not evaluated", error)


class TestDriftComparisonFormatting(unittest.TestCase):
    def test_slugify_drift_name(self):
        self.assertEqual(slugify_drift_name([]), "ordinary")
        self.assertEqual(slugify_drift_name(["sia", "curvature_dem"]), "sia_curvature_dem")
        self.assertEqual(slugify_drift_name("sia + curvature_dem"), "sia_curvature_dem")
        self.assertEqual(slugify_drift_name("ordinary"), "ordinary")

    def test_format_and_save_comparison_tables(self):
        rec1 = DriftComparisonRecord(
            pass_label="Pass 2",
            rank=1,
            model_name="sia + curvature_dem",
            model_slug="sia_curvature_dem",
            terms=["sia", "curvature_dem"],
            cv_rmse=2.5,
            delta_rmse_pct=0.0,
            aicc=100.0,
            delta_aicc=0.0,
            vif_max=1.2,
            rmse_unit="m",
            status="OK",
            thick_mean=50.0,
            thick_median=48.0,
            thick_std=10.0,
            thick_max=85.0,
            bss_mean=90.0,
            bss_median=88.0,
            bss_std=15.0,
            bss_max=120.0,
        )
        rec2 = DriftComparisonRecord(
            pass_label="Pass 2",
            rank=2,
            model_name="ordinary",
            model_slug="ordinary",
            terms=[],
            cv_rmse=3.0,
            delta_rmse_pct=20.0,
            aicc=110.0,
            delta_aicc=10.0,
            vif_max=1.0,
            rmse_unit="m",
            status="FAILED",
            is_failed=True,
        )
        table_str = format_drift_comparison_table([rec1, rec2], "Pass 2")
        self.assertIn("DRIFT MODEL INTERPOLATION COMPARISON TABLE (PASS 2)", table_str)
        self.assertIn("sia + curvature_dem", table_str)
        self.assertIn("FAILED", table_str)

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_paths = save_drift_comparison_files(tmp_dir, post_records=[rec1, rec2])
            self.assertEqual(len(saved_paths), 3)
            for p in saved_paths:
                self.assertTrue(p.exists())
                content = p.read_text(encoding="utf-8")
                self.assertIn("sia + curvature_dem", content)
                self.assertIn("FAILED", content)

    def test_format_and_save_comparison_tables_with_uncertainty(self):
        # Pass 1 with uncertainty
        p1_rec = DriftComparisonRecord(
            pass_label="Pass 1",
            rank=1,
            model_name="sia",
            model_slug="sia",
            terms=["sia"],
            cv_rmse=1.2,
            delta_rmse_pct=0.0,
            aicc=50.0,
            delta_aicc=0.0,
            vif_max=1.0,
            rmse_unit="ns",
            status="VALID",
            target_mean=100.0,
            target_median=95.0,
            target_std=20.0,
            target_max=160.0,
            target_unit="ns",
            unc_mean=4.5,
        )
        p1_str = format_drift_comparison_table([p1_rec], "Pass 1")
        self.assertIn("Mean Unc", p1_str)
        self.assertIn("4.50", p1_str)

        with tempfile.TemporaryDirectory() as tmp_dir:
            p1_saved = save_drift_comparison_files(tmp_dir, pre_records=[p1_rec])
            self.assertEqual(len(p1_saved), 3)
            csv_content = (Path(tmp_dir) / "drift_model_comparison.csv").read_text(encoding="utf-8")
            self.assertIn("traveltime_unc_mean", csv_content)
            self.assertIn("4.5000", csv_content)
            md_content = (Path(tmp_dir) / "drift_model_comparison.md").read_text(encoding="utf-8")
            self.assertIn("Mean Uncertainty", md_content)
            self.assertIn("4.50 ns", md_content)

        # Pass 2 with uncertainty
        p2_rec = DriftComparisonRecord(
            pass_label="Pass 2",
            rank=1,
            model_name="sia",
            model_slug="sia",
            terms=["sia"],
            cv_rmse=2.0,
            delta_rmse_pct=0.0,
            aicc=80.0,
            delta_aicc=0.0,
            vif_max=1.0,
            rmse_unit="m",
            status="VALID",
            thick_mean=45.0,
            thick_median=42.0,
            thick_std=12.0,
            thick_max=80.0,
            bss_mean=85.0,
            bss_median=80.0,
            bss_std=14.0,
            bss_max=115.0,
            unc_mean=3.2,
        )
        p2_str = format_drift_comparison_table([p2_rec], "Pass 2")
        self.assertIn("Mean Unc", p2_str)
        self.assertIn("3.20", p2_str)

        with tempfile.TemporaryDirectory() as tmp_dir:
            p2_saved = save_drift_comparison_files(tmp_dir, post_records=[p2_rec])
            self.assertEqual(len(p2_saved), 3)
            csv_content = (Path(tmp_dir) / "drift_model_comparison.csv").read_text(encoding="utf-8")
            self.assertIn("thickness_unc_mean_m", csv_content)
            self.assertIn("3.2000", csv_content)
            md_content = (Path(tmp_dir) / "drift_model_comparison.md").read_text(encoding="utf-8")
            self.assertIn("Mean Unc (m)", md_content)
            self.assertIn("3.20", md_content)
            self.assertIn("Std Thick (m)", md_content)
            self.assertIn("Std BSS (kPa)", md_content)

        # Unified both Pass 1 and Pass 2
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = save_drift_comparison_files(
                tmp_dir,
                pre_records=[p1_rec],
                post_records=[p2_rec],
            )
            self.assertEqual(len(saved), 3)
            txt_content = (Path(tmp_dir) / "drift_model_comparison.txt").read_text(encoding="utf-8")
            self.assertIn("PASS 1", txt_content)
            self.assertIn("PASS 2", txt_content)

            csv_content = (Path(tmp_dir) / "drift_model_comparison.csv").read_text(encoding="utf-8")
            self.assertIn("Pass 1", csv_content)
            self.assertIn("Pass 2", csv_content)
            self.assertIn("traveltime_unc_mean", csv_content)
            self.assertIn("thickness_unc_mean_m", csv_content)

            md_content = (Path(tmp_dir) / "drift_model_comparison.md").read_text(encoding="utf-8")
            self.assertIn("Pass 1: Traveltime Field Interpolation", md_content)
            self.assertIn("Pass 2: Ice Thickness & Bedrock Topography Interpolation", md_content)
            self.assertIn("Mean Uncertainty", md_content)
            self.assertIn("Mean Unc (m)", md_content)

    def test_format_and_save_comparison_tables_without_uncertainty(self):
        p1_rec = DriftComparisonRecord(
            pass_label="Pass 1",
            rank=1,
            model_name="ordinary",
            model_slug="ordinary",
            terms=[],
            cv_rmse=1.5,
            delta_rmse_pct=0.0,
            aicc=60.0,
            delta_aicc=0.0,
            vif_max=1.0,
            rmse_unit="ns",
            status="VALID",
            target_mean=100.0,
            target_median=95.0,
            target_std=20.0,
            target_max=160.0,
            target_unit="ns",
            unc_mean=None,
        )
        p1_str = format_drift_comparison_table([p1_rec], "Pass 1")
        self.assertNotIn("Mean Unc", p1_str)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_drift_comparison_files(tmp_dir, pre_records=[p1_rec])
            csv_content = (Path(tmp_dir) / "drift_model_comparison.csv").read_text(encoding="utf-8")
            self.assertNotIn("traveltime_unc_mean", csv_content)
            md_content = (Path(tmp_dir) / "drift_model_comparison.md").read_text(encoding="utf-8")
            self.assertNotIn("Mean Uncertainty", md_content)



class TestFarthestPoints(unittest.TestCase):
    def test_deterministic_sorted_and_spread(self):
        rng = np.random.default_rng(0)
        x, y = rng.uniform(0, 1000, 300), rng.uniform(0, 1000, 300)
        first = farthest_point_indices(x, y, 40)
        np.testing.assert_array_equal(first, farthest_point_indices(x, y, 40))
        self.assertEqual(len(set(first.tolist())), 40)
        np.testing.assert_array_equal(first, np.sort(first))
        np.testing.assert_array_equal(farthest_point_indices(x, y, 300), np.arange(300))
        np.testing.assert_array_equal(farthest_point_indices(x, y, 500), np.arange(300))
        # the subset covers the area better than the first 40 points
        sub = np.column_stack((x[first], y[first]))
        head = np.column_stack((x[:40], y[:40]))
        self.assertGreater(
            np.min(np.hypot(*(sub[:, None] - sub[None]).transpose(2, 0, 1))[np.triu_indices(40, 1)]),
            np.min(np.hypot(*(head[:, None] - head[None]).transpose(2, 0, 1))[np.triu_indices(40, 1)]),
        )


class TestMemoryPlan(unittest.TestCase):
    def test_fits_into_memory(self):
        with patch("pysole.memory.available_memory_bytes", return_value=64 * 1024**3):
            plan = plan_cv_memory(5000)
        self.assertEqual(plan, CvMemoryPlan(True, None))

    def test_unknown_memory_assumes_it_fits(self):
        with patch("pysole.memory.available_memory_bytes", return_value=None):
            self.assertEqual(plan_cv_memory(50_000), CvMemoryPlan(True, None))

    def test_lu_only_when_the_inverse_does_not_fit(self):
        n = 10_000  # 3.5 N^2 doubles = 2.8 GB > 2 GB budget, 2.5 N^2 = 2.0 GB + columns of everything fits in 3 GB
        with patch("pysole.memory.available_memory_bytes", return_value=int(2.4 * 1024**3 / 0.5)):
            plan = plan_cv_memory(n, fraction=0.5)
        self.assertFalse(plan.full_inverse)
        self.assertIn("Memory Guard", plan.message)

    def test_subset_floor_and_message(self):
        with patch("pysole.memory.available_memory_bytes", return_value=1024**3):
            plan = plan_cv_memory(50_000, mean_block_size=30.0, fraction=0.5)
        self.assertFalse(plan.full_inverse)
        self.assertEqual(plan.n_eval, CV_MIN_EVALUATION_POINTS)
        self.assertIn("farthest-point subset", plan.message)


class TestMemoryFallbackIsDeterministic(AnalyzerFixture):
    def test_lu_mode_equals_full_inverse(self):
        full = {r.name: r for r in self.run_cv(buffer_radius_m=70.0)}
        with patch("pysole.drift_analyzer.plan_cv_memory", return_value=CvMemoryPlan(False, None, "")):
            lu_only = {r.name: r for r in self.run_cv(buffer_radius_m=70.0)}
        for name, res in full.items():
            self.assertAlmostEqual(lu_only[name].cv_rmse, res.cv_rmse, delta=1e-9 * res.cv_rmse)
            self.assertAlmostEqual(lu_only[name].aicc, res.aicc, delta=1e-7)

    def test_subset_is_reproducible_and_warned(self):
        plan = CvMemoryPlan(False, 30, "   [Memory Guard] test subset")
        runs = []
        for _ in range(2):
            with patch("pysole.drift_analyzer.plan_cv_memory", return_value=plan), \
                    self.assertLogs("pysole", level="WARNING") as cm:
                runs.append([(r.name, r.cv_rmse, r.aicc) for r in self.run_cv(buffer_radius_m=70.0)])
            self.assertTrue(any("test subset" in m for m in cm.output))
        self.assertEqual(runs[0], runs[1])

        # the subset result equals an independent refit of exactly the farthest-point targets
        subset = farthest_point_indices(self.x, self.y, 30)
        xy = np.column_stack((self.x, self.y))
        blocks = [np.flatnonzero(np.hypot(*(xy - xy[i]).T) <= 70.0) for i in subset]
        cols = DriftBasis(["z_dem"], self.x, self.y, geometry=self.geom, drift_grids=self.grids).evaluate(self.x, self.y)
        residual = refit_residuals(self.x, self.y, self.z, cols, blocks, [np.array([i]) for i in subset], (150.0, 4.0, 0.05))
        with patch("pysole.drift_analyzer.plan_cv_memory", return_value=plan), self.assertLogs("pysole", level="WARNING"):
            got = next(r for r in self.run_cv(candidates=[["z_dem"]], buffer_radius_m=70.0))
        self.assertAlmostEqual(got.cv_rmse, float(np.sqrt(np.mean(residual**2))), delta=1e-7 * got.cv_rmse)


class TestInteractiveSelection(AnalyzerFixture):
    def run_interactive(self, answers):
        analyzer = DriftAnalyzer(interpolation_target="D", survey_profile_column=None, show_progress=False)
        with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", side_effect=answers) as mocked, \
                patch("builtins.print") as printed:
            results = analyzer.run_diagnostics(
                self.x, self.y, self.z, self.geom, self.grids, [[], ["z_dem"], ["linear_xy"]],
                "spherical", dict(VARIOGRAM), interactive=True,
            )
        return results, mocked, "\n".join(str(c.args[0]) for c in printed.call_args_list if c.args)

    def test_invalid_answers_are_reported_until_valid(self):
        results, mocked, printed = self.run_interactive(["abc", "7", "banana", "ordinary"])
        self.assertEqual(results[0].terms, [])
        self.assertEqual(mocked.call_count, 4)
        self.assertIn("is not a number; keeping the default radius", printed)
        self.assertIn("Row 7 does not exist", printed)
        self.assertIn("Unknown drift term(s)", printed)

    def test_buffer_radius_prompt(self):
        with patch("builtins.print") as printed:
            for answer, expected in (("", 75.0), ("-3", 75.0), ("nan", 75.0), ("120", 120.0), (" 90.5 ", 90.5)):
                with self.subTest(answer=answer), patch("builtins.input", return_value=answer):
                    self.assertEqual(DriftAnalyzer._prompt_buffer_radius(150.0, 75.0), expected)
            with patch("builtins.input", side_effect=EOFError):
                self.assertEqual(DriftAnalyzer._prompt_buffer_radius(150.0, 75.0), 75.0)
        self.assertIn("must be a positive number", "\n".join(str(c.args[0]) for c in printed.call_args_list))

    def test_empty_answers_keep_the_recommendation(self):
        results, mocked, printed = self.run_interactive(["", ""])
        self.assertEqual(mocked.call_count, 2)
        self.assertIn("dAICc", printed)
        top = sorted(results, key=lambda r: (r.is_penalized, r.aicc))[0]
        self.assertIs(results[0], top)

    def test_choice_moves_to_the_front(self):
        results, _, _ = self.run_interactive(["80", "linear_xy"])
        self.assertEqual(results[0].terms, ["linear_xy"])

    def test_eof_keeps_the_recommendation(self):
        results, _, _ = self.run_interactive([EOFError(), EOFError()])
        self.assertEqual(len(results), 3)

    def test_banner_and_table_header(self):
        with patch("pysole.drift_analyzer.log_section_header") as mock_banner:
            results, mocked, printed = self.run_interactive(["", ""])
            mock_banner.assert_called_once_with("INTERACTIVE DRIFT MODEL SELECTION")
            self.assertIn("DRIFT ANALYZER CANDIDATE MODEL RECOMMENDATIONS (PASS 1)", printed)


class TestProgressDisplay(AnalyzerFixture):
    """The spinner and the loop bar only report: same ranking, right totals, nothing created when disabled."""

    SUITE = [[], ["z_dem"], ["linear_xy"]]

    def run_recorded(self, show_progress, **kwargs):
        bars, spinners = [], []

        class RecordingBar:
            def __init__(self, **bar_kwargs):
                self.kwargs = bar_kwargs
                self.total_updated = 0
                bars.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def update(self, n=1):
                self.total_updated += n

        @contextlib.contextmanager
        def recording_spinner(desc, disable=False, **spinner_kwargs):
            spinners.append((desc, disable))
            yield

        analyzer = DriftAnalyzer(interpolation_target="D", show_progress=show_progress)
        with patch("pysole.drift_analyzer.get_progress_bar", RecordingBar), \
                patch("pysole.drift_analyzer.activity_indicator", recording_spinner):
            results = analyzer.evaluate_spatial_cv(
                self.x, self.y, self.z, self.geom, self.grids, self.SUITE, "spherical", dict(VARIOGRAM), **kwargs
            )
        return results, bars, spinners

    def test_buffer_mode_reports_points(self):
        results, bars, spinners = self.run_recorded(True, cv_mode="buffer", buffer_radius_m=60.0)
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].kwargs["unit"], "points")
        self.assertFalse(bars[0].kwargs["disable"])
        self.assertEqual(bars[0].kwargs["total"], len(self.z))
        self.assertEqual(bars[0].total_updated, len(self.z))
        self.assertEqual(len(spinners), 1)
        self.assertFalse(spinners[0][1])
        self.assertIn(f"N = {len(self.z)}", spinners[0][0])

    def test_lopo_mode_reports_profiles(self):
        profiles = np.arange(len(self.z)) % 5
        _, bars, _ = self.run_recorded(True, cv_mode="lopo", profile_ids=profiles)
        self.assertEqual(bars[0].kwargs["unit"], "profiles")
        self.assertEqual(bars[0].kwargs["total"], 5)
        self.assertEqual(bars[0].total_updated, 5)

    def test_disabled_display_and_identical_results(self):
        shown, _, _ = self.run_recorded(True, cv_mode="buffer", buffer_radius_m=60.0)
        hidden, bars, spinners = self.run_recorded(False, cv_mode="buffer", buffer_radius_m=60.0)
        self.assertTrue(bars[0].kwargs["disable"])
        self.assertTrue(spinners[0][1])
        self.assertEqual([(r.name, r.cv_rmse, r.aicc) for r in shown], [(r.name, r.cv_rmse, r.aicc) for r in hidden])

    def test_default_is_to_show_progress(self):
        self.assertTrue(DriftAnalyzer().show_progress)

    def test_pass_1_and_pass_2_labels_in_progress(self):
        for mode, expected_pass in [("pre_migration", "Pass 1"), ("post_migration", "Pass 2")]:
            analyzer = DriftAnalyzer(mode=mode, interpolation_target="D", show_progress=True)
            self.assertEqual(analyzer.pass_label, expected_pass)
            bars, spinners = [], []

            class RecordingBar:
                def __init__(self, **bar_kwargs):
                    self.kwargs = bar_kwargs
                    bars.append(self)
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False
                def update(self, n=1):
                    pass

            @contextlib.contextmanager
            def recording_spinner(desc, disable=False, **spinner_kwargs):
                spinners.append((desc, disable))
                yield

            with patch("pysole.drift_analyzer.get_progress_bar", RecordingBar), \
                    patch("pysole.drift_analyzer.activity_indicator", recording_spinner):
                analyzer.evaluate_spatial_cv(
                    self.x, self.y, self.z, self.geom, self.grids, self.SUITE, "spherical", dict(VARIOGRAM),
                    cv_mode="buffer", buffer_radius_m=60.0
                )
            self.assertIn(f"[Drift Analyzer] {expected_pass}: factorising kriging system", spinners[0][0])
            self.assertIn(f"[Drift Analyzer] {expected_pass}: leave-block-out CV", bars[0].kwargs["desc"])


class TestSolverUsesFittedVariogram(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        geom, self.dem, self.slope, _, _ = make_asymmetric_fixture()
        x, y, z = make_points(geom, self.dem)
        self.solver = Solver(
            dem=self.dem, bounds=geom.bounds, survey_data_type="depth", post_variogram_model="gaussian",
            output_dir=os.path.join(self._tmp.name, "out"), show_progress=False,
        )
        self.solver.opt_slope = self.slope
        self.solver.opt_kc = 0.03
        pts = np.column_stack((x, y, np.zeros_like(x), z))  # columns: x, y, surface elevation, depth
        self.solver.survey_points = pts.copy()
        self.solver.migrated_points = pts.copy()

    def test_recommend_drift_model_passes_stage_model_and_fitted_parameters(self):
        self.solver.opt_variogram_params = {"range": 123.0, "sill": 7.0, "nugget": 0.1}
        with patch("pysole.solver.DriftAnalyzer") as mock_analyzer:
            mock_analyzer.return_value.run_diagnostics.return_value = []
            self.solver.recommend_drift_model(stage="post_migration", interactive=False)
        kwargs = mock_analyzer.return_value.run_diagnostics.call_args.kwargs
        self.assertEqual(kwargs["variogram_model"], "gaussian")
        self.assertEqual(kwargs["variogram_params"], {"range": 123.0, "sill": 7.0, "nugget": 0.1})
        self.assertEqual(kwargs["candidates"], build_drift_suite(None))
        init = mock_analyzer.call_args.kwargs
        self.assertEqual(init["mode"], "post_migration")
        self.assertEqual(init["target_name"], "depth")

    def test_recommend_drift_model_forwards_the_progress_setting(self):
        for value in (True, False):
            self.solver.show_progress = value
            with patch("pysole.solver.DriftAnalyzer") as mock_analyzer:
                mock_analyzer.return_value.run_diagnostics.return_value = []
                self.solver.recommend_drift_model(stage="post_migration", interactive=False)
            self.assertIs(mock_analyzer.call_args.kwargs["show_progress"], value)

    def test_recommend_drift_model_on_real_analyzer(self):
        # Test single-computation caching via get_sample_points
        pts1 = self.solver.get_sample_points("post_migration", "P")
        pts2 = self.solver.get_sample_points("post_migration", "P")
        self.assertIs(pts1, pts2)

        self.solver.opt_variogram_params = {"range": 120.0, "sill": 4.0, "nugget": 0.05}
        results = self.solver.recommend_drift_model(stage="post_migration", interactive=False)
        self.assertEqual(len(results), 24)
        self.assertEqual(results[0].delta_rmse_pct, 0.0)
        best = [r for r in results if not r.is_penalized]
        self.assertEqual(min(r.delta_aicc for r in best), 0.0)


class TestMultiModelComparisonIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        geom, self.dem, self.slope, _, _ = make_asymmetric_fixture()
        x, y, z = make_points(geom, self.dem)
        self.plots_dir = os.path.join(self._tmp.name, "figures")
        self.output_dir = os.path.join(self._tmp.name, "results")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        cfg = {
            "kriging_parameters": {
                "pre_migration": {
                    "method": "universal",
                    "drift_terms": [["z_dem"], []],
                    "calculate_uncertainty": True,
                },
                "post_migration": {
                    "method": "universal",
                    "drift_terms": [["z_dem"], []],
                    "calculate_uncertainty": True,
                },
            },
            "outputs": {
                "plots_dir": self.plots_dir,
                "output_dir": self.output_dir,
                "save_thickness_grid": True,
                "save_thickness_uncertainty": True,
                "save_traveltime_grid": True,
                "save_traveltime_uncertainty": True,
                "save_migrated_points": True,
                "save_basal_shear_stress": True,
            },
        }

        self.solver = Solver(
            dem=self.dem,
            bounds=geom.bounds,
            survey_data_type="one_way_traveltime",
            pre_kriging_method="universal",
            post_kriging_method="universal",
            pre_variogram_model="spherical",
            post_variogram_model="spherical",
            pre_drift_analyzer=True,
            post_drift_analyzer=True,
            output_dir=self.output_dir,
            plots_dir=self.plots_dir,
            show_progress=False,
            config=cfg,
        )
        self.solver.opt_slope = self.slope
        self.solver.opt_kc = 0.03
        pts = np.column_stack((x, y, np.zeros_like(x), z))
        self.solver.survey_points = pts.copy()

    def test_multi_model_pipeline_execution_and_exports(self):
        # 1. Pass 1: migrate_eikonal()
        with patch.object(self.solver, "_export_optional_points_csv", wraps=self.solver._export_optional_points_csv) as mock_pts_export:
            mig_pts = self.solver.migrate_eikonal(interactive=False)
            # Verify migrated points are exported only ONCE for the production model, not per candidate
            self.assertEqual(mock_pts_export.call_count, 1)

        self.assertIsNotNone(mig_pts)
        self.assertEqual(len(self.solver.pre_selected_drift_models), 2)

        # Check Pass 1 comparison files (saved directly in output_dir)
        for ext in ["txt", "csv", "md"]:
            comp_file = os.path.join(self.output_dir, f"drift_model_comparison.{ext}")
            self.assertTrue(os.path.exists(comp_file), f"Missing {comp_file}")

        p1_txt = Path(self.output_dir, "drift_model_comparison.txt").read_text(encoding="utf-8")
        self.assertIn("Mean Unc", p1_txt)
        p1_csv = Path(self.output_dir, "drift_model_comparison.csv").read_text(encoding="utf-8")
        self.assertIn("traveltime_unc_mean", p1_csv)
        p1_md = Path(self.output_dir, "drift_model_comparison.md").read_text(encoding="utf-8")
        self.assertIn("Mean Uncertainty", p1_md)

        # Check Pass 1 rasters with model slug in flat output_dir
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "traveltime_z_dem.tif")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "traveltime_ordinary.tif")))
        # Check migrated points in root results without model slug
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "migrated_points.csv")))

        # 2. Pass 2: calculate_bedrock() & finalize_bedrock()
        self.solver.calculate_bedrock(interactive=False)
        bed_map = self.solver.finalize_bedrock(interactive=False)
        self.assertIsNotNone(bed_map)
        self.assertEqual(len(self.solver.post_selected_drift_models), 2)

        # Check Pass 2 comparison files (unified files saved directly in output_dir)
        for ext in ["txt", "csv", "md"]:
            comp_file = os.path.join(self.output_dir, f"drift_model_comparison.{ext}")
            self.assertTrue(os.path.exists(comp_file), f"Missing {comp_file}")

        p2_txt = Path(self.output_dir, "drift_model_comparison.txt").read_text(encoding="utf-8")
        self.assertIn("Mean Unc", p2_txt)
        p2_csv = Path(self.output_dir, "drift_model_comparison.csv").read_text(encoding="utf-8")
        self.assertIn("thickness_unc_mean_m", p2_csv)
        self.assertIn("traveltime_unc_mean", p2_csv)
        p2_md = Path(self.output_dir, "drift_model_comparison.md").read_text(encoding="utf-8")
        self.assertIn("Mean Unc (m)", p2_md)
        self.assertIn("Mean Uncertainty", p2_md)

        # Check Pass 2 plots in subfolders
        for slug in ["z_dem", "ordinary"]:
            for fig_num, fig_name in [
                ("06", "bedrock_elevation_and_uncertainty"),
                ("07", "final_embedded_bedrock"),
                ("08", "final_thickness_and_uncertainty"),
                ("09", "final_thickness_histogram"),
                ("10", "final_basal_shear_stress_and_uncertainty"),
                ("11", "final_basal_shear_stress_histogram"),
            ]:
                p = os.path.join(self.plots_dir, slug, f"{fig_num}_{fig_name}_{slug}.png")
                self.assertTrue(os.path.exists(p), f"Missing figure: {p}")

        # Check Pass 2 rasters with model slug in flat output_dir
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "thickness_z_dem.tif")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "thickness_ordinary.tif")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "bedrock_z_dem.tif")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "bedrock_ordinary.tif")))
        # Verify no subdirectories in output_dir
        subdirs = [d for d in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, d))]
        self.assertEqual(len(subdirs), 0, f"Expected no subdirectories in output_dir, found: {subdirs}")

    def test_multi_model_suite_with_drift_analyzer_false_executes_non_interactively(self):
        # Create solver with drift_analyzer explicitly set to False, but with multi-model suites
        geom, dem, slope, _, _ = make_asymmetric_fixture()
        x, y, z = make_points(geom, dem)
        plots = os.path.join(self._tmp.name, "figures_da_false")
        outputs = os.path.join(self._tmp.name, "results_da_false")
        os.makedirs(plots, exist_ok=True)
        os.makedirs(outputs, exist_ok=True)

        solver = Solver(
            dem=dem,
            bounds=geom.bounds,
            survey_data_type="one_way_traveltime",
            pre_kriging_method="universal",
            post_kriging_method="universal",
            pre_drift_terms=[["z_dem"], []],
            post_drift_terms=[["z_dem"], []],
            pre_drift_analyzer=False,
            post_drift_analyzer=False,
            output_dir=outputs,
            plots_dir=plots,
            show_progress=False,
        )
        solver.opt_slope = slope
        solver.opt_kc = 0.03
        solver.survey_points = np.column_stack((x, y, np.zeros_like(x), z))

        # Run pipeline with interactive=False
        mig_pts = solver.migrate_eikonal(interactive=False)
        self.assertIsNotNone(mig_pts)
        self.assertEqual(len(solver.pre_selected_drift_models), 2)

        solver.calculate_bedrock(interactive=False)
        bed_map = solver.finalize_bedrock(interactive=False)
        self.assertIsNotNone(bed_map)
        self.assertEqual(len(solver.post_selected_drift_models), 2)

        # Verify rasters with drift model suffix in flat results
        for slug in ["z_dem", "ordinary"]:
            self.assertTrue(
                os.path.exists(os.path.join(outputs, f"bedrock_{slug}.tif")),
                f"Missing bedrock_{slug}.tif in {outputs}",
            )

        # Verify no subdirectories in outputs
        subdirs = [d for d in os.listdir(outputs) if os.path.isdir(os.path.join(outputs, d))]
        self.assertEqual(len(subdirs), 0, f"Expected no subdirectories in outputs, found: {subdirs}")

        # Verify drift model specific subfolder structure in plots
        for slug in ["z_dem", "ordinary"]:
            slug_plots = os.path.join(plots, slug)
            self.assertTrue(os.path.isdir(slug_plots), f"Missing model plot directory {slug_plots}")
            self.assertTrue(
                os.path.exists(os.path.join(slug_plots, f"06_bedrock_elevation_and_uncertainty_{slug}.png")),
                f"Missing figure 06 in {slug_plots}",
            )

        # Unified comparison files exist directly in outputs and contain both passes
        for ext in ["txt", "csv", "md"]:
            comp_file = os.path.join(outputs, f"drift_model_comparison.{ext}")
            self.assertTrue(os.path.exists(comp_file), f"Missing {comp_file}")

        md_text = Path(outputs, "drift_model_comparison.md").read_text(encoding="utf-8")
        self.assertIn("Pass 1: Traveltime Field Interpolation", md_text)
        self.assertIn("Pass 2: Ice Thickness & Bedrock Topography Interpolation", md_text)
        self.assertIn("`ordinary`", md_text)
        self.assertIn("`z_dem`", md_text)

    def test_drift_comparison_pass_log_labels(self):
        """Verifies that [Drift Comparison] logs use explicit Pass 1 and Pass 2 tags."""
        geom, dem, slope, _, _ = make_asymmetric_fixture()
        x, y, z = make_points(geom, dem)
        plots = os.path.join(self._tmp.name, "figures_log_test")
        outputs = os.path.join(self._tmp.name, "results_log_test")
        os.makedirs(plots, exist_ok=True)
        os.makedirs(outputs, exist_ok=True)

        solver = Solver(
            dem=dem,
            bounds=geom.bounds,
            survey_data_type="one_way_traveltime",
            pre_kriging_method="universal",
            pre_drift_analyzer=False,
            pre_drift_terms=[["z_dem"], []],
            post_kriging_method="universal",
            post_drift_analyzer=False,
            post_drift_terms=[["z_dem"], []],
            plots_dir=plots,
            output_dir=outputs,
            show_progress=False,
        )
        solver.opt_slope = slope
        solver.opt_kc = 0.03
        solver.survey_points = np.column_stack((x, y, np.zeros_like(x), z))

        with self.assertLogs("pysole", level="INFO") as log_ctx:
            solver.migrate_eikonal(interactive=False)
            solver.calculate_bedrock(interactive=False)
            solver.finalize_bedrock(interactive=False)

        all_logs = "\n".join(log_ctx.output)
        self.assertIn("[Drift Comparison] Evaluating 2 candidate drift models via block leave-out", all_logs)
        self.assertIn("[Drift Comparison] 2 models queued for Pass 1 comparison:", all_logs)
        self.assertIn("[Drift Comparison] [Pass 1] Evaluating candidate model", all_logs)
        self.assertIn("[Drift Comparison] 2 models queued for Pass 2 comparison:", all_logs)
        self.assertIn("[Drift Comparison] [Pass 2] Evaluating candidate model", all_logs)


if __name__ == "__main__":
    unittest.main()
