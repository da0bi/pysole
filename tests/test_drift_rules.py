"""
Phase 2 of the multi-drift follow-up: drift rules (``pysole.drift``), fail-fast validation, defaults,
method purge and the removal of the ``Solver.drift_analyzer`` master attribute.
"""

import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pysole import ConfigError
from pysole.config import DEFAULT_CONFIG, load_config
from pysole.drift_analyzer import FAILED_AICC
from pysole.drift import (
    DEFAULT_REGRESSION_DRIFT_TERMS,
    DRIFT_PRIMITIVES,
    build_drift_suite,
    default_model,
    format_drift_name,
    normalize_engine,
    normalize_method,
    parse_drift_terms,
    resolve_drift_plan,
)
from pysole.interpolation import (
    KrigingResult,
    built_in_kriging_interpolation,
    kriging_interpolation,
)
from pysole.raster import GridGeometry
from pysole.solver import Solver

SRC = Path(__file__).resolve().parents[1] / "src" / "pysole"
WUK_DEM = Path(__file__).resolve().parents[1] / "examples" / "wuk" / "input_data" / "dgm_unt_wuk.tif"
BOUNDS = (500000.0, 5200000.0, 500120.0, 5200120.0)
SURVEY = np.array([
    [500015.0, 5200015.0, 1003.0, 40.0],
    [500045.0, 5200030.0, 1012.0, 55.0],
    [500075.0, 5200020.0, 1018.0, 48.0],
    [500100.0, 5200055.0, 1030.0, 35.0],
    [500030.0, 5200085.0, 1022.0, 60.0],
    [500085.0, 5200100.0, 1038.0, 42.0],
])


def messages(plan):
    return [text for _, text in plan.notices]


class TestDriftSuite(unittest.TestCase):
    def test_default_suite_is_ordinary_plus_23_models(self):
        suite = build_drift_suite(None)
        self.assertEqual(len(suite), 24)
        self.assertEqual(suite[0], [])
        self.assertEqual(sum(1 for c in suite if not c), 1)
        self.assertEqual(len({tuple(c) for c in suite}), 24)
        for candidate in suite:
            self.assertFalse({"linear_xy", "quadratic_xy"} <= set(candidate), candidate)
            self.assertEqual(candidate, sorted(candidate, key=DRIFT_PRIMITIVES.index))

    def test_suite_sizes_for_listed_primitives(self):
        cases = {
            ("sia", "z_dem", "curvature_dem"): 7,
            ("sia", "z_dem"): 3,
            ("sia",): 1,
            ("sia", "linear_xy"): 3,
            ("linear_xy", "quadratic_xy"): 2,
            ("sia", "linear_xy", "quadratic_xy"): 5,
            ("sia", "z_dem", "curvature_dem", "linear_xy"): 15,
            ("sia", "z_dem", "curvature_dem", "linear_xy", "quadratic_xy"): 23,
        }
        for terms, expected in cases.items():
            with self.subTest(terms=terms):
                self.assertEqual(len(build_drift_suite(list(terms))), expected)

    def test_nested_empty_list_adds_ordinary(self):
        self.assertEqual(build_drift_suite(["sia", []]), [[], ["sia"]])
        self.assertEqual(len(build_drift_suite(["sia", "z_dem", []])), 4)
        self.assertEqual(build_drift_suite([[]]), [[]])

    def test_term_order_does_not_matter(self):
        self.assertEqual(build_drift_suite(["z_dem", "sia"]), build_drift_suite(["sia", "z_dem"]))
        self.assertEqual(parse_drift_terms(["quadratic_xy", "sia", "sia"])[0], ["sia", "quadratic_xy"])

    def test_nested_empty_list_only_in_suites(self):
        with self.assertRaises(ConfigError):
            parse_drift_terms(["sia", []])  # not an analyzer suite
        with self.assertRaises(ConfigError):
            resolve_drift_plan("universal", False, ["sia", []])
        with self.assertRaises(ConfigError):
            resolve_drift_plan("regression", False, ["sia", []])
        with self.assertRaises(ConfigError):
            parse_drift_terms(["sia", [], []], allow_ordinary_candidate=True)  # only once
        with self.assertRaises(ConfigError):
            parse_drift_terms(["sia", ["z_dem"]], allow_ordinary_candidate=True)  # no other nesting

    def test_top_level_empty_list_is_rejected_use_null(self):
        for method, analyzer in (("universal", False), ("universal", True), ("regression", False), ("regression", True)):
            with self.subTest(method=method, analyzer=analyzer):
                with self.assertRaisesRegex(ConfigError, "null"):
                    resolve_drift_plan(method, analyzer, [])

    def test_unknown_and_shorthand_names_rejected(self):
        for bad in ("sia_space", "full_physical", "quadratic", "regional_linear", "bogus", 7):
            with self.subTest(term=bad):
                with self.assertRaisesRegex(ConfigError, "Shorthand compound names are no longer supported"):
                    parse_drift_terms([bad])
        with self.assertRaises(ConfigError):
            parse_drift_terms("sia")  # a bare string is not a list

    def test_drift_name_formatting(self):
        self.assertEqual(format_drift_name([]), "ordinary")
        self.assertEqual(format_drift_name(["quadratic_xy", "sia"]), "sia + quadratic_xy")


class TestNormalizers(unittest.TestCase):
    def test_method_normalization(self):
        self.assertEqual(normalize_method("Universal_Kriging"), "universal")
        self.assertEqual(normalize_method(" ordinary "), "ordinary")
        self.assertEqual(normalize_method("regression_kriging"), "regression")

    def test_method_sia_is_purged_with_pointer(self):
        with self.assertRaisesRegex(ConfigError, "universal.*\\['sia'\\]"):
            normalize_method("sia")
        with self.assertRaises(ConfigError):
            normalize_method("sia_kriging")
        with self.assertRaises(ConfigError):
            normalize_method("kriging")

    def test_engine_normalization(self):
        self.assertEqual(normalize_engine("PyKrige"), "pykrige")
        self.assertEqual(normalize_engine("native_kriging"), "native")
        with self.assertRaises(ConfigError):
            normalize_engine("gstools")

    def test_config_error_is_a_value_error_and_exported(self):
        self.assertTrue(issubclass(ConfigError, ValueError))
        from pysole.config import ConfigError as from_config
        from pysole.errors import ConfigError as from_errors

        self.assertIs(from_config, ConfigError)
        self.assertIs(from_errors, ConfigError)


class TestRuleTable(unittest.TestCase):
    """One test per row of the rule table (see ``pysole.drift``)."""

    def test_ordinary_overrides_analyzer_and_terms_with_one_warning(self):
        plan = resolve_drift_plan("ordinary", True, ["sia", "z_dem"])
        self.assertEqual((plan.method, plan.analyze, plan.terms, plan.suite), ("ordinary", False, [], []))
        self.assertEqual([lvl for lvl, _ in plan.notices], ["warning"])
        self.assertIn("'drift_analyzer' and 'drift_terms'", messages(plan)[0])

    def test_ordinary_warns_for_either_setting_alone(self):
        self.assertIn("'drift_analyzer'", messages(resolve_drift_plan("ordinary", True, None))[0])
        self.assertNotIn("drift_terms'", messages(resolve_drift_plan("ordinary", True, None))[0].split("overrides")[1].split(":")[0])
        only_terms = resolve_drift_plan("ordinary", False, ["z_dem"])
        self.assertEqual(len(only_terms.notices), 1)
        self.assertIn("'drift_terms'", messages(only_terms)[0])

    def test_ordinary_without_overridden_settings_is_silent(self):
        self.assertEqual(resolve_drift_plan("ordinary", False, None).notices, [])

    def test_ordinary_does_not_validate_ignored_terms(self):
        plan = resolve_drift_plan("ordinary", False, ["not_a_term", []])
        self.assertEqual(plan.method, "ordinary")
        self.assertEqual(len(plan.notices), 1)

    def test_universal_without_analyzer_is_one_multi_drift_model(self):
        plan = resolve_drift_plan("universal", False, ["z_dem", "sia", "linear_xy"])
        self.assertEqual((plan.method, plan.analyze, plan.terms, plan.notices), ("universal", False, ["sia", "z_dem", "linear_xy"], []))

    def test_universal_without_analyzer_linear_and_quadratic_falls_back_with_warning(self):
        plan = resolve_drift_plan("universal", False, ["linear_xy", "quadratic_xy"])
        self.assertEqual(plan.method, "universal")
        self.assertFalse(plan.analyze)
        self.assertEqual(plan.terms, ["quadratic_xy"])
        self.assertEqual(len(plan.notices), 1)
        level, msg = plan.notices[0]
        self.assertEqual(level, "warning")
        self.assertEqual(
            msg,
            "drift_terms: ['linear_xy', 'quadratic_xy'] is falling back to ['quadratic_xy'] "
            "which contains the linear terms [x, y].",
        )

        # Order invariance and multi-term combination
        plan_multi = resolve_drift_plan("universal", False, ["quadratic_xy", "sia", "linear_xy"])
        self.assertEqual(plan_multi.terms, ["sia", "quadratic_xy"])
        self.assertEqual(len(plan_multi.notices), 1)
        self.assertEqual(plan_multi.notices[0][0], "warning")
        self.assertIn("falling back to ['quadratic_xy']", plan_multi.notices[0][1])

    def test_universal_without_analyzer_needs_terms(self):
        with self.assertRaisesRegex(ConfigError, "specify drift_terms or use method 'ordinary'"):
            resolve_drift_plan("universal", False, None)
        with self.assertRaisesRegex(ConfigError, "nested"):  # a nested [] only exists inside an analyzer suite
            resolve_drift_plan("universal", False, [[]])

    def test_universal_with_analyzer_default_suite(self):
        plan = resolve_drift_plan("universal", True, None)
        self.assertEqual((plan.method, plan.analyze, len(plan.suite), plan.notices), ("universal", True, 24, []))

    def test_universal_with_analyzer_listed_suite(self):
        plan = resolve_drift_plan("universal", True, ["sia", "z_dem", "curvature_dem"])
        self.assertTrue(plan.analyze)
        self.assertEqual(len(plan.suite), 7)
        with_ordinary = resolve_drift_plan("universal", True, ["sia", "z_dem", []])
        self.assertEqual(len(with_ordinary.suite), 4)
        self.assertEqual(with_ordinary.suite[0], [])

    def test_single_candidate_overrides_analyzer_with_info(self):
        plan = resolve_drift_plan("universal", True, ["z_dem"])
        self.assertEqual((plan.method, plan.analyze, plan.terms), ("universal", False, ["z_dem"]))
        self.assertEqual([lvl for lvl, _ in plan.notices], ["info"])
        self.assertIn("z_dem", messages(plan)[0])

    def test_ordinary_only_suite_resolves_to_ordinary(self):
        plan = resolve_drift_plan("universal", True, [[]])
        self.assertEqual((plan.method, plan.analyze, plan.terms), ("ordinary", False, []))
        self.assertEqual([lvl for lvl, _ in plan.notices], ["info"])

    def test_regression_terms_are_covariates_and_analyzer_is_skipped(self):
        plan = resolve_drift_plan("regression", False, ["z_dem", "linear_xy"], engine="pykrige")
        self.assertEqual((plan.method, plan.terms, plan.notices), ("regression", ["z_dem", "linear_xy"], []))
        # drift_terms: None defaults to all primitives with quadratic_xy providing coordinates
        self.assertEqual(
            resolve_drift_plan("regression", False, None, engine="pykrige").terms,
            list(DEFAULT_REGRESSION_DRIFT_TERMS),
        )
        # linear_xy + quadratic_xy in regression falls back to quadratic_xy with warning
        warned = resolve_drift_plan("regression", False, ["linear_xy", "quadratic_xy"], engine="pykrige")
        self.assertEqual(warned.terms, ["quadratic_xy"])
        self.assertEqual(len(warned.notices), 1)
        self.assertEqual(warned.notices[0][0], "warning")
        self.assertIn("falling back to ['quadratic_xy']", warned.notices[0][1])

        skipped = resolve_drift_plan("regression", True, None, engine="pykrige")
        self.assertFalse(skipped.analyze)
        self.assertEqual([lvl for lvl, _ in skipped.notices], ["warning"])

    def test_regression_on_the_native_engine_says_pykrige_is_used(self):
        plan = resolve_drift_plan("regression", False, ["sia"], engine="native")
        self.assertEqual([lvl for lvl, _ in plan.notices], ["info"])
        self.assertIn("only available through PyKrige", plan.notices[0][1])
        self.assertEqual(resolve_drift_plan("regression", False, ["sia"]).notices, plan.notices)  # native is the default
        both = resolve_drift_plan("regression", True, None, engine="native")
        self.assertEqual([lvl for lvl, _ in both.notices], ["info", "warning"])
        self.assertEqual(resolve_drift_plan("regression", False, ["sia"], engine="pykrige").notices, [])

    def test_analyzer_with_pykrige_ranks_natively_and_applies_the_winner_with_pykrige(self):
        plan = resolve_drift_plan("universal", True, ["sia", "z_dem"], engine="pykrige")
        self.assertTrue(plan.analyze)
        self.assertEqual([lvl for lvl, _ in plan.notices], ["info", "warning"])
        self.assertIn("PyKrige", plan.notices[0][1])
        self.assertEqual(resolve_drift_plan("universal", True, ["sia", "z_dem"], engine="native").notices, [])

    def test_pykrige_universal_warns_about_the_slow_constructor_statistics(self):
        warned = resolve_drift_plan("universal", False, ["sia", "z_dem"], engine="pykrige")
        self.assertEqual([lvl for lvl, _ in warned.notices], ["warning"])
        self.assertIn("very slow", warned.notices[0][1])
        self.assertIn("engine 'native'", warned.notices[0][1])
        single = resolve_drift_plan("universal", True, ["sia"], engine="pykrige")  # a suite of one overrides the analyzer
        self.assertEqual([lvl for lvl, _ in single.notices], ["info", "warning"])
        # no warning where PyKrige's UniversalKriging is not used or not slow
        for method, analyzer, terms, engine in (
            ("universal", False, ["sia", "z_dem"], "native"),
            ("universal", True, [[]], "pykrige"),  # Ordinary-only suite resolves to ordinary
            ("ordinary", False, None, "pykrige"),
            ("regression", False, ["sia"], "pykrige"),
        ):
            with self.subTest(method=method, analyzer=analyzer, terms=terms, engine=engine):
                plan = resolve_drift_plan(method, analyzer, terms, engine=engine)
                self.assertFalse([m for lvl, m in plan.notices if lvl == "warning" and "very slow" in m])

    def test_invalid_engine_is_a_config_error(self):
        for engine in ("pykrig", "", None):
            with self.subTest(engine=engine):
                with self.assertRaises(ConfigError):
                    resolve_drift_plan("universal", False, ["sia"], engine=engine)

    def test_invalid_method_names_are_config_errors(self):
        for method in ("sia", "krige", None):
            with self.subTest(method=method):
                with self.assertRaises(ConfigError):
                    resolve_drift_plan(method, False, ["sia"])


class TestDefaultModel(unittest.TestCase):
    def test_direct_target_without_analyzer_defaults_to_universal_sia(self):
        self.assertEqual(default_model(None, None, "T", "T", False), ("universal", ["sia"]))
        self.assertEqual(default_model(None, None, "d", "D", False), ("universal", ["sia"]))

    def test_direct_target_with_analyzer_defaults_to_default_suite(self):
        self.assertEqual(default_model(None, None, "T", "T", True), ("universal", None))

    def test_direct_target_keeps_given_terms(self):
        self.assertEqual(default_model(None, ["z_dem"], "T", "T", False), ("universal", ["z_dem"]))

    def test_product_target_defaults_to_ordinary(self):
        self.assertEqual(default_model(None, None, "P", "T", False), ("ordinary", None))
        self.assertEqual(default_model(None, None, "D", "T", False), ("ordinary", None))  # D is not the pre-migration target
        # terms / analyzer given without a method fall under the ordinary override (warning in the plan)
        method, terms = default_model(None, ["sia"], "P", "D", True)
        self.assertEqual(method, "ordinary")
        self.assertEqual(len(resolve_drift_plan(method, True, terms).notices), 1)

    def test_explicit_method_is_never_replaced(self):
        self.assertEqual(default_model("universal", None, "T", "T", False), ("universal", None))
        self.assertEqual(default_model("ordinary", ["sia"], "T", "T", False), ("ordinary", ["sia"]))
        with self.assertRaises(ConfigError):  # explicit universal without terms stays an error
            resolve_drift_plan(*[default_model("universal", None, "T", "T", False)[0], False, None])


class TestLoadConfigRules(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = {"output_dir": self._tmp.name}

    def cfg(self, **kriging):
        return load_config({"outputs": self.out, "kriging_parameters": kriging})

    def test_defaults_are_ordinary_with_null_terms(self):
        for stage in ("pre_migration", "post_migration"):
            self.assertIsNone(DEFAULT_CONFIG["kriging_parameters"][stage]["drift_terms"])
        kp = self.cfg()["kriging_parameters"]
        for stage in ("pre_migration", "post_migration"):
            self.assertEqual((kp[stage]["method"], kp[stage]["drift_terms"]), ("ordinary", None))

    def test_direct_target_defaults(self):
        kp = self.cfg(
            pre_migration={"interpolation_target": "T"}, post_migration={"interpolation_target": "D"}
        )["kriging_parameters"]
        for stage in ("pre_migration", "post_migration"):
            self.assertEqual((kp[stage]["method"], kp[stage]["drift_terms"]), ("universal", ["sia"]))

    def test_cli_analyzer_flag_is_applied_before_defaults(self):
        cfg = load_config(
            {"outputs": self.out, "kriging_parameters": {"pre_migration": {"interpolation_target": "T"}}},
            drift_analyzer=True,
        )
        pre, post = cfg["kriging_parameters"]["pre_migration"], cfg["kriging_parameters"]["post_migration"]
        self.assertEqual((pre["method"], pre["drift_terms"], pre["drift_analyzer"]), ("universal", None, True))
        # product target: the flag is set but ordinary overrides it (warning when the Solver is built)
        self.assertEqual((post["method"], post["drift_analyzer"]), ("ordinary", True))

    def test_analyzer_flag_none_or_false_keeps_configured_value(self):
        for flag in (None, False):
            cfg = load_config({"outputs": self.out}, drift_analyzer=flag)
            self.assertFalse(cfg["kriging_parameters"]["pre_migration"]["drift_analyzer"])

    def test_explicit_universal_without_terms_is_rejected(self):
        for stage in ("pre_migration", "post_migration"):
            with self.subTest(stage=stage):
                with self.assertRaisesRegex(ConfigError, f"kriging_parameters.{stage}.*drift_terms"):
                    self.cfg(**{stage: {"method": "universal"}})
                with self.assertRaises(ConfigError):
                    self.cfg(**{stage: {"method": "universal", "drift_terms": None}})
                with self.assertRaisesRegex(ConfigError, "null"):
                    self.cfg(**{stage: {"method": "universal", "drift_terms": []}})

    def test_method_sia_rejected(self):
        with self.assertRaisesRegex(ConfigError, "universal"):
            self.cfg(pre_migration={"method": "sia"})

    def test_invalid_engine_rejected(self):
        with self.assertRaises(ConfigError):
            self.cfg(engine="gstools")

    def test_unknown_term_rejected_but_not_validated_under_ordinary(self):
        with self.assertRaises(ConfigError):
            self.cfg(post_migration={"method": "universal", "drift_terms": ["sia_space"]})
        cfg = self.cfg(post_migration={"method": "ordinary", "drift_terms": ["sia_space"]})
        self.assertEqual(cfg["kriging_parameters"]["post_migration"]["method"], "ordinary")

    def test_valid_suites_pass(self):
        cfg = self.cfg(
            pre_migration={"method": "universal", "drift_analyzer": True, "drift_terms": ["sia", "z_dem", []]},
            post_migration={"method": "universal", "drift_analyzer": True},
        )
        self.assertEqual(cfg["kriging_parameters"]["pre_migration"]["drift_terms"], ["sia", "z_dem", []])


class TestSolverConstructorRules(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        geom = GridGeometry.create((12, 12), dx=10.0, dy=10.0, bounds=BOUNDS)
        xx, yy = geom.meshgrid
        self.dem = 1000.0 + 0.2 * (xx - BOUNDS[0]) + 0.1 * (yy - BOUNDS[1])
        self.out = os.path.join(self._tmp.name, "out")

    def solver(self, **kwargs) -> Solver:
        kwargs.setdefault("output_dir", self.out)
        s = Solver(dem=self.dem, bounds=BOUNDS, survey_data_type="depth", **kwargs)
        s.opt_slope = np.full((12, 12), 0.2)
        s.survey_points = SURVEY.copy()
        s.migrated_points = SURVEY.copy()
        return s

    def test_defaults_follow_the_target(self):
        bare = self.solver()
        self.assertEqual((bare.pre_kriging_method, bare.pre_drift_terms), ("ordinary", None))
        self.assertEqual((bare.post_kriging_method, bare.post_drift_terms), ("ordinary", None))
        direct = self.solver(pre_interpolation_target="T", post_interpolation_target="D")
        for method, terms in ((direct.pre_kriging_method, direct.pre_drift_terms), (direct.post_kriging_method, direct.post_drift_terms)):
            self.assertEqual((method, terms), ("universal", ["sia"]))

    def test_invalid_configuration_fails_before_any_data_is_read(self):
        """The DEM path does not exist: a ConfigError (not FileNotFoundError) proves validation comes first."""
        missing = os.path.join(self._tmp.name, "missing.tif")
        bad = [
            {"pre_kriging_method": "universal"},  # explicit universal without terms / analyzer
            {"post_kriging_method": "universal", "post_drift_terms": []},
            {"pre_kriging_method": "sia"},
            {"post_kriging_method": "universal", "post_drift_terms": ["sia_space"]},
            {"pre_kriging_method": "universal", "pre_drift_terms": ["sia", []]},  # nested [] without analyzer
            {"kriging_engine": "gstools"},
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ConfigError):
                    Solver(dem=missing, **kwargs)

    def test_ordinary_override_logs_one_warning_per_stage_even_for_invalid_terms(self):
        with self.assertLogs("pysole", level="WARNING") as cm:
            solver = self.solver(
                pre_kriging_method="ordinary", pre_drift_analyzer=True, pre_drift_terms=["not_a_term"],
                post_kriging_method="ordinary", post_drift_terms=["sia"],
            )
        overrides = [m for m in cm.output if "overrides" in m]
        self.assertEqual(len(overrides), 2, cm.output)
        self.assertIn("Pass 1 - Constructing Traveltime Grid", overrides[0])
        self.assertIn("Pass 2 - Constructing Depth Grid", overrides[1])
        self.assertFalse(solver._drift_plan("pre_migration").analyze)

    def test_single_candidate_override_logs_info(self):
        with self.assertLogs("pysole", level="INFO") as cm:
            solver = self.solver(
                post_kriging_method="universal", post_drift_analyzer=True, post_drift_terms=["z_dem"]
            )
        self.assertTrue(any("exactly one candidate" in m for m in cm.output), cm.output)
        plan = solver._drift_plan("post_migration")
        self.assertEqual((plan.method, plan.analyze, plan.terms), ("universal", False, ["z_dem"]))

    def test_master_drift_analyzer_attribute_is_gone(self):
        self.assertFalse(hasattr(self.solver(), "drift_analyzer"))

    def test_from_config_resolves_defaults_and_applies_cli_flag(self):
        if not WUK_DEM.exists():
            self.skipTest("WUK example DEM not available")
        config = {
            "inputs": {"dem_path": str(WUK_DEM), "show_progress": False},
            "outputs": {"output_dir": self.out},
            "kriging_parameters": {"pre_migration": {"interpolation_target": "T"}},
        }
        solver = Solver.from_config(config)
        self.assertEqual((solver.pre_kriging_method, solver.pre_drift_terms, solver.pre_drift_analyzer), ("universal", ["sia"], False))
        self.assertEqual((solver.post_kriging_method, solver.post_drift_terms), ("ordinary", None))
        with_flag = Solver.from_config(config, drift_analyzer=True)
        self.assertTrue(with_flag.pre_drift_analyzer and with_flag.post_drift_analyzer)
        self.assertEqual((with_flag.pre_kriging_method, with_flag.pre_drift_terms), ("universal", None))
        self.assertEqual(len(with_flag._drift_plan("pre_migration").suite), 24)
        with self.assertRaises(ConfigError):
            Solver.from_config({**config, "kriging_parameters": {"post_migration": {"method": "universal"}}})


class TestKrigingPassUsesPlan(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        geom = GridGeometry.create((12, 12), dx=10.0, dy=10.0, bounds=BOUNDS)
        xx, yy = geom.meshgrid
        self.dem = 1000.0 + 0.2 * (xx - BOUNDS[0]) + 0.1 * (yy - BOUNDS[1])
        self.fake = KrigingResult(bedrock_grid=np.full((12, 12), 40.0), variance_grid=np.zeros((12, 12)))

    def run_pass(self, winner_terms=None, winner_aicc=0.0, batch=False, **kwargs):
        solver = Solver(
            dem=self.dem, bounds=BOUNDS, survey_data_type="depth",
            output_dir=os.path.join(self._tmp.name, "out"), **kwargs,
        )
        solver.opt_slope = np.full((12, 12), 0.2)
        solver.survey_points = SURVEY.copy()
        solver.migrated_points = SURVEY.copy()
        solver.is_batch_mode = batch
        with patch("pysole.solver.DriftAnalyzer") as analyzer, \
                patch("pysole.solver.sys.stdin", SimpleNamespace(isatty=lambda: True)), \
                patch("pysole.solver.kriging_interpolation", return_value=self.fake) as kriging:
            if winner_terms is not None:
                analyzer.return_value.run_diagnostics.return_value = [
                    SimpleNamespace(terms=winner_terms, aicc=winner_aicc)
                ]
            solver._execute_kriging_pass(
                target_type="D", points=solver.migrated_points, krig_method=solver.post_kriging_method,
                var_model="spherical", zero_boundary=False, pass_name="Pass 2 - Constructing Depth Grid",
            )
        return analyzer, kriging

    def test_default_suite_is_handed_to_the_analyzer(self):
        analyzer, _ = self.run_pass(
            winner_terms=["sia", "z_dem"], post_kriging_method="universal", post_drift_analyzer=True
        )
        candidates = analyzer.return_value.run_diagnostics.call_args.kwargs["candidates"]
        self.assertEqual(len(candidates), 24)

    def test_batch_mode_disables_the_analyzer_prompts_even_in_a_terminal(self):
        for batch in (False, True):
            with self.subTest(batch=batch):
                analyzer, _ = self.run_pass(
                    winner_terms=["z_dem"], batch=batch, post_kriging_method="universal", post_drift_analyzer=True
                )
                self.assertEqual(analyzer.return_value.run_diagnostics.call_args.kwargs["interactive"], not batch)

    def test_listed_suite_is_handed_to_the_analyzer(self):
        analyzer, _ = self.run_pass(
            winner_terms=["z_dem"], post_kriging_method="universal", post_drift_analyzer=True,
            post_drift_terms=["sia", "z_dem", "curvature_dem"],
        )
        self.assertEqual(len(analyzer.return_value.run_diagnostics.call_args.kwargs["candidates"]), 7)

    def test_winner_decides_the_final_method(self):
        _, kriging = self.run_pass(
            winner_terms=["sia", "z_dem"], post_kriging_method="universal", post_drift_analyzer=True
        )
        self.assertEqual(kriging.call_args.kwargs["method"], "universal")
        self.assertEqual(kriging.call_args.kwargs["drift_terms"], ["sia", "z_dem"])
        _, kriging = self.run_pass(winner_terms=[], post_kriging_method="universal", post_drift_analyzer=True)
        self.assertEqual(kriging.call_args.kwargs["method"], "ordinary")
        self.assertFalse(kriging.call_args.kwargs["drift_terms"])
        self.assertIsNone(kriging.call_args.kwargs["external_drift_grid"])

    def test_failed_winner_falls_back_to_ordinary(self):
        with self.assertLogs("pysole", level="WARNING") as cm:
            _, kriging = self.run_pass(
                winner_terms=["sia", "z_dem"], winner_aicc=FAILED_AICC,
                post_kriging_method="universal", post_drift_analyzer=True,
            )
        self.assertTrue(any("could not cross-validate any candidate" in m for m in cm.output))
        self.assertEqual(kriging.call_args.kwargs["method"], "ordinary")
        self.assertFalse(kriging.call_args.kwargs["drift_terms"])

    def test_ordinary_never_runs_the_analyzer(self):
        analyzer, kriging = self.run_pass(
            post_kriging_method="ordinary", post_drift_analyzer=True, post_drift_terms=["sia"]
        )
        analyzer.assert_not_called()
        self.assertEqual(kriging.call_args.kwargs["method"], "ordinary")
        self.assertIsNone(kriging.call_args.kwargs["external_drift_grid"])

    def test_single_model_is_applied_directly(self):
        analyzer, kriging = self.run_pass(post_kriging_method="universal", post_drift_terms=["sia", "z_dem"])
        analyzer.assert_not_called()
        self.assertEqual(kriging.call_args.kwargs["drift_terms"], ["sia", "z_dem"])
        self.assertEqual(set(kriging.call_args.kwargs["external_drift_grid"]), {"sia", "z_dem"})

    def test_solver_logs_warning_for_linear_and_quadratic_terms(self):
        with self.assertLogs("pysole", level="WARNING") as cm:
            analyzer, kriging = self.run_pass(
                post_kriging_method="universal",
                post_drift_terms=["linear_xy", "quadratic_xy"],
            )
        analyzer.assert_not_called()
        self.assertEqual(kriging.call_args.kwargs["drift_terms"], ["quadratic_xy"])
        self.assertTrue(any("drift_terms: ['linear_xy', 'quadratic_xy'] is falling back to ['quadratic_xy']" in m for m in cm.output))

    def test_calculate_bedrock_method_override_is_validated(self):
        solver = Solver(dem=self.dem, bounds=BOUNDS, survey_data_type="depth",
                        output_dir=os.path.join(self._tmp.name, "out"))
        solver.opt_slope = np.full((12, 12), 0.2)
        solver.migrated_points = SURVEY.copy()
        with self.assertRaises(ConfigError):
            solver.calculate_bedrock(method="sia")
        with self.assertRaises(ConfigError):  # explicit universal override without drift terms
            solver.calculate_bedrock(method="universal")


class TestLowLevelEngineRules(unittest.TestCase):
    def setUp(self):
        self.geom = GridGeometry.create((12, 12), dx=10.0, dy=10.0, bounds=BOUNDS)
        self.dem = 1000.0 + 0.2 * (self.geom.meshgrid[0] - BOUNDS[0]) + 0.1 * (self.geom.meshgrid[1] - BOUNDS[1])

    def krige(self, **kwargs):
        return kriging_interpolation(
            sample_points=SURVEY[:, [0, 1, 3]], geometry=self.geom, dem_grid=self.dem,
            variogram_params={"range": 80.0, "sill": 100.0, "nugget": 1.0}, show_progress=False, **kwargs,
        )

    def test_method_sia_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "universal"):
            self.krige(method="sia")
        with self.assertRaises(ValueError):
            built_in_kriging_interpolation(
                SURVEY[:, [0, 1, 3]], self.geom.x_coords, self.geom.y_coords, method="sia", show_progress=False
            )

    def test_universal_needs_a_drift(self):
        for terms in (None, []):
            with self.subTest(terms=terms):
                with self.assertRaises(ValueError):
                    self.krige(method="universal", drift_terms=terms)
        with self.assertRaisesRegex(ValueError, "no drift column"):
            built_in_kriging_interpolation(
                SURVEY[:, [0, 1, 3]], self.geom.x_coords, self.geom.y_coords, method="universal", show_progress=False
            )

    def test_flat_raster_alone_is_not_a_silent_quadratic_fallback(self):
        with self.assertRaisesRegex(ValueError, "no drift column"):
            built_in_kriging_interpolation(
                SURVEY[:, [0, 1, 3]], self.geom.x_coords, self.geom.y_coords, method="universal", show_progress=False,
                drift_terms=["z_dem"], external_drift_grid={"z_dem": np.full((12, 12), 5.0)},
            )

    def test_raster_term_without_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "z_dem"):
            built_in_kriging_interpolation(
                SURVEY[:, [0, 1, 3]], self.geom.x_coords, self.geom.y_coords, method="universal", show_progress=False,
                drift_terms=["z_dem", "linear_xy"],
            )

    def test_ordinary_ignores_drifts_with_a_warning(self):
        plain = self.krige(method="ordinary")
        with self.assertLogs("pysole", level="WARNING") as cm:
            ignored = self.krige(method="ordinary", drift_terms=["sia", "linear_xy"])
        self.assertTrue(any("ordinary" in m and "ignores" in m for m in cm.output), cm.output)
        np.testing.assert_array_equal(plain.bedrock_grid, ignored.bedrock_grid)

    def test_universal_sia_term_equals_former_sia_semantics(self):
        """Universal + ['sia'] builds the SIA grid itself (no method 'sia' needed)."""
        slope = np.full((12, 12), 0.2) + 0.01 * np.arange(12)[None, :]
        res = self.krige(method="universal", drift_terms=["sia"], opt_slope_grid=slope)
        self.assertTrue(np.all(np.isfinite(res.bedrock_grid)))

    def test_universal_linear_and_quadratic_falls_back_with_warning(self):
        with self.assertLogs("pysole", level="WARNING") as cm:
            res = self.krige(method="universal", drift_terms=["linear_xy", "quadratic_xy"])
        self.assertTrue(any("drift_terms: ['linear_xy', 'quadratic_xy'] is falling back to ['quadratic_xy']" in m for m in cm.output))
        self.assertEqual(res.bedrock_grid.shape, self.geom.shape)


class TestNoLegacyNamesInSource(unittest.TestCase):
    def source(self):
        return {p.name: p.read_text(encoding="utf-8") for p in SRC.glob("*.py")}

    def test_removed_symbols_are_gone(self):
        for name, text in self.source().items():
            for symbol in ("CANDIDATE_DRIFT_MODELS", "CANONICAL_DRIFT_ORDER", "SUPPORTED_PRIMITIVE_DRIFTS"):
                self.assertNotIn(symbol, text, f"{name} still mentions {symbol}")

    def test_no_legacy_shorthand_or_sia_method_names(self):
        legacy = re.compile(
            r"[`\"'](sia_thickness|regional_linear|sia_drift|sia_space|sia_z_dem|sia_curvature_dem|"
            r"z_dem_curvature_dem|sia_z_dem_curvature_dem|full_physical|full_spatial_physical)[`\"']"
        )
        sia_method = re.compile(
            r"method\s*=\s*[\"']sia[\"']|method_clean\s*==\s*[\"']sia[\"']|valid_methods\s*=.*[\"']sia[\"']"
            r"|\[\"universal\",\s*\"universal_kriging\",\s*\"sia\"\]|\{\"universal\",\s*\"sia\"|\"universal\",\s*\"sia\","
        )
        for name, text in self.source().items():
            # pykrige's own drift name 'regional_linear' is a PyKrige API string, not a PySole drift term
            cleaned = text.replace('"regional_linear"', "")
            self.assertIsNone(legacy.search(cleaned), f"{name}: legacy drift shorthand")
            self.assertIsNone(sia_method.search(text), f"{name}: method 'sia'")

    def test_solver_has_no_master_drift_analyzer_flag(self):
        text = self.source()["solver.py"]
        self.assertNotIn("self.drift_analyzer", text)
        self.assertNotIn('getattr(self, "drift_analyzer"', text)

    def test_drift_rules_live_only_in_drift_module(self):
        text = self.source()
        self.assertIn("def build_drift_suite", text["drift.py"])
        for name, content in text.items():
            if name != "drift.py":
                self.assertNotIn("def build_drift_suite", content)
                self.assertNotIn("def format_drift_name", content)


class TestSingleModelAutoDisableAndNonFlatSuites(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        geom = GridGeometry.create((12, 12), dx=10.0, dy=10.0, bounds=BOUNDS)
        xx, yy = geom.meshgrid
        self.dem = 1000.0 + 0.2 * (xx - BOUNDS[0]) + 0.1 * (yy - BOUNDS[1])
        self.out = os.path.join(self._tmp.name, "out")
        os.makedirs(self.out, exist_ok=True)

    def test_non_flat_suite_with_analyzer_false_resolves_non_interactive_plan(self):
        plan = resolve_drift_plan("universal", False, [["sia", "z_dem"], ["sia"], []])
        self.assertEqual(plan.method, "universal")
        self.assertTrue(plan.analyze)
        self.assertFalse(plan.interactive)
        self.assertEqual(plan.suite, [["sia", "z_dem"], ["sia"], []])
        self.assertTrue(any("drift_analyzer: false" in msg for _, msg in plan.notices))

    def test_duplicate_non_flat_suite_with_analyzer_false_unwraps(self):
        plan = resolve_drift_plan("universal", False, [["sia"], ["sia"]])
        self.assertEqual(plan.method, "universal")
        self.assertFalse(plan.analyze)
        self.assertEqual(plan.terms, ["sia"])

    def test_single_non_flat_candidate_with_analyzer_false_unwraps(self):
        plan = resolve_drift_plan("universal", False, [["sia", "z_dem"]])
        self.assertEqual(plan.method, "universal")
        self.assertEqual(plan.terms, ["sia", "z_dem"])
        self.assertFalse(plan.analyze)

    def test_load_config_auto_disables_analyzer_for_single_model(self):
        # Flat single primitive
        cfg1 = load_config({
            "outputs": {"output_dir": self.out},
            "kriging_parameters": {
                "post_migration": {
                    "method": "universal",
                    "drift_analyzer": True,
                    "drift_terms": ["sia"],
                }
            }
        })
        self.assertFalse(cfg1["kriging_parameters"]["post_migration"]["drift_analyzer"])

        # Non-flat single candidate
        cfg2 = load_config({
            "outputs": {"output_dir": self.out},
            "kriging_parameters": {
                "post_migration": {
                    "method": "universal",
                    "drift_analyzer": True,
                    "drift_terms": [["sia", "z_dem"]],
                }
            }
        })
        self.assertFalse(cfg2["kriging_parameters"]["post_migration"]["drift_analyzer"])

        # Multi-model suite keeps drift_analyzer: True
        cfg3 = load_config({
            "outputs": {"output_dir": self.out},
            "kriging_parameters": {
                "post_migration": {
                    "method": "universal",
                    "drift_analyzer": True,
                    "drift_terms": [["sia", "z_dem"], ["sia"], []],
                }
            }
        })
        self.assertTrue(cfg3["kriging_parameters"]["post_migration"]["drift_analyzer"])

    def test_solver_auto_disables_analyzer_attribute_for_single_model(self):
        s1 = Solver(
            dem=self.dem,
            bounds=BOUNDS,
            survey_data_type="depth",
            post_kriging_method="universal",
            post_drift_analyzer=True,
            post_drift_terms=["sia"],
            output_dir=self.out,
        )
        self.assertFalse(s1.post_drift_analyzer)

        s2 = Solver(
            dem=self.dem,
            bounds=BOUNDS,
            survey_data_type="depth",
            post_kriging_method="universal",
            post_drift_analyzer=True,
            post_drift_terms=[["sia", "z_dem"]],
            output_dir=self.out,
        )
        self.assertFalse(s2.post_drift_analyzer)


class TestMassBalanceAndVelocityDriftModels(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.geom = GridGeometry.create((12, 12), dx=10.0, dy=10.0, bounds=BOUNDS)
        xx, yy = self.geom.meshgrid
        self.dem = 1000.0 + 0.2 * (xx - BOUNDS[0]) + 0.1 * (yy - BOUNDS[1])
        self.mb = -2.0 + 0.01 * (xx - BOUNDS[0]) - 0.005 * (yy - BOUNDS[1])
        # High-frequency spike + off-glacier positive value in velocity
        self.vel = 15.0 + 0.05 * (xx - BOUNDS[0])
        self.vel[6, 6] += 80.0
        self.mask = np.ones((12, 12), dtype=bool)
        self.mask[0, :] = False  # row 0 is off-glacier
        self.out = os.path.join(self._tmp.name, "out")
        os.makedirs(self.out, exist_ok=True)

    def test_suite_scaling_with_available_optional_terms(self):
        from pysole.drift import OPTIONAL_RASTER_DRIFT_TERMS, resolve_available_optional_terms

        self.assertEqual(len(build_drift_suite(None)), 24)
        mb_only = resolve_available_optional_terms(has_mass_balance=True, has_velocity=False)
        self.assertEqual(mb_only, ["mass_balance"])
        self.assertEqual(len(build_drift_suite(None, available_optional_terms=mb_only)), 48)

        vel_only = resolve_available_optional_terms(has_mass_balance=False, has_velocity=True)
        self.assertEqual(vel_only, ["velocity", "velocity_sia"])
        self.assertEqual(len(build_drift_suite(None, available_optional_terms=vel_only)), 60)

        both = resolve_available_optional_terms(has_mass_balance=True, has_velocity=True)
        self.assertEqual(both, list(OPTIONAL_RASTER_DRIFT_TERMS))
        suite_120 = build_drift_suite(None, available_optional_terms=both)
        self.assertEqual(len(suite_120), 120)
        for cand in suite_120:
            s = set(cand)
            self.assertFalse({"linear_xy", "quadratic_xy"} <= s)
            if "velocity_sia" in s:
                self.assertNotIn("sia", s)
                self.assertNotIn("velocity", s)

    def test_velocity_sia_redundancy_pruning_with_warning(self):
        plan1 = resolve_drift_plan("universal", False, ["sia", "velocity_sia"])
        self.assertEqual(plan1.terms, ["velocity_sia"])
        self.assertEqual(len(plan1.notices), 1)
        self.assertEqual(plan1.notices[0][0], "warning")
        self.assertIn("already couples 'velocity' and 'sia'", plan1.notices[0][1])

        plan2 = resolve_drift_plan("universal", False, ["velocity", "velocity_sia", "z_dem"])
        self.assertEqual(plan2.terms, ["z_dem", "velocity_sia"])
        self.assertEqual(len(plan2.notices), 1)
        self.assertIn("already couples 'velocity' and 'sia'", plan2.notices[0][1])

    def test_fail_fast_when_optional_raster_missing(self):
        with self.assertRaisesRegex(ConfigError, "mass_balance_path"):
            load_config({
                "outputs": {"output_dir": self.out},
                "kriging_parameters": {
                    "post_migration": {
                        "method": "universal",
                        "drift_analyzer": False,
                        "drift_terms": ["mass_balance"],
                    }
                }
            })
        with self.assertRaisesRegex(ConfigError, "velocity_path"):
            load_config({
                "outputs": {"output_dir": self.out},
                "kriging_parameters": {
                    "post_migration": {
                        "method": "universal",
                        "drift_analyzer": False,
                        "drift_terms": ["velocity_sia"],
                    }
                }
            })
        with self.assertRaisesRegex(ConfigError, "mass_balance_path"):
            Solver(
                dem=self.dem,
                bounds=BOUNDS,
                survey_data_type="depth",
                post_kriging_method="universal",
                post_drift_analyzer=False,
                post_drift_terms=["sia", "mass_balance"],
                output_dir=self.out,
            )

    def test_solver_keeps_mass_balance_unsmoothed_and_smooths_velocity_with_zero_clamping(self):
        solver = Solver(
            dem=self.dem,
            bounds=BOUNDS,
            outline=self.mask,
            mass_balance=self.mb,
            velocity=self.vel,
            survey_data_type="depth",
            post_kriging_method="universal",
            post_drift_analyzer=False,
            post_drift_terms=["mass_balance", "velocity_sia"],
            output_dir=self.out,
        )
        solver.opt_kc = 0.05
        grids = solver.get_drift_grids(["mass_balance", "velocity", "velocity_sia"])
        # mass_balance is exact / unsmoothed
        np.testing.assert_allclose(grids["mass_balance"], self.mb)
        # velocity is smoothed (peak at [6,6] attenuated) and non-negative
        self.assertTrue(np.all(grids["velocity"] >= 0.0))
        self.assertLess(grids["velocity"][6, 6], self.vel[6, 6])
        self.assertTrue(np.all(np.isfinite(grids["velocity_sia"])))
        self.assertTrue(np.all(grids["velocity_sia"] >= 0.0))


if __name__ == "__main__":
    unittest.main()

