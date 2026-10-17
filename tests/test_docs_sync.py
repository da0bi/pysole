"""
README / docs <-> code synchronisation tests (v0.5.0, Phase 6).

The drift documentation is executable: the documented model lists, ``n_drift`` values, candidate counts, rule table
and JSON examples are parsed from the Markdown and checked against ``pysole.drift``, the drift basis and
``load_config``, so documentation and code cannot drift apart silently.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

from pysole import ConfigError
from pysole.config import DEFAULT_CONFIG, load_config
from pysole.drift import DRIFT_PRIMITIVES, build_drift_suite, resolve_drift_plan
from pysole.interpolation import DriftBasis
from tests.test_drift_covariates import make_asymmetric_fixture, make_points

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "drift_analyzer_&_survey_planner.md"
LIVING_DOCS = (README, GUIDE)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def json_blocks(text: str) -> list[dict]:
    return [json.loads(block) for block in re.findall(r"```json\n(.*?)```", text, re.S)]


def leaves(tree: dict, prefix: str = "") -> list[tuple[str, object]]:
    found = []
    for key, value in tree.items():
        if isinstance(value, dict):
            found += leaves(value, f"{prefix}{key}.")
        else:
            found.append((f"{prefix}{key}", value))
    return found


@unittest.skipUnless(README.exists() and GUIDE.exists(), "not running from a source checkout")
class TestDriftModelListsMatchCode(unittest.TestCase):
    def test_guide_lists_every_multi_drift_model_with_the_n_drift_of_the_code(self):
        text = read(GUIDE)
        section = text[text.index("### 1.8 Systematic Multi-Drift Model Suite"):text.index("### 1.9")]
        documented = {}
        for terms, n_drift in re.findall(r"\*\*`(\[[^`]*\])` \(\$n_\{\\text\{drift\}\} = (\d+)\$\)\*\*", section):
            documented[frozenset(json.loads(terms))] = int(n_drift)

        expected = {frozenset(t) for t in build_drift_suite(None) if len(t) >= 2}
        self.assertEqual(len(expected), 18)
        self.assertEqual(set(documented), expected)

        geom, dem, _, _, grids = make_asymmetric_fixture()
        x, y, _ = make_points(geom, dem)
        for terms, n_drift in documented.items():
            ordered = [t for t in DRIFT_PRIMITIVES if t in terms]
            columns = DriftBasis(ordered, x, y, geom, grids).n_columns
            self.assertEqual(n_drift, columns + 1, f"{ordered}: n_drift counts the constant, n_terms does not")

    def test_readme_lists_the_same_multi_drift_models(self):
        text = read(README)
        section = text[text.index("**4 Multi-Raster Physical Combinations"):text.index("For detailed mathematical formulations")]
        documented = {frozenset(json.loads(m)) for m in re.findall(r"^- `(\[[^`]*\])`$", section, re.M)}
        self.assertEqual(documented, {frozenset(t) for t in build_drift_suite(None) if len(t) >= 2})

    def test_guide_documents_the_single_primitives_and_ordinary(self):
        text = read(GUIDE)
        for term in DRIFT_PRIMITIVES:
            self.assertIn(f"`{term}`", text)
        self.assertEqual(len(build_drift_suite(None)), 24)
        self.assertIn("**24 candidates**", text)
        self.assertIn("24 candidates", read(README))


@unittest.skipUnless(README.exists() and GUIDE.exists(), "not running from a source checkout")
class TestSuiteCountsAndRulesMatchCode(unittest.TestCase):
    def test_documented_candidate_counts(self):
        text = read(GUIDE)
        section = text[text.index("#### Suite expansion"):text.index("#### Winner handling")]
        rows = re.findall(r"^\| `(.+?)` \| (\d+|error)[ |(]", section, re.M)
        self.assertGreaterEqual(len(rows), 6)
        for terms_text, count in rows:
            terms = None if terms_text == "null" else json.loads(terms_text)
            with self.subTest(drift_terms=terms_text):
                if count == "error":
                    with self.assertRaises(ConfigError):
                        resolve_drift_plan("universal", True, terms)
                else:
                    self.assertEqual(len(build_drift_suite(terms)), int(count))

    def test_single_candidate_suite_overrides_the_analyzer(self):
        plan = resolve_drift_plan("universal", True, ["sia"])
        self.assertEqual((plan.method, plan.analyze, plan.terms), ("universal", False, ["sia"]))
        self.assertEqual([lvl for lvl, _ in plan.notices], ["info"])

    def test_rule_table_rows(self):
        guide = read(GUIDE)
        for phrase in (
            "`ordinary` overrides `drift_analyzer` and `drift_terms`",
            "ONE multi-drift model",
            "default suite",
            "Random Forest covariates",
            "A top-level `[]` is never accepted",
        ):
            self.assertIn(phrase, guide)

        ordinary = resolve_drift_plan("ordinary", True, ["sia"])
        self.assertEqual((ordinary.method, ordinary.analyze, ordinary.terms), ("ordinary", False, []))
        self.assertEqual([lvl for lvl, _ in ordinary.notices], ["warning"])
        self.assertEqual(resolve_drift_plan("ordinary", False, None).notices, [])

        one_model = resolve_drift_plan("universal", False, ["sia", "z_dem"])
        self.assertEqual((one_model.method, one_model.analyze, one_model.terms), ("universal", False, ["sia", "z_dem"]))
        for terms in (None, []):
            with self.assertRaises(ConfigError):
                resolve_drift_plan("universal", False, terms)

        default_suite = resolve_drift_plan("universal", True, None)
        self.assertTrue(default_suite.analyze)
        self.assertEqual(len(default_suite.suite), 24)

        covariates = resolve_drift_plan("regression", False, ["z_dem", "linear_xy"], engine="pykrige")
        self.assertEqual((covariates.method, covariates.terms), ("regression", ["z_dem", "linear_xy"]))
        skipped = resolve_drift_plan("regression", True, None, engine="pykrige")
        self.assertFalse(skipped.analyze)
        self.assertEqual([lvl for lvl, _ in skipped.notices], ["warning"])

    def test_readme_states_the_same_rules(self):
        text = read(README)
        for phrase in (
            "`ordinary` overrides the analyzer and the terms",
            "ONE multi-drift model",
            "24 candidates",
            "An empty list `[]` at the top level is rejected",
            "only available through PyKrige",
        ):
            self.assertIn(phrase, text)

    def test_ranking_wording_matches_the_sort_key(self):
        """S12: the guide must not claim an RMSE-first ranking; the code sorts by (penalized, AICc, RMSE)."""
        text = read(GUIDE)
        self.assertIn("AICc ascending", text)
        self.assertNotIn("sorted primarily by **$\\text{RMSE}_{\\text{CV}}$**", text)
        source = read(ROOT / "src" / "pysole" / "drift_analyzer.py")
        self.assertIn("results.sort(key=lambda r: (r.is_penalized, r.aicc, r.cv_rmse))", source)


@unittest.skipUnless(README.exists() and GUIDE.exists(), "not running from a source checkout")
class TestJsonExamplesMatchCode(unittest.TestCase):
    def test_readme_configuration_example_equals_the_default_configuration(self):
        example = json_blocks(read(README))[0]
        self.assertEqual(dict(leaves(example)), dict(leaves(DEFAULT_CONFIG)))

    def test_every_kriging_example_loads(self):
        with tempfile.TemporaryDirectory() as out:
            for md in LIVING_DOCS:
                for block in json_blocks(read(md)):
                    if "kriging_parameters" not in block:
                        continue
                    with self.subTest(file=md.name, kriging=block["kriging_parameters"]):
                        load_config({"outputs": {"output_dir": out}, "kriging_parameters": block["kriging_parameters"]})

    def test_readme_parameter_reference_lists_every_configuration_key(self):
        text = read(README)
        start = text.index("## JSON Configuration Parameter Reference")
        reference = text[start:text.index("## Technical & Methodological Notes", start)]
        missing = []
        for path, _ in leaves(DEFAULT_CONFIG):
            parts = path.split(".")
            needle = ".".join(parts[1:]) if parts[0] == "kriging_parameters" and len(parts) == 3 else parts[-1]
            if f"`{needle}`" not in reference:
                missing.append(path)
        self.assertEqual(missing, [])


@unittest.skipUnless(README.exists() and GUIDE.exists(), "not running from a source checkout")
class TestDocumentationIntegrity(unittest.TestCase):
    def test_docs_have_no_absolute_or_dead_links(self):
        """README / CHANGELOG / docs links are relative and resolve to existing files."""
        root = ROOT
        md_files = [root / "README.md", root / "CHANGELOG.md", *sorted((root / "docs").glob("*.md"))]
        dead = []
        for md in md_files:
            text = md.read_text(encoding="utf-8")
            self.assertNotIn("file://", text, md.name)
            for m in re.finditer(r"\]\(([^)\s]+)\)", text):
                url = m.group(1)
                if url.startswith(("http", "mailto:", "#")):
                    continue
                rel = url.split("#")[0]
                if rel and not (md.parent / rel).exists():
                    dead.append(f"{md.name}: {url}")
        self.assertEqual(dead, [])

    def test_documented_drift_term_names_are_valid(self):
        """Drift names written in README / living docs are accepted by the code (no removed legacy aliases)."""
        root = ROOT
        living = [
            root / "README.md",
            *(root / "docs" / n for n in (
                "pysole_interpolation_practice_guide.md",
                "gok_thickness_analysis_report.md",
                "drift_analyzer_&_survey_planner.md",
                "kriging_performance_report.md",
                "release_guide.md",
            )),
        ]
        legacy = re.compile(r"[`\"](sia_thickness|z_surface|regional_linear|quadratic|sia_drift|sia_space|sia_z_dem|sia_curvature_dem|z_dem_curvature_dem|sia_z_dem_curvature_dem|full_physical|full_spatial_physical)[`\"]")
        lists = re.compile(r'\[((?:\s*"[A-Za-z_]+"\s*,?)+)\]')
        problems = []
        for md in living:
            if not md.exists():
                continue
            for no, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
                # ``z_surface`` is a legitimate CSV column of the migrated-points export, not a drift name
                if legacy.search(line.replace("z_surface, depth_migrated", "")):
                    problems.append(f"{md.name}:{no}: legacy drift name")
                if "drift" in line.lower() or "sia" in line:
                    for m in lists.finditer(line):
                        bad = [n for n in re.findall(r'"([A-Za-z_]+)"', m.group(1)) if n not in DRIFT_PRIMITIVES]
                        if bad:
                            problems.append(f"{md.name}:{no}: non-primitive drift name(s) {bad}")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
