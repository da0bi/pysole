# PySole v0.5.1: Independent Codebase and Documentation Audit Report

## 0. Pinned state and method

| Item | Value |
|---|---|
| Commit | `ad9408fe9abce517617c527fbb380accd4e5cf28` |
| Tag (`git describe --tags`) | `v0.5.1` (= HEAD = origin/main), 146 tracked files |
| Date of audit | 2026-10-14 |
| Python / NumPy / SciPy / pandas / PyKrige | 3.13.5 / 2.5.1 / 1.18.0 / 3.0.5 / 1.7.3 |
| Machine | Linux, 16 logical / 8 physical cores, 42 GB RAM |
| Reviewer | Antigravity (the same assistant family that wrote much of v0.5.0 and v0.5.1) |

**Self-review disclosure.** I helped write the code under review, so I am not a blind reviewer. Mitigations used: oracles were written from textbook formulas in `/tmp` without copying PySole code; every finding below carries a command or code reference; comments, docstrings and CHANGELOG sentences were treated as claims; two of my own initial suspicions were withdrawn after rechecking (section 10). A later independent review of the same commit should repeat sections 2 and 3 first.

**Method.** Pass 1: independent oracles (`/tmp/audit/o1`, `o6` to `o11`): dense brute-force kriging with `numpy.linalg.solve` (Ordinary and Universal, three variogram models), brute-force per-fold refits against `LeaveBlockOut`, a closed-form geometry oracle for Eikonal migration, a memory scaling measurement, PyKrige comparisons. Pass 2: reading the code. Read-only on the repository; the only repository write is this file. `git status` was clean before and after (tests run from the repository root create `pysole/pysole.log`; I removed it after the early runs, and later test runs were done in a `git archive` copy under `/tmp/audit/tcopy`). The evidence scripts and outputs stay in `/tmp/audit/` (not part of the repository; copy them if you want to keep them).

Legend: V = verified by running, R = verified by reading only, N = not verified.

## 1. Executive summary and verdict

**Verdict: v0.5.1 is in good shape for scientific use of its core numerical path, with no Critical and no High finding.** The kriging engine, the exact leave-block-out cross-validation and the 3D Eikonal migration reproduce independent references to rounding or regularisation level. The weaknesses are at the edges: input and configuration validation, a silent-wrong-answer path in `LeaveBlockOut` for singular per-block drift systems, a memory model that is documented as `3 N^2` but measures about `4.3 N^2`, CI that does not exercise PyKrige, and repository hygiene.

| Area | Grade | Basis |
|---|---|---|
| Kriging engine (`KrigingSystem`) | A | V: matches dense oracle to 1e-13 (regularised) and 1.5e-5 (unregularised) |
| Leave-block-out (`LeaveBlockOut`) | B+ | V: 1.8e-13 vs brute force for well-posed systems; F-01 for singular block systems |
| Drift suite, rules (`drift.py`) | A | R, V: 24 candidates, rule table clean, 66 rule tests pass |
| Drift Analyzer statistics | B | AICc on correlated CV residuals is a heuristic (DW-01); metrics match definitions |
| Eikonal migration | A- | V: 0.00 m error vs closed-form geometry on 5 plane configurations; curved beds not covered |
| Robustness and input validation | C+ | F-03, F-04: most numeric parameters unvalidated; raw numpy errors on NaN |
| Performance and memory | B | V: timings faster than claimed; F-02 memory formula wrong |
| Architecture | B- | `solver.py` 108 KB (was about 81 KB at v0.4.4), 37 broad `except` sites |
| Tests | B | V: 329 tests pass in 153 s; golden test is circular; Eikonal tests do not test the gradient term |
| Documentation | B+ | V: parity of keys and defaults is complete; stale `release_guide.md`; wrong memory formula |
| CI and release | C+ | no PyKrige in CI, one OS, no tag-vs-version check, publish not gated by tests |
| Packaging and hygiene | B- | V: wheel clean and installs; sdist ships half a test suite; tracked stray outputs |

**Safe to rely on today:** native Ordinary and Universal Kriging estimates and variances (up to the documented `1e-6 * sill` regularisation), Drift Analyzer rankings for well-conditioned drift columns, Eikonal migration on smooth geometries, PyKrige parity for the tested models, the config key and default documentation, the wheel from PyPI.

**Not yet safe to rely on:** Drift Analyzer results for candidates whose drift columns degenerate inside a removed block (F-01); typo detection in numeric config values and `variogram_model` (F-03); the documented memory figures for N above about 8000 (F-02); any claim about glaciological correctness of `sia` on the product target `P` or of BSS smoothing (outside this audit, see section 9).

## 2. Findings

Severity scale: `[Critical]` wrong results without warning in normal use, `[High]` wrong results in plausible use or data loss, `[Medium]` wrong results in rare use or a guard that is off, `[Low]` robustness or maintenance, `[Info]` observation. "Recent" marks code written in v0.5.0 or v0.5.1.

### 2.1 Confirmed bugs and behavioural defects

| ID | Sev | Area | Location | Evidence | Impact | Fix (effort) |
|---|---|---|---|---|---|---|
| F-01 | [Medium] Recent | `LeaveBlockOut` | `kriging_core.py` L413-432 (`_prepare`), L506-513 | V `o9`: drift column nonzero only inside block 0: LBO residual -81.06 vs per-fold refit NaN. Two columns identical outside block 0: LBO -2.943 vs refit -0.3445. Well-posed cases agree to 4 digits (-0.4059 / -0.4059, ...). | The conditioning check (`cond(S) > 1e14`) uses the full data only. A block whose retained observations make the drift system singular or near-singular goes through the Woodbury update and returns finite garbage; only non-finite rows reach `_direct_residuals`. The analyzer then ranks the candidate on a wrong residual with no flag. Likely to inflate that candidate's RMSE (fail-safe direction) but not guaranteed. | Check `cond(S + W)` per block (batched `np.linalg.cond` or `slogdet`), route cond > 1e12 to `_direct_residuals` and mark NaN when the retained drift system is singular; add a regression test. S-M |
| F-02 | [Medium] Recent | Memory plan | `memory.py` L348-352, L382-383, L406; guide L153 | V `o5_mem`: `LeaveBlockOut` full-inverse path adds 301 MB at N=3000 and 821 MB at N=5000 (ru_maxrss increments), i.e. 4.1-4.4 N^2 doubles, not 3. The `full_inverse=False` path measures 2.4 N^2 vs 2.5 planned (fine). | Docs claim N=5000: 0.6 GB, N=10 000: 2.4 GB; measured about 0.86 GB and about 3.4 GB. The plan (`3.5 N^2 + 256 MB`) underestimates by about 11 % at N=10 000 and about 17 % at N=20 000. With the default `max_memory_fraction = 0.5` there is headroom; a user setting 0.8 can run out of memory. Probable cause: `lu_solve(lu, np.eye(...))` holds matrix, LU, identity and inverse (4 matrices) transiently. | Raise `_CV_MATRICES_FULL` to 4.5 or build the inverse in place (`scipy.linalg.lapack.dgetri`); correct docstring, guide L153 and README. S |
| F-04 | [Low] Recent | Input validation | `drift_analyzer.py` L417-422; `kriging_core.py` (`KrigingSystem.__init__`); `raster.py` L1070+ | V `o6`: NaN in `z` raises raw `LinAlgError: SVD did not converge` (from VIF) plus LAPACK `DLASCL parameter 4`; NaN in `x` raises raw cKDTree `ValueError`. V `o7`: `load_survey_points` keeps NaN rows, keeps `inf`, accepts negative traveltimes. The production path filters NaN (`interpolation.py` L496, L747, L975) but not `inf`. | Public-API users get cryptic errors; `inf` in a survey file propagates to NaN grids. | Check `np.isfinite` once in `load_survey_points` (drop with a warning or raise `ConfigError`) and in `KrigingSystem.__init__`. S |
| F-05 | [Low] Recent | `KrigingSystem` degenerate inputs | `kriging_core.py` `resolve_variogram_params` (dict path, no guard for `sill = 0`) | V `o1`/`o2`: explicit zero sill, or constant `z` with a dict lacking `sill`: NaN estimates with only a NumPy "Singular matrix" warning. Collinear points plus a linear drift: NaN silently. A NaN coordinate gives finite estimates (the variogram maps NaN distance to the sill). Negative sill and `range <= 0` are accepted. | Silent NaN or silently wrong output when the engine is called directly. The `None` parameter path and the production pipeline guard most of this. | Validate `range > 0`, `sill > 0`, `nugget >= 0`, finite inputs, and raise `ValueError`. S |
| F-06 | [Low] | Unit heuristic | `solver.py` L1484 | R: with `perform_migration=false` the column is multiplied by velocity only if `max() > 15.0`. Carry-over of earlier finding N4-L. | A traveltime survey with all picks below 15 ns would not be converted; implausible in practice but contradicts the explicit `survey_data_type`. | Use `survey_data_type` only. S |

### 2.2 Documentation defects

| ID | Sev | Location | Evidence | Fix |
|---|---|---|---|---|
| D-01 | [Medium] | `memory.py` L382-383, `docs/drift_analyzer_&_survey_planner.md` L153, `README.md` | The `3 N^2` memory formula and the 0.6 GB / 2.4 GB figures are wrong (F-02). | Correct together with F-02. |
| D-02 | [Low] | `docs/release_guide.md` L18-20, L65-75, L91-103 | Hard-coded 0.4.3 examples; the version-file list omits `uv.lock`. | Use `X.Y.Z` placeholders; add `uv.lock`. |
| D-03 | [Low] | `CHANGELOG.md` L494 | References `scripts/plot_package_structure.py`; `scripts/` is gitignored, so the file does not exist in a clean clone. | Remove the path or track the script. |
| D-04 | [Info] | `CHANGELOG.md`, guide L152 | Timings (WUK 3.1/3.3 s, GOK 7.0 s) are machine dependent; V: I measured 1.9 s and 4.5 s for the default 24-candidate suite on this 16-thread machine. | Keep, with the machine named (already stated). |

### 2.3 Design weaknesses

| ID | Sev | Location | Observation | Suggestion |
|---|---|---|---|---|
| F-03 | [Medium] Recent | `config.py` (only `validate_memory_fraction` L68, L587-588 and boolean coercion), `interpolation.py` L985-988, `variogram.py` L451-481 | V `o7`: `load_config` accepts `-1`, `0`, `"foo"`, `1e9` for `velocity`, `g`, `n_cores`, `dx`, `dy`, `kc_min`, `kc_max`, `lambda_min`, `lambda_max`, `n_steps`, `nrbins`, `slope_floor_deg`, and `"foo"` for `variogram_model`. `Solver(post_variogram_model="foo")` also constructs. The model name is validated only inside `kriging_interpolation`, after DEM load, migration and BSS optimisation; in between the Drift Analyzer silently evaluates an unknown name as spherical or exponential (substring match). Drift rules, by contrast, are fail-fast. | Validate numeric ranges and the model name in `load_config` and `Solver.__init__` with `ConfigError`, as done for drift rules. S-M |
| DW-01 | [Medium] | `drift_analyzer.py` L476-527 | AICc is computed from spatially correlated, heteroscedastic leave-block-out residuals, with `k = n_drift + 1` added on top of out-of-sample residuals (double penalty). The ranking is effectively RMSE-driven. VIF is centred and misses near-collinearity with the intercept (the `_prepare` conditioning check covers singular drift). Statistical caveat, not a bug; see question Q2. | State in the guide that AICc is a heuristic ranking score; consider ranking by CV RMSE and reporting AICc as secondary. |
| DW-02 | [Low] | `solver.py` (2340 lines, 108 KB), 37 `except Exception` sites in `src/pysole` (12 in `plotting.py`) | Monolith grew since v0.4.4; broad excepts hide defects outside plotting. | Split by pipeline stage in v0.6; narrow excepts outside plotting. |
| DW-03 | [Low] | `logging.py` L12-26 | A `tqdm` fallback stub exists although `tqdm>=4.65.0` is a hard dependency (`pyproject.toml` L66): dead compatibility shim. | Remove. |
| DW-04 | [Low] | `raster.py` L1200-1210 | Duplicate consolidation rounds coordinates to 0.1 m and loops in Python (V: 0.27 s at N=5000, 2.1 s at N=20 000 with one duplicate). Points closer than about 0.05 m are merged. | Vectorise with `np.add.at`/`pandas.groupby`; document the tolerance. |
| DW-05 | [Info] | `logging.py` L141-189 | `activity_indicator`: V no leaked thread after body exceptions or a raised `KeyboardInterrupt` (simulated, not a real SIGINT), no output and no thread when `disable=True`. The `tqdm` bar is created in the worker thread, so a concurrent log line can interleave (cosmetic). | None. |
| DW-06 | [Info] | WUK example | V: peak RSS 1.8 GB for N about 1100 points; not investigated. | Profile in v0.6. |
| DW-07 | [Info] | `pyproject.toml` L42 | `license = { text = "MIT" }` is the pre-PEP 639 style. | `license = "MIT"` plus `license-files` when setuptools >= 77 is required. |

### 2.4 Test gaps

| ID | Sev | Observation | Fix |
|---|---|---|---|
| T-01 | [Medium] | `tests/test_eikonal_migration_plane.py`: the six "plane over plane" cases call `migrate_points`, which builds a constant traveltime grid (`migration.py` L144-149). The traveltime gradient term (`s1`, `s2`) is never exercised, and cases 5 and 6 ("non-parallel planes") expect the same surface-normal displacement as the parallel cases; the bedrock plane never enters the expected value. V `o10`: with an analytic `T = d_perp / v` field (flat, tilted, parallel, non-parallel, opposite-dip beds) `migrate_eikonal_points` hits the perpendicular foot point with 0.00 m error in x, y and z. | Add the closed-form oracle from `o10` as a test (about 40 lines). |
| T-02 | [Low] | `_block_solve` and `_direct_residuals` are not referenced by any test (grep over `tests/`). | Add a test with an ill-conditioned block (F-01). |
| T-03 | [Low] | Golden test (`tests/data/golden_kriging_engine.npz`) pins the engine to its own earlier output, so it is circular; V `o1` now confirms the engine independently. | Add a dense-solve oracle test (about 30 lines, as in `o1`). |
| T-04 | [Low] | `tests/test_docs_sync.py` covers drift lists, rule table and JSON examples only; not numeric claims, not `release_guide.md`. | Extend to version strings in docs. |
| T-05 | [Low] | No test for config numeric validation (none exists, F-03); notebook test is opt-in (`PYSOLE_TEST_NOTEBOOK=1`) and skipped by default. | With F-03. |
| T-06 | [Low] | `test_audit_regressions.py` and `test_review_v044.py` write `pysole/pysole.log` into the working directory (V per-module run, `o8_writes.txt`); other modules write nothing. Harmless because `/pysole/` is gitignored, but tests should use temporary directories. | Run in `tempfile` cwd. |

## 3. Claim verification matrix

| Claim (source) | Result |
|---|---|
| Analyzer residuals equal an independent per-fold refit to 1e-9 (CHANGELOG 0.5.0) | Confirmed for well-posed systems: max abs diff 1.8e-13 on residual scale 4.7 (V). Wrong for singular per-block drift systems (F-01). |
| Output identical to 0.4.4 within 1e-9, golden-tested (CHANGELOG 0.5.0) | Golden test passes (V); 0.4.4 not available to re-run (N). Independent oracle agrees with the engine (V). |
| Single regularisation `1e-6 * sill` on the point block (CHANGELOG, `kriging_core.py` L33-34) | Confirmed (V): effect: estimates at data points differ from z by about 2e-5, variance at data points clipped to 0. Kriging is not exact; document as an implicit nugget. |
| Native variance equals an independent dense solve (CHANGELOG 0.5.0) | Confirmed (V): 1e-13 vs regularised oracle, at most 1e-5 vs unregularised. |
| PyKrige reproduces native within 1e-4 for Ordinary and all 23 drift models | Tests pass with PyKrige (13 tests, V). Gaussian range conversion `7/sqrt(48)` is correct (`interpolation.py` L664-680, V against PyKrige formula). Not independently re-derived for all 23 models. |
| PyKrige universal is slow: 400 points 2 s, 800 points 19 s (CHANGELOG, README L570) | Confirmed (V): 1.49 s and 22.8 s on this machine; warning exists (`drift.py` L41-46, L300). |
| WUK 3.1 s / 3.3 s, GOK 7.0 s analyzer timings | Confirmed as an upper bound: 1.9 s (N=982) and 4.5 s (N=2033) (V). |
| 0.4.4 needed more than 400 s on GOK | N (version not run). |
| BLAS cap: equal at N=1000, 8 to 17 % faster for N>=2000, 17 to 23 ms at N=500 (CHANGELOG 0.5.1) | Confirmed (V): N=500 17 -> 23 ms, N=1000 equal, N=2000 0.38 -> 0.35 s, N=3000 0.96 -> 0.80 s, N=6000 5.5 -> 4.8 s. |
| Residual difference capped vs uncapped at most 1.1e-12 relative | Not reproduced (N). |
| Progress display costs nothing and is bit-identical (CHANGELOG 0.5.1) | Confirmed (V): N=3000 1.124 vs 1.113 s, N=6000 4.786 vs 4.803 s, results identical. |
| Memory about `3 N^2` doubles; N=5000 0.6 GB; N=10 000 2.4 GB (`memory.py`, guide L153) | Wrong: about 4.3 N^2 (F-02). |
| Grid kriging 1.3-2.8 times faster, WUK/GOK 3.2 / 1.7 times faster (README L463, CHANGELOG 0.5.0) | Not reproduced (N). |
| Default suite is 24 candidates (5 single + 18 multi + Ordinary) | Confirmed (V): combinatorics and `test_docs_sync.py`. |
| `--batch` shows no progress; `show_progress=False` starts no thread | Confirmed for the analyzer spinner (V `o6`); full `--batch` run of WUK from the installed wheel exits 0 in 10.65 s (V). |
| README `nrbins = min(30, max(3, in_range_pairs // 30))` | Confirmed (R, `README.md` L409-415). |
| Same results for same input (no hidden randomness) | Confirmed (V `o6`: identical rankings on repeat; the only RNG is `default_rng(42)` in RF gap filling, `interpolation.py` L1114). |
| `pysole --init`, `--drift-analyzer`, `--profile-col`, `-v`, `--batch`, `-V` exist | Confirmed (V `pysole --help` from the wheel). |
| Versions agree | Confirmed (V): `pyproject.toml`, `__init__.py`, `uv.lock`, `CHANGELOG.md` top = 0.5.1; `release_guide.md` stale (D-02). |

## 4. Status of earlier audit findings

The `v050` report contains suggestions, not IDs; its items are listed by their own labels. Items marked N were not re-verified.

| Earlier ID | Topic | Status at v0.5.1 |
|---|---|---|
| claude H1 | `coords_to_grid_indices` Y-mirrored | Fixed: method deleted, test `test_coords_to_grid_indices_removed` (R, `test_drift_covariates.py` L65-67); bottom-up sampling tests pass (V) |
| claude H2, M7 | Drift normalised per call; analyzer vs production drift definitions | Fixed: `DriftBasis` with grid-level standardisation, shared `KrigingSystem` (R, V) |
| claude H6 | Variogram heuristic, not fitted | Fixed: fitted parameters passed to all passes and the analyzer (R, CHANGELOG 0.5.0) |
| claude H8 | GPX/GeoJSON in projected coordinates | Mitigated: WGS84 export when a CRS is configured, warning otherwise (R, test logs); N |
| claude M8 | AICc comparability | Fixed: all candidates share the evaluated points (R); statistical caveat remains (DW-01) |
| claude M9 | RF importance mislabelled | Removed with `screen_feature_importances` (R) |
| claude H3, H4, H5, H7, M13 | Outline mask, NaN rings, planner, slope clamping, raster round trip | N (not re-verified); README L331 documents NaN-separated rings |
| v043 B6 | `--verbose` overridden by config | Fixed (R: `pipeline.py` L146, L166, L187 pass `log_level`); N not run end-to-end |
| v043 P6 / N4-M13 | Notebook `cv_mode="lopo"` | Fixed: no match in `examples/*.ipynb`, README, current docs (V grep) |
| v043 P8 | Unknown config keys silently ignored | Fixed: warning with "did you mean" (`config.py` L121-171; V `o7` shows the warning) |
| v043 P9 | Undocumented `inputs.output_dir`, `output_bedrock_map` | Fixed: no longer read (V grep) |
| v043 N4-M11 | Stale `_sample_pts_cache` | Fixed: cleared in `solver.py` L1463 (R) |
| v043 N4-L | `np.random.seed`, `max() > 15` | Seed replaced by `default_rng(42)` (fixed); `max() > 15.0` still present (F-06) |
| v043 B4 / N5-M1 | `linear` variogram invalid in 2-D | Mitigated: native engine falls back to spherical with a warning (`variogram.py` L451-481); PyKrige keeps linear |
| v043 B5 / P1 | `nrbins` doc mismatch | Fixed (README L409-415) |
| v043 N5-M3 | Numeric profile ID position | Fixed for CSV with `profile_column` (drops the column by name, `raster.py` L1139-1140) |
| v043 D3 / D4 | Explicit inverse, variance always computed | Inverse is now deliberate (leave-block-out); variance flag N |
| v043 D7 | `optimize_bss_variance` memory linear in `n_steps` | N (not re-measured) |
| v050 M1, M2 | Local neighbourhood kriging; anisotropy | Open (features, not defects) |
| v050 A1 | Split `solver.py` | Open and worse: 108 KB (DW-02) |
| v050 A2 | Deprecate PyKrige | Not done; PyKrige now supports all drift terms |
| v050 A3 | Strict schema contracts | Open (F-03) |
| v050 U1 | Geographic CRS detection | Done (`raster.py` L453 message) |
| v050 U2 | `pysole.schema.json` | Open (no schema file tracked) |
| v050 U3 | DPI/colormap config | Open (no `dpi` key in `DEFAULT_CONFIG`) |
| v050 P2 | FFT spectrum caching | Already present since 0.4.4 |

No regression of an earlier fix was found.

## 5. Schema and documentation parity

V `o4_parity.py` (leaf keys of `DEFAULT_CONFIG` compared with every file):

| Artefact | Result |
|---|---|
| `DEFAULT_CONFIG` (`config.py`) | 61 leaf keys |
| `pysole.json` | Same 61 keys, no extra, no missing |
| `examples/wuk/pysole_wuk.json`, `examples/gok/pysole_gok.json` | Same 61 keys, intentional value differences only |
| README parameter table | Every key documented (rows for `pre_migration.*`, `post_migration.*`, `output_format` L302, `crs` L264, `drift_terms` L283 and L290). Default of `method` is shown as "by target" (by design, not a mismatch) |
| README JSON snippet vs `DEFAULT_CONFIG` | Equal (also enforced by `test_docs_sync.py`) |
| Relative markdown links in tracked `*.md` | 0 broken in the working tree. Caveat: the check ran against a working tree that also contains ignored files; no tracked document links to an ignored file (grep) |
| Version strings | consistent except `docs/release_guide.md` (0.4.3) |
| Numeric validation parity (docs promise vs code) | Only `max_memory_fraction` is range-validated (F-03) |

## 6. Test and CI assessment

- V: `PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests`: 329 tests, OK, 1 skipped (opt-in notebook), about 155 s.
- V: with PyKrige blocked (`sys.modules["pykrige"] = None`): 329 tests, OK, 15 skipped. The 13 PyKrige parity tests (`test_pykrige_drifts.py`) and the notebook test do not run in CI because `ci.yml` installs only `pip install -e .` without the `pykrige` or `dev` extras. The CHANGELOG parity claim is therefore unguarded in CI.
- Per-module run in a clean copy: slowest modules `test_audit_regressions.py` 52 s, `test_solver.py` 28 s, `test_config.py` 13 s, `test_drift_analyzer.py` 13 s, `test_optional_outputs.py` 11 s. No timing- or thread-count-based assertions were found in the files read.
- Strong tests: 66 drift rule tests, 38 analyzer engine tests, 15 leave-block-out tests against per-fold refits, 11 docs-sync tests.
- Weak or circular: see T-01 to T-06.
- `ci.yml`: Python 3.10-3.13 on `ubuntu-latest` only; installs `libgdal-dev` although binary wheels exist; no macOS or Windows; dependency lower bounds in `pyproject.toml` (`numpy>=1.24`, `scipy>=1.10`, `pandas>=2.0`, ...) are never tested because CI resolves the newest versions; `pytest` is installed but unused.
- `publish-pypi.yml` (trigger: tag `v*`): builds with `python -m build`, runs `twine check`, publishes with trusted publishing. No check that the tag equals `pyproject.toml` version, and no `needs:` dependency on the test workflow: a tag pushed on a red commit still publishes.
- Can CI pass while the published wheel is broken? Unlikely for module contents (V: wheel contains `pysole/*.py` and `py.typed`, installs, runs `pysole --help` and the WUK example), but possible for dependency bounds and optional extras (above).

## 7. Hygiene and packaging checklist

| Item | Result |
|---|---|
| Wheel | V: only `pysole/*.py`, `py.typed`, dist-info; `twine check` passes; clean install in a fresh venv works |
| sdist | Builds. Contains 27 `tests/` files but not `tests/data/golden_kriging_engine.npz`, `tests/golden_fixture.py`, `tests/__init__.py`: tests cannot run from the sdist (golden test fails). Fix: `graft tests` in `MANIFEST.in`, or exclude tests |
| `MANIFEST.in` | Lists ignored paths in exclude lines (`scratch`, `logo`, `matlab`, `scripts`, `git_token.txt`, `pypi_request.txt`); none are pulled in; the `git_token.txt` line is odd and should go |
| Clean-install quick start and WUK example | V: `pysole --batch pysole_wuk.json` from the installed wheel: exit 0, 10.65 s, figures and `wuk_final_bedrock.tif` written, `pysole.log` written into the working directory |
| `.gitignore` vs tracked files | Contradictory: ignores `examples/**/figures/`, `examples/**/*.tif`, `*.png` (except `images/*.png`), but tracked are 20+ example figures, three `*_final_bedrock.tif`, and stray output folders `examples/wuk/input_data/figures/` and `examples/wuk/input_data/pysole/figures/` (duplicates of `examples/wuk/figures/`), `examples/wuk/wuk_final_bedrock.tif` plus a duplicate under `input_data/`. New example files are silently ignored. `.gitignore` also contains `/tmp/` and `.antigravityrules` |
| Size | Tracked total about 42 MB; largest `images/pysole_package_structure.png` 7 MB, `examples/gok/gok_final_bedrock.tif` 3.4 MB |
| Examples | `examples/apo/input_data/*` is tracked without any config or README mention |
| Ignored but referenced | `scripts/plot_package_structure.py` (CHANGELOG L494). `CONTRIBUTING.md`, `paper.md`, `paper.bib`, `logo/`, `matlab/` are ignored and therefore not in the public repository (decision for the maintainer, Q4) |
| License, conduct | `LICENSE` and `CODE_OF_CONDUCT.md` tracked; `license = { text = "MIT" }` (DW-07) |
| Versions, tag | consistent (section 5); tag `v0.5.1` = HEAD |
| Tests write into the repository | `pysole/pysole.log` from `test_audit_regressions.py`, `test_review_v044.py` (T-06) |

## 8. Prioritised roadmap

**v0.5.2 (small, no API change):**
1. F-01 per-block conditioning check and fallback, with test (T-02).
2. F-02 and D-01: correct memory constant and all documented figures.
3. F-03: validate numeric config values and `variogram_model` in `load_config` and `Solver.__init__`.
4. F-04 and F-05: finite and positive checks in `load_survey_points` and `KrigingSystem`.
5. T-01 and T-03: add the Eikonal closed-form oracle and the dense-solve kriging oracle as tests.
6. CI: install `.[all,dev]` (or add a PyKrige job) so the 13 parity tests and an optional notebook run execute; add a tag-equals-version check and `needs: test` to the publish workflow.
7. Packaging and hygiene: `graft tests` or drop tests from the sdist; untrack stray example outputs and duplicate `.tif` files; fix or remove the `scripts/` reference; update `docs/release_guide.md` (D-02); make tests use temporary working directories (T-06); remove the `tqdm` stub (DW-03) and the `git_token.txt` manifest line.

**v0.6:** split `solver.py` by pipeline stage; narrow the 37 broad excepts; schema-based config (JSON Schema file and/or dataclass validation); reduce peak memory of the inverse; DW-01 statistics decision (Q2); optional macOS/Windows CI jobs and a lower-bounds job; vectorised duplicate consolidation; features from the v050 roadmap (anisotropy, local neighbourhood kriging) if wanted.

**Leave as is:** the `1e-6 * sill` regularisation (document it as an implicit nugget), the Gaussian-range conversion for PyKrige, seeded RF gap filling, PyKrige as an optional engine.

## 9. What I could not verify and why

- Glaciological correctness: `sia` drift on the product target `P`, BSS optimisation and smoothing conventions, velocity and ns conventions: judgement calls (Q1, Q7).
- Eikonal migration on curved or real beds: the oracle covers planes only; the smoothed-grid gradient on real data has no analytical reference.
- `smoothing.py` (FFT padding, frequency sweep), `plotting.py`, `survey_planner.py` internals, random-forest gap filling, the `regression` kriging method (only through the test suite), interactive prompts: not audited line by line.
- Real `KeyboardInterrupt` (SIGINT) during the spinner: simulated with a raised exception only.
- Runs with N between 10 000 and 20 000: only the memory plan was exercised (with patched available memory) and measured up to N=5000.
- Python 3.10 to 3.12, macOS, Windows, minimum dependency versions: not available locally (CI result "green" was reported by the maintainer).
- Earlier-audit items marked N in section 4; the old 0.4.4 timings and output equality.
- Whether GitHub renders README relative links: only file existence was checked.

**Claims accepted on trust:** all pre-0.5.0 CHANGELOG entries; physical background text in README and guides; contents of `uv.lock`; the PyPI trusted-publisher configuration; that CI was green for v0.5.1.

## 10. Open questions for the maintainer

- Q1. Is a `sia` drift on the product target `P = D sin(alpha)` physically acceptable once it is ranked normally (it only triggers a warning)? The audit treats this as your decision.
- Q2. Should the analyzer rank by CV RMSE with AICc as secondary information, given DW-01?
- Q3. Is the `1e-6 * sill` regularisation meant as an implicit nugget? Today the kriging surface is not exact at data points (about 2e-5 deviation on a residual scale of about 1) and the variance there is 0.
- Q4. Should `CONTRIBUTING.md`, `paper.md`, `paper.bib`, `scripts/` belong in the public repository?
- Q5. Do you accept stricter config validation (typos in numeric values and `variogram_model` become `ConfigError`)? It is technically a behaviour change for configs that silently worked.
- Q6. Should tracked example outputs (figures, `*_final_bedrock.tif`, duplicate folders) stay as showcase material, or move to release assets?
- Q7. Eikonal migration assumes the smoothed interpolated traveltime grid defines the ray direction; accepted for curved beds?

**Withdrawn during the audit (for the second reviewer):** (a) "Gaussian variogram differs between native and PyKrige": true for PyKrige used directly, but PySole rescales the range by `7/sqrt(48)` (`interpolation.py` L664-680) and the engines agree; (b) "`outputs.output_format` missing from the README table": it is documented at README L302.
