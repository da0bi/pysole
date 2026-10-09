# PySole v0.4.3 — Final Audit (Pass 6, all findings closed)

Scope: all of `src/pysole/*.py`, the three `pysole*.json` configs plus the `--init` template, README and `docs/`, CLI, tests and the quick-start notebook. Every item below was reproduced or verified with a command or script.

## 1. Verdict

**All findings (N6-M1, N6-L1…L7) are resolved. No open findings remain. GO for release** of the code in this working copy. What is still to be done is mechanical (section 5).

| Gate | Result |
|---|---|
| Full suite (`unittest discover -s tests`) | **113 tests, OK, 1 skipped** (398 s; test 39 added afterwards and run separately: OK). The skip is the opt-in notebook-execution test, passed separately in the previous pass (`PYSOLE_TEST_NOTEBOOK=1`). 108 earlier tests + 5 new (34–38). |
| Deprecation / Future warnings in the full log | **0** |
| `compileall` with `-W error` (src + tests) | OK |
| CLI smoke (`python -m pysole`) | `--version` → `PySole 0.4.3`; `--init` prints the confirmation; missing config, invalid JSON and `"maybe"` as a boolean each give a one-line `pysole: error: …` and rc 1; with `-v` the traceback is shown; `plan-survey` without `--dem` → rc 2 |
| WUK end-to-end (`pysole_wuk.json --batch`, from a copy) | rc 0, ~18 s, 0 warnings/errors, 11 figures, bedrock raster **byte-identical** to the one produced before this pass |
| Config parity | root `pysole.json` == `DEFAULT_CONFIG`; WUK and GOK configs: 0 missing, 0 unknown keys |
| Link check (README, CHANGELOG, docs) | 0 absolute links, 0 dead links (also enforced by test 38) |
| Versions | `pyproject.toml` = `__version__` = 0.4.3 |

## 2. Findings resolved in this pass

| ID | Resolution | Proof |
|---|---|---|
| N6-M1 | All absolute links to a local checkout replaced by relative links in README, CHANGELOG and `docs/`. A link to the non-existent `docs/drift_analyzer.md` became plain text. | link checker: 0 dead; test 38 |
| N6-L1 | `coerce_bool()` validates every boolean leaf in `load_config()` and `OutputsConfig.from_dict()`. `"false"`/`"no"`/`0` → `False`; `null` → default; garbage raises `ConfigError` (a `ValueError`). | test 34; CLI smoke |
| N6-L2 | Invalid JSON / non-object file → `ConfigError` naming the file. CLI prints `pysole: error: …`, rc 1 (traceback with `-v`). | tests 35, 36; CLI smoke |
| N6-L3 | `pysole --init` prints "Created template configuration file at: …". | test 36; CLI smoke |
| N6-L4 | README lists `--version`/`-V`, `python -m pysole`, the error behaviour and accepted boolean spellings. | README CLI section |
| N6-L5 | Unused imports and locals removed in 9 modules. `OutputsConfig` and `ConfigError` are exported from the package root. An AST guard test keeps imports clean. | test 37 |
| N6-L6 | Parameter docstrings completed for `built_in_kriging_interpolation`, `kriging_interpolation`, `migrate_eikonal_points`, `Solver.__init__`. | source |
| N6-L7 | "Historical document" banner on the six old review reports; `compute_uncertainty` section added to the interpolation practice guide. | docs |

### One more defect found and fixed while closing N6-L7
`docs/pysole_interpolation_practice_guide.md` and `docs/gok_thickness_analysis_report.md` still used drift names that were removed in 0.4.1 (`sia_thickness`, `z_surface`, `regional_linear`, `quadratic`). A user copying them into `drift_terms` gets `ValueError: Unknown drift term …` (verified by running `kriging_interpolation`). An earlier review had listed this, but it was never fixed. Corrected to `sia`, `z_dem`, `linear_xy`, `quadratic_xy`. Old CHANGELOG entries keep the historical names on purpose.

### Behaviour changes users should know about (documented in CHANGELOG `[0.4.3]`)
- A quoted `"false"` for a boolean no longer silently means `true`; unrecognised values now stop the run with a clear error.
- `"compute_uncertainty": null` now means "default (`true`)", not "skip".
- `OutputsConfig` is no longer re-exposed by `pysole.pipeline`; import it from `pysole` or `pysole.config`.

## 3. Open findings

None.

## 4. Corrections to earlier statements
- The previous audit said WUK produces 23 figures. The WUK example produces **11** (identical set before and after this pass); 23 was a counting error.

## 5. What is left before the release is actually published

1. **Transfer to the main repository.** This directory is a review copy. The changes (30 modified files, 3 new: `docs/pysole_v043_final_audit.md`, `src/pysole/__main__.py`, `tests/test_notebook.py`) have to be brought into the main PySole repository; that repo cannot be verified from the review copy.
2. `uv lock --check` / `uv sync` there (the `dev` extra now includes `nbclient`, `nbformat`, `ipykernel`).
3. Build and check: `uv build` (or `python -m build`), then `twine check dist/*`.
4. Commit, tag `v0.4.3`, publish per [release_guide.md](release_guide.md).
5. Optional: run the opt-in notebook test once more after the transfer (`PYSOLE_TEST_NOTEBOOK=1`).
