# PySole v0.4.3 — Fourth-Pass Review (Tasks A–D verification & release readiness)

> **Historical document.** This report was written against an earlier development state of PySole and is kept for traceability only. Findings listed here were addressed in later releases (see [CHANGELOG](../CHANGELOG.md) and [pysole_v043_final_audit.md](pysole_v043_final_audit.md)); line numbers and parameter names may no longer match the current code.

Scope: `src/pysole/`, `tests/`, `README.md`, `pysole.json`, `examples/`, `docs/`, `CHANGELOG.md`.
Mode: strictly read-only. All scratch material is in `/tmp/pysole_review2/` (`v4_checks.py`, `v4_e2e.py`, `v4_taskb.py`, `v4_parity.py`).
Evidence tags: **[V]** reproduced numerically or end-to-end by me, **[C]** confirmed by reading code, **[U]** could not be verified here.

> [!IMPORTANT]
> **I could not run the repository test suite.** `shapely`, `pandas`, `scikit-learn`, `rasterio` and `pykrige` are not installed in this sandbox, there is no network, and the repo has no `.venv/`. Any "N tests pass without warnings" statement (the brief says 54+; I count **87** `def test_` functions) is therefore **[U]**. I ran the pure-NumPy/SciPy parts of the real modules through a stub harness instead.

---

## 1. Executive Summary & Release Verdict

**Verdict: NO-GO for tagging `v0.4.3` as it stands. It is a narrow no-go: a short, mostly documentation-level fix list (R1–R7, §6) turns it into GO.**

Progress since the previous pass is real:

- The three functional blockers from pass 3 (B1 bedrock export, B2 output-dir plumbing, B3 engine string) are fixed and verified end-to-end **[V]**.
- Profile IDs now survive loading, 3D migration and `get_sample_points()` **[V]**.
- Schema parity is now clean: `pysole.json`, `DEFAULT_CONFIG`, the README JSON snippet and both example configs agree key-for-key **[V]**.
- The numerical core was already sound and did not regress.

It is still a no-go for four reasons:

1. **Several Task claims are false or overstated** (CHANGELOG and brief). A release note that says "verified" about something that does not work is a release blocker for me:
   - CLI `--verbose`/`--debug` precedence does **not** work **[V]**.
   - "Memory streaming in BSS sweeps" is **not** implemented; memory is still linear in `n_steps` **[V]**.
   - "Conditional variance evaluation" is **not** implemented **[C]**.
   - The quickstart notebook still raises `TypeError` **[C]**.
2. **The new `"linear"` variogram model is not a valid 2-D variogram** (44 negative covariance eigenvalues in a 300-point test; 2.2 % of kriging cells get variance clamped to 0) **[V]**. It is advertised in the README for both passes.
3. **The README still documents a different `nrbins` formula than the code uses**, and the optimiser's log message still reports the old one **[C]**.
4. **Repo hygiene.** A stray, un-ignored `pysole/final_bedrock.tif` sits in the repo root, and the test suite writes outputs outside temp dirs (N5-M5).

I found no new numerical defect that would produce wrong science in the default workflow.

---

## 2. Tasks A–D & Audit Resolution Matrix

Legend: ✅ fixed & verified · 🟡 partially / claim overstated · ❌ not done · ❔ unverifiable here.

### 2.1 Tasks A–D (from the brief and `CHANGELOG.md`)

| Item | Claim | Status | Evidence |
|---|---|---|---|
| **A1** Default bedrock export `<prefix>_bedrock.<ext>` via `save_bedrock_elevation_map` | `run_from_config` writes it | ✅ **[V]** | e2e: `res_bedrock.npy` created in the configured `output_dir` and in the default `<survey_dir>/pysole/`; `save_bedrock_elevation_map=false` → no file. Written by `finalize_bedrock` → `export_outputs("finalization")` ([solver.py L1311](../src/pysole/solver.py#L1311), [L404](../src/pysole/solver.py#L404)). Your own `examples/wuk/wuk_final_bedrock.tif` has a 21:26 mtime. |
| **A2** `resolve_output_dir()` keyword args | no stray log in cwd | ✅ **[V]** | [pipeline.py L87-91](../src/pysole/pipeline.py#L87-L91); run with relative `output_dir` left cwd empty; log in `<cfg>/out3/pysole.log`. |
| **A3** Engine string propagation | `"pykrige"` now reachable | ✅ code / 🟡 test | [solver.py L734](../src/pysole/solver.py#L734) passes `self.engine_type`. The regression test ([test_audit_regressions.py L316-324](../tests/test_audit_regressions.py#L316-L324)) only asserts `solver.engine_type == "native"`. It never reaches `kriging_interpolation`, so it is vacuous. CHANGELOG names `interpolate_kriging()` but the change is in `_execute_kriging_pass` (that method is only an alias). |
| **B1** `pd.factorize` string/numeric profile IDs | | ✅ for string IDs; 🟡 numeric | [raster.py L836-895](../src/pysole/raster.py#L836-L895). String IDs work. A *numeric* profile column is only honoured if it is the **5th** numeric column (see N5-M3). |
| **B2** Profile ID through Eikonal migration | | ✅ **[V]** | 5-col in → 5-col migrated, IDs identical row-wise ([migration.py L222-225](../src/pysole/migration.py#L222-L225)). |
| **B3** Profile ID in `get_sample_points()` | | ✅ **[V]** | Targets D, P, T all return `[X,Y,val,prof]`. But `perform_migration=False` drops the column ([solver.py L817](../src/pysole/solver.py#L817), **[V]**), and `Solver.recommend_drift_model` never passes `profile_data` ([L1346-1362](../src/pysole/solver.py#L1346-L1362)), so LOPO only engages through the `drift_analyzer` flag inside `_execute_kriging_pass` ([L697-710](../src/pysole/solver.py#L697-L710)). |
| **B4** `"linear"` variogram | accepted | 🟡 accepted but mathematically invalid in 2-D | N5-M1. The test passes `{"slope": 0.1}`; that key is ignored (changing it to 99 changes the output by 0.0 **[V]**), so the test proves nothing about the model. |
| **B5** `nrbins = min(30, max(3, in_range//30))` | | 🟡 | Correct in `calculate_variogram` (both branches) and its docstring. Still wrong in the README ([L384-390](../README.md#L384-L390)) and in the optimiser's log ([variogram.py L471](../src/pysole/variogram.py#L471) has no cap and uses all pairs, not in-range pairs). The interactive prompt prints `nrbins = None` ([L600](../src/pysole/variogram.py#L600)). |
| **B6** CLI `--verbose/--debug` precedence over config | "Verified" | ❌ **[V]** | `run_from_config` calls `Solver.from_config(config_file)` without `log_level` ([pipeline.py L101, L121](../src/pysole/pipeline.py#L121)). `load_config` then resets logging to the file value ([config.py L295-296](../src/pysole/config.py#L295-L296)). Measured: DEBUG → INFO after `load_config`; after a full `run_from_config(log_level="DEBUG")` the logger is at INFO. |
| **C1** JSON templates + README table | `save_bedrock_elevation_map`, target `"D"` | ✅ **[V]** | Automated diff: `pysole.json`, the README snippet and `DEFAULT_CONFIG` are identical (59 keys, types included). Examples differ only by intentional values. README target now reads `"D"` ([L263](../README.md#L263)). |
| **D1** NumPy 2.5 `src.read(1)` fix | | ❔ | [raster.py L481](../src/pysole/raster.py#L481) now uses `np.array(..., dtype=float64)`. That is harmless but always copies (transient 2× memory). If the deprecation is raised inside rasterio's `read`, a wrapper cannot remove it. Not verifiable without rasterio. |
| **D2** Tikhonov block scoping `K[:N,:N]` | | ✅ scope / 🟡 sign | Scoped correctly ([interpolation.py L366-367](../src/pysole/interpolation.py#L366-L367)); the Lagrange block stays exact. The sign issue (N4-M9) is unchanged (see N5-L5). |
| **D3** Direct LU solve | | 🟡 | `lu_factor/lu_solve` for the weights, but an explicit `K_inv = lu_solve(lu, I)` is still formed ([L374-376](../src/pysole/interpolation.py#L374-L376)): O(n³) extra and O((N+d)²) memory. |
| **D4** Conditional variance evaluation | | ❌ **[C]** | Variance is computed unconditionally in every chunk ([L431-440](../src/pysole/interpolation.py#L431-L440)): an extra `K_inv @ RHS` GEMM per chunk, whether or not any output needs it. No flag exists. |
| **D5** Kernel-scaled FFT padding ⌈4σ_px⌉ | | ✅ | [smoothing.py L97-101](../src/pysole/smoothing.py#L97-L101); optimiser path is `kc_min`-aware (verified in pass 3). `Solver._get_fft_dem_grids` still calls it without `kc` (N5-L2). |
| **D6** Compound drift expansion | | ✅ expansion / 🟡 inputs | Expansion is correct ([interpolation.py L680-698](../src/pysole/interpolation.py#L680-L698)). Compound names that contain curvature still get *raw-DEM* curvature (N5-M4). |
| **D7** Memory streaming in `optimize_bss_variance()` | | ❌ **[V]** | `tracemalloc` peak on a 400×400 DEM: **84.8 / 277.5 / 534.9 MB at n_steps = 6 / 24 / 48**, with a 1.28 MB array. Linear in `n_steps`, about 11 MB per step (≈ 8 grids). Cause: every step's DEM and slope are kept and copied ([variogram.py L643-661](../src/pysole/variogram.py#L643-L661)); pruning only happens after the sweep ([L727-731](../src/pysole/variogram.py#L727-L731)). Extrapolation: a 5 MP DEM at 50 steps is roughly 8 GB. |

### 2.2 Carry-over from `pysole_v043_release_review.md` (pass 3)

| Pass-3 finding | Status now |
|---|---|
| N4-H1 bedrock never written by `run_from_config` | ✅ fixed **[V]** |
| N4-M1 engine ignored | ✅ fixed (test vacuous) |
| N4-M2 `resolve_output_dir` positional bug | ✅ fixed **[V]** |
| N4-M3 LOPO dead | 🟡 plumbing fixed; public `recommend_drift_model` still ignores profile; numeric IDs fragile |
| N4-M4 `--drift-analyzer` no-op / `--verbose` overridden | `--drift-analyzer` ✅ ([solver.py L670](../src/pysole/solver.py#L670) reads the attribute). `--verbose` ❌ |
| N4-M5 API vs config defaults differ | ❌ unchanged (docs only; low impact) |
| N4-M6 `"linear"` rejected | accepted, but see N5-M1 |
| N4-M7 compound-drift raw curvature / un-padded FFT | ❌ unchanged |
| N4-M8 `nrbins` doc/log mismatch | 🟡 code ✅, README + log ❌ |
| N4-M9 Tikhonov sign | ❌ unchanged (low impact) |
| N4-M10 memory/cost | ❌ unchanged (D3/D4/D7) |
| N4-M11 interactive scope narrower than documented; `_sample_pts_cache` never invalidated | ❌ unchanged **[V]** (cache stale after re-migration) |
| N4-M12 duplicate exports | ❌ still 3 write paths (N5-L3) |
| N4-M13 notebook / stale docs | ❌ notebook still has `cv_mode="lopo"` ([ipynb L77](../examples/pysole_quickstart.ipynb)) |
| N4-L (σ_px formula in CHANGELOG, `np.random.seed`, `max()>15`, double docstring) | ❌ all still present ([CHANGELOG L72](../CHANGELOG.md#L72), [interpolation.py L825](../src/pysole/interpolation.py#L825), [solver.py L816](../src/pysole/solver.py#L816), [L1104-1122](../src/pysole/solver.py#L1104-L1122)) |

Earlier audits (`claude_codebase_audit`, `pysole_v042_code_review`, `pysole_v042_review_pass3`): their High items stay fixed (Eikonal sign, `kc`-aware padding, invalid-input guards, compound-drift expansion). I re-checked these by reading the code; no regressions.

---

## 3. Mathematical & Feature Integrity Review

**Fourier sweep (`optimize_bss_variance`).**

- Half-domain default `kc_min = 4π/L_max` gives λ_max = L/2 ([L416](../src/pysole/variogram.py#L416)); correct.
- `lambda_min`/`lambda_max` conversion is `k = 2π/λ`. Precedence is explicit: metric `"wavelength"` prefers λ; `"wavenumber"` prefers `kc`; null falls back through the other pair to the defaults.
- All guards behave: non-positive λ and non-positive `kc_min` raise, `kc_min ≥ kc_max` raises, `kc_max` is clamped to Nyquist.
- `lambda_min`/`lambda_max` are kept, as you asked. I recommend one sentence in the README stating "if both pairs are set, the pair matching `fft_filter_metric` wins" (currently only logged at INFO).
- The `n_steps` count is based on the *linear* mode count `⌊(kc_max−kc_min)·L/2π⌋` but the sampling is geometric. For default ranges this is `N/2−2`, so it clips to 50 for any DEM with N ≥ 104. Harmless; the doc could say "50 for typical DEMs".
- `np.geomspace(kc_max, kc_min, n)` is fine for `kc_min > 0`, which is now guaranteed.
- Memory: see D7. This is the main scalability weakness.

**Lag binning (`nrbins`).** `calculate_variogram` uses `min(30, max(3, in_range_pairs//30))` in the dense branch and `min(30, max(3, n_pairs//30))` in the N > 5000 chunked branch. Both are fine. The chunked branch is a Python loop over N (≈ N numpy calls), acceptable. An explicit `nrbins < 3` is silently raised to 3. The optimiser's log message is inconsistent with the bins actually used (N5-M2).

**Dual Kriging engine.**

- Weights come from one LU factorisation; prediction is an O(N) dot per cell; kriging is invariant to sill scaling (verified in pass 3).
- Tikhonov ε is correctly confined to `K[:N,:N]`, so `Fᵀw = 0` holds exactly.
- ε is added to the *diagonal of Γ* (a semivariogram matrix). In γ-form the stabiliser should be −ε (equivalently +ε on the covariance diagonal). As a result conditioning is slightly worse than unregularised in a narrow near-duplicate regime. Because `mean(diag(K_sample)) ≡ 0`, ε reduces to `1e-6·sill`, so the practical effect is tiny. The CHANGELOG formula ε·mean(diag K) is misleading.
- `lu_factor` on a singular matrix only warns; the `except` fallback to `lstsq/pinv` is therefore rarely reached.
- Chunked variance does `K_inv @ K_rhs` per chunk: O(chunk·(N+d)²). For N = 5 000 samples this dominates run time and is not skippable.

**3-D Eikonal migration.** Closed form `p_z = (B − √(B²−AC))/A` with `A = 1+zx²+zy²`, `B = s1·zx+s2·zy`, `C = |s|²−1/v²` ([migration.py L142-163](../src/pysole/migration.py#L142-L163)). Evanescent cells are clamped to the real part and counted. The sign fix from pass 3 holds. Boundary fallback keeps the unmigrated depth. `valid_d = d_mig > 0` also sends legitimate zero-depth picks to the fallback (low). The profile column passes through unchanged.

**Profile-ID / LOPO.** Works for string IDs on the CSV path and for 5-column arrays. Fragile for numeric IDs, see N5-M3. LOPO is only used when ≥ 3 unique profiles exist (falls back to buffer-LOOCV otherwise; sensible).

**Pipeline / export architecture.** `finalize_bedrock` exports ([L1311](../src/pysole/solver.py#L1311)), then `PipelineExporter.export_all()` exports again ([pipeline.py L152-153](../src/pysole/pipeline.py#L152-L153)), and `Solver.run_pipeline` saves a third time ([solver.py L1495](../src/pysole/solver.py#L1495)). The results are identical, so this is only wasted I/O. `PipelineExporter` is a four-line shim over private `Solver` methods. A consolidation (single `Solver.export_outputs()` call at the end of both entry points; delete `PipelineExporter` or make it real) remains the right streamlining step. The undocumented legacy key `outputs.output_bedrock_map` ([pipeline.py L155](../src/pysole/pipeline.py#L155)) is a fourth, off-schema path.

**pykrige path (newly reachable).** `kriging_interpolation` forwards only the first external drift and discards `variogram_params`, `drift_terms` and `n_cores` ([interpolation.py L741-749](../src/pysole/interpolation.py#L741-L749)). So `engine: "pykrige"` silently ignores the optimised variogram and any multi-drift model. Untested **[U]**.

---

## 4. Schema & Documentation Parity Audit

| Artifact | Parity with `DEFAULT_CONFIG` ([config.py L70-148](../src/pysole/config.py#L70-L148)) | Notes |
|---|---|---|
| `pysole.json` | ✅ identical **[V]** | all 59 keys, values and int/float types |
| README JSON snippet | ✅ identical **[V]** | |
| `examples/wuk/pysole_wuk.json`, `examples/gok/pysole_gok.json` | ✅ no unknown keys, none missing | intentional differences only: paths, `nrbins: 15`, `smooth_bedrock: true`, `output_dir: "."`, prefixes |
| README parameter table | ✅ all keys present | issues below |
| `docs/release_guide.md` | ✅ version strings `0.4.3` consistent | `pyproject.toml` L7 and `__init__.py` L18 agree |
| `docs/drift_analyzer_&_survey_planner.md` | ✅ `--drift-analyzer --profile-col` now valid | `kc` default 0.0314 consistent |
| `CHANGELOG.md` | 🟡 | see overclaims below |

Remaining parity defects:

| # | Location | Defect |
|---|---|---|
| P1 | [README L384-390](../README.md#L384-L390) | `nrbins = max(3, N_pairs/30)`; code is `min(30, max(3, in_range_pairs//30))`. |
| P2 | [README L229](../README.md#L229) | Survey layout `[(profile_id), X, Y, value]` (ID first) is not what the loader does: a numeric ID in column 0 shifts every column (**[V]**: x-range 0–4 instead of ≈ 5e5). Only *string* IDs work in that position. |
| P3 | [README L247](../README.md#L247) | `kc_min` "half of the DEM extent" is ambiguous. It is `4π/L_max`, i.e. λ_max = L/2. |
| P4 | README L260/L267 | `"linear"` advertised without the 2-D validity caveat. |
| P5 | README CLI section ([L636-650](../README.md#L636-L650)) | claims `--verbose` overrides the config; it does not (B6). Also lists neither `--drift-analyzer`, `--profile-col`, `--init` nor `plan-survey`. |
| P6 | `examples/pysole_quickstart.ipynb` L77 | `recommend_drift_model(..., cv_mode="lopo")` is not in the signature → `TypeError`; the return value is also not a 2-tuple. |
| P7 | CHANGELOG | L72 states σ_px = 1/(√2·π·k_c·Δx); the code is σ_px = 1/(k_c·Δ) ([smoothing.py L98](../src/pysole/smoothing.py#L98)). L56 names the old `<prefix>_bedrock_elevation_map.tif`. L76 describes ε·mean(diag K) (≡ 0). Task B claims CLI precedence is "verified" (it is not). |
| P8 | `config.py` | No schema validation: unknown keys are silently ignored and `OutputsConfig.from_dict` drops them. A typo such as `save_bedrock_elevation_maps` produces no warning. |
| P9 | Config reads not in schema | `outputs.output_bedrock_map`, `inputs.output_dir` (used only for the log directory) and `inputs.drift_analyzer` are read by code but are neither in `DEFAULT_CONFIG` nor documented. |

---

## 5. New Findings & Edge Cases

IDs are `N5-*`; **[V]/[C]/[U]** as above.

### [High]
None.

### [Medium]

- **N5-M1 — `"linear"` variogram is invalid in 2-D [V].** [variogram.py L271-275](../src/pysole/variogram.py#L271-L275) is `n + c·min(h/a,1)`. The implied covariance is the cone `(1−r/a)₊`, which is positive-definite only in ℝ¹ (Askey: `(1−r)₊^ν` needs ν ≥ (d+1)/2; here ν = 1, d = 2).
  - Measured on 300 random 2-D points: `min eig(C) = −0.447`, 44 negative eigenvalues; spherical: `+2.05e-3`, 0 negative.
  - Kriging with the model gives variance 0 in 2.2 % of cells (clamped negatives); spherical gives 0 %.
  - The same bounded form is used by `DualKrigingSolver` ([interpolation.py L1046-1047](../src/pysole/interpolation.py#L1046-L1047)).
  - Fix: either drop `"linear"` from the user-facing lists, or document "1-D profiles only / may give unreliable variances in 2-D", or substitute a valid model (e.g. Matérn-3/2).
- **N5-M2 — Optimiser log reports the wrong bin count [C].** [variogram.py L471-492](../src/pysole/variogram.py#L471-L492) uses `max(3, n_pairs//30)` (no cap, all pairs). With N = 140 it reports ≈ 324 bins while `calculate_variogram` uses 30. Together with P1, the "standardised" formula is still not consistent across code, log and docs.
- **N5-M3 — Profile-column ingestion is positional, not name-based [V/C].** [raster.py L836-895](../src/pysole/raster.py#L836-L895). String profile columns are dropped by `select_dtypes(number)` and then re-appended (works). A *numeric* profile column stays inside `pts` and is interpreted by position:
  - `X,Y,value,profile` (4 numeric columns) → `value` is read as `Z_surf` and the profile ID as the observation.
  - `profile,X,Y,value` (README layout) → coordinates shifted.
  - Silent garbage, no error. Fix: when `profile_column` is given, read it from the DataFrame by name, drop it from the numeric block, then append.
- **N5-M4 — Compound drifts use raw-DEM curvature [C].** [solver.py L718](../src/pysole/solver.py#L718) tests `"curvature_dem" in drift_terms` (exact element). `sia_curvature_dem`, `z_dem_curvature_dem`, `sia_z_dem_curvature_dem`, `full_physical` and `full_spatial_physical` never match, so [interpolation.py L732](../src/pysole/interpolation.py#L732) computes curvature from the *unsmoothed* DEM (pixel-scale noise), while the plain `curvature_dem` gets the `kc_opt`-smoothed field. Same drift, two different covariates.
- **N5-M5 — Repo and test hygiene [V].**
  - `pysole/final_bedrock.tif` (30×30, 7.4 kB) sits in the repo root and is **not** matched by `.gitignore` (it ignores only `examples/**/*.tif`). It is not from my review (my last check was clean); it is a byproduct of a run with cwd = repo root. I did not touch it.
  - Because `save_bedrock_elevation_map` is now `True` by default and `finalize_bedrock()` exports, any API call without an output directory or survey path writes `<cwd>/pysole/…` ([config.py L186-189](../src/pysole/config.py#L186-L189)).
  - `tests/test_solver.py` (no `output_dir`) writes into `examples/wuk/input_data/pysole/` and figures there.
  - Fix: add `/pysole/` and `examples/**/pysole/` to `.gitignore`, give those tests `output_dir=tmp`, and note the new default side-effect in the README API section.
- **N5-M6 — `--verbose` / `--debug` ineffective [V].** See B6. Small fix: pass `log_level=log_lvl` to both `Solver.from_config(...)` calls ([pipeline.py L101, L121](../src/pysole/pipeline.py#L121)) and to the earlier `load_config(config_file)` at L81.
- **N5-M7 — Task D claims not met [V/C].** D4 (conditional variance) and D7 (memory streaming); see §2.1. Either implement them or remove them from the release notes. Suggested implementation:
  - `return_variance: bool` in `built_in_kriging_interpolation`, skipped when no uncertainty output is requested;
  - replace explicit `K_inv` with `lu_solve(lu, K_rhs_sub)` per chunk;
  - in the BSS sweep keep only a running best `(kc, slope, dem)` unless `interactive` or `plots_dir` is set, and drop the `.copy()`s.
- **N5-M8 — Weak or vacuous regression tests [C].**
  - `test_kriging_engine_string_propagation` ([L316-324](../tests/test_audit_regressions.py#L316-L324)) never calls kriging.
  - `test_linear_variogram_model_support` ([L360-373](../tests/test_audit_regressions.py#L360-L373)) passes an ignored key.
  - `test_kc_min_half_domain_limit_default` ([test_parameter_defaults.py L22-34](../tests/test_parameter_defaults.py#L22-L34)) asserts only "not None".
  - `test_compound_drift_expansion` asserts "not None".
  - `test_eikonal_migration_oblique_slope` asserts only finiteness.
  - The LOPO test checks plumbing only, not that a fold is actually executed.

### [Low]

- **N5-L1 — Dead branch in `blend_margin_topography`.** The point-array path raises `UnboundLocalError: tapered_thickness` ([interpolation.py L203](../src/pysole/interpolation.py#L203)). **[V]** Nothing in the package calls it that way, but the signature advertises it.
- **N5-L2 — `Solver._get_fft_dem_grids` pads for kc = 0.01** ([solver.py L463](../src/pysole/solver.py#L463)). Any cache-miss `get_smoothed_dem(kc)` with a smaller `kc` under-pads (boundary leakage). Hits on `opt_kc` are fine because the optimiser pre-populates the cache.
- **N5-L3 — Triple export** of the bedrock raster (see §3) plus the legacy `output_bedrock_map` key.
- **N5-L4 — `_sample_pts_cache` never invalidated [V].** Re-running `migrate_eikonal` with a new velocity leaves the post-migration sample points stale (cache identical although depths changed). Also affects the interactive "test another velocity" loop if a kriging pass ran in between.
- **N5-L5 — Tikhonov sign / `mean(diag(K))`** (N4-M9, unchanged). Practical impact ≈ 1e-6·sill.
- **N5-L6 — `perform_migration=False` drops profile ID** and keeps the `max()>15` unit heuristic ([solver.py L816-817](../src/pysole/solver.py#L816-L817)).
- **N5-L7 — `Solver.recommend_drift_model` ignores `survey_profile_column`**, and has no `cv_mode` parameter (see P6).
- **N5-L8 — Interactive prompt prints `nrbins = None`** ([variogram.py L600, L602](../src/pysole/variogram.py#L600)).
- **N5-L9 — `lambda_min < λ_Nyquist` warning fires even when λ is ignored** (wavenumber metric) ([variogram.py L429](../src/pysole/variogram.py#L429)).
- **N5-L10 — `finalize_bedrock` has a duplicated docstring** (second string is a no-op expression) ([solver.py L1104-1122](../src/pysole/solver.py#L1104-L1122)).
- **N5-L11 — `np.random.seed(42)` mutates global RNG state** ([interpolation.py L825](../src/pysole/interpolation.py#L825)); use `np.random.default_rng(42)`.
- **N5-L12 — pykrige path ignores variogram params / multi-drift / n_cores** (§3). Unverified **[U]**.
- **N5-L13 — Import-time backend switch** in [plotting.py L12-16](../src/pysole/plotting.py#L12-L16) is now conditional (no `DISPLAY`), acceptable.
- **N5-L14 — Dedup is O(N·U)** with a Python loop ([raster.py L909-916](../src/pysole/raster.py#L909-L916)); vectorise with `np.add.at` / `bincount` if N grows.
- **N5-L15 — `lu_factor` on singular `K` only warns**; the `except` fallback is rarely reached ([interpolation.py L373-379](../src/pysole/interpolation.py#L373-L379)). Check `np.isfinite(w_z).all()` instead.
- **N5-L16 — Test fixture typo:** `nnrows 30` in the ASC header in `test_optional_outputs.py` L23; harmless because the parser falls back to the grid shape.

---

## 6. Definitive Release Readiness Conclusion

**Do not tag `v0.4.3` yet.** Nothing found is a numerical defect in the default pipeline, and the three functional blockers from the previous pass are genuinely fixed. But the release notes currently assert behaviour that does not exist, one advertised model is mathematically invalid in 2-D, and the first things a new user touches (CLI `--verbose`, the quickstart notebook, the `nrbins` formula in the README) are wrong.

**Required before tagging (all small):**

| # | Action | Size |
|---|---|---|
| R1 | Pass `log_level` into `load_config`/`Solver.from_config` in `pipeline.py` so `--verbose/--debug` win. Add a test that asserts `logger.level == DEBUG` after `run_from_config(log_level="DEBUG")`. | 3 lines |
| R2 | Fix `examples/pysole_quickstart.ipynb` L77 (drop `cv_mode`, fix the return unpacking), and the stale doc claims in P5 and P6. | docs |
| R3 | Align the `nrbins` formula in README L384-390 and in the optimiser log ([variogram.py L471](../src/pysole/variogram.py#L471)) with `min(30, max(3, in_range//30))`. | 5 lines |
| R4 | `"linear"` variogram: remove it from the README lists, or add an explicit 1-D-only caveat and a runtime warning. | docs / 3 lines |
| R5 | Make the CHANGELOG truthful: remove or implement the D4 and D7 claims; fix L56, L72, L76 and the B6 "verified" line. | docs |
| R6 | Hygiene: delete the stray `pysole/final_bedrock.tif`, add `/pysole/` and `examples/**/pysole/` to `.gitignore`, point `test_solver.py` outputs at a temp dir. | trivial |
| R7 | **Run the full suite (`MPLBACKEND=Agg MPLCONFIGDIR=/tmp ./.venv/bin/python -m unittest discover -s tests`) in an environment that has all dependencies** and confirm no failures and no `DeprecationWarning`s. I could not do this. | mandatory |

**Strongly recommended (does not block, but should be on the v0.4.4 list):**
- Implement D4 and D7 (see N5-M7).
- Name-based profile-column ingestion (N5-M3).
- Smoothed curvature for compound drifts (N5-M4).
- Replace the vacuous tests (N5-M8).
- Consolidate exports into one call and delete or promote `PipelineExporter`.
- Invalidate `_sample_pts_cache` when migrated points change.

Once R1–R7 are done, I would classify the codebase as ready for publication.
