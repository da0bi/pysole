# PySole v0.4.2 — Third-Pass Code Review (read-only)

> **Historical document.** This report was written against an earlier development state of PySole and is kept for traceability only. Findings listed here were addressed in later releases (see [CHANGELOG](../CHANGELOG.md) and [pysole_v043_final_audit.md](pysole_v043_final_audit.md)); line numbers and parameter names may no longer match the current code.

Scope: `src/pysole/` at `/home/db/Software/pysole-review`. Baseline: `docs/pysole_v042_code_review.md`.

**Method and limits**
- I read the code line by line. Only numpy 2.2.4, scipy 1.15.3, matplotlib and pyproj were available.
- shapely, pandas, sklearn, rasterio, geopandas and pytest are missing, and there is no network.
- The repo test suite could not be run: `unittest discover` fails at import, 15/15 errors.
- Instead I ran stubbed numerical harnesses in `/tmp/pysole_review2/`. Findings marked **[V]** were reproduced numerically.
- Unmarked findings come from code reading only.
- `.git` is a broken worktree, so I could not diff against the previous version.
- Nothing in the repo was modified. No `__pycache__` or `.pyc` files were left, and no background processes remain.
- The e2e run (`run_from_config(is_batch=False)`) was killed. It blocked on the interactive velocity prompt (see N3-M5).
- Not read: `logging.py`, `__init__.py`, the CHANGELOG body, `docs/*.md`, `examples/*`, most other tests, and `raster.py` L230-345 (CRS checks). Schema/doc parity for examples is therefore only partly verified (see section 3, N3-L1).

---

## 1. Executive Summary

The v0.4.2 hardening pass fixed most of the crash-class and numerical bugs from the previous review.
- Fixed and verified: the missing `sys`/`logger` imports, NaN-aware FFT smoothing, centred quadratic drift, `resample_dem` NaN handling, outline transform, compound drift-term expansion, and the RF thickness target.
- Also changed: the Eikonal closed form was introduced, and the fitted variogram params are now plumbed into kriging.

The pass also introduced a regression and left structural problems open.
1. **[High] The 3D Eikonal migration has a sign error** (`migration.py` L126-165). It was verified against exact plane-over-plane solutions. It displaces picks in the wrong direction and by the wrong amount whenever the surface or bed is tilted. This is a regression from the older single-axis code.
2. **[High] Compound drift names are only partly honoured by the built-in kriging engine.** `sia_space` and `full_spatial_physical` silently lose their spatial terms (up to 32 m difference). Silent quadratic fallbacks remain.
3. **[Medium] The new `fft_filter_metric` has no numerical effect.** It only changes prompt labels. `lambda_*` silently overrides `kc_*`. The linear-in-k sweep puts about 80 % of samples at λ < 100 m.
4. **[Medium] FFT padding is not tied to `kc`**, so the default `kc_min = 4π/L_max` gives an FFT-vs-spatial error of 12-23 m at the largest σ.
5. **[Medium] Output-directory and log plumbing is wrong in `pipeline.py` L142.** It passes the config file as `survey_data_path` and reads the wrong config key. The e2e log confirms it: two `pysole.log` files were created, in `out/` and in `<cfg>/pysole/`.
6. **[Medium] `interactive_optimization` and `interactive_migration` are still ignored by `run_from_config`.** With `is_batch=False` and a TTY the pipeline blocks on prompts.
7. **[Medium] LOPO drift analysis is still dead code.** `survey_profile_column` is a no-op end to end, and the README layout is wrong.
8. **[Medium] Fitted-sill kriging is not scale-covariant** (see M5 in the matrix). The kriged mean shifts up to 7.6 m for a 4e4× sill change, which now matters because fitted sills are wired in.

Test coverage improved on paper (`tests/test_audit_regressions.py`), but most new tests are placeholders and none would have caught the migration sign error.

---

## 2. Audit Resolution Matrix

Legend: ✅ fixed, 🟡 partial, ❌ open, ⬜ not rechecked.
IDs follow my previous report. Line numbers are anchors in the current code.

### 2.1 High findings

| ID | Topic | Status | Evidence / residual |
|---|---|---|---|
| N-H1 | Missing `sys` / `logger` imports | ✅ | `pipeline.py` L9, `config.py` L12. `test_run_from_config_batch_false` still passes `is_batch=True` (misnomer). |
| N-H2 | CLI/config params not forwarded | 🟡 | `velocity` and `finalize_bedrock(**fin_cfg)` are now forwarded (`pipeline.py` L186-199). `resolve_path` arg order fixed (L206). **Open:** `interactive_*` config flags are ignored (L181). `Solver.run_pipeline` is a diverging second pipeline (saves `<prefix>_bedrock`, `show_progress` doubles as a batch flag). |
| N-H3 | NaN in FFT smoothing | ✅ [V] | Normalised convolution, matches the `gaussian_filter` ratio to 0.000 m. NaNs restored (`smoothing.py` L246-256). |
| N-H4 | Quadratic drift conditioning | ✅ | Centre, then square (`interpolation.py` L913-922). |
| N-H5 | `config_path` reset / LOPO / profile column | ❌ | `Solver.__init__` resets `config_path`/`config` (L193-194). LOPO dead: `sample_pts` has 3 columns, so `prof_data` is always None (`solver.py` L677). `survey_profile_column` is never read from the CSV. `--drift-analyzer` is a no-op because the per-stage flag is used (L650). |
| N-H6 | Compound drift names | 🟡 [V] | Expansion and `ValueError` on unknown terms done (`interpolation.py` L677-695). **Residual (N3-H2):** the built-in engine receives the unexpanded names (L742). |
| N-H7 | Default `kc = 0.5` | ❌ | Unchanged in `SurveyPlanner.plan_survey` (L235), `Solver.plan_survey` (L1344), CLI `--kc 0.5` (`config.py` L406). The planner uses fixed 40 px padding and forces dx = dy = 10 for path DEMs (L55). |
| H6 | Variogram params unused | 🟡 | `opt_variogram_params` is now assigned and passed (`solver.py` L935, L707). Caveats: fitted at `kc_max` (least smoothed), always "spherical", applied to T/D/P targets (sill has P units). `recommend_drift_model` still hard-codes range = 100, sill = 1 (L1338). |

### 2.2 Medium findings

| ID | Topic | Status | Evidence / residual |
|---|---|---|---|
| N-M1 | `resample_dem` NaN | ✅ | Only out-of-range cells are filled (`raster.py` L390-408). Bilinear still spreads NaN by one cell and does not anti-alias. |
| N-M2 | Outline transform | ✅ | Rebuilt with `from_bounds` (`raster.py` L709). |
| N-M3 | FFT padding vs kc | ❌ [V] | `precompute_fft_grid(dem, dx, dy)` is called without `kc` (`variogram.py` L438, `solver.py` L451, `smoothing.py` L290). Error vs `gaussian_filter(mode='mirror')` for σ_px = 16 / 32 / 64: 0.65 / 12.0 / 23.4 m at default pad, about 0.003 m when `kc` is passed. |
| N-M4 | kc-sweep robustness | 🟡 | Bin count is `min(30, max(3, in_range//30))` (`variogram.py` L207-213). `counts` weighting exists. **Open:** `fit_variogram_model` is called without `counts` (L553, L585), so the weighting is dead. The first kc is evaluated twice. `kc_min > kc_max` is silently rewritten to `0.5·kc_max` (L422-423). |
| N-M5 | Memory | ❌ | All smoothed DEMs and slopes are retained (+`.copy()`, `variogram.py` L634-635). Pruning happens only after the sweep (L702-705). For N > 5000 the chunked branch uses `n_pairs//30` bins (L165, about 6e5 bins at N = 6000). Log at L445 reports `n_pairs//30` bins, but at most 30 are used. |
| M5 | Dual kriging | ❌ [V] | Explicit `K_inv` (`interpolation.py` L373). Tikhonov `max(1e-6·sill, 1e-6)` also hits the Lagrange block (L363-364). Mean is not sill-scale-covariant (sill ×400 → 1.26 m, ×4e4 → 7.6 m). `DualKrigingSolver` in the analyzer uses an absolute 1e-8 and duplicated variogram code, and its `success=False` is silent. |
| N-M7 | `finalize_bedrock` | 🟡 | Non-NaN `np.max` fixed (`solver.py` L971-973). Clip counts logged, but over the whole rectangle rather than inside the mask. **Open:** double docstring (L1082-1100), `interactive=True` default, `kriged_std` does not describe the final field. |
| N-M8 | RF holes | 🟡 | Regresses thickness, excludes the ring (`interpolation.py` L795-826). **Open:** zero-thickness holes use ≤ 0.1, no training subsample, fake progress bar. |
| N-M9 | Eikonal migration | ❌ (regressed) | Closed form with `delta_zs` implemented, but **sign error** (see N3-H1). |
| N-M10 | Drift analyzer parity | ❌ | Analyzer curvature is raw while production is smoothed. The solver injects smoothed curvature only when the literal `"curvature_dem"` is present (`solver.py` L698); compound names fall back to raw curvature (`interpolation.py` L720). `slope_floor_deg` not forwarded. `target_name[0].upper()` bug remains. Analyzer folds are O(N³) each. |
| N-M11 | Config side effects / merge | ❌ | `load_config` creates dirs and re-inits logging. It does a shallow merge: a user `pre_migration` dict replaces the whole default sub-dict (`config.py` L256-261). Duplicate exporters remain (`Solver.export_outputs` vs `PipelineExporter.export_all`) with different raster names. Default prefix "final" gives `final_final_bedrock_elevation_map`. `OutputsConfig` duplicate removed ✅. New output-dir bug: see N3-M3. |
| N-M12 | Plotting | 🟡 | `_get_cell_edge_extent` fixes the half-pixel plot offset ✅. **Open:** `if not os.environ.get("DISPLAY"): matplotlib.use("Agg")` (`plotting.py` L12-13), the mean line includes zeros while the histogram excludes them, linear kc axis for `all_kc_variances`. |

### 2.3 Low and design findings

| ID | Topic | Status | Note |
|---|---|---|---|
| N-L2 | `getattr(self, "pre_kriging_points", None)` | ❌ | `solver.py` L588 |
| N-L3 | `get_drift_functions` dead/buggy | ❌ | Imported but unused |
| N-L4 | O(N·U) dedup in `load_survey_points` | ❌ | `raster.py` L849-853 |
| N-L5 | BSSOptimizer fallback with no survey points | ❌ | `solver.py` L886 |
| N-L8 | `load_outline` silent fallback | ❌ | `raster.py` L722-725, L781. Should raise when `outline_path` was given. |
| N-L9 | `_transform_coords_to_wgs84` returns projected coords on failure | ❌ | |
| N-L10 | Stencil spacing | ⬜ | Not rechecked |
| N-L11 | `compute_cutoff_wavelength(kc)` | ✅ | Single argument |
| M10 | Unit heuristic `max()>15`; `valid_d = d_mig > 0` conflates NaN with ≤ 0 | ❌ | `solver.py` L795, `migration.py` L201 |
| M11 | `sigma=1.0` px in `blend_margin_topography` | ❌ | `interpolation.py` L203 |
| M12 | Zero-boundary points fixed ≈ 100, no dedup | ❌ | `interpolation.py` L625 |
| M13 | `xllcenter` | ✅ | `raster.py` L483-491 |
| — | `.asc` export uses `cellsize` only, ignoring dy | ❌ | `raster.py` L203 |
| — | Unused imports (logging, LinAlgError, RegularGridInterpolator, resolve_input_path, MigrationResult, OptimizationResult, fft_gaussian_smooth, GridGeometry, compute_gradients, Any); `dataclass` import mid-file in `smoothing.py` | ❌ | |
| — | Silent `except` in `drift_analyzer.py` L250/L272, interactive loops | ❌ | |
| — | CHANGELOG mislabels | ⬜ | |

Other IDs from the earlier report that are not listed above were not rechecked in this pass.

### 2.4 Regression tests

`tests/test_audit_regressions.py` (about 12 tests) now exists, but its quality is weak.
- Outline orientation and nan_rings only exercise numpy arrays.
- The tracks test only checks dict keys.
- The NPY round-trip loads a headerless array.
- The compound-drift test only asserts "not None".
- The oblique-migration test only asserts finiteness, which is why the sign bug passed.
- `test_migration` uses a T = 0 grid and asserts shapes only.
- There is no real `is_batch=False` test for N-H1.

---

## 3. New Findings and Edge Cases

### [High]

**N3-H1 — 3D Eikonal migration sign error** [V] — `src/pysole/migration.py` L126-165.
- The code uses `s1, s2 = +∂T/∂x, +∂T/∂y`.
- The correct horizontal slowness components are `u = −∂T/∂x`, `w = −∂T/∂y`, with:
  - `A = zx² + zy² + 1`
  - `B = u·zx + w·zy`
  - `C = u² + w² − v⁻²`
  - `p_z = (B − √(B² − A·C)) / A`
  - `p_x = u − p_z·zx`
  - `p_y = w − p_z·zy`
  - displacement `= T·v²·p`
- I checked against the exact plane-over-plane solution in 6 cases. The corrected sign matches to 0.01 m. Three representative cases:

| Case | Exact (dx, dy, dz) | Current code |
|---|---|---|
| flat surface, bed dip bx = 0.1 | (+7.87, 0, −78.7) | (−7.87, 0, −78.7) |
| surface zx = 0.2, flat bed | (0, 0, −291) | (+111.9, 0, −268.6) |
| zx = 0.3, zy = 0.1; bed (−0.04, 0.03) | (−17.2, +12.9, −431) | (250, 64.7, −345.6) |

- Flat/flat gives zero displacement under either sign, so trivial tests pass.
- `d_mig` is near-correct only for planar beds, so picks are relocated to the wrong places on real beds. This is a regression from the older single-axis code.
- **Fix:** negate the gradient, and add exact plane-over-plane tests with non-zero T grids.

**N3-H2 — Drift expansion not applied in built-in kriging** [V] — `src/pysole/interpolation.py` L742, L308-309.
- `built_in_kriging_interpolation` receives the unexpanded `drift_terms` and tests only for the literal `"linear_xy"` and `"quadratic_xy"`.
- As a result `sia_space` ≡ `sia` (max difference to `[sia, linear_xy]` = 32.4 m).
- `full_spatial_physical` drops `linear_xy` (23.4 m).
- `sia_z_dem` ≡ `[sia, z_dem]` (0.0), which is fine.
- Other silent fallbacks to a 6-term quadratic [V]:
  - `universal` with `drift_terms=[]`.
  - `method="sia"` without slope/dem grids.
- **Fix:** pass the expanded list to the engine and raise instead of silently falling back.

### [Medium]

**N3-M1 — `fft_filter_metric` is cosmetic** [V] — `variogram.py` L408-412, L513.
- `use_wavelength` only changes prompt labels.
- `lambda_min`/`lambda_max` override `kc_max`/`kc_min` for either metric, with no warning (`kc_max=0.05, lambda_min=200` → `kc_max` used = 0.0314).
- With metric "wavelength" and null lambdas, it falls back to `kc_*`. This matches the README, but the README does not say the metric is cosmetic.
- Either document it as a display-only option, or make the metric choose which pair of bounds is authoritative and warn on conflict.

**N3-M2 — Linear sampling in k is badly distributed in scale** [V] — `variogram.py` L390-429.
- Default grid: kc 0.314 → 0.00628 (dx 10 m, 2 km domain), n = 50.
- 40 of 50 samples have λ < 100 m (essentially unsmoothed). Only 3 of 50 lie in λ ∈ [300, 3000] m.
- In the synthetic test, `optimal_kc = kc_min`.
- Use geometric spacing in k, or uniform spacing in σ = 1/kc. Warn when the optimum is at a boundary.

**N3-M3 — Output dir / log mis-plumbed** [V] — `pipeline.py` L142.
- `resolve_output_dir(inputs_cfg.get("output_dir"), config_file)` passes the config file as `survey_data_path` and reads `inputs.output_dir`. The schema key is `outputs.output_dir`.
- In the e2e run, `pysole.log` was created in both `e2e/out/` and `e2e/pysole/`, and an empty `e2e/pysole/figures` was created.
- This is the same class of bug as N-H2 (positional-argument mix-up).

**N3-M4 — `kc ≤ 0` semantic inversion** [V] — `smoothing.py` L234-235.
- `kc ≤ 0` returns an unfiltered DEM, but kc → 0 means maximum smoothing (`kc = 1e-9` gives a flat field).
- A user `kc_min = 0` or `lambda_max = inf` therefore puts the raw DEM into the sweep as a candidate. `_eval_single_kc` rejects only `kc < 0`.
- Return the mean field, or reject `kc ≤ 0` with a clear error.

**N3-M5 — Interactive flags ignored; e2e blocks** [V] — `pipeline.py` L181.
- `interactive_flag = not is_batch and isatty()` ignores the config flags `interactive_optimization` and `interactive_migration`.
- The e2e run with `is_batch=False` stopped at "Test another migration velocity? [y/N]:", and `plt.pause` warned on the Agg backend.

**N3-M6 — `n_steps` handling** [V] — `variogram.py`.
- The discrete Fourier-mode count is correct math (Δk = 2π/L).
- On real domains it always saturates at the cap of 50. It matters only for tiny domains: a 400 m domain gave 18 modes with step 0.0166 vs fundamental 0.0157.
- `n_steps ∈ {1, 2}` is silently raised to 3.
- The README says δk = Δ/n_steps, but `linspace` gives Δ/(n_steps − 1).
- Interactive prompts skip the `kc_min < kc_max` re-validation and do not recompute `n_steps`.

**N3-M7 — Boundary and input edge cases for λ bounds** [V] — `variogram.py` L390-429.
- `lambda_min = 0` gives `ZeroDivisionError`.
- A negative `lambda_max` is silently dropped (`kc ≥ 0` filter).
- `lambda_min` below 2·dx is silently clamped to k_Nyquist, with no log.
- `kc_min > kc_max` is silently rewritten to `0.5·kc_max`.
- Validate and raise with explicit messages.

**N3-M8 — Half-Domain `kc_min = 4π/L_max`** — `variogram.py` L420-423.
- The math is correct: λ_max = L/2, σ_max = 0.08·L_max, so σ_px,max = N/(4π).
- It interacts badly with the fixed 40 px FFT pad (N-M3): the error is 12-23 m at the largest σ.
- `L_max = max(extent)` ignores long, narrow domains. Consider using the shorter side for the lower bound.
- Gaussian "corner" kc is not a hard cutoff (gain 0.607 at k = kc), but the docs call λ_c a "cutoff".

**N3-M9 — Fitted-variogram reuse** — `variogram.py` L553, L585; `solver.py` L935.
- The variogram range used for the `mean_variance` cut-off is fitted on the least-smoothed kc.
- It is not re-fit when the kc range is changed interactively.
- `fit_variogram_model` is hard-coded "spherical" and ignores `counts`.
- The fitted P-sill is applied to T/D targets, so their kriging variance is mis-scaled.

**N3-M10 — Kriging mean depends on the sill** [V] — `interpolation.py` L363-364. See M5 above. It matters now that fitted sills are wired in. Use a relative regularisation (`ε·mean diag K`), excluding the Lagrange block.

### [Low]

- **N3-L1 — Doc/schema mismatches:**
  - The `calculate_variogram` docstring says `max(3, N_pairs//30)` and "`warn_low_pairs` defaults to False" (actual default True).
  - The README documents the survey CSV as `[(profile_id), X, Y, value]`, but `load_survey_points` is positional with X, Y first.
  - The README `n_steps` formula is off by one.
  - `pysole.json` / README do not state that `fft_filter_metric` is cosmetic.
  - Objective 3 (100 % parity) is only partly verified. I compared `config.py`, `pysole.json`, and the README sections I read (L120-290, L330-400). I did not read `examples/` or `docs/`.
- **N3-L2 — `pipeline.py` / tests:** the `test_run_from_config_batch_false` name is misleading (it passes `is_batch=True`).
- **N3-L3 — Plot:** `plt.pause(0.5)` is called on a non-interactive backend (`plotting.py` L262).
- **N3-L4 — Default `output_prefix` "final"** produces `final_final_bedrock_elevation_map`.
- **N3-L5 — `.asc` export** ignores dy (`raster.py` L203).

---

## 4. Mathematical and Streamlining Opportunities

**Numerical precision**
1. **Pass `kc` into `precompute_fft_grid`.** It is the one-line fix for the 12-23 m padding error. Alternatively, derive padding as `ceil(4σ_px)` capped at N.
2. **Replace explicit `K_inv`** with a Cholesky or `solve` on the augmented system. Scale the regularisation by `trace(K)/n`, and keep it off the Lagrange rows.
3. **Sweep spacing:** use `kc = geomspace(kc_max, kc_min, n)`. This gives uniform sampling in log λ, which suits the self-similar scale dependence of the variance.
4. **Fix the migration sign**, and add analytic plane-over-plane unit tests with non-zero T. Treat NaN separately from non-positive `d_mig`.
5. **Enforce sill-scaling invariance in tests:** multiply the sill by 400 and assert the mean changes by less than 1e-6.

**Memory (O(N) vs O(N²))**
6. Stream the kc sweep. Compute the variogram for each kc, keep a running best plus scalar metrics, and discard the smoothed DEM and slope arrays. This removes O(n_kc·Ny·Nx) retention and the `.copy()` calls.
7. The chunked `calculate_variogram` branch must use the same capped bin count as the dense branch. Otherwise it creates about 6e5 bins at N = 6000.
8. In the analyzer, use one Cholesky factorisation per fold, and the cheap LOO identity (`e_i = [K⁻¹z]_i / [K⁻¹]_ii`) instead of O(N³) per fold.
9. Vectorise `load_survey_points` dedup (`np.unique` on rounded coordinates, or a KD-tree).

**Architecture**
10. Merge the two pipelines (`Solver.run_pipeline` and `pipeline.run_from_config`) and the two exporters (`Solver.export_outputs` and `PipelineExporter.export_all`) into one code path with one naming scheme.
11. Make `load_config` pure: parse, deep-merge defaults, validate. Do directory and log setup in the pipeline. A recursive deep merge fixes N-M11.
12. Move the schema (keys, defaults, types, bounds) into one declarative table in `config.py`. Generate `pysole.json`, the README parameter table and CLI flags from it, to guarantee parity.
13. Remove dead code and unused imports: `get_drift_functions`, the mid-file `dataclass` import, and the silent `except` blocks.
14. Replace the DISPLAY-based `matplotlib.use("Agg")` at import time with explicit lazy backend selection.

**Priority order for fixes**
1. Migration sign (N3-H1)
2. Drift-term expansion (N3-H2)
3. Output-dir plumbing (N3-M3)
4. FFT padding with `kc` (N-M3)
5. Interactive flags (N3-M5)
6. kc ≤ 0 semantics and λ-bound validation (N3-M4, N3-M7)
7. Kriging regularisation (N3-M10)
8. Remaining items
