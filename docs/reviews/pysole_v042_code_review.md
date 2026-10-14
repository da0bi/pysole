# PySole v0.4.2 — Read-Only Code Review (second pass)

> **Historical document.** This report was written against an earlier development state of PySole and is kept for traceability only. Findings listed here were addressed in later releases (see [CHANGELOG](../CHANGELOG.md) and [pysole_v043_final_audit.md](pysole_v043_final_audit.md)); line numbers and parameter names may no longer match the current code.

**Scope:** `src/pysole/` (14 modules, ~7,000 lines), all read line by line. Includes the previously unreviewed `optimize_bss_variance`, `calculate_bedrock`, `finalize_bedrock`, `smooth_bedrock_dem`, `plotting.py`, `config.py` and `pipeline.py`.
**Mode:** Read-only. No file in the repo was modified or created. I checked afterwards that no `__pycache__` or other new files exist in the repo. Test scripts are in `/tmp/pysole_review/` (`verify.py`, `verify_mig.py`, `static_checks.py`).

**Evidence tags**
- **[V]** reproduced by running code. The output is quoted.
- **[CODE]** derived from reading the code; not executed.

> [!NOTE]
> **What I could not run.** The sandbox has no `pytest`, `sklearn`, `pandas`, `shapely`, `rasterio` or `geopandas`, and no network. I did not run the repo's test suite. The harness stubs the missing imports (`shapely`, `sklearn`, `pandas`) and runs the real NumPy/SciPy code paths. Anything that needs rasterio or geopandas (vector-outline rasterisation, GeoTIFF I/O) or scikit-learn (RF gap filling) is [CODE] only. I did not measure performance.

---

## 1. Executive summary

v0.4.2 fixes most of the first audit's geometric and statistical defects, and I confirmed this by running the code. The Y-mirroring in `coords_to_grid_indices`, the per-call drift normalisation (leave-one-out RMSE is now 1.12 against 163 before), the NaN-ring outline, the NPY round-trip, the variogram scale-invariance and the survey-planner mirroring and budget are all fixed.

**The release is not ready to ship.** The remaining problems fall into three groups.

1. **Two audit items are marked fixed but are not.** H6 (the fitted variogram is never used) has plumbing in `kriging_interpolation`, but nothing ever assigns `Solver.opt_variogram_params`. M5 (dual-kriging regularisation and inverse) is untouched, although the changelog says it was fixed.
2. **New entry-point and wiring defects.**
   - `run_from_config(is_batch=False)` crashes with a `NameError`.
   - `--init` crashes with a `NameError`.
   - `run_from_config` silently drops `migration_parameters.velocity` and `finalization_parameters`.
   - `--drift-analyzer` and `--profile-col` are parsed and then ignored.
   - `Solver(survey_profile_column=...)` is overwritten with `None` at the end of `__init__`.
   - The Drift Analyzer never receives `profile_data`, so LOPO-CV is dead code from the solver.
3. **New numerical defects.**
   - The NaN-aware FFT smoothing is wrong. I measured a +88 % elevation error next to NaN cells.
   - Drift names that `DriftAnalyzer` recommends (`sia_z_dem`, `full_physical`, …) silently become a quadratic-XY drift in production. The difference between the two grids is exactly 0.0.
   - `DriftBasis` quadratic terms are ill-conditioned in UTM coordinates, which gives VIF 999 and a condition number of 8·10¹².
   - The SurveyPlanner default `kc = 0.5` rad/m is now a 13 m wavelength, which is essentially no smoothing.
   - The Eikonal migration deviates from the exact solution when the surface tilts in both x and y.

**Test gap.** None of the 7 regression tests proposed in the first audit exist. The existing suite exercises only `is_batch=True`, which is why the `sys` `NameError` was never seen.

| Area | Health |
|---|---|
| Grid geometry / orientation | Good |
| Variogram fitting / BSS search | Good, with robustness gaps |
| Smoothing | Needs work (NaN handling, padding) |
| Kriging core (dual form) | Mathematically correct; M5 open, H6 not wired |
| Drift Analyzer | Structurally fixed, but not trustworthy (conditioning, name mismatch, dead LOPO, hard-coded variogram) |
| Migration | Close to correct; oblique-slope error and depth reference bias |
| Pipeline / config / CLI | Several hard failures and silent config drops |
| Survey planner | Geometry fixed; default `kc` regression |

---

## 2. Audit Resolution Matrix

✅ resolved · 🟡 partial · ❌ open

### HIGH
| ID | Finding | Status | Evidence / residual |
|---|---|---|---|
| H1 | `coords_to_grid_indices` Y-mirrored | ✅ | [V] y=0.5→row 0, y=3.5→row 3. See [raster.py:91-95](../src/pysole/raster.py#L91-L95). |
| H2 | Drift re-normalised per call | ✅ | [V] LOO RMSE 1.12 (σ=1). `DriftBasis` is fixed once. New defects in N-H4, N-H5 and N-M10. |
| H3 | Vector outline Y-flip | ✅ [CODE] | [raster.py:702](../src/pysole/raster.py#L702) flips. New defect N-M2: the transform is stale after resampling. Not executed (no rasterio). |
| H4 | NaN-ring outlines ignored | ✅ | [V] outer + hole → True fraction 0.84 (expected 0.84). Failures now log a warning, but the fallback is still "whole DEM is glacier" (see N-L8). |
| H5 | SurveyPlanner mirrored / budget | 🟡 | [V] 0 vertices outside the mask, 820 m ≤ 1000 m budget. Still open: gaps across rock are joined by straight segments ([survey_planner.py:171](../src/pysole/survey_planner.py#L171)), and the flowline is still a straight column (D6). |
| **H6** | **Kriging variogram heuristic** | ❌ | `kriging_interpolation(variogram_params=…)` exists, but `opt_variogram_params` is **never assigned** (grep: only read at [solver.py:671](../src/pysole/solver.py#L671) and [solver.py:704](../src/pysole/solver.py#L704)). The result is `None`, so range = 0.6·max distance, sill = var(z), nugget = 0, as before. The BSS fit in [variogram.py:558](../src/pysole/variogram.py#L558) is discarded. |
| H7 | Product slope clamp asymmetric | ✅ [CODE] | The same `slope_floor_deg` is used at [solver.py:609](../src/pysole/solver.py#L609) and [solver.py:719](../src/pysole/solver.py#L719). |
| H8 | GPX/GeoJSON projected coords | ✅ | `pyproj` reprojection ([survey_planner.py:31-43](../src/pysole/survey_planner.py#L31-L43)). [V] UTM33 → (14.01°E, 47.05°N). Residual N-L9: on failure it silently writes projected values as lat/lon. |

### MEDIUM
| ID | Finding | Status | Evidence / residual |
|---|---|---|---|
| M1 | `kc` not rad/m | ✅ | `fftfreq(N, d=dx)·2π` ([smoothing.py:172](../src/pysole/smoothing.py#L172)). **Side effect:** the old defaults were never rescaled, see N-H6. |
| M2 | FFT boundaries / NaN | 🟡 | Reflect padding exists, but it is fixed at 32 px (N-M3: 20 m edge error [V]). The NaN "normalised convolution" is **mathematically wrong** (N-H3). NaNs are not restored. |
| M3 | `fit_variogram_model` | ✅ | [V] fitted range 600 and sill 40·scale at scales 1, 1e-2 and 1e-4. `model_type` is honoured and failures are logged. |
| M4 | O(N²) memory | 🟡 | The chunked branch only runs when `precomputed_dists is None`. `optimize_bss_variance` always precomputes a full `pdist` ([variogram.py:406](../src/pysole/variogram.py#L406)), so the main path is unchanged. See N-M5. |
| **M5** | **Dual-kriging cost / regularisation** | ❌ | Explicit `K_inv` is still built ([interpolation.py:371](../src/pysole/interpolation.py#L371)). `K += eye(K.shape[0])·reg` hits the **whole** matrix including the Lagrange block ([interpolation.py:361-362](../src/pysole/interpolation.py#L361-L362)). The `except` fallback is still unreachable. The variance clamp is silent. `DualKrigingSolver` uses an absolute `1e-8` ([interpolation.py:1020](../src/pysole/interpolation.py#L1020)). The CHANGELOG claim ("point-point block only") is **incorrect**. |
| M6 | Curvature recomputed per call | ✅ [CODE] | `DriftBasis` caches the grids. |
| M7 | Analyzer ≠ production | 🟡 | The SIA floor is aligned, but the user's `slope_floor_deg` is not forwarded. Still different: curvature (raw vs `kc`-smoothed), sampling (nearest cell vs bilinear), variogram (hard-coded range 100 m), zero-boundary points (ignored in CV). See N-M10. |
| M8 | AICc comparability | ✅ [CODE] | A common-intersection mask exists ([drift_analyzer.py:289-295](../src/pysole/drift_analyzer.py#L289-L295)). One failing candidate shrinks the mask for all. |
| M9 | MDI vs permutation | ✅ [CODE] | Permutation importance and NaN masking are implemented. The method is not called from `run_diagnostics`, and the importance is computed in-sample. |
| M10 | Migration edge cases | 🟡 | Fallback and evanescent counts are now logged. Still open: `valid_d = d_mig > 0` conflates NaN with ≤0; `v_sq` vs `inv_v_sq` use different velocities ([migration.py:143,174](../src/pysole/migration.py#L143)); the `max() > 15.0` unit heuristic remains ([solver.py:792](../src/pysole/solver.py#L792)). |
| M11 | `blend_margin_topography` | 🟡 | Continuous blend ✅ and `sampling=(dy,dx)` ✅. Still open: `sigma=1.0` in pixels, the EDT border treated as non-background, and the grid-vs-points shape ambiguity. |
| M12 | Zero-boundary points | 🟡 | `border_value=1` ✅. Still open: fixed ~100 points, no de-duplication. |
| M13 | I/O round trip | 🟡 | [V] NPY marker row 0→0. The CSV path is [CODE] only. Still open: `.asc` `cellsize` ignores `dy` ([raster.py:203](../src/pysole/raster.py#L203)). New bug: `xllcenter` is treated as `xllcorner` ([raster.py:472](../src/pysole/raster.py#L472)). |
| M14 | Input validation | 🟡 | `DriftBasis` raises on unknown terms ✅; `max_length_km ≤ 0` raises ✅. Still open: [V] `variogram_model="foo", method="bar"` is accepted silently, and the CHANGELOG says this raises. Unknown or compound `drift_terms` in `kriging_interpolation` are silently ignored (N-H2). No `kc_min < kc_max` check (N-M6). RF NaN holes are never selected. |

### LOW and design items
| Item | Status | Note |
|---|---|---|
| Wasted `compute_gradients` | 🟡 | The BSS loop is fixed. Still used for slope only in [interpolation.py:664,748](../src/pysole/interpolation.py#L664), [solver.py:463](../src/pysole/solver.py#L463), and on the T-grid in migration. |
| Silent `except` sites | ❌ | Still at [drift_analyzer.py:250,272](../src/pysole/drift_analyzer.py#L250), `DualKrigingSolver.success=False`, and the `raster.py` CRS helpers. |
| Duplicated variogram code | 🟡 | `built_in` uses `evaluate_variogram_model`. `DualKrigingSolver` still has two inline copies. |
| Docstring mismatches | 🟡 | `warn_low_pairs` doc says False but the default is True. `migrate_eikonal(plotit)` is unused. |
| Double `[INFO]` prefix | ❌ | Many sites. |
| Unused imports | ❌ | [V] AST scan: `logging`, `LinAlgError`, `get_drift_functions` (drift_analyzer); `RegularGridInterpolator` (migration); `resolve_input_path` (pipeline); `MigrationResult`, `OptimizationResult`, `fft_gaussian_smooth` (solver); `logging`, `GridGeometry` (survey_planner); `compute_gradients` (variogram); `Any` (smoothing). |
| Type hints | ❌ | `list[Any]` for drift callables; `plan_survey` returns a dict. |
| D1 `DriftBasis` | ✅ (defects) | See N-H4, N-M10. |
| D2 `VariogramParams` | ❌ | A dict is plumbed but never populated. |
| D3 Orientation API | 🟡 | Only `coords_to_grid_indices`. |
| D4 Wavelength cutoff | 🟡 | Dual API ✅; padding not scaled to σ. |
| D5 Robust variogram | ❌ | |
| D6 Flowline planner | 🟡 | Reprojection and clipping ✅; no flowline tracing. |
| D7 Error counters | ❌ | |
| 7 suggested regression tests | ❌ | grep finds none of them. |

> [!IMPORTANT]
> The CHANGELOG's "Codebase Audit Implementations" uses H1…H8 labels that do **not** match the audit's IDs. For example, it lists "H2 Outline Orientation", "H4 Spearman" and "H8 Survey Planner Boundary Masking". That makes the changelog unreliable for tracing. It also describes `PipelineManager`/`PipelineConfig` classes that do not exist, and a ~150-line `config.py` that is actually 428 lines.

---

## 3. New findings and deep-dive review

### HIGH

**N-H1. `run_from_config` crashes with `NameError` when `is_batch=False` (the default) [V]**
[pipeline.py:222](../src/pysole/pipeline.py#L222) uses `sys.stdin.isatty()`, but `sys` is not imported. AST scan: undefined name `['sys']`. Tests pass only because they call `run_from_config(..., is_batch=True)`, and `not True and …` short-circuits.
Related [V]: [config.py:300](../src/pysole/config.py#L300) `create_template_config` uses `logger`, which is not imported at module level. `pysole --init` writes the file and then raises `NameError: name 'logger' is not defined`.

**N-H2. Config is silently dropped; two diverging pipelines [CODE]**
- `pipeline.run_from_config` calls `solver.migrate_eikonal(interactive=…)` **without** `velocity=migration.get("velocity")` ([pipeline.py:224](../src/pysole/pipeline.py#L224)). `migrate_eikonal` hard-codes 0.16 m/ns ([solver.py:781](../src/pysole/solver.py#L781)). A user's `velocity` is ignored. The example configs use 0.16, which hides this.
- `finalize_bedrock(interactive=…)` ([pipeline.py:227](../src/pysole/pipeline.py#L227)) never receives `finalization_parameters`. RF filling, margin blending and smoothing from JSON never run through the main entry point.
- `Solver.run_pipeline` ([solver.py:1388-1483](../src/pysole/solver.py#L1388-L1483)) does handle both, but nothing calls it. **Fix:** make `run_from_config` a thin wrapper over `Solver.run_pipeline`.
- `interactive_optimization: false` in the config is overridden by TTY detection.
- Line 234 calls `resolve_path(output_bedrock_file, config_file, solver.output_dir)` with **positional arguments in the wrong order**. The signature is `(path, output_dir, survey_data_path, config_path)`. [V] `resolve_path("out.tif", "<cfg>.json", "<dir>")` returns `<cfg>.json/out.tif` and **creates a directory named like the config file**.

**N-H3. FFT "normalised convolution" is mathematically wrong [V]**
[smoothing.py:157-158,246](../src/pysole/smoothing.py#L157-L158) mean-fills the NaNs, filters the filled data, then divides by the smoothed mask. Normalised convolution needs `smooth(data·mask) / smooth(mask)`, with zeros in the masked cells. As written, valid cells near a NaN edge are scaled by 1/mask (up to 10⁶).

> Test: cone DEM 3100 m, left 40 columns NaN, σ = 5 px. First valid column: FFT **5823 m** vs correct **3085 m** (true 3100). Max error in the 20 adjacent columns is **2738 m**.

This hits `optimize_bss`, the optimal slope and curvature drift, and the BSS output for any DEM with nodata. The same function backs `SurveyPlanner`.

**N-H4. `DriftBasis` quadratic terms are ill-conditioned in real coordinates [V]**
[interpolation.py:871-880](../src/pysole/interpolation.py#L871-L880) evaluates `x**2`, `y**2` and `x*y` on raw coordinates. It then "centres" them with `m_x`, `m_y` and `m_x·m_y`. These are the means of x and y, not of x², y² and xy. `x²` is not centred correctly.
> 200 points over a 2 km square: UTM cond([1,X]) = **8.3·10¹²**, max VIF = **999**; local coordinates give cond 15 and VIF 5.8.

Consequences: `quadratic_xy` is flagged multicollinear and penalised in every real dataset, the CV solves are near-singular, and production (which centres correctly) differs. **Fix:** centre first, `((x-mx)/sx)²`.

**N-H5. DriftAnalyzer never reaches LOPO-CV, and `survey_profile_column` is lost [V + CODE]**
- `run_diagnostics(profile_data=…)` is the only way to enable LOPO ([drift_analyzer.py:371,386](../src/pysole/drift_analyzer.py#L371)). Neither call site passes it ([solver.py:677](../src/pysole/solver.py#L677), [solver.py:1331](../src/pysole/solver.py#L1331)). Only a unit test does, so every real run uses buffer-LOOCV.
- [V] `Solver(survey_profile_column="profile", config_path=…)` returns `None, None` because [solver.py:192-194](../src/pysole/solver.py#L192-L194) re-initialises `config`, `config_path` and `survey_profile_column` after they were set. `recommend_drift_model` reads the attribute ([solver.py:1324](../src/pysole/solver.py#L1324)), while `_execute_kriging_pass` reads the config dict ([solver.py:667](../src/pysole/solver.py#L667)).
- Because `opt_variogram_params` is never set (H6), `a_0 = 100 m` and the buffer radius is 50 m. `recommend_drift_model` hard-codes `range=100, sill=1` ([solver.py:1341](../src/pysole/solver.py#L1341)).

**N-H6. Recommended compound drifts silently become quadratic-XY in production [V]**
`CANDIDATE_DRIFT_MODELS` returns terms like `["sia_z_dem"]` and `["full_physical"]`. `DriftBasis` expands them, but `kriging_interpolation` tests `"sia" in drift_terms` by list membership ([interpolation.py:653-655](../src/pysole/interpolation.py#L653-L655)). The compound names match nothing, so `built_in` falls into its default 6-term quadratic drift.
> [V] `max|krig(["sia_z_dem"]) − krig(["quadratic_xy"])| = 0.0`, and `max|krig(["sia_z_dem"]) − krig(["sia","z_dem"])| = 3.5`.

**Fix:** expand names once (the `DriftBasis.primitives` logic) and raise on unknown terms. The same applies to `drift_terms=["sia"]` when no slope grid is available, which also silently becomes quadratic.

**N-H7. SurveyPlanner / `plan_survey` / CLI default `kc = 0.5` rad/m is now λ_c ≈ 13 m [V]**
After the M1 fix, kc is true rad/m, but the defaults ([survey_planner.py:83,235](../src/pysole/survey_planner.py#L83), [solver.py:1347](../src/pysole/solver.py#L1347), [config.py:395](../src/pysole/config.py#L395)) were not rescaled. At dx = 10 m the filter gain is 0.82 at Nyquist and 0.99 at λ = 100 m, so there is no smoothing.
> Synthetic glacier: D_SIA mean **40 m** with kc = 0.5 vs **67 m** with kc = 0.01 (λ = 628 m). The planner default is biased thin and noisy.

Specify the cutoff as λ (≈ 4–10 × expected thickness) instead.

### MEDIUM

**N-M1. `resample_dem` erases NaN/nodata [V]** ([raster.py:390-399](../src/pysole/raster.py#L390-L399))
All NaN cells, including interior holes and a clipped-to-glacier mask, are nearest-neighbour filled. [V] 100 NaN cells → 0. With `outline_path = null` the mask is later derived from NaNs, so after resampling the whole DEM becomes "glacier". Only fill cells that were out of range of the source grid, and use area averaging when coarsening.

**N-M2. Vector outline uses a stale transform after resampling [CODE]** ([raster.py:688-700](../src/pysole/raster.py#L688-L700))
`meta["transform"]` is the native-resolution file transform, but `out_shape` is the resampled shape. The mask is mis-scaled whenever `dx`/`dy` differ from native. Rebuild the transform with `from_bounds(*bounds, width, height)` from the final geometry.

**N-M3. FFT padding is a fixed 32 px, regardless of σ [V]** ([smoothing.py:160-161](../src/pysole/smoothing.py#L160-L161))
σ_px = 1/(kc·dx). [V] For dx = 5 m, kc = 0.01 (σ = 20 px) the edge error vs `gaussian_filter(mode="reflect")` is **20.4 m** (interior 4·10⁻⁴ m). Use pad ≥ 4σ_px, capped at the grid size. The same issue applies to `smooth_bedrock_dem(method="fft_lowpass")` ([solver.py:1051-1068](../src/pysole/solver.py#L1051-L1068)), which has no padding at all and re-implements the FFT.

**N-M4. `optimize_bss_variance` failure paths [V]** ([variogram.py:696-697](../src/pysole/variogram.py#L696-L697))
`if best_kc is None: best_kc = float(kc_max)`:
- `kc_min > kc_max` with `kc_max=None` → `TypeError: float() argument … 'NoneType'`.
- Wavelength mode, or fewer than 3 valid picks → the same `TypeError`.
- If `kc_max` is given, it silently returns the **unsmoothed** slope grid labelled kc = kc_max [V].

Validate the ranges and raise a clear `ValueError`. Other issues in this function:
- `use_wavelength` silently flips if any `lambda_*` is given, even when `fft_filter_metric="wavenumber"`.
- The default kc grid is not dx-aware: at dx = 50 m the search has only ~6 values.
- `mean_variance` is an unweighted mean over lag bins, so a bin with 1 pair counts as much as one with 500. Use pair-count weights.
- The first kc is evaluated twice.

**N-M5. Memory behaviour of the kc sweep [CODE]**
- `optimize_bss_variance` keeps **every** smoothed DEM and slope, with additional `.copy()` ([variogram.py:638-639](../src/pysole/variogram.py#L638-L639)). `Solver` then copies all of them into `_smoothed_dem_cache` ([solver.py:937-938](../src/pysole/solver.py#L937-L938)). Only the optimal one is ever used. For a 5000² DEM and 50 kc values this is ~10 GB.
- A full `pdist` is built for any N ([variogram.py:406](../src/pysole/variogram.py#L406)), which defeats the chunked branch.
- Each kc re-bins the same distances and re-allocates three N²/2 arrays inside a ThreadPool.

**N-M6. Bin-count formula [V]** ([variogram.py:208-213](../src/pysole/variogram.py#L208-L213))
`max(3, n_pairs // 30)` counts all pairs, but only those within `maxdist` are binned. [V] N = 800 gives **10,530 bins**, median 24 pairs per bin, and 11 % of bins with < 10 pairs. `curve_fit` is unweighted. Choose ~20–40 bins by lag and weight the fit by √pairs.

**N-M7. `calculate_bedrock` / `finalize_bedrock` [CODE + V]**
- `max_thickness = max(max(pts)*1.5, 500)` becomes `nan` if any value is NaN ([solver.py:973](../src/pysole/solver.py#L973)). [V] `max(nan*1.5, 500.0) = nan`, so `np.clip(…, nan)` yields an all-NaN grid. Use `np.nanmax`.
- Negative kriged thickness and thickness above 1.5 × max are clipped **silently** ([solver.py:974](../src/pysole/solver.py#L974)). Log the clipped fraction.
- `kriged_std` is computed from the kriging variance before smoothing, RF fill and blending, and without the clip. The exported "uncertainty" does not describe `final_thickness`.
- `export_outputs`: `res` for `save_basal_shear_stress_uncertainty` is never appended to `saved` ([solver.py:392-394](../src/pysole/solver.py#L392-L394)).
- The double docstring ([solver.py:1085-1103](../src/pysole/solver.py#L1085-L1103)) makes the second string a no-op; the smoothing parameter docs are lost.
- The signature defaults `interactive=True`, so `input()` is called from library use.

**N-M8. RF hole definition [CODE]** ([interpolation.py:756,778](../src/pysole/interpolation.py#L756))
"Holes" are defined as `mask & (thickness ≤ 0.1)`. That includes legitimate margin cells (zero-boundary pseudo-points force D → 0 there) and clipped cells. RF replaces the margin constraint with a guess made from raw X/Y/Z/slope, which cannot extrapolate. NaN cells are never holes. Predict thickness (≥ 0) rather than elevation, exclude the margin ring, and subsample the training set (up to M·N rows × 100 trees × depth 15).

**N-M9. Migration geometry [V]** ([migration.py:141-177](../src/pysole/migration.py#L141-L177))
I compared the displacement vector against the exact Eikonal solution on a planar surface over a planar bed (`verify_mig.py`):

| Surface slopes (zx, zy) | exact Δx | code Δx | note |
|---|---|---|---|
| 0.2, 0.05 | −41.54 | −44.04 | 6 % |
| 0.3, 0.1 | −51.63 | −61.20 | **19 %** |
| zy = 0 (3 cases) | exact | exact | |

The formula is exact only when the surface tilts in one axis; the `s12_quadr` cross-term form deviates for oblique tilt. Δz is accurate to 0.3 %. Check against `MIG.m`. A closed form that needs no trig grids: with `u=-∂T/∂x`, `w=-∂T/∂y`, `A=1+zx²+zy²`, `B=u·zx+w·zy`, `C=u²+w²−v⁻²`, take `p_z=(B−√(B²−AC))/A`, `p_x=u−p_z·zx`, `p_y=w−p_z·zy`, and displacement `=T·v²·(p_x,p_y,p_z)`. The evanescent condition is `B²−AC<0`. This matched truth to 0.01 m in all 5 test cases.
Second issue: `d_mig = −dz` is measured from the **source** elevation. The thickness at the migrated location is `−dz + [z_s(x_mig,y_mig) − z_s(x₀,y₀)]`. In the test (α ≈ 11°, 60 m shift) the omitted term is ~6 m (1.5 % of 415 m). Confirm this against MIG.m intent.

**N-M10. Analyzer ≠ production, remaining [CODE]**
- Curvature is computed from the raw DEM ([interpolation.py:849](../src/pysole/interpolation.py#L849)), while production uses the `kc`-smoothed curvature ([solver.py:697](../src/pysole/solver.py#L697)). The CHANGELOG claims they are aligned.
- Drift features use nearest-cell lookup ([interpolation.py:889](../src/pysole/interpolation.py#L889)), production uses bilinear.
- `include_zero_boundary` is stored but never used in CV.
- `slope_floor_deg` is not forwarded to `DriftBasis`.
- `target_name[0].upper()` prints "O*sin(alpha)" ([drift_analyzer.py:377](../src/pysole/drift_analyzer.py#L377)).
- Scalability: 12 candidates × up to 100 folds, each a fresh dense O(N³) `solve` plus an N×N `cdist` ([drift_analyzer.py:238-273](../src/pysole/drift_analyzer.py#L238-L273)). This is impractical above a few thousand picks. Thin by along-track binning, or use block-removal (Schur-complement) updates from one factorisation.

**N-M11. CLI and config [V + CODE]**
- `--drift-analyzer` and `--profile-col` are parsed and never read [V] ([config.py:375-388](../src/pysole/config.py#L375-L388)).
- `pysole plan-survey` parses correctly [V]. However, `SurveyPlanner(dem=path)` forces `dx = dy = 10` on `load_dem`, so any DEM is resampled to 10 m ([survey_planner.py:62-63](../src/pysole/survey_planner.py#L62-L63)).
- `load_config` has side effects: it creates `pysole/` and `pysole/figures/` directories and re-initialises logging on every call. It also silently uses defaults if the config file does not exist, does a shallow merge (a user `pre_migration` dict replaces the default dict), and does no key validation (typos are ignored).
- `run_from_config` reads `inputs.output_dir` ([pipeline.py:188](../src/pysole/pipeline.py#L188)), while the schema puts it in `outputs`.
- Two `OutputsConfig` classes with **different defaults** (`config.py` vs `pipeline.py`). `Solver.export_outputs` and `PipelineExporter.export_all` export overlapping rasters twice under different names.
- Method `ordinary` with explicit `drift_terms` still runs universal kriging, because `has_ext_drifts` overrides the method ([interpolation.py:689](../src/pysole/interpolation.py#L689)).

**N-M12. Plotting [CODE]**
- `if not os.environ.get("DISPLAY"): matplotlib.use("Agg")` ([plotting.py:12-13](../src/pysole/plotting.py#L12-L13)) is a Linux-only test. On macOS/Windows it forces Agg and interactive plots never show. `__init__` imports `plotting`, so a bare `import pysole` changes the global backend.
- Extents are inconsistent. `imshow` should use cell-edge `bounds` and `contour` uses centres. The migration plot uses centres for both ([plotting.py:154](../src/pysole/plotting.py#L154)), and the other plots use `bounds` for both, so one layer is off by ½ pixel.
- The thickness histogram excludes zeros, but the "Mean" line includes them.
- The filename prefix is ignored for all plots except the BSS pair, so stage-1 and stage-2 plots overwrite each other.

### LOW
- **N-L1.** `Solver.__init__` defaults (`pre_kriging_method="universal"`, `pre_drift_terms=["sia"]`, target `"P"`) disagree with the config defaults (ordinary, `[]`). The API default triggers the "SIA on P" warning.
- **N-L2.** [solver.py:588](../src/pysole/solver.py#L588) reads `pre_kriging_points`, which is never defined. `_sample_pts_cache` is never invalidated after `optimize_bss`.
- **N-L3.** `get_drift_functions` is dead code and buggy: [V] late-binding makes all functions return the last evaluator (f0 == f1). It is also imported unused in `drift_analyzer`.
- **N-L4.** `load_survey_points` de-duplication averages **all** columns, including any profile-ID column. It is an O(N·U) Python loop ([raster.py:830-834](../src/pysole/raster.py#L830-L834)). Use `bincount`/`np.add.at`.
- **N-L5.** `BSSOptimizer` fallback with no survey points ([solver.py:884-885](../src/pysole/solver.py#L884-L885)) builds a fake pick set from the DEM with thickness 10 m. The resulting "optimal kc" is meaningless.
- **N-L6.** `survey_planner.max_depth_m` becomes NaN (and invalid JSON) when the DEM has NaNs, because `np.gradient` spreads NaN and only `isnan(dem)` is zeroed.
- **N-L7.** `__init__.py` hard-codes `__version__`; `survey_planner` imports `pyplot` at module level.
- **N-L8.** `load_outline` falls back to "whole DEM" after a failed outline parse, with only a warning. An explicit `outline_path` that fails should raise.
- **N-L9.** `_transform_coords_to_wgs84` returns projected coordinates on failure, which then get written as lat/lon.
- **N-L10.** `compute_surface_curvature` uses `np.gradient(np.gradient(f))` (stencil spacing 2·dx). The compact 3-point stencil resolves 2·dx features.
- **N-L11.** `compute_cutoff_wavelength(kc, dx, dy)` ignores `dx`/`dy`.

---

## 4. Mathematical and streamlining opportunities

**Numerical precision**
1. **Exact Eikonal closed form** (N-M9). It replaces ~10 trig grids and `s12_quadr`, removes the oblique-slope error, and the discriminant gives the evanescent mask directly.
2. **Normalised convolution done correctly** (N-H3): `smooth(data·mask)/smooth(mask)`, restore NaN where `smooth(mask) < ε`. Use one routine for FFT and spatial smoothing.
3. **Centre-then-square** for quadratic drift (N-H4). It brings the condition number from 10¹² to ~15 and makes VIF meaningful.
4. **Spectral slope.** Multiply the smoothed spectrum by `i·kx`, `i·ky` and `ifft` once per component. This avoids the extra transfer function `sin(k·dx)/dx` that `np.gradient` applies after smoothing.
5. **Pair-weighted** variogram objective and fit (N-M4, N-M6). A robust estimator (Cressie–Hawkins) suits heavy-tailed BSS products near margins.
6. **Calibrated variogram** fitted on drift-detrended residuals of the *kriged target* (T, P or D), not the unfiltered stage-1 P product. Carry it as a `VariogramParams` dataclass (D2).
7. **Positivity.** Kriging thickness and clipping at 0 biases the mean at the margins. Consider kriging `log(D+ε)` or reporting the clipped fraction.

**Performance and memory**
1. **Evaluate kriging only inside the outline mask** (plus the blend margin). `built_in` evaluates all M·N cells, and callers discard outside values ([solver.py:976-987](../src/pysole/solver.py#L976-L987)). A glacier typically fills 10–40 % of the DEM rectangle, so this is a 2.5–10× saving for free.
2. **Replace `K_inv @ K_rhs` with an LU solve** per chunk, make variance optional (D1/M5), and apply the regularisation to the point block only.
3. **Keep only the optimal smoothed DEM/slope**; recompute on demand. Drop `.copy()`.
4. **Precompute `np.digitize` bin indices once** for the kc sweep, since the distances do not change, and accumulate with `np.bincount(weights=…)`. Cap the full `pdist` and subsample or block it above ~5000 picks.
5. **`rfft2`/`irfft2` with `next_fast_len` padding.** Real input halves memory and time. Optionally store spectra as `complex64`.
6. **DriftAnalyzer folds.** One factorisation with block removal instead of a fresh O(N³) solve per fold; thin the picks first.

**Architecture**
1. One pipeline: `run_from_config` → `Solver.run_pipeline`. Delete the duplicate `OutputsConfig`, and let `Solver.export_outputs` be the only exporter.
2. One drift-name resolver shared by `DriftBasis` and `kriging_interpolation`. Raise on unknown terms (N-H6).
3. `DualKrigingSolver` should call `evaluate_variogram_model`; delete its inline copies.
4. Validate config against a schema (unknown keys, ranges, `kc_min < kc_max`), and make `load_config` side-effect free.
5. Replace swallowed exceptions with counters in the final log (D7), e.g. failed CV folds.

**Tests to add** (the first audit's 7, plus new): the 7 from the first audit; `run_from_config(is_batch=False)` with a non-TTY stub; the config velocity actually reaching migration; compound drift names equal the explicit primitives; NaN-edge smoothing vs the reference (error < 1 m); the quadratic-basis condition number in UTM coordinates; Eikonal vs exact on a planar surface and bed with oblique slopes; SurveyPlanner default `kc` against a reference; `optimize_bss` with `kc_min > kc_max` raises `ValueError`.

## 5. Suggested fix order
1. **N-H1, N-H2** (the entry point works and honours config), then **N-H5, N-H6** (analyzer correctness).
2. **N-H3, N-M3** (smoothing) and **N-H4** (conditioning), then **H6/M5** (wire `VariogramParams`, fix the regularisation).
3. **N-H7** (planner default), **N-M1/M2** (resampling vs outline), **N-M9** (migration).
4. Remaining MEDIUM, then LOW, plus the test additions above.
