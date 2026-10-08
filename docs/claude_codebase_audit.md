# PySole Codebase Audit (read-only)

**Scope:** `/home/db/Software/pysole/src/pysole/` (13 modules, ~6,550 lines). Source folder verified before starting.
**Mode:** Read-only. No source file was modified. Verification scripts live in `scratch/audit_verify.py` and `scratch/audit_verify2.py` in the artifacts folder.
**Reviewer stance:** Senior code reviewer and applied mathematician.

## Coverage and confidence

| Module | Depth |
|---|---|
| `migration.py`, `smoothing.py`, `variogram.py`, `interpolation.py`, `drift_analyzer.py`, `survey_planner.py` | Full line-by-line read |
| `raster.py` | Full read of `GridGeometry`, `BedrockMap`, `load_dem`, `load_outline`, `load_survey_points`. Remaining helpers skimmed |
| `solver.py` | Lines 566-860 and 1275-end read in full. Lines 1-565 and 860-1275 only grep-scanned (`optimize_bss`, `calculate_bedrock`, `finalize_bedrock` **not** reviewed in depth) |
| `plotting.py`, `config.py`, `pipeline.py`, `logging.py` | Not reviewed beyond grep. Out of scope for the numerics |

Each finding carries one of two tags:
- **[VERIFIED]**: reproduced by running code. Evidence is given.
- **[CODE]**: derived from reading the code. I did not execute it, so treat it as high-confidence but unproven.

No linter (ruff, pyflakes) is installed in `.venv`, so unused-import claims are marked "likely".

---

## Executive summary

Four independent defects are confirmed. They make the **DriftAnalyzer ranking unreliable** and cause **spatial misalignment (Y-mirroring)** in the outline and survey-planner paths.

1. **H1** `coords_to_grid_indices` mirrors Y, so drift covariates are sampled at the wrong locations.
2. **H2** Drift features are re-normalised per `predict()` call, so cross-validation scores are meaningless.
3. **H3** Vector outlines (`.shp`/`.geojson`) are rasterised top-down but the DEM is bottom-up.
4. **H5** Survey-planner tracks are mirrored relative to the SIA raster.

H1 and H2 together mean the v0.4.0 DriftAnalyzer headline feature currently ranks models on invalid CV scores. I recommend fixing these before tagging or pushing the release.

---

## HIGH severity

### H1. `coords_to_grid_indices` is Y-mirrored **[VERIFIED]**
[raster.py:73-95](file:///home/db/Software/pysole/src/pysole/raster.py#L73-L95)

The grid is bottom-up. `y_coords` is ascending and `load_dem` flips files on load. The method nevertheless computes `rows = (maxy - y) / dy`, which assumes a top-down grid.

Evidence on a 4x3 grid with `bounds=(0,0,3,4)`:
```
y=0.5 -> row 3   (bottom-up grid needs row 0)
y=3.5 -> row 0   (bottom-up grid needs row 3)
```
**Impact:** every `z_dem`, `sia` and `curvature_dem` drift covariate in `get_drift_functions` ([interpolation.py:818-847](file:///home/db/Software/pysole/src/pysole/interpolation.py#L818-L847)) is sampled at the Y-mirrored location. This affects VIF, CV and the recommended model. The production kriging path uses `RegularGridInterpolator` and is **not** affected, so analyzer and production disagree.

**Fix:** `rows = clip(((y - miny) / dy).astype(int), 0, M-1)`. Prefer `np.floor`, since `.astype(int)` truncates toward zero for negatives.

---

### H2. Drift features are normalised per call, which invalidates CV **[VERIFIED]**
[interpolation.py:811-852](file:///home/db/Software/pysole/src/pysole/interpolation.py#L811-L852), [interpolation.py:960-967](file:///home/db/Software/pysole/src/pysole/interpolation.py#L960-L967)

Every drift lambda standardises with statistics of **the points passed in**: `np.mean(x)`, `np.ptp(x)`, `np.nanmean(vals)`, `np.nanstd(vals)`.
- At fit time the statistics come from the training set.
- At `predict()` time they come from the validation set.
- In buffer-LOOCV the validation set is a **single point**. Then `x - mean(x) = 0` and `std = 0`, so **every drift feature is 0** at the held-out point.

Evidence: pure linear field `z = 0.5x + 0.2y + noise(σ=1)`, drift `linear_xy`, leave-one-out:
```
LOO RMSE = 163.66   (expected ~1)
```
**Impact:** all candidate CV scores (RMSE, R², AICc) are dominated by this artefact, so the ranking is not trustworthy.

**Fix (design suggestion D1):** build a `DriftBasis` object once. It evaluates and normalises each covariate at all sample points with fixed global statistics, stores the `(mean, scale)` pair, and exposes `design_matrix(idx)`. CV then just slices rows. This also removes H1, M6 and the analyzer/production inconsistencies in M7.

---

### H3. Vector outline mask is vertically mirrored vs the DEM **[VERIFIED]**
[raster.py:688-701](file:///home/db/Software/pysole/src/pysole/raster.py#L688-L701)

`features.rasterize` with `from_bounds(...)` or the file transform returns a **top-down** array. The DEM was flipped to bottom-up in `load_dem` ([raster.py:557-558](file:///home/db/Software/pysole/src/pysole/raster.py#L557-L558)), and the returned mask is never flipped. The CSV-polygon branch uses `ensure_spatial_coords` (bottom-up) and is consistent, so the two outline paths disagree.

Evidence: a polygon covering the **south** half (y in 0..100) of a 20x20 grid:
```
mask True rows: 10..19    (bottom-up grid => should be 0..9)
```
**Impact:** glacier mask applied to the wrong half of the DEM. This affects zero-boundary points, masking, `tt_grid[~mask]=nan`, blending and the SIA depth.

> [!WARNING]
> I tested a synthetic asymmetric polygon. Please re-check the WUK/GOK examples: either the outlines happen to be near-symmetric, they are supplied as raster masks, or results were visually accepted despite the shift.

**Fix:** `return mask[::-1].astype(bool)` (rasterise with a top-down transform, then flip). Add an asymmetric-polygon regression test.

---

### H4. NaN-separated ring outlines are silently ignored **[VERIFIED]**
[raster.py:716-756](file:///home/db/Software/pysole/src/pysole/raster.py#L716-L756)

The rasterisation block (`MplPath ... return outer_mask & ~hole_mask`) is indented **inside the `else:`** branch, which is taken only when there are no NaN rows. When NaN separators are present (outer ring plus nunatak holes, the feature documented), `rings` is built and then discarded. The function falls through to `return ~np.isnan(dem_grid)`.

Evidence: outer square plus hole, expected True fraction ~0.60:
```
mask True fraction: 1.0   (outline silently ignored)
```
Compounding this, the surrounding `except Exception: pass` blocks ([raster.py:704](file:///home/db/Software/pysole/src/pysole/raster.py#L704), [raster.py:757](file:///home/db/Software/pysole/src/pysole/raster.py#L757)) turn *any* failure (missing geopandas, bad file, CRS mismatch) into "whole DEM is the glacier" with no log line.

**Fix:** de-indent the rasterisation block. Replace the blanket `pass` with `logger.error` plus `raise`, or at least a prominent warning.

---

### H5. SurveyPlanner tracks are mirrored; budget not enforced **[VERIFIED + CODE]**
[survey_planner.py:102-103](file:///home/db/Software/pysole/src/pysole/survey_planner.py#L102-L103), [survey_planner.py:118](file:///home/db/Software/pysole/src/pysole/survey_planner.py#L118), [survey_planner.py:130-132](file:///home/db/Software/pysole/src/pysole/survey_planner.py#L130-L132)

**(a) Mirroring [VERIFIED]:** `y_coords = np.linspace(maxy, miny, ny)` is **descending**, but `d_sia` is bottom-up (row 0 = Y_min) and the figure uses `origin="lower"`. Evidence: glacier mask in rows 5-25 (y = 50-250 m), yet:
```
L1_longitudinal y_range=[356, 549]    <- outside the glacier
T1_transverse   y=508 ;  T2_transverse y=366
```
**(b) Half-cell offset [CODE]:** `linspace(minx, maxx, nx)` places samples on cell *edges* with effective spacing `(maxx-minx)/(nx-1) != dx`. Cell centres are `minx + (i+0.5)*dx`. Reuse `GridGeometry.create(...)`.

**(c) Budget [CODE]:** `long_len = min(long_len, 0.4*L_max)` caps only the **reported number**. The coordinates are not truncated, so the exported track can exceed the budget. `n_cross = max(2, ...)` forces at least 2 cross-profiles even if the remaining budget is ~0, and the cross-profile lengths are never summed against `L_max`. In my synthetic run the total was 788 m under 1000 m, so the budget violation arises only when the flowline exceeds 0.4 L_max.

**(d) Gaps [CODE]:** the `> 0` masks connect disjoint segments with straight lines across rock outcrops, and `np.diff` counts those jumps in `length_m`.

---

### H6. Kriging variogram is a heuristic, not fitted **[CODE]**
[interpolation.py:244-250](file:///home/db/Software/pysole/src/pysole/interpolation.py#L244-L250)

`built_in_kriging_interpolation` ignores the fitted variogram: `range_a = 0.6 * max pairwise distance`, `sill = var(z)`, `nugget = 0`. `variogram_model` is only a shape name.
- For non-stationary targets (bedrock with a trend), `var(z)` includes trend variance, so the sill is inflated.
- The "uncertainty" grids (`kriged_std`) are therefore **not calibrated** estimates.
- Because weights depend on range and shape, the interpolated surface itself is also driven by this arbitrary range.
- The fitted `a_range`/`sill`/`nugget` from `fit_variogram_model` are never passed to the final interpolation.

**Fix:** pass `VariogramParams(model, nugget, psill, range)` through `kriging_interpolation`, fitted on drift-detrended residuals.

---

### H7. Product-target slope clamping is asymmetric (low-slope bias) **[CODE]**
[solver.py:604-606](file:///home/db/Software/pysole/src/pysole/solver.py#L604-L606) vs [solver.py:711-715](file:///home/db/Software/pysole/src/pysole/solver.py#L711-L715)

- Forward: `P = D * max(sin α, 1e-4)`. NaN slope falls back to an arbitrary `0.1`.
- Inverse: `D = P / max(sin α, sin(slope_floor_deg))`, with the floor defaulting to 5 degrees.

At a data point with α = 2 degrees, the kriged P honours `D·sin 2°`, but the back-transform divides by `sin 5°`. The recovered thickness at the data point is `D · sin2°/sin5° ≈ 0.40·D`. Kriging no longer reproduces the data in low-slope zones, which is where glaciers are thickest. Use the **same** floor in both directions.

---

### H8. GPX/GeoJSON written in projected coordinates **[CODE + log evidence]**
[survey_planner.py:154-197](file:///home/db/Software/pysole/src/pysole/survey_planner.py#L154-L197)

GPX `lat`/`lon` are filled with raw projected X/Y (the test log shows e.g. `[424843.37, 210813.23]`). That is invalid GPX (|lat| > 90) and useless on a handheld GPS, which is the stated use. GeoJSON (RFC 7946) must be WGS84; projected coordinates will be misplaced by GIS tools. Reproject with `pyproj` from `self.crs` to EPSG:4326 and warn if `crs is None`.

---

## MEDIUM severity

### M1. `kc` is not in rad/m; it scales with dx² **[VERIFIED]**
[smoothing.py:134-135](file:///home/db/Software/pysole/src/pysole/smoothing.py#L134-L135), [variogram.py:25-32](file:///home/db/Software/pysole/src/pysole/variogram.py#L25-L32)

`kx = fftshift(fftfreq(N)) * 2π|dx|` multiplies by dx, where a physical wavenumber requires `2π·fftfreq(N, d=dx)` (division). The code is internally consistent with `compute_cutoff_wavelength`, but `kc` has implicit units that depend on dx. Docstrings and logs say rad/m.

Evidence, same `kc = 0.5`:
```
dx= 2 m -> 50 m      dx= 5 m -> 314 m
dx=10 m -> 1257 m    dx=20 m -> 5027 m
```
The same default smooths over very different physical scales. `SurveyPlanner`'s default `kc=0.5` with `dx=10` gives a 1.26 km cutoff. This is presumably the legacy MATLAB convention, but it should be a deliberate, documented choice. Since backward compatibility is no longer a constraint, I suggest specifying the cutoff **as a wavelength in metres**.

### M2. FFT smoothing boundary handling [CODE]
[smoothing.py:131-143](file:///home/db/Software/pysole/src/pysole/smoothing.py#L131-L143)
- FFT implies periodic boundaries, so opposite DEM edges bleed into each other. Use reflect/edge padding (e.g. pad by ~3σ).
- NaNs are mean-filled (a step edge at the glacier/rock boundary, producing ringing and slope artefacts near the margin), and NaN positions are not restored afterwards. Prefer normalised convolution (smooth `data·mask` and `mask` separately, then divide).

### M3. `fit_variogram_model` issues **[VERIFIED]**
[variogram.py:199-251](file:///home/db/Software/pysole/src/pysole/variogram.py#L199-L251)
- `var_val = np.var(semivars)` has units of value⁴ while the sill has units value². Sill bounds `[1e-6, 10*var_val]` are scale-dependent. Same shaped variogram, three scales:
  ```
  scale 1e+00: true sill~45     fitted 40
  scale 1e-02: true sill~0.45   fitted 0.124   range 113  (true 600)
  scale 1e-04: true sill~0.0045 fitted 1.24e-06 range 500
  ```
  Use `sill0 = mean(γ[last third])` and bounds relative to `max(γ)`.
- `model_type` is accepted and ignored (always spherical), yet downstream code selects exp/gauss/lin by name.
- Bare `except Exception` falls back silently; log the failure.

### M4. `calculate_variogram` is O(N²) in memory [CODE]
[variogram.py:145-177](file:///home/db/Software/pysole/src/pysole/variogram.py#L145-L177)

Holds `pdist` (float64), `sq_diffs` (float64) and `bin_indices` (int64), each `N(N-1)/2` long. For N = 20,000 picks that is ~2·10⁸ pairs, roughly 1.6 GB per array and ~5 GB peak. Compute in row blocks with `np.bincount` accumulation, or randomly subsample pairs. Along-track picks also make the first lag bin dominated by near-duplicate neighbours; consider a minimum-lag or per-profile declustering.

### M5. Dual-Kriging variance cost [CODE]
[interpolation.py:366-432](file:///home/db/Software/pysole/src/pysole/interpolation.py#L366-L432)
- An explicit inverse `K_inv = lu_solve(lu, eye)` is formed even when only the mean is needed, and every chunk does `K_inv @ K_rhs_sub`. Total cost is O(M_grid·N²), and the inverse is O(N³) and N² memory.
- Prefer `lu_solve(lu, K_rhs_sub)` per chunk (better conditioned) and make variance optional (`return_variance=True`).
- `scipy.linalg.lu_factor` only **warns** on exact singularity; it does not raise. The `except Exception` fallback is therefore rarely reached. Check `np.isfinite` on `w_z` or use `cho`/`lstsq` on a conditioned matrix.
- The `1e-6·I` regularisation is added to the whole augmented matrix (including the zero Lagrange block). It is scale-blind relative to γ, whose magnitude is `sill = var(z)`. I tested 3 exact duplicate points: it did **not** crash and gave plausible output, so this is a robustness concern, not a confirmed failure.
- `var_sub` is clamped with `np.maximum(…, 0)`, which hides negative-variance (ill-conditioning) signals; log how many cells were clamped.

### M6. DriftAnalyzer recomputes the full DEM curvature per call [CODE]
[interpolation.py:837-847](file:///home/db/Software/pysole/src/pysole/interpolation.py#L837-L847), [interpolation.py:902-908](file:///home/db/Software/pysole/src/pysole/interpolation.py#L902-L908), [interpolation.py:960-965](file:///home/db/Software/pysole/src/pysole/interpolation.py#L960-L965)

`eval_curvature_dem` runs `compute_surface_curvature(dem)` (4 `np.gradient` over M×N) and builds a `GridGeometry` on every invocation. `get_drift_functions` is called in `__init__` **and** in every `predict()`. With 12 candidates x ≤100 LOO points x 2 calls, that is thousands of full-grid evaluations. This explains the ~225 s for the drift-analyzer and interpolation tests. `DriftBasis` (D1) fixes it.

### M7. Analyzer and production use different drift definitions **[CODE]**

| Aspect | DriftAnalyzer | Production kriging |
|---|---|---|
| SIA slope floor | `max(α,1°)`, `max(sin,1e-3)` ([interpolation.py:833-834](file:///home/db/Software/pysole/src/pysole/interpolation.py#L833-L834)) | `slope_floor_deg` (default 5°) ([interpolation.py:666](file:///home/db/Software/pysole/src/pysole/interpolation.py#L666)) |
| Curvature | raw DEM | `kc`-smoothed ([solver.py:689-691](file:///home/db/Software/pysole/src/pysole/solver.py#L689-L691)) |
| Variogram | `sill=1`, `nugget=0`, range from BSS fit; `recommend_drift_model` hard-codes **range=100 m** ([solver.py:1315](file:///home/db/Software/pysole/src/pysole/solver.py#L1315)) | heuristic (H6) |
| Z-scoring | per-point-set statistics (H2) | whole-grid statistics |

The model ranked #1 is therefore not the model that will actually run.

### M8. AICc comparability [CODE]
[drift_analyzer.py:274-306](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py#L274-L306)

AICc is computed on CV residuals with a different `n_valid` per candidate, because LOPO/buffer skips differ. AIC/AICc are comparable only on the same data. Evaluate every model on the common set of successfully predicted indices. Also `k_param = n_drift + 1` excludes the constant term. State the convention in the docs.

### M9. RF importance is MDI, but docs say "Permutation" [CODE]
[drift_analyzer.py:125-129](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py#L125-L129)

`rf.feature_importances_` is impurity-based (MDI), which is biased toward high-variance continuous features and unstable under collinearity. `CHANGELOG.md` and the docs describe it as *Permutation* importance. Either use `sklearn.inspection.permutation_importance` (preferably on out-of-bag or held-out data) or correct the text. The docstring also says "Partial Correlations" but computes marginal Pearson/Spearman and omits Spearman. NaN rows are not masked: `pearsonr` returns NaN and RF may raise.

### M10. Migration edge cases [CODE]
[migration.py:141-212](file:///home/db/Software/pysole/src/pysole/migration.py#L141-L212), [solver.py:785](file:///home/db/Software/pysole/src/pysole/solver.py#L785)
- `tt_grid[~outline_mask] = nan` followed by `np.gradient` spreads NaN one cell into the glacier. Interpolated `dz` is NaN there, so `d_mig = max(-nan,0)` falls back to the **unmigrated** depth with no log line. Picks near the margin silently skip migration. Log the fallback count.
- `d_mig = max(-dzi, 0)` then `valid_d = d_mig > 0`: any legitimate zero or positive `dz` is also replaced by unmigrated depth, which is discontinuous. Distinguish NaN from ≤0.
- `s3 = sqrt(max(1/v² − |s_h|², 0))`: the clamp hides evanescent cells (|s_h| > 1/v), i.e. kriged traveltime gradients inconsistent with the velocity. Count and warn.
- `inv_v_sq` uses `max(velocity,1e-4)` but `v_sq = velocity**2` uses the raw value; use one `v_eff`.
- [solver.py:785](file:///home/db/Software/pysole/src/pysole/solver.py#L785): `pts[:,3]*v_eff if pts[:,3].max() > 15 else pts[:,3]` is a magic-number guess of units. The data type is already known (depth returned earlier), so this branch always holds travel times. Remove the heuristic.
- Algebra check: `A = cos²α_y cos²α_x + sin²α_y cos²α_x + sin²α_x cos²α_y = 1 − sin²α_x sin²α_y`, and `(cos²α_y + sin²α_y)/A = 1/A`. Also `dem_grads` already stores `sin/cos(atan(slope))`; the code recomputes `arctan`, `sin`, `cos` over the grid.

### M11. `blend_margin_topography` [CODE]
[interpolation.py:165-174](file:///home/db/Software/pysole/src/pysole/interpolation.py#L165-L174)
- `np.where(weight > 0.8, thickness, smoothed)` creates a step between raw and smoothed thickness at the 0.8 contour. Blend continuously: `w*raw + (1-w)*smoothed`.
- `gaussian_filter(sigma=1.0)` is in pixels, so the smoothing scale changes with resolution.
- `distance_transform_edt(...)*cellsize` ignores anisotropy; use `sampling=(dy, dx)`.
- EDT treats the array border as non-background, so glacier cells at the DEM edge receive full weight.

### M12. Zero-boundary pseudo-points [CODE]
[interpolation.py:611-632](file:///home/db/Software/pysole/src/pysole/interpolation.py#L611-L632)
- `binary_erosion` defaults to `border_value=0`, so glacier pixels on the DEM edge are classed as boundary and receive spurious zero constraints where the glacier actually continues beyond the DEM. Use `border_value=1`.
- The count is hard-coded to ~100 (`stride = len//100`), independent of perimeter length or the kriging range.
- Added points are not de-duplicated against survey picks.

### M13. Raster I/O orientation round trip **[VERIFIED]**
[raster.py:216-220](file:///home/db/Software/pysole/src/pysole/raster.py#L216-L220) vs [raster.py:557-558](file:///home/db/Software/pysole/src/pysole/raster.py#L557-L558)

`BedrockMap.save` writes `tif`/`asc` flipped to top-down, but `csv`/`npy` unflipped (bottom-up). `load_dem` flips **every** file input including csv/npy:
```
round trip marker row:  tif 0 | asc 0 | csv 5 | npy 5     (expected 0)
```
Reloading a saved CSV/NPY mirrors it. Also the `asc` header uses `cellsize = (maxx-minx)/width` only, ignoring `dy`.

### M14. Input validation gaps [CODE]
- `get_drift_functions`: an unknown or misspelled term (`"sai"`) is added to `primitives` and silently produces **no** function. If nothing matches, the fallback is a single `x`-linear drift (no `y`). This silently substitutes an unrelated model. Raise `ValueError` listing the valid terms.
- `kriging_interpolation`: unknown `method` or `variogram_model` falls through to ordinary or spherical with no message. Validate against an enum or literal set.
- `optimize_bss_variance`: no check that `kc_min < kc_max`. `d_kc <= 0` is silently replaced by 0.1.
- `plan_survey`: no checks for `max_length_km <= 0`, `tau_0 <= 0`, or an empty outline mask. `slope_floor_deg` is not exposed through `plan_survey`.
- `random_forest_hole_filling`: NaN cells inside the mask are never selected as holes, because `NaN <= 0.1` is False. It also relies on `ravel()` returning a view for the assignment.

---

## LOW severity

- **Wasted compute:** `compute_gradients` always builds 10 output grids (including curvature, 4 `np.gradient` calls). The BSS loop calls it once per `kc` but uses only `["slope_rad"]`, roughly 3x more work than needed. Add `compute_slope_rad(dem, dx, dy)` and call it in the loop.
- **Silent exception swallowing** at 13 sites: [drift_analyzer.py:231](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py#L231), [233](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py#L233), [269](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py#L269), [271](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py#L271), [371](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py#L371), [408](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py#L408), `raster.py:288/346/612/704/757`, `solver.py:780/849`. Several are `except LinAlgError: pass` followed by `except Exception: pass`, where the first is redundant. `DualKrigingSolver` sets `success=False` and `predict` returns NaN with no log. Count and report failed CV folds.
- **Duplicated variogram code:** the same four model branches appear in `variogram_func` ([interpolation.py:252](file:///home/db/Software/pysole/src/pysole/interpolation.py#L252)), `DualKrigingSolver.__init__`, `predict`, and `spherical_variogram`. Extract one `variogram_model(name, h, nugget, psill, range)`.
- **Docstring/behaviour mismatches:** `calculate_variogram(warn_low_pairs)` documents default False but is True. `compute_cutoff_wavelength` mentions `ds = sqrt(|dx*dy|)` but computes `2π·dx·dy/kc`. `migrate_eikonal` has an unused `plotit` parameter.
- **Cosmetic:** double `[INFO]` prefix in log lines (logger already adds level). The migration progress bar wraps a single vectorised call. RF "trees" progress is a fake 50/50 split.
- **Likely unused imports** (no linter available): `logging` in `drift_analyzer.py` and `survey_planner.py`, `Path` in `migration.py`, `ThreadPoolExecutor`/`Any` usage worth checking.
- **`plan_survey` plotting:** imports `pyplot` at module top and uses global state (`plt.close()`). `plot_path` is built independently of what `_save_figure` actually writes. One test run wrote to `<cwd>/pysole/figures/` (seen in the test log), polluting the repository working directory.
- **Type hints:** `list[Any]` for drift callables (use a `Callable[..., np.ndarray]` alias or Protocol), `crs: Any`, `recommend_drift_model -> list[Any]` (should be `list[CandidateDriftResult]`), and a dict-of-strings return from `plan_survey` (a `dataclass` would be clearer).
- **`blend_margin_topography` ambiguity:** `bedrock_input.shape == (M, N)` distinguishes grid from point array, which is ambiguous if the point count equals `M` and `N == 4`.

---

## Design suggestions

| # | Suggestion | Resolves |
|---|---|---|
| D1 | `DriftBasis` class: evaluate and z-score covariates once with fixed global statistics. Provide `design_matrix(idx)` and `evaluate(x, y)`. Share between analyzer and production kriging. | H1, H2, M6, M7 |
| D2 | `VariogramParams` dataclass (model, nugget, psill, range) fitted on drift-detrended residuals and passed to all kriging paths. | H6, M3, M7, Low duplication |
| D3 | One orientation convention in `GridGeometry`: `row_of(y)`, `col_of(x)`, `to_internal(top_down_array)`, `to_file_order(array)`. All I/O and mask code goes through it. | H1, H3, H5, M13 |
| D4 | Specify the smoothing cutoff as a **wavelength in metres** (`2π·fftfreq(N, d=dx)`) with reflect padding. | M1, M2 |
| D5 | Robust variogram estimator (Cressie-Hawkins or MAD-based) with pair declustering, since BSS products are heavy-tailed near margins. | M4 |
| D6 | Survey planner: trace flowlines on the SIA/DEM gradient field instead of a straight column, clip geometry to the budget, and reproject outputs. | H5, H8 |
| D7 | Replace blanket `except Exception` with specific exceptions plus a counter surfaced in the final log summary. | Low |

---

## Suggested regression tests (all four verified defects are directly testable)

1. `test_coords_to_grid_indices_bottom_up`: y_min cell maps to row 0 (H1).
2. `test_dual_kriging_loo_linear_field`: linear field, `linear_xy` drift, LOO RMSE < 3·σ_noise (H2).
3. `test_outline_vector_orientation`: asymmetric south-half polygon gives rows 0..9 true (H3).
4. `test_outline_nan_rings_hole`: outer plus hole returns True fraction ≈ 0.60 (H4).
5. `test_survey_tracks_inside_mask`: every track vertex lies inside the outline mask, and total length ≤ L_max (H5).
6. `test_csv_npy_roundtrip_orientation` (M13).
7. `test_fit_variogram_scale_invariance`: fitted sill scales with γ over 6 orders of magnitude (M3).

## Recommended fix order

1. **H1, H2** (with D1): unblock a meaningful DriftAnalyzer for v0.4.0.
2. **H3, H4, M13** (with D3): orientation and outline correctness affect every run that uses a vector outline.
3. **H5, H8**: survey planner, before anyone takes the GPX into the field.
4. **H6, H7, M7** (with D2): scientific consistency and honest uncertainty.
5. Remaining Medium and Low items as time permits.

> [!NOTE]
> I did not review `optimize_bss`, `calculate_bedrock`, `finalize_bedrock`, or `plotting.py`/`config.py`/`pipeline.py` in depth. Say so if you want a second pass over those.
