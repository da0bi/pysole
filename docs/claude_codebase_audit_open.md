# PySole Codebase Audit - Open & Remaining Items

**Scope:** Un-implemented numerical, architectural, and design recommendations from `docs/claude_codebase_audit.md`.  
**Status:** Audit report of open items remaining for future releases (post-v0.4.1).

---

## Executive Summary of Completed vs. Open Items

The primary high-priority defects (**H1, H2, H3, H4, H5, H8, M12**) affecting spatial grid alignment, top-down vs. bottom-up outline orientation, GPX/GeoJSON reprojection, Spearman correlation integration, and cross-validation feature standardization have been **fully resolved** in `v0.4.1`.

The items detailed below represent the **remaining open items**, categorized by severity level and functional domain.

---

## HIGH Severity (Remaining Open Items)

### H6. Kriging variogram parameter fitting in final interpolation
**Location:** [interpolation.py](file:///home/db/Software/pysole/src/pysole/interpolation.py)

- `built_in_kriging_interpolation` defaults to heuristic variogram parameters: `range_a = 0.6 * max pairwise distance`, `sill = var(z)`, and `nugget = 0`.
- For non-stationary targets (bedrock with linear/SIA trend), `var(z)` includes trend variance, inflating the sill.
- As a result, the uncertainty grids (`kriged_std`) are uncalibrated estimates.
- **Recommended Fix:** Pass fitted `VariogramParams(model, nugget, psill, range)` (fitted on drift-detrended residuals) directly into `kriging_interpolation`.

### H7. Product-target slope clamping asymmetry (Low-Slope Bias)
**Location:** [solver.py](file:///home/db/Software/pysole/src/pysole/solver.py)

- Forward transformation: $P = D \cdot \max(\sin\alpha, 10^{-4})$.
- Inverse transformation: $D = P / \max(\sin\alpha, \sin(\text{slope\_floor\_deg}))$, with default floor $5^\circ$.
- At low slope angles ($\alpha = 2^\circ$), forward product uses $\sin 2^\circ$, but back-transformation divides by $\sin 5^\circ$, causing thickness underestimation at low-slope glacier interiors ($D \cdot \sin 2^\circ / \sin 5^\circ \approx 0.40 D$).
- **Recommended Fix:** Enforce identical slope floor clamping thresholds in both forward product generation and inverse thickness recovery.

---

## MEDIUM Severity (Remaining Open Items)

### M1. Wavenumber cutoff $k_c$ scaling with $\mathrm{d}x$
**Location:** [smoothing.py](file:///home/db/Software/pysole/src/pysole/smoothing.py), [variogram.py](file:///home/db/Software/pysole/src/pysole/variogram.py)

- `kx = fftshift(fftfreq(N)) * 2 * pi * dx` multiplies by $\mathrm{d}x$, whereas physical wavenumber requires division ($2\pi / \mathrm{d}x$).
- While internally consistent with `compute_cutoff_wavelength`, the default $k_c = 0.5$ produces different physical cutoff wavelengths across varying grid resolutions (e.g. $50\text{ m}$ for $\mathrm{d}x=2\text{ m}$ vs. $1257\text{ m}$ for $\mathrm{d}x=10\text{ m}$).
- **Recommended Fix:** Express cutoff parameters explicitly as physical wavelength in meters $\lambda_c$ ($2\pi / k_c$).

### M2. FFT smoothing boundary padding & NaN handling
**Location:** [smoothing.py](file:///home/db/Software/pysole/src/pysole/smoothing.py)

- FFT assumes periodic boundary conditions, causing opposite DEM edges to bleed into each other.
- NaNs inside/outside glacier masks are mean-filled, creating sharp step edges at margins that introduce ringing artifacts.
- **Recommended Fix:** Implement reflect/edge padding (by $\approx 3\sigma$) and normalized convolution ($\text{smooth}(data \cdot mask) / \text{smooth}(mask)$).

### M3. `fit_variogram_model` scale sensitivity & model selection
**Location:** [variogram.py](file:///home/db/Software/pysole/src/pysole/variogram.py)

- `var_val = np.var(semivars)` uses variance of semivariance values ($[units]^4$) rather than variance of data ($[units]^2$), making sill optimization bounds scale-dependent.
- `model_type` string is accepted but fitting currently evaluates spherical variograms.
- **Recommended Fix:** Base initial sill guess $S_0$ on average tail semivariance $\text{mean}(\gamma_{\text{last third}})$, and support exponential/gaussian fitting dynamically.

### M4. Memory footprint of `calculate_variogram` for large pick sets
**Location:** [variogram.py](file:///home/db/Software/pysole/src/pysole/variogram.py)

- Pairwise distance matrices (`pdist`, `sq_diffs`, `bin_indices`) scale as $O(N^2)$ memory ($N(N-1)/2$ entries). For $N=20,000$ points, memory usage exceeds 5 GB.
- **Recommended Fix:** Process pairwise distance binning in chunked row blocks using `np.bincount` accumulators or spatial subsampling.

### M5. Dual Kriging matrix inversion & variance optimization
**Location:** [interpolation.py](file:///home/db/Software/pysole/src/pysole/interpolation.py)

- An explicit matrix inverse `K_inv` is calculated even when variance grids are not requested.
- Regularization $\lambda \cdot I$ is applied across the entire augmented matrix (including Lagrange blocks).
- **Recommended Fix:** Use `lu_solve(lu, K_rhs)` per grid chunk, make variance evaluation optional (`return_variance=False`), and apply regularizer relative to sill magnitude.

### M6. Recomputation of DEM curvature in Drift Analyzer
**Location:** [drift_analyzer.py](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py), [interpolation.py](file:///home/db/Software/pysole/src/pysole/interpolation.py)

- `eval_curvature_dem` recalculates `compute_surface_curvature(dem)` across the full M×N grid on every `predict()` call during cross-validation folds.
- **Recommended Fix:** Cache grid derivative maps (elevation, slope, curvature) once upon analyzer initialization.

### M7. Parameter alignment between Drift Analyzer and production Solver
**Location:** [drift_analyzer.py](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py), [solver.py](file:///home/db/Software/pysole/src/pysole/solver.py)

- Discrepancies exist between analyzer default settings (e.g. SIA slope floor $1^\circ$, un-smoothed curvature) and production `Solver` settings (slope floor $5^\circ$, $k_c$-smoothed curvature).
- **Recommended Fix:** Standardize feature generation primitives between `DriftAnalyzer` and `Solver` pipelines.

### M8. AICc comparability across varying profile fold counts
**Location:** [drift_analyzer.py](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py)

- LOPO-CV and Spatial Buffer LOOCV may yield slightly different valid sample counts $N_{\text{valid}}$ across candidate models if singular matrix folds are skipped.
- **Recommended Fix:** Restrict AICc comparison strictly to the common intersection of successfully predicted points across all candidate models.

### M10. Eikonal 3D migration edge cases
**Location:** [migration.py](file:///home/db/Software/pysole/src/pysole/migration.py)

- Traveltime gradient NaNs near glacier boundaries fall back to unmigrated depths without logging count.
- Evanescent wave conditions ($|s_h| > 1/v$) are clamped silently.
- **Recommended Fix:** Add diagnostic logging for margin fallback counts and evanescent wave clamping triggers.

### M11. Continuous margin blending in `blend_margin_topography`
**Location:** [interpolation.py](file:///home/db/Software/pysole/src/pysole/interpolation.py)

- Thresholding `np.where(weight > 0.8, thickness, smoothed)` creates a minor step at the 0.8 weight contour.
- `gaussian_filter(sigma=1.0)` operates in pixel units rather than metric spatial distances.
- **Recommended Fix:** Replace step threshold with linear/logistic continuous weight blending $w \cdot H_{\text{raw}} + (1-w) \cdot H_{\text{smoothed}}$, and supply physical spatial sampling `(dy, dx)` to distance transforms.

### M13. Raster file format orientation symmetry
**Location:** [raster.py](file:///home/db/Software/pysole/src/pysole/raster.py)

- `BedrockMap.save` exports GeoTIFF/ASC in top-down orientation, while CSV/NPY formats export in internal bottom-up order. `load_dem` flips all input formats uniformly.
- **Recommended Fix:** Standardize export orientation across all raster file formats or add format-aware orientation tags.

### M14. Input validation & error feedback
**Location:** [drift_analyzer.py](file:///home/db/Software/pysole/src/pysole/drift_analyzer.py), [interpolation.py](file:///home/db/Software/pysole/src/pysole/interpolation.py), [survey_planner.py](file:///home/db/Software/pysole/src/pysole/survey_planner.py)

- Unknown or misspelled drift strings fall back to linear $x$-drift without raising explicit error.
- **Recommended Fix:** Raise explicit `ValueError` when invalid drift terms, invalid variogram models, or negative survey planner budgets are supplied.

---

## LOW Severity & Code Quality Enhancements

1. **Gradients Optimization**: Create a lightweight `compute_slope_rad(dem, dx, dy)` function to avoid computing full curvature/aspect grids during $k_c$ variance optimization loops.
2. **Exception Logging**: Replace silent `except Exception: pass` blocks with explicit exception types (`LinAlgError`, `ValueError`) and diagnostic warnings.
3. **Variogram Code Consolidation**: Consolidate spherical, exponential, and gaussian math into a single unified `eval_variogram_model(name, h, nugget, psill, range)` helper.
4. **Type Hints**: Upgrade `list[Any]` return types to explicit protocols or dataclasses (`CandidateDriftResult`, `SurveyPlanResult`).
