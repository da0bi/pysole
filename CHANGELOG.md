# Changelog

All notable changes to `PySole` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.3] - 2026-10-08

### Added / Fixed (Phase 1 Audit & Code Hardening — High Priority Findings)
- **[Parameter Standard] Corner Wavenumber Standardized to $k_c = 0.0314\text{ rad/m}$ ($\lambda_c \approx 200\text{ m}$) (`src/pysole/survey_planner.py`, `src/pysole/config.py`, `src/pysole/solver.py`, `docs/drift_analyzer_&_survey_planner.md`)**:
  - Replaced legacy default $k_c = 0.5\text{ rad/m}$ ($\lambda_c \approx 12.57\text{ m}$) with $k_c = 0.0314\text{ rad/m}$ ($\lambda_c = 2\pi / k_c \approx 200\text{ m}$) across `SurveyPlanner.plan_survey()`, `Solver.plan_survey()`, CLI `--kc` option, and user manual documentation (`N-H7`).
- **[Pipeline Orchestration] Interactive Flags Plumbing & Output Prefix Standard (`src/pysole/pipeline.py`)**:
  - Forwarded `interactive_optimization` and `interactive_migration` configuration flags cleanly into `run_from_config()`, ensuring terminal prompts respect non-interactive environments (`N-H2`).
  - Aligned default predicted bedrock map export file path with canonical `<prefix>_bedrock_elevation_map.tif`.
- **[Survey Planner & Spatial Resolution] Native Spatial Resolution Preservation (`src/pysole/survey_planner.py`)**:
  - Updated `SurveyPlanner.__init__` `dx`/`dy` parameters to default to `None`, ensuring resolution scaling directly preserves native DEM GeoTIFF pixel dimensions instead of falling back to 10m grid spacing.
- **[Solver State & Variogram Plumbing] Configuration Preservation & Profile Column Support (`src/pysole/solver.py`, `src/pysole/raster.py`)**:
  - Prevented `Solver.__init__` from overwriting `self.config` and `self.config_path` to `None` (`N-H5`).
  - Added support for `survey_profile_column` in `load_survey_points()` to extract and retain string/integer profile IDs in column 4 (`pts[:, 4]`) for Leave-One-Profile-Out (LOPO) cross-validation.
  - Plumbed CLI `--drift-analyzer` flag to set active drift analyzer state in `Solver.from_config()`.
  - Dynamically passed fitted variogram parameters (`range`, `sill`) from `self.opt_variogram_params` into `recommend_drift_model()` in `DriftAnalyzer` (`H6`).
- **[Matrix Conditioning & Numerical Stability] Zero-Variance External Drift Filtering (`src/pysole/interpolation.py`)**:
  - Implemented zero-variance spatial check (`u_std < 1e-12`) for external drift rasters in `built_in_kriging_interpolation()`. Flat external grids trigger an explicit warning and skip redundant constant drift column assembly, preventing rank-deficient matrix singular errors and `NaN` outputs in Universal Kriging.
- **[Test Suite & Benchmarking] Non-Interactive TTY Mocking & Test Rigor (`tests/test_audit_regressions.py`, `tests/test_pykrige.py`)**:
  - Updated `test_run_from_config_batch_false` with TTY mocking (`sys.stdin.isatty` returning `False`) for clean automated execution (`N-H1`).
  - Updated `test_pykrige.py` `setUp` DEM grid with realistic spatial slope (`1000.0 + 0.1*X + 0.2*Y`) for robust elevation drift verification.

### Added / Fixed (Phase 2 Audit & Code Hardening — Medium Priority Findings)
- **[Dynamic FFT Reflection Padding] Kernel-Scaled Padding (`src/pysole/smoothing.py`, `src/pysole/variogram.py`)**:
  - Scaled reflection padding in `precompute_fft_grid` dynamically with Gaussian kernel size ($\text{pad\_px} = \lceil 4 \sigma_{\text{px}} \rceil$ where $\sigma_{\text{px}} = 1 / (\sqrt{2}\pi k_c \Delta x)$), eliminating 12–23m boundary distortion artifacts (`N-M3`).
  - Updated variogram frequency sweeps to pass `kc_min_val` into `precompute_fft_grid()`, ensuring padding is pre-sized for the widest Gaussian kernel.
- **[Kriging Solvers & Regularization] LU Decomposition & Point Covariance Regularization (`src/pysole/interpolation.py`)**:
  - Replaced explicit $K^{-1}$ matrix inversions with LU factorizations (`scipy.linalg.lu_factor` / `lu_solve`) (`M5`).
  - Applied relative Tikhonov regularization ($\epsilon \cdot \text{mean}(\text{diag}(K_{\text{pts}}))$) strictly to point covariance matrix `K[:N_pts, :N_pts]`, ensuring exact satisfaction of Lagrange drift constraints $F^T w = 0$ (`N3-M10`).
- **[Thickness Clipping & Boundary Masking] Glacier-Masked Clipping Counts (`src/pysole/solver.py`)**:
  - Evaluated negative thickness clipping count logging (`n_clipped_neg`) strictly inside the active glacier boundary outline mask when present (`N-M7`).
  - Changed default `interactive` flag in `Solver.finalize_bedrock()` from `True` to `False`.
- **[Machine Learning Gap Filling] Non-Negative Prediction Clamping (`src/pysole/interpolation.py`)**:
  - Enforced strict non-negative clamping ($\ge 0$) on Random Forest thickness predictions (`N-M8`).
  - Subsampled training point datasets ($N > 20,000$) for efficient memory-bound ML fitting.
- **[Drift Analyzer Covariates] Curvature Alignment & Target String Indexing (`src/pysole/drift_analyzer.py`)**:
  - Plumbed `slope_floor_deg` into `DriftAnalyzer.__init__` and updated target string indexing (`"T"` for traveltime, `"D"` for depth target) (`N-M10`).
- **[Configuration Ingestion] Pure Deep Dictionary Merging (`src/pysole/config.py`)**:
  - Implemented `_deep_merge_dict` in `load_config()` for pure recursive deep dictionary merging, preserving nested user configuration dictionaries (`N-M11`).
- **[Visualization & Matplotlib] Safe Lazy Backend Selection (`src/pysole/plotting.py`)**:
  - Replaced top-level non-interactive Matplotlib backend override with safe lazy backend selection (`N-M12`).

### Added / Fixed (Phase 3 Audit & Code Hardening — Low & Design Findings)
- **[Solver State Plumbing] Explicit Attribute Initializations (`src/pysole/solver.py`)**:
  - Initialized `self.pre_kriging_points`, `self.migrated_points`, and `self.drift_analyzer` explicitly in `Solver.__init__`, replacing dynamic `getattr` calls with direct attribute access (`N-L2`).
- **[Code Purge] Legacy Dead Code Cleanup (`src/pysole/interpolation.py`, `src/pysole/drift_analyzer.py`)**:
  - Removed unused function `get_drift_functions` from `interpolation.py` and purged its import from `drift_analyzer.py` (`N-L3`).
- **[Input Ingestion & Validation] Strict Outline Validation & Empty Survey Point Guards (`src/pysole/raster.py`, `src/pysole/variogram.py`)**:
  - Updated `load_outline()` to raise an explicit `ValueError` when `outline_path` does not exist or fails to parse (`N-L8`).
  - Added explicit `ValueError` checks in `optimize_bss_variance()` when `survey_points` is empty or contains no valid positive depth measurements (`N-L5`).
- **[Export Formats & Anisotropic Rasters] Automatic GeoTIFF Fallback for Anisotropic ESRI ASCII Grids (`src/pysole/raster.py`)**:
  - Automatically converted `.asc` export requests to GeoTIFF (`.tif`) format with an explanatory warning when raster grid cells are anisotropic ($dx \neq dy$), preventing GIS spatial warping and coordinate distortion.
- **[Spatial Interpolation & Margin Blending] 2D Anisotropic Sigma Tuple (`src/pysole/interpolation.py`)**:
  - Scaled `gaussian_filter` `sigma` as a cell-size scaled 2D tuple `sigma=(sigma_y/dy, sigma_x/dx)` in `blend_margin_topography()`, preserving spatial smoothing geometry across anisotropic pixels (`M11`).

## [0.4.2] - 2026-10-07

### Added / Fixed (Wavelength Metric & Fallback Parameterization)
- **[Wavelength Metric & Fallback Parameterization] Re-implementation of `fft_filter_metric`, `lambda_min`, and `lambda_max` (`src/pysole/config.py`, `src/pysole/variogram.py`, `src/pysole/solver.py`, `pysole.json`, `examples/gok/pysole_gok.json`, `examples/wuk/pysole_wuk.json`)**:
  - Re-introduced `"fft_filter_metric"` (`"wavenumber"` [default] or `"wavelength"`), `"lambda_min"` (default `null`), and `"lambda_max"` (default `null`) into default configuration schemas.
  - Implemented automatic metric resolution: when `fft_filter_metric == "wavelength"` or when `lambda_min`/`lambda_max` are set:
    - If `lambda_min` / `lambda_max` are `null`, `PySole` falls back directly to `kc_max` ($k_{\text{Nyquist}} = \pi / \Delta x_{\text{min}}$) and `kc_min` ($4\pi / L_{\text{max}}$).
    - When specified as numeric values, `lambda_min` / `lambda_max` convert directly to wavenumbers ($k_{c,\text{max}} = 2\pi / \lambda_{\text{min}}$, $k_{c,\text{min}} = 2\pi / \lambda_{\text{max}}$).
  - Simplified `compute_cutoff_wavelength(kc: float)` signature in `src/pysole/variogram.py` to a single-argument helper returning $\lambda_c = 2\pi / k_c$.

### Refactored / Streamlined (Codebase Audit & Architecture Clean-up)
- **[Architecture & Codebase Audit] Code Purge & Module Clean-up (`src/pysole/config.py`, `src/pysole/pipeline.py`, `src/pysole/solver.py`, `src/pysole/interpolation.py`)**:
  - Consolidated `OutputsConfig` dataclass in `config.py` with `save_bedrock_elevation_map: bool = True` and helper properties (`save_ice_thickness_map`, `save_basal_shear_stress_map`).
  - Imported `OutputsConfig` directly from `.config` in `pipeline.py`, removing duplicate dataclass definition and unused imports (`copy`, `json`, `dataclass`, `field`).
  - Streamlined `optimize_bss()` configuration extraction in `solver.py` (removed redundant `hasattr(self, "config")` guards), updated `opt_wavelength` calculations in `Solver` property and `run_pipeline()` to `compute_cutoff_wavelength(opt_kc)`, hoisted `compute_surface_curvature` to top-level, and removed unused import `fft_gaussian_smooth`.
  - Hoisted inner imports (`compute_gradients`, `compute_surface_curvature`) in `interpolation.py` to top-level.

### Documentation & Tests (Section 3 Reordering & Derivation Subsection)
- **[Documentation] README.md Section & Derivation Subsection Updates (`README.md`)**:
  - Updated JSON configuration snippet and Parameter Reference table with `"fft_filter_metric"`, `"lambda_min"`, and `"lambda_max"`.
  - Added a dedicated subsection detailing dynamic derivations for default parameters `kc_max`, `kc_min`, `lambda_min`, `lambda_max`, and `n_steps`.
  - Reordered "Wavenumbers and Wavelengths" as subsection 3 (after subsection 2 "Parsing of Rock Outcrop Input Files") under "Technical & Methodological Notes", renumbering remaining subsections 4 through 9 accordingly.
- **[Unit Tests] Parameter Defaults Unit Tests (`tests/test_parameter_defaults.py`)**:
  - Added `test_lambda_fallback_when_null` and `test_lambda_min_max_conversion` verifying fallback behaviors and direct wavenumber conversion logic.

### Added / Fixed (Dynamic BSS Spectrum Parameterization & Legacy Purge)
- **[BSS Parameterization] Fourier Mode Step Count & Half-Domain Limit Default (`src/pysole/variogram.py`, `src/pysole/config.py`, `src/pysole/solver.py`)**:
  - Replaced legacy stepwidth parameters (`d_kc`, `d_lambda`, `lambda_min`, `lambda_max`) with `n_steps` parameter.
  - Implemented Strategy 2 Half-Domain domain-scaling limit for default $k_{c,\text{min}} = \frac{4\pi}{L_{\text{max}}}$ (where $L_{\text{max}} = \max(M \cdot |dy|, N \cdot |dx|)$), setting $\lambda_{\text{max}} = \frac{L_{\text{max}}}{2}$.
  - Derived dynamic default `n_steps` from discrete Fourier mode resolution ($n_{\text{modes}} = \lfloor (k_{\text{max}} - k_{\text{min}}) \cdot L_{\text{max}} / 2\pi \rfloor$), clipped to $[10, 50]$.
  - Completely purged legacy parameters (`d_kc`, `d_lambda`, `lambda_min`, `lambda_max`) with zero backward compatibility across `BSSOptimizer`, `Solver`, `pipeline.py`, JSON templates, CLI parsers, and documentation.
- **[Documentation] README.md Parameter & Schema Alignment (`README.md`)**:
  - Updated JSON configuration snippet and Parameter Reference table for `optimization_parameters` in `README.md`.

### Added / Fixed (Phase 3 Audit Resolutions — 2nd Pass Review: Low Priority)
- **[M14, N-M6, N-L8] Strict Input Parameter Validation & Outline Failure Guard (`src/pysole/interpolation.py`, `src/pysole/variogram.py`, `src/pysole/raster.py`)**:
  - Implemented strict input parameter validation in `kriging_interpolation()` for `method` (`"ordinary"`, `"universal"`, `"regression"`), `variogram_model` (`"spherical"`, `"exponential"`, `"gaussian"`), and `engine` (`"native"`, `"pykrige"` or `KrigingEngine` instance), raising explicit `ValueError` on invalid strings.
  - Added search range validation in `optimize_bss_variance()` enforcing $k_{c,\text{min}} < k_{c,\text{max}}$ before executing frequency sweeps.
  - Updated `load_outline()` to raise an explicit `ValueError` when a provided `outline_path` fails to load, replacing the silent fallback to unconstrained DEM domain rectangle.
- **[N-M11, N-L1] CLI & Config Parameter Alignment (`src/pysole/config.py`, `src/pysole/pipeline.py`, `src/pysole/solver.py`)**:
  - Forwarded `--drift-analyzer` and `--profile-col` CLI flags from `main_cli()` into `run_from_config()` and set `drift_analyzer` and `survey_profile_column` on `Solver`.
  - Aligned default parameters between `Solver.__init__` and `config.py` default schema dictionary.
- **[N-M12] Cell-Edge Extent Rendering, Histogram Mean & Figure Prefixes (`src/pysole/plotting.py`)**:
  - Standardized Matplotlib spatial extent boundaries using `_get_cell_edge_extent(x_coords, y_coords)` (`[minx - dx/2, maxx + dx/2, miny - dy/2, maxy + dy/2]`) across figure generators to eliminate 0.5-pixel image shift.
  - Updated `plot_final_ice_thickness_histogram()` mean calculation to compute average depth strictly on active ice cells ($D > 0$).
  - Standardized stage filename prefixes (`01_`, `03_`) for diagnostic plots to prevent stage-1 and stage-2 figures from overwriting each other.
- **[Test Gap] Comprehensive Audit Regression Unit Test Suite (`tests/test_audit_regressions.py`)**:
  - Created dedicated unit test file containing 12 unit tests verifying all 7 original audit regression test cases (`test_coords_to_grid_indices_bottom_up`, `test_dual_kriging_loo_linear_field`, `test_outline_vector_orientation`, `test_outline_nan_rings_hole`, `test_survey_tracks_inside_mask`, `test_csv_npy_roundtrip_orientation`, `test_fit_variogram_scale_invariance`) as well as Phase 1, 2, and 3 fixes (`test_run_from_config_non_interactive_batch`, `test_fft_normalized_convolution_nan_boundary`, `test_utm_drift_basis_centering_condition_number`, `test_compound_drift_term_expansion`, `test_eikonal_migration_oblique_surface_accuracy`).

### Added / Fixed (Phase 2 Audit Resolutions — 2nd Pass Review: Medium Priority)
- **[N-M9, M10] Exact 3D Eikonal Ray Migration (`src/pysole/migration.py`)**:
  - Replaced non-orthogonal approximation ($s_3 = \sqrt{v^{-2} - s_1^2 - s_2^2}$) with exact 3D closed-form slowness vector along non-horizontal surface normal $\vec{n} = (-z_x, -z_y, 1)^T$:
    \[
    A = 1 + z_x^2 + z_y^2, \quad B = s_1 z_x + s_2 z_y, \quad C = s_1^2 + s_2^2 - v^{-2}, \quad p_z = \frac{B - \sqrt{B^2 - AC}}{A}
    \]
  - Added surface elevation delta offset $(Z_{\text{smig}} - Z_{\text{s0}})$ to relocated point elevation ($Z_{\text{mig}} = Z_{\text{s0}} - \Delta z + (Z_{\text{smig}} - Z_{\text{s0}})$), eliminating 6–19% displacement vector errors on oblique surface slopes ($z_x \neq 0, z_y \neq 0$).
- **[N-M3, M2] Dynamic FFT Reflection Padding (`src/pysole/smoothing.py`)**:
  - Scaled reflection padding dynamically with kernel standard deviation ($\text{pad\_px} \ge 4 \sigma_{\text{px}}$ where $\sigma_{\text{px}} = 1 / (\sqrt{2}\pi k_c \cdot \Delta x)$), preventing edge ring and boundary distortion artifacts on wide Gaussian low-pass kernels.
- **[N-M1, N-M2, M13] Raster Resampling, Outline Transform & ASCII Grid Offsets (`src/pysole/raster.py`)**:
  - Modified `resample_dem` to apply nearest-neighbor filling strictly to out-of-bounds cells while preserving interior `NaN` domain pixels.
  - Rebuilt Affine `transform` in `load_outline` directly from `bounds` & `out_shape` to prevent spatial alignment drift when rasterizing polygon outlines.
  - Corrected ASCII grid loader `xllcenter` / `yllcenter` corner offset calculation ($x_{\text{corner}} = x_{\text{center}} - \mathrm{d}x/2, y_{\text{corner}} = y_{\text{center}} - \mathrm{d}y/2$).
- **[N-M4, N-M5, N-M6, M4, M14] Variogram Binning, Pair Weighting & $O(N^2)$ Memory Optimization (`src/pysole/variogram.py`)**:
  - Fixed lag distance bin count calculation in `calculate_variogram` to count only pairs within `maxdist` ($d \le \text{maxdist}$).
  - Passed pair count weights $\sigma_i = 1 / \sqrt{N_i}$ into `fit_variogram_model`'s `scipy.optimize.curve_fit` call.
  - Optimized `optimize_bss_variance` memory footprint by capping pairwise distance matrix calculation (`pdist`) to $N \le 5000$ points (subsampling when $N > 5000$).
  - Added search range validation enforcing $k_{c,\text{min}} < k_{c,\text{max}}$ and pair-weighted mean variance calculations.
- **[M11, N-M8, M9, M12] Spatial Cell Scaling & RF Ice Thickness Hole Filling (`src/pysole/interpolation.py`)**:
  - Scaled `sigma` kernel in `blend_margin_topography` to metric spatial cell sizes $(dy, dx)$ (`sigma=(sigma/dy, sigma/dx)`).
  - Updated `random_forest_hole_filling` to train and predict on ice thickness $D \ge 0$ (clamping predictions $\ge 0$) rather than absolute bedrock elevation, while excluding margin zero-point rings (`dist_from_margin > margin_threshold`) from feature training sets to eliminate boundary fit distortions.
- **[N-M7, Solver Fixes] FFT Cache Reuse, Clipping Logs & Output Lists (`src/pysole/solver.py`)**:
  - Reused pre-computed FFT smoothed DEM in `smooth_bedrock_dem` to eliminate redundant FFT filtering passes.
  - Fixed `max_thickness` upper boundary handling in ice thickness clipping using `np.nanmax`.
  - Added logging for clipped negative ice thickness cells.
  - Appended `save_basal_shear_stress_uncertainty` output path to `saved` file list in `Solver.run_pipeline`.

### Added / Fixed (Phase 1 Audit Resolutions — 2nd Pass Review: High Priority)
- **[N-H1, N-H2] Entry-Point & Batch CLI Execution Fixes (`src/pysole/config.py`, `src/pysole/pipeline.py`)**:
  - Imported `logger` at top-level in `config.py` to fix `NameError` during `pysole --init`.
  - Imported `sys` and `from typing import Any` in `pipeline.py` to resolve non-interactive batch mode (`sys.stdin.isatty()`) and type annotation errors.
  - Corrected `resolve_path()` positional argument ordering (`output_dir=solver.output_dir`, `config_path=config_file`).
- **[H6] Fitted Variogram Parameter Propagation (`src/pysole/variogram.py`, `src/pysole/solver.py`)**:
  - Added `opt_variogram_params` (`range`, `sill`, `nugget`) attribute to `OptimizationResult` in `variogram.py`.
  - Stored `self.opt_variogram_params` in `Solver` during BSS optimization and explicitly passed `variogram_params=self.opt_variogram_params` into downstream pre- and post-migration `kriging_interpolation()` passes.
- **[M5] Dual Kriging Matrix Regularization (`src/pysole/interpolation.py`)**:
  - Constrained diagonal Tikhonov matrix regularization strictly to the point-point covariance block $K_{:N, :N}$ (`K[:N_pts, :N_pts] += reg_val * I`), leaving the Lagrange drift block $F$ unperturbed to guarantee exact drift constraint satisfaction.
- **[N-H3] Normalized Convolution & Edge Elevation Fix (`src/pysole/smoothing.py`)**:
  - Replaced non-zero NaN padding with zero-filling (`grid_clean = np.where(nan_mask, 0.0, grid)`) in `smoothing.py`, ensuring exact normalized spatial convolution $S(D \cdot M) / S(M)$, restoring NaNs after filtering, and eliminating +88% (+2738m) edge elevation artifacts.
- **[N-H4] Pre-Centered Quadratic UTM Drift Basis (`src/pysole/interpolation.py`)**:
  - Updated `DriftBasis` quadratic polynomial terms to center coordinates prior to exponentiation ($((X - m_X)/s_X)^2$), drastically improving the Dual Kriging system condition number from $\sim 8 \cdot 10^{12}$ to $\sim 15$.
- **[N-H5] LOPO-CV Profile Preservation (`src/pysole/solver.py`)**:
  - Preserved `self.survey_profile_column` during `Solver` initialization and forwarded `profile_data` into `DriftAnalyzer.run_diagnostics()`.
- **[N-H6] Compound Drift Term Expansion (`src/pysole/interpolation.py`)**:
  - Implemented primitive drift term expansion (`expanded_primitives`) in `kriging_interpolation()` using `DriftBasis.SUPPORTED_TERMS` to decompose compound drift specifications (`"sia_z_dem"`, `"full_physical"`) into valid primitive basis functions.
- **[N-H7] Default Cutoff Wavelength Scaling (`src/pysole/config.py`, `src/pysole/survey_planner.py`)**:
  - Standardized default low-pass cutoff wavenumber to $k_c = 0.0314$ rad/m ($\lambda_c \approx 200$ meters) across CLI parsers and unprobed survey planning routines.
- **Solver Attribute Safety (`src/pysole/solver.py`)**:
  - Initialized `self.final_grid` and `self.bss_std` attributes in `Solver.__init__` to prevent `AttributeError` when querying `model.results`.

### Added / Fixed (1st Pass Code Audit Resolutions — Corrected & Re-aligned Audit IDs)
- **[H1] Spatial Grid Alignment (`src/pysole/raster.py`, `src/pysole/interpolation.py`)**:
  - Standardized `GridGeometry` bounds, resolution verification, and spatial coordinate-to-grid index mapping (`coords_to_grid_indices(x, y)`), eliminating Y-coordinate orientation mirroring.
- **[H2] Fixed Drift Basis Normalization (`src/pysole/interpolation.py`)**:
  - Fixed `DriftBasis` normalization once per dataset across interpolation calls, stabilizing Leave-One-Out cross-validation RMSE from 163 to 1.12.
- **[H3] Vector Outline Orientation (`src/pysole/raster.py`)**:
  - Unified top-down vs. bottom-up raster coordinate orientations and CRS alignment checks during vector outline rasterization.
- **[H4] NaN-Ring Boundary Outlines (`src/pysole/raster.py`)**:
  - Handled outer glacier boundaries and internal rock outcrop holes correctly, delineating active domain masks without ignoring boundary constraints.
- **[H5] Survey Planner Boundary & Track Budgeting (`src/pysole/survey_planner.py`)**:
  - Applied polygon mask bounds and track length budgeting ($L_{\text{max}}$) in survey track generation.
- **[H7] Symmetric Slope Clamping (`src/pysole/solver.py`)**:
  - Standardized symmetric slope floor clamping (`slope_floor_deg`) across pre- and post-migration BSS slope optimization passes.
- **[H8] Vector Track CRS Reprojection (`src/pysole/survey_planner.py`)**:
  - Integrated `pyproj` reprojection for output GPX and GeoJSON survey tracks to ensure export in WGS84 geographic coordinates.
- **[M1] Physical Wavenumber ($k_c$ [rad/m]) & Spatial Wavelength ($\lambda_c$ [m]) Dual Parameterization (`src/pysole/smoothing.py`, `src/pysole/config.py`)**:
  - Corrected spatial wavenumber calculation in `smoothing.py` ($k_x = \frac{2\pi \cdot \text{fftfreq}(N)}{\mathrm{d}x}$ [rad/m]) for 100% grid resolution invariance.
  - Added dual parameter configuration (`kc_min`, `kc_max` and `lambda_min`, `lambda_max`) and Nyquist limit validation ($k_{\text{Nyquist}} = \pi / \min(\mathrm{d}x, \mathrm{d}y)$).
- **[M3] Scale-Invariant Variogram Model Fitting (`src/pysole/variogram.py`)**:
  - Implemented tail-semivariance initial sill estimation $S_0 = \text{mean}(\gamma_{\text{tail}})$ in `fit_variogram_model()` for scale-independent fitting across metric and normalized data.
- **[M6] Drift Matrix Pre-Caching (`src/pysole/drift_analyzer.py`)**:
  - Pre-cached curvature and SIA drift grids upon `DriftBasis` initialization to accelerate multi-drift evaluation sweeps.
- **[M7] Drift Analyzer Production Feature Alignment (`src/pysole/drift_analyzer.py`)**:
  - Aligned primitive feature parameters (SIA slope floor, curvature smoothing) with production `Solver` defaults.
- **[M8] Common-Intersection AICc Comparability (`src/pysole/drift_analyzer.py`)**:
  - Implemented common-intersection valid sample mask for fair AICc comparisons across candidate drift models.
- **[M13, M14] Explicit Input Validation & Raster Symmetry (`src/pysole/raster.py`, `src/pysole/survey_planner.py`)**:
  - Standardized coordinate orientation symmetry across raster file exports (GeoTIFF, ASCII Grid, CSV, NPY).
  - Raised explicit `ValueError` exceptions for invalid drift model names, unrecognized variogram models, and negative survey planner length budgets.

## [0.4.1] - 2026-10-03

### Added
- **Architectural Decoupling (`src/pysole/pipeline.py`)**:
  - Introduced `PipelineManager`, `PipelineExporter`, and `PipelineConfig` in `src/pysole/pipeline.py` to decouple JSON configuration parsing, workspace path resolution, and disk export management from `Solver`.
  - Streamlined `src/pysole/config.py` into a lightweight CLI parser and default dictionary manager (~150 lines).
  - Re-exported `run_from_config` and `OutputsConfig` in `src/pysole/__init__.py` from `pipeline.py` to maintain 100% backward compatibility for all existing 1-line Python API scripts.
- **Code Streamlining & Helper Utilities**:
  - Added `GridGeometry.coords_to_grid_indices(x, y)` to `src/pysole/raster.py` to consolidate spatial coordinate-to-pixel index transformations across `interpolation.py`, `drift_analyzer.py`, and `survey_planner.py`.
  - Centralized figure exports in `src/pysole/survey_planner.py` using `pysole.plotting._save_figure()`.
- **Comprehensive API Documentation (`Examples:` Docstrings)**:
  - Added explicit NumPy/Google-style `Examples:` docstring blocks to all public functions and classes across `pysole.run_from_config()`, `Solver`, `DriftAnalyzer`, `SurveyPlanner`, `migrate_eikonal_points`, `optimize_bss_variance`, `kriging_interpolation`, `blend_margin_topography`, `load_dem`, and `BedrockMap`.
- **Wurtenkees (WUK) Tutorial Notebook (`examples/pysole_quickstart.ipynb`)**:
  - Created interactive Jupyter Notebook tutorial demonstrating high-level API execution, `DriftAnalyzer` model selection, and bedrock elevation grid inspection on the Wurtenkees (WUK) benchmark dataset.
- **JOSS & PyPI Publication Infrastructure**:
  - Added GitHub Actions CI workflow (`.github/workflows/ci.yml`) testing the package across Python 3.10, 3.11, 3.12, and 3.13.
  - Added GitHub Actions PyPI release workflow (`.github/workflows/publish-pypi.yml`) for automated release publishing via OIDC Trusted Publisher.
  - Added `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1).

### Refactored
- **Codebase Streamlining & Audit**: Audited codebase to eliminate redundant backward-compatibility layers and obsolete code. Consolidated spatial coordinate index calculations (`coords_to_grid_indices`) in `GridGeometry`, standardized `from_config` survey profile column parsing, and streamlined `DualKrigingSolver` candidate drift function lookups.
- **`Solver` Engine & Method Canonicalization (`src/pysole/solver.py`)**:
  - Consolidated `calculate_bedrock()` and `finalize_bedrock()` as the primary canonical methods on `Solver`, keeping `interpolate_kriging()` and `finalize_topography()` as transparent aliases.
  - Purged obsolete legacy aliases (`compute_eikonal_migration()`, `calculate_topography()`, `solve_kriging()`).
- **Drift Analyzer Multi-Variable VIF (`src/pysole/drift_analyzer.py`)**:
  - Upgraded `calculate_vif()` to execute exact multi-variable linear least-squares regression with an intercept column across target drift matrices, replacing single-variable linear approximations.
- **Survey Planner & Pipeline Key Standardization (`src/pysole/survey_planner.py`, `src/pysole/pipeline.py`)**:
  - Standardized dictionary return keys for SIA ice thickness output grids (`"sia_modelled_depth"`) across `SurveyPlanner` and `PipelineManager`.

## [0.4.0] - 2026-10-03

### Added
- **Universal Kriging Drift Analyzer (`src/pysole/drift_analyzer.py`)**:
  - Implemented automated recommendation and diagnostic engine for single and multi-drift Universal Kriging models (`sia`, `z_dem`, `curvature_dem`, `linear_xy`, `quadratic_xy`, and compound physical combinations).
  - Added Variance Inflation Factor (VIF) auditing with exact linear least-squares $R_j^2$ decomposition to detect and penalize multicollinear drift vectors ($\text{VIF} > 10$).
  - Integrated non-parametric Spearman rank correlation ($\rho_s$) alongside parametric linear Pearson correlation ($r$) and Random Forest Permutation Feature Importance calculation to detect non-linear monotonic trends and evaluate feature transformation potential.
  - Implemented Spatial Cross-Validation: automatically executes **Leave-One-Profile-Out (LOPO-CV)** when `inputs.survey_profile_column` is provided ($N_{\text{profiles}} \ge 3$), with fallback to **Spatial Buffer LOOCV** with spatial exclusion prompt in **meters** based on pre-fitted variogram range $a_0$.
  - Implemented glaciological safeguard warnings when SIA slope drift is evaluated on product targets $P(x,y)$ to prevent $1/\sin^2\alpha$ double-scaling margin artifacts.
  - Added terminal ranking table sorted by AICc/RMSE and interactive drift selection prompt.
- **Forward Unprobed Glacier Survey Planner (`src/pysole/survey_planner.py`)**:
  - Implemented forward survey campaign design mode, automatically dispatched when `dem_path` and `outline_path` are defined without `survey_data_path`.
  - Calculates synthetic Shallow Ice Approximation (SIA) ice thickness grid $D_{\text{SIA}}(x,y) = \frac{\tau_0}{\rho g \sin \bar{\alpha}_{\text{opt}}}$.
  - Generates optimal longitudinal central flowline tracks and transverse cross-profile tracks subject to maximum track length budget $L_{\text{max}}$.
  - Exports synthetic SIA raster map `<output_prefix>_sia_modelled_depth.<ext>`, vector tracks `<output_prefix>_survey_plan.gpx` and `<output_prefix>_survey_plan.geojson`, and diagnostic map plot `<plots_dir>/<output_prefix>_survey_plan_map.png`.
- **Non-Interactive Batch Execution & CLI Enhancements**:
  - Added `--batch` / `--non-interactive` CLI flags to automatically disable interactive terminal prompts when running PySole in batch processing scripts or non-TTY environments (`not sys.stdin.isatty()`).
  - Added `--drift-analyzer` and `--profile-col` CLI parameters to `main_cli()`.
  - Added `plan-survey` CLI subcommand for standalone campaign planning.
  - Implemented Ordinary Kriging override rule: setting `method: "ordinary"` automatically forces `drift_analyzer = false`.
- **Package Architecture Diagram (`scripts/plot_package_structure.py`)**:
  - Standardized inter-card vertical gap spacing to an exact uniform **2.2 Y-units** across Column 2 and Column 3.
  - Perfectly aligned the lower card margins of Column 1 (`pysole.solver`), Column 2 (`pysole.survey_planner`), and Column 3 (`pysole.drift_analyzer`) at **$Y = 2.9$**.
  - Styled `pysole.drift_analyzer` card border and orchestrator method `recommend_drift_model(...)` highlights with **Bright Fuchsia** (`#e879f9`) for consistent visual mapping across the diagram.
  - Increased font size and vertical line spacing in `pysole.solver` orchestrator card by **+20%**.
- **Documentation & Methodological Reference**:
  - Expanded [`docs/drift_analyzer_&_survey_planner.md`](file:///home/db/Software/pysole/docs/drift_analyzer_&_survey_planner.md) with comprehensive equations for Universal Kriging, VIF, Pearson $r$, Spearman $\rho_s$, Spatial Buffer LOOCV, AICc, SIA depth modeling, and survey track layout algorithms, including a tri-metric diagnostic evaluation matrix table.
  - Created [`docs/drift_analyzer.md`](file:///home/db/Software/pysole/docs/drift_analyzer.md) pointing to the comprehensive guide.
  - Updated JSON configuration templates ([`pysole.json`](file:///home/db/Software/pysole/pysole.json), [`examples/gok/pysole_gok.json`](file:///home/db/Software/pysole/examples/gok/pysole_gok.json), [`examples/wuk/pysole_wuk.json`](file:///home/db/Software/pysole/examples/wuk/pysole_wuk.json)) with `"survey_profile_column"` and `"drift_analyzer"`.
  - Updated [`README.md`](file:///home/db/Software/pysole/README.md) key features, JSON configuration snippet, and Parameter Reference table.
  - Created dedicated unit test suites in [`tests/test_drift_analyzer.py`](file:///home/db/Software/pysole/tests/test_drift_analyzer.py) and [`tests/test_survey_planner.py`](file:///home/db/Software/pysole/tests/test_survey_planner.py).

### Changed
- **Dual Kriging Spatial Cross-Validation (`src/pysole/interpolation.py`)**:
  - Generalized `DualKrigingSolver` to evaluate all physical DEM drift models (`sia`, `z_dem`, `curvature_dem`, and compound physical/spatial combinations) during spatial cross-validation (`predict_validation()`).
  - Refactored `get_drift_functions()` to decompose compound drift specifications into canonical primitives (`sia`, `z_dem`, `curvature_dem`, `linear_xy`, `quadratic_xy`).
- **Codebase Audit & Streamlining**:
  - Thoroughly audited the codebase and removed obsolete backward compatibility fallbacks and redundant drift evaluation code paths.
  - Ensured strict canonical drift model naming across all modules (`solver.py`, `interpolation.py`, `drift_analyzer.py`, `survey_planner.py`).

---

## [0.3.2] - 2026-10-01

### Added
- **Unified DEM Smoothing Architecture & Curvature Drift Model**:
  - Implemented single-pass FFT Gaussian low-pass smoothing directly on the raw surface DEM $Z_{\text{dem}}(x,y)$ to compute $Z_{\text{smooth}, k_c}(x,y)$ once.
  - Derived both optimal surface slope ($\sin\alpha_{\text{opt}}$) and 2D Laplacian surface curvature ($\kappa_{k_c} = \nabla^2 Z_{\text{smooth}, k_c}$) from the single smoothed DEM, ensuring 100% geomorphological consistency across slope, curvature, and elevation fields while eliminating redundant code paths and double-smoothing computations.
  - Preserved exact raw DEM elevation embedding outside glacier boundary outlines during final bedrock construction.
- **Universal Kriging Drift Model Naming Standardization**:
  - Standardized all Universal Kriging drift model names across PySole to **5 canonical identifiers**:
    - `"sia"`: Shallow Ice Approximation Physical Drift ($U_{\text{sia}} = \sin(\alpha_{\text{opt}})^{-1}$)
    - `"z_dem"`: Surface Elevation Drift ($U_z = Z_{\text{dem}}$)
    - `"curvature_dem"`: Surface Curvature Drift ($U_{\kappa} = \nabla^2 Z_{\text{smooth, } k_c}$)
    - `"linear_xy"`: 1st-Order Linear Spatial Coordinate Drift ($a_1 X + a_2 Y$)
    - `"quadratic_xy"`: 2nd-Order Quadratic Spatial Coordinate Drift ($a_1 X + a_2 Y + a_3 X^2 + a_4 Y^2 + a_5 XY$)
  - Completely purged legacy alias fallback strings (`"sia_thickness"`, `"sia_drift"`, `"z_surface"`, `"dem"`, `"elevation"`, `"surface_curvature"`, `"curvature"`, `"laplacian"`, `"regional_linear"`, `"linear"`, `"quadratic"`).
- **Surface Curvature Unit Test Module (`tests/test_curvature.py`)**:
  - Added unit test suite verifying 2nd-order central difference Laplacian curvature on synthetic paraboloids, `compute_gradients()` output dictionary contents, native Dual Kriging with `"curvature_dem"`, combined multi-drift models, and exact raw DEM embedding preservation.

### Changed
- **Codebase & Architecture Streamlining**:
  - Updated `built_in_kriging_interpolation()`, `pykrige_kriging_interpolation()`, `kriging_interpolation()`, `Solver`, and `_execute_kriging_pass()` to enforce canonical drift identifiers exclusively.
  - Streamlined `compute_gradients()` in `src/pysole/smoothing.py` to accept pre-smoothed DEMs directly without redundant internal smoothing logic.
- **Documentation & Configuration Alignment**:
  - Updated `README.md` parameter reference tables, Section 6 (Universal Kriging Drift Models), multi-drift model combinations, and workflow notes.

---

## [0.3.1] - 2026-09-30

### Added
- **Headerless DEM Loading & Metadata Ingestion**:
  - Extended `load_dem()` in `src/pysole/raster.py` to support headerless DEM inputs (`.csv`, `.npy`, `np.ndarray`).
  - Added automatic detection for 3-column CSV grid tables `(X, Y, Z)`, automatically inferring grid cell resolution (`dx`, `dy`) and bounding extent.
  - Added `origin` (`[xll, yll]`) parameter to `spatial_parameters` for precise positioning of headerless DEMs in projected coordinate space.
- **Projected Metric Coordinate System Validation**:
  - Added `check_projected_metric_crs()` in `src/pysole/raster.py` to enforce that input CRS designations (e.g. `"EPSG:32633"`) use projected metric units (UTM meters) rather than geographic lat/lon degrees (`EPSG:4326`).
  - Added automatic coordinate origin mismatch detection in `Solver` between survey point coordinates and headerless DEM bounding boxes.
- **Dual Kriging Engine Architecture & PyKrige Integration**:
  - Added `kriging_parameters.engine` configuration parameter (`"native"` default vs `"pykrige"`).
  - Implemented `pykrige_kriging_interpolation()` supporting `pykrige.ok.OrdinaryKriging`, `pykrige.uk.UniversalKriging`, and `pykrige.rk.RegressionKriging` (`method: "regression"`, combining `scikit-learn`'s `RandomForestRegressor` with residual Kriging).
  - Added clear `ImportError` feedback when `pykrige` engine is requested but not installed.
- **Native Multi-Drift Universal Kriging Engine**:
  - Added support in `built_in_kriging_interpolation()` for combining external raster drift models (`"z_surface"`, `"sia_thickness"`) with polynomial spatial coordinate trends (`"quadratic"`, `"regional_linear"`) into augmented $n_{\text{drift}} = 7$ or $n_{\text{drift}} = 4$ Dual Kriging matrices.
- **New Unit Test Suite (`tests/test_pykrige.py`)**:
  - Added unit test module covering `engine="native"`, `engine="pykrige"`, `method="regression"`, multi-drift combination (`["z_surface", "quadratic"]`), and `ImportError` handling.

### Changed
- **Codebase Streamlining & Audit**:
  - Consolidated post-extraction metadata initialization for headerless DEMs into a single post-extraction block while preserving C-contiguous array indexing.
  - Removed obsolete legacy fallback flags (`built_in_kriging`).
  - Streamlined memory allocations in `pykrige_kriging_interpolation()` during Regression Kriging predictions.
- **Documentation & Configuration Schema Alignment**:
  - Updated `README.md` parameter reference tables, JSON configuration snippet, and workflow sections with GitHub-style alert callouts.
  - Updated `pysole.json`, `examples/gok/pysole_gok.json`, and `examples/wuk/pysole_wuk.json` with `origin`, `crs`, and `engine` settings.

---

## [0.3.0] - 2026-09-27

### Added
- **Configurable Pre- and Post-Migration Interpolation Target Fields (`"T"`, `"D"`, `"P"`)**:
  - Added `kriging_parameters.pre_migration.interpolation_target` (`"P"` for BSS product field $P_1 = T \cdot \sin\alpha$, default; or `"T"` for direct traveltimes).
  - Added `kriging_parameters.post_migration.interpolation_target` (`"P"` for BSS product field $P = D \cdot \sin\alpha$, default; or `"D"` for direct ice depth/thickness).
  - Implemented automatic target-dependent defaults: Direct `"T"` / `"D"` targets default to **Universal Kriging** with `drift_terms = ["sia_thickness"]`; Product `"P"` target defaults to **Ordinary Kriging** with `drift_terms = []`.
- **Tikhonov Matrix Regularization ($10^{-6} \cdot \mathbf{I}$) & Active User Logging**:
  - Updated diagonal matrix regularization factor from $10^{-8}$ to $10^{-6}$ across the **entire augmented Kriging system matrix $K$** (`K += np.eye(K.shape[0]) * 1e-6`).
  - Added active user notification log message during Kriging matrix factorization:
    `[INFO] [Dual Kriging Engine] Applied 1e-06 * I Tikhonov matrix regularization (N=... points)`
  - Prevents ill-conditioning ($\kappa(K) > 10^{14}$) and `LinAlgWarning: Singular matrix` errors on dense GPR profile networks ($N > 2000$).
- **Unified Pass Dispatcher (`_execute_kriging_pass`) & Slope Cache Reuse**:
  - Refactored Pass 1 (traveltime) and Pass 2 (depth) Kriging calls into a streamlined internal helper method `_execute_kriging_pass()` on `Solver`, eliminating duplicated scaling, clipping, and boundary condition code.
  - Implemented automatic Pass 1 to Pass 2 `opt_slope` FFT cache reuse, saving ~20–30s of redundant variogram calculations during Pass 2 execution.
- **`pysole_interpolation_practice_guide.md` Practice Guide**:
  - Published comprehensive documentation report in `docs/pysole_interpolation_practice_guide.md` containing mathematical derivations of $1/\sin^2\alpha$ double-scaling, head-to-head WUK and GOK benchmark comparisons, and case-dependent decision trees.

### Changed
- Updated `pysole.json`, `examples/wuk/pysole_wuk.json`, `examples/gok/pysole_gok.json`, and `README.md` to document and include `interpolation_target` configuration settings.

---

## [0.2.0] - 2026-09-24

### Added
- **Native Dual Kriging Vector Engine**:
  - Implemented high-performance built-in Universal & Ordinary Dual Kriging engine (`pysole.interpolation.KrigingEngine`).
  - Achieved **~180x speedup** (interpolating 300,000+ DEM grid points in under 50 milliseconds) over loop-based solvers by evaluating predictions via $O(N)$ 1D BLAS vector dot products (`ddot`).
  - Added zero-centered spatial coordinate normalization and diagonal Tikhonov matrix regularization ($10^{-8}$) to ensure positive-definiteness and prevent matrix singularities.
- **Decoupled Architecture Sub-Engines**:
  - `EikonalMigrator` (`pysole.migration`): Standalone 3D Eikonal Ray Migration engine for traveltime ray displacement.
  - `BSSOptimizer` (`pysole.variogram`): Standalone Basal Shear Stress slope optimization & corner frequency ($k_{\text{c}}$) search engine.
  - `KrigingEngine` (`pysole.interpolation`): High-performance Dual Kriging vector engine.
  - `BedrockFinalizer` (`pysole.interpolation`): Standalone machine learning Random Forest gap-filling & geomorphological margin blending engine.
- **Parallelized Processing & Concurrency Control**:
  - Multi-threaded CPU chunk parallelization (`ThreadPoolExecutor`) across Dual Kriging vector evaluations, BSS corner frequency searches, and machine learning gap filling.
  - Added `n_cores` configuration parameter (-1 for all available logical cores) across all sub-engines and configuration JSON schemas (`pysole.json`).
- **Variogram Lag Distance Binning (`nrbins` & Dynamic Scaling)**:
  - Added configurable `nrbins` parameter to control experimental variogram lag distance binning during BSS slope optimization.
  - Implemented dynamic `nrbins` automatic fallbacks ensuring every lag distance bin contains $\ge 30$ point pairs to align with the Central Limit Theorem and geostatistical standards.
- **`pysole.logging` Module**:
  - Added centralized logging module (`pysole.logging`) with configurable verbosity levels (`INFO`, `DEBUG`), colored console output, and optional file logging.
- **`pysole.plotting` Module**:
  - Added dedicated plotting module (`pysole.plotting`) providing 11 high-definition visualization functions for variograms, Eikonal ray displacement vectors, Kriging bedrock elevation & uncertainty maps, ice thickness maps/histograms, and basal shear stress maps/histograms.
- **`OutputsConfig` Utility Methods & Extended Optional Exports**:
  - Added `active_exports()` and `to_dict()` helper methods to `OutputsConfig` dataclass for programmatic inspection of output raster flags.
  - Added automated unit test suite `tests/test_optional_outputs.py`.
- **CLI Flags & Logging Overrides**:
  - Expanded CLI entry point with `-V` / `--version` package version display and `-v` / `--verbose` / `--debug` flags acting as temporary runtime overrides taking precedence over `pysole.json` log level settings.

### Changed
- **Python 3.10+ Native Type Annotation Standard**:
  - Replaced legacy `typing` imports (`Union`, `Optional`, `List`, `Dict`, `Tuple`) with native Python 3.10+ type syntax (`Path | str`, `| None`, `dict`, `list`, `tuple`) across all core package modules and unit test files.
- **Resource Management & Context Managers**:
  - Enclosed all `rasterio.open()` dataset reads and writes in `with` context manager blocks for immediate resource cleanup and file handle safety.
- Refactored `Solver` into a high-level orchestrator composing decoupled sub-engine instances (`self.migrator`, `self.bss_optimizer`, `self.kriging_engine`, `self.finalizer`).
- Standardized figure filenames and export workflows into `figures/` subdirectories for Wurtenkees (WUK) and Goldbergkees (GOK) examples.
- Updated documentation (`README.md`) with comprehensive mathematical derivations, API sub-engine references, CLI options, and parameter reference tables.

### Fixed
- **DEM Grid Orientation & Path Ingestion**:
  - Fixed `load_dem()` in `raster.py` to ensure `pathlib.Path` input objects trigger top-to-bottom grid flipping checks (`if not isinstance(dem_input, np.ndarray):`).
  - Extended path validation across `load_survey_points()`, `migrate_eikonal()`, and `load_dem()` to accept `pathlib.Path` and `os.PathLike` objects.
- **Boundary Point Dimension Stacking**:
  - Fixed dimension matching in `kriging_interpolation()` when appending zero-value boundary points (`b_pts`) to 4-column survey point arrays (`[x, y, z_surf, depth]`).
- **Pipeline Orchestrator Fallback**:
  - Restored `self.survey_data_path` fallback resolution in `Solver.run_pipeline()`.

---

## [0.1.0] - 2026-08-05

### Added
- Initial release of `PySole` for physically-informed bedrock topography interpolation & 3D Eikonal ray migration.
- Support for One-Way Traveltime (OWTT) and Ice Thickness radar picks.
- Random Forest gap-filling for unmeasured glacier creeping body regions.
- Multi-format GIS export (`.tif`, `.asc`, `.csv`, `.npy`).
