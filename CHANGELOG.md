# Changelog

All notable changes to `PySole` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
