# Changelog

All notable changes to `PySole` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Changed
- Refactored `Solver` into a high-level orchestrator composing decoupled sub-engine instances (`self.migrator`, `self.bss_optimizer`, `self.kriging_engine`, `self.finalizer`).
- Standardized figure filenames and export workflows into `figures/` subdirectories for Wurtenkees (WUK) and Goldbergkees (GOK) examples.
- Updated documentation (`README.md`) with comprehensive mathematical derivations, API sub-engine references, and parameter reference tables.

---

## [0.1.0] - 2026-08-05

### Added
- Initial release of `PySole` for physically-informed bedrock topography interpolation & 3D Eikonal ray migration.
- Support for One-Way Traveltime (OWTT) and Ice Thickness radar picks.
- Random Forest gap-filling for unmeasured glacier creeping body regions.
- Multi-format GIS export (`.tif`, `.asc`, `.csv`, `.npy`).
