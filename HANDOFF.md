# Project Handoff Document: PySole

This document provides a comprehensive state-of-the-project summary for **`PySole`**. It is intended to allow any developer, collaborator, or AI coding assistant to immediately resume work with complete situational awareness, architectural context, operational commands, and recent design decisions.

---

## 1. Executive Summary & Project Identity

* **Project Purpose**: `PySole` is a physically-informed bedrock interpolation and 3D ray-based migration engine designed for sparse geophysical datasets (e.g., ground-penetrating radar, seismic sounding) across glaciers and gravity-driven viscous flows.
* **Core Inverse Principle**: Based on the Shallow Ice Approximation (SIA), creeping viscous bodies exhibit an inverse relationship between ice depth and surface slope (greater depth corresponds to gentler surface slopes). `PySole` optimizes spatial smoothing scales ($k_c$, $\lambda_c$) from real field data to calibrate physically-informed Universal Kriging drift models.
* **Workflow Architecture**:
  * **Pass 1 (Pre-Migration)**: Interpolates radar traveltimes $T(x,y)$ or BSS product $P(x,y)$ across the glacier domain using Ordinary or Universal Kriging.
  * **3D Eikonal Ray Normal Migration**: Solves the 3D Eikonal equation over dipping surface topography to displace scattered surface soundings back along true wavefront normals into 3D space $(X_{\text{mig}}, Y_{\text{mig}}, Z_{\text{mig}}, D_{\text{mig}})$.
  * **Pass 2 (Post-Migration)**: Interpolates migrated soundings into continuous ice thickness $D(x,y)$, bedrock elevation $Z_{\text{bed}} = Z_{\text{surf}} - D$, and basal shear stress $\tau_b(x,y)$.
* **Current Version**: `0.6.0` (active development on `main`).
* **Repository & Branch Status**: Branch `main`, clean working tree, fully in sync with `origin/main` (latest commit `92d414c`).

---

## 2. System Architecture & Module Map

| Component / File | Purpose & Responsibilities |
| :--- | :--- |
| [`src/pysole/solver.py`](src/pysole/solver.py) | **Primary Orchestrator (`Solver`)**: High-level workflow orchestration, state management, configuration ingestion, DEM loading, two-pass execution loops, quantitative volume and area attributes, terminal summary cards. |
| [`src/pysole/interpolation.py`](src/pysole/interpolation.py) | **Dual Kriging Vector Engine**: High-performance geostatistical engine. LU factorization (`scipy.linalg.lu_factor`), Level-3 BLAS vector dot-products ($O(N \cdot m)$ predictions), chunked estimation variance evaluation ($O(N^2 \cdot m)$), adaptive diagonal Tikhonov regularization ($\lambda_{\text{reg}} = 10^{-6} \cdot S$). |
| [`src/pysole/drift.py`](src/pysole/drift.py) | **Drift Primitives & Combinatorics**: Implements the 8 drift primitives (`"sia"`, `"z_dem"`, `"curvature_dem"`, `"mass_balance"`, `"velocity"`, `"velocity_sia"`, `"linear_xy"`, `"quadratic_xy"`), dynamic candidate suite building (24, 48, 60, 120 models), collinearity pruning, execution plan resolution. |
| [`src/pysole/drift_analyzer.py`](src/pysole/drift_analyzer.py) | **Cross-Validation & Ranking**: Evaluates candidate models via Leave-One-Profile-Out cross-validation (LOPO-CV with $\ge 3$ tracks) or Spatial Buffer LOOCV ($r_{\text{buffer}} = a_0 / 2$). Computes AICc, RMSE, penalized variance. Exports unified `drift_model_comparison.{txt,csv,md}`. |
| [`src/pysole/migration.py`](src/pysole/migration.py) | **3D Eikonal Ray Migration**: Computes apparent surface slownesses, incorporates topographic slope derivatives, solves the quadratic Eikonal condition for vertical slowness $p_z$, and evaluates 3D displacement vectors $(\Delta x, \Delta y, \Delta z)$. |
| [`src/pysole/raster.py`](src/pysole/raster.py) | **Raster & Geometry Engine**: Raster I/O (GeoTIFF, ASCII, CSV, NPY), coordinate transformations, DEM spatial interpolation, survey points loading and validation (`load_survey_points`), quantitative ice volume calculation (`calculate_ice_volume`). |
| [`src/pysole/plotting.py`](src/pysole/plotting.py) | **Diagnostic Figures Engine**: Automated generation of publication-quality diagnostic plots `01_` through `11_` (variograms, optimization curves, 3D migration vectors, bedrock topography, uncertainty maps, thickness and shear stress distributions). |
| [`src/pysole/config.py`](src/pysole/config.py) | **Configuration & Schema Engine**: Ingests and validates `pysole.json`, populates `DEFAULT_CONFIG`, CLI argument parser (`pysole` command). |

---

## 3. Recent Major Developments (Post-v0.6.0 & Recent Sessions)

### A. Glaciological Drift Primitives (v0.6.0 Release)
* Integrated three new glaciological covariates:
  * `"mass_balance"`: Decadal surface mass balance field $U_{\text{mb}}(x,y) = \dot{b}(x,y)$, unsmoothed to preserve localized accumulation/ablation anomalies.
  * `"velocity"`: Off-glacier zero-clamped surface flow speed, spectrally low-pass filtered at optimal wavenumber $k_c$.
  * `"velocity_sia"`: Coupled SIA-velocity creep scaling $U_{\text{vel\_sia}}(x,y) = (u_{s,\text{smooth},k_c})^{1/4} / (\sin\alpha_{\text{opt}})^{3/4}$ derived from Glen's flow law ($n = 3$).
* Expanded dynamic candidate suites: **24** (DEM only), **48** (DEM + mass balance), **60** (DEM + velocity), and **120** (DEM + mass balance + velocity) models.
* Dynamic collinearity pruning: `"velocity_sia"` is never combined with `"sia"` or `"velocity"`.

### B. Quantitative Ice Volume & Autocorrelated Uncertainty
* Implemented quantitative domain area ($A_{\text{total}} = N \cdot |dx \cdot dy|$), total ice volume ($V = A_{\text{total}} \cdot \bar{D}$ [$\text{m}^3$, $\text{km}^3$]), and spatially autocorrelated Kriging volume uncertainty:
  $$\sigma_V = A_{\text{total}} \cdot \bar{\sigma}_D = \sum_{i \in \text{glacier}} \sigma_{D, i} \cdot \Delta A \quad \Longleftrightarrow \quad \frac{\sigma_V}{V} = \frac{\bar{\sigma}_D}{\bar{D}}$$
  following glaciological error propagation methodology (Binder et al., 2009; Fischer, 2009).
* Added structured terminal summary cards reporting volume and distribution statistics.
* Added calculated total volume to diagnostic Figure 09 (`09_<output_prefix>_final_thickness_histogram.png`).

### C. Migration Visualization & Colormap Upgrades
* **Figure 03 (`03_<output_prefix>_3d_migration.png`) Contour Labels**:
  * If `migration_parameters.horizontal_displacement_visualization == "grad_traveltime_grid"`, contour line numeric labels are omitted (`ax1.clabel` skipped), displaying clean, unobstructed contour lines over the traveltime gradient field.
* **Figure 03 Companion Colormap Upgrade**:
  * Replaced the gradient colormap in `plot_migration_horizontal_displacements` with **Early Amber-Red** (`_amber_red_gradient_cmap`), resolving low-to-medium traveltime gradients with distinct contrast compared to near-white.
* **Figure 08 Thickness Uncertainty Colormap Standardization**:
  * Standardized Figure 08 thickness uncertainty colormap to `_get_transparent_cmap("plasma")`, matching Figures 06 and 10.

### D. Survey Profile Column Default & Survey CSV Parsing Documentation
* Changed `inputs.survey_profile_column` default from `null` to `"profile_id"`.
* Added Subsection 3 under `## Technical & Methodological Notes` in `README.md`: **"Parsing of Survey Data CSV Files"**.
* Documented the core parsing behaviors:
  * **3-Column Inputs $[X, Y, \text{value}]$**: Automatically bilinearly samples $Z_{\text{surface}}$ from the surface DEM (`inputs.dem_path`) using `RegularGridInterpolator` and inserts it as Column 2.
  * **4-Column Inputs $[X, Y, Z, \text{value}]$**: Preserves explicit $Z$ directly without DEM modification.
  * **Files WITH a Header**: Identified by name (`inputs.survey_profile_column`). Profile column can appear in any column position; remaining columns follow the standard $[X, Y, (Z), \text{value}]$ sequence. Enables LOPO-CV when $\ge 3$ unique profiles exist.
  * **Files WITHOUT a Header**: Positionally read via `np.loadtxt`. Must strictly follow standard positional sequences ($[X, Y, \text{value}]$, $[X, Y, Z, \text{value}]$, or $[X, Y, Z, \text{value}, \text{profile\_id}]$). Setting `"survey_profile_column": null` avoids spurious "header not found" warnings.
  * Cautions against positional traps (e.g. profile ID placed in Column 0 triggering DEM bounds error; or headerless 4-column $[X, Y, \text{TWT}, \text{profile\_id}]$ misinterpreting TWT as $Z_{\text{surface}}$).
* Provided the concise **Rule of Thumb for Survey CSVs**.

### E. CI & Test Suite Synchronization
* Made `tests/test_docs_sync.py` section heading matching regex-flexible (`#{2,4}`), ensuring tests do not break if markdown headings switch between `###` and `####`.
* All 388 test cases pass cleanly.

---

## 4. Key Design Decisions & Critical Invariants

1. **Standard Survey Point Matrix Shape**:
   Internally, `load_survey_points` in [`src/pysole/raster.py`](src/pysole/raster.py) always builds an array with shape $(N, 4)$ or $(N, 5)$:
   $$\begin{bmatrix} X & Y & Z_{\text{surface}} & \text{value} & [\text{profile\_id}] \end{bmatrix}$$
   Downstream solvers (Eikonal migration, Kriging systems) depend strictly on:
   * Col 0: Easting $X$ [m]
   * Col 1: Northing $Y$ [m]
   * Col 2: Surface Elevation $Z_{\text{surface}}$ [m]
   * Col 3: Measured sounding value (TWT [ns], OWT [ns], or thickness [m])
   * Col 4 (optional): Discrete integer/categorical profile identifier.

2. **Kriging Uncertainty vs. Sample Standard Deviation**:
   * Kriging standard error $\sigma_D(x,y) = \sqrt{\sigma^2(x,y)}$ represents spatial geostatistical estimation uncertainty based on sounding layout and variogram structure.
   * Domain-average Kriging uncertainty $\bar{\sigma}_D$ is distinct from the empirical standard deviation of ice thickness measurements ($\text{std}(D)$), even if they occasionally have comparable magnitudes in sparse datasets.

3. **Output Files Hierarchy**:
   * All primary spatial raster grids (`bedrock`, `thickness`, `traveltime`, `basal_shear_stress`, and uncertainties) and comparison tables (`drift_model_comparison.{txt,csv,md}`) are saved directly to `<output_dir>`.
   * Raster grids have applied drift model slugs appended: `<output_prefix>_<grid_type>_<model_slug>.<ext>`.
   * Migrated points CSV is saved to `<output_dir>/<output_prefix>_migrated_points.csv`.
   * Diagnostic figures are saved to `<output_dir>/figures/` (or `<output_dir>/figures/<model_slug>/` for multi-model suites).

4. **HTML Inline Math in Markdown Documentation**:
   * Per project standards, all inline math in `README.md` must use HTML tags (`<i>X</i>`, `<i>Z</i><sub>surface</sub>`, Unicode symbols) rather than LaTeX `$inline$` math to avoid vertical clipping in GitHub Markdown rendering. Display blocks use `$$...$$`.
   * Enforced automatically by `test_readme_has_no_inline_latex_math` in `tests/test_docs_sync.py`.

---

## 5. Operational Verification & Testing Commands

All commands should be executed from the repository root `/home/db/Software/pysole` using the local virtual environment:

### Virtual Environment Python
```bash
./.venv/bin/python
```

### Run the Fast Documentation Sync Test (< 0.1s)
Checks drift models, JSON configuration consistency, relative links, and HTML inline math:
```bash
./.venv/bin/python -m unittest tests/test_docs_sync.py
```

### Run Configuration & Audit Regression Tests (~35s)
```bash
./.venv/bin/python -m unittest tests/test_config.py tests/test_audit_regressions.py
```

### Run the Complete Test Suite (~3–5 mins)
Executes all 388 unit and integration tests:
```bash
./.venv/bin/python -m unittest discover -s tests -f -b
```

### Package Installation & Build
```bash
pip install -e .
```

---

## 6. Collaboration Rules & Project Constraints

1. **Git Remote Push**:
   * **Never execute `git push` autonomously.** The user explicitly performs all pushes to the remote GitHub repository.
2. **Codebase Cleanliness & Comments**:
   * Maintain existing docstrings and comments. Do not delete explanatory notes unless explicitly asked.
3. **Markdown File Links**:
   * In assistant responses, always use clickable `file://` links for file paths and symbol definitions.
4. **No Legacy Aliases**:
   * Modern primitive names are strictly enforced (`"sia"`, `"z_dem"`, `"curvature_dem"`, `"mass_balance"`, `"velocity"`, `"velocity_sia"`, `"linear_xy"`, `"quadratic_xy"`). Legacy term aliases have been deprecated and pruned.

---

## 7. Current Status & Potential Next Steps

* **Current Status**: All code, documentation, and tests are clean, consistent, and validated.
* **Potential Next Steps**:
  1. **Version Release Preparation**: If releasing a patch/minor update (e.g. `v0.6.1`), bump version in `pyproject.toml`, update `CHANGELOG.md`, and verify release build with `python -m build`.
  2. **Manuscript Alignment**: If preparing academic publication materials, update or synchronize `paper.md` with recent glaciological drift models, volume calculations, and migration enhancements.
  3. **New Features / Glaciological Enhancements**: As requested by the user during interactive sessions.
