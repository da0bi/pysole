<p align="center">
  <img src="images/pysole_nano_logo_v2_transparent.png" width="360" alt="PySole Logo">
</p>
<p style="text-align: center; font-size: 20px;"><strong>Physically-Informed Bedrock Interpolation & 3D Migration for Sparse Geophysical Datasets</strong></p>

`PySole` is designed to reconstruct the thickness distribution and basal topography (sole) of glaciers, landslides, and other gravity-driven, viscous flow phenomena. It adapts the **Shallow Ice Approximation** to estimate the thickness distribution based on the fundamental inverse relation between depth of the creeping body and its surface slope, where greater depths correspond to gentler surface slopes and vice versa. It is specifically engineered for sparse geophysical datasets where 3D wavefield migration to image the bedrock is impossible due to insufficient spatial sampling. `PySole` transforms limited survey points into robust, physically-constrained 3D bedrock models.

---

## Table of Contents

[Key Features](#key-features)<br><br>
[Workflow & Methodology](#workflow-and-methodology)<br><br>
[Installation](#installation)<br><br>
[Configuration Guide (`pysole.json`)](#configuration-guide)<br><br>
[Configuration Parameter Reference](#configuration-parameter-reference)<br><br>
[Technical & Methodological Notes](#technical-and-methodological-notes)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[1. Supported DEM Input Formats](#supported-dem-input-formats)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2. Parsing of Rock Outcrop & Nunatak Input Files](#parsing-rock-outcrops)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3. Dynamic Variogram Binning with Minimum Pair Threshold](#dynamic-variogram-binning)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4. High-Performance Dual Kriging Vector Engine](#dual-kriging-vector-engine)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5. Shallow Ice Approximation Drift Model](#shallow-ice-approximation-custom-drift)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6. Depth Uncertainty Derivation](#depth-uncertainty-derivation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7. Spatial Smoothing of the Calculated DEMs](#dem-spatial-smoothing)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[8. Multi-Format DEM Export](#multi-format-dem-export)<br><br>
[Package Architecture](#package-architecture)<br><br>
[Command-Line Interface (CLI) Execution](#cli-execution)<br><br>
[Python API & Quick Start](#python-api-and-quick-start)<br><br>
[Real-World Example: Wurtenkees Glacier](#real-world-example-wurtenkees-glacier)<br><br>
[Citation & References](#citation-and-references)

---

<a id="key-features"></a>
## Key Features

* **JSON Configuration & Terminal CLI Driven:** All processing workflow options can be defined in a `pysole.json` file and executed via Python API or directly from the terminal using the `pysole` command line tool.
* **Multi-Core Parallel Acceleration:** `PySole` supports multi-core CPU parallelization across computationally intensive processing steps.
* **Automated High-Resolution Diagnostic Plots:** Automatically generates and optionally exports diagnostic figures for each key processing milestone.
* **5 Supported Digital Elevation Model (DEM) Input and Output Formats:** Seamless loading and exporting of GeoTIFFs (`.tif`), ESRI ASCII Grids (`.asc`, `.txt`), CSV matrices (`.csv`), NumPy binary arrays (`.npy`), and in-memory NumPy 2D arrays (`np.ndarray`).
* **Strict CRS & Spatial Alignment Verification:** Performs strict verification across all input layers (DEM, boundary outline, survey points). If any layer uses a different Coordinate Reference System or falls outside the DEM spatial extent, processing halts with an explicit error.
* **Rock Outcrop & Nunatak Hole Support:** Native parsing of interior vector polygon holes. When boundary conditions are enabled, zero-traveltime/-thickness constraints are automatically applied along internal hole perimeters.
* **Flexible Survey Data Types:** `PySole` accepts one- or two-way signal traveltimes as well as direct thickness/depth measurements as survey data type. In case of direct thickness/depth data the 3D ray-based migration is automatically skipped.
* 💡**DEM Surface Slope Smoothing💡:** The degree of DEM surface slope smoothing is crucial when estimating ice thickness with the Shallow Ice Approximation (SIA), which assumes a constant basal shear stress. By relaxing this rigid baseline constraint, Binder et al. (2009) derived an objective optimization criterion for the surface slope smoothing process, which is implemented in `PySole`. The optimal degree of surface slope smoothing is derived by enforcing minimum spatial variance in basal shear stress as the optimization criterion:
  <p align="center">
    <font size="+1"><b>min<sub><i>k</i><sub>c</sub></sub> Var<sub><i>xy</i></sub>(<i>τ</i><sub>b</sub>)</b></font>
  </p>

  In shallow ice dynamics, basal shear stress is given by:

  <p align="center">
    <font size="+1"><b><i>τ</i><sub>b</sub> = <i>ρ</i><sub>ice</sub> <i>g</i> <i>D</i> sin(<i>α</i>)</b></font>
  </p>

  where ice density, <i>ρ</i><sub>ice</sub>, and gravitational acceleration, <i>g</i>, are assumed to be constant. Thus, just the product of the two variables ice depth and surface slope, <i>P</i> = <i>D</i> sin(<i>α</i>), is evaluated during the optimization process. Surface slope smoothing is performed in the frequency domain using <i>Fast Fourier Transform</i> (FFT) filtering, while spatial variance is quantified via variogram analysis. An interactive mode allows users to test varying degrees of smoothing across spatial wavenumber cutoffs (<i>k</i><sub>c</sub>) and refine the variogram correlation range. This surface slope optimization methodology is an integral component for interpolating both pre-migration wavefront traveltimes and post-migration depths. To accelerate spatial wavenumber evaluations across high-resolution DEM grids, <i>k</i><sub>c</sub> filtering is executed via multi-threaded CPU parallelization.
* **3D Ray-Based Migration:** `PySole` features an optional 3D ray-based migration—introduced by Binder et al. (2009) and engineered specifically to process geophysical signal traveltimes with sparse spatial coverage.
* **Kriging Interpolation & Boundary Condition:** Provides a native, numerically optimized, and parallelized 2D Kriging algorithm supporting both Ordinary and Universal Kriging (default, with data drift correction). Optionally, `PySole` supports 2D Universal, Ordinary, and Regression Kriging via the [`PyKrige`](https://geostat-framework.readthedocs.io/projects/pykrige) package. A custom Universal Kriging drift model based on the SIA is available by default. Perimeter boundary conditions (zero traveltime <i>T</i> = 0 s and zero thickness <i>D</i> = 0 m) can optionally be enforced alongside corresponding Kriging estimation uncertainty fields.
* **ML Hole Filling & Geomorphological Margin Blending:** Employs the parallelized [`scikit-learn`](https://scikit-learn.org) Random Forest regression to patch blank regions and ensure complete spatial coverage after Kriging interpolation (optional step). Furthermore, geomorphological margin blending can be applied to smoothly taper bedrock elevations into the surrounding surface DEM terrain.
* **Final DEMs Spatial Smoothing:** As a post-processing step, spatial smoothing options are available for the calculated DEMs.

---

<a id="workflow-and-methodology"></a>
## Workflow & Methodology

<p align="center">
  <a href="images/pysole_processing_pipeline.png">
    <img src="images/pysole_processing_pipeline.png" width="100%" alt="PySole Processing Pipeline Workflow">
  </a>
  <br>
  <em>Figure 1:  End-to-end computational workflow of the PySole solver processing pipeline. Click diagram to view in high resolution.</em>
</p>

#### 1. DEM Loading & Spatial Resampling
Grid spacing (`dx`, `dy`) and bounding extent are automatically extracted from DEM metadata. If target `dx` and `dy` pixel sizes are specified, 2D bilinear grid resampling is performed automatically.

#### 2. Pre-Migration Traveltime Interpolation
Applies the optimization criterion to determine the optimal surface slope smoothing degree by evaluating the product of traveltime observations and corresponding smoothed surface slopes, <i>P</i><sub>T,i</sub> = <i>T</i><sub>i</sub> sin(<i>α</i><sub>smoothed,i</sub>). Once the optimal smoothing degree is determined, `PySole` interpolates <i>P</i><sub>T,i</sub> using Kriging (with optional zero-traveltime boundary conditions <i>T</i> = 0 s) to receive the continuous product field <i>P</i><sub>T</sub>(<i>x</i>,<i>y</i>). The continuous signal traveltime field <i>T</i>(<i>x</i>,<i>y</i>) is ultimately reconstructed by dividing <i>P</i><sub>T</sub>(<i>x</i>,<i>y</i>) by the optimal smoothed surface slope field sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>)):

  <p align="center">
    <font><i>T</i>(<i>x</i>,<i>y</i>) = <i>P</i><sub>T</sub>(<i>x</i>,<i>y</i>) / sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>))</font>
  </p>

A minimum smoothed surface slope threshold of 2.0° is enforced to prevent numerical instabilities and unphysical singularities in low-gradient regions.

#### 3. 3D Ray-Based Migration
The migration algorithm solves the Eikonal equation to relocate subsurface reflection points, particuarly improving the imaging of steep slopes and overdeepenings. An interactive mode allows users to test different signal propagation velocities and evaluate them through visualizations of migrated depths and the corresponding horizontal survey point displacements induced by the migration process.

#### 4. Post-Migration Surface Slope Optimization
Analogous to the pre-migration interpolation step, `PySole` applies the optimization criterion to determine the optimal surface slope smoothing degree, sin(<i>α</i><sub>opt</sub>), across spatial wavenumber cutoffs <i>k</i><sub>c</sub>. In this second optimization pass the product of migrated depths (<i>D</i><sub>i</sub>) and the corresponding smoothed surface slopes, <i>P</i><sub>D,i</sub> = <i>D</i><sub>i</sub> sin(<i>α</i><sub>smoothed,i</sub>), is evaluated.

#### 5. Final Depth Interpolation & Uncertainty Display
Performs spatial Kriging interpolation on the products <i>P</i><sub>D,i</sub> = <i>D</i><sub>i</sub> sin(<i>α</i><sub>opt,i</sub>) to reconstruct continuous depth <i>D</i>(<i>x</i>,<i>y</i>) and bedrock elevation <i>Z</i><sub>bed</sub>(<i>x</i>,<i>y</i>) fields. The Kriging standard error for the interpolated depths is converted to meters to quantify depth uncertainty.

---

<a id="installation"></a>
## Installation

### Standard Installation via PyPI <strong><i> -> !!! NOT AVAILABLE YET !!!</i></strong>

```bash
pip install pysole
```

### Local / Development Installation

To install `PySole` directly from source in editable mode:

```bash
git clone https://github.com/da0bi/pysole.git
cd pysole
pip install -e .
```

---

<a id="configuration-guide"></a>
## Configuration Guide (`pysole.json`)

All execution options can be fully defined in a single `pysole.json` configuration file:

```json
{
    "inputs": {
        "dem_path": null,
        "outline_path": null,
        "survey_data_path": null,
        "base_dir": null,
        "survey_data_type": "one_way_travel_time",
        "ice_density": 900.0,
        "g": 9.81,
        "n_cores": -1,
        "log_level": "INFO"
    },
    "spatial_parameters": {
        "dx": 5.0,
        "dy": 5.0,
        "bounds": null
    },
    "migration_parameters": {
        "perform_migration": true,
        "velocity": 0.16,
        "interactive_migration": false
    },
    "optimization_parameters": {
        "kc_max": 10.0,
        "kc_min": 0.01,
        "d_kc": 0.1,
        "nrbins": null,
        "slope_floor_deg": 5.0,
        "interactive_optimization": false
    },
    "kriging_parameters": {
        "built_in_kriging": true,
        "pre_migration": {
            "method": "universal",
            "drift_terms": ["sia_thickness"],
            "variogram_model": "spherical",
            "include_zero_boundary_condition": true
        },
        "post_migration": {
            "method": "universal",
            "drift_terms": ["sia_thickness"],
            "variogram_model": "spherical",
            "include_zero_boundary_condition": true
        }
    },
    "finalization_parameters": {
        "random_forest_gap_filling": false,
        "apply_margin_blend": false,
        "min_gap_dist": 50.0,
        "smooth_bedrock": false,
        "smoothing_method": "gaussian",
        "smoothing_sigma": 1.5,
        "smoothing_kernel_size": 3,
        "smoothing_kc_cutoff": null
    },
    "outputs": {
        "output_format": "tif",
        "output_name": "final_bedrock",
        "plots_dir": "figures"
    }
}
```

---

<a id="configuration-parameter-reference"></a>
## Configuration Parameter Reference

| Section | Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`inputs`** | `dem_path` | `str` | `null` | **(Required)** File path to the surface Digital Elevation Model (`.asc`, `.tif`, `.csv`, `.npy`). |
| | `outline_path` | `str` | `null` | File path to creeping body / glacier boundary polygon (`.shp`, `.geojson`, `.gpkg`, `.csv`). If `null`, domain is derived from non-NaN DEM pixels. |
| | `survey_data_path` | `str` | `null` | **(Required)** File path to signal travel time or thickness observations CSV `[X, Y, value]`. |
| | `base_dir` | `str` | `null` | General workspace directory for all output files. If `null` (default), automatically falls back to the parent directory of `survey_data_path`. All relative output paths are resolved relative to `base_dir`. Absolute output paths override `base_dir`. |
| | `survey_data_type` | `str` | `"one_way_travel_time"` | Observation data type: `"one_way_travel_time"`, `"two_way_travel_time"`, or `"thickness"` / `"ice_thickness"` (skips ray migration). |
| | `ice_density` | `float` | `900.0` | Density of the creeping medium in kg/m³ (e.g. `900.0` kg/m³ for temperate glacier ice). Used to calculate basal shear stress $\tau_{\text{b}}$. |
| | `g` | `float` | `9.81` | Gravitational acceleration constant in m/s² (`9.81` m/s²). Used to calculate basal shear stress $\tau_{\text{b}}$. |
| | `n_cores` | `int` | `-1` | Number of CPU cores applied across all parallelized processes (`-1` for all available cores). |
| | `log_level` | `str` | `"INFO"` | Package logging level verbosity: `"INFO"` (default), `"DEBUG"`, `"WARNING"`, `"ERROR"`, or `"CRITICAL"`. Appends timestamped logs to `pysole.log`. |
| **`spatial_parameters`** | `dx` | `float` | `null` | Target grid resolution along X in meters. If defined, automatically resamples the DEM grid. If `null`, native resolution is kept. |
| | `dy` | `float` | `null` | Target grid resolution along Y in meters. If defined, automatically resamples the DEM grid. If `null`, native resolution is kept. |
| | `bounds` | `list[float]` | `null` | Spatial bounding box `[minx, miny, maxx, maxy]`. If `null`, extracted directly from DEM raster metadata. |
| **`migration_parameters`** | `perform_migration` | `bool` | `true` | If `true`, performs 3D ray-based migration on signal travel times. If `false`, migration is skipped. |
| | `velocity` | `float` | `0.16` | Signal propagation velocity (e.g. `0.16` m/ns for GPR radar wave propagation in temperate ice). |
| | `interactive_migration` | `bool` | `false` | If `true`, enables interactive velocity testing with visual displacement vector plots. |
| **`optimization_parameters`** | `kc_max` | `float` | `10.0` | Maximum corner frequency for FFT Gaussian low-pass smoothing. |
| | `kc_min` | `float` | `0.01` | Minimum corner frequency for FFT Gaussian low-pass smoothing. |
| | `d_kc` | `float` | `0.1` | Corner frequency stepwidth for evaluating basal shear stress spatial variance Var<sub><i>xy</i></sub>(<i>τ</i><sub>b</sub>). |
| | `nrbins` | `int` | `null` | Number of variogram lag distance bins. If `null`, dynamically calculated to guarantee at least 30 point pairs per bin. |
| | `slope_floor_deg` | `float` | `5.0` | Minimum surface slope angle threshold in degrees [°] enforced during surface slope optimization to prevent numerical division singularities. |
| | `interactive_optimization` | `bool` | `false` | If `true`, enables interactive CLI prompt to inspect BSS variance curve and adjust corner frequency spectrum parameters (`kc_min`, `kc_max`, `d_kc`), lag distance bin count (`nrbins`), and correlation range (<i>a</i><sub>range</sub>). |
| **`kriging_parameters`** | `built_in_kriging` | `bool` | `true` | If `true` (default), uses native numerically optimized and parallelized Dual Kriging engine (supporting Ordinary and Universal Kriging). If `false`, Kriging implementations from the PyKrige package are applied. |
| | `pre_migration` | `dict` | *Sub-section* | Configuration for pre-migration traveltime interpolation (<i>T</i>(<i>x</i>,<i>y</i>)). |
| | `pre_migration.method` | `str` | `"universal"` | Kriging approach: `"universal"` (default), `"ordinary"`, or `"regression"`. Note that `"regression"` mode requires the optional [`PyKrige`](https://geostat-framework.readthedocs.io/projects/pykrige) package. |
| | `pre_migration.drift_terms` | `list[str]` | `["sia_thickness"]` | Drift terms for Universal Kriging: `["sia_thickness"]` (SIA-based physical drift model), `["quadratic"]`, or `["regional_linear"]`. |
| | `pre_migration.variogram_model` | `str` | `"spherical"` | Theoretical variogram model (`"spherical"`, `"exponential"`, `"gaussian"`, `"linear"`). |
| | `pre_migration.include_zero_boundary_condition` | `bool` | `true` | If `true` (default), includes zero traveltime boundary points (<i>T</i> = 0 ns) along the perimeter and rock outcrop/nunatak margin outline(s). |
| | `post_migration` | `dict` | *Sub-section* | Configuration for final bedrock depth interpolation (<i>D</i>(<i>x</i>,<i>y</i>) [m]). |
| | `post_migration.method` | `str` | `"universal"` | Kriging approach: `"universal"` (default), `"ordinary"`, or `"regression"`. Note that `"regression"` mode requires the optional [`PyKrige`](https://geostat-framework.readthedocs.io/projects/pykrige) package. |
| | `post_migration.drift_terms` | `list[str]` | `["sia_thickness"]` | Drift terms for Universal Kriging: `["sia_thickness"]` (SIA-based physical drift model), `["quadratic"]`, or `["regional_linear"]`. |
| | `post_migration.variogram_model` | `str` | `"spherical"` | Theoretical variogram model (`"spherical"`, `"exponential"`, `"gaussian"`, `"linear"`). |
| | `post_migration.include_zero_boundary_condition` | `bool` | `true` | If `true` (default), includes zero thickness boundary points (<i>D</i> = 0 m) along the perimeter and rock outcrop/nunatak margin outline(s). |
| **`finalization_parameters`** | `random_forest_gap_filling` | `bool` | `false` | If `true`, applies Random Forest machine learning gap filling across unmeasured interior regions before margin blending. |
| | `apply_margin_blend` | `bool` | `false` | If `true`, applies geomorphological margin blending to seamlessly transition calculated bedrock elevation to surrounding surface DEM terrain. |
| | `min_gap_dist` | `float` | `50.0` | Minimum gap distance in meters [m] inside which bedrock is smoothly tapered and blended into surface DEM terrain. |
| | `smooth_bedrock` | `bool` | `false` | If `true`, applies spatial DEM post-processing smoothing directly to the ice depth field <i>D</i>(<i>x</i>,<i>y</i>) to eliminate high-frequency slope-division noise. |
| | `smoothing_method` | `str` | `"gaussian"` | Final bedrock DEM smoothing algorithm choice: `"gaussian"` (default), `"median"`, or `"fft_lowpass"`. |
| | `smoothing_sigma` | `float` | `1.5` | Smoothing strength (radius in pixels) for `"gaussian"` filtering. Higher values produce smoother bedrock terrain. |
| | `smoothing_kernel_size` | `int` | `3` | Window kernel size (<i>k</i> × <i>k</i>) for `"median"` filtering (must be an odd integer). Higher values produce smoother bedrock terrain. |
| | `smoothing_kc_cutoff` | `float` | `null` | Corner frequency cutoff wavenumber (<i>k</i><sub>c,smooth</sub>) for `"fft_lowpass"`. If `null`, defaults to <i>k</i><sub>c,opt</sub>. <i>Lower</i> values produce smoother bedrock terrain. |
| **`outputs`** | `output_format` | `str` / `list[str]` | `"tif"` | Desired export format(s): `"tif"`, `"asc"`, `"csv"`, `"npy"`, a list of formats (e.g. `["tif", "asc", "csv"]`), or `"all"` to export all four formats. |
| | `output_name` | `str` | `"final_bedrock"` | Filename or absolute filename path of the final bedrock elevation raster(s). Do not use a file extension—extension(s) are strictly determined by `output_format` and appended automatically. Relative names resolve inside `base_dir`; absolute paths override `base_dir`. |
| | `plots_dir` | `str` | `"figures"` | Output directory for saving diagnostic figures. All generated figures are saved automatically. If `null`, figures are saved into a `"figures"` folder inside `base_dir`. An absolute path overrides `base_dir/figures`. |

---

<a id="technical-and-methodological-notes"></a>
## Technical & Methodological Notes

<a id="supported-dem-input-formats"></a>
#### 1. Supported DEM Input Formats
`PySole` supports 5 distinct DEM input formats in `load_dem()`:

- **GeoTIFF (`.tif`, `.tiff`, `.geotiff`)**: *Recommended*. Automatically extracts spatial bounds, transform, CRS, pixel resolution, and `nodata` values using [`rasterio`](https://rasterio.readthedocs.io).
- **ESRI ASCII Grid (`.asc`, `.txt`)**: Standard 6-line header GIS raster format. Automatically extracts `ncols`, `nrows`, `xllcorner`, `yllcorner`, `cellsize`, and `NODATA_value`.
- **CSV Matrix (`.csv`)**: 2D comma-separated matrix of elevation values.
- **NumPy Binary Array (`.npy`)**: Fast 2D binary array loaded via `np.load()`.
- **NumPy 2D Array (`np.ndarray`)**: Direct in-memory array passed into the `Solver` constructor (`dem=dem_grid`).

<a id="parsing-rock-outcrops"></a>
#### 2. Parsing of Rock Outcrop & Nunatak Input Files
Vector polygon files (`.shp`, `.geojson`, `.gpkg`) or CSV outline files containing interior rings (separated by `NaN` rows) are automatically parsed as polygon holes. `PySole` treats pixels inside rock outcrop holes as exposed bedrock (<i>D</i> = 0 m).

For rock outcrop holes to be detected correctly from a Shapefile (`.shp`):

- **Topology**: The outcrop must be stored as an interior ring (`polygon.interiors`) within a single `Polygon` / `MultiPolygon` feature (e.g. created using QGIS "Add Ring" or ArcGIS "Construct Hole").
- **Winding Order**: Standard OGC orientation (Clockwise exterior boundary, Counter-Clockwise interior hole rings).
- **CRS Alignment**: The shapefile's Coordinate Reference System must match the DEM raster projection.
- **Valid Geometries**: Rings must not intersect themselves (`PySole` automatically executes `validate_and_extract_polygons()` on load to auto-repair geometries or fall back to the outer boundary shell if holes fail criteria).

<a id="dynamic-variogram-binning"></a>
#### 3. Dynamic Variogram Binning with Minimum Pair Threshold
Experimental variogram lag distance bins are calculated from the pairwise Euclidean distances ($d_{ij}$) between all survey points. Users can specify a fixed number of lag bins via `nrbins` under `optimization_parameters` in `pysole.json`, or during the `interactive_optimization` procedure. When `nrbins` is set to `null` (default), PySole dynamically determines the minimum distance bin count based on the total number of survey point pairs ($N_{\text{pairs}} = \frac{N(N-1)}{2}$):

<p align="center">
  <i>nrbins</i> = <i>max(3, N<sub>pairs</sub> / 30)</i>
</p>

Enforcing a minimum threshold of at least **30 point pairs per lag bin** aligns with established geostatistical literature (e.g. Webster and Oliver, 2007), ensuring robust experimental variogram estimation and stable theoretical model curve fitting. If a user-specified `nrbins` yields fewer than 30 average point pairs per bin, a diagnostic warning is emitted while honoring the user's explicit bin choice.

<a id="dual-kriging-vector-engine"></a>
#### 4. High-Performance Dual Kriging Vector Engine
`PySole` features a native, numerically optimized geostatistical engine based on **Dual Kriging** (Matheron, 1981). Unlike standard Kriging implementations (Primal Kriging) that solve node-specific linear systems point-by-point for every target grid node (requiring millions of repetitive matrix inversions across a high-resolution DEM), Dual Kriging solves the global linear system only once for the entire sample observation set:

<p align="center">
  <i>K</i> <i>w</i><sub>z</sub> = z<sub>aug</sub>
</p>

where <i>K</i> is the augmented sample-to-sample covariance/variogram matrix, <i>z</i><sub>aug</sub> = [<i>z</i><sub>1</sub>, ..., <i>z</i><sub><i>N</i></sub>, 0, ..., 0]<sup>T</sup> contains the known data points augmented with zero drift constraints, and <i>w</i><sub>z</sub> = [<i>w</i><sub>sample</sub><sup>T</sup>, <i>w</i><sub>drift</sub><sup>T</sup>]<sup>T</sup> = [<i>b</i><sub>1</sub>, ..., <i>b</i><sub><i>N</i></sub>, <i>a</i><sub>1</sub>, ..., <i>a</i><sub><i>L</i></sub>]<sup>T</sup> is the single global dual weight vector solved via <i>Lower-Upper</i> (LU) matrix decomposition. Once <i>w</i><sub>z</sub> is computed, spatial interpolation across all target grid nodes simplifies to a single <i>Basic Linear Algebra Subprograms</i> (BLAS)-accelerated 1D vector dot product:

<p align="center">
  <i>Z</i><sub>grid</sub> = <i>w</i><sub>sample</sub> · <i>Γ</i><sub>grid</sub> + <i>w</i><sub>drift</sub> · <i>F</i><sub>grid</sub>
</p>

where:
- <i>Z</i><sub>grid</sub> is the predicted output value (e.g., bedrock elevation or depth) at target grid node (<i>x</i>, <i>y</i>).
- <i>w</i><sub>sample</sub> = [<i>b</i><sub>1</sub>, ..., <i>b</i><sub><i>N</i></sub>] are the solved dual spatial weights for each of the <i>N</i> data points.
- <i>Γ</i><sub>grid</sub> = [&gamma;(<i>x</i><sub>1</sub>, <i>x</i><sub>grid</sub>), ..., &gamma;(<i>x</i><sub><i>N</i></sub>, <i>x</i><sub>grid</sub>)]<sup>T</sup> is the 1D sample-to-grid cross-variogram vector measuring spatial correlation between each data point and target node (<i>x</i>, <i>y</i>).
- <i>w</i><sub>drift</sub> = [<i>a</i><sub>1</sub>, ..., <i>a</i><sub><i>L</i></sub>] are the solved dual drift model coefficients.
- <i>F</i><sub>grid</sub> is the drift function vector evaluated at target node (<i>x</i>, <i>y</i>) (e.g. constant mean, coordinate trends, or SIA physical ice thickness drift).

To guarantee numerical stability during matrix decomposition, diagonal Tikhonov regularization adds a microscopic offset (10<sup>−8</sup>) to the main diagonal of <i>K</i>, ensuring positive-definiteness and preventing matrix singularities. Combined with zero-centered spatial coordinate normalization and multi-threaded CPU chunk parallelization (`ThreadPoolExecutor`), PySole's Dual Kriging Vector Engine achieves a **~180x speedup** over loop-based solvers (interpolating 300,000+ DEM grid points in under 50 milliseconds) while maintaining complete mathematical parity with standard Universal Kriging.

<a id="shallow-ice-approximation-custom-drift"></a>
#### 5. Shallow Ice Approximation Drift Model for Universal Kriging Interpolation
`PySole` offers a physically-informed custom drift model based on the **Shallow Ice Approximation (SIA)**. Re-arranging the basal shear stress <i>τ</i><sub>b</sub> for ice depth <i>D</i> yields the inverse relationship between <i>D</i>(<i>x</i>,<i>y</i>) and sin(<i>α</i>(<i>x</i>,<i>y</i>)). Setting `"drift_terms": ["sia_thickness"]` informs Universal Kriging of the **relative thickness distribution pattern** driven directly by the optimized DEM surface slope:

  <p align="center">
    <font><i>D</i> &prop; sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>))<sup>-1</sup></font>
  </p>

Thus, producing a terrain-conforming, physically realistic background trend across unmeasured gap regions without requiring assumptions about absolute <i>τ</i><sub>b</sub> values. The custom physical SIA drift model is available for both pre- and post-migration Universal Kriging interpolations, and is used by default for the final interpolation of migrated depth data.

<a id="depth-uncertainty-derivation"></a>
#### 6. Depth Uncertainty Derivation in Meters
Kriging interpolation provides uncertainty estimates by variance of the product field <i>σ</i><sub>P</sub><sup>2</sup>(<i>x</i>,<i>y</i>) [m<sup>2</sup>]. The 2D depth estimation variance field <i>σ</i><sub>D</sub><sup>2</sup>(<i>x</i>,<i>y</i>) [m<sup>2</sup>] is obtained via linear error propagation:

<p align="center">
  <i>σ</i><sub>D</sub><sup>2</sup>(<i>x</i>,<i>y</i>) = <i>σ</i><sub>P</sub><sup>2</sup>(<i>x</i>,<i>y</i>) / sin<sup>2</sup>(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>)) &nbsp;&nbsp; [m<sup>2</sup>]
</p>

Taking the square root converts the variance field into the **Kriging Standard Error <i>σ</i><sub>D</sub>(<i>x</i>,<i>y</i>) in meters**:

<p align="center">
  <i>σ</i><sub>D</sub>(<i>x</i>,<i>y</i>) = √(<i>σ</i><sub>D</sub><sup>2</sup>(<i>x</i>,<i>y</i>)) &nbsp;&nbsp; [± m]
</p>

Under Gaussian linear estimation theory, ± 1.00 <i>σ</i><sub>D</sub>(<i>x</i>,<i>y</i>) represents the 68.3% confidence margin of error, while ± 1.96 <i>σ</i><sub>D</sub>(<i>x</i>,<i>y</i>) represents the 95% confidence margin of error.

<a id="dem-spatial-smoothing"></a>
#### 7. Spatial Smoothing of the Calculated Depth and Bedrock DEMs
The depth field <i>D</i>(<i>x</i>,<i>y</i>) is obtained by dividing the Kriged product field <i>P</i><sub>D</sub>(<i>x</i>,<i>y</i>) with the optimal smoothed surface slope field sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>)). When post-processing DEM spatial smoothing is enabled (`smooth_bedrock: true`), `PySole` applies the spatial smoothing operator <i>S</i> **directly to the ice depth field <i>D</i>(<i>x</i>,<i>y</i>)**:

<p align="center" style="line-height: 1.8;">
  <i>D</i><sub>smooth</sub>(<i>x</i>,<i>y</i>) = <i>S</i>(<i>D</i>(<i>x</i>,<i>y</i>))<br>
  <i>Z</i><sub>bed</sub>(<i>x</i>,<i>y</i>) = <i>Z</i><sub>surface</sub>(<i>x</i>,<i>y</i>) − <i>D</i><sub>smooth</sub>(<i>x</i>,<i>y</i>)
</p>

Applying smoothing directly to <i>D</i>(<i>x</i>,<i>y</i>) prevents the high-frequency surface DEM roughness residual (<i>Z</i><sub>surface</sub> − <i>S</i>(<i>Z</i><sub>surface</sub>)) from superimposing rectangular grid artifacts onto the ice thickness map, ensuring that both <i>D</i>(<i>x</i>,<i>y</i>) and <i>Z</i><sub>bed</sub>(<i>x</i>,<i>y</i>) remain smooth and continuous. The available spatial smoothing operators are `"gaussian"`, `"median"`, and `"fft_lowpass"`.

<a id="multi-format-dem-export"></a>
#### 8. Multi-Format DEM Export
Under `outputs` in `pysole.json`, users can specify via `output_format` which file format(s) to export calculated depth and bedrock DEMs:<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;`output_format: "tif"` (or `"asc"`, `"csv"`, `"npy"`): Exports a single specified format.<br>
&nbsp;&nbsp;&nbsp;&nbsp;`output_format: ["tif", "asc", "csv", "npy"]`: Exports a list of specified formats.<br>
&nbsp;&nbsp;&nbsp;&nbsp;`output_format: "all"`: Exports all four formats simultaneously.

---

<a id="package-architecture"></a>
## Package Architecture

<p align="center">
  <a href="images/pysole_package_structure.png">
    <img src="images/pysole_package_structure.png" width="100%" alt="PySole Package Structure & Submodules">
  </a>
  <br>
  <em>Figure 2: Overview of PySole package submodules, class structure, and core processing functions. Solver functions and related submodules are color-coded. Click diagram to view in high resolution.</em>
</p>

---

<a id="cli-execution"></a>
## Command-Line Interface (CLI) Execution

`PySole` provides a convenient terminal entry point for executing processing workflows directly from your command line without writing Python scripts.

### 1. Convenient Execution & `$PATH` Setup

When you install `PySole` (`pip install .` or `pip install -e .`), `pip` automatically creates the `pysole` executable binary script in your Python environment's binary directory (`bin/` on Linux/macOS or `Scripts\` on Windows).

* **Virtual Environment (Recommended):**
  When your virtual environment is activated (`source .venv/bin/activate` or `conda activate myenv`), its `bin/` directory is automatically prepended to your `$PATH`. You can immediately execute `pysole` from **any** working directory:

  ```bash
  pysole pysole.json
  ```

* **User Installation without Virtual Environment (`pip install --user .`):**
  If installed without a virtual environment using `--user`, Python places entry point scripts in `~/.local/bin` (Linux/macOS) or `%APPDATA%\Python\Scripts` (Windows). If `pysole` is not recognized by your terminal, add `~/.local/bin` to your `$PATH` variable by placing the following line in your `~/.bashrc` or `~/.zshrc`:

  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  ```

### 2. Available CLI Commands

* **Generate a Default `pysole.json` Template:**
  ```bash
  pysole --init
  ```
  *(Creates a clean, fully commented `pysole.json` configuration file in your current working directory).*

* **Run the PySole Solver with a Configuration File:**
  ```bash
  pysole pysole.json
  ```
  *(Executes the full pipeline defined in `pysole.json` and exports predicted bedrock rasters and diagnostic plots).*

* **Display CLI Help & Usage Options:**
  ```bash
  pysole --help
  ```

---

<a id="python-api-and-quick-start"></a>
## Python API & Quick Start

`PySole` provides a dual-layer Python API: a **High-Level Solver Orchestrator** for streamlined pipeline execution, and **Decoupled Specialized Sub-Engines** for custom advanced research workflows.

### 1. High-Level One-Liner Execution

Run the complete pipeline from a `pysole.json` configuration file in a single line:

```python
import pysole

# Execute complete workflow defined in pysole.json
bedrock_map = pysole.run_from_config("pysole.json")
```

---

### 2. High-Level `Solver` Orchestrator API

Initialize the central `Solver` facade to steer the pipeline through processing milestones. When instantiated, `Solver` creates and manages underlying specialized sub-engine instances (`self.migrator`, `self.bss_optimizer`, `self.kriging_engine`, `self.finalizer`):

```python
import pysole

# 1. Initialize High-Level Solver Orchestrator
model = pysole.Solver(
    dem="surface_dem.asc",
    outline="creeping_body.shp",
    survey_data_type="one_way_travel_time",
    pre_kriging_method="universal",
    post_kriging_method="universal",
    perform_migration=True,
)

# Inspect underlying specialized sub-engine instances managed by Solver
print(model.geometry)        # GridGeometry spatial container instance
print(model.migrator)        # EikonalMigrator sub-engine instance
print(model.bss_optimizer)   # BSSOptimizer sub-engine instance
print(model.kriging_engine)  # KrigingEngine sub-engine instance
print(model.finalizer)       # BedrockFinalizer sub-engine instance

# 2. Migrate sparse GPR/Seismic travel times (delegates to model.migrator)
bedrock_pts = model.migrate_eikonal(
    travel_times="sparse_survey.csv",
    velocity=0.16,
)

# 3. Iterative BSS variance optimization (delegates to model.bss_optimizer)
model.optimize_bss(kc_max=10.0, kc_min=0.01, d_kc=0.1)

# 4. Primary Kriging spatial interpolation (delegates to model.kriging_engine)
kriged_bedrock, kriged_variance = model.interpolate_kriging(
    method="universal",
    plotit=True,
)

# 5. Finalize topography (delegates to model.finalizer for gap filling & margin blending)
bedrock_map = model.finalize_topography(
    interactive=True,
    smooth_bedrock=True,
    smoothing_method="gaussian",
)

# 6. Export predicted bedrock elevation raster
bedrock_map.save("final_bedrock.tif")
```

---

### 3. Decoupled Sub-Engines API (Standalone Custom Workflows)

Advanced users and researchers can bypass the `Solver` orchestrator entirely and instantiate specialized sub-engines directly for modular, custom research scripts:

```python
import pysole
from pysole import (
    load_dem,
    load_outline,
    load_survey_points,
    GridGeometry,
    EikonalMigrator,
    BSSOptimizer,
    KrigingEngine,
    BedrockFinalizer,
)

# 1. Load raster inputs and construct spatial GridGeometry container
dem_grid, meta = load_dem("surface_dem.asc")
outline_mask = load_outline("creeping_body.shp", dem_grid=dem_grid, meta=meta)
geometry = GridGeometry.create(dem_grid.shape, dx=meta["dx"], dy=meta["dy"], bounds=meta["bounds"])
survey_pts = load_survey_points("sparse_survey.csv", bounds=geometry.bounds, dem_grid=dem_grid)

# 2. Standalone 3D Ray Migration Sub-Engine (pysole.migration.EikonalMigrator)
migrator = EikonalMigrator(dem=dem_grid, geometry=geometry, outline_mask=outline_mask)
# mig_res = migrator.migrate(travel_time_grid=tt_grid, survey_points=survey_pts, velocity=0.16)

# 3. Standalone Basal Shear Stress Optimization Sub-Engine (pysole.variogram.BSSOptimizer)
bss_optimizer = BSSOptimizer(dem=dem_grid, geometry=geometry)
opt_res = bss_optimizer.optimize(survey_points=survey_pts, kc_max=10.0, kc_min=0.01, d_kc=0.1)
print(f"Optimal Corner Frequency: {opt_res.optimal_kc:.4f} rad/m")

# 4. Standalone Dual Kriging Vector Sub-Engine (pysole.interpolation.KrigingEngine)
krig_engine = KrigingEngine(dem=dem_grid, geometry=geometry, outline_mask=outline_mask)
krig_res = krig_engine.interpolate(sample_points=survey_pts, method="universal", variogram_model="spherical")
# Returns typed KrigingResult dataclass containing krig_res.bedrock_grid and krig_res.variance_grid

# 5. Standalone Bedrock Finalizer Sub-Engine (pysole.interpolation.BedrockFinalizer)
finalizer = BedrockFinalizer(dem=dem_grid, geometry=geometry, outline_mask=outline_mask)
rf_filled = finalizer.fill_holes(krig_res.bedrock_grid)
blended_bedrock = finalizer.blend_margin(rf_filled, min_gap_dist=50.0)
```

---

<a id="real-world-example-wurtenkees-glacier"></a>
## Real-World Example: Wurtenkees Glacier

You can run a complete real-world demonstration on the **Wurtenkees Glacier** dataset (Hohe Tauern, Eastern Alps, Austria) using the provided configuration file [`examples/wuk/pysole_wuk.json`](file:///home/db/Software/pysole/examples/wuk/pysole_wuk.json).

An executable example script is provided in [`examples/wuk/run_wuk_example.py`](file:///home/db/Software/pysole/examples/wuk/run_wuk_example.py):

```bash
python examples/wuk/run_wuk_example.py
```

The GPR dataset and DEM inputs are sourced from the Master's thesis by Binder (2011, written in German and available from [ResearchGate](https://www.researchgate.net/publication/369660356_Bestimmung_der_Eismachtigkeitsverteilung_dreier_Gletscher_der_Hohen_Tauern_auf_Basis_von_Ground_Penetrating_Radar_GPR_Daten)).

---

<a id="citation-and-references"></a>
## Citation & References

If you use `PySole` for your publications, please cite the underlying methodology introduced by Binder et al. (2009).

### APA
* **Binder, D., Brückl, E., Roch, K.H., Behm, M., Schöner, W., & Hynek, B. (2009).** Determination of total ice volume and ice-thickness distribution of two glaciers in the Hohe Tauern region, Eastern Alps, from GPR data. *Annals of Glaciology*, 50(51), 71–79. [doi:10.3189/172756409789097522](https://doi.org/10.3189/172756409789097522)
* **Binder, D. (2011).** *Bestimmung der Eismächtigkeitsverteilung dreier Gletscher der Hohen Tauern auf Basis von Ground Penetrating Radar (GPR) Daten* (Master's thesis, Vienna University of Technology, Vienna, Austria). Available from [ResearchGate](https://www.researchgate.net/publication/369660356_Bestimmung_der_Eismachtigkeitsverteilung_dreier_Gletscher_der_Hohen_Tauern_auf_Basis_von_Ground_Penetrating_Radar_GPR_Daten).
* **Matheron, G. (1981).** Splines and kriging: Their formal equivalence. In D. F. Merriam (Ed.), Down-to-Earth statistics: Solutions looking for geological problems (Vol. 8, pp. 77–95). Syracuse University. (Syracuse University Geology Contribution No. 8).
* **Webster, R., & Oliver, M. A. (2007).** Geostatistics for Environmental Scientists (2nd ed.). John Wiley & Sons. [doi:10.1002/9780470517277](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470517277)

### BibTeX
```bibtex
@article{binder2009determination,
  title        = {Determination of total ice volume and ice-thickness distribution of two glaciers in the Hohe Tauern region, Eastern Alps, from GPR data},
  author       = {Binder, Daniel and Br{\"u}ckl, Ewald and Roch, Karl-Heinz and Behm, Michael and Sch{\"o}ner, Wolfgang and Hynek, Bernhard},
  journal      = {Annals of Glaciology},
  volume       = {50},
  number       = {51},
  pages        = {71--79},
  year         = {2009},
  publisher    = {Cambridge University Press},
  doi          = {10.3189/172756409789097522}
}

@mastersthesis{binder2011bestimmung,
  author       = {Binder, Daniel},
  title        = {Bestimmung der Eism{\"a}chtigkeitsverteilung dreier Gletscher der Hohen Tauern auf Basis von Ground Penetrating Radar (GPR) Daten},
  school       = {Vienna University of Technology (TU Wien)},
  year         = {2011},
  address      = {Vienna, Austria},
  url          = {https://www.researchgate.net/publication/369660356_Bestimmung_der_Eismachtigkeitsverteilung_dreier_Gletscher_der_Hohen_Tauern_auf_Basis_von_Ground_Penetrating_Radar_GPR_Daten}
}

@incollection{matheron1981splines,
  author    = {Matheron, Georges},
  title     = {Splines and Kriging: Their Formal Equivalence},
  editor    = {Merriam, D. F.},
  booktitle = {Down-to-Earth Statistics: Solutions Looking for Geological Problems},
  series    = {Syracuse University Geology Contribution},
  volume    = {8},
  pages     = {77--95},
  publisher = {Syracuse University},
  address   = {Syracuse, New York},
  year      = {1981}
}

@book{webster2007geostatistics,
  title     = {Geostatistics for Environmental Scientists},
  author    = {Webster, Richard and Oliver, Margaret A.},
  edition   = {2nd},
  year      = {2007},
  publisher = {John Wiley \& Sons},
  address   = {Chichester, UK},
  isbn      = {978-0-470-84418-2},
  doi       = {10.1002/9780470517277}
}
```
