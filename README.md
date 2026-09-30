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
[JSON Configuration File](#json-configuration)<br><br>
[JSON Configuration Parameter Reference](#json-configuration-parameter-reference)<br><br>
[Technical & Methodological Notes](#technical-and-methodological-notes)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[1. Supported DEM Formats](#supported-dem-formats)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[2. Parsing of Rock Outcrop Input Files](#parsing-rock-outcrops)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3. Variogram Binning with Minimum Pair Threshold](#variogram-binning)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4. High-Performance Dual Kriging Vector Engine](#dual-kriging-vector-engine)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5. Recommended Interpolation Strategies](#interpolation-strategies)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6. Universal Kriging Drift Models](#universal-kriging-drift-models)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7. Depth Uncertainty Derivation](#depth-uncertainty-derivation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[8. Spatial Smoothing of the Calculated DEMs](#dem-spatial-smoothing)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[9. Conversion of Wavenumber to Wavelength](#wavenumber-to-wavelength-conversion)<br><br>
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
* **Automated High-Resolution Diagnostic Plots:** Automatically generates and exports diagnostic figures for each key processing milestone.
* **5 Supported Digital Elevation Model (DEM) Input and Output Formats:** Seamless loading and exporting of GeoTIFFs (`.tif`), ESRI ASCII Grids (`.asc`, `.txt`), 3-column-CSV matrices (`.csv`, [X, Y, Z]), 2D-CSV matrices (`.csv`, [Z<sub>nx</sub> x Z<sub>ny</sub>]), NumPy binary arrays (`.npy`), and in-memory NumPy 2D arrays (`np.ndarray`).
* **Strict CRS & Spatial Alignment Verification:** Performs strict verification across all input layers (DEM, boundary outline, survey points). If any layer uses a different Coordinate Reference System or falls outside the DEM spatial extent, processing halts with an explicit error.
* **Flexible Survey Data Types:** `PySole` accepts one- or two-way signal traveltimes as well as direct thickness/depth measurements as survey data type. In case of direct thickness/depth data the 3D migration is automatically skipped.
* 💡**DEM Surface Slope Smoothing💡:** The degree of DEM surface slope smoothing is crucial when estimating ice thickness with the <i>Shallow Ice Approximation</i> (SIA), which assumes a constant basal shear stress. By relaxing this rigid baseline constraint, Binder et al. (2009) derived an objective optimization criterion for the surface slope smoothing process, which is implemented in `PySole`. The optimal degree of surface slope smoothing is derived by enforcing minimum spatial variance in basal shear stress as the optimization criterion:
  <p align="center">
    <font size="+1"><b>min<sub><i>k</i><sub>c</sub></sub> Var<sub><i>xy</i></sub>(<i>τ</i><sub>b</sub>)</b></font>
  </p>

  In shallow ice dynamics, basal shear stress is given by:

  <p align="center">
    <font size="+1"><b><i>τ</i><sub>b</sub> = <i>ρ</i><sub>ice</sub> <i>g</i> <i>D</i> sin(<i>α</i>)</b></font>
  </p>

  where ice density, <i>ρ</i><sub>ice</sub>, and gravitational acceleration, <i>g</i>, are assumed to be constant. Thus, just the product of the two variables ice depth and surface slope, <i>P</i> = <i>D</i> sin(<i>α</i>), is evaluated during the optimization process. Surface slope smoothing is performed in the frequency domain using a <i>Fast Fourier Transform</i> (FFT) low-pass filter defined by the spatial corner wavenumber (<i>k</i><sub>c</sub>). The spatial variance of <i>τ</i><sub>b</sub> is then quantified via variogram analysis. An interactive mode allows users to test varying degrees of smoothing across wavenumber cutoffs and refine the variogram parameters. This surface slope optimization methodology is an integral component for interpolating both pre-migration wavefront traveltimes and post-migration depths. To accelerate the optimization process, both the FFT low-pass filtering and the corresponding product variogram evaluations are executed via multi-threaded CPU parallelization.
* **3D Ray-Based Migration:** `PySole` features an optional 3D ray-based migration—introduced by Binder et al. (2009) and engineered specifically to process geophysical signal traveltimes with sparse spatial coverage.
* **Kriging Interpolation:** Provides a native, numerically optimized, and parallelized 2D Kriging algorithm supporting both Ordinary and Universal Kriging. Two distinct interpolation strategies are recommended: <i>Ordinary Kriging</i> is recommended for interpolating basal shear stress (BSS) derived products based on the assumption of a constant spatial mean (no external drift). For direct interpolation of signal traveltimes or (migrated) depths, <i>Universal Kriging</i> is recommended using the custom `PySole` SIA-based drift model. Corresponding Kriging estimation uncertainty fields are calculated alongside all predicted grids.
* **Boundary Conditions:** Perimeter and rock outcrop margin boundary conditions (zero traveltime <i>T</i> = 0 s and zero thickness <i>D</i> = 0 m) are enforced by default (and can optionally be toggled off). Interior rock outcrops, or vector polygon holes, are natively parsed.
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
  <em>Figure 1:  End-to-end computational workflow of the PySole solver dual-path processing pipeline. Click diagram to view in high resolution.</em>
</p>

#### 1. DEM Loading & Metadata
Grid spacing (`dx`, `dy`) and spatial bounds are automatically extracted from GeoTIFF and ESRI ASCII DEM metadata headers (and autodetected for 3-column CSV tables). If target `dx` and `dy` pixel resolutions are specified, 2D bilinear grid resampling is performed automatically.

For headerless DEM formats (2D `.csv` matrices, `.npy`, or `np.ndarray`), spatial parameters (`origin`, `crs`, `dx`, `dy`) should be provided under `spatial_parameters` to build the spatial metadata object.

> [!IMPORTANT]
> All input datasets (DEM, survey picks, and outline geometries) **must share the same projected, metric coordinate system** (e.g., UTM in meters). Geographic coordinates (latitude/longitude in degrees) will cause invalid distance, surface slope, and basal shear stress calculations.

#### 2. Pre-Migration Traveltime Interpolation
Across spatial wavenumber cutoffs <i>k</i><sub>c</sub>, the point products of traveltime observations and corresponding low-pass filtered surface slopes, <i>P</i><sub>T,i</sub> = <i>T</i><sub>i</sub> sin(<i>α</i><sub>smoothed,i</sub>), are evaluated. Once the optimization criterion is satisfied, the optimally smoothed surface slope, sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>)), is deployed in the subsequent Kriging interpolation. By default, the recommended interpolation strategy depends on `pre_migration.interpolation_target`:
- **BSS-derived Product `"P"` (Default)**: `PySole` interpolates <i>P</i><sub>T,i</sub> using <b>Ordinary Kriging</b> with optional zero-traveltime boundary conditions (<i>T</i> = 0 s) by default to produce the continuous product field <i>P</i><sub>T</sub>(<i>x</i>,<i>y</i>). The continuous signal traveltime field <i>T</i>(<i>x</i>,<i>y</i>) is then reconstructed by dividing <i>P</i><sub>T</sub>(<i>x</i>,<i>y</i>) by the optimally smoothed surface slope field sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>)):

  <p align="center">
    <font><i>T</i>(<i>x</i>,<i>y</i>) = <i>P</i><sub>T</sub>(<i>x</i>,<i>y</i>) / sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>))</font>
  </p>

- **Direct `"T"`**: Directly interpolates signal traveltimes <i>T</i><sub>i</sub> using <b>Universal Kriging</b> with the <b>SIA physical drift model</b> by default.

- **Smoothed Surface Slope Safeguard**: `PySole` employs a user-defined minimum threshold of the smoothed surface slopes (`slope_floor_deg`) to prevent numerical division singularities in low-gradient regions (default is <i>5°</i>).

#### 3. 3D Ray-Based Migration
The migration algorithm solves the Eikonal equation to relocate subsurface reflection points, particularly improving the imaging of steep slopes and overdeepenings. An interactive mode allows users to test different signal propagation velocities and evaluate them through visualizations of migrated depths and the corresponding horizontal survey point displacements induced by the migration process.

#### 4. Post-Migration Surface Slope Optimization
Analogous to the pre-migration optimization pass, `PySole` applies the optimization criterion to determine the post-migration optimally smoothed surface slope, sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>)), across spatial wavenumber cutoffs <i>k</i><sub>c</sub>. The post-migration surface slope optimization evaluates, however, the products of migrated depths and corresponding smoothed surface slopes, <i>P</i><sub>D,i</sub> = <i>D</i><sub>i</sub> sin(<i>α</i><sub>smoothed,i</sub>).

#### 5. Final Depth Interpolation & Uncertainty Display
Reconstructs continuous thickness <i>D</i>(<i>x</i>,<i>y</i>) and bedrock elevation <i>Z</i><sub>bed</sub>(<i>x</i>,<i>y</i>) fields. By default, the recommended interpolation strategy depends on `pre_migration.interpolation_target`:
- **BSS-derived Product `"P"` (Default)**: Interpolates point products <i>P</i><sub>D,i</sub> = <i>D</i><sub>i</sub> sin(<i>α</i><sub>opt,i</sub>) using <b>Ordinary Kriging</b> with optional zero-thickness boundary conditions (<i>D</i> = 0 m) by default. The continuous product field, <i>P</i><sub>D</sub>(<i>x</i>,<i>y</i>), is divided by the optimally smoothed surface slope field, sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>)), to receive the final depth field <i>D</i>(<i>x</i>,<i>y</i>).
- **Direct `"D"`**: Directly interpolates (migrated) survey point depths <i>D</i><sub>i</sub> using <b>Universal Kriging</b> with the <b>SIA physical drift model</b> by default.

The Kriging standard error for the interpolated depths is converted to meters to quantify depth uncertainty.

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

<a id="json-configuration"></a>
## JSON Configuration File

All execution options can be fully defined in a single JSON configuration file, which by default is named `pysole.json`:

```json
{
    "inputs": {
        "dem_path": null,
        "outline_path": null,
        "survey_data_path": null,
        "survey_data_type": "one_way_travel_time",
        "ice_density": 900.0,
        "g": 9.81,
        "n_cores": -1,
        "log_level": "INFO",
        "show_progress": true
    },
    "spatial_parameters": {
        "dx": 5.0,
        "dy": 5.0,
        "bounds": null,
        "origin": null,
        "crs": null
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
        "engine": "native",
        "pre_migration": {
            "interpolation_target": "P",
            "method": "ordinary",
            "drift_terms": [],
            "variogram_model": "spherical",
            "include_zero_boundary_condition": true
        },
        "post_migration": {
            "interpolation_target": "P",
            "method": "ordinary",
            "drift_terms": [],
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
        "output_dir": null,
        "output_format": "tif",
        "output_prefix": "final",
        "plots_dir": "figures",
        "save_traveltime_grid": false,
        "save_migrated_points": false,
        "save_thickness_grid": false,
        "save_thickness_uncertainty": false,
        "save_basal_shear_stress": false,
        "save_basal_shear_stress_uncertainty": false
    }
}
```

---

<a id="json-configuration-parameter-reference"></a>
## JSON Configuration Parameter Reference

| Section | Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`inputs`** | `dem_path` | `str` | `null` | **(Required)** File path to the surface Digital Elevation Model (`.asc`, `.tif`, `.csv`, `.npy`). |
| | `outline_path` | `str` | `null` | File path to creeping body / glacier boundary polygon (`.shp`, `.geojson`, `.gpkg`, `.csv`). If `null`, domain is derived from non-NaN DEM pixels. |
| | `survey_data_path` | `str` | `null` | **(Required)** File path to signal traveltime or thickness observations CSV `[X, Y, value]`. |
| | `survey_data_type` | `str` | `"one_way_travel_time"` | Observation data type: `"one_way_travel_time"`, `"two_way_travel_time"`, or `"thickness"` / `"ice_thickness"` (skips 3D migration). |
| | `ice_density` | `float` | `900.0` | Density of the creeping medium in kg/m³ (`900.0` kg/m³ for temperate glacier ice by default). Used to calculate basal shear stress $\tau_{\text{b}}$. |
| | `g` | `float` | `9.81` | Gravitational acceleration constant in m/s² (`9.81` m/s²). Used to calculate basal shear stress $\tau_{\text{b}}$. |
| | `n_cores` | `int` | `-1` | Number of CPU cores applied across all parallelized processes (`-1` for all available cores). |
| | `log_level` | `str` | `"INFO"` | Package logging level verbosity: `"INFO"` (default), `"DEBUG"`, `"WARNING"`, `"ERROR"`, or `"CRITICAL"`. Appends timestamped logs to `pysole.log`. |
| | `show_progress` | `bool` | `true` | If `true` (default), displays terminal progress bars during heavy processing steps. Set to `false` if running `PySole` in batch processing scripts. |
| **`spatial_parameters`** | `dx` | `float` | `null` | Target grid resolution along X in meters. If defined, automatically resamples the DEM grid. If `null`, native resolution is kept. |
| | `dy` | `float` | `null` | Target grid resolution along Y in meters. If defined, automatically resamples the DEM grid. If `null`, native resolution is kept. |
| | `bounds` | `list[float]` | `null` | Optional spatial bounding box `[minx, miny, maxx, maxy]`. Leave `null` by default. Use only to manually override invalid or missing spatial bounds in DEM raster headers (`.asc`, `.tif`). `bounds` are automatically calculated from `origin`, `dx`, `dy`, and grid dimensions for headerless DEMs (`.csv`, `.npy`). |
| | `origin` | `list[float]` | `null` | Lower-left coordinate origin `[xll, yll]` in projected metric units for headerless DEM formats (`.csv`, `.npy`, `np.ndarray`). If `null`, defaults to `(0.0, 0.0)`. |
| | `crs` | `str` / `int` | `null` | Coordinate Reference System (e.g. `"EPSG:32633"`, `32633`, or PROJ string). Automatically checked upon load to enforce projected metric coordinate systems. GeoTIFF is the only self-contained format among the supported input formats that embeds spatial projection metadata (e.g., EPSG code or WKT string) directly inside the file header.  |
| **`migration_parameters`** | `perform_migration` | `bool` | `true` | If `true`, performs 3D ray-based migration on signal traveltimes. If `false`, migration is skipped. |
| | `velocity` | `float` | `0.16` | Signal propagation velocity (default value of `0.16` m/ns is characteristic for radar wave propagation in temperate ice). |
| | `interactive_migration` | `bool` | `false` | If `true`, enables interactive velocity testing with visual migrated depths and horizontal displacement vector plots. |
| **`optimization_parameters`** | `kc_max` | `float` | `10.0` | Maximum corner frequency for FFT Gaussian low-pass smoothing. |
| | `kc_min` | `float` | `0.01` | Minimum corner frequency for FFT Gaussian low-pass smoothing. |
| | `d_kc` | `float` | `0.1` | Corner frequency stepwidth  for FFT Gaussian low-pass smoothing. |
| | `nrbins` | `int` | `null` | Number of variogram lag distance bins. If `null` (default), dynamically calculated to receive ~30 point pairs per bin, and an absolute minimum floor of 3 bins. |
| | `slope_floor_deg` | `float` | `5.0` | Minimum surface slope angle threshold in degrees [°] enforced during surface slope optimization to prevent numerical division singularities. |
| | `interactive_optimization` | `bool` | `false` | If `true`, enables interactive CLI prompt to inspect BSS variance curve and adjust corner frequency spectrum parameters (`kc_min`, `kc_max`, `d_kc`), lag distance bin count (`nrbins`), and correlation range (`a_range`). |
| **`kriging_parameters`** | `engine` | `str` | `"native"` | Kriging calculation engine: `"native"` (default, high-performance Dual Kriging solver) or `"pykrige"` (uses external [`PyKrige`](https://geostat-framework.readthedocs.io/projects/pykrige) package - optional dependency in `pyproject.toml`). |
| | `pre_migration` | `dict` | *Sub-section* | Configuration for pre-migration traveltime field, <i>T</i>(<i>x</i>,<i>y</i>), interpolation. |
| | `pre_migration.interpolation_target` | `str` | `"P"` | Pre-migration interpolation targets: `"P"` for BSS-derived products (<i>P</i> = <i>T</i><sub>i</sub> · sin <i>α</i><sub>opt, i</sub>, default) or `"T"` for direct signal traveltimes (<i>T</i><sub>i</sub>). |
| | `pre_migration.method` | `str` | `"ordinary"` | Kriging approach: `"ordinary"` (the default for `pre_migration.interpolation_target`: `"P"`), `"universal"` (the default for `pre_migration.interpolation_target`: `"T"`), or `"regression"` (only available for `engine`: `"pykrige"`). |
| | `pre_migration.drift_terms` | `list[str]` | `[]` | Available drift models for Universal Kriging: `["sia_thickness"]` (SIA physical drift model - the default for `pre_migration.interpolation_target`: `"T"`), `["z_surface"]` (or `["dem"]` / `["elevation"]` - surface elevation drift model), `["regional_linear"]` (or `["x", "y"]` - 1st-order linear coordinate trend), or `["quadratic"]` (2nd-order quadratic coordinate trend). If left empty `[]` (the default for `pre_migration.interpolation_target`: `"P"`) applies a constant mean, resulting in Ordinary Kriging. |
| | `pre_migration.variogram_model` | `str` | `"spherical"` | Theoretical variogram model (`"spherical"`, `"exponential"`, `"gaussian"`, `"linear"`). |
| | `pre_migration.include_zero_boundary_condition` | `bool` | `true` | If `true` (default), includes zero traveltime boundary points (<i>T</i> = 0 s) along the perimeter and rock outcrop margin outline(s). |
| | `post_migration` | `dict` | *Sub-section* | Configuration for final bedrock depth, <i>D</i>(<i>x</i>,<i>y</i>), interpolation. |
| | `post_migration.interpolation_target` | `str` | `"P"` | Post-migration interpolation targets: `"P"` for BSS-derived products (<i>P</i> = <i>D</i><sub>i</sub> · sin <i>α</i><sub>opt, i</sub>, default) or `"T"` for direct (migrated) depths (<i>D</i><sub>i</sub>). |
| | `post_migration.method` | `str` | `"ordinary"` | Kriging approach: `"ordinary"` (the default for `post_migration.interpolation_target`: `"P"`), `"universal"` (the default for `post_migration.interpolation_target`: `"T"`), or `"regression"`. |
| | `post_migration.drift_terms` | `list[str]` | `[]` | Available drift models for Universal Kriging: `["sia_thickness"]` (SIA physical drift model, the default for `post_migration.interpolation_target`: `"D"`), `["z_surface"]` (or `["dem"]` / `["elevation"]` - surface elevation drift model), `["regional_linear"]` (or `["x", "y"]` - 1st-order linear coordinate trend), or `["quadratic"]` (2nd-order quadratic coordinate trend). If left empty `[]` (the default for `post_migration.interpolation_target`: `"P"`) applies a constant mean, resulting in Ordinary Kriging. |
| | `post_migration.variogram_model` | `str` | `"spherical"` | Theoretical variogram model (`"spherical"`, `"exponential"`, `"gaussian"`, `"linear"`). |
| | `post_migration.include_zero_boundary_condition` | `bool` | `true` | If `true` (default), includes zero thickness boundary points (<i>D</i> = 0 m) along the perimeter and rock outcrop margin outline(s). |
| **`finalization_parameters`** | `random_forest_gap_filling` | `bool` | `false` | If `true`, applies Random Forest machine learning gap filling across unmeasured interior regions. |
| | `apply_margin_blend` | `bool` | `false` | If `true`, applies geomorphological margin blending to seamlessly transition calculated bedrock elevation to surrounding surface DEM terrain. |
| | `min_gap_dist` | `float` | `50.0` | Minimum gap distance in meters inside which bedrock is smoothly tapered and blended into surface DEM terrain. |
| | `smooth_bedrock` | `bool` | `false` | If `true`, applies spatial DEM post-processing smoothing directly to the depth field <i>D</i>(<i>x</i>,<i>y</i>) to eliminate high-frequency slope-division noise. |
| | `smoothing_method` | `str` | `"gaussian"` | Final bedrock DEM smoothing algorithm choice: `"gaussian"` (default), `"median"`, or `"fft_lowpass"`. |
| | `smoothing_sigma` | `float` | `1.5` | Smoothing strength (radius in pixels) for `"gaussian"` filtering. Higher values produce smoother bedrock terrain. |
| | `smoothing_kernel_size` | `int` | `3` | Window kernel size (<i>k</i> × <i>k</i>) for `"median"` filtering (must be an odd integer). Higher values produce smoother bedrock terrain. |
| | `smoothing_kc_cutoff` | `float` | `null` | Corner frequency cutoff wavenumber (<i>k</i><sub>c,smooth</sub>) for `"fft_lowpass"`. If `null`, defaults to <i>k</i><sub>c,opt</sub> from the surface slope smoothing optimization. <i>Lower</i> values produce smoother bedrock terrain. |
| **`outputs`** | `output_dir` | `str` | `null` | General workspace directory for all output files. `"."` defines `output_dir` as the parent directory of the `pysole.json` configuration file. If `null` (default), automatically creates a `pysole` folder inside the parent directory of `survey_data_path`. All relative output paths are resolved relative to `output_dir`. Absolute output paths override `output_dir`. |
| | `output_format` | `str` / `list[str]` | `"tif"` | Desired export format(s): `"tif"`, `"asc"`, `"csv"`, `"npy"`, a list of formats (e.g. `["tif", "asc", "csv"]`), or `"all"` to export all four formats. |
| | `output_prefix` | `str` | `"final"` | Filename prefix or absolute filepath prefix used to construct export filenames. Do not include file extensions—format extension(s) are determined by `output_format`. Relative prefix names resolve inside `output_dir`; absolute filepaths override `output_dir`. 🚨 The final bedrock grid `<output_prefix>_bedrock.<ext>` is exported by default. 🚨 |
| | `plots_dir` | `str` | `"figures"` | Output directory for saving diagnostic figures. All generated figures are saved automatically. If `null`, figures are saved into a `"figures"` folder inside `output_dir`. An absolute path overrides `output_dir/figures`. |
| | `save_traveltime_grid` | `bool` | `false` | If `true`, exports the pre-migration interpolated traveltime raster grid, <i>T</i>(<i>x</i>,<i>y</i>), masked by the creeping body outline to `<output_prefix>_traveltime.<ext>`. |
| | `save_migrated_points` | `bool` | `false` | If `true`, exports the 3D ray-migrated survey points to `<output_prefix>_migrated_points.csv` (`x, y, z_surface, depth_migrated`). If migration is skipped, it is automatically set to `false` and a log message is printed. |
| | `save_thickness_grid` | `bool` | `false` | If `true`, exports the final thickness raster grid, <i>D</i>(<i>x</i>,<i>y</i>), masked by the creeping body outline to `<output_prefix>_thickness.<ext>`. |
| | `save_thickness_uncertainty` | `bool` | `false` | If `true`, exports the Kriging thickness uncertainty raster grid, <i>σ<sub>D</sub></i>(<i>x</i>,<i>y</i>), masked by the creeping body outline to `<output_prefix>_thickness_uncertainty.<ext>`. |
| | `save_basal_shear_stress` | `bool` | `false` | If `true`, exports the final basal shear stress raster grid, <i>τ</i><sub>b</sub>(<i>x</i>,<i>y</i>), masked by the creeping body outline to `<output_prefix>_basal_shear_stress.<ext>`. |
| | `save_basal_shear_stress_uncertainty` | `bool` | `false` | If `true`, exports the basal shear stress Kriging uncertainty raster grid, <i>σ</i><sub><i>τ</i><sub>b</sub></sub>(<i>x</i>,<i>y</i>), masked by the creeping body outline to `<output_prefix>_basal_shear_stress_uncertainty.<ext>`. |

---

<a id="technical-and-methodological-notes"></a>
## Technical & Methodological Notes

<a id="supported-dem-formats"></a>
#### 1. Supported DEM Formats
`PySole` supports 5 distinct DEM formats:

- **GeoTIFF (`.tif`, `.tiff`, `.geotiff`)**: *Recommended*. Automatically extracts spatial bounds, affine transform, CRS, pixel resolution, and `nodata` values using [`rasterio`](https://rasterio.readthedocs.io).
- **ESRI ASCII Grid (`.asc`, `.txt`)**: Standard 6-line header GIS raster format. Automatically extracts `ncols`, `nrows`, `xllcorner`, `yllcorner`, `cellsize`, and `nodata_value`. Optional CRS projection can be specified via `spatial_parameters.crs`.
- **CSV Grid (`.csv`)**: Supports both **2D elevation matrices** and **3-column XYZ grid tables** `(X, Y, Z)`. For 3-column tables, cell spacing (`dx`, `dy`) and bounds are automatically inferred; for 2D matrices, metadata is built from `spatial_parameters`.
- **NumPy Binary Array (`.npy`)**: Fast 2D binary array format. Spatial metadata (`origin`, `crs`, `dx`, `dy`) is defined via `spatial_parameters`.
- **NumPy 2D Array (`np.ndarray`)**: In-memory array passed directly into the `Solver` constructor (`dem=dem_grid`). Spatial metadata is defined via `spatial_parameters`.

<a id="parsing-rock-outcrops"></a>
#### 2. Parsing of Rock Outcrop Input Files
Vector polygon files (`.shp`, `.geojson`, `.gpkg`) or CSV outline files containing interior rings (separated by `NaN` rows) are automatically parsed as polygon holes. `PySole` treats pixels inside rock outcrop holes as exposed bedrock (<i>Z</i> = <i>Z</i><sub>Surface</sub>). When boundary conditions are enabled, the perimeters of interior rings are included, applying <i>T</i> = 0 s or <i>D</i> = 0 m.

For rock outcrop holes to be detected correctly from a Shapefile (`.shp`):

- **Topology**: The outcrop must be stored as an interior ring (`polygon.interiors`) within a single `Polygon` / `MultiPolygon` feature (e.g. created using QGIS "Add Ring" or ArcGIS "Construct Hole").
- **Winding Order**: Standard OGC orientation (Clockwise exterior boundary, Counter-Clockwise interior hole rings).
- **CRS Alignment**: The shapefile's Coordinate Reference System must match the DEM raster projection.
- **Valid Geometries**: Rings must not intersect themselves (`PySole` automatically executes `validate_and_extract_polygons()` on load to auto-repair geometries or fall back to the outer boundary shell if holes fail criteria).

<a id="variogram-binning"></a>
#### 3. Variogram Binning with Minimum Pair Threshold
Experimental variogram lag distance bins are calculated from the pairwise Euclidean distances across a total of <i>N</i> survey points. Users can specify a fixed number of lag bins via `nrbins` under `optimization_parameters` in `pysole.json`, or during the `interactive_optimization` procedure. When `nrbins` is set to `null` (default), PySole initially determines a minimum distance bin count based on the total number of survey point pairs ($N_{\text{pairs}} = \frac{N(N-1)}{2}$):

<p align="center">
  <i>nrbins</i> = max( 3, <i>N</i><sub>pairs</sub> / 30 )
</p>

Enforcing a minimum threshold of at least **30 point pairs per lag bin** aligns with established geostatistical literature (e.g. Webster and Oliver, 2007), ensuring robust experimental variogram estimation and stable theoretical model curve fitting. If a user-specified `nrbins` yields, however, fewer than 30 average point pairs per bin, a diagnostic warning is emitted while honoring the user's explicit bin choice. An absolute lower floor of 3 lag distance bins is enforced across all calculations.

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

To guarantee numerical stability during matrix decomposition, diagonal Tikhonov regularization adds a small offset (10<sup>−6</sup>) to the main diagonal of <i>K</i>, ensuring positive-definiteness and preventing matrix singularities. Combined with zero-centered spatial coordinate normalization and multi-threaded CPU chunk parallelization (`ThreadPoolExecutor`), PySole's Dual Kriging Vector Engine achieves a **~180x speedup** over loop-based solvers (interpolating 300,000+ DEM grid points in under 50 milliseconds) while maintaining complete mathematical parity with standard Universal Kriging.

<a id="interpolation-strategies"></a>
#### 5. Recommended Interpolation Strategies
While users can combine any available interpolation options, two primary strategies are recommended. These are implemented in `PySole` as the default approaches based on the target variable (`interpolation_target`: `"P"`, `"T"`, or `"D"`):

- **BSS-derived Product `"P"` Strategy**: Interpolates the BSS-derived product field <i>P</i> = <i>T</i> · sin <i>α</i><sub>opt</sub> (pre-migration) or <i>P</i> = <i>D</i> · sin <i>α</i><sub>opt</sub> (post-migration). **Ordinary Kriging** is recommended for BSS-derived product targets.<br>
  > [!IMPORTANT]
  > Interpolating a BSS-derived product `"P"` with Universal Kriging and the `"sia_thickness"` drift model creates a 1/sin<sup>2</sup>(<i>α</i>) double-scaling artifact. This artifact leads to implausibly large depths at low slopes and is therefore strongly discouraged.
- **Direct `"T"` or `"D"` Strategy**: Directly interpolates signal traveltimes <i>T</i><sub>i</sub> (pre-migration), or (migrated) depths <i>D</i><sub>i</sub>. **Universal Kriging** with the `"sia_thickness"` drift model is recommended.

<a id="universal-kriging-drift-models"></a>
#### 6. Universal Kriging Drift Models
The following drift models are implemented in `PySole` for Universal Kriging:

- **Shallow Ice Approximation Physical Drift Model (`["sia_thickness"]`)**:
   `PySole` offers the physically-informed custom `"sia_thickness"` drift model. Re-arranging the basal shear stress <i>τ</i><sub>b</sub> for ice depth <i>D</i> yields the inverse relationship between <i>D</i>(<i>x</i>,<i>y</i>) and sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>)). Setting the drift term parameter to `["sia_thickness"]` informs Universal Kriging of the relative thickness distribution pattern driven directly by the optimized DEM surface slope:

   <p align="center">
     <i>D</i> &prop; sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>))<sup>-1</sup>
   </p>

   Thus, producing a terrain-conforming, physically realistic background trend across unmeasured gap regions without requiring assumptions about absolute <i>τ</i><sub>b</sub> values. The custom physical SIA drift model is available for both pre- and post-migration Universal Kriging interpolations, and is the default for `interpolation_target`: `"T"` or `"D"`. A surface slope floor safeguard (`slope_floor_deg`, default 5.0°) clamps ultra-low slope angles prior to computing the inverse-sine drift, preventing matrix singularities.

- **Surface Elevation Drift Model (`["z_surface"`], [`"dem"`], or [`"elevation"`])**:
   Uses the DEM surface elevation <i>Z</i><sub>surface</sub>(<i>x</i>,<i>y</i>) as a spatial drift variable:
   <p align="center">
     <i>U</i>(<i>x</i>,<i>y</i>) = <i>Z</i><sub>surface</sub>(<i>x</i>,<i>y</i>)
   </p>
   This models the glaciological elevation-dependent ice thickness pattern (thicker ice in lower valley basins/confluences, thinner ice on high-altitude ridges) without relying on surface slope angles.

- **Linear Surface Drift Model (`["regional_linear"`] or `["x", "y"]`)**:
   Fits a 1st-order bivariate spatial coordinate trend surface across the <i>X</i> and <i>Y</i> grid axes:
   <p align="center">
     <i>U</i>(<i>x</i>,<i>y</i>) = <i>a</i><sub>1</sub> <i>X</i> + <i>a</i><sub>2</sub> <i>Y</i>
   </p>

- **Quadratic Surface Drift Model (`["quadratic"`] or `["x", "y", "x2", "y2", "xy"]`)**:
   Fits a 2nd-order bivariate polynomial trend surface across <i>X</i> and <i>Y</i> coordinates:
   <p align="center">
     <i>U</i>(<i>x</i>,<i>y</i>) = <i>a</i><sub>1</sub> <i>X</i> + <i>a</i><sub>2</sub> <i>Y</i> + <i>a</i><sub>3</sub> <i>X</i><sup>2</sup> + <i>a</i><sub>4</sub> <i>Y</i><sup>2</sup> + <i>a</i><sub>5</sub> <i>X Y</i>
   </p>

- **Empty Drift Model (`[]`)**:
   An empty drift model parameter assumes a constant local spatial mean (no external drift), which mathematically equates to an Ordinary Kriging approach.

<a id="depth-uncertainty-derivation"></a>
#### 7. Depth Uncertainty Derivation in Meters
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
#### 8. Spatial Smoothing of the Calculated Depth and Bedrock DEMs
The depth field <i>D</i>(<i>x</i>,<i>y</i>) is obtained by dividing the Kriged product field <i>P</i><sub>D</sub>(<i>x</i>,<i>y</i>) with the optimally smoothed surface slope field sin(<i>α</i><sub>opt</sub>(<i>x</i>,<i>y</i>)). When post-processing DEM spatial smoothing is enabled (`smooth_bedrock: true`), `PySole` applies the spatial smoothing operator <i>S</i> **directly to the ice depth field <i>D</i>(<i>x</i>,<i>y</i>)**:

<p align="center" style="line-height: 1.8;">
  <i>D</i><sub>smooth</sub>(<i>x</i>,<i>y</i>) = <i>S</i>(<i>D</i>(<i>x</i>,<i>y</i>))<br>
  <i>Z</i><sub>bed</sub>(<i>x</i>,<i>y</i>) = <i>Z</i><sub>surface</sub>(<i>x</i>,<i>y</i>) − <i>D</i><sub>smooth</sub>(<i>x</i>,<i>y</i>)
</p>

Applying smoothing directly to <i>D</i>(<i>x</i>,<i>y</i>) prevents the high-frequency surface DEM roughness residual (<i>Z</i><sub>surface</sub> − <i>S</i>(<i>Z</i><sub>surface</sub>)) from superimposing rectangular grid artifacts onto the ice thickness map, ensuring that both <i>D</i>(<i>x</i>,<i>y</i>) and <i>Z</i><sub>bed</sub>(<i>x</i>,<i>y</i>) remain smooth and continuous. The available spatial smoothing operators are `"gaussian"`, `"median"`, and `"fft_lowpass"`.

<a id="wavenumber-to-wavelength-conversion"></a>
#### 9. Conversion of Wavenumber to Wavelength
In `PySole` 2D lowpass spatial smoothing operates in the discrete frequency domain. Spatial wavenumber components along the orthogonal grid axes <i>X</i> and <i>Y</i> are constructed as:

<p align="center">
  <i>k</i><sub>x</sub> = <i>f</i><sub>x,pixel</sub> · (2&pi; · |<i>dx</i>|) &nbsp;&nbsp;&nbsp;&nbsp; [rad]<br>
  <i>k</i><sub>y</sub> = <i>f</i><sub>y,pixel</sub> · (2&pi; · |<i>dy</i>|) &nbsp;&nbsp;&nbsp;&nbsp; [rad]
</p>

where <i>f</i><sub>x,pixel</sub>, <i>f</i><sub>y,pixel</sub> &in; [−0.5, +0.5] are discrete frequencies in **[cycles / pixel]**, and <i>dx</i>, <i>dy</i> are grid pixel spacings in **[meters / pixel]**.

Because physical spatial frequencies are <i>f</i><sub>x,phys</sub> = <i>f</i><sub>x,pixel</sub> / <i>dx</i> and <i>f</i><sub>y,phys</sub> = <i>f</i><sub>y,pixel</sub> / <i>dy</i> [cycles / m], the physical spatial wavenumbers <i>k</i><sub>x,phys</sub>, <i>k</i><sub>y,phys</sub> [rad / m] relate to the code wavenumbers by:

<p align="center">
  <i>k</i><sub>x,phys</sub> = 2&pi; <i>f</i><sub>x,phys</sub> = <i>k</i><sub>x</sub> / <i>dx</i><sup>2</sup> &nbsp;&nbsp; [rad / m]<br>
  <i>k</i><sub>y,phys</sub> = 2&pi; <i>f</i><sub>y,phys</sub> = <i>k</i><sub>y</sub> / <i>dy</i><sup>2</sup> &nbsp;&nbsp; [rad / m]
</p>

Thus, the directional physical spatial cutoff wavelengths &lambda;<sub>c,x</sub> and &lambda;<sub>c,y</sub> [meters] corresponding to a corner frequency cutoff <i>k</i><sub>c</sub> are:

<p align="center">
  &lambda;<sub>c,x</sub> = 2&pi; / <i>k</i><sub>x,phys</sub> = (2&pi; · <i>dx</i><sup>2</sup>) / <i>k</i><sub>c</sub> &nbsp;&nbsp; [m]<br>
  &lambda;<sub>c,y</sub> = 2&pi; / <i>k</i><sub>y,phys</sub> = (2&pi; · <i>dy</i><sup>2</sup>) / <i>k</i><sub>c</sub> &nbsp;&nbsp; [m]
</p>

The overall 2D effective spatial cutoff wavelength &lambda;<sub>c,eff</sub> (geometric mean across both coordinate axes) is:

<p align="center">
  &lambda;<sub>c,eff</sub> = &radic;(&lambda;<sub>c,x</sub> · &lambda;<sub>c,y</sub>) = (2&pi; · |<i>dx</i> · <i>dy</i>|) / <i>k</i><sub>c</sub> = (2&pi; · <i>ds</i><sup>2</sup>) / <i>k</i><sub>c</sub> &nbsp;&nbsp; [meters]
</p>

where <i>ds</i> = &radic;(|<i>dx</i> · <i>dy</i>|) represents the effective spatial grid cell resolution (or grid cell area scale <i>ds</i><sup>2</sup> = |<i>dx</i> · <i>dy</i>|).

For example, on an isotropic grid with <i>dx</i> = <i>dy</i> = 5.0 m (<i>ds</i> = 5.0 m, <i>ds</i><sup>2</sup> = 25.0 m<sup>2</sup>), an optimal corner frequency <i>k</i><sub>c,opt</sub> = 0.4000 corresponds to a physical spatial cutoff wavelength:

<p align="center">
  &lambda;<sub>c,opt</sub> = (2&pi; · 25.0) / 0.4000 &approx; 392.70 meters
</p>

This physical cutoff wavelength is reported alongside <i>k</i><sub>c,opt</sub> in the `PySole` logging outputs (`pysole.log`).

---

<a id="package-architecture"></a>
## Package Architecture

<p align="center">
  <a href="images/pysole_package_structure.png">
    <img src="images/pysole_package_structure.png" width="100%" alt="PySole Package Structure & Submodules">
  </a>
  <br>
  <em>Figure 2: Overview of PySole package architecture, class structure, sub-engine modules, and API methods. High-level Solver methods are color-coded by their underlying sub-engine domain. Click diagram to view in high resolution.</em>
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
  If installed without a virtual environment using `--user`, Python places entry point scripts in `~/.local/bin` (Linux/macOS) or `%APPDATA%\Python\Scripts` (Windows). If `pysole` is not recognized by your terminal, add `~/.local/bin` to your `$PATH` variable by placing the following line in your `~/.bashrc`:

  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  ```

### 2. Available CLI Commands

* **Generate a Default `pysole.json` Template:**
  ```bash
  pysole --init
  ```
  *(Creates a clean template `pysole.json` configuration file in your current working directory).*

* **Run PySole with Default `pysole.json` Configuration:**
  ```bash
  pysole
  ```
  *(Automatically loads and executes `pysole.json` in the current working directory).*

* **Run PySole with a Custom Configuration File:**
  ```bash
  pysole path/to/custom_config.json
  ```
  *(Executes the full pipeline defined in `custom_config.json`).*

* **Run with Verbose / Debug Logging:**
  ```bash
  pysole pysole.json -v
  # or
  pysole pysole.json --verbose
  ```
  *(Enables `DEBUG` level logging verbosity for detailed computational diagnostics. Acts as a temporary CLI runtime override taking precedence over the `log_level` defined in `pysole.json`).*

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

# Execute complete workflow using default 'pysole.json'
final_bedrock = pysole.run_from_config()

# Or execute with a custom configuration file and log level override
final_bedrock = pysole.run_from_config("custom_config.json", log_level="DEBUG")

# Inspect spatial metadata and export final bedrock raster
print(final_bedrock.shape, final_bedrock.bounds)
final_bedrock.save("final_bedrock.tif")
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

# Option A: Run complete end-to-end pipeline in a single call
final_bedrock = model.run_pipeline(survey_data_path="sparse_survey.csv")

# Option B: Steer workflow step-by-step through individual milestones:
# 2. Migrate sparse GPR/Seismic traveltimes (delegates to model.migrator)
bedrock_pts = model.migrate_eikonal(
    travel_times="sparse_survey.csv",
    velocity=0.16,
)

# 3. Iterative BSS variance optimization (delegates to model.bss_optimizer)
model.optimize_bss(kc_max=10.0, kc_min=0.01, d_kc=0.1)

# 4. Primary Kriging spatial interpolation (delegates to model.kriging_engine)
kriged_bedrock, kriged_variance = model.interpolate_kriging(
    method="universal",
)

# 5. Finalize topography (delegates to model.finalizer for gap filling & margin blending)
final_bedrock = model.finalize_topography(
    interactive=False,
    smooth_bedrock=True,
    smoothing_method="gaussian",
)

# 6. Export predicted bedrock elevation raster
final_bedrock.save("final_bedrock.tif")
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

If you use `PySole` in your research, please cite the underlying methodology introduced by Binder et al. (2009):

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
```

### Full Reference List (APA)
* **Binder, D., Brückl, E., Roch, K.H., Behm, M., Schöner, W., & Hynek, B. (2009).** Determination of total ice volume and ice-thickness distribution of two glaciers in the Hohe Tauern region, Eastern Alps, from GPR data. *Annals of Glaciology*, 50(51), 71–79. [doi:10.3189/172756409789097522](https://doi.org/10.3189/172756409789097522)
* **Binder, D. (2011).** *Bestimmung der Eismächtigkeitsverteilung dreier Gletscher der Hohen Tauern auf Basis von Ground Penetrating Radar (GPR) Daten* (Master's thesis, Vienna University of Technology, Vienna, Austria). Available from [ResearchGate](https://www.researchgate.net/publication/369660356_Bestimmung_der_Eismachtigkeitsverteilung_dreier_Gletscher_der_Hohen_Tauern_auf_Basis_von_Ground_Penetrating_Radar_GPR_Daten).
* **Matheron, G. (1981).** Splines and kriging: Their formal equivalence. In D. F. Merriam (Ed.), Down-to-Earth statistics: Solutions looking for geological problems (Vol. 8, pp. 77–95). Syracuse University. (Syracuse University Geology Contribution No. 8).
* **Webster, R., & Oliver, M. A. (2007).** Geostatistics for Environmental Scientists (2nd ed.). John Wiley & Sons. [doi:10.1002/9780470517277](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470517277)
