<a id="survey_planner"></a>
# Survey Planner Guide

This document provides a comprehensive methodological and practical reference for the **Unprobed Glacier Survey Planner (`SurveyPlanner`)** in `PySole`, a forward campaign design engine for generating synthetic Shallow Ice Approximation (SIA) ice thickness targets and laying out optimal longitudinal and transverse survey tracks.

> [!TIP]
> For post-survey Universal Kriging drift evaluation, cross-validation diagnostics, and multi-model comparisons once sounding observations have been collected, see the dedicated [Universal Kriging Drift Analyzer Guide](drift_analyzer.md).

---

## 1. Algorithmic Workflow & Overview

When a user provides a surface DEM (`dem_path`) and glacier outline (`outline_path`), but **no survey dataset** (`survey_data_path = null`), `PySole` automatically dispatches the **Survey Planner**.

```
[ Surface DEM & Glacier Outline ]
               │
               ▼
┌──────────────────────────────┐
│  FFT Gaussian DEM Smoothing  │ ──► Compute Z_smooth,kc(x,y)
│    (Corner Frequency k_c)    │ ──► Derive Clamped Slope alpha_clamped(x,y)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Synthetic SIA Target Modeling│ ──► D_SIA(x,y) = tau_0 / (rho * g * sin(alpha))
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Track Layout Algorithm    │ ──► Longitudinal Central Flowline (L_1)
│    (Budget Limit L_max)      │ ──► Transverse Cross-Profiles (T_1 ... T_M)
└──────────────┬───────────────┘
               │
               ▼
[ Export: GeoTIFF, GPX, GeoJSON, Map Plot ]
```

---

### 1.1 Required User Input Data & Default Parameters

To execute forward campaign survey planning on unprobed glaciers, `SurveyPlanner` requires two core spatial input files:

1. **Surface DEM (`dem` / `dem_path`)**: High-resolution Digital Elevation Model raster (GeoTIFF `.tif`, ESRI ASCII `.asc`, or NumPy `ndarray`) containing surface elevation values $Z_{\text{dem}}(x,y)$ [m].
2. **Glacier Boundary Outline (`outline` / `outline_path`, Optional)**: Vector boundary polygon (Shapefile `.shp`, GeoJSON `.geojson`, or 2D boolean mask) delineating the active extent of the creeping body. If `outline_path` is `null` (or omitted), `PySole` automatically detects `NaN` entries in the DEM raster to construct the active domain mask:
   - **Outer Perimeter Margins**: Exterior `NaN` padding surrounding the creeping body defines the outer boundary shell.
   - **Interior Rock Outcrops (Nunataks)**: Interior `NaN` entries within the glacier domain are automatically delineated as rock outcrop holes where SIA ice thickness is masked to zero ($D_{\text{SIA}} = 0\text{ m}$).
   - **Automated Logging**: Emits an `INFO` log detailing the count of `NaN` pixels and valid domain pixels. If no `NaN` values exist in the DEM and no outline is provided, emits a `WARNING` indicating that the full rectangular grid will be treated as active domain without boundary constraints.

#### Default Engine Parameters
When optional parameters are omitted, `SurveyPlanner` enforces the following default values:

| Parameter | Identifier / Argument | Default Value | Physical / Technical Description |
| :--- | :--- | :--- | :--- |
| **FFT DEM Cutoff Wavelength** | `lambda_c` / `--lambda-c` (alias `--wavelength`) or `kc` / `--kc` | **`200.0`** m ($k_c \approx 0.0314\text{ rad/m}$) | Physical spatial corner cutoff wavelength ($\lambda_c = 200.0\text{ m}$, corresponding to angular corner wavenumber frequency $k_c = 2\pi / \lambda_c \approx 0.0314\text{ rad/m}$) applied during 2D Fourier frequency-domain Gaussian DEM surface smoothing. |
| **Nominal Basal Shear Stress** | `tau_0` / `tau_p` / `--tau` | **`100`** kPa ($100,000\text{ Pa}$) | Target nominal basal shear stress ($\tau_0 = 100\text{ kPa}$) used in Shallow Ice Approximation (SIA) ice thickness depth modeling $D_{\text{SIA}}(x,y) = \frac{\tau_0}{\rho g \sin \alpha}$. |
| **Total Survey Track Budget** | `max_length_km` / `max_km` / `--max-km` | **`5.0`** km ($5,000\text{ m}$) | Maximum total survey length budget ($L_{\text{max}} = 5.0\text{ km}$) allocated across longitudinal flowlines and transverse cross-profiles. |
| **Minimum Surface Slope Floor** | `slope_floor_deg` | `5.0`° | Minimum surface slope angle threshold ($\alpha_{\text{floor}} = 5^\circ$) to prevent numerical division singularities in low-gradient accumulation basins. |
| **Ice Density** | `ice_density` | `900.0` kg/m³ | Bulk ice density ($\rho_{\text{ice}} = 900\text{ kg/m}^3$). |
| **Gravitational Acceleration** | `g` | `9.81` m/s² | Standard gravitational acceleration ($g = 9.81\text{ m/s}^2$). |

---

### 1.2 Synthetic SIA Ice Thickness Modeling

#### 1. Frequency-Domain Surface DEM Smoothing
Raw DEM elevations $Z_{\text{dem}}(x,y)$ contain high-frequency micro-topography. `PySole` applies a 2D Gaussian low-pass filter in the frequency domain using spatial corner wavenumber $k_c$ (default $k_c = 0.0314\text{ rad/m}$):

$$Z_{\text{smooth}, k_c}(x,y) = \text{Re}\left[ \mathcal{F}^{-1} \left( \mathcal{F}\left[ Z_{\text{dem}}(x,y) \right] \cdot \exp\left( -\frac{k_x^2 + k_y^2}{2 k_c^2} \right) \right) \right]$$

where $k_x, k_y$ are spatial wavenumbers ($2\pi / \lambda$).

#### 2. Surface Slope Angle Derivation & Clamping
Spatial surface slope gradients are calculated via 2nd-order central finite differences:

$$\frac{\partial Z_{\text{smooth}}}{\partial x}, \quad \frac{\partial Z_{\text{smooth}}}{\partial y}$$

$$\alpha(x,y) = \arctan\left( \sqrt{\left(\frac{\partial Z_{\text{smooth}}}{\partial x}\right)^2 + \left(\frac{\partial Z_{\text{smooth}}}{\partial y}\right)^2} \right)$$

To prevent division by zero or extreme thickness over-estimates in flat regions (e.g., upper accumulation basins), slope angles are clamped against a minimum slope floor $\alpha_{\text{floor}}$ (default 5.0°):

$$\alpha_{\text{clamped}}(x,y) = \max(\alpha(x,y), \, \alpha_{\text{floor}})$$

#### 3. Shallow Ice Approximation Thickness Field $D_{\text{SIA}}(x,y)$
Assuming ice density $\rho_{\text{ice}} = 900\text{ kg/m}^3$, gravitational acceleration $g = 9.81\text{ m/s}^2$, and nominal basal shear stress $\tau_0 = 100\text{ kPa}$:

$$D_{\text{SIA}}(x,y) = \frac{\tau_0}{\rho_{\text{ice}} \, g \, \sin(\alpha_{\text{clamped}}(x,y))}$$

The thickness grid is masked to zero outside the active glacier outline $O(x,y)$:

$$D_{\text{SIA}}(x,y) = 0 \quad \forall (x,y) \notin O(x,y)$$

---

### 1.3 Survey Profile Track Layout Algorithm

The survey planner automatically designs field campaign GPR/seismic tracks subject to a total track length budget $L_{\text{max}}$ (default $L_{\text{max}} = 5.0\text{ km}$).

#### 1. Central Longitudinal Flowline ($L_1$)
- **Maximum Thickness Axis**: The planner identifies the primary ice flow column $j_{\text{center}}$ corresponding to maximum integrated synthetic ice volume:

  $$j_{\text{center}} = \arg\max_j \sum_{i=1}^{N_y} D_{\text{SIA}}(i, j)$$

- **Profile Coordinates**: Traces the central longitudinal profile coordinates along $j_{\text{center}}$ across all active ice nodes ($D_{\text{SIA}} > 0$).
- **Budget Allocation**: Allocates up to $\sim 40\%$ of total budget $L_{\text{max}}$ to the main longitudinal profile:

  $$L_{\text{long}} = \min(L_{\text{flowline}}, \, 0.40 \cdot L_{\text{max}})$$

#### 2. Transverse Cross-Profiles ($T_1, T_2, \dots, T_M$)
- **Remaining Budget**: $L_{\text{rem}} = L_{\text{max}} - L_{\text{long}}$.
- **Number of Cross-Profiles ($M$)**:

  $$M = \max\left(2, \, \left\lfloor \frac{L_{\text{rem}}}{0.20 \cdot L_{\text{max}}} \right\rfloor\right)$$

- **Row Distribution**: $M$ cross-profile row indices $i_1, \dots, i_M$ are spaced evenly across the central $15\%\text{--}85\%$ of the active glacier length.
- **Margin-to-Margin Spanning**: For each cross-profile row $i_k$, coordinates span orthogonal to the main axis across active ice columns where $D_{\text{SIA}}(i_k, j) > 0$.

---

### 1.4 Output Formats & Field Exports

The `SurveyPlanner` generates four field-ready deliverables:

| Deliverable | File Naming Pattern | Description |
| :--- | :--- | :--- |
| **GeoTIFF Raster** | `<output_prefix>_sia_modelled_depth.tif` | 2D synthetic SIA ice thickness grid $D_{\text{SIA}}(x,y)$ [m]. |
| **GPX Vector** | `<output_prefix>_survey_plan.gpx` | GPS track file containing `<trk>` segments and `<trkpt>` nodes for field navigation. |
| **GeoJSON Vector** | `<output_prefix>_survey_plan.geojson` | Standard GIS LineString features with `track_id`, `length_m`, and `max_depth_m` attributes. |
| **Map Plot** | `<plots_dir>/<output_prefix>_survey_plan_map.png` | Diagnostic map plot displaying $D_{\text{SIA}}(x,y)$ color map overlaid with longitudinal ($L_1$, red) and transverse ($T_i$, cyan) tracks. |

---

## 2. Configuration & CLI Examples

### 2.1 Automated Dispatch via JSON (`pysole.json`)

When `inputs.survey_data_path` is omitted or set to `null` in `pysole.json`, `PySole` automatically dispatches the Survey Planner using the configured DEM raster and glacier boundary outline:

```json
{
  "inputs": {
    "dem_path": "surface_dem.tif",
    "outline_path": "glacier_outline.shp",
    "survey_data_path": null
  },
  "outputs": {
    "output_dir": "results",
    "output_prefix": "final"
  }
}
```

### 2.2 CLI Commands

```bash
# Interactive Survey Planning (prompts for omitted parameters: tau_0, lambda_c, max-km)
pysole plan-survey --dem surface_dem.tif --outline glacier_outline.shp

# Fully specified non-interactive execution (or using --batch)
pysole plan-survey --dem surface_dem.tif --outline glacier_outline.shp --lambda-c 200 --tau 100 --max-km 5.0 --out-dir results --format tif
```

#### `pysole plan-survey` CLI Options & Path Resolution

Computes the shallow-ice-approximation (SIA) thickness model and proposes survey tracks within a length budget.

- **Interactive Parameter Configuration**:
  When running via the Python API (`SurveyPlanner.plan_survey()`, `Solver.plan_survey()`, or automated dispatch when `inputs.survey_data_path` is `null`/omitted) or the CLI (`pysole plan-survey`), PySole interactively prompts for any omitted parameters (`tau_0`, `lambda_c`, `max_length_km`) via terminal `input()`. Each prompt displays the default value in brackets (e.g. `[default: 200.0]`). Pressing <kbd>Enter</kbd> (empty response) automatically adopts the default value. Non-numeric or negative entries trigger a helpful warning and re-prompt. In non-TTY environments (CI, background scripts, pipes) or when `--batch` / `interactive=False` is passed, prompts are bypassed and defaults are used automatically.

- **Options:**
  - `--dem`: File path to surface DEM raster (required).
  - `--outline`: File path to creeping body / glacier boundary polygon (optional).
  - `--lambda-c`, `--wavelength`: Corner frequency cutoff wavelength $\lambda_c$ in meters (default `200.0` m).
  - `--kc`: FFT corner wavenumber $k_c$ in rad/m (default `0.0314` rad/m, retained for backwards compatibility).
  - `--tau`: Target basal shear stress $\tau_0$ in kPa (default `100`).
  - `--max-km`: Survey length budget in km (default `5.0`).
  - `--batch`: Force non-interactive batch mode (accept defaults for omitted parameters without terminal prompts).
  - `--prefix`: Output prefix (default `final`).
  - `--out-dir`: Output directory for generated deliverables.
  - `--plots-dir`: Output directory for diagnostic figures.
  - `--format`: Export raster format (`tif` or `asc`, default `tif`).

- **Output Path Resolution:** Outputs are kept together: files go to `--out-dir` (current directory if omitted); an absolute `--prefix` places all files beside it; figures are saved in `figures/` next to the data files unless `--plots-dir` is an absolute path.
