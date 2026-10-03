# PySole Drift Analyzer & Unprobed Glacier Survey Planner Methodological Guide

This document provides a comprehensive methodological and mathematical reference for two key modules in `PySole`:
1. **Universal Kriging Drift Analyzer (`DriftAnalyzer`)**: An automated diagnostic and model selection engine for evaluating and ranking spatial drift models.
2. **Unprobed Glacier Survey Planner (`SurveyPlanner`)**: A forward campaign design engine for generating synthetic Shallow Ice Approximation (SIA) ice thickness targets and laying out optimal longitudinal and transverse survey tracks.

---

## 1. Universal Kriging Drift Analyzer (`DriftAnalyzer`)

### 1.1 Mathematical Formulation of Universal Kriging

Universal Kriging models a spatially continuous target variable $Z(\mathbf{x})$ at spatial coordinates $\mathbf{x} = (x, y)^T$ as the sum of a deterministic spatial trend surface (the **drift** $m(\mathbf{x})$) and a zero-mean, intrinsically stationary random residual process $\epsilon(\mathbf{x})$:

$$Z(\mathbf{x}) = m(\mathbf{x}) + \epsilon(\mathbf{x}) = \sum_{k=0}^K \beta_k f_k(\mathbf{x}) + \epsilon(\mathbf{x})$$

where:
- $f_0(\mathbf{x}) = 1$ is the constant baseline term ($\beta_0$),
- $f_k(\mathbf{x})$ for $k = 1, \dots, K$ are known spatial covariate functions (drifts),
- $\beta_k$ are unknown drift coefficients estimated simultaneously with Kriging weights,
- $\epsilon(\mathbf{x})$ is governed by a spatial covariance function $C(\mathbf{h})$ or semivariogram $\gamma(\mathbf{h}) = \frac{1}{2} E\left[(Z(\mathbf{x}+\mathbf{h}) - Z(\mathbf{x}))^2\right]$.

In `PySole`, candidate spatial covariates $f_k(\mathbf{x})$ include:
1. **`ordinary`**: Baseline constant mean $m(\mathbf{x}) = \beta_0$ (Ordinary Kriging).
2. **`sia`**: Inverse sine of surface slope $\sin(\alpha_{\text{opt}}(\mathbf{x}))^{-1}$, representing the Shallow Ice Approximation thickness driver.
3. **`z_dem`**: Surface elevation $Z_{\text{dem}}(\mathbf{x})$, capturing elevation-dependent ice accumulation or valley depth trends.
4. **`curvature_dem`**: Surface Laplacian curvature $C_{k_c}(\mathbf{x}) = \nabla^2 Z_{\text{smooth}, k_c}(\mathbf{x}) = \frac{\partial^2 Z_{\text{smooth}}}{\partial x^2} + \frac{\partial^2 Z_{\text{smooth}}}{\partial y^2}$, capturing convex ridges ($C > 0$) versus concave troughs ($C < 0$).
5. **`linear_xy`**: Planar spatial coordinates $(X, Y)$.
6. **`quadratic_xy`**: 2nd-order spatial polynomial $(X, Y, X^2, Y^2, XY)$.
7. **Compound Combinations**: Physical multi-drift combinations such as `["z_dem", "sia"]`, `["z_dem", "curvature_dem"]`, and `["z_dem", "sia", "curvature_dem"]`.

---

### 1.2 Multi-Tier Diagnostic Evaluation Pipeline

The `DriftAnalyzer` evaluates 14 candidate single and multi-drift models across three diagnostic tiers:

```
[ Input Survey Data & Covariates ]
               │
               ▼
┌──────────────────────────────┐
│  Tier 1: Feature Screening   │ ──► Multicollinearity Audit (VIF > 10)
│   (VIF, Pearson, RF Perm)    │ ──► Pearson Correlation & RF Importance
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Tier 2: Spatial CV & AICc   │ ──► Leave-One-Profile-Out (LOPO-CV) [if profile_col]
│   (RMSE, MAE, R², AICc)      │ ──► Spatial Buffer LOOCV (r_buffer = a_0 / 2) [fallback]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Tier 3: Glaciological Rules │ ──► Safeguard Warning (sia drift on product P)
└──────────────┬───────────────┘
               │
               ▼
[ Ranked Drift Model Recommendations ]
```

---

### 1.3 Tier 1: Statistical Feature Screening & Multicollinearity Auditing

#### 1. Variance Inflation Factor (VIF) Multicollinearity Audit
When multiple spatial drift terms $f_1, \dots, f_K$ are combined in a Universal Kriging model, high collinearity between covariates can cause severe numerical instability (ill-conditioning) in the Dual Kriging coefficient matrix $K$.

To detect collinearity, the `DriftAnalyzer` computes the **Variance Inflation Factor (VIF)** for each covariate $f_j$ within a multi-drift set:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the coefficient of determination obtained by regressing covariate $f_j$ against all remaining $K-1$ covariates in the drift set.
- **$\text{VIF}_j = 1$**: Complete orthogonality (no collinearity).
- **$\text{VIF}_j > 10$**: Severe multicollinearity. Multi-drift combinations containing any covariate with $\text{VIF}_j > 10$ are flagged as **penalized** (`is_penalized = True`) and deprioritized in ranking.

#### 2. Linear Pearson Correlation Analysis
Measures the parametric linear bivariate association between each spatial covariate $f_k(\mathbf{x})$ and the target observation vector $\mathbf{y} = (y_1, \dots, y_N)^T$:

$$r(f_k, \mathbf{y}) = \frac{\sum_{i=1}^N (f_{k,i} - \bar{f}_k)(y_i - \bar{y})}{\sqrt{\sum_{i=1}^N (f_{k,i} - \bar{f}_k)^2 \sum_{i=1}^N (y_i - \bar{y})^2}}$$

Because Universal Kriging constructs a linear trend model $\sum \beta_k f_k(\mathbf{x})$, Pearson correlation directly measures the exact linear effectiveness of covariate $f_k$ in the Dual Kriging system.

#### 3. Non-Parametric Spearman Rank Correlation Analysis
Measures the monotonic (non-linear) rank association between spatial covariate $f_k(\mathbf{x})$ and target vector $\mathbf{y}$:

$$\rho_s(f_k, \mathbf{y}) = 1 - \frac{6 \sum_{i=1}^N d_i^2}{N(N^2 - 1)}$$

where $d_i = \text{rank}(f_{k,i}) - \text{rank}(y_i)$.
Spearman rank correlation provides critical advantages in glaciological terrain analysis:
- **Outlier Robustness**: Immune to extreme elevation, slope, or traveltime outliers (e.g. steep rock cliffs or near-zero slopes) that would skew Pearson linear estimates.
- **Monotonic Trend Detection**: Detects strong physical trends even when the raw relationship is non-linear (e.g., ice thickness $D$ vs. surface slope $\alpha$ or elevation $Z_{\text{dem}}$).

#### 4. Random Forest Permutation Feature Importance
To capture non-linear terrain relationships (e.g., localized overdeepenings or multi-variable slope-trough interactions), the engine fits an ensemble Random Forest regressor ($\text{RF}$) mapping feature matrix $X = [f_1, \dots, f_K]$ to $\mathbf{y}$. Permutation feature importance for covariate $f_j$ is computed by shuffling feature vector $f_j$ across sample rows and measuring the decrease in out-of-bag $R^2$ accuracy:

$$\text{Importance}(f_j) = R^2_{\text{baseline}} - R^2_{\text{permuted}(f_j)}$$

#### 5. Methodological Insights: Tri-Metric Diagnostic Synthesis

Comparing Pearson ($r$), Spearman ($\rho_s$), and Random Forest Permutation Importance provides a comprehensive diagnostic breakdown:

| Diagnostic Metric | Statistical Focus | Alignment with Universal Kriging | Sensitivity to Outliers | Diagnostic Insight |
| :--- | :--- | :--- | :--- | :--- |
| **Pearson ($r$)** | Parametric **Linear** | **Direct**: Matches UK trend form $\sum \beta_k f_k(\mathbf{x})$ | Sensitive to extreme outliers | Quantifies direct linear coefficient $\beta_k$ efficiency |
| **Spearman ($\rho_s$)** | Non-parametric **Monotonic** | **Diagnostic**: Identifies non-linear monotonic trends | **Highly Robust** (Rank-based) | Uncovers strong physical drivers prior to non-linear transformation |
| **RF Importance** | Non-parametric **Non-Linear & Interactive** | **Exploratory**: Detects multi-scale terrain interactions | **Highly Robust** | Identifies complex localized terrain drivers |

**Interpretation Matrix**:
- **High Pearson ($r$) + High Spearman ($\rho_s$)**: Optimal linear drift candidate as-is (e.g. $f_{\text{SIA}} = \sin(\alpha_{\text{opt}})^{-1}$).
- **Low Pearson ($r$) + High Spearman ($\rho_s$)**: Strong physical relationship exists, but non-linear in raw space; indicates feature transformation (e.g., $\sin(\alpha)^{-1}$ or quadratic $Z^2$) will dramatically improve Kriging accuracy.
- **High RF Importance + Low Pearson/Spearman**: Indicates localized, non-monotonic spatial interactions (e.g. overdeepened valley basins or subglacial troughs).

---

### 1.4 Tier 2: Geostatistical Spatial Cross-Validation & Ranking Metrics

Standard Leave-One-Out Cross-Validation (LOOCV) can severely underestimate generalization errors on dense line-survey data (such as GPR profiles) due to strong along-track spatial autocorrelation. The `DriftAnalyzer` addresses this by deploying two spatial cross-validation strategies:

#### 1. Leave-One-Profile-Out Cross-Validation (LOPO-CV)
Activated when `inputs.survey_profile_column` (e.g., `line_id`, `profile_id`) is specified and contains at least 3 distinct profile tracks ($M \ge 3$).
- For each profile track $m \in \{1, \dots, M\}$, all observations belonging to profile $m$ are held out simultaneously as validation set $V_m$.
- Dual Kriging is trained on remaining profiles $T_m = \bigcup_{k \neq m} V_k$.
- Predictions $\hat{y}_i$ are evaluated for all points $i \in V_m$.
- Measures true inter-track interpolation performance across unmeasured glacier sectors.

#### 2. Spatial Buffer LOOCV (Fallback)
Used when no profile column is defined. For each validation node $\mathbf{x}_i$:
- All neighboring survey observations $\mathbf{x}_j$ located within a spatial exclusion buffer radius $r_{\text{buffer}}$ are excluded from training:

$$d(\mathbf{x}_i, \mathbf{x}_j) < r_{\text{buffer}} \implies \mathbf{x}_j \notin \text{Training Set}$$

- The buffer radius is linked to the empirical variogram spatial range $a_0$:

$$r_{\text{buffer}} = \frac{a_0}{2}$$

- Eliminates along-track autocorrelation leakage during validation.

#### 3. Cross-Validation Performance Metrics
For observed values $y_i$ and cross-validated predictions $\hat{y}_i$ ($i = 1, \dots, N$):

- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

- **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

- **Corrected Akaike Information Criterion (AICc)**:
  Balancing residual sum of squares against model parameter complexity (number of drift terms $K$):

  $$\text{AICc} = N \ln\left(\frac{\text{SS}_{\text{res}}}{N}\right) + 2K + \frac{2K(K+1)}{N - K - 1}$$

  where $\text{SS}_{\text{res}} = \sum_{i=1}^N (y_i - \hat{y}_i)^2$.

Models are sorted primarily by **$\text{RMSE}_{\text{CV}}$** (ascending) and **AICc** (ascending), while penalizing models with $\text{VIF} > 10$.

---

### 1.5 Tier 3: Glaciological Safeguards

The `DriftAnalyzer` incorporates domain-specific glaciological safeguards:
- **Product Target Safeguard**: When interpolating BSS product targets $P(x,y) = D(x,y) \sin \alpha(x,y)$, applying slope-dependent drifts (`sia`, $\sin(\alpha)^{-1}$) causes double-scaling ratio artifacts at low-slope margins ($P / \sin \alpha \to D / \sin^2 \alpha$). The analyzer flags `sia` on product targets $P$ with a safeguard warning and recommends `ordinary` Kriging or elevation drifts (`z_dem`).

---

## 2. Unprobed Glacier Survey Planner (`SurveyPlanner`)

When a user provides a surface DEM (`dem_path`) and glacier outline (`outline_path`), but **no survey dataset** (`survey_data_path = null`), `PySole` automatically dispatches the **Unprobed Glacier Survey Planner**.

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

### 2.1 Synthetic SIA Ice Thickness Modeling

#### 1. Frequency-Domain Surface DEM Smoothing
Raw DEM elevations $Z_{\text{dem}}(x,y)$ contain high-frequency micro-topography. `PySole` applies a 2D Gaussian low-pass filter in the frequency domain using spatial corner wavenumber $k_c$:

$$Z_{\text{smooth}, k_c}(x,y) = \text{Re}\left( \mathcal{F}^{-1} \left\{ \mathcal{F}\{Z_{\text{dem}}(x,y)\} \cdot \exp\left(-\frac{k_x^2 + k_y^2}{2 k_c^2}\right) \right\} \right)$$

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

### 2.2 Survey Profile Track Layout Algorithm

The survey planner automatically designs field campaign GPR/seismic tracks subject to a total track length budget $L_{\text{max}}$ (e.g., 10.0 km).

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

### 2.3 Output Formats & Field Exports

The `SurveyPlanner` generates four field-ready deliverables:

| Deliverable | File Naming Pattern | Description |
| :--- | :--- | :--- |
| **GeoTIFF Raster** | `<output_prefix>_sia_modelled_depth.tif` | 2D synthetic SIA ice thickness grid $D_{\text{SIA}}(x,y)$ [m]. |
| **GPX Vector** | `<output_prefix>_survey_plan.gpx` | GPS track file containing `<trk>` segments and `<trkpt>` nodes for field navigation. |
| **GeoJSON Vector** | `<output_prefix>_survey_plan.geojson` | Standard GIS LineString features with `track_id`, `length_m`, and `max_depth_m` attributes. |
| **Map Plot** | `<plots_dir>/<output_prefix>_survey_plan_map.png` | Diagnostic map plot displaying $D_{\text{SIA}}(x,y)$ color map overlaid with longitudinal ($L_1$, red) and transverse ($T_i$, cyan) tracks. |

---

## 3. Configuration & CLI Examples

### 3.1 JSON Configuration Schema

```json
{
  "inputs": {
    "dem_path": "surface_dem.tif",
    "outline_path": "glacier_outline.shp",
    "survey_data_path": "gpr_picks.csv",
    "survey_data_type": "one_way_traveltime",
    "survey_profile_column": "line_id"
  },
  "kriging_parameters": {
    "pre_migration": {
      "interpolation_target": "P",
      "method": "universal",
      "drift_analyzer": true
    },
    "post_migration": {
      "interpolation_target": "D",
      "method": "universal",
      "drift_analyzer": true
    }
  }
}
```

### 3.2 CLI Commands

```bash
# Execute PySole with interactive Drift Analyzer and profile grouping
pysole pysole.json --drift-analyzer --profile-col line_id

# Force non-interactive batch mode
pysole pysole.json --batch

# Standalone Unprobed Glacier Survey Planning
pysole plan-survey --dem surface_dem.tif --outline glacier_outline.shp --kc 3.0 --tau 100 --max-km 10.0 --output proposed_survey
```
