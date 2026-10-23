# Universal Kriging Drift Analyzer Guide

This document provides a methodological and mathematical reference for the **Universal Kriging Drift Analyzer (`DriftAnalyzer`)** in `PySole`, an automated diagnostic and model selection engine for evaluating, ranking, and comparing spatial drift models.

> [!TIP]
> For forward campaign design on unprobed glaciers without radar sounding data, see the dedicated [Survey Planner Guide](survey_planner.md).

---

## 1. Universal Kriging Drift Analyzer (`DriftAnalyzer`)

### 1.1 Mathematical Formulation & Available Drift Primitives

Universal Kriging models a spatially continuous target variable $Z(\mathbf{x})$ at spatial coordinates $\mathbf{x} = (x, y)^T$ as the sum of a deterministic spatial trend surface (the **drift** $m(\mathbf{x})$) and a zero-mean, intrinsically stationary random residual process $\epsilon(\mathbf{x})$:

$$Z(\mathbf{x}) = m(\mathbf{x}) + \epsilon(\mathbf{x}) = \sum_{k=0}^K \beta_k f_k(\mathbf{x}) + \epsilon(\mathbf{x})$$

where:
- $f_0(\mathbf{x}) = 1$ is the constant baseline term ($\beta_0$),
- $f_k(\mathbf{x})$ for $k = 1, \dots, K$ are known spatial covariate functions (drifts),
- $\beta_k$ are unknown drift coefficients estimated simultaneously with Kriging weights,
- $\epsilon(\mathbf{x})$ is governed by a spatial covariance function $C(\mathbf{h})$ or semivariogram $\gamma(\mathbf{h}) = \frac{1}{2} E\left[(Z(\mathbf{x}+\mathbf{h}) - Z(\mathbf{x}))^2\right]$.

#### Single Drift Models & Spatial Covariate Construction

`PySole` provides **8 primitive drift terms** (`sia`, `z_dem`, `curvature_dem`, `mass_balance`, `velocity`, `velocity_sia`, `linear_xy`, `quadratic_xy`) plus the constant-mean **Ordinary Kriging** baseline (`ordinary`). Below, $n_{\text{drift}} = K + 1$ denotes the total number of drift columns in the augmented Dual Kriging system **including the constant mean** $f_0(\mathbf{x}) = 1$:

| Primitive / Model | Covariate Basis $f_k(\mathbf{x})$ | $n_{\text{drift}}$ | Required Input | Spectral Filtering & Domain Handling | Glaciological Interpretation |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`ordinary`** (`[]`) | $f_0(\mathbf{x}) = 1$ | 1 | DEM only | None | Constant local/global mean baseline (Ordinary Kriging). |
| **`"sia"`** | $\sin(\alpha_{\text{opt}}(\mathbf{x}))^{-1}$ | 2 | DEM only | Slope from 2D FFT Gaussian-smoothed DEM at $k_c$; floored at `slope_floor_deg` ($5^\circ$) | Shallow Ice Approximation ($D \propto 1/\sin\alpha$): thicker ice on flat plateaus, thinner on steep slopes. |
| **`"z_dem"`** | $Z_{\text{dem}}(\mathbf{x})$ | 2 | DEM only | Raw surface DEM on `GridGeometry` | Valley hypsometry and elevation-dependent accumulation/thickness gradients. |
| **`"curvature_dem"`** | $C_{k_c}(\mathbf{x}) = \nabla^2 Z_{\text{smooth}, k_c}(\mathbf{x})$ | 2 | DEM only | Laplacian $\frac{\partial^2 Z}{\partial x^2} + \frac{\partial^2 Z}{\partial y^2}$ of $k_c$-smoothed DEM | Topographic focusing: concave glacial troughs ($C > 0$) vs. convex ridges ($C < 0$). |
| **`"mass_balance"`** | $\dot{b}(\mathbf{x})$ | 2 | `inputs.mass_balance_path` | Co-registered to DEM grid; **kept unsmoothed** | Decadal-mean surface mass balance flux divergence; preserves local debris-cover, avalanche, and wind-drift anomalies. |
| **`"velocity"`** | $u_{s,\text{smooth}, k_c}(\mathbf{x})$ | 2 | `inputs.velocity_path` | Zero-clamped outside `outline_mask`; 2D FFT low-pass filtered at $k_c$ and clamped $\ge 0$ | Decadal-mean surface flow speed (creep + basal sliding); longitudinal coupling scale matched to $k_c$. |
| **`"velocity_sia"`** | $\frac{(\max(u_{s,\text{smooth}, k_c}(\mathbf{x}), 0))^{1/4}}{(\sin\alpha_{\text{opt}}(\mathbf{x}))^{3/4}}$ | 2 | `inputs.velocity_path` | Uses $k_c$-smoothed, zero-clamped velocity and $k_c$-smoothed, floored SIA slope | Coupled Glen's flow law ($n = 3$) internal ice creep scaling ($D \propto u_s^{1/4} / (\sin\alpha)^{3/4}$). |
| **`"linear_xy"`** | $(X, Y)$ | 3 | DEM only | Zero-centered, aspect-ratio-scaled planar coordinates | 1st-order regional planar tilt across the glacier catchment. |
| **`"quadratic_xy"`** | $(X, Y, X^2, Y^2, XY)$ | 6 | DEM only | Zero-centered, aspect-ratio-scaled 2nd-order polynomial | 2nd-order parabolic basin or dome curvature across the glacier domain. |

#### Dynamic Default Candidate Suite Scaling

Combining the 8 single drift primitives yields up to **111 non-redundant multi-drift combinations** (Section 1.8). When `drift_analyzer: true` and `drift_terms: null`, `PySole` dynamically scales the evaluated candidate suite based on the configured input rasters:

| Configured Input Rasters | Active Primitives | Single Drift | Multi Drift | Ordinary | Total Suite Size |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DEM only** *(default)* | `sia`, `z_dem`, `curvature_dem`, `linear_xy`, `quadratic_xy` (5) | 5 | 18 | 1 | **24 candidates** |
| **DEM + `mass_balance_path`** | Core 5 + `mass_balance` (6) | 6 | 41 | 1 | **48 candidates** |
| **DEM + `velocity_path`** | Core 5 + `velocity`, `velocity_sia` (7) | 7 | 52 | 1 | **60 candidates** |
| **DEM + `mass_balance_path` + `velocity_path`** | All 8 primitives | **8** | **111** | **1** | **120 candidates** |

> [!IMPORTANT]
> Explicitly requesting `"mass_balance"` without configuring `inputs.mass_balance_path` (or `Solver(..., mass_balance=...)`), or requesting `"velocity"` / `"velocity_sia"` without configuring `inputs.velocity_path` (or `Solver(..., velocity=...)`), raises an immediate `ConfigError` before any data is processed.

---

### 1.2 Configuration: Which Models Are Evaluated and Applied

`kriging_parameters.<stage>.drift_terms` is `null` (omitted) or a flat/nested list of the eight primitive terms `sia`, `z_dem`, `curvature_dem`, `mass_balance`, `velocity`, `velocity_sia`, `linear_xy`, `quadratic_xy`. Inside an analyzer suite, `[]` represents the stand-alone Ordinary Kriging candidate. A top-level `[]` is never accepted for `universal` and `regression` (use `null` to omit the terms). Together with `method` and `drift_analyzer`, the configuration is interpreted as follows across JSON, CLI, and Python API:

| `method` | `drift_analyzer` | `drift_terms` syntax | Meaning & Executed Action |
| :--- | :--- | :--- | :--- |
| `ordinary` | any | any or `null` | **`ordinary` overrides `drift_analyzer` and `drift_terms`**: both are ignored (terms are not validated) and a single warning is logged if either was set. |
| `universal` | `false` | flat list (e.g. `["sia", "z_dem"]`) or single nested list (`[["sia", "z_dem"]]`) | Applied directly as **ONE multi-drift model**. Redundant collinear terms (`linear_xy` with `quadratic_xy`; `sia` or `velocity` with `velocity_sia`) are automatically dropped with a warning. `null` or `[]` raises a `ConfigError`. |
| `universal` | `false` | multi-model nested list (e.g. `[["sia", "z_dem"], ["sia"], []]`) | **Non-interactive multi-model suite**: evaluates all listed candidates via block leave-out CV using the default exclusion buffer radius ($r_{\text{buffer}} = a_0 / 2$) and exports all model rasters, subfolders, and comparison tables without terminal prompts. |
| `universal` | `true` | `null` | Evaluates the **default suite** (**24** candidates with DEM only, **48** with `mass_balance_path`, **60** with `velocity_path`, or **120** with both). |
| `universal` | `true` | flat list $\ge 2$ terms (e.g. `[[], "sia", "z_dem", "mass_balance"]`) | Evaluates the **combinatorial suite** of all valid non-redundant subsets of the listed primitives (plus Ordinary if `[]` is included). |
| `universal` | `true` | explicit multi-model nested list (e.g. `[["velocity_sia", "z_dem"], ["sia"], []]`) | Evaluates the **explicit suite** of listed candidate models directly without combinatorial expansion. |
| `universal` | `true` | single candidate (`["sia"]`, `[["sia", "z_dem"]]`, or `[[]]`) | **Single-model auto-disable**: logs an informational notice, sets `drift_analyzer: false`, and applies the single model directly without running cross-validation. |
| `regression` | `false` or `true` | flat list or `null` | **Random Forest covariates**: trains a Random Forest trend model on the listed features before Ordinary Kriging of the residuals (executed via PyKrige; analyzer is skipped with a warning if `true`). **Default (`null`)**: uses all non-redundant primitives (`["sia", "z_dem", "curvature_dem", "quadratic_xy"]` plus `"mass_balance"` and `"velocity"` when configured); spatial coordinates only: `["linear_xy"]`. |

If `method` is omitted (`null`), the interpolation target decides: direct targets (`"T"` pre-migration, `"D"` post-migration) default to `universal` with `["sia"]` (or the default suite when `drift_analyzer: true` and `drift_terms: null`), whereas the BSS product target `"P"` defaults to `ordinary`.

#### Suite expansion

When `drift_analyzer: true` and `drift_terms` is a flat list (or `null`), the combinatorial suite expands to every non-empty subset of the active primitives **except**:
1. subsets containing both `linear_xy` and `quadratic_xy` (`quadratic_xy` already contains the linear $[X, Y]$ columns), and
2. subsets containing `velocity_sia` together with `sia` or `velocity` (`velocity_sia` already couples `velocity` and `sia`).

Terms are ordered canonically (`sia + z_dem`), duplicates are removed, and a nested `[]` adds the Ordinary Kriging candidate at the front of the suite:

| `drift_terms` (analyzer on) | Candidates |
| :--- | :--- |
| `null` | 24 (all 5 core primitives + Ordinary; 48 with `mass_balance_path`, 60 with `velocity_path`, 120 with both) |
| `["sia", "z_dem", "curvature_dem", "linear_xy", "quadratic_xy"]` | 23 |
| `["sia", "z_dem", "curvature_dem", "mass_balance", "velocity", "velocity_sia", "linear_xy", "quadratic_xy"]` | 119 |
| `[[], "sia", "z_dem", "curvature_dem", "mass_balance", "velocity", "velocity_sia", "linear_xy", "quadratic_xy"]` | 120 |
| `[[], "sia", "z_dem", "curvature_dem", "mass_balance", "linear_xy", "quadratic_xy"]` | 48 |
| `[[], "sia", "z_dem", "curvature_dem", "velocity", "velocity_sia", "linear_xy", "quadratic_xy"]` | 60 |
| `[[], "sia", "z_dem"]` | 4 (Ordinary, `sia`, `z_dem`, `sia + z_dem`) |
| `["sia", "z_dem", "curvature_dem"]` | 7 |
| `["sia", "z_dem", "quadratic_xy"]` | 7 |
| `["sia"]` | 1 (auto-disables analyzer) |
| `[]` | error (use `null`) |

#### Winner handling

The best-ranked candidate is applied (or becomes the primary model when multiple models are selected). In an interactive terminal, the ranking table is printed and the user can select one or more models by row number (`3`, `#3`), row ranges (`1-5`, `1, 3, 5-8`), printed name (`sia + z_dem`), term list (`['sia', 'z_dem']`), `ordinary`, or `[]`; pressing `[Enter]` accepts `#1`. In batch mode (`--batch` or non-interactive multi-model suites), no prompt is shown. Penalized candidates ($\text{VIF} > 10$ or numerical CV failure) are always ranked after all non-penalized candidates. If no candidate can be cross-validated, a warning is logged and Ordinary Kriging is applied.

#### Engine support

| Engine | `ordinary` | `universal` (all 119 drift models) | `regression` | `drift_analyzer` |
| :--- | :--- | :--- | :--- | :--- |
| `"native"` (default) | yes | yes (fast Dual Kriging + exact variance) | PyKrige is used for the stage (logged) | yes |
| `"pykrige"` | yes | yes, same prediction as native (max. difference $< 10^{-4}$ of value range); **slow for large $N$** | yes | candidates are ranked with the native engine; winner is applied with PyKrige |

> [!WARNING]
> **Speed of `"pykrige"` with `"universal"`:** PyKrige's `UniversalKriging` computes a leave-one-out fit statistic inside its constructor whose runtime scales as $O(N^3)$ to $O(N^4)$ in the number of survey points (several minutes for $N \gtrsim 1{,}000$), independent of the drift terms or grid size. Use `engine: "native"` for Universal Kriging on large datasets; `"ordinary"` and `"regression"` with PyKrige are unaffected.

---

### 1.3 Multi-Tier Diagnostic Evaluation Pipeline

The `DriftAnalyzer` evaluates all candidates in the active suite (dynamically **24**, **48**, **60**, or **120** candidates by default, or any custom suite) across three diagnostic tiers:

```
[ Input Survey Data & Spatial Covariates (DEM + Optional Mass Balance / Velocity) ]
                                        │
                                        ▼
               ┌────────────────────────────────────────────────┐
               │ Tier 1: Multicollinearity Audit                │
               │   • Computes max VIF across drift columns      │
               │   • Flags VIF > 10 as PENALIZED                │
               └────────────────────────┬───────────────────────┘
                                        │
                                        ▼
               ┌────────────────────────────────────────────────┐
               │ Tier 2: Exact Block Leave-Out Spatial CV       │
               │   • LOPO-CV (if survey_profile_column ≥ 3)     │
               │   • Spatial Buffer LOOCV (r_buffer = a_0 / 2)  │
               │   • Computes RMSE, MAE, R², AICc, ΔAICc        │
               └────────────────────────┬───────────────────────┘
                                        │
                                        ▼
               ┌────────────────────────────────────────────────┐
               │ Tier 3: Glaciological Target Check             │
               │   • Warns on 'sia' or 'velocity_sia' with 'P'  │
               └────────────────────────┬───────────────────────┘
                                        │
                                        ▼
         [ Ranked Candidates by (is_penalized, AICc, RMSE); Winner Applied ]
```

---

### 1.4 Tier 1: Multicollinearity Auditing

When multiple spatial drift terms $f_1, \dots, f_K$ are combined in a Universal Kriging model, strong collinearity between covariates causes severe ill-conditioning in the Dual Kriging matrix. The `DriftAnalyzer` computes the **Variance Inflation Factor (VIF)** for each covariate $f_j$ in a candidate's standardized design matrix:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the coefficient of determination obtained by regressing covariate $f_j$ against the remaining $K-1$ covariates:
- **$\text{VIF}_j = 1$**: Complete orthogonality (single-drift and Ordinary models always have $\text{VIF} = 1.0$).
- **$\text{VIF}_j > 10$**: Severe multicollinearity. Any candidate with $\max_j \text{VIF}_j > 10$ (or a numerical CV failure) is flagged with `is_penalized = True` (`PENALIZED`) and sorted below all valid candidates to prevent runaway extrapolation spikes in unmeasured glacier sectors.

---

### 1.5 Tier 2: Geostatistical Spatial Cross-Validation & Ranking Metrics

Standard point-wise Leave-One-Out Cross-Validation (LOOCV) severely underestimates generalization error on dense line-survey data (such as GPR profiles) due to along-track spatial autocorrelation. The `DriftAnalyzer` eliminates autocorrelation leakage by holding out entire spatial blocks:

#### 1. Leave-One-Profile-Out Cross-Validation (LOPO-CV)
Activated automatically when `inputs.survey_profile_column` (or `--profile-col`) is provided and contains at least 3 distinct profile tracks ($M \ge 3$). Each profile track $m \in \{1, \dots, M\}$ is held out in full while the Kriging system is fitted on the remaining $M - 1$ profiles, directly measuring inter-profile interpolation accuracy across unmeasured glacier sectors.

#### 2. Spatial Buffer LOOCV (Fallback)
Used when no profile column is configured (or fewer than 3 profiles exist). For each validation sounding $\mathbf{x}_i$, all neighboring observations within a spatial exclusion buffer radius $r_{\text{buffer}}$ are excluded from the training set:

$$d(\mathbf{x}_i, \mathbf{x}_j) \le r_{\text{buffer}} \implies \mathbf{x}_j \notin \text{Training Set}, \qquad r_{\text{buffer}} = \frac{a_0}{2}$$

where $a_0$ is the fitted variogram range (configurable interactively at the terminal prompt).

#### 3. Fast Exact Block Leave-Out Matrix Update
All candidates (whether 24 or 120) and all blocks are evaluated **exact up to machine precision** from a **single** $O(N^3)$ factorization of the base Ordinary Kriging matrix $A_0 = \begin{pmatrix} \Gamma & \mathbf{1} \\ \mathbf{1}^T & 0 \end{pmatrix}$ into $P = A_0^{-1}$. Each candidate's drift columns $G$ are incorporated via a fast low-rank Schur-complement update of $P$ to obtain dual weights $\mathbf{c}$, and the exact leave-block-out residual vector for held-out block $B$ is evaluated via the block-generalized Dubrule identity:

$$\mathbf{e}_B = \left(P_{BB}\right)^{-1} \mathbf{c}_B, \qquad \hat{\mathbf{y}}_B = \mathbf{y}_B - \mathbf{e}_B$$

- **Deterministic full-sample evaluation ($n = N$)**: Every survey point is evaluated without random sub-sampling. Blocks leaving fewer than $n_{\text{drift}} + 3$ training points are skipped.
- **Memory Guard**: Full inversion requires $\sim 3 N^2$ float64 words ($\sim 0.6\text{ GB}$ at $N = 5{,}000$). If this exceeds `max_memory_fraction` of available RAM, `PySole` retains the LU factors and evaluates a deterministic farthest-point subset ($n \ge 100$) with a logged warning.

#### 4. Cross-Validation Performance Metrics
For observed values $y_i$ and cross-validated predictions $\hat{y}_i$ across the $n$ evaluated points ($i = 1, \dots, n$), with residual sum of squares $\text{SS}_{\text{res}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2$ and parameter count $k = n_{\text{drift}}$ (constant + drift columns):

$$\text{RMSE} = \sqrt{\frac{\text{SS}_{\text{res}}}{n}}, \qquad \text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|, \qquad R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

$$\text{AICc} = n \ln\left(\frac{\text{SS}_{\text{res}}}{n}\right) + 2k + \frac{2k(k+1)}{n - k - 1}$$

#### 5. Ranking Key, ΔAICc / ΔRMSE % & Diagnostic Case Study
Candidates are sorted deterministically by the 3-tier tuple:

$$\text{Sort Key} = \left(\text{is\_penalized},\ \text{AICc ascending},\ \text{CV-RMSE ascending}\right)$$

1. **Non-penalized first** (`is_penalized = False` before `True`),
2. **AICc ascending** (balancing fit quality against parameter count $k = n_{\text{drift}}$),
3. **CV-RMSE ascending** (tie-breaker).

Two relative columns quantify the distance to the top-ranked model (`#1`):
- **$\Delta\text{AICc}$** $= \text{AICc} - \text{AICc}_{\text{best non-penalized}}$
- **$\Delta\text{RMSE \%}$** $= 100 \cdot \left(\text{RMSE} / \text{RMSE}_{\text{top}} - 1\right)$

The complete 24-candidate ranking table below (APO glacier dataset, Pass 2, product target $P = D \cdot \sin\alpha_{\text{opt}}$, $N = 8{,}939$, $r_{\text{buffer}} = 1{,}700\text{ m}$) illustrates how `VALID`, `WARNING`, and `PENALIZED` models interact:

```
================================================================================================================
       DRIFT ANALYZER CANDIDATE MODEL RECOMMENDATIONS (PASS 2)
================================================================================================================
#   Drift Model                                  RMSE (m)   dRMSE %   AICc       dAICc     VIF      Status
----------------------------------------------------------------------------------------------------------------
1   ordinary                                     6.38       0.00      33129.6    0.0       1.0      VALID
2   sia                                          6.45       1.06      33320.0    190.3     1.0      WARNING
3   linear_xy                                    6.47       1.43      33387.9    258.2     1.2      VALID
4   z_dem                                        6.49       1.76      33443.0    313.3     1.0      VALID
5   curvature_dem                                6.51       2.02      33489.0    359.4     1.0      VALID
6   z_dem + linear_xy                            6.51       2.00      33489.3    359.6     1.8      VALID
7   curvature_dem + linear_xy                    6.59       3.36      33725.8    596.2     1.3      VALID
8   sia + curvature_dem                          6.61       3.64      33772.4    642.8     1.0      WARNING
9   sia + linear_xy                              6.61       3.66      33778.6    649.0     1.3      WARNING
10  sia + z_dem                                  6.63       3.87      33811.8    682.2     1.0      WARNING
11  sia + z_dem + curvature_dem                  6.66       4.38      33901.8    772.2     1.2      WARNING
12  sia + z_dem + curvature_dem + linear_xy      6.72       5.30      34063.5    933.9     1.8      WARNING
13  sia + z_dem + linear_xy                      6.74       5.72      34132.8    1003.1    1.8      WARNING
14  sia + curvature_dem + linear_xy              6.77       6.11      34197.6    1068.0    1.3      WARNING
15  z_dem + curvature_dem + linear_xy            6.79       6.42      34249.8    1120.2    1.8      VALID
16  curvature_dem + quadratic_xy                 6.82       6.98      34347.7    1218.0    9.8      VALID
17  z_dem + curvature_dem                        6.84       7.22      34380.0    1250.4    1.1      VALID
18  quadratic_xy                                 6.85       7.35      34406.9    1277.2    9.8      VALID
19  sia + curvature_dem + quadratic_xy           7.11       11.53     35093.9    1964.2    9.9      WARNING
20  sia + quadratic_xy                           7.15       12.05     35176.5    2046.8    9.8      WARNING
21  sia + z_dem + curvature_dem + quadratic_xy   6.35       -0.52     33053.1    -76.5     11.4     PENALIZED
22  z_dem + curvature_dem + quadratic_xy         6.39       0.18      33175.5    45.9      10.8     PENALIZED
23  z_dem + quadratic_xy                         6.53       2.44      33572.4    442.8     10.4     PENALIZED
24  sia + z_dem + quadratic_xy                   6.97       9.30      34733.3    1603.7    10.7     PENALIZED
----------------------------------------------------------------------------------------------------------------
```

- **Why `#1 ordinary` Wins on Target `"P"`**: Because $P(x,y) = D \cdot \sin\alpha_{\text{opt}}$ already incorporates the first-order SIA slope scaling, its spatial mean is nearly stationary ($n_{\text{drift}} = 1$, $\text{AICc} = 33129.6$).
- **Why Rows 21–24 Are `PENALIZED` Despite Lower Raw RMSE**: Row 21 achieves a slightly lower raw RMSE ($6.35\text{ m}$, $\Delta\text{RMSE} = -0.52\%$) by combining `z_dem` with `quadratic_xy` ($\text{VIF} = 11.4 > 10$). Because valley elevation correlates strongly with a 2nd-order spatial polynomial, the solver assigns large opposing weights to `z_dem` and `quadratic_xy` that cancel along survey tracks but diverge in unprobed areas. The $\text{VIF} > 10$ gate automatically relegates rows 21–24 to the bottom of the ranking.

---

### 1.6 Reading the AICc: ΔAICc, ΔRMSE % and the Role of $n$

For two candidates differing by $\Delta k = k_{\text{complex}} - k_{\text{simple}}$ parameters, the AICc difference is approximately:

$$\text{AICc}_{\text{simple}} - \text{AICc}_{\text{complex}} \approx 2n \ln\left(\frac{\text{RMSE}_{\text{simple}}}{\text{RMSE}_{\text{complex}}}\right) - 2\,\Delta k$$

Because the log-likelihood term scales linearly with sample count $n = N$ while the parameter penalty $2\,\Delta k$ is constant, large surveys ($N > 1{,}000$) can produce a large $\Delta\text{AICc}$ ($> 10$) from a very small relative RMSE reduction ($< 0.5\%$).

**Best-Practice Guidelines:**
- **Inspect $\Delta\text{AICc}$ and $\Delta\text{RMSE \%}$ together**: $\Delta\text{AICc} < 2$ indicates statistically indistinguishable models; $\Delta\text{AICc} > 10$ indicates strong statistical support. However, if the top candidates differ by $\Delta\text{RMSE \%} \lesssim 1\%$, prefer the **simpler, physically best-justified** model.
- **Prefer LOPO-CV (`survey_profile_column`)**: Whenever profile line IDs exist, LOPO-CV provides the most realistic test of cross-glacier interpolation skill.
- **Heed `WARNING` and `PENALIZED` flags**: Avoid selecting `WARNING` models on product target `"P"` (Section 1.9) or `PENALIZED` models ($\text{VIF} > 10$) even if their raw RMSE is slightly lower.

---

### 1.7 Variogram, Standardization & Shared Kriging System

The `DriftAnalyzer` and the production Kriging pass share the exact same mathematical objects (`KrigingSystem` and `DriftBasis`), ensuring that cross-validation rankings reflect production behavior without approximation:

1. **Single Shared Variogram & Regularization**: One theoretical variogram model (`range`, `sill`, `nugget`) is fitted once per stage on the BSS product $P$ and shared across all candidates and the production solve, with identical Tikhonov diagonal regularization ($10^{-6} \times \text{sill}$).
2. **Grid Standardization & Bilinear Sampling**: All active raster covariates (`sia`, `z_dem`, `curvature_dem`, `mass_balance`, `velocity`, `velocity_sia`; see table in Section 1.1) are constructed once per stage on the bottom-up `GridGeometry`, standardized by their full-grid mean and standard deviation (with NaNs filled as `0.0` post-standardization), and sampled at survey coordinates via bilinear interpolation. Coordinate drifts (`linear_xy`, `quadratic_xy`) use zero-centered, aspect-ratio-scaled coordinates in $[-1, 1]$.

---

### 1.8 Systematic Multi-Drift Model Suite

Universal Kriging in `PySole` combines the 6 physical raster primitives (`"sia"`, `"z_dem"`, `"curvature_dem"`, `"mass_balance"`, `"velocity"`, `"velocity_sia"`) with the 2 polynomial coordinate trends (`"linear_xy"`, `"quadratic_xy"`) into an augmented linear drift basis:

$$U(x,y) = \beta_0 + \sum_{r \in \mathcal{R}} \beta_r f_r(x,y) + p_{\text{coord}}(x,y)$$

After pruning collinear combinations (`linear_xy` + `quadratic_xy` and `velocity_sia` + `sia`/`velocity`), the complete suite comprises **111 systematic multi-drift combinations** (**18 core DEM combinations** always available + **93 auxiliary glaciological combinations** enabled when `mass_balance_path` and/or `velocity_path` are configured), organized into three structural groups. Below, $n_{\text{drift}}$ counts all columns of the augmented system **including the constant** $\beta_0$ (`CandidateDriftResult.n_terms = n_drift - 1`, and AICc uses $k = n_{\text{drift}}$).

#### 1. 33 Multi-Raster Physical Combinations (No Coordinate Trend)

These models combine 2 to 5 physical raster covariates ($p_{\text{coord}}(x,y) = 0$) without imposing a planar or parabolic coordinate tilt:

$$U(x,y) = \beta_0 + \sum_{r \in \mathcal{R}} \beta_r f_r(x,y), \qquad n_{\text{drift}} = 1 + |\mathcal{R}|$$

##### Core DEM Multi-Raster Combinations (4 models — always available)
Couple SIA slope scaling $\sin(\alpha_{\text{opt}})^{-1}$, surface elevation $Z_{\text{dem}}$, and $k_c$-smoothed Laplacian curvature $\nabla^2 Z_{\text{smooth}, k_c}$:
- **`["sia", "z_dem"]` ($n_{\text{drift}} = 3$)**: SIA ice-flux slope dynamics coupled with macroscopic valley hypsometry.
- **`["sia", "curvature_dem"]` ($n_{\text{drift}} = 3$)**: SIA slope dynamics coupled with concave trough focusing ($\nabla^2 Z > 0$) and convex ridge thinning ($\nabla^2 Z < 0$).
- **`["z_dem", "curvature_dem"]` ($n_{\text{drift}} = 3$)**: Slope-independent terrain morphometry (safe on both direct targets `"T"`/`"D"` and product target `"P"`).
- **`["sia", "z_dem", "curvature_dem"]` ($n_{\text{drift}} = 4$)**: Complete core DEM physical trio uniting SIA slope, elevation hypsometry, and valley curvature.

##### Surface Mass Balance (`mass_balance`) Combinations (7 models — requires `mass_balance_path`)
Couple unsmoothed decadal-mean surface mass balance $\dot{b}(x, y)$ (capturing localized debris-cover insulation, avalanche accumulation cones, and wind-drift bowls) with SIA slope and DEM morphometry:
- **`["sia", "mass_balance"]` ($n_{\text{drift}} = 3$)**
- **`["z_dem", "mass_balance"]` ($n_{\text{drift}} = 3$)**
- **`["curvature_dem", "mass_balance"]` ($n_{\text{drift}} = 3$)**
- **`["sia", "z_dem", "mass_balance"]` ($n_{\text{drift}} = 4$)**
- **`["sia", "curvature_dem", "mass_balance"]` ($n_{\text{drift}} = 4$)**
- **`["z_dem", "curvature_dem", "mass_balance"]` ($n_{\text{drift}} = 4$)**
- **`["sia", "z_dem", "curvature_dem", "mass_balance"]` ($n_{\text{drift}} = 5$)**

##### Surface Flow Velocity (`velocity`) Combinations (15 models — requires `velocity_path`; 8 also require `mass_balance_path`)
Couple $k_c$-smoothed, off-glacier zero-clamped surface velocity $u_{s,\text{smooth}, k_c}(x, y)$ as an independent additive covariate alongside SIA slope, DEM morphometry, and mass balance (allowing the solver to weight basal sliding/creep speed and surface slope independently):
- **`["sia", "velocity"]` ($n_{\text{drift}} = 3$)**
- **`["z_dem", "velocity"]` ($n_{\text{drift}} = 3$)**
- **`["curvature_dem", "velocity"]` ($n_{\text{drift}} = 3$)**
- **`["mass_balance", "velocity"]` ($n_{\text{drift}} = 3$)**
- **`["sia", "z_dem", "velocity"]` ($n_{\text{drift}} = 4$)**
- **`["sia", "curvature_dem", "velocity"]` ($n_{\text{drift}} = 4$)**
- **`["sia", "mass_balance", "velocity"]` ($n_{\text{drift}} = 4$)**
- **`["z_dem", "curvature_dem", "velocity"]` ($n_{\text{drift}} = 4$)**
- **`["z_dem", "mass_balance", "velocity"]` ($n_{\text{drift}} = 4$)**
- **`["curvature_dem", "mass_balance", "velocity"]` ($n_{\text{drift}} = 4$)**
- **`["sia", "z_dem", "curvature_dem", "velocity"]` ($n_{\text{drift}} = 5$)**
- **`["sia", "z_dem", "mass_balance", "velocity"]` ($n_{\text{drift}} = 5$)**
- **`["sia", "curvature_dem", "mass_balance", "velocity"]` ($n_{\text{drift}} = 5$)**
- **`["z_dem", "curvature_dem", "mass_balance", "velocity"]` ($n_{\text{drift}} = 5$)**
- **`["sia", "z_dem", "curvature_dem", "mass_balance", "velocity"]` ($n_{\text{drift}} = 6$)**

##### Coupled SIA Creep Velocity (`velocity_sia`) Combinations (7 models — requires `velocity_path`; 4 also require `mass_balance_path`)
Couple Glen's flow law ($n = 3$) internal deformation thickness proxy $u_{s,\text{smooth}, k_c}^{1/4} / (\sin\alpha_{\text{opt}})^{3/4}$ with DEM morphometry and surface mass balance:
- **`["z_dem", "velocity_sia"]` ($n_{\text{drift}} = 3$)**
- **`["curvature_dem", "velocity_sia"]` ($n_{\text{drift}} = 3$)**
- **`["mass_balance", "velocity_sia"]` ($n_{\text{drift}} = 3$)**
- **`["z_dem", "curvature_dem", "velocity_sia"]` ($n_{\text{drift}} = 4$)**
- **`["z_dem", "mass_balance", "velocity_sia"]` ($n_{\text{drift}} = 4$)**
- **`["curvature_dem", "mass_balance", "velocity_sia"]` ($n_{\text{drift}} = 4$)**
- **`["z_dem", "curvature_dem", "mass_balance", "velocity_sia"]` ($n_{\text{drift}} = 5$)**

---

#### 2. 39 Multi-Raster + 1st-Order Linear Spatial Trend (`linear_xy`)

These models superimpose any non-empty combination of the physical raster covariates ($\mathcal{R}$) onto a 1st-order planar regional coordinate gradient ($p_{\text{coord}}(x,y) = a_X X + a_Y Y$, adding 2 coordinate columns):

$$U(x,y) = \beta_0 + \sum_{r \in \mathcal{R}} \beta_r f_r(x,y) + a_X X + a_Y Y, \qquad n_{\text{drift}} = 3 + |\mathcal{R}|$$

##### Core DEM + `linear_xy` Combinations (7 models — always available)
- **`["sia", "linear_xy"]` ($n_{\text{drift}} = 4$)**
- **`["z_dem", "linear_xy"]` ($n_{\text{drift}} = 4$)**
- **`["curvature_dem", "linear_xy"]` ($n_{\text{drift}} = 4$)**
- **`["z_dem", "curvature_dem", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["sia", "z_dem", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["sia", "curvature_dem", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["sia", "z_dem", "curvature_dem", "linear_xy"]` ($n_{\text{drift}} = 6$)**

##### Surface Mass Balance (`mass_balance`) + `linear_xy` Combinations (8 models — requires `mass_balance_path`)
- **`["mass_balance", "linear_xy"]` ($n_{\text{drift}} = 4$)**
- **`["sia", "mass_balance", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["z_dem", "mass_balance", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["curvature_dem", "mass_balance", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["sia", "z_dem", "mass_balance", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["sia", "curvature_dem", "mass_balance", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["z_dem", "curvature_dem", "mass_balance", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["sia", "z_dem", "curvature_dem", "mass_balance", "linear_xy"]` ($n_{\text{drift}} = 7$)**

##### Surface Flow Velocity (`velocity`) + `linear_xy` Combinations (16 models — requires `velocity_path`; 8 also require `mass_balance_path`)
- **`["velocity", "linear_xy"]` ($n_{\text{drift}} = 4$)**
- **`["sia", "velocity", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["z_dem", "velocity", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["curvature_dem", "velocity", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["mass_balance", "velocity", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["sia", "z_dem", "velocity", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["sia", "curvature_dem", "velocity", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["sia", "mass_balance", "velocity", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["z_dem", "curvature_dem", "velocity", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["z_dem", "mass_balance", "velocity", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["curvature_dem", "mass_balance", "velocity", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["sia", "z_dem", "curvature_dem", "velocity", "linear_xy"]` ($n_{\text{drift}} = 7$)**
- **`["sia", "z_dem", "mass_balance", "velocity", "linear_xy"]` ($n_{\text{drift}} = 7$)**
- **`["sia", "curvature_dem", "mass_balance", "velocity", "linear_xy"]` ($n_{\text{drift}} = 7$)**
- **`["z_dem", "curvature_dem", "mass_balance", "velocity", "linear_xy"]` ($n_{\text{drift}} = 7$)**
- **`["sia", "z_dem", "curvature_dem", "mass_balance", "velocity", "linear_xy"]` ($n_{\text{drift}} = 8$)**

##### Coupled SIA Creep Velocity (`velocity_sia`) + `linear_xy` Combinations (8 models — requires `velocity_path`; 4 also require `mass_balance_path`)
- **`["velocity_sia", "linear_xy"]` ($n_{\text{drift}} = 4$)**
- **`["z_dem", "velocity_sia", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["curvature_dem", "velocity_sia", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["mass_balance", "velocity_sia", "linear_xy"]` ($n_{\text{drift}} = 5$)**
- **`["z_dem", "curvature_dem", "velocity_sia", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["z_dem", "mass_balance", "velocity_sia", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["curvature_dem", "mass_balance", "velocity_sia", "linear_xy"]` ($n_{\text{drift}} = 6$)**
- **`["z_dem", "curvature_dem", "mass_balance", "velocity_sia", "linear_xy"]` ($n_{\text{drift}} = 7$)**

---

#### 3. 39 Multi-Raster + 2nd-Order Quadratic Spatial Trend (`quadratic_xy`)

These models superimpose any non-empty combination of the physical raster covariates ($\mathcal{R}$) onto a 2nd-order parabolic regional surface ($p_{\text{coord}}(x,y) = a_X X + a_Y Y + a_{XX} X^2 + a_{YY} Y^2 + a_{XY} XY$, adding 5 coordinate columns):

$$U(x,y) = \beta_0 + \sum_{r \in \mathcal{R}} \beta_r f_r(x,y) + a_X X + a_Y Y + a_{XX} X^2 + a_{YY} Y^2 + a_{XY} XY, \qquad n_{\text{drift}} = 6 + |\mathcal{R}|$$

##### Core DEM + `quadratic_xy` Combinations (7 models — always available)
- **`["sia", "quadratic_xy"]` ($n_{\text{drift}} = 7$)**
- **`["z_dem", "quadratic_xy"]` ($n_{\text{drift}} = 7$)**
- **`["curvature_dem", "quadratic_xy"]` ($n_{\text{drift}} = 7$)**
- **`["z_dem", "curvature_dem", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["sia", "z_dem", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["sia", "curvature_dem", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["sia", "z_dem", "curvature_dem", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**

##### Surface Mass Balance (`mass_balance`) + `quadratic_xy` Combinations (8 models — requires `mass_balance_path`)
- **`["mass_balance", "quadratic_xy"]` ($n_{\text{drift}} = 7$)**
- **`["sia", "mass_balance", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["z_dem", "mass_balance", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["curvature_dem", "mass_balance", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["sia", "z_dem", "mass_balance", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["sia", "curvature_dem", "mass_balance", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["z_dem", "curvature_dem", "mass_balance", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["sia", "z_dem", "curvature_dem", "mass_balance", "quadratic_xy"]` ($n_{\text{drift}} = 10$)**

##### Surface Flow Velocity (`velocity`) + `quadratic_xy` Combinations (16 models — requires `velocity_path`; 8 also require `mass_balance_path`)
- **`["velocity", "quadratic_xy"]` ($n_{\text{drift}} = 7$)**
- **`["sia", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["z_dem", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["curvature_dem", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["mass_balance", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["sia", "z_dem", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["sia", "curvature_dem", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["sia", "mass_balance", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["z_dem", "curvature_dem", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["z_dem", "mass_balance", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["curvature_dem", "mass_balance", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["sia", "z_dem", "curvature_dem", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 10$)**
- **`["sia", "z_dem", "mass_balance", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 10$)**
- **`["sia", "curvature_dem", "mass_balance", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 10$)**
- **`["z_dem", "curvature_dem", "mass_balance", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 10$)**
- **`["sia", "z_dem", "curvature_dem", "mass_balance", "velocity", "quadratic_xy"]` ($n_{\text{drift}} = 11$)**

##### Coupled SIA Creep Velocity (`velocity_sia`) + `quadratic_xy` Combinations (8 models — requires `velocity_path`; 4 also require `mass_balance_path`)
- **`["velocity_sia", "quadratic_xy"]` ($n_{\text{drift}} = 7$)**
- **`["z_dem", "velocity_sia", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["curvature_dem", "velocity_sia", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["mass_balance", "velocity_sia", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**
- **`["z_dem", "curvature_dem", "velocity_sia", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["z_dem", "mass_balance", "velocity_sia", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["curvature_dem", "mass_balance", "velocity_sia", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**
- **`["z_dem", "curvature_dem", "mass_balance", "velocity_sia", "quadratic_xy"]` ($n_{\text{drift}} = 10$)**

---

### 1.9 Tier 3: Glaciological Rules

When interpolating the BSS product target $P(x,y) = D(x,y) \cdot \sin\alpha_{\text{opt}}(x,y)$, dividing the recovered product by $\sin\alpha_{\text{opt}}(x,y)$ while also using an inverse-slope drift (`"sia"`, $\propto \sin(\alpha)^{-1}$, or `"velocity_sia"`, $\propto u_s^{1/4} \sin(\alpha)^{-3/4}$) introduces a $1/\sin^2\alpha$ or $1/\sin^{7/4}\alpha$ double-scaling artifact at low-slope glacier margins.
- Every candidate containing `"sia"` or `"velocity_sia"` evaluated on target `"P"` logs a warning and is marked `WARNING` in the ranking table, but is **ranked normally** (not penalized).
- For target `"P"`, `ordinary` or slope-free drifts (`z_dem`, `curvature_dem`, `mass_balance`, `velocity`, `linear_xy`, `quadratic_xy`) are recommended, whereas `"sia"` and `"velocity_sia"` are designed for direct targets (`"T"` or `"D"`).

---

### 1.10 Multi-Model Comparison Suite, Directory Architecture & Comparison Tables

To systematically compare how competing drift hypotheses affect reconstructed bedrock topography, ice thickness, and basal shear stress, `PySole` supports evaluating and exporting multiple drift models in a single run:

#### 1. Selecting Multiple Models (Interactive or JSON)
- **Interactive Terminal Prompt**: Enter a single row (`3`), a hyphenated range (`1-5`), a comma-separated list (`1, 3, 5`), or mixed combinations (`1-3, 7`). The first model listed becomes the **primary production model** (driving Pass 1 migration soundings and return values), while all remaining models are queued for comparative evaluation.
- **Pre-Defined JSON Suite**: Pass a nested list of candidate models in `kriging_parameters.<stage>.drift_terms` (e.g. `[["velocity_sia", "z_dem"], ["sia", "mass_balance"], ["sia"], []]`, with `drift_analyzer: true` or `false`; see Section 2.2). All listed models are evaluated and exported non-interactively without terminal prompts.

#### 2. Output Directory & Figure Subfolder Architecture
- **Diagnostic Plots (`figures/` or `<plots_dir>`)**:
  - Pre-interpolation plots (`01`–`05`: variograms, $k_c$ optimization curves, 3D migration vectors) are saved in root `figures/`.
  - Post-interpolation plots (`06`–`11`: bedrock, embedded bedrock, thickness map & histogram, BSS map & histogram) are isolated into `figures/<model_slug>/` with `<model_slug>` appended to each filename (`06_<prefix>bedrock_elevation_and_uncertainty_<model_slug>.png`, ..., `11_...`).
- **Spatial Rasters & Comparison Tables (`results/` or `<output_dir>`)**:
  - All raster grids append `<model_slug>` (`[<output_prefix>_]bedrock_<model_slug>.tif`, `thickness_<model_slug>.tif`, `traveltime_<model_slug>.tif`, `basal_shear_stress_<model_slug>.tif`, and `*_uncertainty_<model_slug>.tif`) and are saved **flat inside `<output_dir>`**.
  - Migrated sounding coordinates (`[<output_prefix>_]migrated_points.csv`) are exported once for the primary model in root `<output_dir>`.
  - Unified cross-stage comparison tables (`drift_model_comparison.txt`, `drift_model_comparison.csv`, `drift_model_comparison.md`) are saved directly in `<output_dir>`, contrasting CV metrics (`RMSE`, `ΔRMSE %`, `AICc`, `ΔAICc`, `VIF`, `Status`) against full-domain traveltime $T(x,y)$ (Pass 1), ice thickness $H(x,y)$ (Pass 2), basal shear stress $\tau_b(x,y)$ (Pass 2), and optional mean Kriging uncertainty (`Mean Unc`, when `calculate_uncertainty: true`).

---

## 2. Configuration, CLI & Python API Examples

### 2.1 Dynamic Default Suite with Glaciological Rasters (24 / 48 / 60 / 120 Candidates)

When `drift_analyzer: true` and `drift_terms` is omitted (`null`), `PySole` automatically evaluates **24 candidates** (DEM only), **48 candidates** (with `mass_balance_path`), **60 candidates** (with `velocity_path`), or all **120 candidates** (when both `mass_balance_path` and `velocity_path` are provided, as shown below):

```json
{
  "inputs": {
    "dem_path": "surface_dem.tif",
    "outline_path": "glacier_outline.shp",
    "mass_balance_path": "decadal_smb.tif",
    "velocity_path": "decadal_velocity.tif",
    "survey_data_path": "gpr_picks.csv",
    "survey_data_type": "one_way_traveltime",
    "survey_profile_column": "line_id"
  },
  "kriging_parameters": {
    "pre_migration": {
      "interpolation_target": "T",
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

### 2.2 Explicit Single-Model, Combinatorial Subset & Non-Interactive Multi-Model Suites

- **Single Multi-Drift Model (`drift_analyzer: false` with a flat list)** in Pass 1 (`["velocity_sia", "mass_balance", "z_dem"]`) combined with a **Combinatorial Subset Suite (`drift_analyzer: true` with a flat list)** in Pass 2:

```json
{
  "inputs": {
    "dem_path": "surface_dem.tif",
    "outline_path": "glacier_outline.shp",
    "mass_balance_path": "decadal_smb.tif",
    "velocity_path": "decadal_velocity.tif",
    "survey_data_path": "gpr_picks.csv"
  },
  "kriging_parameters": {
    "pre_migration": {
      "interpolation_target": "T",
      "method": "universal",
      "drift_analyzer": false,
      "drift_terms": ["z_dem", "mass_balance", "velocity_sia"]
    },
    "post_migration": {
      "interpolation_target": "D",
      "method": "universal",
      "drift_analyzer": true,
      "drift_terms": [[], "sia", "z_dem", "mass_balance", "velocity_sia"]
    }
  }
}
```

- **Non-Interactive Explicit Multi-Model Comparison Suite (`drift_analyzer: false` with nested candidate lists)**: Evaluates and exports only the specified candidate models using the default spatial exclusion buffer radius ($r_{\text{buffer}} = a_0 / 2$) without interactive prompts:

```json
{
  "inputs": {
    "dem_path": "surface_dem.tif",
    "outline_path": "glacier_outline.shp",
    "mass_balance_path": "decadal_smb.tif",
    "velocity_path": "decadal_velocity.tif",
    "survey_data_path": "gpr_picks.csv"
  },
  "kriging_parameters": {
    "pre_migration": {
      "interpolation_target": "T",
      "method": "universal",
      "drift_analyzer": false,
      "drift_terms": [
        ["z_dem", "velocity_sia"],
        ["sia", "z_dem"],
        []
      ]
    },
    "post_migration": {
      "interpolation_target": "D",
      "method": "universal",
      "drift_analyzer": false,
      "drift_terms": [
        ["z_dem", "mass_balance", "velocity_sia"],
        ["sia", "z_dem", "mass_balance"],
        ["sia", "z_dem"],
        []
      ]
    }
  }
}
```

### 2.3 Command-Line Interface (CLI) Examples

```bash
# Run with the interactive Drift Analyzer enabled across both passes and LOPO-CV profile grouping
pysole pysole.json --drift-analyzer --profile-col line_id

# Run non-interactively in batch mode (automatically selects #1 ranked model or runs explicit suite)
pysole pysole.json --drift-analyzer --batch
```

### 2.4 Python API Examples (`Solver` & `recommend_drift_model`)

```python
import pysole

# Initialize Solver with optional decadal mass balance and surface flow velocity rasters
solver = pysole.Solver(
    dem="surface_dem.tif",
    outline="glacier_outline.shp",
    mass_balance="decadal_smb.tif",
    velocity="decadal_velocity.tif",
    survey_data_path="gpr_picks.csv",
    survey_profile_column="line_id",
    pre_interpolation_target="T",
    pre_kriging_method="universal",
    pre_drift_analyzer=True,
    post_interpolation_target="D",
    post_kriging_method="universal",
    post_drift_analyzer=True,
)

# Inspect the full 120-candidate Drift Analyzer ranking programmatically
rankings = solver.recommend_drift_model(stage="post_migration", interactive=False)
print(f"Top model: {rankings[0].name} (AICc = {rankings[0].aicc:.1f}, RMSE = {rankings[0].cv_rmse:.2f} m)")
```
