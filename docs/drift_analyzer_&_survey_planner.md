# PySole Drift Analyzer & Survey Planner Methodological Guide

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
4. **`curvature_dem`**: Surface Laplacian curvature $C_{k_c}(\mathbf{x}) = \nabla^2 Z_{\text{smooth}, k_c}(\mathbf{x}) = \frac{\partial^2 Z_{\text{smooth}}}{\partial x^2} + \frac{\partial^2 Z_{\text{smooth}}}{\partial y^2}$, capturing convex ridges ($C < 0$) versus concave troughs ($C > 0$).
5. **`linear_xy`**: Planar spatial coordinates $(X, Y)$.
6. **`quadratic_xy`**: 2nd-order spatial polynomial $(X, Y, X^2, Y^2, XY)$.
7. **Multi-Drift Combinations**: Systematic suite of 18 multi-drift combinations combining physical raster covariates (`["sia"]`, `["z_dem"]`, `["curvature_dem"]`) with planar (`["linear_xy"]`) and quadratic (`["quadratic_xy"]`) regional trends (Section 1.8). Together with the 5 single terms this gives 23 drift models; with Ordinary Kriging the default suite has **24 candidates**.

---

### 1.2 Configuration: Which Models Are Evaluated and Applied

`kriging_parameters.<stage>.drift_terms` is `null` (omitted) or a flat list of the five primitive terms `sia`, `z_dem`, `curvature_dem`, `linear_xy`, `quadratic_xy`. Inside an analyzer suite exactly one nested `[]` is the stand-alone Ordinary Kriging candidate. A top-level `[]` is never accepted for `universal` and `regression` (use `null` to omit the terms). Together with `method` and `drift_analyzer` the list is interpreted as follows (one rule table for the configuration file, the CLI and the Python API):

| `method` | `drift_analyzer` | Meaning of `drift_terms` |
| :--- | :--- | :--- |
| `ordinary` | any | **`ordinary` overrides `drift_analyzer` and `drift_terms`**: both are ignored (the terms are not even validated); one warning is logged if either is set. The analyzer never runs for `ordinary`. |
| `universal` | `false` | The list is **ONE multi-drift model** (all terms together). `null` or `[]` is an error ("specify drift_terms or use method ordinary"). |
| `universal` | `true` | `null`: the **default suite** (24 candidates). A list: the **suite** of all combinations of the listed terms. The winner decides the final model: the Ordinary candidate gives `ordinary`, any drift model gives `universal` with its terms. |
| `regression` | `false` | The terms are the **Random Forest covariates** (`null`: x and y); works for every primitive. |
| `regression` | `true` | The analyzer is skipped with a warning (it ranks Universal Kriging drift models). |

If the method is omitted altogether, the target decides: the direct targets (`"T"` before, `"D"` after migration) use `universal` with `["sia"]` (with `drift_analyzer: true` and `null` terms: the default suite), the product target `"P"` uses `ordinary`.

#### Suite expansion

The suite is every non-empty subset of the listed primitives **except subsets containing both `linear_xy` and `quadratic_xy`** (the quadratic trend already contains the linear one), in canonical order and names (`sia + z_dem`), duplicates removed; a nested `[]` adds the Ordinary candidate (never combined with terms).

| `drift_terms` (analyzer on) | Candidates |
| :--- | :--- |
| `null` | 24 (all 5 primitives + Ordinary) |
| `["sia", "z_dem", "curvature_dem", "linear_xy", "quadratic_xy"]` | 23 |
| `[[], "sia", "z_dem"]` | 4 (Ordinary, `sia`, `z_dem`, `sia + z_dem`) |
| `["sia", "z_dem", "curvature_dem"]` | 7 |
| `["sia", "z_dem", "quadratic_xy"]` | 7 |
| `["sia"]` | 1 |
| `[]` | error (use `null`) |

A suite of exactly **one** candidate (a single-term list such as `["sia"]`) overrides the analyzer: no analysis is run (an info message is logged) and that model is applied. Note that with the analyzer on, a list of several terms is always a suite; for ONE multi-drift model such as `["sia", "z_dem"]` set `drift_analyzer: false`.

#### Winner handling

The best-ranked candidate is applied. In an interactive terminal the ranking table is shown and the model can be chosen by row number (`3`, `#3`), printed name, term list (`['sia', 'z_dem']`, `sia + z_dem`), `ordinary` or `[]`; Enter keeps #1. In batch mode (`--batch`, no terminal) #1 is applied. Penalized candidates (VIF above 10 or a numerical failure) are always ranked after all other candidates. If no candidate can be cross-validated, a warning is logged and Ordinary Kriging is applied. For the product target `"P"` a warning is logged for the finally applied model if it contains `sia`.

#### Engine support

| Engine | `ordinary` | `universal` (all 23 drift models) | `regression` | `drift_analyzer` |
| :--- | :--- | :--- | :--- | :--- |
| `"native"` (default) | yes | yes | PyKrige is used for the stage (logged) | yes |
| `"pykrige"` | yes | yes, same prediction as native (max. difference below 1e-4 of the value range); **slow for many survey points** (warning, see below) | yes | candidates are ranked with the native engine, the winner is applied with PyKrige |

**Speed of `"pykrige"` with `"universal"`:** PyKrige's `UniversalKriging` computes a leave-one-out fit statistic every time it is created (the class has no option to switch it off, unlike `OrdinaryKriging`). Its run time grows roughly with N³ to N⁴ in the number of survey points, independent of the drift terms and of the grid size: about 2 s for 400 points, 19 s for 800 and several minutes for the 1098 points of the WUK example, while the kriging itself is fast. PySole logs a warning when `engine: "pykrige"` is combined with `method: "universal"`. Use the `"native"` engine (all drift terms, kriging variance) for Universal Kriging with more than a few hundred survey points; `"ordinary"` and `"regression"` with PyKrige are not affected.

PyKrige uses the fitted variogram of `PySole` (so both engines interpolate with the same model) and accepts every drift term: raster terms as several `specified` drifts, `linear_xy` and `quadratic_xy` as scaled functional drifts. For `regression` the covariates are the standardised raster values and the polynomial columns of the listed terms. PyKrige's own estimation variance is about 1 % below the native one (the native variance equals a dense solve of the kriging system); `regression` provides no variance. Only PyKrige supports the 1-D `linear` variogram model; the native engine falls back to `spherical`.

---

### 1.3 Multi-Tier Diagnostic Evaluation Pipeline

The `DriftAnalyzer` evaluates all candidates of the suite (by default 24: 23 single and multi-drift models and Ordinary Kriging) across three diagnostic tiers:

```
[ Input Survey Data & Covariates ]
               │
               ▼
┌──────────────────────────────┐
│ Tier 1: Multicollinearity    │ ──► VIF audit of every candidate (VIF > 10 → penalized)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Tier 2: Exact Spatial CV     │ ──► Leave-One-Profile-Out blocks (LOPO-CV) [if profile_col]
│   (RMSE, MAE, R², AICc)      │ ──► Spatial buffer blocks (r_buffer = a_0 / 2) [fallback]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Tier 3: Glaciological Rules  │ ──► Warning (sia drift on product P)
└──────────────┬───────────────┘
               │
               ▼
[ Ranked Candidates with ΔAICc / ΔRMSE %; the best one is applied ]
```

---

### 1.4 Tier 1: Multicollinearity Auditing

When multiple spatial drift terms $f_1, \dots, f_K$ are combined in a Universal Kriging model, high collinearity between covariates can cause severe numerical instability (ill-conditioning) in the Dual Kriging coefficient matrix $K$.

To detect collinearity, the `DriftAnalyzer` computes the **Variance Inflation Factor (VIF)** for each covariate $f_j$ within a multi-drift set:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the coefficient of determination obtained by regressing covariate $f_j$ against all remaining $K-1$ covariates in the drift set.
- **$\text{VIF}_j = 1$**: Complete orthogonality (no collinearity).
- **$\text{VIF}_j > 10$**: Severe multicollinearity. Multi-drift combinations containing any covariate with $\text{VIF}_j > 10$ are flagged as **penalized** (`is_penalized = True`) and ranked after all other models. A numerical failure of the cross-validation also penalizes a candidate.

---

### 1.5 Tier 2: Geostatistical Spatial Cross-Validation & Ranking Metrics

Standard Leave-One-Out Cross-Validation (LOOCV) can severely underestimate generalization errors on dense line-survey data (such as GPR profiles) due to strong along-track spatial autocorrelation. The `DriftAnalyzer` addresses this by removing whole spatial *blocks* of observations before predicting them:

#### 1. Leave-One-Profile-Out Cross-Validation (LOPO-CV)
Activated when `inputs.survey_profile_column` (e.g., `line_id`, `profile_id`) is specified and contains at least 3 distinct profile tracks ($M \ge 3$).
- For each profile track $m \in \{1, \dots, M\}$, all observations belonging to profile $m$ are held out simultaneously as validation set $V_m$.
- The kriging system is trained on the remaining profiles $T_m = \bigcup_{k \neq m} V_k$ and the predictions $\hat{y}_i$ are evaluated for all points $i \in V_m$.
- Measures true inter-track interpolation performance across unmeasured glacier sectors.

#### 2. Spatial Buffer LOOCV (Fallback)
Used when no profile column is defined (or fewer than 3 profiles exist). For each validation node $\mathbf{x}_i$ all neighboring survey observations $\mathbf{x}_j$ located within a spatial exclusion buffer radius $r_{\text{buffer}}$ are excluded from training:

$$d(\mathbf{x}_i, \mathbf{x}_j) < r_{\text{buffer}} \implies \mathbf{x}_j \notin \text{Training Set}$$

The buffer radius is linked to the fitted variogram range $a_0$ (in an interactive terminal it can be changed at the prompt):

$$r_{\text{buffer}} = \frac{a_0}{2}$$

This eliminates along-track autocorrelation leakage during validation.

#### 3. Exact Block Leave-Out Evaluation
All candidates and all blocks are evaluated **exactly** from **one** factorization. The kriging matrix of the Ordinary system $A_0 = \begin{pmatrix} \Gamma & \mathbf{1} \\ \mathbf{1}^T & 0 \end{pmatrix}$ (variogram and regularization of the production run, the same for every candidate) is factorized once and inverted to $P = A_0^{-1}$. A candidate with drift columns $G$ is a low-rank update of $P$ (all primitives are columns of one matrix), which gives its dual weights $\mathbf{c}$. Removing a block $B$ gives the leave-block-out residuals (the leave-one-out identity of Dubrule, generalized to blocks)

$$\mathbf{e}_B = \left(P_{BB}\right)^{-1} \mathbf{c}_B, \qquad \hat{\mathbf{y}}_B = \mathbf{y}_B - \mathbf{e}_B$$

These residuals equal those of a freshly fitted kriging system per fold up to rounding error (about 1e-11 in the tests). Consequently:
- **Every survey point is evaluated** ($n = N$, deterministic; there is no fold count, no random sub-sampling and no tuning knob). All candidates are scored on the same points: a block that leaves fewer than $n_{\text{drift}} + 3$ training points is skipped for the candidate.
- The cost is one $O(N^3)$ factorization plus cheap low-rank updates. Measured on a 16-thread workstation for the complete default suite (24 candidates, all points): Wuk (N = 982) 3.1 s pre- and 3.3 s post-migration, Gok (N = 2033) 7.0 s.
- The memory demand is about $3 N^2$ doubles (N = 5000: 0.6 GB, N = 10 000: 2.4 GB). If this exceeds `max_memory_fraction` of the available memory, the LU factors are kept instead of the inverse and only a deterministic farthest-point subset of the evaluation points (at least 100) is evaluated; a warning is logged and $n$ is then the size of the subset.
- With `inputs.show_progress` (default `true`) the terminal shows an elapsed-time line while the base system is factorised (it appears only after about one second) and one progress bar over the blocks (points, or profiles for LOPO). The display only reports; it never changes or slows the calculation (results are bit-identical with it switched off).

#### 4. Cross-Validation Performance Metrics
For observed values $y_i$ and cross-validated predictions $\hat{y}_i$ of the $n$ evaluated points ($i = 1, \dots, n$):

- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

- **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

- **Corrected Akaike Information Criterion (AICc)**:
  Balancing the residual sum of squares against the number of model parameters $k = n_{\text{drift}}$ (the constant plus all drift columns, see Section 1.8):

  $$\text{AICc} = n \ln\left(\frac{\text{SS}_{\text{res}}}{n}\right) + 2k + \frac{2k(k+1)}{n - k - 1}$$

  where $\text{SS}_{\text{res}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2$.

#### 5. Ranking and the ΔAICc / ΔRMSE % Columns
Candidates are ranked by (1) **not penalized first** (penalized: VIF above 10 or a numerical failure), (2) **AICc ascending** and (3) CV-RMSE as the tie-break. The primary ranking key is therefore the AICc, not the RMSE: models with almost identical RMSE can swap places because of their different number of parameters (rows 5 and 6 below). Two columns show the distance to the best model:

- **ΔAICc** = AICc − AICc of the best non-penalized model.
- **ΔRMSE %** = $100 \cdot (\text{RMSE} / \text{RMSE}_{\#1} - 1)$, relative to the best-ranked model.

Example (Wuk, pre-migration, product target `"P"`, N = 982, default suite of 24 candidates, as printed by the Drift Analyzer; rows 9-15 and 17-24 are omitted here):

```
================================================================================================================
       DRIFT ANALYZER CANDIDATE MODEL RECOMMENDATIONS (PRE_MIGRATION)
================================================================================================================
#   Drift Model                                  RMSE (m)   dRMSE %   AICc       dAICc     VIF      Status
----------------------------------------------------------------------------------------------------------------
1   z_dem + curvature_dem                        13.77      0.00      5156.0     0.0       1.1      VALID
2   z_dem                                        14.93      8.43      5312.9     156.9     1.0      VALID
3   sia + z_dem                                  15.33      11.37     5367.6     211.6     2.4      WARNING
4   curvature_dem + linear_xy                    15.99      16.18     5452.5     296.5     1.2      VALID
5   ordinary                                     16.91      22.84     5555.9     399.9     1.0      VALID
6   sia + curvature_dem + linear_xy              16.88      22.62     5560.5     404.5     2.9      WARNING
7   quadratic_xy                                 17.18      24.78     5596.9     440.9     2.9      VALID
8   curvature_dem + quadratic_xy                 17.18      24.77     5598.8     442.8     4.8      VALID
...
16  z_dem + curvature_dem + linear_xy            16.19      17.60     5478.4     322.4     93.5     PENALIZED
----------------------------------------------------------------------------------------------------------------
WARNING: SIA slope drift on Product P target causes 1/sin^2(alpha) double-scaling margin artifacts.
```

`WARNING` marks a candidate with a note (here the `sia` drift on a product target, Section 1.9); `PENALIZED` models (VIF > 10) are listed last even though several of them (for example row 16) have a lower RMSE than the Ordinary model (row 5).

---

### 1.6 Reading the AICc: ΔAICc, ΔRMSE % and the Role of $n$

The cross-validation residuals are already out-of-sample, but the AICc still weighs them against the number of parameters. Because $\text{SS}_{\text{res}}$ is summed over $n$ points, the likelihood term of the AICc **grows with $n$** whereas the penalty $2k + 2k(k+1)/(n-k-1)$ does not. For two candidates, a complex one (more parameters, $\Delta k = k_1 - k_2$) and a simple one,

$$\text{AICc}_{\text{simple}} - \text{AICc}_{\text{complex}} \approx n \ln\frac{\text{SS}_{\text{simple}}}{\text{SS}_{\text{complex}}} - 2\,\Delta k = 2n \ln\frac{\text{RMSE}_{\text{simple}}}{\text{RMSE}_{\text{complex}}} - 2\,\Delta k ,$$

so the complex model wins as soon as $n \ln(1 + g) > \Delta k$ for the relative RMSE gain $g$. The more survey points exist, the smaller the RMSE gain that suffices.

**Worked example.** A relative RMSE gain of only $g = 0.14\,\%$ with $\Delta k = 1$ gives $2n \ln(1.0014) \approx 0.0028\,n$:
- $n = 400$: 1.1 < $2\Delta k$ = 2, the simpler model wins.
- $n = 982$: 2.7 > 2, the more complex model wins (break-even at $n \approx 700$).

The same data set can therefore select different models if $n$ changes (for example by sub-sampling the survey points), although the physical benefit of the extra term is identical. The analyzer evaluates **all** survey points, so $n = N$ is a property of the data set and not a policy choice; AICc values of PySole 0.4.4 (which used 100 random points) are not comparable with the present ones.

A real example from the Gok data set (post-migration, N = 2033) shows a statistically clear but practically small difference:

```
#   Drift Model                                  RMSE (m)   dRMSE %   AICc       dAICc     VIF      Status
----------------------------------------------------------------------------------------------------------------
1   sia + linear_xy                              4.11       0.00      5756.3     0.0       1.5      WARNING
2   linear_xy                                    4.13       0.52      5775.4     19.1      1.1      VALID
```

Adding `sia` to `linear_xy` reduces the RMSE by only 0.52 %, but the ΔAICc of 19.1 is large because $n$ is large.

**Best practice**
- Read **ΔAICc together with ΔRMSE %**: ΔAICc tells whether a difference is detectable with $n$ points, ΔRMSE % whether it matters. A rule of thumb for ΔAICc: below 2 the models are practically indistinguishable, 4-7 clearly less supported, above 10 essentially unsupported.
- If the best models are within ΔRMSE % of about 1 % (or ΔAICc below about 2), prefer the **simpler and physically better justified** model; statistical rankings do not replace physical judgment.
- Use `inputs.survey_profile_column` (LOPO blocks) whenever profile IDs exist. Without profiles check that the top models are stable when the buffer radius (default $a_0/2$, asked for in the interactive prompt) is changed.
- Treat `WARNING` (for example `sia` on a product target) and `PENALIZED` (VIF) models with care even if they rank high.

---

### 1.7 Covariates and Variogram Used by the Analyzer

The analyzer uses exactly the covariates and the variogram of the production run:

- **Variogram:** one variogram model is fitted once per stage and shared by every candidate and by the production Kriging (`PySole` fits a spherical model to the BSS product $P$ for all targets; with direct targets `"T"` / `"D"` this fit is used as it is). Candidates differ only in their drift columns; the regularization (relative Tikhonov offset of $10^{-6}$ times the sill) is the same.
- **Raster covariates** are built once per stage: `sia` is $1/\sin(\alpha_{\text{opt}})$ with the slope floor `slope_floor_deg` applied, `curvature_dem` is the Laplacian of the DEM smoothed with the optimized corner wavenumber $k_c$ and `z_dem` is the DEM. All `PySole` grids are bottom-up (row 0 is the southernmost row); the covariates are sampled at the survey points by bilinear interpolation on the grid coordinates with the same code as in the production run, so analyzer and production see identical values. Every raster covariate is standardized with the mean and standard deviation of its full grid (cells without data are set to 0 after standardization); the coordinate trends use centered and scaled coordinates.
- **Analyzer and production** share one Kriging system (`KrigingSystem`) and one drift construction, so a ranking is not an approximation of the production run.

---

### 1.8 Systematic Multi-Drift Model Suite

Universal Kriging in `PySole` natively supports combining external raster drifts (`["sia"]`, `["z_dem"]`, `["curvature_dem"]`) with polynomial spatial coordinates (`["linear_xy"]`, `["quadratic_xy"]`) into an augmented trend model $U(x,y) = \sum_{k=0}^K \beta_k f_k(x,y)$. The complete suite comprises **18 systematic multi-drift combinations** categorized into three structural groups. $n_{\text{drift}}$ in the headings counts all columns of the augmented system **including the constant** (Ordinary baseline); the number of drift columns without the constant is `CandidateDriftResult.n_terms = n_drift - 1`, and the AICc uses $k = n_{\text{drift}}$:

#### 1. Multi-Raster Physical Combinations (No Coordinate Trend)

These models couple multiple terrain raster covariates without assuming a regional spatial coordinate tilt:

- **`["sia", "z_dem"]` ($n_{\text{drift}} = 3$)**:
  $$U(x,y) = a_1 \sin(\alpha_{\text{opt}}(x,y))^{-1} + a_2 Z_{\text{dem}}(x,y)$$
  *Glaciological Rationale*: Couples local slope-driven ice flux dynamics (Shallow Ice Approximation) with macroscopic valley hypsometry (elevation lapse rate and accumulation gradients). Intended for direct targets (`"T"` or `"D"`); on product targets $P$ it is flagged with a warning.

- **`["sia", "curvature_dem"]` ($n_{\text{drift}} = 3$)**:
  $$U(x,y) = a_1 \sin(\alpha_{\text{opt}}(x,y))^{-1} + a_2 \nabla^2 Z_{\text{smooth}}(x,y)$$
  *Glaciological Rationale*: Couples SIA slope dynamics with 2D terrain curvature, capturing enhanced overdeepenings in concave bowls ($\nabla^2 Z > 0$) and reduced thickness across divergent convex ridges ($\nabla^2 Z < 0$). Intended for direct targets (`"T"` or `"D"`); on product targets $P$ it is flagged with a warning.

- **`["z_dem", "curvature_dem"]` ($n_{\text{drift}} = 3$)**:
  $$U(x,y) = a_1 Z_{\text{dem}}(x,y) + a_2 \nabla^2 Z_{\text{smooth}}(x,y)$$
  *Glaciological Rationale*: Terrain morphometry model independent of slope singularity; combines regional elevation gradients with local concave/convex topographic focusing. Contains no `sia` term, hence no double scaling on product targets $P$.

- **`["sia", "z_dem", "curvature_dem"]` ($n_{\text{drift}} = 4$)**:
  $$U(x,y) = a_1 \sin(\alpha_{\text{opt}}(x,y))^{-1} + a_2 Z_{\text{dem}}(x,y) + a_3 \nabla^2 Z_{\text{smooth}}(x,y)$$
  *Glaciological Rationale*: Full physical multi-raster model uniting viscous ice flow physics (SIA), valley hypsometry ($Z_{\text{dem}}$), and cross-sectional bowl concavity ($\nabla^2 Z_{\text{smooth}}$). Intended for direct targets (`"T"` or `"D"`); on product targets $P$ it is flagged with a warning.

#### 2. Multi-Raster + 1st-Order Linear Spatial Trend (`linear_xy`)

These models overlay physical raster covariates onto a flat, planar regional spatial gradient ($a_X X + a_Y Y$):

- **`["sia", "linear_xy"]` ($n_{\text{drift}} = 4$)**:
  $$U(x,y) = a_1 \sin(\alpha_{\text{opt}}(x,y))^{-1} + a_2 X + a_3 Y$$
  *Glaciological Rationale*: Viscous flow physics modulated by a constant regional geographic planar gradient across the catchment. Intended for direct targets (`"T"` or `"D"`); on product targets $P$ it is flagged with a warning.

- **`["z_dem", "linear_xy"]` ($n_{\text{drift}} = 4$)**:
  $$U(x,y) = a_1 Z_{\text{dem}}(x,y) + a_2 X + a_3 Y$$
  *Glaciological Rationale*: Elevation-dependent valley thickening superimposed on an asymmetric regional tectonic or climatic tilt. Contains no `sia` term, hence no double scaling on product targets $P$.

- **`["curvature_dem", "linear_xy"]` ($n_{\text{drift}} = 4$)**:
  $$U(x,y) = a_1 \nabla^2 Z_{\text{smooth}}(x,y) + a_2 X + a_3 Y$$
  *Glaciological Rationale*: Morphometric terrain concavity/convexity combined with a planar spatial gradient across the glacier domain. Contains no `sia` term, hence no double scaling on product targets $P$.

- **`["z_dem", "curvature_dem", "linear_xy"]` ($n_{\text{drift}} = 5$)**:
  $$U(x,y) = a_1 Z_{\text{dem}}(x,y) + a_2 \nabla^2 Z_{\text{smooth}}(x,y) + a_3 X + a_4 Y$$
  *Glaciological Rationale*: Joint elevation and curvature morphometry overlaid on a 1st-order regional geographic trend. Contains no `sia` term, hence no double scaling on product targets $P$.

- **`["sia", "z_dem", "linear_xy"]` ($n_{\text{drift}} = 5$)**:
  $$U(x,y) = a_1 \sin(\alpha_{\text{opt}}(x,y))^{-1} + a_2 Z_{\text{dem}}(x,y) + a_3 X + a_4 Y$$
  *Glaciological Rationale*: SIA slope dynamics and elevation hypsometry augmented with a linear geographic planar tilt. Intended for direct targets (`"T"` or `"D"`); on product targets $P$ it is flagged with a warning.

- **`["sia", "curvature_dem", "linear_xy"]` ($n_{\text{drift}} = 5$)**:
  $$U(x,y) = a_1 \sin(\alpha_{\text{opt}}(x,y))^{-1} + a_2 \nabla^2 Z_{\text{smooth}}(x,y) + a_3 X + a_4 Y$$
  *Glaciological Rationale*: SIA slope dynamics and terrain Laplacian concavity augmented with a linear geographic planar tilt. Intended for direct targets (`"T"` or `"D"`); on product targets $P$ it is flagged with a warning.

- **`["sia", "z_dem", "curvature_dem", "linear_xy"]` ($n_{\text{drift}} = 6$)**:
  $$U(x,y) = a_1 \sin(\alpha_{\text{opt}}(x,y))^{-1} + a_2 Z_{\text{dem}}(x,y) + a_3 \nabla^2 Z_{\text{smooth}}(x,y) + a_4 X + a_5 Y$$
  *Glaciological Rationale*: Complete physical raster trio (SIA, elevation, curvature) combined with a regional linear planar gradient. Intended for direct targets (`"T"` or `"D"`); on product targets $P$ it is flagged with a warning.

#### 3. Multi-Raster + 2nd-Order Quadratic Spatial Trend (`quadratic_xy`)

These models overlay physical raster covariates onto a 2nd-order parabolic spatial surface ($a_X X + a_Y Y + a_{XX} X^2 + a_{YY} Y^2 + a_{XY} XY$):

- **`["sia", "quadratic_xy"]` ($n_{\text{drift}} = 7$)**:
  $$U(x,y) = a_1 \sin(\alpha_{\text{opt}}(x,y))^{-1} + a_2 X + a_3 Y + a_4 X^2 + a_5 Y^2 + a_6 XY$$
  *Glaciological Rationale*: SIA slope physics combined with a 2nd-order parabolic spatial surface capturing large-scale ice-cap dome or basin curvature. Intended for direct targets (`"T"` or `"D"`); on product targets $P$ it is flagged with a warning.

- **`["z_dem", "quadratic_xy"]` ($n_{\text{drift}} = 7$)**:
  $$U(x,y) = a_1 Z_{\text{dem}}(x,y) + a_2 X + a_3 Y + a_4 X^2 + a_5 Y^2 + a_6 XY$$
  *Glaciological Rationale*: Elevation hypsometry combined with a 2D quadratic spatial surface, ideal for radial ice caps or complex branched catchments. Contains no `sia` term, hence no double scaling on product targets $P$.

- **`["curvature_dem", "quadratic_xy"]` ($n_{\text{drift}} = 7$)**:
  $$U(x,y) = a_1 \nabla^2 Z_{\text{smooth}}(x,y) + a_2 X + a_3 Y + a_4 X^2 + a_5 Y^2 + a_6 XY$$
  *Glaciological Rationale*: Morphometric terrain concavity/convexity combined with a quadratic regional surface. Contains no `sia` term, hence no double scaling on product targets $P$.

- **`["z_dem", "curvature_dem", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**:
  $$U(x,y) = a_1 Z_{\text{dem}}(x,y) + a_2 \nabla^2 Z_{\text{smooth}}(x,y) + a_3 X + a_4 Y + a_5 X^2 + a_6 Y^2 + a_7 XY$$
  *Glaciological Rationale*: Dual elevation-curvature morphometry combined with a 2nd-order spatial polynomial. Recommended for alpine valley glaciers and ice caps. Contains no `sia` term, hence no double scaling on product targets $P$.

- **`["sia", "z_dem", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**:
  $$U(x,y) = a_1 \sin(\alpha_{\text{opt}}(x,y))^{-1} + a_2 Z_{\text{dem}}(x,y) + a_3 X + a_4 Y + a_5 X^2 + a_6 Y^2 + a_7 XY$$
  *Glaciological Rationale*: SIA slope physics and elevation hypsometry combined with 2D quadratic spatial coordinates. Intended for direct targets (`"T"` or `"D"`); on product targets $P$ it is flagged with a warning.

- **`["sia", "curvature_dem", "quadratic_xy"]` ($n_{\text{drift}} = 8$)**:
  $$U(x,y) = a_1 \sin(\alpha_{\text{opt}}(x,y))^{-1} + a_2 \nabla^2 Z_{\text{smooth}}(x,y) + a_3 X + a_4 Y + a_5 X^2 + a_6 Y^2 + a_7 XY$$
  *Glaciological Rationale*: SIA slope physics and Laplacian concavity combined with 2D quadratic spatial coordinates. Intended for direct targets (`"T"` or `"D"`); on product targets $P$ it is flagged with a warning.

- **`["sia", "z_dem", "curvature_dem", "quadratic_xy"]` ($n_{\text{drift}} = 9$)**:
  $$U(x,y) = a_1 \sin(\alpha_{\text{opt}}(x,y))^{-1} + a_2 Z_{\text{dem}}(x,y) + a_3 \nabla^2 Z_{\text{smooth}}(x,y) + a_4 X + a_5 Y + a_6 X^2 + a_7 Y^2 + a_8 XY$$
  *Glaciological Rationale*: Comprehensive physical and spatial representation combining all three raster drivers with a full 2nd-order spatial polynomial. Intended for direct targets (`"T"` or `"D"`); on product targets $P$ it is flagged with a warning.

---

### 1.9 Tier 3: Glaciological Rules

The `DriftAnalyzer` applies one domain-specific rule:
- **Product Target Warning**: When interpolating BSS product targets $P(x,y) = D(x,y) \sin \alpha(x,y)$, applying slope-dependent drifts (`sia`, $\sin(\alpha)^{-1}$) causes double-scaling ratio artifacts at low-slope margins ($P / \sin \alpha \to D / \sin^2 \alpha$). Every tested candidate containing `sia` on a product target $P$ logs **one warning** and is flagged in the table (`WARNING`), but all metrics are computed and the candidate is **ranked normally** (it is not penalized). The slope floor `slope_floor_deg` only keeps the drift finite in nearly flat cells; there is no further safeguard against the double scaling, so `ordinary` or terrain drifts (`z_dem`, `curvature_dem`) are recommended for product targets. The warning is logged again for the finally applied model if it contains `sia`.

---

## 2. Survey Planner (`SurveyPlanner`)

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

### 2.1 Required User Input Data & Default Parameters

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
| **FFT DEM Cutoff Wavelength** | `kc` / `--kc` | **`0.0314`** rad/m | Corner wavenumber frequency cutoff ($k_c = 0.0314\text{ rad/m}$, corresponding to cutoff wavelength $\lambda_c = 2\pi / k_c \approx 200\text{ m}$) applied during frequency-domain Gaussian DEM surface smoothing. |
| **Nominal Basal Shear Stress** | `tau_0` / `tau_p` / `--tau` | **`100`** kPa ($100,000\text{ Pa}$) | Target nominal basal shear stress ($\tau_0 = 100\text{ kPa}$) used in Shallow Ice Approximation (SIA) ice thickness depth modeling $D_{\text{SIA}}(x,y) = \frac{\tau_0}{\rho g \sin \alpha}$. |
| **Total Survey Track Budget** | `max_length_km` / `max_km` / `--max-km` | **`5.0`** km ($5,000\text{ m}$) | Maximum total survey length budget ($L_{\text{max}} = 5.0\text{ km}$) allocated across longitudinal flowlines and transverse cross-profiles. |
| **Minimum Surface Slope Floor** | `slope_floor_deg` | `5.0`° | Minimum surface slope angle threshold ($\alpha_{\text{floor}} = 5^\circ$) to prevent numerical division singularities in low-gradient accumulation basins. |
| **Ice Density** | `ice_density` | `900.0` kg/m³ | Bulk ice density ($\rho_{\text{ice}} = 900\text{ kg/m}^3$). |
| **Gravitational Acceleration** | `g` | `9.81` m/s² | Standard gravitational acceleration ($g = 9.81\text{ m/s}^2$). |

---

### 2.2 Synthetic SIA Ice Thickness Modeling

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

### 2.3 Survey Profile Track Layout Algorithm

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

### 2.4 Output Formats & Field Exports

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

With `drift_analyzer: true` and `drift_terms` omitted (`null`) the default suite of 24 candidates is evaluated; `drift_terms` select the suite or the model:

```json
{
  "kriging_parameters": {
    "pre_migration": {
      "interpolation_target": "T",
      "method": "universal",
      "drift_analyzer": false,
      "drift_terms": ["sia", "z_dem"]
    },
    "post_migration": {
      "interpolation_target": "D",
      "method": "universal",
      "drift_analyzer": true,
      "drift_terms": [[], "sia", "z_dem"]
    }
  }
}
```

The pre-migration stage applies ONE multi-drift model (`sia` + `z_dem`); the post-migration stage ranks the 4 candidates Ordinary, `sia`, `z_dem` and `sia + z_dem` and applies the best one.

### 3.2 CLI Commands

```bash
# Execute PySole with interactive Drift Analyzer and profile grouping
pysole pysole.json --drift-analyzer --profile-col line_id

# Force non-interactive batch mode
pysole pysole.json --batch

# Standalone Survey Planning (defaults: k_c=0.0314, tau=100 kPa, max-km=5.0 km)
pysole plan-survey --dem surface_dem.tif --outline glacier_outline.shp --kc 0.0314 --tau 100 --max-km 5.0 --out-dir results --format tif
```

#### `pysole plan-survey` CLI Options & Path Resolution

Computes the shallow-ice-approximation (SIA) thickness model and proposes survey tracks within a length budget.

- **Options:**
  - `--dem`: File path to surface DEM raster (required).
  - `--outline`: File path to creeping body / glacier boundary polygon (optional).
  - `--kc`: FFT corner wavenumber in rad/m (default `0.0314`).
  - `--tau`: Target basal shear stress $\tau_0$ in kPa (default `100`).
  - `--max-km`: Survey length budget in km (default `5.0`).
  - `--prefix`: Output prefix (default `final`).
  - `--out-dir`: Output directory for generated deliverables.
  - `--plots-dir`: Output directory for diagnostic figures.
  - `--format`: Export raster format (`tif` or `asc`, default `tif`).

- **Output Path Resolution:** Outputs are kept together: files go to `--out-dir` (current directory if omitted); an absolute `--prefix` places all files beside it; figures are saved in `figures/` next to the data files unless `--plots-dir` is an absolute path.

