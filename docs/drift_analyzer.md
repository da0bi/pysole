# Universal Kriging Drift Analyzer Guide

This document provides a comprehensive methodological and mathematical reference for the **Universal Kriging Drift Analyzer (`DriftAnalyzer`)** in `PySole`, an automated diagnostic and model selection engine for evaluating, ranking, and comparing spatial drift models.

> [!TIP]
> For forward campaign design on unprobed glaciers without radar sounding data, see the dedicated [Survey Planner Guide](survey_planner.md).

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
| `universal` | `false` | The list is **ONE multi-drift model** (all terms together). `null` or `[]` is an error ("specify drift_terms or use method ordinary"). If both `linear_xy` and `quadratic_xy` are specified, `linear_xy` is dropped with a warning (fallback to `["quadratic_xy"]` which contains the linear terms [x, y]). |
| `universal` | `true` | `null`: the **default suite** (24 candidates). A list: the **suite** of all combinations of the listed terms. The winner decides the final model: the Ordinary candidate gives `ordinary`, any drift model gives `universal` with its terms. |
| `regression` | `false` | The terms are the **Random Forest covariates** (predictor features, e.g. `["sia", "z_dem"]`) used to train a Random Forest model to predict the large-scale trend before Ordinary Kriging interpolates the residuals. **Default (`null`)**: trains on all available primitives (`["sia", "z_dem", "curvature_dem", "quadratic_xy"]`); spatial coordinates only: `["linear_xy"]`; works for every primitive; executed via PyKrige. |
| `regression` | `true` | The analyzer is skipped with a warning (the Drift Analyzer evaluates Universal Kriging linear drift models; Regression Kriging proceeds directly using the specified `drift_terms`). |

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

#### Complete Model Suite & Drift Analyzer State Matrix

The configuration interaction between Kriging `method`, `drift_analyzer`, and `drift_terms` is fully governed by the following state matrix:

| Kriging `method` | `drift_analyzer` | `drift_terms` syntax | Interpreted Action | Effective `drift_analyzer` |
| :--- | :--- | :--- | :--- | :--- |
| `"ordinary"` | `false` or `true` | any or omitted | Ordinary Kriging (drift terms ignored) | `false` |
| `"universal"` | `false` | flat list, e.g. `["sia", "z_dem"]` | Single multi-drift model applied directly | `false` |
| `"universal"` | `false` | single non-flat, e.g. `[["sia", "z_dem"]]` | Unwrapped to single model `["sia", "z_dem"]` | `false` |
| `"universal"` | `false` | multi-model suite, e.g. `[["sia", "z_dem"], ["sia"]]` | Non-interactive multi-model run with default CV exclusion buffer radius | `false` |
| `"universal"` | `true` | flat single primitive, e.g. `["sia"]` | Single model applied directly; analyzer auto-disabled | `false` (auto-disabled) |
| `"universal"` | `true` | flat list $\ge 2$ terms, e.g. `["sia", "z_dem"]` | Combinatorial candidate suite evaluated and ranked | `true` |
| `"universal"` | `true` | flat list with `[]`, e.g. `[[], "sia", "z_dem"]` | Combinatorial candidate suite + Ordinary evaluated | `true` |
| `"universal"` | `true` | explicit multi-model suite, e.g. `[["sia", "z_dem"], ["sia"], []]` | Explicit candidate suite evaluated (no combinatorial expansion) | `true` |
| `"universal"` | `true` | explicit single-model suite, e.g. `[["sia", "z_dem"]]` | Single model applied directly; analyzer auto-disabled | `false` (auto-disabled) |
| `"regression"` | `false` or `true` | flat list or omitted | Random Forest Regression Kriging (PyKrige) | `false` |

> [!NOTE]
> When `drift_analyzer: true` is configured with a single model definition (e.g. flat `["sia"]` or non-flat `[["sia", "z_dem"]]`), `PySole` logs an informational notice and automatically sets `drift_analyzer: false` across configuration dictionaries and solver states, directly executing Universal Kriging without interactive prompts or redundant candidate loops.


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
- **ΔRMSE %** = $`100 \cdot (\text{RMSE} / \text{RMSE}_{\#1} - 1)`$, relative to the best-ranked model.

Example (Wuk, pre-migration, product target `"P"`, N = 982, default suite of 24 candidates, as printed by the Drift Analyzer; rows 9-15 and 17-24 are omitted here):

```
================================================================================================================
       DRIFT ANALYZER CANDIDATE MODEL RECOMMENDATIONS (PASS 1)
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

#### 6. Detailed 24-Candidate Case Study: Understanding Multicollinearity & `PENALIZED` Models (APO Dataset)

The following complete 24-candidate ranking table was generated for the **APO** glacier dataset during post-migration (Pass 2, target $P = D \cdot \sin(\alpha_{\text{opt}})$, $N = 8{,}939$ soundings, spatial exclusion buffer $r_{\text{buffer}} = 1{,}700\text{ m}$):

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

#### Detailed Breakdown of the Ranking Dynamics

1. **Hierarchical 3-Tier Sort Key**:
   Candidates are evaluated and sorted by:
   $$\text{Sort Key} = \left(\text{penalized},\ \text{AICc},\ \text{RMSE}\right)$$
   Any candidate that is **not penalized** (`is_penalized = False`, equivalent to 0) strictly outranks **every penalized model** (`is_penalized = True`, equivalent to 1). Within each group, models are ranked by ascending AICc (favoring parsimonious models that balance residual variance against parameter count $k$), using RMSE as the final tie-breaker.

2. **Why Model #1 (`ordinary`) Wins**:
   In Pass 2, the interpolation target is the BSS-derived product $P(x,y) = D \cdot \sin(\alpha_{\text{opt}})$. Under the Shallow Ice Approximation, basal shear stress is assumed to be roughly uniform across the glacier sole, meaning $P(x,y)$ has a constant spatial mean. Ordinary Kriging captures this stationary mean directly with zero external drift parameters ($n_{\text{drift}} = 1$), achieving the best parsimony score ($\text{AICc} = 33129.6$). Adding external drift functions increases parameter complexity without providing genuine physical signal, leading to higher AICc penalties.

3. **Why Models with `WARNING` Appear (Rows 2, 8–14, 19–20)**:
   All candidates displaying `WARNING` contain the `sia` drift ($1/\sin\alpha$). Applying an SIA slope drift to a target that is *already* scaled by $\sin\alpha$ introduces an unphysical $1/\sin^2(\alpha)$ double-scaling artifact at low-gradient glacier margins. The Drift Analyzer evaluates and ranks them objectively, but alerts the user to this physical risk.

4. **The `PENALIZED` Models (Rows 21–24) and the Multicollinearity Gate**:
   - **The Apparent Paradox**: Model #21 (`sia + z_dem + curvature_dem + quadratic_xy`) actually achieves the **lowest cross-validation RMSE ($6.35\text{ m}$ vs $6.38\text{ m}$)** and **lowest raw AICc ($33053.1$ vs $33129.6$)** of all 24 candidates ($\text{dRMSE} = -0.52\%$, $\text{dAICc} = -76.5$). Despite this seemingly superior fit, it is relegated to the bottom of the table.
   - **The Common Denominator**: All four penalized models (#21, #22, #23, #24) combine **glacier surface elevation (`z_dem`)** with the **2D quadratic spatial polynomial (`quadratic_xy`: $X, Y, X^2, Y^2, XY$)**. In typical mountain glaciers and ice caps, valley elevation correlates strongly with a second-order polynomial surface ($R^2 > 0.90$), yielding $\text{VIF} > 10$.
   - **Why VIF $> 10$ is Dangerous in Universal Kriging**:
     When drift columns are highly collinear, the Universal Kriging matrix $F^T \Gamma^{-1} F$ becomes severely ill-conditioned. To minimize fitting residuals, the solver assigns enormous opposing coefficients to the redundant terms (e.g. $+50{,}000 \cdot Z_{\text{dem}}$ vs $-50{,}000 \cdot \text{quadratic}$). Near dense survey soundings, these opposing weights cancel out (giving deceptively low RMSE). However, in unmeasured sectors or near glacier margins, the terms diverge, creating unphysical runaway elevation spikes, deep craters, or numerical breakdown.
   - **The Quality Gate**: Flagging models with $\text{VIF} > 10$ as `PENALIZED` acts as an automated protective barrier that ensures ill-conditioned drift combinations are never recommended or applied in production runs.

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

### 1.10 Multi-Model Comparison Suite, Directory Architecture & Comparison Tables

When evaluating drift models, glaciologists frequently need to systematically contrast how different theoretical assumptions (e.g. baseline Ordinary Kriging vs. SIA physical drift vs. elevation hypsometry) impact the spatial geometry of the interpolated bedrock surface, ice thickness distribution, and basal shear stress fields. `PySole` provides an automated multi-model comparison workflow:

#### 1. Interactive Multi-Model Selection
In interactive terminal sessions, when presented with the Drift Analyzer candidate recommendation table, the user can select one or multiple models simultaneously:
- **Recommended Default**: Press `[Enter]` or type `#` to accept the top-ranked model (`#1`).
- **Single Model**: Enter a row number (`3`, `#3`), canonical model name (`sia + z_dem`), or `ordinary`.
- **Row Ranges**: Hyphenated intervals (e.g. `1-5` or `#1-#5`) select all candidates within the range.
- **Comma-Separated Lists**: Discrete model numbers (e.g. `1, 3` or `1, 2, 5`).
- **Mixed Combinations**: Arbitrary combinations of lists and ranges (e.g. `1-3, 5` or `3, 1-2`).
- **Deduplication & Order of Precedence**: Inputs are automatically validated and deduplicated. The first model entered becomes the **primary production model** that determines downstream migration soundings and final returns, while all remaining models are queued for comparative multi-model evaluation.

#### 2. Configuration via JSON (`pysole.json`)
Users can pre-define an explicit candidate suite in `pysole.json` under `kriging_parameters.<stage>.drift_terms` as a list of model candidate lists:
```json
{
  "kriging_parameters": {
    "pre_migration": {
      "method": "universal",
      "drift_analyzer": true,
      "drift_terms": [
        ["sia", "z_dem"],
        ["sia"],
        []
      ]
    },
    "post_migration": {
      "method": "universal",
      "drift_analyzer": true,
      "drift_terms": [
        ["sia", "z_dem", "curvature_dem"],
        ["sia", "z_dem"],
        ["z_dem"],
        []
      ]
    }
  }
}
```
When an explicit suite of candidate models is configured in JSON, the Drift Analyzer automatically evaluates all specified models and executes the multi-model comparison pipeline **non-interactively without prompt interruption**.

#### 3. Data Storage & Output Directory Architecture
To keep output repositories organized and avoid collision between competing models without deep directory nesting:

- **Diagnostic Plots (`figures/`)**:
  - Global pre-interpolation diagnostic plots (`01` through `05`: variograms, BSS optimization curves, 3D ray migration vectors) remain in root `figures/`.
  - Post-interpolation diagnostic plots (`06` through `11`: bedrock elevation map, embedded bedrock map, ice thickness map and histogram, basal shear stress map and histogram) are routed into model subdirectories:
    ```
    figures/
    ├── 01_pass1_variogram.png
    ├── 02_pass1_kc_variogram_optimization.png
    ├── 03_3d_migration.png
    ├── 04_pass2_variogram.png
    ├── 05_pass2_kc_variogram_optimization.png
    ├── sia_z_dem/
    │   ├── 06_bedrock_elevation_and_uncertainty_sia_z_dem.png
    │   ├── 07_final_embedded_bedrock_sia_z_dem.png
    │   ├── 08_final_thickness_and_uncertainty_sia_z_dem.png
    │   ├── 09_final_thickness_histogram_sia_z_dem.png
    │   ├── 10_final_basal_shear_stress_and_uncertainty_sia_z_dem.png
    │   └── 11_final_basal_shear_stress_histogram_sia_z_dem.png
    └── ordinary/
        ├── 06_bedrock_elevation_and_uncertainty_ordinary.png
        └── ...
    ```
- **Spatial Rasters & Tables (`results/` or configured `<output_dir>`)**:
  - All GeoTIFF and raster outputs append the applied drift model slug at the end of the filename (e.g. `[<output_prefix>_]bedrock_<model_slug>.tif`, `[<output_prefix>_]thickness_<model_slug>.tif`, `[<output_prefix>_]traveltime_<model_slug>.tif`, `[<output_prefix>_]basal_shear_stress_<model_slug>.tif`, and uncertainty rasters `[<output_prefix>_]*_uncertainty_<model_slug>.tif`) and are saved **directly into the defined `<output_dir>`** (flat output directory structure; no model subdirectories are generated in `<output_dir>`).
  - Cross-model comparison tables (`drift_model_comparison.txt`, `drift_model_comparison.csv`, and `drift_model_comparison.md`) are exported **directly into `<output_dir>`**.
  - Global survey observation soundings and migration coordinate tables (`[<output_prefix>_]migrated_points.csv`) remain in the root `<output_dir>` without a drift model slug.

#### 4. Cross-Model Comparison Tables
When multiple drift models are evaluated or selected for production comparison, `PySole` formats and exports comprehensive comparison files in three synchronized formats directly under `<output_dir>`:
1. `.txt`: ASCII console table (`drift_model_comparison.txt`) combining Pass 1 and Pass 2 interpolation results.
2. `.csv`: Machine-readable tabular file (`drift_model_comparison.csv`) with a unified RFC 4180 schema and a `stage` column identifying "Pass 1" vs. "Pass 2".
3. `.md`: GitHub-Flavored Markdown document (`drift_model_comparison.md`) with clean, perfectly-aligned tables and KaTeX math support.

- **Pass 1 Section**:
  Contrasts cross-validation rankings (Rank, Drift Model, CV RMSE, $\Delta$RMSE %, AICc, $\Delta$AICc, VIF, Status) with full domain traveltime distribution metrics $T(x,y)$ (Mean, Median, Std, Max in `ns` or `m`). If `pre_migration.calculate_uncertainty: true`, an additional column `Mean Uncertainty` is dynamically appended, reporting the spatial mean traveltime Kriging uncertainty $\sigma_T$ in traveltime units over all glaciated cells within the glacier outline (`outline_mask & (traveltime > 0)`).
- **Pass 2 Section**:
  Contrasts cross-validation rankings with both ice thickness distribution metrics $H(x,y)$ (Mean, Median, Std, Max in meters) and basal shear stress distribution metrics $\tau_b(x,y)$ (Mean, Median, Std, Max in kPa). If `post_migration.calculate_uncertainty: true`, an additional column `Mean Unc (m)` is dynamically appended under the Ice Thickness section, reporting the spatial mean ice thickness Kriging uncertainty $\sigma_D$ in meters over all glaciated cells within the glacier outline (`outline_mask & (thickness > 0)`).
- **Uncertainty Toggle**: When `calculate_uncertainty: false` (the default setting to avoid unnecessary $O(N^2 \cdot m)$ operations), the uncertainty column is omitted, keeping comparison tables concise and performant.
- **Graceful Error Handling**: If any candidate model fails numerical matrix inversion or regularized Cholesky decomposition, it is marked `FAILED` in the table with safe zeroed metrics, allowing the remaining candidates in the suite to complete cleanly.

---

## 2. Configuration & CLI Examples

### 2.1 JSON Configuration Schema

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

### 2.2 CLI Commands

```bash
# Execute PySole with interactive Drift Analyzer and profile grouping
pysole pysole.json --drift-analyzer --profile-col line_id

# Force non-interactive batch mode
pysole pysole.json --batch
```

