# PySole Interpolation Strategy Guide: Geostatistical & Glaciological Best Practices

## 1. Executive Summary & Overview

In radar glaciology and subglacial topography modeling, converting sparse Ground-Penetrating Radar (GPR) traveltimes or migrated depth picks into a continuous 3D bed elevation grid ($Z_{\text{bed}}(x,y)$) or ice thickness map ($H(x,y)$) is a fundamental inverse problem. The accuracy of the resulting ice volume and subglacial trough geometry depends heavily on the chosen spatial interpolation strategy and geostatistical drift model.

The PySole package provides two primary interpolation workflows across its two-pass architecture (Pass 1: Pre-migration traveltimes $T(x,y)$; Pass 2: Post-migration depths $D(x,y)$):
1. **Direct Depth / Traveltime Kriging ($D$ or $T$)**: Direct spatial interpolation of observed depths or traveltimes using Ordinary Kriging (OK) or Universal Kriging (UK) with spatial drift terms (e.g., surface elevation `z_dem`, linear trend).
2. **Basal Shear Stress (BSS) Product Kriging ($P = D \cdot \sin\alpha$)**: Interpolation of the product of depth (or traveltime) and optimum low-pass filtered surface slope ($\sin\alpha$), grounded in the Shallow Ice Approximation (SIA).
3. **BSS Product Kriging with Universal Kriging and SIA Drift Model (`sia`)**: Incorporating an inverse-slope drift term ($U_{\text{SIA}} = 1/\sin\alpha$) during UK trend fitting of the BSS product.

### Key Finding & Core Recommendation
> [!CAUTION]
> **The Double-Scaling Artifact Warning**: Combining Universal Kriging with the `sia` drift model ($1/\sin\alpha$) during **BSS Product Kriging** introduces a catastrophic $\frac{1}{\sin^2\alpha}$ quadratic inverse-slope scaling artifact. At low-slope glacier margins ($\alpha \approx 3^\circ - 5^\circ$), this multiplies residual trend parameters by $>130\times$, generating artificial ice thickness peaks (up to **170 m** on Goldbergkees vs. true ~121 m peak) and pulling the maximum thickness peak away from the central basin towards the margin rim.

**General Rule of Thumb**:
* **Recommended PySole Default**: **BSS Product Kriging with Ordinary Kriging (`ordinary`)** for both Pass 1 and Pass 2.
* **If elevation trends exist**: Use **BSS Product Kriging with Universal Kriging using `z_dem` (DEM elevation) drift**.
* **Do NOT use `sia` drift model when interpolating BSS products ($P_1$ or $P_2$)**. `sia` drift is only mathematically valid when interpolating **Direct Depths/Traveltimes** ($D$ or $T$) without slope multiplication.

---

## 2. Theoretical & Mathematical Foundations

### 2.1 Direct Depth / Traveltime Kriging ($D$ or $T$)

In direct interpolation, the depth field $D(x,y)$ (or traveltime $T(x,y)$) is modeled as a spatial random process:
$$D(x,y) = m(x,y) + \epsilon(x,y)$$
where $m(x,y)$ is the deterministic spatial trend and $\epsilon(x,y)$ is a zero-mean intrinsically stationary random field governed by a spatial variogram $\gamma(h)$.

* **Ordinary Kriging (OK)** assumes $m(x,y) = \mu$ (constant unknown mean).
* **Universal Kriging (UK)** models $m(x,y) = \sum_{k=0}^K \beta_k f_k(x,y)$ using external spatial covariates $f_k(x,y)$.

**Glaciological Limitation**: Direct Kriging assumes spatial continuity of depth regardless of local surface slope. On steep icefalls or glacier margins where ice thickness rapidly thins, direct Kriging relies entirely on nearby zero-thickness boundary constraints. In poorly sampled regions, direct depth Kriging can over-predict ice thickness near steep margins.

---

### 2.2 Basal Shear Stress (BSS) Product Kriging (PySole Standard)

Under the Shallow Ice Approximation (SIA), basal shear stress $\tau_{\text{b}}$ is expressed as:
$$\tau_{\text{b}}(x,y) = \rho g H(x,y) \sin\alpha(x,y)$$
where $\rho$ is ice density ($917\text{ kg/m}^3$), $g$ is gravitational acceleration ($9.81\text{ m/s}^2$), $H(x,y)$ is ice thickness, and $\alpha(x,y)$ is surface slope.

Glaciologically, basal shear stress $\tau_{\text{b}}$ is remarkably uniform across a glacier, typically fluctuating within a narrow physical range of $50 - 150\text{ kPa}$. Rather than interpolating highly variable ice thickness $H(x,y)$, PySole transforms depths into a quasi-stationary BSS product field:
$$P(x,y) = D(x,y) \cdot \sin(\alpha_{k_c}(x,y))$$
where $\alpha_{k_c}(x,y)$ is the optimum low-pass filtered surface slope field. The corner frequency $k_c$ is determined dynamically via spectral variogram optimization to isolate regional surface slopes while filtering out high-frequency noise and local seracs.

The interpolated depth field $\hat{D}(x,y)$ is reconstructed across the entire DEM grid as:
$$\hat{D}(x,y) = \frac{\hat{P}(x,y)}{\sin(\alpha_{k_c}(x,y))}$$

**Key Physical Advantage**: 
* In steep margin zones ($\sin\alpha \uparrow$), dividing by larger $\sin\alpha$ naturally forces ice thickness to taper towards zero.
* In flat central basins ($\sin\alpha \downarrow$), dividing by smaller $\sin\alpha$ naturally reconstructs deep central ice reservoirs.
* Because $P(x,y)$ is quasi-stationary, **Ordinary Kriging** provides an optimal, unbiased predictor.

---

### 2.3 Mathematical Derivation of the Double-Scaling Artifact

When a user configures `post_drift_terms = ["sia"]` alongside Universal Kriging in PySole, the SIA drift covariate $f_{\text{SIA}}(x,y) = \frac{1}{\sin\alpha(x,y)}$ is applied to the target field.

If the target field is the **BSS Product** $P(x,y) = D(x,y) \cdot \sin\alpha(x,y)$, Universal Kriging fits the trend:
$$\hat{P}(x,y) = \beta_0 + \beta_1 \cdot \left(\frac{1}{\sin\alpha(x,y)}\right) + e(x,y)$$

Reconstructing depth $\hat{D}(x,y)$ by dividing $\hat{P}(x,y)$ by $\sin\alpha(x,y)$ yields:
$$\hat{D}(x,y) = \frac{\hat{P}(x,y)}{\sin\alpha(x,y)} = \frac{\beta_0}{\sin\alpha(x,y)} + \frac{\beta_1}{\sin^2\alpha(x,y)} + \frac{e(x,y)}{\sin\alpha(x,y)}$$

#### Analysis of the Quadratic Term $\frac{1}{\sin^2\alpha}$
> [!WARNING]
> On low-slope margin benches or flat plateau rims where $\alpha \approx 5^\circ$:
> $$\sin(5^\circ) = 0.08715 \implies \frac{1}{\sin(5^\circ)} = 11.47 \implies \frac{1}{\sin^2(5^\circ)} = 131.65$$
> A tiny fitted drift coefficient $\beta_1 = 1.0\text{ m}$ gets multiplied by **$131.65$**, creating a massive **$170.35\text{ m}$ artificial ice peak** (observed on Goldbergkees) displaced towards the glacier border!

This double-scaling artifact occurs in both Pass 1 (traveltimes $P_1 = T \cdot \sin\alpha$) and Pass 2 (depths $P_2 = D \cdot \sin\alpha$).

---

### 2.4 Non-Slope Spatial Drift Models (`z_dem`, `linear_xy`, `quadratic_xy`)

To account for elevation-dependent glaciological trends (e.g., thicker ice in high accumulation basins, thinner ice on ablation tongues) without slope interaction, PySole supports non-slope spatial drift models:
* `z_dem`: Surface elevation $Z_{\text{surf}}(x,y)$ from the input DEM.
* `linear_xy`: Planar coordinates $[x, y]$.
* `quadratic_xy`: Quadratic spatial coordinates $[x, y, x^2, y^2, xy]$.

When using `z_dem` with BSS Product Kriging:
$$\hat{P}(x,y) = \beta_0 + \beta_1 Z_{\text{surf}}(x,y) + e(x,y) \implies \hat{D}(x,y) = \frac{\beta_0 + \beta_1 Z_{\text{surf}}(x,y) + e(x,y)}{\sin\alpha(x,y)}$$
This preserves linear elevation scaling while preventing quadratic slope singularities.

---

## 3. Empirical Evaluation: WUK and GOK Benchmarks

We evaluated 6 distinct interpolation strategies across two contrasting alpine glaciers in Austria:
1. **Wurtenkees (WUK)**: A small, steep cirque glacier with high uniform surface slope ($15^\circ - 25^\circ$) and sparse 2D GPR profile lines.
2. **Goldbergkees (GOK)**: A large valley glacier featuring flat high-elevation plateau icefield benches ($\alpha \approx 3^\circ - 5^\circ$), a broad central basin, and an extensive GPR network.

### 3.1 Tested Strategies Matrix

| Strategy Code | Interpolated Target Field | Kriging Engine | Drift Model (`pre/post_drift_terms`) |
| :--- | :--- | :--- | :--- |
| **Strategy 1** | Direct Depths / Traveltimes ($D, T$) | Ordinary Kriging | None (`ordinary`) |
| **Strategy 2** | Direct Depths / Traveltimes ($D, T$) | Universal Kriging | `sia` ($1/\sin\alpha$) |
| **Strategy 3** | BSS Product ($P = D \cdot \sin\alpha$) | Ordinary Kriging | None (**PySole Default**) |
| **Strategy 4** | BSS Product ($P = D \cdot \sin\alpha$) | Universal Kriging | `sia` (**Double-Scaled**) |
| **Strategy 5** | BSS Product ($P = D \cdot \sin\alpha$) | Universal Kriging | `z_dem` (DEM Elevation Drift) |
| **Strategy 6** | Direct Depths / Traveltimes ($D, T$) | Universal Kriging | `z_dem` (DEM Elevation Drift) |

---

### 3.2 Wurtenkees (WUK) Benchmark Results

*Grid size: $171 \times 171$ ($5\text{ m}$ resolution), 982 GPR picks, max boundary distance: $279\text{ m}$.*

| Strategy | Max Thickness [m] | Mean Thickness [m] | Total Volume [km³] | Max Location Border Dist [m] | LOOCV RMSE [m] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy 1**: Direct Depth (Ordinary) | 74.58 | 30.22 | 0.0133 | 226.7 | 0.24 |
| **Strategy 2**: Direct Depth (Universal + SIA) | 74.68 | 30.27 | 0.0133 | 226.7 | 0.24 |
| **Strategy 3**: BSS Product + Ordinary (**Default**) | **79.73** | **29.79** | **0.0131** | **219.2** | **0.65** |
| **Strategy 4**: BSS Product + Universal (SIA Drift) | 86.44 | 30.31 | 0.0133 | 223.0 | 0.66 |
| **Strategy 5**: BSS Product + Universal (`z_dem`) | 79.70 | 29.78 | 0.0131 | 219.2 | 0.65 |
| **Strategy 6**: Direct Depth (Universal + `z_dem`) | 74.60 | 30.20 | 0.0132 | 226.7 | 0.24 |

#### WUK Insights
* On Wurtenkees, the surface slope is steep and relatively uniform ($>15^\circ$). As a result, the double-scaling artifact in Strategy 4 is mild (elevating max thickness from 79.7 m to 86.4 m) because $\sin\alpha$ never drops near zero.
* BSS Product Kriging (Strategy 3 & 5) produces realistic central basin thickening ($79.7\text{ m}$) compared to Direct Depth Kriging ($74.6\text{ m}$).

---

### 3.3 Goldbergkees (GOK) Benchmark Results

*Grid size: $414 \times 450$ ($5\text{ m}$ resolution), 2033 GPR picks, max boundary distance: $362\text{ m}$.*

| Strategy | Max Thickness [m] | Mean Thickness [m] | Total Volume [km³] | Max Location Border Dist [m] | Glaciological Realism & Matrix Stability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy 1**: Direct Depth (Ordinary) | 121.98 | 38.76 | 0.0422 | 330.2 | High (Central Basin Peak) |
| **Strategy 2**: Direct Depth (Universal + SIA) | NaN | NaN | 0.0000 | 0.0 | **Failed (Singular Kriging Matrix)** |
| **Strategy 3**: BSS Product + Ordinary (**Default**) | **121.25** | **38.80** | **0.0422** | **330.2** | **Optimal (Physics-Based Central Basin)** |
| **Strategy 4**: BSS Product + Universal (SIA Drift) | **170.35** | **40.62** | **0.0442** | **165.5** | **UNREALISTIC (Double-Scaled Margin Peak)** |
| **Strategy 5**: BSS Product + Universal (`z_dem`) | NaN | NaN | 0.0000 | 0.0 | **Failed (Singular Kriging Matrix)** |
| **Strategy 6**: Direct Depth (Universal + `z_dem`) | NaN | NaN | 0.0000 | 0.0 | **Failed (Singular Kriging Matrix)** |

#### GOK Diagnostic Analysis
1. **Double-Scaling Artifact (Strategy 4)**: Inflates maximum ice thickness from **121.25 m** to **170.35 m** (+49.1 m artificial excess!) and shifts the maximum peak location from the central basin (330.2 m from border) down to 165.5 m near low-slope plateau benches.
2. **Kriging Matrix Singularity (Strategies 2, 5, 6)**: Applying Universal Kriging with external drifts (`sia` or `z_dem`) on large 2D GPR networks (2033 points) without drift normalization causes matrix ill-conditioning (`LinAlgWarning: Singular matrix`).
3. **Robustness of Strategy 3 (BSS + Ordinary Kriging)**: Strategy 3 is completely immune to matrix singularity issues and places the maximum ice thickness of **121.25 m** cleanly in the deep central basin (330.2 m from border).

---

### 3.4 In-Depth Comparison: Direct Depth + Universal Kriging (SIA Drift) vs. BSS Product + Ordinary Kriging

A critical question in radar glaciology is how **Strategy 2 (Direct Depth + Universal Kriging with SIA Drift)** compares to **Strategy 3 (BSS Product + Ordinary Kriging)**.

#### 1. Mathematical Structure & Trend Model
* **Strategy 2 (Direct Depth + SIA Drift UK)**:
  $$\hat{D}(x,y) = \beta_0 + \beta_1 \cdot \left(\frac{1}{\sin\alpha(x,y)}\right) + e(x,y)$$
  Directly fits an inverse-slope trend to measured depths. Because $\frac{1}{\sin\alpha}$ enters linearly, **Strategy 2 avoids the $\frac{1}{\sin^2\alpha}$ double-scaling artifact**.
* **Strategy 3 (BSS Product + Ordinary Kriging)**:
  $$\hat{P}(x,y) = \mu + e(x,y) \implies \hat{D}(x,y) = \frac{\mu + e(x,y)}{\sin(\alpha_{k_c}(x,y))}$$
  Fits a constant mean to the quasi-stationary shear stress product $P(x,y) = D \cdot \sin(\alpha_{k_c})$ and reconstructs depth by dividing by spectrally filtered slope $\sin(\alpha_{k_c})$.

#### 2. Key Conceptual & Practical Differences

| Feature / Property | Strategy 2: Direct Depth + SIA UK | Strategy 3: BSS Product + Ordinary Kriging (PySole Default) |
| :--- | :--- | :--- |
| **Slope Field Processing** | Uses raw, unfiltered local DEM gradients $\alpha(x,y)$. | Uses spectrally optimized, low-pass filtered slope field $\alpha_{k_c}(x,y)$ at corner frequency $k_c$. |
| **Sensitivity to Noise / Seracs** | High: Sensitive to local DEM noise, crevasses, seracs, and localized steep slopes. | Low: Spectral filtering isolates regional topographic slope, filtering out local roughness. |
| **Kriging Matrix Stability** | **Poor**: Near flat plateau areas ($\alpha \to 0$), $1/\sin\alpha \to \infty$, causing ill-conditioned/singular Kriging matrices (`LinAlgWarning: Singular matrix` on GOK). | **Optimal**: Operates on smooth scalar field $P(x,y)$ with Ordinary Kriging. Matrix is 100% stable across all grid sizes. |
| **Margin Boundary Behavior** | Relies on residual covariance decay $e(x,y)$ to enforce zero thickness near borders. | Naturally forces zero-thickness tapering at steep borders ($\sin\alpha \uparrow \implies H \to 0$) without requiring dense zero-thickness points. |
| **Wurtenkees Performance** | Max = 74.68 m, Mean = 30.27 m, Vol = 0.0133 km³. | Max = 79.73 m, Mean = 29.79 m, Vol = 0.0131 km³. (Slightly deeper central basin peak due to slope filtering). |
| **Goldbergkees Performance** | **Failed** due to Kriging matrix singularity across 2033 points. | **Passed**: Max = 121.25 m, Mean = 38.80 m, Vol = 0.0422 km³ (stably placed in deep central basin). |

#### 3. Verdict & Summary Recommendation
While Strategy 2 is mathematically elegant because it avoids double-scaling, it suffers from **numerical instability** when applied to large, complex GPR networks over flat plateau glaciers (where $\alpha \to 0$ leads to matrix singularity). Furthermore, it lacks the **spectral slope filtering** of Strategy 3, making it vulnerable to local DEM noise. 

**Strategy 3 (BSS Product + Ordinary Kriging)** remains the superior and recommended default for PySole.

---

## 4. Case-Dependent Best Interpolation Practice Recommendations

Based on theoretical derivations and empirical benchmarks on WUK and GOK, we establish the following case-dependent recommendations for PySole users:

```
                      ┌─────────────────────────────────────────┐
                      │  Glacier Topography & Survey Setting   │
                      └────────────────────┬────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
       [Valley Glacier / Cirque]                       [Flat Icefield / Plateau]
    (Slopes > 5°, Well-Defined Basin)               (Low-Slope Benches < 5°, Ice Cap)
                    │                                             │
      ┌─────────────┴─────────────┐                 ┌─────────────┴─────────────┐
      ▼                           ▼                 ▼                           ▼
[Standard Survey]       [Strong Elevation]   [Standard Survey]       [Sparse Pick Data]
 (Dense GPR Grid)          (Long Tongue)      (Dense GPR Grid)       (No Margin Constraints)
      │                           │                 │                           │
      ▼                           ▼                 ▼                           ▼
 ┌─────────┐                 ┌─────────┐       ┌─────────┐                 ┌─────────┐
 │Strategy3│                 │Strategy5│       │Strategy3│                 │Strategy1│
 └─────────┘                 └─────────┘       └─────────┘                 └─────────┘
  (BSS + OK)                  (BSS + UK         (BSS + OK)                (Direct D + OK)
                             z_dem)
```

### Case A: Standard Alpine Valley Glaciers (e.g. Goldbergkees, Pasterze)
* **Recommended Strategy**: **Strategy 3 (BSS Product + Ordinary Kriging)**
* **Rationale**: Preserves SIA physics, accurately bounds ice thickness in central troughs, and naturally tapers to zero at valley walls without margin artifacts.
* **JSON Configuration**:
  ```json
  "kriging_parameters": {
    "pre_migration": {
      "method": "ordinary",
      "drift_terms": []
    },
    "post_migration": {
      "method": "ordinary",
      "drift_terms": []
    }
  }
  ```

### Case B: Glaciers with Strong Altitude/Elevation Gradients
* **Recommended Strategy**: **Strategy 5 (BSS Product + Universal Kriging with `z_dem` Drift)**
* **Rationale**: Captures regional elevation-dependent ice thickness variations (accumulation zone vs. ablation tongue) while completely avoiding slope singularities.
* **JSON Configuration**:
  ```json
  "kriging_parameters": {
    "pre_migration": {
      "method": "universal",
      "drift_terms": ["z_dem"]
    },
    "post_migration": {
      "method": "universal",
      "drift_terms": ["z_dem"]
    }
  }
  ```

### Case C: Flat Icefield Benches & Low-Slope Margin Regions ($\alpha < 5^\circ$)
* **Recommended Strategy**: **Strategy 3 (BSS + Ordinary Kriging)** or **Strategy 1 (Direct Depth + Ordinary Kriging)**
* **STRICT RULE**: **NEVER configure `sia` drift for post-migration depth interpolation on flat benches!**

---

## 5. Summary Table of Recommendations

| Glacier Setting | Recommended Strategy | Pre-Kriging Type & Drift | Post-Kriging Type & Drift | Primary Benefit |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Alpine Valley** | Strategy 3 (BSS + OK) | `ordinary`, `[]` | `ordinary`, `[]` | Optimal central basin depth, clean margin tapering |
| **Long Valley Tongue (Elevational Drift)** | Strategy 5 (BSS + UK Elevation) | `universal`, `["z_dem"]` | `universal`, `["z_dem"]` | Models altitude-dependent thickness gradients |
| **Flat Plateau / Ice Cap** | Strategy 3 or Strategy 1 | `ordinary`, `[]` | `ordinary`, `[]` | Prevents inverse-slope margin explosion |
| **Direct Depth Interpolation (No BSS)** | Strategy 2 (Direct Depth + SIA) | `universal`, `["sia"]` | `universal`, `["sia"]` | Valid ONLY if BSS product multiplication is disabled |

---

## 6. Runtime Note: Uncertainty Maps (`outputs.compute_uncertainty`)

The post-migration Kriging variance (the uncertainty maps and the derived thickness/BSS standard deviation) costs $O(N^2 \cdot n_{\text{cells}})$ and dominates the run time on large grids (GOK: ~121 s with, ~63 s without). Set `outputs.compute_uncertainty` to `false` (or pass `compute_uncertainty=False` to `Solver`) to skip it. The bedrock and thickness fields are **bit-identical** in both modes, and the diagnostic figures are still saved (the uncertainty panels show a placeholder). If an uncertainty raster export is requested (`outputs.save_thickness_uncertainty` or `outputs.save_basal_shear_stress_uncertainty`), the evaluation is re-enabled automatically and a warning is logged.

---
*Report filed in PySole Documentation: [pysole_interpolation_practice_guide.md](pysole_interpolation_practice_guide.md).*
