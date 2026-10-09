# Goldbergkees (GOK) Ice Thickness Peak & Drift Analysis Report

**Date**: September 27, 2026  
**Package**: PySole (Python Solver for Glacier Bedrock Topography & Ice Thickness)  
**Topic**: Root Cause Analysis and Physical Correction of Margin-Adjacent Ice Thickness Peak on Goldbergkees  

---

## 1. Executive Summary

During testing of the Goldbergkees (GOK) glacier example dataset, an unexpected maximum ice thickness of **~160 m** was observed at pixel `(132, 271)`. Crucially, this peak occurred **10 m (2 pixels)** away from the glacier boundary outline, rather than in the deep, central basin of the glacier where the greatest ice thickness was glaciologically expected.

A comprehensive mathematical and physical investigation revealed that this anomaly was caused by a **$\frac{1}{\sin^2(\alpha)}$ double-scaling artifact** occurring during Pass 2 (post-migration Kriging) when Universal Kriging with Shallow Ice Approximation (`sia`) drift was applied to the slope-scaled Basal Shear Stress (BSS) product field $P(x,y)$.

Following the correction of Pass 2 Kriging dynamics to **Ordinary Kriging** (or non-slope spatial drifts), the maximum ice thickness on Goldbergkees shifted to **110.15 m** located at pixel `(259, 219)` in the widest central basin of the glacier (**195.0 m** inside the outline).

---

## 2. Root Cause Analysis

### 2.1 Mathematical Mechanism: Double-Scaling ($\frac{1}{\sin^2 \alpha}$)

1. **Pass 2 Interpolated Quantity**:
   In PySole's 2-pass workflow, Pass 2 interpolates the slope-scaled BSS Product field $P(x, y)$:
   $$P(x, y) = D(x, y) \cdot \sin(\alpha(x, y)) = \frac{\tau_{\text{basal}}}{\rho_{\text{ice}} \cdot g}$$

2. **SIA Drift Term Definition**:
   The `sia` drift term represents the Shallow Ice Approximation thickness proxy:
   $$h_{\text{SIA}}(x, y) = \frac{\tau}{\rho_{\text{ice}} \cdot g \cdot \sin(\alpha(x, y))}$$

3. **Trend Model Fitting**:
   When Universal Kriging was configured for Pass 2 with `"drift_terms": ["sia"]`, the trend model fitted the product field $P(x,y)$ as a linear function of $h_{\text{SIA}}$:
   $$P(x, y) \approx \beta_0 + \beta_1 \cdot h_{\text{SIA}}(x,y) = \beta_0 + \beta_1 \cdot \frac{\tau}{\rho_{\text{ice}} \cdot g \cdot \sin(\alpha(x, y))}$$

4. **Thickness Reconstruction**:
   To calculate ice thickness $D(x, y)$, the solver divided the interpolated product grid $P(x, y)$ by $\sin(\alpha(x, y))$:
   $$D(x, y) = \frac{P(x, y)}{\sin(\alpha(x, y))} = \frac{\beta_0}{\sin(\alpha(x, y))} + \frac{\beta_1 \cdot \tau}{\rho_{\text{ice}} \cdot g \cdot \sin^2(\alpha(x, y))}$$

This created an unintentional **$\frac{1}{\sin^2(\alpha)}$ amplification factor** ($131.6\times$ multiplier at $5.0^\circ$ slope) for low-slope pixels.

---

### 2.2 Topographical Trigger on Goldbergkees

* **Terrain Geometry**: Pixel `(132, 271)` on Goldbergkees lies on a flat side-bench / shoulder near the glacier boundary where the DEM surface slope drops to the $5.0^\circ$ floor threshold ($\sin(5.0^\circ) = 0.08715$).
* **Artifact Generation**: Because the slope at this margin bench is flat ($5.0^\circ$), the $\frac{1}{\sin^2(5^\circ)}$ factor artificially inflated $D(x,y)$ to **160.01 m** at that margin pixel.
* **Contrast with Wurtenkees (WUK)**: On Wurtenkees, the lowest DEM surface slopes coincide with the central valley basin, so $1/\sin(\alpha)$ aligned naturally with the glacier center. On Goldbergkees, side-bench topography exposed the unphysical nature of the double-scaled drift.

---

## 3. Physical Principles & Algorithmic Solution

### 3.1 Physical Principles for BSS Product Kriging
For the BSS product field $P(x, y) = D(x, y) \cdot \sin(\alpha(x, y)) = \frac{\tau}{\rho_{\text{ice}} g}$, substituting the SIA thickness equation gives:
$$P_{\text{SIA}}(x, y) = \left( \frac{\tau}{\rho_{\text{ice}} g \sin(\alpha)} \right) \cdot \sin(\alpha) = \frac{\tau}{\rho_{\text{ice}} g} = \text{constant}$$

Thus, the SIA physical drift for the product field $P(x,y)$ is **constant**. This constant trend is inherently represented by **Ordinary Kriging**. `sia` drift ($1/\sin\alpha$) is designed for direct Depth $D(x,y)$ Kriging, not BSS Product $P(x,y)$ Kriging.

---

### 3.2 Configuration & User Control

1. **User Control in `src/pysole/solver.py`**:
   `Solver.calculate_bedrock` passes `self.post_drift_terms` directly to `kriging_interpolation` without any hardcoded programmatic overrides, leaving full control in the user's hands to configure `"ordinary"` or `"universal"` Kriging as desired.

2. **Updated Recommended Configurations**:
   Updated `DEFAULT_CONFIG` in `src/pysole/config.py`, `pysole.json`, `examples/gok/pysole_gok.json`, and `examples/wuk/pysole_wuk.json` to set `post_migration.method` to `"ordinary"` and `post_migration.drift_terms` to `[]`.

---

## 4. Quantitative Results Comparison (Goldbergkees)

| Parameter | Previous Result (Double-Scaled Drift) | Corrected Result (Ordinary Kriging) |
| :--- | :--- | :--- |
| **Max Ice Thickness** | `160.01 m` | **`110.15 m`** |
| **Max Thickness Location (Row, Col)** | `(132, 271)` (Margin Bench) | **`(259, 219)` (Central Basin)** |
| **Distance to Glacier Outline** | `10.0 m` (2 pixels from edge) | **`195.0 m` (Deep Inside Center)** |
| **Surface Elevation at Max** | `2954.00 m` | **`2845.00 m`** |
| **Glacier Bedrock Elevation Range** | `[2242.37 m, 3112.38 m]` | **`[2300.00 m, 3095.66 m]`** |

---

## 5. Verification & Test Suite Status

* **Unit Test Suite**: All 37 unit tests in `tests/` pass 100% (`37/37 OK`).
* **Example Workflows**: Both `examples/wuk/run_wuk_example.py` and `examples/gok/run_gok_example.py` run cleanly to completion.
* **Output Logging**: All output rasters, CSV point callsets, PNG diagnostic figures, and log files state their full absolute paths in the terminal and log files.
