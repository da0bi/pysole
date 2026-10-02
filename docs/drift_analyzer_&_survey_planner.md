# PySole Drift Analyzer & Survey Planner Guide

`PySole` includes an automated diagnostic helper tool (**Drift Analyzer**) for selecting optimal Universal Kriging single/multi-drift models and a forward **Survey Planner** for designing GPR/Seismic campaigns on unprobed glaciers.

---

## 1. Drift Analyzer (`DriftAnalyzer`)

Universal Kriging models spatial variation as a deterministic trend surface (drift $m(x,y)$) combined with a stationary random residual field ($\epsilon(x,y)$):
$$Z(x,y) = m(x,y) + \epsilon(x,y)$$

The `DriftAnalyzer` evaluates 14 candidate single and multi-drift combinations across three diagnostic tiers:

1. **Tier 1: Linear & Non-Linear Feature Screening**:
   - **Partial Correlation Analysis**: Measures correlation between target variables and spatial covariates ($Z_{\text{dem}}$, $C_{k_c}$, $\sin(\alpha_{\text{opt}})^{-1}$) while controlling for $(X,Y)$.
   - **Random Forest Permutation Importance**: Quantifies non-linear terrain drivers (e.g., localized valley trough deepening).
   - **Variance Inflation Factor (VIF) Multicollinearity Audit**: Calculates $\text{VIF}_j = \frac{1}{1 - R_j^2}$. Combinations with $\text{VIF} > 10$ are flagged and penalized to prevent ill-conditioned Dual Kriging matrices ($K$).

2. **Tier 2: Geostatistical Spatial Cross-Validation**:
   - **Leave-One-Profile-Out (LOPO-CV)**: Activated when `inputs.survey_profile_column` is specified in configuration or CLI. Systematically omits entire profile tracks, measuring inter-profile interpolation capability.
   - **Spatial Buffer LOOCV**: Used when `survey_profile_column` is `null`. Omits all survey points within an exclusion radius $r_{\text{buffer}}$ (in meters) around each validation node, eliminating along-track autocorrelation leakage. Guided interactively by the fitted empirical variogram spatial range $a_0$.

3. **Tier 3: Glaciological Safeguard Audit**:
   - Automatically warns against applying slope-dependent drifts (`sia`) to BSS product targets ($P = D \sin \alpha$) to prevent $1/\sin^2\alpha$ margin boundary artifacts.

### Configuration Schema

Add `"survey_profile_column"` under `"inputs"` and `"drift_analyzer"` under `"pre_migration"` / `"post_migration"`:

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
    "engine": "native",
    "pre_migration": {
      "interpolation_target": "P",
      "method": "universal",
      "drift_analyzer": true,
      "drift_terms": ["sia"],
      "include_zero_boundary_condition": true
    }
  }
}
```

> [!NOTE]
> If `method` is set to `"ordinary"`, `drift_analyzer` is automatically overridden to `false` (no drift terms to evaluate).

---

## 2. Unprobed Glacier Survey Planner (`SurveyPlanner`)

When `dem_path` and `outline_path` are defined, but **no `survey_data_path`** is provided in `pysole.json`, `PySole` automatically switches to the **Survey Planner**:

1. **Synthetic SIA Model**:
   Calculates synthetic ice thickness $D_{\text{SIA}}(x,y) = \frac{\tau_0}{\rho g \sin(S(Z_{\text{dem}}))}$ masked inside the glacier outline.
2. **Track Layout Engine**:
   Lays out a longitudinal central flowline track along the maximum thickness ridge ($\sim 35\%$ of budget $L_{\text{max}}$) and orthogonal transverse cross-profiles.
3. **Outputs & Exports**:
   - **Raster**: `<output_dir>/<output_prefix>_sia_modelled_depth.<ext>`
   - **GIS Tracks**: `<output_dir>/<output_prefix>_survey_plan.gpx` & `.geojson`
   - **Map Plot**: `<plots_dir>/<output_prefix>_survey_plan_map.png`

---

## 3. Command Line Interface (CLI)

```bash
# Interactive Drift Model Recommendation
pysole pysole.json --drift-analyzer --profile-col line_id

# Force Non-Interactive Batch Mode
pysole pysole.json --batch

# Unprobed Glacier Survey Planner
pysole plan-survey --dem surface.tif --outline glacier.shp --kc 3.0 --tau 100 --max-km 12.5 --output proposed.gpx
```
