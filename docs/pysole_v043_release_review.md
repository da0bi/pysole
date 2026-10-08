# PySole post-v0.4.2 Review — Release Readiness for v0.4.3 (read-only)

Scope: `src/pysole/` (14 modules, ~7,000 lines), `tests/`, `README.md`, `pysole.json`, `examples/`, `docs/`.
Baselines: `claude_codebase_audit.md` (A1) → `pysole_v042_code_review.md` (A2) → `pysole_v042_review_pass3.md` (A3).

**How this was verified**
- Code was read line by line for the changed modules. The docs/schema parity audit was delegated to a read-only sub-agent. I re-checked each of its High/Medium claims against the source.
- **[V]** marks findings I reproduced numerically with stubbed harnesses in `/tmp/pysole_review2/`.
- Unmarked findings come from code reading only.
- Available libraries: numpy 2.2.4, scipy 1.15.3, matplotlib and pyproj.
- **The repo test suite cannot run in this sandbox.** `MPLBACKEND=Agg python -m unittest discover -s tests` fails with 16/16 import errors (`ModuleNotFoundError: shapely`). The same happens with `PYTHONPATH=src`. Pass/fail of the 83 test functions is therefore **unverified**.
- shapely, pandas, sklearn, rasterio and pytest are missing, and there is no network.
- Nothing in the repo was modified. No `__pycache__`, `.pyc` or other file was left in the repo. No background process remains.

---

## 1. Executive Summary & Release Readiness Verdict

### Verdict: **NO-GO for v0.4.3 as it stands.** It becomes **GO after six small, well-bounded fixes** (section 6).

**What is now sound.** Several mathematically important items have been fixed and verified.

| Area | Result |
|---|---|
| 3D Eikonal migration sign (A3 N3-H1) | ✅ [V] Matches the exact plane-over-plane solution to 0.01–0.13 m in 6 oblique cases (sloped surface plus sloped bed, non-constant T). |
| FFT padding vs kernel size (A2 N-M3) | ✅ [V] on the optimiser path. Error vs `gaussian_filter(mode='mirror')` at σ_px = 16/32/64 is 0.003/0.003/0.118 m with `kc`-aware padding. It was 0.65/12.0/23.4 m. |
| Dual kriging scale invariance (A3 N3-M10) | ✅ [V] Sill ×4, ×400 and ×4·10⁴ change the grid by at most 1.1·10⁻¹³ m. |
| Log-uniform k_c sweep (A3 N3-M2) | ✅ [V] 16 of 50 samples fall in λ ∈ [300, 3000] m. It was 3 of 50. |
| `kc ≤ 0`, `λ ≤ 0`, `kc_min ≥ kc_max` | ✅ [V] Now raise explicit `ValueError`s. |
| End-to-end synthetic run (100×100 DEM, 140 picks, `is_batch=True`) | ✅ [V] Runs in 7 s. Thickness RMSE is 7.3 m on a 83 m mean (bias +3.5 m) with v = 0.17 vs true 0.168. |

**What blocks release.**
1. **[High] `run_from_config` / CLI never writes the final bedrock raster.** [V] The main product is silently missing. README, CHANGELOG and examples say otherwise.
2. **[Medium] The pipeline output directory is still resolved wrongly** (`pipeline.py` L87), although the CHANGELOG says it is fixed. [V] This leaves a stray `pysole.log` in the working directory.
3. **[Medium] Documented options are silently non-functional or misleading.**
   - `kriging_parameters.engine: "pykrige"` is ignored.
   - CLI `--drift-analyzer` is a no-op.
   - Leave-One-Profile-Out CV (LOPO) is still dead.
   - CLI `--verbose` is overridden by the config.
   - `variogram_model: "linear"` is documented but rejected.
4. **[Medium] Release materials are partly untrue.** `examples/pysole_quickstart.ipynb` raises `TypeError`. Two docs use drift names that now raise `ValueError`. Several CHANGELOG "fixed" claims are false (see section 2.5).
5. **[Gate] The tests could not be run here.** They must be run in a complete environment before tagging. No test covers items 1–3.

**Everything else is non-blocking** (section 5, Low items, and the streamlining list). It includes the Tikhonov sign convention, the explicit `K_inv`, O(n_kc·M·N) sweep memory and duplicate exports.

---

## 2. Audit Resolution Matrix

✅ resolved · 🟡 partial · ❌ open · ⬜ not rechecked. **[V]** = reproduced.

### 2.1 Audit A1 (`claude_codebase_audit.md`)

| ID | Finding | Status | Evidence now |
|---|---|---|---|
| H1 | `coords_to_grid_indices` Y-mirrored | ✅ | Tests exist in `test_audit_regressions.py` (not run here). |
| H2 | Drift re-normalised per call | ✅ | `DriftBasis` fixes mean/scale once (`interpolation.py` L881-972). |
| H3 | Outline Y-flip | ✅ | Unchanged since A2. |
| H4 | NaN-ring outlines | ✅ | `load_outline` now raises on missing/unparseable `outline_path` (`raster.py` L729, L764, L818). |
| H5 | SurveyPlanner mirrored / budget | 🟡 | Straight segments across gaps and straight flowline (D6) unchanged. |
| H6 | Variogram heuristic | 🟡 | Fitted params now flow `optimize_bss` → `kriging_interpolation` (`solver.py` L945, L716). Caveats: fitted on the least-smoothed kc, always "spherical", P-units sill reused for T/D targets. |
| H7 | Product slope clamp asymmetric | ✅ | Same `slope_floor_deg` at `solver.py` L618 and L731. |
| H8 | GPX/GeoJSON projected coords | 🟡 | `pyproj` reprojection works; on failure it silently exports projected values (`survey_planner.py` L36-43, = A2 N-L9). |
| M1 | `kc` not in rad/m | ✅ | `fftfreq·2π` (`smoothing.py` L112-113). |
| M2 | FFT boundary / NaN | ✅ [V] | Normalised convolution matches reference to 0.000 m; NaNs restored (`smoothing.py` L170-182). Padding: see A2 N-M3. |
| M3 | `fit_variogram_model` | ✅ | Scale-invariant bounds; `counts` weighting is now live (`variogram.py` L580, L612). |
| M4 | O(N²) variogram memory | 🟡 | Dense `pdist` only for N ≤ 5000. The N > 5000 branch is uncapped in bins and loops in Python (`variogram.py` L158-196). |
| M5 | Dual-kriging cost / regularisation | 🟡 | LU factorisation ✅. Regularisation limited to `K[:N,:N]` ✅ (`interpolation.py` L366-367). Scale invariance ✅ [V]. **Still open:** an explicit `K_inv` is formed (L376). The variance is always computed, at O(N²) per cell. The `DualKrigingSolver` uses an absolute 1e-8 (L1066). Its `success=False` is silent. The Tikhonov sign is inverted for the γ-form (section 3.2). |
| M6 | Curvature recomputed per call | ✅ | Cached. |
| M7 | Analyzer ≠ production drifts | ❌ | Raw vs smoothed curvature, nearest-cell vs bilinear sampling, `slope_floor_deg` not forwarded (`solver.py` L672-678). Details in A2 N-M10. |
| M8 | AICc comparability | ✅ | Common-intersection mask. |
| M9 | MDI vs permutation importance | ✅ | Not rechecked. |
| M10 | Migration edge cases | 🟡 | `v` vs `1/v` now consistent. Still open: `valid_d = d_mig > 0` conflates NaN and 0 (`migration.py` L205). The `max()>15` unit heuristic remains (`solver.py` L805). |
| M11 | `blend_margin_topography` | ✅ | σ is now an anisotropic tuple in pixels (`interpolation.py` L171). |
| M12 | Zero-boundary points | ❌ | Fixed ~100 points, no de-duplication (`interpolation.py` L628). |
| M13 | I/O round trip | ✅ | `xllcenter` ✅. `.asc` with dx ≠ dy now falls back to GeoTIFF with a warning (`raster.py` L206-237). |
| M14 | Input validation | ✅ | `method`, `engine`, `variogram_model` validated (`interpolation.py` L655-672). But "linear" is rejected while documented (README L259/L266, `solver.py` L83/L91). |
| Low: wasted `compute_gradients` | ✅ | `compute_slope_rad` in the BSS loop. |
| Low: silent `except` | 🟡 | `drift_analyzer.py` L252/L274 now log at DEBUG. Interactive loops still `except Exception: break/pass` (`solver.py` L800-801, L869-870). |
| Low: duplicated variogram code | 🟡 | `DualKrigingSolver` still has two inline copies (`interpolation.py` L1042-1052, L1102-1112). |
| Low: docstring mismatches | 🟡 | `warn_low_pairs` fixed. `calculate_variogram` bin formula docs are wrong (section 3.4). |
| Low: double `[INFO]` prefix | ❌ | e.g. `pipeline.py` L95, `solver.py` L663. |
| Low: unused imports | 🟡 | Fixed: `LinAlgError`, `get_drift_functions`, `fft_gaussian_smooth`. Remaining (AST scan): `drift_analyzer.logging`, `migration.RegularGridInterpolator`, `pipeline.{dataclass, resolve_input_path, OutputsConfig, save_points_csv}`, `smoothing.Any`, `solver.{MigrationResult, OptimizationResult}`, `survey_planner.{logging, GridGeometry}`, `variogram.compute_gradients`. |
| D1 `DriftBasis` | ✅ | |
| D2 `VariogramParams` dataclass | ❌ | Still a dict. |
| D3 orientation API | 🟡 | |
| D4 wavelength cutoff | ✅ | Dual `kc`/`λ` API; padding scaled to σ on the optimiser path. |
| D5 robust variogram | ❌ | |
| D6 flowline planner | 🟡 | |
| D7 failure counters | 🟡 | Debug logs only. |
| 7 regression tests | 🟡 | Several now exist; most are weak (section 5). |

### 2.2 Audit A2 (`pysole_v042_code_review.md`)

| ID | Status | Evidence now |
|---|---|---|
| N-H1 `sys`/`logger` imports | ✅ | `pipeline.py` L9, `config.py` L12. |
| N-H2 config/CLI plumbing | 🟡 | `interactive_*` flags and `finalization_parameters` are forwarded (`pipeline.py` L126-146). **Open:** `Solver.run_pipeline` is still a diverging second pipeline (section 3.5). |
| N-H3 FFT normalised convolution | ✅ [V] | See A1 M2. |
| N-H4 quadratic drift conditioning | ✅ | |
| N-H5 `config_path`, LOPO, profile column | ❌ | `config`/`config_path` are preserved ✅ (`solver.py` L117, L195). **Still open:** `from_config` overwrites `config_path` with `str(dict)` for dict configs (L576). LOPO is still dead (see N-NEW-M3). |
| N-H6 compound drifts | ✅ | The expanded primitives now reach the engine (`interpolation.py` L759). Unknown terms raise (L698). Missing grids / empty `drift_terms` raise (L722-737). |
| N-H7 default `kc = 0.5` | ✅ | Now 0.0314 (`solver.py` L1355, `survey_planner.py` L84, `config.py` L410). |
| N-M1 `resample_dem` NaN | ✅ | |
| N-M2 stale outline transform | ✅ | |
| N-M3 FFT padding | 🟡 | Optimiser path ✅ [V] (`variogram.py` L464). **Residual:** `Solver._get_fft_dem_grids` still calls `precompute_fft_grid(dem, dx, dy)` without `kc` (`solver.py` L459). It feeds `get_smoothed_dem` / `get_smoothed_curvature` at the optimal kc. |
| N-M4 sweep robustness | ✅ | `counts` weighting live. `kc_min ≥ kc_max` raises. The first kc is still evaluated twice (minor). |
| N-M5 sweep memory | ❌ | All smoothed DEMs and slopes are kept (`variogram.py` L660-661, plus the `executor.map` result list). They are pruned only afterwards (L727-731). |
| N-M6 bin-count formula | ❌ [V] | Docs, log and code disagree (section 3.4). |
| N-M7 `calculate_bedrock` / `finalize_bedrock` | 🟡 | Clip counts inside the mask ✅ (`solver.py` L985-987). `interactive=False` default ✅. Still open: double docstring (L1093-1111). `kriged_std` does not describe the smoothed/RF/blended final field. |
| N-M8 RF holes | 🟡 | Non-negative clamp ✅. Subsample at N>20 000 ✅. Fake progress bar remains. `np.random.seed(42)` mutates global RNG state (`interpolation.py` L825). |
| N-M9 Eikonal geometry | ✅ [V] | Section 3.3. |
| N-M10 analyzer vs production | ❌ | Raw vs smoothed curvature; the literal `"curvature_dem"` is the only name that triggers smoothed injection (`solver.py` L707). Compound names use raw curvature. `slope_floor_deg` is not forwarded. |
| N-M11 config / exports | 🟡 | Deep merge ✅ (`config.py` L225-232). `load_config` still has side effects (dirs, logging, L291-294). Output-dir plumbing still wrong (N-NEW-M2). |
| N-M12 plotting | 🟡 | `_get_cell_edge_extent` ✅. `_safe_interactive_pause` ✅. The backend is still switched at import time (guarded by DISPLAY/backend checks, `plotting.py` L12-16). |
| N-L2 `getattr(pre_kriging_points)` | ✅ | Explicit attributes. |
| N-L3 `get_drift_functions` | ✅ | Removed. |
| N-L4 O(N·U) dedup | ❌ | `raster.py` L901-905. It also `nanmean`s the profile-ID column. |
| N-L5 empty survey fallback | ✅ | Raises. |
| N-L8 outline fallback | ✅ | Raises. |
| N-L9 WGS84 fallback | ❌ | |
| N-L10 stencil spacing | ⬜ | |
| N-L11 `compute_cutoff_wavelength(kc)` | ✅ | |

### 2.3 Audit A3 (`pysole_v042_review_pass3.md`)

| ID | Status | Evidence now |
|---|---|---|
| N3-H1 migration sign | ✅ [V] | Section 3.3. |
| N3-H2 drift expansion in engine | ✅ | `interpolation.py` L759. |
| N3-M1 `fft_filter_metric` | ✅ | Precedence implemented and logged (`variogram.py` L418-427). Table in section 3.1. Still undocumented in README (README L246/L359). |
| N3-M2 linear k spacing | ✅ [V] | `np.geomspace` (`variogram.py` L564). |
| N3-M3 output dir / log | ❌ [V] | CHANGELOG claims it is fixed. `pipeline.py` L87 still passes the config file as the 2nd **positional** argument, which is `survey_data_path`. |
| N3-M4 `kc ≤ 0` | ✅ | `smoothing.py` L155-159. `kc=0` returns the DC component. |
| N3-M5 interactive flags | 🟡 | Plumbed (`pipeline.py` L130-131). Schema default is False. Scope narrower than documented (N-NEW-M6). |
| N3-M6 `n_steps` | 🟡 | Clamped to [3, 50] silently. README documents only [10, 50]. |
| N3-M7 λ-bound validation | ✅ [V] | `λ ≤ 0` and `kc_min ≥ kc_max` raise. `λ_min < λ_Nyquist` warns. |
| N3-M8 Half-Domain kc_min | ✅ | Math correct; padding fixed on the optimiser path; residual at `solver.py` L459. |
| N3-M9 fitted-variogram reuse | 🟡 | Counts live. `range_fix` still from the first kc and not re-fit after interactive range changes. |
| N3-M10 sill dependence | ✅ [V] | 1.1·10⁻¹³ m. |
| N3-L1 doc mismatches | 🟡 | `warn_low_pairs` ✅. Bin formula ❌. CSV layout doc ❌. |
| N3-L2 test misnomer | ✅ | Renamed `test_run_from_config_interactive_mode`. |
| N3-L3 `plt.pause` on Agg | ✅ | `_safe_interactive_pause`. |
| N3-L4 `final_final_*` | ✅ [V] | Outputs are now `res_thickness.npy`, etc. |
| N3-L5 `.asc` dy | ✅ | GeoTIFF fallback. |

### 2.4 Streamlining tasks promised by the CHANGELOG ("Phase 8")

| Task | Status |
|---|---|
| Export consolidation (`PipelineExporter` delegates to `Solver.export_outputs`) | 🟡 It is a thin shim (`pipeline.py` L18-37). Bedrock export was **lost** in the move (N-NEW-H1). Duplicate writes remain (section 3.5). |
| One pipeline | ❌ `pipeline.run_from_config` and `Solver.run_pipeline` differ in defaults, bedrock saving, and batch semantics. |
| Pure `load_config` | ❌ Side effects remain. |
| Single schema source | ❌ Defaults repeated in `config.py`, `pipeline.py` (`.get(..., True)` / `.get(..., False)`) and `Solver.run_pipeline`. |

### 2.5 CHANGELOG claims that do not match the code

| CHANGELOG | Reality |
|---|---|
| L15 "default bedrock export `<prefix>_bedrock_elevation_map.tif`" | The string does not exist in `src/`. `save_bedrock_elevation_map` (`config.py` L28) is never read. |
| L21 "CLI `--drift-analyzer` plumbed" | `solver.drift_analyzer` is set (`pipeline.py` L121-122) but never read. The per-stage JSON flag is what counts (`solver.py` L659). |
| L20 "string/integer profile IDs retained" | `pd.to_numeric(errors="coerce")` turns strings into NaN (`raster.py` L843). |
| L43 "`slope_floor_deg` plumbed into DriftAnalyzer" | The `__init__` argument exists. `Solver` never passes it (`solver.py` L672-678). |
| L47 "safe **lazy** backend selection" | Still import-time `matplotlib.use("Agg")` under DISPLAY/backend guards. |
| L17 "SurveyPlanner native resolution" | True for path DEMs only. An ndarray DEM still falls back to 10 m (`survey_planner.py` L67-68). |
| L31 "σ_px = 1/(√2·π·k_c·Δx)" | The code uses σ_px = 1/(k_c·Δx) (`smoothing.py` L98-99). The code is right (angular k); the CHANGELOG formula is wrong. |
| L92 "output dir plumbing fixed" | Not fixed [V]. |
| L35 "relative Tikhonov ε·mean(diag K)" | `mean(diag(K_sample))` is identically 0 (γ(0)=0), so ε = 10⁻⁶·sill (`interpolation.py` L366). |
| L129-L135 vs L155-158 in 0.4.2 | Same release purges and re-introduces `lambda_min/max`. Two "Phase 7" headings. |

---

## 3. Mathematical & Feature Integrity Review

### 3.1 Fourier frequency sweep (`variogram.py` L390-565, `smoothing.py`)

**Filter definition.** `H(k) = exp(−k²/(2k_c²))` with angular `k` [rad/m].
- Spatial Gaussian σ = 1/k_c, so σ_px = 1/(k_c·Δ).
- λ_c = 2π/k_c is a *corner* wavelength: gain 0.607 at k = k_c, −3 dB at k = 0.833·k_c (λ = 1.2·λ_c).
- The docs call it a "cutoff". This is acceptable if stated once as a corner definition.
- `k_c,max` default = k_Nyquist = π/Δ → σ = 0.32·Δ, effectively an identity filter. That is a sound upper end.

**Half-Domain `k_c,min = 4π/L_max`** (λ_max = L/2).
- σ_max = L/(4π) = 0.080·L, so σ_px,max = N/(4π).
- Reflect-padding with 4σ gives 0.32·N per side. `pad` is clipped to `[32, N]` (`smoothing.py` L100-101). Valid.
- **Edge case:** for a long, narrow domain L_max = max(extent) makes k_c,min too small for the short side. A reasonable alternative is min(extent).

**Precedence of `kc_*` vs `lambda_*`** [V]:

| Input | Resulting search range |
|---|---|
| defaults | kc_max = π/Δ = 0.3142, kc_min = 4π/L = 0.00628, n = 50 |
| `wavenumber`, `kc_max=0.05`, `lambda_min=200` | kc_max = **0.0500** (kc wins), n = 13 |
| `wavelength`, `kc_max=0.05`, `lambda_min=200` | kc_max = **0.0314** (λ wins), n = 10 |
| `wavelength`, λ ∈ [100, 1500] | kc ∈ [0.00419, 0.0628], n = 18 |
| `wavelength`, λ null | falls back to `kc_*` (same grid as defaults) |
| `lambda_min = 5` (< 2Δ) | clamped to k_Nyquist, WARNING logged |
| `n_steps = 1` / `200` | silently 3 / 50 |
| `kc_min=0` / `λ ≤ 0` / `kc_min ≥ kc_max` | `ValueError` |

The `lambda_min`/`lambda_max` parameters are retained as requested. The design is coherent. What is missing:
- The README does not state the precedence (README L246-248, L359-367).
- No log appears when only one of the pairs is ignored.
- The `λ_min < λ_Nyquist` warning (`variogram.py` L429) fires even when the λ value is ignored in "wavenumber" mode.

**`n_steps`.**
- The discrete Fourier-mode count `floor((kc_max − kc_min)·L_max/2π)` clipped to [10, 50] is correct math for linear k (Δk = 2π/L).
- It is now coupled to a *geometric* grid, so it is a heuristic, not a resolution rule.
- Narrow ranges get few samples: λ ∈ [100, 1500] gives only 18 points, a 16 % k-step.
- A per-decade rule (for example 15–20 samples/decade) would be cleaner. This is Low severity.
- At the low-k end consecutive geometric samples are closer than Δk, so some are redundant. Fine for cost, but not "discrete modes".

**Padding** [V]: the optimiser path (`variogram.py` L464) passes `kc_min_val`. All good. `Solver._get_fft_dem_grids` (`solver.py` L459) does not, so at σ_px = 32 the cached smoothed DEM differs from the optimised one by about 12 m. This only affects `curvature_dem` drifts.

**Remaining sweep issues.**
- Interactive entry of `kc_min = 0` crashes with `ValueError: Geometric sequence cannot include zero` (verified for the call). The interactive path is not re-validated (`variogram.py` L538-562).
- The interactive n_steps prompt has no upper clamp (L560).
- Best-kc selection uses `γ/mean²` averaged within the range fitted on the first (least-smoothed) kc.
- The boundary-warning logic (L733-737) is correct.

### 3.2 Dual kriging engine (`interpolation.py` L210-473)

**Verified good.**
- **Block regularisation:** ε is added only to `K[:N,:N]`. The Lagrange rows/columns are untouched, so `Fᵀw = 0` is exact.
- **Scale invariance:** sill ×4·10⁴ gives Δ ≤ 1.1·10⁻¹³ m [V].
- **Fixes:** LU solve (`lu_factor`/`lu_solve`) and a flat-external-drift guard (L297-300).
- **Coordinate normalisation:** drift columns are centred and scaled.

**Findings.**
1. **[Medium] Tikhonov sign convention is inverted for the semivariogram form** [V].
   - The system is built with Γ (γ(0) = 0 on the diagonal). Adding +ε to Γ's diagonal corresponds to *subtracting* ε from the covariance diagonal. That is negative noise.
   - The correct stabiliser in γ-form is −ε (equivalent to a +ε noise variance on C).
   - Conditioning of the augmented matrix for a near-duplicate pair (sill 1, ε = 10⁻⁶):

   | pair separation h | cond, +ε (code) | cond, −ε | cond, none |
   |---|---|---|---|
   | 10⁻³ m | 4.8·10⁵ | 4.2·10⁵ | 4.5·10⁵ |
   | 10⁻⁴ m | 1.4·10⁷ | 2.7·10⁶ | 4.5·10⁶ |
   | 6.7·10⁻⁵ m | **1.4·10⁹** | 3.4·10⁶ | 6.7·10⁶ |
   | 10⁻⁵ m | 8.0·10⁶ | 5.9·10⁶ | 4.5·10⁷ |

   - The "+ε" form is *worse than no regularisation* for h ≈ 10⁻⁴–10⁻³ m, and nearly singular at 2γ(h) ≈ ε (h ≈ 7·10⁻⁵ m).
   - Practical impact is small: picks are de-duplicated at 0.1 m. Migrated picks could still converge. Also, `np.mean(np.diag(K_sample))` is always 0, so ε = 10⁻⁶·sill.
   - Fix: `K[:N,:N] -= ε·I`, with ε = 10⁻⁶·sill (or relative to `trace(Γ_offdiag)`). Re-run the scale-invariance test.
2. **[Medium] An explicit `K_inv` is still formed** (L376, `lu_solve(lu_piv, eye)`).
   - Cost is O(N³) plus (N+d)² memory.
   - Variance is evaluated at O(N²) per grid cell (L436), even when no uncertainty output is requested.
   - Suggestion: `lu_solve(lu_piv, K_rhs_sub)` per chunk (same cost, no stored inverse), and skip the variance branch when it is unused.
3. **[Low] Silent quadratic fallback in the engine itself.**
   - If the external-drift raster is flat (L297-300) and no polynomial term is requested, `n_drift` becomes 6 (a quadratic) (L323-326).
   - Method `ordinary` combined with any external grid is upgraded to universal (`kriging_interpolation`, L756). Both are undocumented.
4. **[Low] Variogram type mismatch.** The P-product fitted sill is used for T/D targets. With scale invariance now fixed, only the **variance scale** of T/D maps is mis-scaled.

### 3.3 3D Eikonal ray migration (`migration.py` L125-222)

Horizontal slownesses are u = −∂T/∂x and w = −∂T/∂y. With A = 1 + z_x² + z_y², B = u·z_x + w·z_y and C = u² + w² − v⁻²:

`p_z = (B − √(B² − AC))/A`, `p_x = u − p_z·z_x`, `p_y = w − p_z·z_y`, displacement `= T·v²·p`.

**[V] Direct call to `migrate_eikonal_points`** versus exact plane-over-plane bed location (v = 0.17, `T = H_normal/v`):

| Surface slopes (zx, zy); bed dips (bx, by) | Exact bed point (x, y, z) | `migrate_eikonal_points` |
|---|---|---|
| (0, 0); (0.1, 0) | (712.87, 605.00, 2921.29) | (712.87, 605.00, 2921.29) |
| (0.2, 0); (0, 0) | (705.00, 605.00, 2850.00) | (705.00, 605.00, 2850.00) |
| (0.2, 0.05); (0.05, −0.02) | (719.86, 599.06, 2874.01) | (719.86, 599.06, 2874.01) |
| (0.3, 0.1); (−0.04, 0.03) | (687.76, 617.93, 2841.03) | (687.76, 617.93, 2841.03) |
| (−0.25, 0.15); (0.06, 0.06) | (704.16, 604.16, 2928.50) | (704.16, 604.16, 2928.63) (my synthetic bed lies above the surface here, so thickness is clamped) |
| (0, 0.2); (0, 0.1) | (705.00, 625.84, 2912.58) | (705.00, 625.84, 2912.58) |

- The thickness formula `d = −dz + (z_s(x_mig) − z_s(x))` is correct.
- The evanescent clamp (`disc < 0` → s₃ = B/A) and its counter are logged.

**Remaining issues.**
- The repo test `test_eikonal_migration_plane.py` cases 1–6 all use a *constant* T (parallel planes → zero gradient). Cases 5/6 are named "non-parallel" but are parallel.
- Only `test_inclined_bed_traveltime_gradient` has a non-zero T-gradient, and only with a flat surface. A test with a sloped surface and a sloped bed would have caught the old regression.
- Displacement uses the *gridded, smoothed* T at the pick location, not the pick's own T. This is by design (MIG.m) but should be documented.
- `valid_d = d_mig > 0` replaces zero migrated thickness with the unmigrated value (L205-212).
- The unit heuristic `pts[:,3].max() > 15` remains (`solver.py` L805), although `survey_data_type` is known.

### 3.4 `nrbins` lag binning [V]

| Quantity | Value |
|---|---|
| README L250/L385, `calculate_variogram` docstring L141-142, log at `variogram.py` L471/L488 | `max(3, N_pairs//30)` bins, uncapped |
| Code (`variogram.py` L207-213) | `min(30, max(3, in_range_pairs//30))` |

- N = 140 → docs/log say 324 bins; actual **30** bins (avg 239 pairs/bin).
- N = 3000 → docs/log say 149 950 bins; actual **30**.
- The e2e log printed "324 bins … avg = 30 pairs/bin" while the variogram used 30 bins.
- A **user-set** `nrbins=15` is honoured (floor 3).
- For N > 5000 the chunked branch of `calculate_variogram` (L158-196) uses the **uncapped** formula on *all* pairs. It also loops in Python: about 6·10⁵ bins at N = 6000.
- Recommendation: keep the cap (30–50 bins is a sane default), and make code, log, docstring and README agree. Share one helper between `calculate_variogram` and the log.

### 3.5 Export / pipeline architecture

Current paths that write rasters:

| Writer | Exports |
|---|---|
| `Solver.migrate_eikonal` (`solver.py` L825-826, L872-873) | traveltime, migrated points |
| `Solver.finalize_bedrock` → `export_outputs("finalization")` (L1300) | thickness, uncertainty, BSS (+unc.) |
| `pipeline.run_from_config` → `PipelineExporter.export_all()` (L149) | **repeats all of the above** |
| `pipeline.run_from_config` L151-160 | final bedrock only if the undocumented `outputs.output_bedrock_map` is set |
| `Solver.run_pipeline` L1481-1489 | `<prefix>_bedrock` (the only place it is written by default) |

[V] A default batch run produced `res_thickness.npy` etc. twice in the log and **no bedrock file**.

Recommended consolidation:
1. One `Solver.export_all(include_bedrock=…)` that honours `OutputsConfig.save_bedrock_elevation_map` and uses one naming scheme.
2. Call it exactly once from `run_from_config`.
3. Make `Solver.run_pipeline` delegate to `pipeline.run_from_config` (or the reverse) so that defaults live in one place.
4. Remove `PipelineExporter`, or turn it into the real owner of the logic.

---

## 4. Schema & Documentation Parity Audit

Sources: `config.py` `DEFAULT_CONFIG` (L69-146), `pysole.json`, `README.md` (JSON snippet L139-216, table L224-285), `examples/{gok,wuk}/*.json`, `docs/*.md`, module docstrings.

### 4.1 Parameter table

| Parameter | config.py | pysole.json | README | examples | docs / other | Status |
|---|---|---|---|---|---|---|
| all `inputs.*` keys incl. `survey_profile_column` | ✔ | ✔ identical | ✔ | ✔ | — | ✅ names/defaults/types |
| `inputs.drift_analyzer` | read in `solver.py` L577, not in schema | ✘ | ✘ | ✘ | — | ❌ undocumented |
| `inputs.output_dir` | read in `pipeline.py` L87 only as fallback | ✘ | ✘ | ✘ | — | ❌ legacy undocumented |
| `spatial_parameters.*` | ✔ | ✔ | ✔ | ✔ | — | ✅ |
| `migration_parameters.{perform_migration, velocity 0.16, interactive_migration false}` | ✔ | ✔ | ✔ | ✔ | — | ✅ (velocity prompt unreachable from the pipeline) |
| `optimization.fft_filter_metric` | "wavenumber" | ✔ | ✔ L244 | ✔ | CHANGELOG | 🟡 precedence undocumented |
| `optimization.kc_max` | null → π/Δ | ✔ | L245 ✔ | ✔ | — | 🟡 silent Nyquist clamp (`variogram.py` L435) |
| `optimization.kc_min` | null → 4π/L_max | ✔ | L246 says "half of the DEM extent" | ✔ | — | 🟡 wording ambiguous (it is λ_max = L/2) |
| `optimization.lambda_min / lambda_max` | null | ✔ | L247-248 ✔ | ✔ | — | 🟡 as above |
| `optimization.n_steps` | null → clip(modes,10,50); user value clipped [3,50] | ✔ | L249: [10, 50] only | ✔ | — | 🟡 |
| `optimization.nrbins` | null | ✔ | L250/L385 formula wrong | 15 (int) | docstring L141 | ❌ semantics (section 3.4) |
| `optimization.slope_floor_deg` | 5.0 | ✔ | ✔ | ✔ | drift doc | 🟡 not forwarded to `DriftAnalyzer`/`SurveyPlanner` |
| `optimization.interactive_optimization` | false | ✔ | L252 ✔ | ✔ | — | 🟡 stage 1 never interactive |
| `kriging_parameters.engine` | "native" | ✔ | L253 ✔ | ✔ | perf report uses removed `built_in_kriging` | ❌ **ignored by code** |
| `kriging.{pre,post}_migration.interpolation_target` | "P" | ✔ | L262 says "T" for post | ✔ | — | ❌ README typo (code: "D") |
| `…method`, `…drift_analyzer`, `…drift_terms` | ✔ | ✔ | L256-258, 263-265 (L265 copy-paste) | ✔ | practice guide uses stale names | 🟡 |
| `…variogram_model` | "spherical" | ✔ | lists "linear" | ✔ | `solver.py` L83/L91 lists "linear" | ❌ "linear" rejected (`interpolation.py` L668) |
| `…include_zero_boundary_condition` | True | ✔ | ✔ | ✔ | — | ✅ |
| `finalization_parameters.*` (8 keys) | ✔ | ✔ | ✔ | smooth_bedrock=true | — | ✅ |
| `outputs.output_dir` | None | ✔ | L276 | "." | — | ❌ resolved inconsistently (N3-M3) |
| `outputs.output_format / output_prefix / plots_dir` | "tif" / "final" / "figures" | ✔ | ✔ | ✔ | — | ✅ |
| `outputs.save_*` (6 keys) | False | ✔ | L280-285 | ✔ | — | ✅ (README L281 says `save_migrated_points` is auto-false; code only logs) |
| `outputs.save_bedrock_elevation_map` | dataclass True; not in `DEFAULT_CONFIG` | ✘ | ✘ | ✘ | CHANGELOG L140 | ❌ dead field |
| `outputs.output_bedrock_map` | read at `pipeline.py` L151 | ✘ | ✘ | ✘ | — | ❌ undocumented |
| env `PYSOLE_LOG_LEVEL` | read at `config.py` L293 | — | ✘ | — | — | 🟡 undocumented |

### 4.2 Other parity results

- **Template ↔ code:** `pysole.json` and the README JSON snippet equal `DEFAULT_CONFIG` key for key and value for value, including int vs float. `pysole --init` dumps `DEFAULT_CONFIG`, so it is identical too.
- **Examples:** both example configs differ from the template only in paths, `nrbins: 15` (int ✅), `smooth_bedrock: true`, `output_dir: "."` and the prefix. All keys are in the schema, so they pass. There is **no** config validation: unknown keys and wrong types are merged silently.
- **Versions:** 0.4.3 is consistent in `pyproject.toml` L7, `__init__.py` L18, CHANGELOG L8 and `docs/release_guide.md`.
- **kc default:** 0.0314 everywhere in code and user docs. 0.5 remains only in historical audits.
- **CLI flags:** the README documents only `--init`, config, `--batch`, `--verbose` and `--help`. It omits `-V`, `--non-interactive`, `--debug`, `--drift-analyzer`, `--profile-col` and the `plan-survey` subcommand.
- **Docs using removed identifiers** (raise `ValueError: Unknown drift term`):
  - `docs/pysole_interpolation_practice_guide.md`: `sia_thickness`, `z_surface`, `regional_linear`, `quadratic` (lines 10, 14, 19, 60, 78-83, 99-106, 242, 251, 253).
  - `docs/gok_thickness_analysis_report.md`: lines 13, 28, 32, 57.
  - `docs/drift_analyzer_&_survey_planner.md` L330: `--output proposed_survey` is not a valid flag.
- **Other doc inconsistencies:**
  - The curvature sign differs between the drift doc L27 and the code/README L453.
  - The drift doc says 14 candidate models; the code has 12.
  - The practice guide uses ρ = 917 vs 900 in code.
  - `kriging_performance_report.md` uses removed keys (`built_in_kriging`, `d_kc`) and a Tikhonov 1e-8 that is not what the code uses.
  - README L790/L792 and the practice guide L256 have absolute `file:///home/db/Software/pysole/...` links.
- **Examples directory:**
  - `examples/pysole_quickstart.ipynb` cell 3 calls `solver.recommend_drift_model(stage=..., cv_mode="lopo")`. The signature is `(stage, interactive)` → `TypeError`.
  - Cells 2-3 use `"wuk/pysole_wuk.json"`, which only works when cwd is `examples/`.
  - `examples/{gok,wuk}/run_*_example.py` print "Final Bedrock Raster Saved" while nothing is saved (new High).
  - The committed `*_final_bedrock.tif` files are leftovers from an older version.
  - `examples/apo/` has data only.

---

## 5. New Findings & Edge Cases

IDs: N4 = new in this pass.

### [High]

**N4-H1 — The main entry point never writes the final bedrock raster** [V]
- [pipeline.py L148-162](file:///home/db/Software/pysole-review/src/pysole/pipeline.py#L148-L162): `export_all()` only exports optional grids. The bedrock map is saved only when the undocumented key `outputs.output_bedrock_map` is present.
- README L278 claims `<output_prefix>_bedrock.<ext>` is exported by default. CHANGELOG L15 claims `<prefix>_bedrock_elevation_map.tif`. `OutputsConfig.save_bedrock_elevation_map` ([config.py L28](file:///home/db/Software/pysole-review/src/pysole/config.py#L28)) is dead.
- [V] The e2e output folder held `res_thickness.npy`, `res_thickness_uncertainty.npy`, `res_basal_shear_stress*.npy`, `res_traveltime.npy` and `res_migrated_points.csv`, but **no bedrock file**. The function returned the grid in memory only.
- The CLI (`pysole pysole.json`) and both example scripts leave users without the primary product.
- Fix:
  - Export the bedrock when `save_bedrock_elevation_map` (default True) is set, using `<prefix>_bedrock`.
  - Add a regression test asserting the file exists after `run_from_config`.
  - Update README L278 and CHANGELOG L15.

### [Medium]

**N4-M1 — `kriging_parameters.engine` is ignored** — [solver.py L723](file:///home/db/Software/pysole-review/src/pysole/solver.py#L723), [L173](file:///home/db/Software/pysole-review/src/pysole/solver.py#L173)
- `kriging_interpolation(engine=self.kriging_engine)` passes the `KrigingEngine` *object* (L173). The string is stored in `self.engine_type` (L158).
- `interpolation.py` L649-652 treats any non-string as "native". `"pykrige"` is therefore unreachable from JSON.
- Fix: `engine=self.engine_type`.

**N4-M2 — Output directory / log resolution still wrong** [V] — [pipeline.py L87](file:///home/db/Software/pysole-review/src/pysole/pipeline.py#L87)
- `resolve_output_dir(outputs.output_dir or inputs.output_dir, config_file)` passes the config file as `survey_data_path`.
- A relative `output_dir` resolves against the **cwd** here, but against the config directory in `Solver`. [V] My run left a stray `pysole.log` at `<cwd>/out/pysole.log`.
- With `output_dir: null`, the log goes to `<cfg_dir>/pysole/`, while outputs go to `<survey_dir>/pysole/` (confirmed with `resolve_output_dir`: `cfg/../data/pysole` vs `cfg/pysole`).
- Fix: `resolve_output_dir(output_dir=outputs.get("output_dir"), survey_data_path=inputs.get("survey_data_path"), config_path=config_file)`.

**N4-M3 — LOPO cross-validation is still dead; the profile column is lossy**
- [solver.py L622-624](file:///home/db/Software/pysole-review/src/pysole/solver.py#L622-L624): `get_sample_points` returns 3 columns, so `prof_data` at [L686](file:///home/db/Software/pysole-review/src/pysole/solver.py#L686) is always `None`. `recommend_drift_model` passes none either.
- `migrate_eikonal_points` returns 4 columns, so the profile column is lost after migration.
- [raster.py L843](file:///home/db/Software/pysole-review/src/pysole/raster.py#L843): string IDs → NaN. Duplicate consolidation `nanmean`s the ID column (L905).
- README's `[(profile_id), X, Y, value]` layout is not what the loader accepts.

**N4-M4 — CLI `--drift-analyzer` is a no-op; CLI `--verbose` is overridden**
- `solver.drift_analyzer` is set ([pipeline.py L121-122](file:///home/db/Software/pysole-review/src/pysole/pipeline.py#L121-L122)) but never read; the per-stage JSON flag is used (`solver.py` L659). `Solver.is_batch_mode` (L667) is never set.
- `--debug` sets `log_level` (L90). `Solver.from_config` then calls `load_config(..., log_level=None)` ([solver.py L502](file:///home/db/Software/pysole-review/src/pysole/solver.py#L502)), which re-initialises logging from the config/env ([config.py L293-294](file:///home/db/Software/pysole-review/src/pysole/config.py#L293-L294)). README L648 claims the CLI wins.

**N4-M5 — Python-API defaults differ from the config defaults**
- `Solver.__init__` defaults to `universal` + `["sia"]` (`solver.py` L37, L42, L144, L150) with target "P". The config defaults are `ordinary` + `[]`.
- The README's Python example (L690-697) therefore runs the SIA-drift-on-product combination that README L420 and `solver.py` L652-656 warn about.

**N4-M6 — `variogram_model="linear"` documented but rejected** — [interpolation.py L668-672](file:///home/db/Software/pysole-review/src/pysole/interpolation.py#L668-L672); docs in `solver.py` L83/L91 and README L259/L266.

**N4-M7 — Compound drifts use raw-DEM curvature; solver FFT cache lacks `kc`**
- [solver.py L707](file:///home/db/Software/pysole-review/src/pysole/solver.py#L707): only the literal `"curvature_dem"` triggers the smoothed curvature. `sia_curvature_dem`, `z_dem_curvature_dem`, `full_physical`, `full_spatial_physical` fall back to raw curvature (`interpolation.py` L730-732). That is a noisy second-difference field.
- [solver.py L459](file:///home/db/Software/pysole-review/src/pysole/solver.py#L459): `precompute_fft_grid` is called without `kc`. The 40 px default pad gives up to 12 m error at σ_px = 32 [V].

**N4-M8 — Lag-bin documentation / log mismatch** [V] — section 3.4. Also the N > 5000 chunked branch ([variogram.py L158-196](file:///home/db/Software/pysole-review/src/pysole/variogram.py#L158-L196)).

**N4-M9 — Tikhonov sign convention** [V] — section 3.2, item 1 ([interpolation.py L366-367](file:///home/db/Software/pysole-review/src/pysole/interpolation.py#L366-L367)).

**N4-M10 — Memory and cost**
- The kc sweep keeps every smoothed DEM and slope (`.copy()` and the materialised `executor.map` list, [variogram.py L644-661](file:///home/db/Software/pysole-review/src/pysole/variogram.py#L644-L661)). That is O(n_kc·M·N). For a 2000² DEM at 50 steps it is about 3 GB.
- Explicit `K_inv`. Variance always computed at O(N²) per cell.
- `load_survey_points` de-duplication loop is O(N·U) in Python.

**N4-M11 — Interactive scope narrower than documented**
- Stage-1 BSS optimisation is invoked inside `get_sample_points` with `interactive=False` ([solver.py L607-609](file:///home/db/Software/pysole-review/src/pysole/solver.py#L607-L609)).
- The velocity prompt (L795) is unreachable from the pipeline, since the velocity is always passed.
- `.get("interactive_*", True)` ([pipeline.py L130-131](file:///home/db/Software/pysole-review/src/pysole/pipeline.py#L130-L131)) contradicts the schema default False (dead after the merge).
- `Solver.run_pipeline` uses `show_progress` as a batch flag (L1447).
- `_sample_pts_cache` is not invalidated when `optimize_bss` or `migrate_eikonal` is re-run via the API.

**N4-M12 — Duplicate exports and two pipelines** — section 3.5.

**N4-M13 — Release documents broken** — the notebook `TypeError` and stale drift identifiers (section 4.2).

### [Low]

- **N4-L1** `n_steps` heuristic is coupled to linear k while sampling is geometric (section 3.1). `n_steps ∈ {1,2}` and `>50` are clamped silently. The interactive prompt skips re-validation: `kc_min = 0` crashes in `np.geomspace`.
- **N4-L2** The `λ_min < λ_Nyquist` warning (`variogram.py` L429) fires although the λ value is ignored under the "wavenumber" metric. `kc_max` above Nyquist is clamped silently (L435).
- **N4-L3** `finalize_bedrock` has two consecutive string literals, so the Parameters docs are not the real docstring (`solver.py` L1093-1111).
- **N4-L4** `np.random.seed(42)` mutates global RNG state (`interpolation.py` L825). Use `np.random.default_rng(42)`.
- **N4-L5** `unmig_depths` unit heuristic (`solver.py` L805) when migration is skipped. `valid_d = d_mig > 0` (`migration.py` L205).
- **N4-L6** `plotting.py` L12-16 still switches the backend at import. `[INFO]` doubled prefixes remain.
- **N4-L7** `from_config` stores `str(dict)` as `config_path` for dict configs ([solver.py L576](file:///home/db/Software/pysole-review/src/pysole/solver.py#L576)). Resolvers then treat it as a file path.
- **N4-L8** Unused imports (section 2.1). `finalize_bedrock`/`smooth_bedrock_dem` smooth in pixel units (σ in px) on anisotropic grids.
- **N4-L9** CHANGELOG internal contradictions and the wrong σ_px formula (L31). `release_guide.md` L45 calls WUK "Weighted Universal Kriging" (README: Wurtenkees). `pysole --init` overwrites an existing config silently. README CLI section incomplete.
- **N4-L10** Tests: no tests for `fft_filter_metric` precedence beyond the basics, `engine` honouring, or bedrock-file output. `test_compound_drift_expansion` asserts "not None". `test_eikonal_migration_oblique_slope` asserts finiteness. The Eikonal "non-parallel" cases are parallel.

---

## 6. Definitive Release Readiness Conclusion

**Do not publish v0.4.3 yet.** The numerical core is now in good shape, but the user-facing path is not.

### Must fix before tagging (blockers)

| # | Item | Where | Effort |
|---|---|---|---|
| B1 | Write the final bedrock raster in `run_from_config`, honouring `save_bedrock_elevation_map`; add a test asserting the file exists | `pipeline.py` L148-162, `config.py` L28 | small |
| B2 | Fix output-dir/log resolution (keyword args, `outputs.output_dir`) | `pipeline.py` L87 | trivial |
| B3 | Pass the engine **string** to `kriging_interpolation`; add a test | `solver.py` L723 | trivial |
| B4 | Make docs and CHANGELOG truthful, or implement: `--drift-analyzer`, LOPO/profile column, `--verbose` precedence, `"linear"` variogram, `nrbins` formula and log, `post_migration` target typo | README, CHANGELOG, `solver.py` | small |
| B5 | Repair the quickstart notebook; drop or rewrite stale drift identifiers in the practice guide, the GOK report and the drift doc `--output` example; fix example-script messages | `examples/`, `docs/` | small |
| B6 | Run `MPLBACKEND=Agg python -m unittest discover -s tests` in a complete environment (shapely, pandas, sklearn, rasterio, pykrige) and confirm it passes | CI / maintainer | — |

After B1–B6 I would rate the codebase **GO for a 0.4.3 beta tag** (`Development Status :: 4 - Beta` in `pyproject.toml`).

### Should fix (not blocking)

1. Flip the Tikhonov sign (`K[:N,:N] -= ε·I`) and drop the dead `mean(diag)` term.
2. Replace the explicit `K_inv`, and skip the variance branch when it is unused.
3. Pass `kc` into `Solver._get_fft_dem_grids`, and inject smoothed curvature for compound drift names.
4. Stream the kc sweep (keep a running best, discard other DEMs).
5. Add precedence documentation for `kc_*` vs `lambda_*`, and a boundary warning when the λ value is ignored.
6. Add a non-constant-T, sloped-surface Eikonal test.
7. Consolidate `Solver.run_pipeline` and `pipeline.run_from_config`, and the exporters, into one owner.

### Defer to 0.4.4+

LOPO end to end (profile column through get_sample_points and migration), a per-decade `n_steps` rule, a robust variogram estimator (A1 D5), `VariogramParams` dataclass (D2), flowline tracing (D6), a single declarative schema (`config.py`) that generates `pysole.json`, the README table and the CLI.

### Repository state

No file in the repository was created or modified. All harnesses and outputs are in `/tmp/pysole_review2/`.
