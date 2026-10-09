"""
Drift Analyzer helper tool for Universal Kriging drift model recommendation.
Analyzes input survey datasets and surface DEM covariates to evaluate, rank,
and recommend optimal single and multi-drift models for Universal Kriging.
"""

from dataclasses import dataclass
from typing import Any
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from .logging import logger
from .interpolation import DriftBasis, DualKrigingSolver


@dataclass
class CandidateDriftResult:
    """Diagnostic evaluation metrics for a candidate Universal Kriging drift model."""
    name: str
    terms: list[str]
    n_terms: int
    vif_max: float
    is_multicollinear: bool
    cv_rmse: float
    cv_mae: float
    cv_r2: float
    aicc: float
    is_penalized: bool = False
    warning_msg: str | None = None


# Supported single and multi-drift candidate combinations
CANDIDATE_DRIFT_MODELS: list[dict[str, Any]] = [
    {"name": "linear_xy", "terms": ["linear_xy"], "type": "single"},
    {"name": "quadratic_xy", "terms": ["quadratic_xy"], "type": "single"},
    {"name": "z_dem", "terms": ["z_dem"], "type": "single"},
    {"name": "sia", "terms": ["sia"], "type": "single"},
    {"name": "curvature_dem", "terms": ["curvature_dem"], "type": "single"},
    {"name": "sia_space", "terms": ["sia_space"], "type": "multi"},
    {"name": "sia_z_dem", "terms": ["sia_z_dem"], "type": "multi"},
    {"name": "sia_curvature_dem", "terms": ["sia_curvature_dem"], "type": "multi"},
    {"name": "z_dem_curvature_dem", "terms": ["z_dem_curvature_dem"], "type": "multi"},
    {"name": "sia_z_dem_curvature_dem", "terms": ["sia_z_dem_curvature_dem"], "type": "multi"},
    {"name": "full_physical", "terms": ["full_physical"], "type": "multi"},
    {"name": "full_spatial_physical", "terms": ["full_spatial_physical"], "type": "multi"},
]


class DriftAnalyzer:
    """
    Automated diagnostic recommendation engine for Universal Kriging drift models.
    """

    def __init__(
        self,
        mode: str = "pre_migration",
        target_name: str = "traveltime",
        interpolation_target: str = "P",
        include_zero_boundary: bool = True,
        survey_profile_column: str | None = None,
        vif_threshold: float = 10.0,
        slope_floor_deg: float = 5.0,
    ):
        self.mode = mode
        self.target_name = target_name
        self.interpolation_target = interpolation_target
        self.include_zero_boundary = include_zero_boundary
        self.survey_profile_column = survey_profile_column
        self.vif_threshold = vif_threshold
        self.slope_floor_deg = slope_floor_deg

    def calculate_vif(self, X_drift: np.ndarray) -> tuple[float, list[float]]:
        """
        Calculates Variance Inflation Factor (VIF) for each column in a drift basis matrix using exact linear least-squares regression.
        """
        n_samples, n_features = X_drift.shape
        if n_features <= 1:
            return 1.0, [1.0]

        vifs: list[float] = []
        for i in range(n_features):
            y_i = X_drift[:, i]
            if np.std(y_i) <= 1e-12:
                vifs.append(1.0)
                continue
            X_others = np.delete(X_drift, i, axis=1)
            A = np.column_stack([np.ones(n_samples), X_others])
            try:
                coef = np.linalg.lstsq(A, y_i, rcond=None)[0]
                y_pred = A @ coef
                ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
                if ss_tot <= 1e-12:
                    r_sq = 0.0
                else:
                    ss_res = np.sum((y_i - y_pred) ** 2)
                    r_sq = max(0.0, 1.0 - (ss_res / ss_tot))
            except Exception:
                r_sq = 0.0

            if r_sq >= 0.9999:
                vif = 999.0
            else:
                vif = 1.0 / (1.0 - r_sq)
            vifs.append(float(vif))

        return float(np.max(vifs)), vifs

    def screen_feature_importances(
        self,
        covariates_df: pd.DataFrame,
        target_values: np.ndarray,
    ) -> pd.DataFrame:
        """
        Computes linear Partial Correlations (Pearson & Spearman) and Random Forest Permutation Importances.
        """
        feature_cols = [c for c in covariates_df.columns if c not in ["X", "Y"]]
        if not feature_cols:
            return pd.DataFrame()

        results = []
        X_mat = covariates_df[feature_cols].values

        # Clean NaNs in features or target for Random Forest fitting
        valid_rows = ~(np.isnan(X_mat).any(axis=1) | np.isnan(target_values))
        X_clean = X_mat[valid_rows]
        y_clean = target_values[valid_rows]

        rf_importances = np.zeros(len(feature_cols))
        if len(y_clean) >= 5:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            try:
                rf.fit(X_clean, y_clean)
                try:
                    from sklearn.inspection import permutation_importance
                    perm_res = permutation_importance(rf, X_clean, y_clean, n_repeats=5, random_state=42)
                    rf_importances = np.maximum(perm_res.importances_mean, 0.0)
                except Exception:
                    rf_importances = getattr(rf, "feature_importances_", np.zeros(len(feature_cols)))
            except Exception:
                pass

        for idx, col in enumerate(feature_cols):
            x_v = covariates_df[col].values
            valid_p = ~(np.isnan(x_v) | np.isnan(target_values))
            if np.sum(valid_p) >= 3 and np.std(x_v[valid_p]) > 1e-12 and np.std(target_values[valid_p]) > 1e-12:
                p_corr, _ = stats.pearsonr(x_v[valid_p], target_values[valid_p])
                s_rho, _ = stats.spearmanr(x_v[valid_p], target_values[valid_p])
            else:
                p_corr, s_rho = 0.0, 0.0

            results.append({
                "feature": col,
                "pearson_corr": float(p_corr) if not np.isnan(p_corr) else 0.0,
                "spearman_rho": float(s_rho) if not np.isnan(s_rho) else 0.0,
                "rf_importance": float(rf_importances[idx]) if idx < len(rf_importances) and not np.isnan(rf_importances[idx]) else 0.0,
            })

        return pd.DataFrame(results).sort_values(by="rf_importance", ascending=False)

    def evaluate_spatial_cv(
        self,
        x_pts: np.ndarray,
        y_pts: np.ndarray,
        z_values: np.ndarray,
        dem_grid: np.ndarray,
        dx: float,
        dy: float,
        bounds: tuple[float, float, float, float],
        alpha_opt_deg: np.ndarray,
        variogram_model: str,
        variogram_params: dict[str, float],
        cv_mode: str = "buffer",
        buffer_radius_m: float = 50.0,
        profile_ids: np.ndarray | None = None,
    ) -> list[CandidateDriftResult]:
        """
        Runs fast Dual Kriging Spatial Cross-Validation across candidate single/multi-drift models.
        """
        n_points = len(z_values)
        c0 = variogram_params.get("nugget", 0.0)
        c = variogram_params.get("sill", 1.0)
        a = variogram_params.get("range", 100.0)

        model_candidates_data = []

        # Determine spatial CV evaluation indices
        if cv_mode == "lopo" and profile_ids is not None:
            eval_indices = np.arange(n_points)
        else:
            if n_points > 100:
                rng = np.random.default_rng(42)
                eval_indices = rng.choice(n_points, size=100, replace=False)
            else:
                eval_indices = np.arange(n_points)

        for model_cfg in CANDIDATE_DRIFT_MODELS:
            name = model_cfg["name"]
            drift_terms = model_cfg["terms"]

            is_sia_slope = any("sia" in t for t in drift_terms)
            warning_msg = None
            if self.interpolation_target == "P" and is_sia_slope:
                warning_msg = "SIA slope drift on Product P target causes 1/sin^2(alpha) double-scaling margin artifacts."

            try:
                drift_basis = DriftBasis(
                    drift_terms=drift_terms,
                    x_ref=x_pts,
                    y_ref=y_pts,
                    dem_grid=dem_grid,
                    dx=dx,
                    dy=dy,
                    bounds=bounds,
                    alpha_opt_deg=alpha_opt_deg,
                )
                n_drift = len(drift_basis.evaluators)
            except Exception as e:
                logger.warning(f"   Skipping drift model '{name}': {e}")
                continue

            X_drift = drift_basis.evaluate(x_pts, y_pts)
            vif_max, _ = self.calculate_vif(X_drift)
            is_multicollinear = vif_max > self.vif_threshold

            preds = np.full(n_points, np.nan)

            if cv_mode == "lopo" and profile_ids is not None:
                unique_profs = np.unique(profile_ids)
                for prof in unique_profs:
                    val_mask = profile_ids == prof
                    train_mask = ~val_mask
                    if np.sum(train_mask) < n_drift + 3:
                        continue
                    try:
                        solver = DualKrigingSolver(
                            x_pts[train_mask],
                            y_pts[train_mask],
                            z_values[train_mask],
                            variogram_model=variogram_model,
                            nugget=c0,
                            sill=c,
                            range_param=a,
                            drift_terms=drift_terms,
                            drift_basis=drift_basis,
                        )
                        preds[val_mask] = solver.predict(x_pts[val_mask], y_pts[val_mask])
                    except Exception as e:
                        logger.debug(f"LOPO fold prediction failed for profile '{prof}': {e}")
            else:
                for i in eval_indices:
                    dist = np.hypot(x_pts - x_pts[i], y_pts - y_pts[i])
                    train_mask = dist > buffer_radius_m
                    if np.sum(train_mask) < n_drift + 3:
                        continue
                    try:
                        solver = DualKrigingSolver(
                            x_pts[train_mask],
                            y_pts[train_mask],
                            z_values[train_mask],
                            variogram_model=variogram_model,
                            nugget=c0,
                            sill=c,
                            range_param=a,
                            drift_terms=drift_terms,
                            drift_basis=drift_basis,
                        )
                        p_val = solver.predict(np.array([x_pts[i]]), np.array([y_pts[i]]))
                        preds[i] = p_val[0]
                    except Exception as e:
                        logger.debug(f"Buffer LOO fold prediction failed for index {i}: {e}")

            model_candidates_data.append({
                "name": name,
                "drift_terms": drift_terms,
                "n_drift": n_drift,
                "vif_max": vif_max,
                "is_multicollinear": is_multicollinear,
                "warning_msg": warning_msg,
                "preds": preds,
            })

        if not model_candidates_data:
            return []

        # Find common non-NaN evaluation mask across valid candidate predictions for comparable AICc
        valid_masks = [~np.isnan(item["preds"][eval_indices]) for item in model_candidates_data]
        common_eval_mask_sub = np.all(valid_masks, axis=0) if valid_masks else np.array([])
        if np.sum(common_eval_mask_sub) < 5:
            # Fall back to per-candidate valid mask if common mask is too small
            common_eval_indices = eval_indices
        else:
            common_eval_indices = eval_indices[common_eval_mask_sub]

        results: list[CandidateDriftResult] = []
        for item in model_candidates_data:
            name = item["name"]
            drift_terms = item["drift_terms"]
            n_drift = item["n_drift"]
            vif_max = item["vif_max"]
            is_multicollinear = item["is_multicollinear"]
            warning_msg = item["warning_msg"]
            preds = item["preds"]

            valid_mask = common_eval_indices[~np.isnan(preds[common_eval_indices])]
            n_valid = len(valid_mask)

            if n_valid < n_drift + 3:
                results.append(CandidateDriftResult(
                    name=name,
                    terms=drift_terms,
                    n_terms=n_drift,
                    vif_max=vif_max,
                    is_multicollinear=is_multicollinear,
                    cv_rmse=9999.0,
                    cv_mae=9999.0,
                    cv_r2=-1.0,
                    aicc=99999.0,
                    is_penalized=True,
                    warning_msg=warning_msg or "Numerical singularity during CV.",
                ))
                continue

            residuals = z_values[valid_mask] - preds[valid_mask]
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            mae = float(np.mean(np.abs(residuals)))
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((z_values[valid_mask] - np.mean(z_values[valid_mask])) ** 2))
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

            # Akaike Information Criterion corrected (AICc)
            k_param = n_drift + 1
            if n_valid - k_param - 1 > 0 and ss_res > 0:
                aicc = float(n_valid * np.log(ss_res / n_valid) + 2 * k_param + (2 * k_param * (k_param + 1)) / (n_valid - k_param - 1))
            else:
                aicc = 99999.0

            is_penalized = is_multicollinear or (warning_msg is not None)

            results.append(CandidateDriftResult(
                name=name,
                terms=drift_terms,
                n_terms=n_drift,
                vif_max=vif_max,
                is_multicollinear=is_multicollinear,
                cv_rmse=rmse,
                cv_mae=mae,
                cv_r2=r2,
                aicc=aicc,
                is_penalized=is_penalized,
                warning_msg=warning_msg,
            ))

        results.sort(key=lambda r: (r.is_penalized, r.aicc, r.cv_rmse))
        return results

    def run_diagnostics(
        self,
        x_pts: np.ndarray,
        y_pts: np.ndarray,
        z_values: np.ndarray,
        dem_grid: np.ndarray,
        dx: float,
        dy: float,
        bounds: tuple[float, float, float, float],
        alpha_opt_deg: np.ndarray,
        variogram_model: str,
        variogram_params: dict[str, float],
        profile_data: np.ndarray | None = None,
        interactive: bool = True,
    ) -> list[CandidateDriftResult]:
        """
        Executes complete diagnostic pipeline and handles interactive terminal session.
        """
        t_char = "T" if "travel" in str(self.target_name).lower() else "D"
        target_descr = f"Product P(x,y) = {t_char}*sin(alpha)" if self.interpolation_target == "P" else f"Direct Field {self.interpolation_target}(x,y)"
        logger.info(f"[INFO] DriftAnalyzer initialized (Mode: {self.mode.title()}, Target: {self.target_name} [{target_descr}], Boundary Conditions: {self.include_zero_boundary}).")
        logger.info(f"[INFO] Extracting spatial covariates (Z_dem, C_kc, sin(alpha_opt)^-1, X, Y) at N = {len(z_values)} sample locations...")

        a_0 = float(variogram_params.get("range", 100.0))
        cv_mode = "buffer"
        buffer_radius_m = a_0 / 2.0
        profile_ids = None

        if self.survey_profile_column and profile_data is not None:
            profile_ids = np.asarray(profile_data)
            n_profs = len(np.unique(profile_ids))
            if n_profs >= 3:
                cv_mode = "lopo"
                logger.info(f"[INFO] Cross-Validation Mode: Leave-One-Profile-Out (LOPO-CV) using column '{self.survey_profile_column}' (N_profiles = {n_profs}).")
            else:
                logger.info(f"[INFO] Fewer than 3 unique profile IDs found (N_profiles = {n_profs}). Falling back to Spatial Buffer LOOCV.")

        if cv_mode == "buffer":
            if interactive and sys.stdin.isatty():
                print(f"\n[DriftAnalyzer] Fitted spatial range: a_0 = {a_0:.1f} meters.")
                try:
                    user_input = input(f"[DriftAnalyzer] Enter spatial exclusion buffer radius in meters [default: {buffer_radius_m:.1f} m]: ").strip()
                    if user_input:
                        buffer_radius_m = float(user_input)
                except Exception:
                    pass
            logger.info(f"[INFO] Fitted spatial range: a_0 = {a_0:.1f} meters. Spatial exclusion buffer radius set to r_buffer = {buffer_radius_m:.1f} meters.")

        logger.info(f"[INFO] Evaluating {len(CANDIDATE_DRIFT_MODELS)} candidate single and multi-drift models via Dual Kriging {cv_mode.upper()}...")
        eval_results = self.evaluate_spatial_cv(
            x_pts, y_pts, z_values, dem_grid, dx, dy, bounds, alpha_opt_deg,
            variogram_model, variogram_params, cv_mode=cv_mode,
            buffer_radius_m=buffer_radius_m, profile_ids=profile_ids,
        )

        top_model = eval_results[0] if eval_results else None
        if top_model:
            logger.info(f"[INFO] DriftAnalyzer ranking complete. Top model: '{top_model.name}' (RMSE = {top_model.cv_rmse:.2f}m, AICc = {top_model.aicc:.1f}).")

        # Interactive Terminal Ranking & Selection Table
        if interactive and sys.stdin.isatty() and eval_results:
            print("\n" + "=" * 80)
            print(f"       DRIFT ANALYZER CANDIDATE MODEL RECOMMENDATIONS ({self.mode.upper()})")
            print("=" * 80)
            print(f"{'#':<3} {'Model Name':<22} {'Terms':<25} {'RMSE (m)':<10} {'AICc':<10} {'VIF':<8} {'Status'}")
            print("-" * 80)
            for idx, res in enumerate(eval_results, 1):
                terms_str = "+".join(res.terms)[:24]
                status_str = "PENALIZED" if res.is_penalized else "VALID"
                print(f"{idx:<3} {res.name:<22} {terms_str:<25} {res.cv_rmse:<10.2f} {res.aicc:<10.1f} {res.vif_max:<8.1f} {status_str}")
            print("-" * 80)
            print(f"[DriftAnalyzer] Recommended model: #1 '{eval_results[0].name}'")
            try:
                sel = input(f"[DriftAnalyzer] Select drift model to apply [1-{len(eval_results)}, default: 1 ('{eval_results[0].name}')]: ").strip()
                if sel and sel.isdigit():
                    idx_sel = int(sel) - 1
                    if 0 <= idx_sel < len(eval_results):
                        chosen = eval_results[idx_sel]
                        # Move chosen model to top of list
                        eval_results.remove(chosen)
                        eval_results.insert(0, chosen)
            except Exception:
                pass

            logger.info(f"[INFO] User selected drift model #1 ('{eval_results[0].name}'). Updating configuration and launching Dual Kriging Solver.")

        return eval_results
