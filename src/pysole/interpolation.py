"""
Interpolation, Margin Blending, Kriging, and Random Forest Hole Filling.
Ported from MATLAB script INTERPOL.m by Daniel Binder (2011) and PySole modern workflow.
Supports Ordinary Kriging, Universal Kriging (default SIA physical drift & quadratic drift), and Regression Kriging.
Uses high-performance built-in vector Dual Kriging.
"""

from dataclasses import dataclass
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata, RBFInterpolator, RegularGridInterpolator
from scipy.linalg import lu_factor, lu_solve
from scipy.ndimage import distance_transform_edt, gaussian_filter
from sklearn.ensemble import RandomForestRegressor
from .raster import GridGeometry
from .logging import logger, get_progress_bar
from .smoothing import compute_gradients, compute_surface_curvature


@dataclass
class KrigingResult:
    """
    Typed data container storing Kriging bedrock elevation and variance interpolation outputs.
    """
    bedrock_grid: np.ndarray   # Interpolated bedrock elevation grid [m]
    variance_grid: np.ndarray  # Kriging estimation variance grid [\sigma^2]


class KrigingEngine:
    """
    Specialized engine executing Ordinary & Universal Kriging,
    Shallow Ice Approximation (SIA) physical drift interpolation, and bedrock reconstruction.
    """

    def __init__(
        self,
        dem: np.ndarray,
        geometry: GridGeometry,
        outline_mask: np.ndarray | None = None,
    ):
        self.dem = dem
        self.geometry = geometry
        self.outline_mask = outline_mask

    def interpolate(
        self,
        sample_points: np.ndarray,
        method: str = "universal",
        variogram_model: str = "spherical",
        opt_slope_grid: np.ndarray | None = None,
        drift_terms: list[str] | None = None,
        include_zero_boundary_condition: bool = True,
        n_cores: int = -1,
        engine: str = "native",
        show_progress: bool = True,
    ) -> KrigingResult:
        """Executes Kriging interpolation using engine spatial geometry settings."""
        return kriging_interpolation(
            sample_points=sample_points,
            geometry=self.geometry,
            method=method,
            variogram_model=variogram_model,
            dem_grid=self.dem,
            opt_slope_grid=opt_slope_grid,
            drift_terms=drift_terms,
            outline_mask=self.outline_mask,
            include_zero_boundary_condition=include_zero_boundary_condition,
            n_cores=n_cores,
            engine=engine,
            show_progress=show_progress,
        )


class BedrockFinalizer:
    """
    Specialized post-processing engine handling Random Forest gap filling,
    geomorphological margin blending, and spatial DEM smoothing.
    """

    def __init__(
        self,
        dem: np.ndarray,
        geometry: GridGeometry,
        outline_mask: np.ndarray | None = None,
    ):
        self.dem = dem
        self.geometry = geometry
        self.outline_mask = outline_mask

    def fill_holes(self, bedrock_grid: np.ndarray, n_cores: int = -1, show_progress: bool = True) -> np.ndarray:
        """Trains Random Forest ML model to fill bedrock holes inside creeping body."""
        return random_forest_hole_filling(
            dem=self.dem,
            bedrock_grid=bedrock_grid,
            boundary_mask=self.outline_mask,
            geometry=self.geometry,
            n_cores=n_cores,
            show_progress=show_progress,
        )

    def blend_margin(
        self,
        bedrock_input: np.ndarray | tuple[np.ndarray, ...],
        min_gap_dist: float | None = None,
    ) -> np.ndarray:
        """Applies geomorphological margin blending to surrounding terrain DEM."""
        return blend_margin_topography(
            dem=self.dem,
            bedrock_input=bedrock_input,
            boundary_mask=self.outline_mask,
            geometry=self.geometry,
            min_gap_dist=min_gap_dist,
        )


def blend_margin_topography(
    dem: np.ndarray,
    bedrock_input: np.ndarray | tuple[np.ndarray, ...],
    boundary_mask: np.ndarray,
    geometry: GridGeometry,
    min_gap_dist: float | None = None,
) -> np.ndarray:
    """
    Geomorphological margin blending: Assures a smooth transition from calculated bedrock
    to surrounding known surface DEM at the body margin boundary, pruning bedrock within min_gap_dist.

    Parameters
    ----------
    dem : 2D np.ndarray
        Surface elevation grid (M x N).
    bedrock_input : 2D np.ndarray or Nx4 point array
        Continuous bedrock elevation grid (M x N) or point array [X, Y, Z_surface, depth].
    boundary_mask : 2D np.ndarray
        Boolean grid indicating active creeping body / glacier area.
    geometry : GridGeometry
        Standardized spatial geometry container for raster grids.
    min_gap_dist : float, optional
        Minimum gap distance / margin blend zone width [m]. Prunes bedrock within this distance from the margin.

    Returns
    -------
    blended_dem : 2D np.ndarray
        Harmonized continuous bedrock elevation grid.
    """
    dx = geometry.dx
    dy = geometry.dy
    M, N = dem.shape

    cellsize = (dx + dy) / 2.0
    if min_gap_dist is None or min_gap_dist <= 0:
        margin_width = 3.0 * cellsize
    else:
        margin_width = float(min_gap_dist)

    xx, yy = geometry.meshgrid

    # Handle 2D bedrock elevation grid input
    if isinstance(bedrock_input, np.ndarray) and bedrock_input.shape == (M, N):
        bedrock_grid = bedrock_input.copy()
        thickness = dem - bedrock_grid
        thickness[~boundary_mask] = 0.0

        dist_from_margin = distance_transform_edt(boundary_mask, sampling=(abs(dy), abs(dx)))
        weight = np.clip(dist_from_margin / max(margin_width, 1e-6), 0.0, 1.0)
        weight = 0.5 * (1.0 - np.cos(np.pi * weight))  # smooth cosine transition

        tapered_thickness = thickness * weight
        sigma_px = (max(margin_width / (3.0 * abs(dy)), 0.5), max(margin_width / (3.0 * abs(dx)), 0.5))
        smoothed_thickness = gaussian_filter(tapered_thickness, sigma=sigma_px)
        final_thickness = weight * thickness + (1.0 - weight) * smoothed_thickness
        final_thickness[~boundary_mask] = 0.0

        return dem - final_thickness

    # Handle point array input
    pts = np.array(bedrock_input, dtype=np.float64)
    px = pts[:, 0]
    py = pts[:, 1]

    if pts.shape[1] >= 4:
        depths = pts[:, 3]
    else:
        depths = dem - pts[:, 2]

    try:
        rbf = RBFInterpolator(np.column_stack((px, py)), depths, kernel="thin_plate_spline", smoothing=0.1)
        grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
        thickness_grid = rbf(grid_pts).reshape((M, N))
    except Exception:
        thickness_grid = griddata((px, py), depths, (xx, yy), method="cubic", fill_value=0.0)

    thickness_grid = np.maximum(np.nan_to_num(thickness_grid, nan=0.0), 0.0)
    thickness_grid[~boundary_mask] = 0.0

    dist_from_margin = distance_transform_edt(boundary_mask, sampling=(abs(dy), abs(dx)))
    weight = np.clip(dist_from_margin / max(margin_width, 1e-6), 0.0, 1.0)
    weight = 0.5 * (1.0 - np.cos(np.pi * weight))

    tapered_thickness = thickness_grid * weight
    sigma_px = (max(margin_width / (3.0 * abs(dy)), 0.5), max(margin_width / (3.0 * abs(dx)), 0.5))
    smoothed_thickness = gaussian_filter(tapered_thickness, sigma=sigma_px)
    final_thickness = weight * thickness_grid + (1.0 - weight) * smoothed_thickness
    final_thickness[~boundary_mask] = 0.0

    return dem - final_thickness


CURVATURE_DRIFT_TERMS: frozenset[str] = frozenset(
    {
        "curvature_dem",
        "sia_curvature_dem",
        "z_dem_curvature_dem",
        "sia_z_dem_curvature_dem",
        "full_physical",
        "full_spatial_physical",
    }
)
"""Drift term names (primitive and compound) whose expansion includes the surface-curvature raster."""


def built_in_kriging_interpolation(
    sample_points: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    method: str = "universal",
    variogram_model: str = "spherical",
    external_drift_grid: np.ndarray | dict[str, np.ndarray] | None = None,
    drift_terms: list[str] | None = None,
    variogram_params: tuple[float, float, float] | dict[str, float] | None = None,
    n_cores: int = -1,
    show_progress: bool = True,
    return_variance: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust native NumPy/SciPy Ordinary & Universal Kriging solver with zero-centered
    spatial coordinate normalization and diagonal regularization to prevent ill-conditioned matrix explosion.
    Supports default quadratic spatial drift terms (1, x, y, x^2, y^2, x*y), custom external drifts
    (e.g., elevation z_surface, SIA thickness, surface curvature), and multi-drift combinations.
    Accelerated with multi-core CPU chunk parallelization via n_cores.

    Parameters
    ----------
    return_variance : bool, default True
        If True, the Kriging estimation variance is evaluated, which requires the explicit
        inverse of the augmented Kriging matrix and an extra O((N+d)^2) work per grid cell.
        If False, only the O(N) dual-kriging weight dot product is evaluated and the
        returned variance grid is filled with NaN (so that it cannot be misread as zero
        uncertainty). Use False whenever uncertainty maps are not requested.

    Notes
    -----
    The ``"linear"`` variogram model is the bounded 1-D profile model
    ``gamma(h) = nugget + sill * min(h / range, 1)``. It is not a conditionally negative
    definite function in two dimensions, so the 2-D Kriging system may be indefinite and
    the estimation variance may be clamped to zero. Prefer ``"spherical"``,
    ``"exponential"`` or ``"gaussian"`` for areal interpolation.
    """
    valid = ~np.isnan(sample_points[:, 0]) & ~np.isnan(sample_points[:, 1]) & ~np.isnan(sample_points[:, 2])
    pts = sample_points[valid]

    M, N = len(y_coords), len(x_coords)
    M_grid = M * N

    N_pts = len(pts)
    if N_pts == 0:
        return np.zeros((M, N)), np.zeros((M, N))

    effective_n_cores = (os.cpu_count() or 1) if (n_cores == -1 or n_cores is None) else max(1, int(n_cores))

    x_mean, x_scale = float(np.mean(x_coords)), max(float(np.ptp(x_coords)), 1.0)
    y_mean, y_scale = float(np.mean(y_coords)), max(float(np.ptp(y_coords)), 1.0)

    pts_x_norm = (pts[:, 0] - x_mean) / x_scale
    pts_y_norm = (pts[:, 1] - y_mean) / y_scale

    sample_dists = cdist(pts[:, :2], pts[:, :2])
    max_d = float(np.max(sample_dists)) if N_pts > 1 else 100.0

    if variogram_params is not None:
        if isinstance(variogram_params, dict):
            range_a = float(variogram_params.get("range", max(max_d * 0.6, 1.0)))
            sill = float(variogram_params.get("sill", np.var(pts[:, 2]) if N_pts > 1 else 1.0))
            nugget = float(variogram_params.get("nugget", 0.0))
        else:
            range_a, sill, nugget = variogram_params
    else:
        range_a = max(max_d * 0.6, 1.0)
        sill = float(np.var(pts[:, 2])) if N_pts > 1 else 1.0
        if sill == 0:
            sill = 1.0
        nugget = 0.0

    from .variogram import evaluate_variogram_model

    if str(variogram_model).lower().strip() == "linear":
        logger.warning(
            "   [Variogram Warning] The 'linear' variogram model is a bounded 1-D profile model and is not "
            "positive-definite in 2-D. The 2-D Kriging system may be indefinite and the variance may collapse to 0. "
            "Prefer 'spherical', 'exponential' or 'gaussian' for areal interpolation."
        )

    def variogram_func(h: np.ndarray) -> np.ndarray:
        gamma = evaluate_variogram_model(h, variogram_model, range_a, sill, nugget)
        gamma = np.where(h == 0, 0.0, gamma)
        return gamma

    K_sample = variogram_func(sample_dists)

    method_clean = str(method).lower().strip()
    ext_dict: dict[str, np.ndarray] = {}
    if isinstance(external_drift_grid, dict):
        ext_dict = external_drift_grid
    elif isinstance(external_drift_grid, np.ndarray) and external_drift_grid.shape == (M, N):
        ext_dict = {"external_drift": external_drift_grid}

    use_universal = (method_clean in ["universal", "universal_kriging", "sia"]) or (len(ext_dict) > 0)
    use_universal = use_universal and (N_pts >= 4)

    # Process external drift rasters and sample point values
    ext_drifts: list[tuple[np.ndarray, np.ndarray]] = []  # List of (u_flat, pts_u_norm)
    if use_universal and len(ext_dict) > 0:
        if len(y_coords) > 1 and y_coords[1] < y_coords[0]:
            y_asc = y_coords[::-1]
            flip_y = True
        else:
            y_asc = y_coords
            flip_y = False

        pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))  # (Y, X)

        for _, grid in ext_dict.items():
            if grid is not None and grid.shape == (M, N):
                u_std = float(np.nanstd(grid))
                if u_std < 1e-12:
                    logger.warning("External drift raster has zero spatial variance (flat). Skipping redundant constant drift column.")
                    continue
                u_mean = float(np.nanmean(grid))
                u_grid_norm = (grid - u_mean) / u_std
                u_flat = u_grid_norm.ravel()

                u_asc = u_grid_norm[::-1, :] if flip_y else u_grid_norm
                interp_u = RegularGridInterpolator((y_asc, x_coords), u_asc, bounds_error=False, fill_value=0.0)
                pts_u_norm = interp_u(pts_xy)
                pts_u_norm = np.nan_to_num(pts_u_norm, nan=0.0)
                ext_drifts.append((u_flat, pts_u_norm))

    has_poly_quad = (drift_terms is not None) and ("quadratic_xy" in drift_terms)
    has_poly_lin = (drift_terms is not None) and ("linear_xy" in drift_terms)

    if use_universal:
        n_ext = len(ext_drifts)
        if n_ext > 0:
            n_drift = 1 + n_ext
            if has_poly_quad:
                n_drift += 5  # x, y, x^2, y^2, xy
            elif has_poly_lin:
                n_drift += 2  # x, y
        else:
            if has_poly_lin and not has_poly_quad:
                n_drift = 3  # 1, x, y
            else:
                n_drift = 6  # 1, x, y, x^2, y^2, xy
    else:
        n_drift = 1

    K = np.zeros((N_pts + n_drift, N_pts + n_drift))
    K[:N_pts, :N_pts] = K_sample
    K[:N_pts, N_pts] = 1.0
    K[N_pts, :N_pts] = 1.0

    curr_col = 1
    # Add external drift terms to point matrix K
    for _, pts_u_norm in ext_drifts:
        K[:N_pts, N_pts + curr_col] = pts_u_norm
        K[N_pts + curr_col, :N_pts] = pts_u_norm
        curr_col += 1

    # Add spatial polynomial terms to point matrix K
    if use_universal and (has_poly_lin or has_poly_quad or (len(ext_drifts) == 0 and n_drift > 1)):
        # x, y terms
        K[:N_pts, N_pts + curr_col] = pts_x_norm
        K[N_pts + curr_col, :N_pts] = pts_x_norm
        curr_col += 1

        K[:N_pts, N_pts + curr_col] = pts_y_norm
        K[N_pts + curr_col, :N_pts] = pts_y_norm
        curr_col += 1

        if has_poly_quad or (len(ext_drifts) == 0 and n_drift >= 6):
            K[:N_pts, N_pts + curr_col] = pts_x_norm**2
            K[N_pts + curr_col, :N_pts] = pts_x_norm**2
            curr_col += 1

            K[:N_pts, N_pts + curr_col] = pts_y_norm**2
            K[N_pts + curr_col, :N_pts] = pts_y_norm**2
            curr_col += 1

            K[:N_pts, N_pts + curr_col] = pts_x_norm * pts_y_norm
            K[N_pts + curr_col, :N_pts] = pts_x_norm * pts_y_norm
            curr_col += 1

    reg_val = 1e-6 * max(float(sill), float(np.mean(np.diag(K_sample))))
    K[:N_pts, :N_pts] += np.eye(N_pts) * reg_val
    logger.info(f"   [Dual Kriging Engine] Applied {reg_val:.1e} Tikhonov matrix regularization (N={N_pts} points, n_drift={n_drift})")

    z_aug = np.zeros(N_pts + n_drift, dtype=np.float64)
    z_aug[:N_pts] = pts[:, 2]

    K_inv = None
    try:
        lu_piv = lu_factor(K)
        w_z = lu_solve(lu_piv, z_aug)  # Dual Kriging 1D weight vector
        if return_variance:
            K_inv = lu_solve(lu_piv, np.eye(N_pts + n_drift))
    except Exception:
        w_z = np.linalg.lstsq(K, z_aug, rcond=None)[0]
        if return_variance:
            K_inv = np.linalg.pinv(K)

    xx, yy = np.meshgrid(x_coords, y_coords)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()

    chunk_size = max(500, min(10000, 5000000 // max(N_pts, 1)))
    z_interp_flat = np.zeros(M_grid, dtype=np.float64)
    var_interp_flat = np.full(M_grid, np.nan, dtype=np.float64)

    chunks = [(start_idx, min(start_idx + chunk_size, M_grid)) for start_idx in range(0, M_grid, chunk_size)]

    w_sample = w_z[:N_pts]
    w_drift = w_z[N_pts:]

    def _process_chunk(chunk_tuple: tuple[int, int]) -> tuple[int, int, np.ndarray, np.ndarray]:
        start_idx, end_idx = chunk_tuple
        sub_size = end_idx - start_idx
        sub_x = xx_flat[start_idx:end_idx]
        sub_y = yy_flat[start_idx:end_idx]
        sub_grid_coords = np.column_stack((sub_x, sub_y))

        sub_grid_dists = cdist(pts[:, :2], sub_grid_coords)
        K_grid_sub = variogram_func(sub_grid_dists)

        K_rhs_drift_sub = np.zeros((n_drift, sub_size), dtype=np.float64)
        K_rhs_drift_sub[0, :] = 1.0

        sub_col = 1
        for u_flat, _ in ext_drifts:
            K_rhs_drift_sub[sub_col, :] = u_flat[start_idx:end_idx]
            sub_col += 1

        if use_universal and (has_poly_lin or has_poly_quad or (len(ext_drifts) == 0 and n_drift > 1)):
            sub_x_norm = (sub_x - x_mean) / x_scale
            sub_y_norm = (sub_y - y_mean) / y_scale
            K_rhs_drift_sub[sub_col, :] = sub_x_norm
            sub_col += 1
            K_rhs_drift_sub[sub_col, :] = sub_y_norm
            sub_col += 1

            if has_poly_quad or (len(ext_drifts) == 0 and n_drift >= 6):
                K_rhs_drift_sub[sub_col, :] = sub_x_norm**2
                sub_col += 1
                K_rhs_drift_sub[sub_col, :] = sub_y_norm**2
                sub_col += 1
                K_rhs_drift_sub[sub_col, :] = sub_x_norm * sub_y_norm
                sub_col += 1

        # High-performance Dual Kriging elevation prediction (O(N) 1D dot product)
        z_sub = np.dot(w_sample, K_grid_sub) + np.dot(w_drift, K_rhs_drift_sub)

        if K_inv is None:
            # Conditional variance: skip the O((N+d)^2) per-cell variance GEMM entirely
            return start_idx, end_idx, z_sub, None

        # Estimation variance computation via thread-safe precomputed K_inv
        K_rhs_sub = np.zeros((N_pts + n_drift, sub_size), dtype=np.float64)
        K_rhs_sub[:N_pts, :] = K_grid_sub
        K_rhs_sub[N_pts:, :] = K_rhs_drift_sub

        W_sub = np.dot(K_inv, K_rhs_sub)

        weights_sub = W_sub[:N_pts, :]
        mu_drift_sub = np.sum(W_sub[N_pts:, :] * K_rhs_drift_sub, axis=0)
        var_sub = np.maximum(np.sum(weights_sub * K_grid_sub, axis=0) + mu_drift_sub, 0.0)

        return start_idx, end_idx, z_sub, var_sub

    if len(chunks) > 1 and effective_n_cores > 1:
        with ThreadPoolExecutor(max_workers=effective_n_cores) as executor:
            chunk_results = list(
                get_progress_bar(
                    executor.map(_process_chunk, chunks),
                    total=len(chunks),
                    desc="   [Dual Kriging Engine] Interpolating DEM grid",
                    unit="chunks",
                    disable=not show_progress,
                )
            )
    else:
        chunk_results = [
            _process_chunk(chunk_tuple)
            for chunk_tuple in get_progress_bar(
                chunks,
                desc="   [Dual Kriging Engine] Interpolating DEM grid",
                unit="chunks",
                disable=not show_progress,
            )
        ]

    for start_idx, end_idx, z_sub, var_sub in chunk_results:
        z_interp_flat[start_idx:end_idx] = z_sub
        if var_sub is not None:
            var_interp_flat[start_idx:end_idx] = var_sub

    z_interp = z_interp_flat.reshape((M, N))
    var_interp = var_interp_flat.reshape((M, N))

    return z_interp, var_interp


def pykrige_kriging_interpolation(
    sample_points: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    method: str = "universal",
    variogram_model: str = "spherical",
    external_drift_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Executes 3rd-party PyKrige Ordinary, Universal, or Regression Kriging.
    Requires optional 'pykrige' dependency.
    """
    try:
        import pykrige
    except ImportError:
        raise ImportError(
            "PyKrige package is required for engine='pykrige' or method='regression'. "
            "Please install it via: pip install pysole[pykrige] or pip install pykrige"
        )

    valid = ~np.isnan(sample_points[:, 0]) & ~np.isnan(sample_points[:, 1]) & ~np.isnan(sample_points[:, 2])
    pts = sample_points[valid]

    M, N = len(y_coords), len(x_coords)
    if len(pts) == 0:
        return np.zeros((M, N)), np.zeros((M, N))

    method_clean = str(method).lower().strip()

    if method_clean in ["regression", "regression_kriging"]:
        from pykrige.rk import RegressionKriging
        from sklearn.ensemble import RandomForestRegressor

        rk = RegressionKriging(
            regression_model=RandomForestRegressor(n_estimators=50, random_state=42),
            method="ordinary",
            variogram_model=variogram_model,
        )
        P_train = pts[:, :2]
        z_train = pts[:, 2]
        rk.fit(P_train, P_train, z_train)

        xx, yy = np.meshgrid(x_coords, y_coords)
        P_pred = np.column_stack((xx.ravel(), yy.ravel()))
        z_flat = rk.predict(P_pred, P_pred)
        z_b = z_flat.reshape((M, N))
        v_b = np.zeros((M, N))
        return z_b, v_b

    elif method_clean in ["ordinary", "ordinary_kriging"]:
        from pykrige.ok import OrdinaryKriging

        ok = OrdinaryKriging(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False,
        )
        z_b, v_b = ok.execute("grid", x_coords, y_coords)
        return np.asarray(z_b, dtype=np.float64), np.asarray(v_b, dtype=np.float64)

    else:
        from pykrige.uk import UniversalKriging

        first_grid = None
        if isinstance(external_drift_grid, dict) and len(external_drift_grid) > 0:
            first_grid = next(iter(external_drift_grid.values()))
        elif isinstance(external_drift_grid, np.ndarray) and external_drift_grid.shape == (M, N):
            first_grid = external_drift_grid

        if first_grid is not None and first_grid.shape == (M, N):
            from scipy.interpolate import RegularGridInterpolator

            if len(y_coords) > 1 and y_coords[1] < y_coords[0]:
                y_asc = y_coords[::-1]
                u_asc = first_grid[::-1, :]
            else:
                y_asc = y_coords
                u_asc = first_grid

            interp_u = RegularGridInterpolator((y_asc, x_coords), u_asc, bounds_error=False, fill_value=0.0)
            pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))
            pts_u = interp_u(pts_xy)
            pts_u = np.nan_to_num(pts_u, nan=0.0)

            uk = UniversalKriging(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                variogram_model=variogram_model,
                drift_terms=["specified"],
                specified_drift=[pts_u],
                verbose=False,
                enable_plotting=False,
            )
            z_b, v_b = uk.execute("grid", x_coords, y_coords, specified_drift_data=[first_grid])
        else:
            uk = UniversalKriging(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                variogram_model=variogram_model,
                drift_terms=["regional_linear"],
                verbose=False,
                enable_plotting=False,
            )
            z_b, v_b = uk.execute("grid", x_coords, y_coords)

        return np.asarray(z_b, dtype=np.float64), np.asarray(v_b, dtype=np.float64)


def kriging_interpolation(
    sample_points: np.ndarray,
    geometry: GridGeometry,
    method: str = "universal",
    variogram_model: str = "spherical",
    variogram_params: tuple[float, float, float] | dict[str, float] | None = None,
    dem_grid: np.ndarray | None = None,
    opt_slope_grid: np.ndarray | None = None,
    drift_terms: list[str] | None = None,
    outline_mask: np.ndarray | None = None,
    include_zero_boundary_condition: bool = True,
    n_cores: int = -1,
    engine: str = "native",
    slope_floor_deg: float = 5.0,
    show_progress: bool = True,
    external_drift_grid: np.ndarray | dict[str, np.ndarray] | None = None,
    return_variance: bool = True,
) -> KrigingResult:
    """
    Applies Kriging spatial interpolation on scattered points supporting four distinct approaches:
    1. 'universal' / 'universal_kriging' (Universal Kriging with default quadratic spatial drift)
    2. 'sia' (Shallow Ice Approximation custom physical drift U_sia = 1 / sin(alpha_safe))
    3. 'ordinary' / 'ordinary_kriging' (Ordinary Kriging assuming constant mean)
    4. 'regression' / 'regression_kriging' (Regression Kriging combining ML regressor with residual Kriging via PyKrige)

    When include_zero_boundary_condition is True, enforces zero-value boundary points (T=0 ns or D=0 m)
    along both the outer perimeter and any interior rock outcrop/nunatak margin boundaries.

    Parameters
    ----------
    return_variance : bool, default True
        Native engine only. If False, the O(N^2) per-cell Kriging variance evaluation is skipped and
        ``KrigingResult.variance_grid`` is filled with NaN. Set False when uncertainty maps are not needed.

    Notes
    -----
    ``variogram_model="linear"`` selects the bounded 1-D profile model
    ``gamma(h) = nugget + sill * min(h / range, 1)`` in the native engine. This model is not
    conditionally negative definite in 2-D, so areal Kriging systems may be indefinite and the variance
    may be clamped to zero (a runtime warning is logged). Prefer 'spherical', 'exponential' or 'gaussian'.
    """
    M, N = geometry.shape
    x_coords = geometry.x_coords
    y_coords = geometry.y_coords

    if include_zero_boundary_condition and outline_mask is not None and np.any(outline_mask):
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(outline_mask, border_value=1)
        boundary_mask = outline_mask & ~eroded

        b_indices = np.argwhere(boundary_mask)  # (row, col)
        if len(b_indices) > 0:
            stride = max(1, len(b_indices) // 100)
            sub_indices = b_indices[::stride]
            b_x = x_coords[sub_indices[:, 1]]
            b_y = y_coords[sub_indices[:, 0]]
            b_val = np.zeros(len(b_x))
            n_cols = sample_points.shape[1] if sample_points.ndim > 1 else 3
            if n_cols > 3:
                b_pts = np.zeros((len(b_x), n_cols), dtype=np.float64)
                b_pts[:, 0] = b_x
                b_pts[:, 1] = b_y
                b_pts[:, 2] = b_val
            else:
                b_pts = np.column_stack((b_x, b_y, b_val))
            sample_points = np.vstack([sample_points, b_pts])

    valid = ~np.isnan(sample_points[:, 0]) & ~np.isnan(sample_points[:, 1]) & ~np.isnan(sample_points[:, 2])
    pts = sample_points[valid]

    if len(pts) == 0:
        return KrigingResult(bedrock_grid=np.zeros((M, N)), variance_grid=np.zeros((M, N)))

    if isinstance(engine, str):
        engine_clean = str(engine).lower().replace("_kriging", "").strip()
    else:
        engine_clean = "native"
    var_model_clean = str(variogram_model).lower().strip()

    method_clean = str(method).lower().replace("_kriging", "").strip()
    valid_methods = {"ordinary", "universal", "sia", "regression"}
    if method_clean not in valid_methods:
        raise ValueError(
            f"Invalid interpolation method '{method}'. Supported methods: 'ordinary', 'universal', 'sia', 'regression'."
        )

    valid_engines = {"native", "pykrige"}
    if engine_clean not in valid_engines:
        raise ValueError(
            f"Invalid Kriging engine '{engine}'. Supported engines: 'native', 'pykrige'."
        )

    valid_variogram_models = {"spherical", "exponential", "gaussian", "linear"}
    if var_model_clean not in valid_variogram_models:
        raise ValueError(
            f"Invalid variogram model '{variogram_model}'. Supported models: 'spherical', 'exponential', 'gaussian', 'linear'."
        )

    external_drift_grids: dict[str, np.ndarray] = {}
    if isinstance(external_drift_grid, dict):
        external_drift_grids.update(external_drift_grid)
    elif isinstance(external_drift_grid, np.ndarray) and external_drift_grid.shape == (M, N):
        external_drift_grids["external_drift"] = external_drift_grid

    expanded_primitives: set[str] = set()
    if drift_terms:
        for term in drift_terms:
            if term == "sia_space":
                expanded_primitives.update(["sia", "linear_xy"])
            elif term == "sia_z_dem":
                expanded_primitives.update(["sia", "z_dem"])
            elif term == "sia_curvature_dem":
                expanded_primitives.update(["sia", "curvature_dem"])
            elif term == "z_dem_curvature_dem":
                expanded_primitives.update(["z_dem", "curvature_dem"])
            elif term in ["sia_z_dem_curvature_dem", "full_physical"]:
                expanded_primitives.update(["sia", "z_dem", "curvature_dem"])
            elif term == "full_spatial_physical":
                expanded_primitives.update(["sia", "z_dem", "curvature_dem", "linear_xy"])
            elif term in DriftBasis.SUPPORTED_TERMS:
                expanded_primitives.add(term)
            else:
                raise ValueError(f"Unknown drift term '{term}'. Supported terms: {sorted(DriftBasis.SUPPORTED_TERMS)}")

    if method_clean == "sia":
        expanded_primitives.add("sia")

    is_sia_mode = "sia" in expanded_primitives
    is_z_dem_mode = "z_dem" in expanded_primitives
    is_curvature_mode = "curvature_dem" in expanded_primitives

    # Compute external drift grids (SIA 1/sin(alpha), DEM elevation z_dem, surface curvature_dem)
    if is_sia_mode and "sia" not in external_drift_grids:
        if opt_slope_grid is not None and opt_slope_grid.shape == (M, N):
            opt_slope_sin = np.sin(opt_slope_grid)
        elif dem_grid is not None and dem_grid.shape == (M, N):
            grads = compute_gradients(dem_grid, dx=geometry.dx, dy=geometry.dy)
            opt_slope_sin = np.sin(grads["slope_rad"])
        else:
            opt_slope_sin = None

        if opt_slope_sin is not None:
            min_slope_sin = np.sin(np.radians(slope_floor_deg))
            safe_slope_grid = np.maximum(opt_slope_sin, min_slope_sin)
            external_drift_grids["sia"] = 1.0 / safe_slope_grid
        else:
            raise ValueError("Universal Kriging with 'sia' drift requested, but neither surface DEM grid nor slope grid was provided.")

    if is_z_dem_mode and "z_dem" not in external_drift_grids:
        if dem_grid is not None and dem_grid.shape == (M, N):
            external_drift_grids["z_dem"] = dem_grid
        else:
            raise ValueError("Universal Kriging with 'z_dem' elevation drift requested, but surface DEM grid was not provided.")

    if is_curvature_mode and "curvature_dem" not in external_drift_grids:
        if dem_grid is not None and dem_grid.shape == (M, N):
            external_drift_grids["curvature_dem"] = compute_surface_curvature(dem_grid, dx=geometry.dx, dy=geometry.dy)
        else:
            raise ValueError("Universal Kriging with 'curvature_dem' drift requested, but surface DEM grid was not provided.")

    if method_clean in ["universal", "universal_kriging"] and drift_terms is not None and len(drift_terms) == 0 and not external_drift_grids:
        raise ValueError("Universal Kriging requested with empty drift_terms and no external drift grids.")

    has_ext_drifts = len(external_drift_grids) > 0

    if engine_clean in ["pykrige"] or method_clean in ["regression", "regression_kriging"]:
        z_b, v_b = pykrige_kriging_interpolation(
            pts,
            x_coords,
            y_coords,
            method="sia" if has_ext_drifts else method_clean,
            variogram_model=variogram_model,
            external_drift_grid=external_drift_grids if has_ext_drifts else None,
        )
    else:
        # Fast, robust native vector Kriging engine
        z_b, v_b = built_in_kriging_interpolation(
            pts,
            x_coords,
            y_coords,
            method="sia" if has_ext_drifts else method_clean,
            variogram_model=variogram_model,
            external_drift_grid=external_drift_grids if has_ext_drifts else None,
            drift_terms=list(expanded_primitives) if expanded_primitives else drift_terms,
            variogram_params=variogram_params,
            n_cores=n_cores,
            show_progress=show_progress,
            return_variance=return_variance,
        )
    return KrigingResult(bedrock_grid=z_b, variance_grid=v_b)


def random_forest_hole_filling(
    dem: np.ndarray,
    bedrock_grid: np.ndarray,
    boundary_mask: np.ndarray,
    geometry: GridGeometry,
    n_cores: int = -1,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Random Forest Machine Learning model for intelligent bedrock hole filling.
    Trains on known bedrock points inside boundary_mask using spatial features (X, Y, DEM elevation, surface slope),
    and predicts bedrock elevation across remaining unmeasured or missing data regions.

    Parameters
    ----------
    dem : 2D np.ndarray
        Surface elevation grid (M x N).
    bedrock_grid : 2D np.ndarray
        Kriging interpolated bedrock elevation grid (M x N).
    boundary_mask : 2D np.ndarray
        Boolean grid of active creeping body area.
    geometry : GridGeometry
        Standardized spatial geometry container for raster grids.
    n_cores : int
        Number of CPU cores for parallelized tree fitting (-1 for all available cores).
    show_progress : bool
        If True (default), displays progress bar during tree fitting and gap prediction.

    Returns
    -------
    filled_bedrock : 2D np.ndarray
        Seamless bedrock elevation grid with remaining holes filled by Random Forest ML predictions.
    """
    # [VECTORIZATION OPTION 4]: Use cached 2D spatial meshgrid from geometry
    xx, yy = geometry.meshgrid

    from .smoothing import compute_gradients

    grads = compute_gradients(dem, dx=geometry.dx, dy=geometry.dy)
    slope_grid = grads["slope_rad"]

    dist_from_margin = distance_transform_edt(boundary_mask, sampling=(abs(geometry.dy), abs(geometry.dx)))
    dist_flat = dist_from_margin.ravel()
    margin_threshold = max(geometry.dx, geometry.dy) * 2.0

    features = np.column_stack((xx.ravel(), yy.ravel(), dem.ravel(), slope_grid.ravel()))
    bedrock_flat = bedrock_grid.ravel()
    mask_flat = boundary_mask.ravel()

    thickness_target = dem.ravel() - bedrock_flat
    valid_train = mask_flat & ~np.isnan(bedrock_flat) & (thickness_target > 0.1) & (dist_flat > margin_threshold)

    n_valid = int(np.sum(valid_train))
    if n_valid < 10:
        return bedrock_grid.copy()

    train_indices = np.where(valid_train)[0]
    if n_valid > 20000:
        rng = np.random.default_rng(42)
        train_indices = rng.choice(train_indices, size=20000, replace=False)

    effective_n_cores = (os.cpu_count() or 1) if (n_cores == -1 or n_cores is None) else max(1, int(n_cores))

    with get_progress_bar(
        total=100,
        desc="   [RF Gap Filling] Training estimator & predicting gaps",
        unit="trees",
        disable=not show_progress,
    ) as pbar:
        rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=effective_n_cores)
        rf.fit(features[train_indices], thickness_target[train_indices])
        pbar.update(50)

        logger.info("   [RF Gap Filling] Finished learning Random Forest regression model")

        # Predict ice thickness strictly on missing interior data holes, excluding the margin ring
        holes_flat = mask_flat & (np.isnan(bedrock_flat) | (thickness_target <= 0.1)) & (dist_flat > margin_threshold)
        filled_bedrock = bedrock_grid.copy()

        if np.any(holes_flat):
            hole_features = features[holes_flat]
            predicted_thickness = np.maximum(rf.predict(hole_features), 0.0)
            filled_bedrock.ravel()[holes_flat] = dem.ravel()[holes_flat] - predicted_thickness

        pbar.update(50)

    filled_bedrock = np.minimum(filled_bedrock, dem)
    return filled_bedrock


class DriftBasis:
    """
    Unified spatial drift basis evaluator and pre-calculated feature cache.
    Computes drift covariates across spatial coordinates or whole DEM grids,
    maintaining fixed global standardization statistics (mean, scale) per feature column.
    """
    SUPPORTED_TERMS = {
        "linear_xy", "quadratic_xy", "z_dem", "sia", "curvature_dem",
        "sia_space", "sia_z_dem", "sia_curvature_dem", "z_dem_curvature_dem",
        "sia_z_dem_curvature_dem", "full_physical", "full_spatial_physical"
    }

    def __init__(
        self,
        drift_terms: list[str],
        x_ref: np.ndarray,
        y_ref: np.ndarray,
        dem_grid: np.ndarray | None = None,
        dx: float = 10.0,
        dy: float = 10.0,
        bounds: tuple[float, float, float, float] | None = None,
        alpha_opt_deg: np.ndarray | None = None,
        slope_floor_deg: float = 5.0,
    ):
        self.drift_terms = drift_terms or []
        self.dx = dx
        self.dy = dy
        self.bounds = bounds
        self.slope_floor_deg = slope_floor_deg

        primitives: set[str] = set()
        for term in self.drift_terms:
            if term not in self.SUPPORTED_TERMS:
                raise ValueError(f"Unknown drift term '{term}'. Supported terms: {sorted(self.SUPPORTED_TERMS)}")
            if term == "sia_space":
                primitives.update(["sia", "linear_xy"])
            elif term == "sia_z_dem":
                primitives.update(["sia", "z_dem"])
            elif term == "sia_curvature_dem":
                primitives.update(["sia", "curvature_dem"])
            elif term == "z_dem_curvature_dem":
                primitives.update(["z_dem", "curvature_dem"])
            elif term in ["sia_z_dem_curvature_dem", "full_physical"]:
                primitives.update(["sia", "z_dem", "curvature_dem"])
            elif term == "full_spatial_physical":
                primitives.update(["sia", "z_dem", "curvature_dem", "linear_xy"])
            else:
                primitives.add(term)

        self.primitives = primitives

        # Precompute DEM curvature & SIA grids ONCE if dem_grid is provided
        self.curvature_grid = None
        self.sia_grid = None
        if dem_grid is not None:
            if "curvature_dem" in primitives:
                from .smoothing import compute_surface_curvature
                self.curvature_grid = compute_surface_curvature(dem_grid, dx=dx, dy=dy)
            if "sia" in primitives and alpha_opt_deg is not None:
                sin_a = np.sin(np.radians(np.maximum(alpha_opt_deg, slope_floor_deg)))
                self.sia_grid = 1.0 / np.maximum(sin_a, 1e-3)

        x_ref = np.asarray(x_ref, dtype=np.float64)
        y_ref = np.asarray(y_ref, dtype=np.float64)

        self.means: list[float] = []
        self.scales: list[float] = []
        self.evaluators: list[Any] = []

        if "linear_xy" in primitives:
            m_x, s_x = float(np.mean(x_ref)), max(float(np.ptp(x_ref)), 1.0)
            m_y, s_y = float(np.mean(y_ref)), max(float(np.ptp(y_ref)), 1.0)
            self.means.extend([m_x, m_y])
            self.scales.extend([s_x, s_y])
            self.evaluators.extend([
                lambda x, y: x,
                lambda x, y: y,
            ])

        if "quadratic_xy" in primitives:
            m_x, s_x = float(np.mean(x_ref)), max(float(np.ptp(x_ref)), 1.0)
            m_y, s_y = float(np.mean(y_ref)), max(float(np.ptp(y_ref)), 1.0)
            self.means.extend([0.0, 0.0, 0.0])
            self.scales.extend([1.0, 1.0, 1.0])
            self.evaluators.extend([
                lambda x, y, mx=m_x, sx=s_x: ((x - mx) / sx) ** 2,
                lambda x, y, my=m_y, sy=s_y: ((y - my) / sy) ** 2,
                lambda x, y, mx=m_x, sx=s_x, my=m_y, sy=s_y: ((x - mx) / sx) * ((y - my) / sy),
            ])

        if "z_dem" in primitives and dem_grid is not None and bounds is not None:
            geom = GridGeometry.create(dem_grid.shape, dx=dx, dy=dy, bounds=bounds)
            r, c = geom.coords_to_grid_indices(x_ref, y_ref)
            z_vals = dem_grid[r, c]
            m_z, s_z = float(np.nanmean(z_vals)), max(float(np.nanstd(z_vals)), 1e-6)
            self.means.append(m_z)
            self.scales.append(s_z)
            self.evaluators.append(lambda x, y: dem_grid[geom.coords_to_grid_indices(x, y)])

        if "sia" in primitives and self.sia_grid is not None and bounds is not None:
            geom = GridGeometry.create(dem_grid.shape, dx=dx, dy=dy, bounds=bounds)
            r, c = geom.coords_to_grid_indices(x_ref, y_ref)
            sia_vals = self.sia_grid[r, c]
            m_sia, s_sia = float(np.nanmean(sia_vals)), max(float(np.nanstd(sia_vals)), 1e-6)
            self.means.append(m_sia)
            self.scales.append(s_sia)
            self.evaluators.append(lambda x, y: self.sia_grid[geom.coords_to_grid_indices(x, y)])

        if "curvature_dem" in primitives and self.curvature_grid is not None and bounds is not None:
            geom = GridGeometry.create(dem_grid.shape, dx=dx, dy=dy, bounds=bounds)
            r, c = geom.coords_to_grid_indices(x_ref, y_ref)
            curv_vals = self.curvature_grid[r, c]
            m_curv, s_curv = float(np.nanmean(curv_vals)), max(float(np.nanstd(curv_vals)), 1e-6)
            self.means.append(m_curv)
            self.scales.append(s_curv)
            self.evaluators.append(lambda x, y: self.curvature_grid[geom.coords_to_grid_indices(x, y)])

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluates and standardizes drift basis matrix for query coordinates (x, y)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_pts = len(x)
        if not self.evaluators:
            return np.zeros((n_pts, 0))

        cols = []
        for fn, mean_val, scale_val in zip(self.evaluators, self.means, self.scales):
            raw_v = fn(x, y)
            cols.append((raw_v - mean_val) / scale_val)
        return np.column_stack(cols)





class DualKrigingSolver:
    """
    Fast Dual Kriging solver used for point validation in DriftAnalyzer cross-validation.
    """
    def __init__(
        self,
        x_pts: np.ndarray,
        y_pts: np.ndarray,
        z_pts: np.ndarray,
        variogram_model: str = "spherical",
        nugget: float = 0.0,
        sill: float = 1.0,
        range_param: float = 100.0,
        drift_terms: list[str] | None = None,
        drift_basis: DriftBasis | None = None,
        dem_grid: np.ndarray | None = None,
        dx: float = 5.0,
        dy: float = 5.0,
        bounds: tuple[float, float, float, float] | None = None,
        alpha_opt_deg: np.ndarray | None = None,
    ):
        self.x_pts = np.asarray(x_pts, dtype=np.float64)
        self.y_pts = np.asarray(y_pts, dtype=np.float64)
        self.z_pts = np.asarray(z_pts, dtype=np.float64)
        self.variogram_model = str(variogram_model).lower()
        self.nugget = float(nugget)
        self.sill = float(sill) if sill > 0 else 1.0
        self.range_param = float(range_param) if range_param > 0 else 100.0
        self.drift_terms = drift_terms or []

        if drift_basis is not None:
            self.drift_basis = drift_basis
        elif self.drift_terms:
            self.drift_basis = DriftBasis(
                drift_terms=self.drift_terms,
                x_ref=self.x_pts,
                y_ref=self.y_pts,
                dem_grid=dem_grid,
                dx=dx,
                dy=dy,
                bounds=bounds,
                alpha_opt_deg=alpha_opt_deg,
            )
        else:
            self.drift_basis = None

        N = len(self.x_pts)
        pts_xy = np.column_stack((self.x_pts, self.y_pts))
        dists = cdist(pts_xy, pts_xy)

        if "exp" in self.variogram_model:
            K = self.nugget + self.sill * (1.0 - np.exp(-3.0 * dists / max(self.range_param, 1e-6)))
        elif "gauss" in self.variogram_model:
            K = self.nugget + self.sill * (1.0 - np.exp(-3.0 * (dists / max(self.range_param, 1e-6))**2))
        elif "lin" in self.variogram_model:
            K = self.nugget + self.sill * np.clip(dists / max(self.range_param, 1e-6), 0.0, 1.0)
        else:
            h_ratio = np.clip(dists / max(self.range_param, 1e-6), 0.0, 1.0)
            gamma = self.sill * (1.5 * h_ratio - 0.5 * (h_ratio**3))
            gamma[dists > self.range_param] = self.sill
            K = self.nugget + gamma

        np.fill_diagonal(K, 0.0)

        F_list = [np.ones((N, 1))]
        if self.drift_basis is not None:
            F_drift = self.drift_basis.evaluate(self.x_pts, self.y_pts)
            if F_drift.shape[1] > 0:
                F_list.append(F_drift)

        F = np.hstack(F_list)
        n_drift = F.shape[1]

        A = np.zeros((N + n_drift, N + n_drift))
        A[:N, :N] = K + np.eye(N) * 1e-8
        A[:N, N:] = F
        A[N:, :N] = F.T

        rhs = np.zeros(N + n_drift)
        rhs[:N] = self.z_pts

        try:
            sol = np.linalg.solve(A, rhs)
            self.b = sol[:N]
            self.a = sol[N:]
            self.success = True
        except Exception:
            self.success = False

    def predict(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        dem_grid: np.ndarray | None = None,
        dx: float = 5.0,
        dy: float = 5.0,
        bounds: tuple[float, float, float, float] | None = None,
        alpha_opt_deg: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.success:
            return np.full_like(x_val, np.nan)

        x_val = np.asarray(x_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)
        N_val = len(x_val)

        val_xy = np.column_stack((x_val, y_val))
        pts_xy = np.column_stack((self.x_pts, self.y_pts))
        dists = cdist(val_xy, pts_xy)

        if "exp" in self.variogram_model:
            K_val = self.nugget + self.sill * (1.0 - np.exp(-3.0 * dists / max(self.range_param, 1e-6)))
        elif "gauss" in self.variogram_model:
            K_val = self.nugget + self.sill * (1.0 - np.exp(-3.0 * (dists / max(self.range_param, 1e-6))**2))
        elif "lin" in self.variogram_model:
            K_val = self.nugget + self.sill * np.clip(dists / max(self.range_param, 1e-6), 0.0, 1.0)
        else:
            h_ratio = np.clip(dists / max(self.range_param, 1e-6), 0.0, 1.0)
            gamma = self.sill * (1.5 * h_ratio - 0.5 * (h_ratio**3))
            gamma[dists > self.range_param] = self.sill
            K_val = self.nugget + gamma

        F_list = [np.ones((N_val, 1))]
        if self.drift_basis is not None:
            F_drift = self.drift_basis.evaluate(x_val, y_val)
            if F_drift.shape[1] > 0:
                F_list.append(F_drift)

        F_val = np.hstack(F_list)
        return K_val @ self.b + F_val @ self.a
