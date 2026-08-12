"""
Interpolation, Margin Blending, Kriging, and Random Forest Hole Filling.
Ported from MATLAB script INTERPOL.m by Daniel Binder (2011) and PySole modern workflow.
Supports Ordinary Kriging, Universal Kriging (default quadratic drift & SIA custom drift), and Regression Kriging.
"""

from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator, RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial.distance import cdist


def blend_margin_topography(
    dem: np.ndarray,
    bedrock_input: Union[np.ndarray, Tuple[np.ndarray, ...]],
    boundary_mask: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
    min_gap_dist: Optional[float] = None,
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
    dx : float
        Grid spacing along X.
    dy : float
        Grid spacing along Y.
    min_gap_dist : float, optional
        Minimum gap distance / margin blend zone width [m]. Prunes bedrock within this distance from the margin.

    Returns
    -------
    blended_dem : 2D np.ndarray
        Harmonized continuous bedrock elevation grid.
    """
    M, N = dem.shape
    if x_coords is None:
        x_coords = np.arange(N) * dx
    if y_coords is None:
        y_coords = np.arange(M) * dy

    cellsize = (dx + dy) / 2.0
    if min_gap_dist is None or min_gap_dist <= 0:
        margin_width = 3.0 * cellsize
    else:
        margin_width = float(min_gap_dist)

    xx, yy = np.meshgrid(x_coords, y_coords)

    # Handle 2D bedrock elevation grid input
    if isinstance(bedrock_input, np.ndarray) and bedrock_input.shape == (M, N):
        bedrock_grid = bedrock_input.copy()
        thickness = dem - bedrock_grid
        thickness[~boundary_mask] = 0.0

        # Distance transform to boundary margin (in meters)
        dist_from_margin = distance_transform_edt(boundary_mask) * cellsize

        # Smooth taper weight over minimum gap distance: 0 at boundary edge, 1 inside body
        weight = np.clip(dist_from_margin / max(margin_width, 1e-6), 0.0, 1.0)
        weight = 0.5 * (1.0 - np.cos(np.pi * weight))  # smooth cosine transition

        # Prune bedrock thickness within minimum gap distance
        tapered_thickness = thickness * weight
        smoothed_thickness = gaussian_filter(tapered_thickness, sigma=1.0)
        final_thickness = np.where(weight > 0.8, thickness, smoothed_thickness)
        final_thickness[~boundary_mask] = 0.0

        return dem - final_thickness

    # Handle point array input
    pts = np.array(bedrock_input, dtype=np.float64)
    px = pts[:, 0]
    py = pts[:, 1]

    # Interpolate bedrock depths at survey points
    if pts.shape[1] >= 4:
        depths = pts[:, 3]
    else:
        depths = dem - pts[:, 2]

    # Radial Basis Function (RBF) thin-plate spline interpolation for smooth bedrock topography
    try:
        rbf = RBFInterpolator(np.column_stack((px, py)), depths, kernel="thin_plate_spline", smoothing=0.1)
        grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
        thickness_grid = rbf(grid_pts).reshape((M, N))
    except Exception:
        thickness_grid = griddata((px, py), depths, (xx, yy), method="cubic", fill_value=0.0)

    thickness_grid = np.maximum(np.nan_to_num(thickness_grid, nan=0.0), 0.0)
    thickness_grid[~boundary_mask] = 0.0

    dist_from_margin = distance_transform_edt(boundary_mask) * cellsize
    weight = np.clip(dist_from_margin / max(margin_width, 1e-6), 0.0, 1.0)
    weight = 0.5 * (1.0 - np.cos(np.pi * weight))

    tapered_thickness = thickness_grid * weight
    smoothed_thickness = gaussian_filter(tapered_thickness, sigma=1.0)
    final_thickness = np.where(weight > 0.8, thickness_grid, smoothed_thickness)
    final_thickness[~boundary_mask] = 0.0

    return dem - final_thickness


def _built_in_kriging(
    sample_points: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    method: str = "universal",
    variogram_model: str = "spherical",
    external_drift_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust native NumPy/SciPy Ordinary & Universal Kriging solver with zero-centered
    spatial coordinate normalization and diagonal regularization to prevent ill-conditioned matrix explosion.
    Supports default quadratic spatial drift terms (1, x, y, x^2, y^2, x*y) and SIA custom external drift (1, U_sia).
    """
    valid = ~np.isnan(sample_points[:, 0]) & ~np.isnan(sample_points[:, 1]) & ~np.isnan(sample_points[:, 2])
    pts = sample_points[valid]

    M, N = len(y_coords), len(x_coords)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_coords = np.column_stack((xx.ravel(), yy.ravel()))
    M_grid = len(grid_coords)

    N_pts = len(pts)
    if N_pts == 0:
        return np.zeros((M, N)), np.zeros((M, N))

    # Zero-center spatial drift coordinates relative to grid origin to prevent float ill-conditioning
    x0, y0 = x_coords[0], y_coords[0]
    pts_x_norm = pts[:, 0] - x0
    pts_y_norm = pts[:, 1] - y0
    grid_x_norm = grid_coords[:, 0] - x0
    grid_y_norm = grid_coords[:, 1] - y0

    # Estimate empirical variogram parameters
    sample_dists = cdist(pts[:, :2], pts[:, :2])
    max_d = float(np.max(sample_dists)) if N_pts > 1 else 100.0
    range_a = max(max_d * 0.6, 1.0)
    sill = float(np.var(pts[:, 2])) if N_pts > 1 else 1.0
    if sill == 0:
        sill = 1.0
    nugget = 0.0

    def variogram_func(h):
        h_ratio = np.clip(h / max(range_a, 1e-6), 0.0, 1.0)
        v_model = variogram_model.lower()
        if "exp" in v_model:
            gamma = nugget + sill * (1.0 - np.exp(-3.0 * h / max(range_a, 1e-6)))
        elif "gauss" in v_model:
            gamma = nugget + sill * (1.0 - np.exp(-3.0 * (h / max(range_a, 1e-6))**2))
        elif "lin" in v_model:
            gamma = nugget + sill * np.clip(h / max(range_a, 1e-6), 0.0, 1.0)
        else:  # spherical
            gamma = nugget + sill * (1.5 * h_ratio - 0.5 * (h_ratio**3))
            gamma = np.where(h > range_a, nugget + sill, gamma)
        gamma = np.where(h == 0, 0.0, gamma)
        return gamma

    K_sample = variogram_func(sample_dists)

    method_clean = str(method).lower().strip()
    is_sia_mode = (method_clean in ["sia_thickness", "sia", "sia_drift"]) or (external_drift_grid is not None)
    use_universal = (method_clean in ["universal", "universal_kriging", "sia_thickness", "sia", "sia_drift"]) and (N_pts >= 4)

    if use_universal:
        if is_sia_mode and external_drift_grid is not None and external_drift_grid.shape == (M, N):
            # Standardize external SIA drift grid U = 1 / sin(alpha_safe)
            u_mean = float(np.mean(external_drift_grid))
            u_std = float(np.std(external_drift_grid))
            u_std = max(u_std, 1e-6)
            u_grid_norm = (external_drift_grid - u_mean) / u_std

            interp_u = RegularGridInterpolator((y_coords, x_coords), u_grid_norm, bounds_error=False, fill_value=0.0)
            pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))  # (Y, X)
            pts_u_norm = interp_u(pts_xy)
            pts_u_norm = np.nan_to_num(pts_u_norm, nan=0.0)

            # SIA External Drift (1, U_sia) -> 2 drift terms
            n_drift = 2
            K = np.zeros((N_pts + n_drift, N_pts + n_drift))
            K[:N_pts, :N_pts] = K_sample
            K[:N_pts, N_pts] = 1.0
            K[:N_pts, N_pts + 1] = pts_u_norm

            K[N_pts, :N_pts] = 1.0
            K[N_pts + 1, :N_pts] = pts_u_norm

            K_rhs_drift = np.zeros((n_drift, M_grid))
            K_rhs_drift[0, :] = 1.0
            K_rhs_drift[1, :] = u_grid_norm.ravel()
        else:
            # Default quadratic spatial drift terms (1, x, y, x^2, y^2, x*y)
            n_drift = 6
            K = np.zeros((N_pts + n_drift, N_pts + n_drift))
            K[:N_pts, :N_pts] = K_sample
            K[:N_pts, N_pts] = 1.0
            K[:N_pts, N_pts + 1] = pts_x_norm
            K[:N_pts, N_pts + 2] = pts_y_norm
            K[:N_pts, N_pts + 3] = pts_x_norm**2
            K[:N_pts, N_pts + 4] = pts_y_norm**2
            K[:N_pts, N_pts + 5] = pts_x_norm * pts_y_norm

            K[N_pts, :N_pts] = 1.0
            K[N_pts + 1, :N_pts] = pts_x_norm
            K[N_pts + 2, :N_pts] = pts_y_norm
            K[N_pts + 3, :N_pts] = pts_x_norm**2
            K[N_pts + 4, :N_pts] = pts_y_norm**2
            K[N_pts + 5, :N_pts] = pts_x_norm * pts_y_norm

            K_rhs_drift = np.zeros((n_drift, M_grid))
            K_rhs_drift[0, :] = 1.0
            K_rhs_drift[1, :] = grid_x_norm
            K_rhs_drift[2, :] = grid_y_norm
            K_rhs_drift[3, :] = grid_x_norm**2
            K_rhs_drift[4, :] = grid_y_norm**2
            K_rhs_drift[5, :] = grid_x_norm * grid_y_norm
    else:
        n_drift = 1
        K = np.zeros((N_pts + n_drift, N_pts + n_drift))
        K[:N_pts, :N_pts] = K_sample
        K[:N_pts, N_pts] = 1.0
        K[N_pts, :N_pts] = 1.0

        K_rhs_drift = np.zeros((n_drift, M_grid))
        K_rhs_drift[0, :] = 1.0

    # Regularize diagonal to guarantee numerical stability
    K[:N_pts, :N_pts] += np.eye(N_pts) * 1e-8

    grid_dists = cdist(pts[:, :2], grid_coords)
    K_grid = variogram_func(grid_dists)

    K_rhs = np.zeros((N_pts + n_drift, M_grid))
    K_rhs[:N_pts, :] = K_grid
    K_rhs[N_pts:, :] = K_rhs_drift

    try:
        W = np.linalg.solve(K, K_rhs)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(K, K_rhs, rcond=None)[0]

    weights = W[:N_pts, :]
    z_vals = pts[:, 2]
    z_interp = np.dot(z_vals, weights).reshape((M, N))

    mu = W[N_pts, :]
    var_interp = (np.sum(weights * K_grid, axis=0) + mu).reshape((M, N))
    var_interp = np.maximum(var_interp, 0.0)

    return z_interp, var_interp


def _is_valid_kriging_output(z_grid: np.ndarray, pts: np.ndarray) -> bool:
    """
    Sanity check to detect PyKrige ill-conditioned matrix value explosions.
    """
    if z_grid is None or not np.all(np.isfinite(z_grid)):
        return False
    val_range = float(np.ptp(pts[:, 2]))
    val_mean = float(np.mean(pts[:, 2]))
    max_allowed = max(abs(val_mean) + 20.0 * max(val_range, 1.0), 1e4)
    min_allowed = min(val_mean - 20.0 * max(val_range, 1.0), -1e4)

    grid_min = float(np.min(z_grid))
    grid_max = float(np.max(z_grid))
    return (grid_min >= min_allowed) and (grid_max <= max_allowed)


def kriging_interpolation(
    sample_points: np.ndarray,
    target_grid_shape: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
    method: str = "universal",
    variogram_model: str = "spherical",
    dem_grid: Optional[np.ndarray] = None,
    opt_slope_grid: Optional[np.ndarray] = None,
    drift_terms: Optional[List[str]] = None,
    outline_mask: Optional[np.ndarray] = None,
    include_zero_boundary_condition: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Kriging spatial interpolation on scattered points supporting four distinct approaches:
    1. 'universal' / 'universal_kriging' (Universal Kriging with default quadratic spatial drift)
    2. 'sia_thickness' / 'sia' (Shallow Ice Approximation custom physical drift U_sia = 1 / sin(alpha_safe))
    3. 'ordinary' / 'ordinary_kriging' (Ordinary Kriging assuming constant mean)
    4. 'regression' / 'regression_kriging' (Regression Kriging combining ML regressor with residual Kriging)

    Includes native NumPy/SciPy Kriging solver fallback so exact Kriging interpolation is always computed
    even if pykrige is not installed or produces ill-conditioned matrix explosions.
    """
    M, N = target_grid_shape
    x_coords = np.linspace(bounds[0], bounds[2], N)
    y_coords = np.linspace(bounds[1], bounds[3], M)

    if include_zero_boundary_condition and outline_mask is not None and np.any(outline_mask):
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(outline_mask)
        boundary_mask = outline_mask & ~eroded

        b_indices = np.argwhere(boundary_mask)  # (row, col)
        if len(b_indices) > 0:
            stride = max(1, len(b_indices) // 100)
            sub_indices = b_indices[::stride]
            b_x = x_coords[sub_indices[:, 1]]
            b_y = y_coords[sub_indices[:, 0]]
            b_val = np.zeros(len(b_x))
            b_pts = np.column_stack((b_x, b_y, b_val))
            sample_points = np.vstack([sample_points, b_pts])

    valid = ~np.isnan(sample_points[:, 0]) & ~np.isnan(sample_points[:, 1]) & ~np.isnan(sample_points[:, 2])
    pts = sample_points[valid]

    if len(pts) == 0:
        return np.zeros((M, N)), np.zeros((M, N))

    method_clean = str(method).lower().replace("_kriging", "").strip()
    is_sia_mode = (method_clean in ["sia_thickness", "sia", "sia_drift"]) or (drift_terms is not None and "sia_thickness" in drift_terms)

    # Compute external SIA drift grid U_sia = 1 / sin(alpha_safe) if requested
    external_sia_grid = None
    if is_sia_mode:
        if opt_slope_grid is not None and opt_slope_grid.shape == (M, N):
            opt_slope_sin = np.sin(opt_slope_grid)
        elif dem_grid is not None and dem_grid.shape == (M, N):
            from .smoothing import compute_gradients

            grads = compute_gradients(dem_grid, dx=(bounds[2] - bounds[0]) / float(N), dy=(bounds[3] - bounds[1]) / float(M))
            opt_slope_sin = np.sin(grads["slope_rad"])
        else:
            opt_slope_sin = None

        if opt_slope_sin is not None:
            min_slope_sin = np.sin(np.radians(2.0))
            safe_slope_grid = np.maximum(opt_slope_sin, min_slope_sin)
            external_sia_grid = 1.0 / safe_slope_grid

    # --- 1. SIA Custom Functional Drift or Universal Kriging ---
    if is_sia_mode or (method_clean == "universal"):
        try:
            from pykrige.uk import UniversalKriging

            if is_sia_mode and external_sia_grid is not None:
                # Standardize external SIA drift grid
                u_mean = float(np.mean(external_sia_grid))
                u_std = float(np.std(external_sia_grid))
                u_std = max(u_std, 1e-6)
                u_grid_norm = (external_sia_grid - u_mean) / u_std

                interp_u = RegularGridInterpolator((y_coords, x_coords), u_grid_norm, bounds_error=False, fill_value=0.0)
                pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))  # (Y, X)
                pts_u_norm = interp_u(pts_xy)
                pts_u_norm = np.nan_to_num(pts_u_norm, nan=0.0)

                UK = UniversalKriging(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    variogram_model=variogram_model,
                    drift_terms=["specified"],
                    specified_drift=[pts_u_norm],
                    verbose=False,
                    enable_plotting=False,
                )
                z_krig, ss_krig = UK.execute("grid", x_coords, y_coords, specified_drift_arrays=[u_grid_norm])
            else:
                drifts = drift_terms if drift_terms is not None else ["quadratic"]
                UK = UniversalKriging(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    variogram_model=variogram_model,
                    drift_terms=drifts,
                    verbose=False,
                    enable_plotting=False,
                )
                z_krig, ss_krig = UK.execute("grid", x_coords, y_coords)

            z_arr = np.array(z_krig, dtype=np.float64)
            ss_arr = np.array(ss_krig, dtype=np.float64)
            if _is_valid_kriging_output(z_arr, pts):
                return z_arr, ss_arr
            else:
                print(f"\n[Warning] PyKrige UniversalKriging ({'SIA drift' if is_sia_mode else 'quadratic drift'}) produced ill-conditioned matrix values. Using robust built-in Kriging engine.")
        except Exception as e:
            pass

    # --- 2. Ordinary Kriging ---
    if method_clean == "ordinary":
        try:
            from pykrige.ok import OrdinaryKriging

            OK = OrdinaryKriging(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False,
            )
            z_krig, ss_krig = OK.execute("grid", x_coords, y_coords)
            z_arr = np.array(z_krig, dtype=np.float64)
            ss_arr = np.array(ss_krig, dtype=np.float64)
            if _is_valid_kriging_output(z_arr, pts):
                return z_arr, ss_arr
            else:
                print("\n[Warning] PyKrige OrdinaryKriging produced ill-conditioned matrix values. Using robust built-in Kriging engine.")
        except Exception:
            pass

    # --- 3. Regression Kriging ---
    if method_clean == "regression":
        try:
            from pykrige.rk import RegressionKriging
            from sklearn.ensemble import RandomForestRegressor

            if dem_grid is not None and dem_grid.shape == (M, N):
                xx, yy = np.meshgrid(x_coords, y_coords)

                interp_dem = RegularGridInterpolator((y_coords, x_coords), dem_grid, bounds_error=False, fill_value=np.nan)
                pts_z = interp_dem(np.column_stack((pts[:, 1], pts[:, 0])))
                pts_z = np.nan_to_num(pts_z, nan=float(np.mean(dem_grid)))

                X_train = np.column_stack((pts[:, 0], pts[:, 1], pts_z))
                y_train = pts[:, 2]

                X_target = np.column_stack((xx.ravel(), yy.ravel(), dem_grid.ravel()))

                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rk = RegressionKriging(regression_model=rf, method="ordinary", variogram_model=variogram_model)
                rk.fit(X_train, y_train)

                z_pred = rk.predict(X_target).reshape((M, N))
                var_pred = np.ones((M, N)) * np.var(y_train) * 0.2
                if _is_valid_kriging_output(z_pred, pts):
                    return z_pred, var_pred
        except Exception:
            pass

    # Fallback to robust built-in Kriging engine
    return _built_in_kriging(
        pts,
        x_coords,
        y_coords,
        method="sia_thickness" if is_sia_mode else method_clean,
        variogram_model=variogram_model,
        external_drift_grid=external_sia_grid,
    )


def random_forest_hole_filling(
    dem: np.ndarray,
    bedrock_grid: np.ndarray,
    boundary_mask: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
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

    Returns
    -------
    filled_bedrock : 2D np.ndarray
        Seamless bedrock elevation grid with remaining holes filled by Random Forest ML predictions.
    """
    M, N = dem.shape
    x_coords = np.arange(N) * dx
    y_coords = np.arange(M) * dy
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Compute surface DEM slope feature
    from .smoothing import compute_gradients

    grads = compute_gradients(dem, dx=dx, dy=dy)
    slope_grid = grads["slope_rad"]

    # Feature matrix: [X, Y, DEM_Z, Slope]
    features = np.column_stack((xx.ravel(), yy.ravel(), dem.ravel(), slope_grid.ravel()))
    target = bedrock_grid.ravel()
    mask_flat = boundary_mask.ravel()

    # Training data: active body pixels with non-zero valid bedrock thickness
    thickness_flat = dem.ravel() - target
    valid_train = mask_flat & ~np.isnan(target) & (thickness_flat > 0.1)

    if np.sum(valid_train) < 10:
        return bedrock_grid.copy()

    try:
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(features[valid_train], target[valid_train])

        print("\nFinished learning regression model")
        rf_pred = rf.predict(features).reshape((M, N))

        # Fill holes where bedrock is zero or invalid inside boundary_mask
        filled_bedrock = bedrock_grid.copy()
        holes = mask_flat.reshape((M, N)) & ((dem - bedrock_grid) <= 0.1)
        filled_bedrock[holes] = rf_pred[holes]

        # Enforce bedrock elevation <= DEM surface elevation
        filled_bedrock = np.minimum(filled_bedrock, dem)
        return filled_bedrock
    except Exception:
        return bedrock_grid.copy()
