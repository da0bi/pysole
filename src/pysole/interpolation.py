"""
Geostatistical Kriging Interpolation, Physical SIA Drift Modeling, Margin Blending, and ML Gap Filling.

Provides advanced spatial interpolation routines for reconstructing continuous bedrock
topography from scattered radar or seismic sounding soundings:
- Ordinary Kriging assuming constant local or global mean.
- Universal Kriging incorporating glaciological Shallow Ice Approximation (SIA) physical
  drift (U_sia = 1 / sin(alpha)), surface elevation, surface curvature, and spatial polynomials.
- Dual Kriging vector solver for high-performance chunked parallel grid evaluation.
- Geomorphological perimeter margin blending ensuring smooth transitions to proglacial terrain.
- Random Forest machine learning for intelligent gap filling across unmeasured basins.
"""

from dataclasses import dataclass
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator, RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from sklearn.ensemble import RandomForestRegressor
from .raster import GridGeometry
from .drift import (
    DEFAULT_REGRESSION_DRIFT_TERMS,
    FALLBACK_QUADRATIC_NOTICE,
    RASTER_DRIFT_TERMS,
    normalize_engine,
    normalize_method,
    parse_drift_terms,
    prune_redundant_drift_terms,
)
from .logging import logger, get_progress_bar
from .smoothing import compute_gradients, compute_surface_curvature
from .variogram import resolve_native_variogram_model
from .kriging_core import KrigingSystem, max_pairwise_distance, polynomial_drift_columns, resolve_variogram_params
from .memory import MAX_FRACTION_CAP, blas_thread_limit, kriging_chunk_size, plan_native_kriging, resolve_threads


@dataclass
class KrigingResult:
    """
    Typed data container storing Kriging bedrock elevation and variance interpolation outputs.

    Parameters
    ----------
    bedrock_grid : np.ndarray
        2D array of shape (M_rows, N_cols) containing interpolated bedrock elevations in meters.
    variance_grid : np.ndarray
        2D array of shape (M_rows, N_cols) containing Kriging estimation variance in meters^2,
        or filled with NaNs if variance computation was skipped or disabled.
    """
    bedrock_grid: np.ndarray   # Interpolated bedrock elevation grid [m]
    variance_grid: np.ndarray  # Kriging estimation variance grid [\sigma^2 in m^2]


class KrigingEngine:
    """
    Specialized interpolation engine executing Ordinary and Universal Kriging.

    Encapsulates grid geometry, surface DEM, and boundary mask to perform geostatistical
    interpolation with optional Shallow Ice Approximation (SIA) physical drift modeling.

    Parameters
    ----------
    dem : np.ndarray
        2D surface DEM grid of shape (M_rows, N_cols) in meters.
    geometry : GridGeometry
        Standardized spatial grid metadata container.
    outline_mask : np.ndarray, optional
        2D boolean active domain mask of shape (M_rows, N_cols).
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
        """
        Execute Kriging spatial interpolation using this engine's spatial geometry and DEM.

        Parameters
        ----------
        sample_points : np.ndarray
            Array of shape (N, >=3) containing survey points [x, y, value, ...].
        method : {"universal", "ordinary", "regression"}, default="universal"
            Kriging formulation. ``"universal"`` needs ``drift_terms`` (use ``["sia"]`` for the SIA drift).
        variogram_model : {"spherical", "exponential", "gaussian", "linear"}, default="spherical"
            Theoretical semivariogram model function.
        opt_slope_grid : np.ndarray, optional
            Smoothed surface slope grid in radians for SIA drift calculation.
        drift_terms : list of str, optional
            Primitive drift terms of one model (e.g., ['sia', 'linear_xy', 'curvature_dem']).
        include_zero_boundary_condition : bool, default=True
            Whether to enforce zero ice thickness along the perimeter mask boundary.
        n_cores : int, default=-1
            Number of CPU threads for parallel grid evaluation (-1 uses all available).
        engine : {"native", "pykrige"}, default="native"
            Underlying Kriging computation backend.
        show_progress : bool, default=True
            Whether to display a progress bar during computation.

        Returns
        -------
        KrigingResult
            Object holding predicted `bedrock_grid` [m] and estimation `variance_grid` [m^2].
        """
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
    Specialized post-processing engine handling ML gap filling and geomorphological blending.

    Executes Random Forest machine learning hole filling, perimeter margin blending,
    and geomorphological terrain smoothing.

    Parameters
    ----------
    dem : np.ndarray
        2D surface DEM grid of shape (M_rows, N_cols) in meters.
    geometry : GridGeometry
        Standardized spatial grid metadata container.
    outline_mask : np.ndarray, optional
        2D boolean active domain mask of shape (M_rows, N_cols).
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
        """
        Train a Random Forest model on known bedrock soundings to predict unmeasured interior gaps.

        Parameters
        ----------
        bedrock_grid : np.ndarray
            2D array of bedrock elevations in meters containing holes or missing data.
        n_cores : int, default=-1
            Number of CPU cores for parallel tree training (-1 for all cores).
        show_progress : bool, default=True
            Whether to display a progress bar.

        Returns
        -------
        np.ndarray
            2D array of bedrock elevations in meters with missing gaps filled by ML predictions.
        """
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
        """
        Apply geomorphological cosine-weighted margin blending to the surrounding terrain DEM.

        Parameters
        ----------
        bedrock_input : np.ndarray
            2D array of bedrock elevations in meters of shape (M_rows, N_cols) or point array.
        min_gap_dist : float, optional
            Margin blend transition zone width in meters. If None, defaults to 3x cell spacing.

        Returns
        -------
        np.ndarray
            2D array of smoothly blended continuous bedrock elevations in meters.
        """
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
    Harmonize continuous bedrock topography with surrounding terrain along glacier margins.

    Enforces smooth geomorphological boundary convergence by applying a cosine-weighted
    spatial taper: ice thickness smoothly decreases to zero at the glacier outline boundary,
    preventing non-physical cliff artifacts where the interpolated glacier bed meets the ice-free DEM.

    Parameters
    ----------
    dem : np.ndarray
        2D surface DEM grid of shape (M_rows, N_cols) in meters.
    bedrock_input : np.ndarray or tuple of np.ndarray
        Continuous bedrock elevation grid of shape (M_rows, N_cols) or point array [x, y, z_surf, depth].
    boundary_mask : np.ndarray
        2D boolean grid where True indicates the active glaciated domain.
    geometry : GridGeometry
        Standardized spatial geometry container for raster grids.
    min_gap_dist : float, optional
        Width of the margin blending transition zone in meters. Defaults to 3 * cell spacing.

    Returns
    -------
    blended_dem : np.ndarray
        Harmonized continuous bedrock elevation grid of shape (M_rows, N_cols) in meters.
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


def build_drift_grids(
    drift_terms: list[str] | set[str] | None,
    geometry: GridGeometry,
    dem_grid: np.ndarray | None = None,
    opt_slope_grid: np.ndarray | None = None,
    slope_floor_deg: float = 5.0,
    provided: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    Builds the raster covariate grids required by raster drift terms (single source for production and analyzer).

    Parameters
    ----------
    drift_terms : list or set of str
        Primitive drift terms; only ``'sia'``, ``'z_dem'`` and ``'curvature_dem'`` need a grid.
    geometry : GridGeometry
        Grid definition; all returned grids have ``geometry.shape``.
    dem_grid : np.ndarray, optional
        Surface DEM [m]; source of ``z_dem`` and, if not supplied through ``provided``, of ``curvature_dem``
        (raw DEM curvature) and, without ``opt_slope_grid``, of the SIA slope.
    opt_slope_grid : np.ndarray, optional
        Optimised smoothed slope grid [rad]; ``sia`` = ``1 / max(sin(alpha), sin(slope_floor_deg))``.
    slope_floor_deg : float, default 5.0
        Lower bound of the surface slope [deg] for the SIA drift.
    provided : dict of str to np.ndarray, optional
        Pre-computed grids (e.g. the k_c-smoothed ``curvature_dem`` of the Solver, or user-supplied external
        drifts). They are returned unchanged, are never recomputed and may carry any name.

    Returns
    -------
    dict of str to np.ndarray
        ``provided`` plus the grids of all requested raster terms.

    Raises
    ------
    ValueError
        If a requested raster term cannot be built because the required DEM / slope grid is missing.
    """
    grids: dict[str, np.ndarray] = dict(provided) if provided else {}
    terms = set(drift_terms or [])
    shape = tuple(geometry.shape)
    dem_ok = dem_grid is not None and dem_grid.shape == shape

    if "sia" in terms and "sia" not in grids:
        if opt_slope_grid is not None and opt_slope_grid.shape == shape:
            slope_sin = np.sin(opt_slope_grid)
        elif dem_ok:
            slope_sin = np.sin(compute_gradients(dem_grid, dx=geometry.dx, dy=geometry.dy)["slope_rad"])
        else:
            raise ValueError("Universal Kriging with 'sia' drift requested, but neither surface DEM grid nor slope grid was provided.")
        grids["sia"] = 1.0 / np.maximum(slope_sin, np.sin(np.radians(slope_floor_deg)))

    if "z_dem" in terms and "z_dem" not in grids:
        if not dem_ok:
            raise ValueError("Universal Kriging with 'z_dem' elevation drift requested, but surface DEM grid was not provided.")
        grids["z_dem"] = dem_grid

    if "curvature_dem" in terms and "curvature_dem" not in grids:
        if not dem_ok:
            raise ValueError("Universal Kriging with 'curvature_dem' drift requested, but surface DEM grid was not provided.")
        grids["curvature_dem"] = compute_surface_curvature(dem_grid, dx=geometry.dx, dy=geometry.dy)

    return grids


def standardise_drift_grid(name: str, grid: np.ndarray) -> np.ndarray | None:
    """
    Grid-level standardisation (mean 0, standard deviation 1) of a raster drift covariate.

    Both Kriging engines use the same standardised columns, so the drift coefficients are comparable and the
    systems are well conditioned. A flat raster would only duplicate the constant column: it is skipped
    (``None``) with a warning.
    """
    spread = float(np.nanstd(grid))
    if spread < 1e-12:
        logger.warning(f"   [Dual Kriging Engine] External drift raster '{name}' has zero spatial variance (flat surface); omitting redundant constant drift column.")
        return None
    return (grid - float(np.nanmean(grid))) / spread


def sample_drift_grid(
    name: str, standardised: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Samples a standardised raster at the points ``(x, y)`` by bilinear interpolation (rows follow ``y_coords``).

    Points outside the grid and NaN cells get the grid mean (standardised value 0); a warning reports them.
    """
    if len(y_coords) > 1 and y_coords[1] < y_coords[0]:
        y_asc, grid_asc = y_coords[::-1], standardised[::-1, :]
    else:
        y_asc, grid_asc = y_coords, standardised
    values = RegularGridInterpolator((y_asc, x_coords), grid_asc, bounds_error=False, fill_value=0.0)(
        np.column_stack((y, x))
    )
    missing = np.isnan(values)
    if missing.any():
        logger.warning(
            f"   [Dual Kriging Engine] External drift raster '{name}' is NaN at {int(missing.sum())} sample location(s); "
            "imputing the grid mean (standardised value 0)."
        )
    return np.nan_to_num(values, nan=0.0)


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
    max_memory_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust native NumPy/SciPy Ordinary & Universal Kriging solver with zero-centered
    spatial coordinate normalization and diagonal regularization to prevent ill-conditioned matrix explosion.
    Supports the polynomial drift terms ``linear_xy`` (x, y) and ``quadratic_xy`` (x, y, x^2, y^2, x*y), raster
    external drifts (e.g., elevation z_surface, SIA 1/sin(alpha), surface curvature) and their combinations.
    Accelerated with multi-core CPU chunk parallelization via n_cores.

    Parameters
    ----------
    sample_points : np.ndarray, shape (N, >=3)
        Observations with columns ``[x, y, value, ...]``. Rows containing NaN in x, y or value are dropped;
        additional columns (e.g. a profile ID) are ignored.
    x_coords, y_coords : np.ndarray
        1-D cell-centre coordinates [m] of the output grid; the result has shape
        ``(len(y_coords), len(x_coords))``.
    method : str, default "universal"
        ``"ordinary"`` (constant unknown mean; drift terms and external drift grids are ignored with a warning) or
        ``"universal"`` (drift model; needs at least one drift column, otherwise ``ValueError``). Universal Kriging
        needs at least 4 valid points (fewer fall back to Ordinary with a warning).
    variogram_model : str, default "spherical"
        ``"spherical"``, ``"exponential"``, ``"gaussian"`` or ``"linear"`` (see Notes for the caveat).
    external_drift_grid : np.ndarray or dict[str, np.ndarray], optional
        One ``(M, N)`` grid or a dict of named grids used as external drifts (universal only). Each grid is
        standardised and adds one drift column; grids without spatial variance are skipped with a warning. A raster
        term (``"sia"``, ``"z_dem"``, ``"curvature_dem"``) in ``drift_terms`` needs its grid here (key = term name).
    drift_terms : list of str, optional
        ``"linear_xy"`` adds x, y; ``"quadratic_xy"`` adds x, y, x², y², xy (it contains the linear columns).
        Raster terms select the matching ``external_drift_grid`` entries.
    variogram_params : tuple (range, sill, nugget) or dict, optional
        Variogram parameters (dict keys ``"range"``, ``"sill"``, ``"nugget"``). If omitted:
        range = 0.6 x the largest inter-point distance, sill = variance of the values, nugget = 0.
    n_cores : int, default -1
        Number of worker threads for the grid evaluation (-1 uses all CPUs). BLAS threads are capped to the
        physical cores for the factorisation and set to one inside the parallel grid chunks (see
        :func:`pysole.memory.blas_thread_limit`); the algebra itself lives in :class:`pysole.kriging_core.KrigingSystem`.
    show_progress : bool, default True
        Show a progress bar.
    return_variance : bool, default True
        If True, the Kriging estimation variance is evaluated: per grid chunk the augmented Kriging system is
        solved with the LU factorisation (O((N+d)^2) work per cell), which keeps the factorisation in memory
        during the grid evaluation. If False, only the O(N) dual-kriging weight dot product is evaluated and
        the returned variance grid is filled with NaN (so that it cannot be misread as zero uncertainty).
        The memory guard may also switch the variance off (NaN grid, warning) if it does not fit into RAM.
    max_memory_fraction : float, default 0.5
        Memory budget as a fraction of the currently available RAM (``0 < f <= 0.9``). If the estimated peak
        exceeds it, the number of worker threads is reduced first; only if a single thread still does not fit
        is the variance skipped (see :mod:`pysole.memory`).

    Returns
    -------
    z_interp : np.ndarray
        2D array of interpolated values (e.g., bedrock elevation in meters) with shape
        ``(len(y_coords), len(x_coords))``.
    var_interp : np.ndarray
        2D array of Kriging estimation variances [units^2] with shape
        ``(len(y_coords), len(x_coords))``, or filled with NaNs if variance computation
        was skipped or disabled.

    Notes
    -----
    The ``"linear"`` variogram model is a 1-D profile model and is not valid for 2-D native spatial
    Kriging. If specified under the native engine, it automatically falls back to 'spherical' with a warning.
    To use a 2-D linear model, set engine='pykrige'.
    """
    variogram_model = resolve_native_variogram_model(variogram_model)
    if not (0.0 < float(max_memory_fraction) <= MAX_FRACTION_CAP):
        raise ValueError(f"max_memory_fraction must satisfy 0 < f <= {MAX_FRACTION_CAP}, got {max_memory_fraction}.")

    valid = ~np.isnan(sample_points[:, 0]) & ~np.isnan(sample_points[:, 1]) & ~np.isnan(sample_points[:, 2])
    pts = sample_points[valid]

    M, N = len(y_coords), len(x_coords)
    M_grid = M * N

    N_pts = len(pts)
    if N_pts == 0:
        return np.zeros((M, N)), np.zeros((M, N))

    effective_n_cores = resolve_threads(n_cores)

    x_mean, x_scale = float(np.mean(x_coords)), max(float(np.ptp(x_coords)), 1.0)
    y_mean, y_scale = float(np.mean(y_coords)), max(float(np.ptp(y_coords)), 1.0)

    pts_x_norm = (pts[:, 0] - x_mean) / x_scale
    pts_y_norm = (pts[:, 1] - y_mean) / y_scale

    method_clean = normalize_method(method)
    if method_clean == "regression":
        raise ValueError(
            "The native Kriging engine supports the methods 'ordinary' and 'universal'; "
            "'regression' requires the 'pykrige' engine."
        )
    ext_dict: dict[str, np.ndarray] = {}
    if isinstance(external_drift_grid, dict):
        ext_dict = external_drift_grid
    elif isinstance(external_drift_grid, np.ndarray) and external_drift_grid.shape == (M, N):
        ext_dict = {"external_drift": external_drift_grid}
    poly_terms = set(drift_terms or [])

    if method_clean == "ordinary":
        if ext_dict or drift_terms:
            logger.warning("   [Dual Kriging Engine] Kriging method 'ordinary' ignores drift_terms and external drift grids.")
        ext_dict = {}
        poly_terms = set()

    use_universal = method_clean == "universal"
    if use_universal and N_pts < 4:
        logger.warning(f"   [Dual Kriging Engine] Universal Kriging needs at least 4 valid points (got {N_pts}); falling back to Ordinary Kriging.")
        ext_dict = {}
        poly_terms = set()
        use_universal = False

    if use_universal:
        for term in RASTER_DRIFT_TERMS:
            if term in poly_terms and term not in ext_dict:
                raise ValueError(f"Drift term '{term}' needs its raster grid in external_drift_grid, but none was provided.")

    # Process external drift rasters and sample point values
    ext_drifts: list[tuple[np.ndarray, np.ndarray]] = []  # List of (u_flat, pts_u_norm)
    if use_universal:
        for name, grid in ext_dict.items():
            if grid is not None and grid.shape == (M, N):
                u_grid_norm = standardise_drift_grid(name, grid)
                if u_grid_norm is not None:
                    pts_u_norm = sample_drift_grid(name, u_grid_norm, x_coords, y_coords, pts[:, 0], pts[:, 1])
                    ext_drifts.append((u_grid_norm.ravel(), pts_u_norm))

    has_poly_quad = "quadratic_xy" in poly_terms
    has_poly_lin = "linear_xy" in poly_terms
    n_poly = 5 if has_poly_quad else (2 if has_poly_lin else 0)  # x, y[, x^2, y^2, xy]

    n_ext = len(ext_drifts)
    if use_universal and n_ext + n_poly == 0:
        raise ValueError(
            "Universal Kriging has no drift column: give drift_terms with 'linear_xy' / 'quadratic_xy' or a "
            "raster drift grid with spatial variance (use method 'ordinary' for a constant mean)."
        )
    n_drift = 1 + n_ext + n_poly

    # Memory guard: decided before any N x N matrix is allocated.
    mem_plan = plan_native_kriging(
        n_pts=N_pts,
        n_drift=n_drift,
        n_threads=effective_n_cores,
        want_variance=return_variance,
        fraction=max_memory_fraction,
        n_grid_cells=M_grid,
    )
    return_variance = mem_plan.return_variance
    effective_n_cores = mem_plan.n_threads

    # Drift columns at the sample points: external rasters first, then the polynomial terms. The grid rows of the
    # chunk loop below follow exactly the same order.
    point_columns = [pts_u_norm for _, pts_u_norm in ext_drifts]
    if n_poly > 0:
        point_columns += polynomial_drift_columns(pts_x_norm, pts_y_norm, has_poly_quad)
    system = KrigingSystem(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        np.column_stack(point_columns) if point_columns else None,
        variogram_model,
        variogram_params,
    )
    logger.info(
        f"   [Dual Kriging Engine] Applied {system.reg_value:.1e} Tikhonov matrix regularization "
        f"(N={N_pts} points, n_drift={n_drift})"
    )

    # The explicit inverse (variance) is computed before the worker threads start, so that all threads only read
    # shared state and run Level-3 BLAS on it.
    with blas_thread_limit():
        system.factorize()
        if return_variance:
            system.inverse()
        system.release()

    xx, yy = np.meshgrid(x_coords, y_coords)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()

    chunk_size = kriging_chunk_size(N_pts)
    z_interp_flat = np.zeros(M_grid, dtype=np.float64)
    var_interp_flat = np.full(M_grid, np.nan, dtype=np.float64)

    chunks = [(start_idx, min(start_idx + chunk_size, M_grid)) for start_idx in range(0, M_grid, chunk_size)]

    def _process_chunk(chunk_tuple: tuple[int, int]) -> tuple[int, int, np.ndarray, np.ndarray | None]:
        start_idx, end_idx = chunk_tuple
        sub_x = xx_flat[start_idx:end_idx]
        sub_y = yy_flat[start_idx:end_idx]

        rows = [u_flat[start_idx:end_idx] for u_flat, _ in ext_drifts]
        if n_poly > 0:
            rows += polynomial_drift_columns((sub_x - x_mean) / x_scale, (sub_y - y_mean) / y_scale, has_poly_quad)
        z_sub, var_sub = system.predict_points(
            sub_x, sub_y, np.vstack(rows) if rows else None, return_variance=return_variance
        )
        return start_idx, end_idx, z_sub, var_sub

    if len(chunks) > 1 and effective_n_cores > 1:
        # The chunks already run in parallel threads: one BLAS thread each (the cap is process-wide, so it is set
        # around the whole executor instead of inside the threads).
        with blas_thread_limit(1), ThreadPoolExecutor(max_workers=effective_n_cores) as executor:
            chunk_results = list(
                get_progress_bar(
                    executor.map(_process_chunk, chunks),
                    total=len(chunks),
                    desc="   [Dual Kriging Engine] Interpolating grid",
                    unit="chunks",
                    disable=not show_progress,
                )
            )
    else:
        with blas_thread_limit():
            chunk_results = [
                _process_chunk(chunk_tuple)
                for chunk_tuple in get_progress_bar(
                    chunks,
                    desc="   [Dual Kriging Engine] Interpolating grid",
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


def _pykrige_variogram_kwargs(
    variogram_model: str, variogram_params: tuple[float, float, float] | dict[str, float] | None, pts: np.ndarray
) -> dict[str, Any]:
    """
    PyKrige keyword arguments that make it use the given ``(range, sill, nugget)`` variogram of PySole.

    Without parameters (and for the PyKrige-only ``linear`` model, whose parameters are ``[slope, nugget]``) PyKrige
    fits its own variogram. PyKrige's spherical and exponential models equal PySole's; its Gaussian model uses
    ``exp(-d^2 / (4 a / 7)^2)`` instead of ``exp(-3 (d / a)^2)``, so the range is rescaled by ``7 / sqrt(48)``.
    """
    model = str(variogram_model).lower().strip()
    if variogram_params is None or model == "linear":
        return {}
    range_a, sill, nugget = resolve_variogram_params(variogram_params, pts[:, 2], max_pairwise_distance(pts[:, :2]))
    if model == "gaussian":
        range_a *= 7.0 / np.sqrt(48.0)
    return {"variogram_parameters": [sill, range_a, nugget]}


def pykrige_kriging_interpolation(
    sample_points: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    method: str = "universal",
    variogram_model: str = "spherical",
    external_drift_grid: np.ndarray | dict[str, np.ndarray] | None = None,
    drift_terms: list[str] | None = None,
    variogram_params: tuple[float, float, float] | dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Execute 3rd-party PyKrige Ordinary, Universal, or Regression Kriging.

    Provides an alternative Kriging backend leveraging the PyKrige package. Universal Kriging supports every drift
    term of the native engine (any number of raster drifts as ``specified`` drifts and the scaled polynomial columns
    as ``functional`` drifts) and reproduces the native estimates for the same variogram. Regression Kriging uses
    the drift terms as Random Forest covariates.

    Parameters
    ----------
    sample_points : np.ndarray, shape (N, >=3)
        Observations array with columns [x, y, value, ...]. NaN rows are dropped.
    x_coords : np.ndarray
        1D array of output cell center X coordinates in meters.
    y_coords : np.ndarray
        1D array of output cell center Y coordinates in meters.
    method : {"universal", "ordinary", "regression"}, default="universal"
        Kriging formulation.
    variogram_model : {"spherical", "exponential", "gaussian", "linear"}, default="spherical"
        Theoretical semivariogram model function.
    external_drift_grid : np.ndarray or dict of str to np.ndarray, optional
        Raster drift covariates of shape (len(y_coords), len(x_coords)) (a dict keyed by term name; a bare array is
        one covariate). Flat rasters are skipped with a warning.
    drift_terms : list of str, optional
        Primitive drift terms: the polynomial terms ``linear_xy`` / ``quadratic_xy`` select the ``functional``
        drift columns (``x, y`` / ``x, y, x^2, y^2, xy``, scaled like the native engine); the raster terms are taken
        from ``external_drift_grid``. For ``regression`` all terms are Random Forest covariates; without terms the
        covariates are ``x, y``.
    variogram_params : tuple (range, sill, nugget) or dict, optional
        Variogram parameters of PySole (the fitted ones); PyKrige fits its own variogram if omitted. Ignored for
        ``regression`` (its variogram is that of the residuals) and for the ``linear`` model.

    Returns
    -------
    z_interp : np.ndarray
        2D array of interpolated elevations or depths of shape (len(y_coords), len(x_coords)).
    var_interp : np.ndarray
        2D array of estimation variance values, or filled with NaNs if unavailable (e.g. regression).

    Raises
    ------
    ImportError
        If the optional `pykrige` dependency is not installed.
    ValueError
        For universal Kriging without any drift column.
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

    method_clean = normalize_method(method)
    terms = set(drift_terms or [])

    if method_clean == "ordinary":
        from pykrige.ok import OrdinaryKriging

        ok = OrdinaryKriging(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False,
            **_pykrige_variogram_kwargs(variogram_model, variogram_params, pts),
        )
        z_b, v_b = ok.execute("grid", x_coords, y_coords)
        return np.asarray(z_b, dtype=np.float64), np.asarray(v_b, dtype=np.float64)

    # Drift columns shared by universal and regression Kriging: standardised rasters and scaled polynomial terms
    sources: dict[str, np.ndarray] = {}
    if isinstance(external_drift_grid, dict):
        sources = external_drift_grid
    elif isinstance(external_drift_grid, np.ndarray) and external_drift_grid.shape == (M, N):
        sources = {"external_drift": external_drift_grid}
    rasters: list[tuple[str, np.ndarray]] = []
    for name, grid in sources.items():
        if grid is not None and grid.shape == (M, N):
            standardised = standardise_drift_grid(name, grid)
            if standardised is not None:
                rasters.append((name, standardised))

    x_mean, x_scale = float(np.mean(x_coords)), max(float(np.ptp(x_coords)), 1.0)
    y_mean, y_scale = float(np.mean(y_coords)), max(float(np.ptp(y_coords)), 1.0)
    quadratic = "quadratic_xy" in terms
    n_poly = 5 if quadratic else (2 if "linear_xy" in terms else 0)

    def poly_columns(x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
        return polynomial_drift_columns((x - x_mean) / x_scale, (y - y_mean) / y_scale, quadratic)[:n_poly]

    if method_clean == "regression":
        from pykrige.rk import RegressionKriging

        xx, yy = np.meshgrid(x_coords, y_coords)
        grid_x, grid_y = xx.ravel(), yy.ravel()
        train_cols = [sample_drift_grid(name, g, x_coords, y_coords, pts[:, 0], pts[:, 1]) for name, g in rasters]
        grid_cols = [g.ravel() for _, g in rasters]
        if n_poly:
            train_cols += poly_columns(pts[:, 0], pts[:, 1])
            grid_cols += poly_columns(grid_x, grid_y)
        if not train_cols:  # no drift terms: the coordinates are the covariates
            train_cols, grid_cols = [pts[:, 0], pts[:, 1]], [grid_x, grid_y]

        rk = RegressionKriging(
            regression_model=RandomForestRegressor(n_estimators=50, random_state=42),
            method="ordinary",
            variogram_model=variogram_model,
            verbose=False,
        )
        rk.fit(np.column_stack(train_cols), pts[:, :2], pts[:, 2])
        z_flat = rk.predict(np.column_stack(grid_cols), np.column_stack((grid_x, grid_y)))
        # RegressionKriging exposes no estimation variance: NaN (not zero!) marks it as unavailable
        return np.asarray(z_flat, dtype=np.float64).reshape((M, N)), np.full((M, N), np.nan)

    from pykrige.uk import UniversalKriging

    if not rasters and not n_poly:
        raise ValueError(
            "Universal Kriging has no drift column: give drift_terms with 'linear_xy' / 'quadratic_xy' or a "
            "raster drift grid with spatial variance (use method 'ordinary' for a constant mean)."
        )
    drift_names: list[str] = []
    drift_kwargs: dict[str, Any] = {}
    if rasters:
        drift_names.append("specified")
        drift_kwargs["specified_drift"] = [
            sample_drift_grid(name, g, x_coords, y_coords, pts[:, 0], pts[:, 1]) for name, g in rasters
        ]
    if n_poly:
        drift_names.append("functional")
        drift_kwargs["functional_drift"] = [
            lambda x, y, j=j: poly_columns(np.asarray(x), np.asarray(y))[j] for j in range(n_poly)
        ]

    uk = UniversalKriging(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        variogram_model=variogram_model,
        drift_terms=drift_names,
        verbose=False,
        enable_plotting=False,
        **_pykrige_variogram_kwargs(variogram_model, variogram_params, pts),
        **drift_kwargs,
    )
    # PyKrige needs ascending grid axes
    flip = len(y_coords) > 1 and y_coords[1] < y_coords[0]
    execute_kwargs: dict[str, Any] = {}
    if rasters:
        execute_kwargs["specified_drift_arrays"] = [g[::-1, :] if flip else g for _, g in rasters]
    z_b, v_b = uk.execute("grid", x_coords, y_coords[::-1] if flip else y_coords, **execute_kwargs)
    z_b, v_b = np.asarray(z_b, dtype=np.float64), np.asarray(v_b, dtype=np.float64)
    return (z_b[::-1, :], v_b[::-1, :]) if flip else (z_b, v_b)


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
    max_memory_fraction: float = 0.5,
) -> KrigingResult:
    """
    Applies Kriging spatial interpolation on scattered points supporting three methods:
    1. 'universal' / 'universal_kriging' (Universal Kriging with the explicitly listed drift terms, e.g. the
       Shallow Ice Approximation drift U_sia = 1 / sin(alpha_safe) via ``drift_terms=['sia']``)
    2. 'ordinary' / 'ordinary_kriging' (Ordinary Kriging assuming constant mean; ignores drift terms)
    3. 'regression' / 'regression_kriging' (Regression Kriging combining ML regressor with residual Kriging via PyKrige)

    When include_zero_boundary_condition is True, enforces zero-value boundary points (T=0 ns or D=0 m)
    along both the outer perimeter and any interior rock outcrop/nunatak margin boundaries.

    Parameters
    ----------
    sample_points : np.ndarray, shape (N, >=3)
        Observations ``[x, y, value, ...]`` (NaN rows are dropped, extra columns ignored).
    geometry : GridGeometry
        Output grid definition (shape, spacing, cell-centre coordinates).
    method : str, default "universal"
        ``"ordinary"``, ``"universal"`` or ``"regression"`` (the ``*_kriging`` spellings are accepted). The former
        method ``"sia"`` is rejected: use ``"universal"`` with ``drift_terms=["sia"]``.
    variogram_model : str, default "spherical"
        ``"spherical"``, ``"exponential"``, ``"gaussian"`` or ``"linear"``.
    variogram_params : tuple (range, sill, nugget) or dict, optional
        Variogram parameters passed to the engine; if omitted the native engine uses data-driven defaults
        (see ``built_in_kriging_interpolation``).
    dem_grid : np.ndarray, optional
        Surface DEM, required for the ``"z_dem"`` / ``"curvature_dem"`` drifts (and for ``"sia"`` when no
        ``opt_slope_grid`` is given).
    opt_slope_grid : np.ndarray, optional
        Optimised smoothed slope grid [rad] used to build the SIA drift ``1 / sin(alpha)``.
    drift_terms : list of str, optional
        Primitive drift terms of ONE multi-drift model: ``"sia"``, ``"z_dem"``, ``"curvature_dem"``,
        ``"linear_xy"``, ``"quadratic_xy"``. ``"universal"`` needs at least one term (or ``external_drift_grid``);
        ``"ordinary"`` ignores them with a warning. An unknown name raises ``ValueError``.
    outline_mask : np.ndarray, optional
        Boolean glacier mask; also defines the boundary used for the zero-value boundary condition.
    include_zero_boundary_condition : bool, default True
        Add zero-value pseudo-observations along the mask boundary.
    n_cores : int, default -1
        Worker threads (-1 uses all CPUs).
    engine : str, default "native"
        ``"native"`` (NumPy/SciPy) or ``"pykrige"``.
    slope_floor_deg : float, default 5.0
        Lower bound of the surface slope [deg] when forming the SIA drift (avoids ``1/sin(alpha)`` blow-up).
    show_progress : bool, default True
        Show a progress bar.
    external_drift_grid : np.ndarray or dict[str, np.ndarray], optional
        Additional user-supplied external drift grid(s) of the DEM shape.
    return_variance : bool, default True
        Native engine only. If False, the O(N^2) per-cell Kriging variance evaluation is skipped and
        ``KrigingResult.variance_grid`` is filled with NaN. The memory guard may also skip it (NaN, warning).
    max_memory_fraction : float, default 0.5
        Native engine only. RAM budget as a fraction of the available memory (``0 < f <= 0.9``); threads are
        reduced first, the variance is skipped only as a last resort (see ``built_in_kriging_interpolation``).

    Returns
    -------
    KrigingResult
        Data object containing:
        - ``bedrock_grid``: 2D array of interpolated bedrock elevations in meters of shape (M_rows, N_cols).
        - ``variance_grid``: 2D array of Kriging estimation variances in meters^2 (or filled with NaNs if skipped).

    Notes
    -----
    ``variogram_model="linear"`` is supported under ``engine="pykrige"``. Under ``engine="native"``,
    selecting ``"linear"`` logs a warning and automatically falls back to ``"spherical"``
    (the 2-D valid model with near-origin linear slope behavior).

    ``method="regression"`` always runs through PyKrige (also with ``engine="native"``); its ``drift_terms`` are
    the RandomForest covariates. ``engine="pykrige"`` Universal Kriging applies every drift term and uses the
    fitted ``variogram_params``; PyKrige's own kriging variance differs from the native one by about 1 %.
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

    engine_clean = normalize_engine(engine)
    var_model_clean = str(variogram_model).lower().strip()
    method_clean = normalize_method(method)

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

    valid_terms: list[str] = parse_drift_terms(drift_terms)[0] if drift_terms else []

    if method_clean == "ordinary":
        if valid_terms or external_drift_grids:
            logger.warning("   [Dual Kriging Engine] Kriging method 'ordinary' ignores drift_terms and external drift grids.")
        valid_terms = []
        external_drift_grids = {}
    elif method_clean == "regression":
        if drift_terms is None and not external_drift_grids:
            if dem_grid is not None:
                valid_terms = list(DEFAULT_REGRESSION_DRIFT_TERMS)
            else:
                logger.warning(
                    "   [Dual Kriging Engine] DEM grid is missing for Regression Kriging default drift terms; "
                    "falling back to coordinate regression ['linear_xy']."
                )
                valid_terms = ["linear_xy"]
        valid_terms, pruned = prune_redundant_drift_terms(valid_terms)
        if pruned:
            logger.warning(FALLBACK_QUADRATIC_NOTICE)
    elif method_clean == "universal":
        valid_terms, pruned = prune_redundant_drift_terms(valid_terms)
        if pruned:
            logger.warning(FALLBACK_QUADRATIC_NOTICE)

    # Raster drift covariates (SIA 1/sin(alpha), DEM elevation z_dem, surface curvature_dem)
    external_drift_grids = build_drift_grids(
        valid_terms,
        geometry=geometry,
        dem_grid=dem_grid,
        opt_slope_grid=opt_slope_grid,
        slope_floor_deg=slope_floor_deg,
        provided=external_drift_grids,
    )

    if method_clean == "universal" and not valid_terms and not external_drift_grids:
        raise ValueError(
            "Universal Kriging requires drift_terms or an external drift grid - "
            "specify drift_terms or use method 'ordinary'."
        )

    has_ext_drifts = len(external_drift_grids) > 0

    if engine_clean == "pykrige" or method_clean == "regression":
        z_b, v_b = pykrige_kriging_interpolation(
            pts,
            x_coords,
            y_coords,
            method=method_clean,
            variogram_model=variogram_model,
            external_drift_grid=external_drift_grids if has_ext_drifts else None,
            drift_terms=valid_terms,
            variogram_params=variogram_params,
        )
    else:
        # Fast, robust native vector Kriging engine
        z_b, v_b = built_in_kriging_interpolation(
            pts,
            x_coords,
            y_coords,
            method=method_clean,
            variogram_model=variogram_model,
            external_drift_grid=external_drift_grids if has_ext_drifts else None,
            drift_terms=valid_terms,
            variogram_params=variogram_params,
            n_cores=n_cores,
            show_progress=show_progress,
            return_variance=return_variance,
            max_memory_fraction=max_memory_fraction,
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
    Intelligent Random Forest machine learning model for glaciological bedrock hole filling.

    Trains a non-linear Random Forest regressor on known bedrock sounding pixels inside
    the glacier boundary using spatial covariates (easting X, northing Y, surface elevation,
    and local surface slope). Predicts missing ice thickness strictly across internal data gaps
    and unprobed tributary basins, preserving measured soundings and respecting boundary margins.

    Parameters
    ----------
    dem : np.ndarray
        2D surface DEM elevation grid of shape (M_rows, N_cols) in meters.
    bedrock_grid : np.ndarray
        2D Kriging-interpolated bedrock elevation grid of shape (M_rows, N_cols) in meters.
    boundary_mask : np.ndarray
        2D boolean grid where True indicates the active glaciated domain.
    geometry : GridGeometry
        Standardized spatial geometry container for raster grids.
    n_cores : int, default=-1
        Number of CPU cores for parallelized tree fitting (-1 for all available cores).
    show_progress : bool, default=True
        Whether to display a progress bar during model training and gap prediction.

    Returns
    -------
    filled_bedrock : np.ndarray
        Seamless bedrock elevation grid of shape (M_rows, N_cols) in meters with missing holes filled.
    """
    # Use cached 2D spatial meshgrid from geometry
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
    Unified spatial drift basis evaluator for the Drift Analyzer.

    Reproduces the drift columns of the native production engine
    (``built_in_kriging_interpolation``) at arbitrary coordinates: every raster covariate is standardised with
    the mean and standard deviation of its full grid and sampled with the bottom-up bilinear interpolator of
    ``GridGeometry.create_interpolator`` (outside the grid, or at NaN cells, the standardised value is 0); the
    polynomial columns use the grid-coordinate centring and range scaling of the production engine.

    Column order: ``sia``, ``z_dem``, ``curvature_dem`` (raster terms, as available), then ``x``, ``y`` for
    ``linear_xy`` and ``x, y, x^2, y^2, xy`` for ``quadratic_xy`` (``quadratic_xy`` contains the linear
    columns, so ``linear_xy`` + ``quadratic_xy`` does not duplicate them).

    Parameters
    ----------
    drift_terms : list of str
        Primitive drift term identifiers (``'sia'``, ``'z_dem'``, ``'curvature_dem'``, ``'linear_xy'``,
        ``'quadratic_xy'``).
    x_ref : np.ndarray
        Sample easting coordinates in meters; used for the polynomial standardisation only when no
        ``geometry`` is given.
    y_ref : np.ndarray
        Sample northing coordinates in meters; used for the polynomial standardisation only when no
        ``geometry`` is given.
    geometry : GridGeometry, optional
        Spatial grid definition of the drift rasters. Required for the raster terms.
    drift_grids : dict of str to np.ndarray, optional
        Raster covariates of shape ``geometry.shape`` keyed by term name (``'sia'`` = ``1 / sin(alpha)``,
        ``'z_dem'``, ``'curvature_dem'``), e.g. from :func:`build_drift_grids`. A requested raster term without
        its grid raises ``ValueError``.
    """
    def __init__(
        self,
        drift_terms: list[str],
        x_ref: np.ndarray,
        y_ref: np.ndarray,
        geometry: GridGeometry | None = None,
        drift_grids: dict[str, np.ndarray] | None = None,
    ):
        raw_terms = parse_drift_terms(drift_terms)[0] if drift_terms else []
        self.drift_terms, pruned = prune_redundant_drift_terms(raw_terms)
        if pruned:
            logger.warning(FALLBACK_QUADRATIC_NOTICE)
        self.primitives: set[str] = set(self.drift_terms)

        self.column_names: list[str] = []
        self.evaluators: list[Any] = []
        self._nan_warned: set[str] = set()

        for term in RASTER_DRIFT_TERMS:
            if term not in self.primitives:
                continue
            grid = None if drift_grids is None else drift_grids.get(term)
            if geometry is None or grid is None:
                raise ValueError(
                    f"Drift term '{term}' requires its raster grid and the grid geometry, but "
                    f"{'no geometry' if geometry is None else 'no grid'} was provided."
                )
            grid = np.asarray(grid, dtype=np.float64)
            if grid.shape != tuple(geometry.shape):
                raise ValueError(
                    f"Drift raster '{term}' has shape {grid.shape}, expected the grid shape {tuple(geometry.shape)}."
                )
            spread = float(np.nanstd(grid))
            if spread < 1e-12:
                logger.warning(
                    f"   [Drift Analyzer] Drift raster '{term}' has zero spatial variance (flat surface); omitting redundant constant drift column."
                )
                continue
            standardised = (grid - float(np.nanmean(grid))) / spread
            interpolator = geometry.create_interpolator(standardised, fill_value=0.0)
            self._add_column(term, lambda x, y, f=interpolator, t=term: self._sample(f, x, y, t))

        if self.primitives & {"linear_xy", "quadratic_xy"}:
            x_axis = geometry.x_coords if geometry is not None else np.asarray(x_ref, dtype=np.float64)
            y_axis = geometry.y_coords if geometry is not None else np.asarray(y_ref, dtype=np.float64)
            m_x, s_x = float(np.mean(x_axis)), max(float(np.ptp(x_axis)), 1.0)
            m_y, s_y = float(np.mean(y_axis)), max(float(np.ptp(y_axis)), 1.0)
            self._add_column("x", lambda x, y: (x - m_x) / s_x)
            self._add_column("y", lambda x, y: (y - m_y) / s_y)
            if "quadratic_xy" in self.primitives:
                self._add_column("x^2", lambda x, y: ((x - m_x) / s_x) ** 2)
                self._add_column("y^2", lambda x, y: ((y - m_y) / s_y) ** 2)
                self._add_column("x*y", lambda x, y: ((x - m_x) / s_x) * ((y - m_y) / s_y))

    @property
    def n_columns(self) -> int:
        """Number of drift columns (excluding the constant of the Kriging system)."""
        return len(self.evaluators)

    def _add_column(self, name: str, evaluator: Any) -> None:
        self.column_names.append(name)
        self.evaluators.append(evaluator)

    def _sample(self, interpolator: Any, x: np.ndarray, y: np.ndarray, term: str) -> np.ndarray:
        """Samples a standardised raster at (x, y); NaN values are imputed with 0 and reported once per term."""
        values = interpolator(np.column_stack((y, x)))
        missing = np.isnan(values)
        if missing.any():
            if term not in self._nan_warned:
                self._nan_warned.add(term)
                logger.warning(
                    f"   [Drift Analyzer] Drift raster '{term}' is NaN at {int(missing.sum())} sample location(s); "
                    "imputing the grid mean (standardised value 0)."
                )
            values = np.where(missing, 0.0, values)
        return values

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate the standardised drift design matrix for arbitrary query coordinates (x, y).

        Parameters
        ----------
        x : np.ndarray
            1D array of query easting coordinates in meters.
        y : np.ndarray
            1D array of query northing coordinates in meters.

        Returns
        -------
        np.ndarray
            Standardised drift design matrix of shape (len(x), n_columns).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if not self.evaluators:
            return np.zeros((len(x), 0))
        return np.column_stack([fn(x, y) for fn in self.evaluators])
