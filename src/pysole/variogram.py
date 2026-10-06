"""
Spatial Variogram Calculation, Theoretical Model Fitting, and Iterative BSS Optimization.
Ported from MATLAB scripts variogram.m, variogramfit.m, and FFTSmooth.m by Daniel Binder (2011).
Computes experimental isotropic variograms, fits theoretical models (Spherical, Exponential, Gaussian),
and determines optimum DEM surface slope smoothing corner frequency kc.
Saves normalized product variogram comparison plots across all evaluated kc to plots_dir.
"""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
from .smoothing import (
    compute_gradients,
    compute_slope_rad,
    precompute_fft_grid,
    fft_gaussian_smooth_precomputed,
)
from .raster import GridGeometry
from .logging import logger, get_progress_bar


def compute_cutoff_wavelength(kc: float, dx: float = 1.0, dy: float = 1.0) -> float:
    """
    Converts physical 2D corner frequency wavenumber cutoff kc [rad/m] to physical spatial wavelength lambda_c [m].
    """
    if kc <= 0:
        return float("inf")
    return (2.0 * np.pi) / kc


def compute_cutoff_wavenumber(wavelength: float) -> float:
    """
    Converts physical spatial wavelength lambda_c [m] to physical corner frequency wavenumber cutoff kc [rad/m].
    """
    if wavelength <= 0:
        return float("inf")
    return (2.0 * np.pi) / wavelength


@dataclass
class OptimizationResult:
    """
    Typed data container storing BSS corner frequency slope optimization outputs.
    """
    optimal_kc: float
    optimal_slope_grid: np.ndarray
    all_kc_variances: np.ndarray
    all_smoothed_slopes: dict[float, np.ndarray]
    optimal_dem_grid: np.ndarray | None = None
    all_smoothed_dems: dict[float, np.ndarray] | None = None
    dx: float = 1.0
    dy: float = 1.0

    @property
    def optimal_wavelength(self) -> float:
        """
        Returns the physical spatial cutoff wavelength lambda_c [m] corresponding to optimal_kc.
        """
        return compute_cutoff_wavelength(self.optimal_kc, self.dx, self.dy)



class BSSOptimizer:
    """
    Specialized engine executing Basal Shear Stress (BSS) surface slope variance
    optimization across frequency corner spectrum kc.
    """

    def __init__(
        self,
        dem: np.ndarray,
        geometry: GridGeometry,
    ):
        self.dem = dem
        self.geometry = geometry

    def optimize(
        self,
        survey_points: np.ndarray,
        kc_max: float | None = None,
        kc_min: float | None = None,
        d_kc: float | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        d_lambda: float | None = None,
        fft_filter_metric: str = "wavenumber",
        plots_dir: str | Path | None = None,
        prefix: str = "01_",
        stage_name: str = "stage1",
        interactive: bool = False,
        n_cores: int = -1,
        nrbins: int | None = None,
        show_progress: bool = True,
    ) -> OptimizationResult:
        """Executes BSS slope filter optimization across corner frequency spectrum."""
        return optimize_bss_variance(
            dem=self.dem,
            survey_points=survey_points,
            geometry=self.geometry,
            kc_max=kc_max,
            kc_min=kc_min,
            d_kc=d_kc,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            d_lambda=d_lambda,
            fft_filter_metric=fft_filter_metric,
            plots_dir=plots_dir,
            prefix=prefix,
            stage_name=stage_name,
            interactive=interactive,
            n_cores=n_cores,
            nrbins=nrbins,
            show_progress=show_progress,
        )


def calculate_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    maxdist: float | None = None,
    nrbins: int | None = None,
    precomputed_dists: np.ndarray | None = None,
    warn_low_pairs: bool = True,
) -> dict[str, np.ndarray]:
    """
    Computes experimental isotropic 2D variogram.
    Ported from variogram.m.

    Parameters
    ----------
    coords : np.ndarray
        Nx2 coordinate array [X, Y].
    values : np.ndarray
        1D array of values at coordinates (e.g. BSS product values).
    maxdist : float, optional
        Maximum lag distance. Defaults to half the maximum pairwise distance.
    nrbins : int, optional
        Number of distance lag bins. If None, dynamically calculated as
        max(3, N_pairs // 30) (~30 point pairs per bin, floor of 3 bins).
    precomputed_dists : np.ndarray, optional
        Pre-calculated pdist(coords) array to avoid redundant distance calculations.
    warn_low_pairs : bool, optional
        If True, logs a warning if avg point pairs per bin is under 30. Defaults to False.

    Returns
    -------
    result : dict
        dict with keys: 'distance' (lag distances h), 'val' (semivariance gamma(h)),
        'np' (number of point pairs per lag bin).
    """
    n_pts = len(coords)
    if n_pts < 2:
        return {"distance": np.array([]), "val": np.array([]), "np": np.array([])}

    if precomputed_dists is None and n_pts > 5000:
        sub_sample_size = min(1000, n_pts)
        sub_idx = np.linspace(0, n_pts - 1, sub_sample_size, dtype=int)
        sub_dists = pdist(coords[sub_idx])
        calc_maxdist = 0.5 * float(np.max(sub_dists)) if (maxdist is None or maxdist <= 0) else float(maxdist)

        n_pairs = (n_pts * (n_pts - 1)) // 2
        max_bins_for_30_pairs = max(3, n_pairs // 30)
        actual_nrbins = max_bins_for_30_pairs if (nrbins is None or nrbins <= 0) else max(3, int(nrbins))

        bins = np.linspace(0, calc_maxdist, actual_nrbins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        counts = np.zeros(actual_nrbins, dtype=np.int64)
        sums = np.zeros(actual_nrbins, dtype=np.float64)

        chunk_rows = 500
        for i_start in range(0, n_pts - 1, chunk_rows):
            i_end = min(i_start + chunk_rows, n_pts - 1)
            for i in range(i_start, i_end):
                j_idx = np.arange(i + 1, n_pts)
                d_ij = np.hypot(coords[i, 0] - coords[j_idx, 0], coords[i, 1] - coords[j_idx, 1])
                diff_ij = (values[i] - values[j_idx]) ** 2

                b_idx = np.digitize(d_ij, bins) - 1
                valid_m = (b_idx >= 0) & (b_idx < actual_nrbins)
                if np.any(valid_m):
                    counts += np.bincount(b_idx[valid_m], minlength=actual_nrbins)
                    sums += np.bincount(b_idx[valid_m], weights=diff_ij[valid_m], minlength=actual_nrbins)

        with np.errstate(divide="ignore", invalid="ignore"):
            semivars = np.where(counts > 0, 0.5 * (sums / np.maximum(counts, 1)), np.nan)

        valid = ~np.isnan(semivars) & (counts > 0)
        return {
            "distance": bin_centers[valid],
            "val": semivars[valid],
            "np": counts[valid],
        }

    if precomputed_dists is not None:
        dists = precomputed_dists
    else:
        dists = pdist(coords)

    if maxdist is None or maxdist <= 0:
        maxdist = 0.5 * float(np.max(dists)) if len(dists) > 0 else 1.0

    # Dynamic bin calculation enforcing minimum 30 point pairs per bin threshold
    n_pairs = len(dists)
    max_bins_for_30_pairs = max(3, n_pairs // 30)

    if nrbins is None or nrbins <= 0:
        actual_nrbins = max_bins_for_30_pairs
    else:
        actual_nrbins = max(3, int(nrbins))
        avg_pairs = n_pairs / float(actual_nrbins) if actual_nrbins > 0 else 0
        if warn_low_pairs and avg_pairs < 30.0 and n_pairs >= 30:
            logger.warning(
                f"Specified lag bin count (nrbins={actual_nrbins}) yields an average of only {avg_pairs:.1f} "
                f"point pairs per bin (violating recommended minimum threshold of 30 pairs/bin). "
                f"Recommended maximum bin count for this dataset is nrbins={max_bins_for_30_pairs}."
            )

    bins = np.linspace(0, maxdist, actual_nrbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    sq_diffs_vec = pdist(values[:, np.newaxis], metric="sqeuclidean")
    bin_indices = np.digitize(dists, bins) - 1

    valid_bins_mask = (bin_indices >= 0) & (bin_indices < actual_nrbins)
    valid_indices = bin_indices[valid_bins_mask]
    valid_sq_diffs = sq_diffs_vec[valid_bins_mask]

    counts = np.bincount(valid_indices, minlength=actual_nrbins)
    sums = np.bincount(valid_indices, weights=valid_sq_diffs, minlength=actual_nrbins)

    with np.errstate(divide="ignore", invalid="ignore"):
        semivars = np.where(counts > 0, 0.5 * (sums / np.maximum(counts, 1)), np.nan)

    valid = ~np.isnan(semivars) & (counts > 0)
    return {
        "distance": bin_centers[valid],
        "val": semivars[valid],
        "np": counts[valid],
    }


def spherical_variogram(h: np.ndarray, a: float, c: float, n: float) -> np.ndarray:
    """Spherical variogram model: gamma(h) = n + c * [1.5*(h/a) - 0.5*(h/a)^3] for h <= a; n + c for h > a."""
    h = np.asarray(h, dtype=float)
    a_eff = max(float(a), 1e-6)
    gamma = np.full_like(h, n + c)
    mask = h <= a_eff
    h_rel = h[mask] / a_eff
    gamma[mask] = n + c * (1.5 * h_rel - 0.5 * (h_rel**3))
    return gamma


def exponential_variogram(h: np.ndarray, a: float, c: float, n: float) -> np.ndarray:
    """Exponential variogram model: gamma(h) = n + c * [1 - exp(-3h/a)]."""
    h = np.asarray(h, dtype=float)
    a_eff = max(float(a), 1e-6)
    return n + c * (1.0 - np.exp(-3.0 * h / a_eff))


def gaussian_variogram(h: np.ndarray, a: float, c: float, n: float) -> np.ndarray:
    """Gaussian variogram model: gamma(h) = n + c * [1 - exp(-3*(h/a)^2)]."""
    h = np.asarray(h, dtype=float)
    a_eff = max(float(a), 1e-6)
    return n + c * (1.0 - np.exp(-3.0 * (h / a_eff)**2))


def linear_variogram(h: np.ndarray, a: float, c: float, n: float) -> np.ndarray:
    """Linear variogram model: gamma(h) = n + c * min(h/a, 1)."""
    h = np.asarray(h, dtype=float)
    a_eff = max(float(a), 1e-6)
    return n + c * np.clip(h / a_eff, 0.0, 1.0)


def evaluate_variogram_model(
    h: np.ndarray, model_type: str, a: float, c: float, n: float
) -> np.ndarray:
    """Evaluates the specified theoretical variogram model curve at lag distances h."""
    m_type = str(model_type).lower().strip()
    if "exp" in m_type:
        return exponential_variogram(h, a, c, n)
    elif "gauss" in m_type:
        return gaussian_variogram(h, a, c, n)
    elif "lin" in m_type:
        return linear_variogram(h, a, c, n)
    else:
        return spherical_variogram(h, a, c, n)


def fit_variogram_model(
    distances: np.ndarray,
    semivars: np.ndarray,
    model_type: str = "spherical",
    show_progress: bool = False,
) -> tuple[float, float, float, dict[str, np.ndarray]]:
    """
    Fits a theoretical variogram model to experimental variogram data.
    Supports Spherical, Exponential, Gaussian, and Linear model functions.
    Scale-invariant fitting scales initial guesses and bounds relative to experimental semivariance magnitude.

    Returns
    -------
    a_range : float
        Calculated spatial correlation range [m].
    sill : float
        Sill parameter C.
    nugget : float
        Nugget variance C0.
    model_curve : dict
        dict with 'h' and 'gamma' fine curve points for plotting.
    """
    with get_progress_bar(
        total=max(1, len(distances)),
        desc="   [Variogram Fitting] Fitting model curve",
        unit="bins",
        disable=not show_progress,
    ) as pbar:
        if len(distances) < 3 or len(semivars) < 3:
            a_default = float(np.max(distances)) if len(distances) > 0 else 1000.0
            sill_default = float(np.mean(semivars)) if len(semivars) > 0 else 1.0
            h_fine = np.linspace(0, a_default * 1.5, 100)
            pbar.update(max(1, len(distances)))
            return a_default, sill_default, 0.0, {"h": h_fine, "gamma": np.full_like(h_fine, sill_default)}

        max_dist = float(np.max(distances))
        gamma_tail = float(np.mean(semivars[-max(1, len(semivars) // 3) :]))
        sill0 = max(gamma_tail, 1e-12)

        p0 = [max_dist * 0.5, sill0 * 0.8, 0.0]
        bounds = ([1e-3, 1e-15, 0.0], [max_dist * 5.0, sill0 * 20.0, sill0 * 2.0])

        m_type = str(model_type).lower().strip()
        if "exp" in m_type:
            fit_func = exponential_variogram
        elif "gauss" in m_type:
            fit_func = gaussian_variogram
        elif "lin" in m_type:
            fit_func = linear_variogram
        else:
            fit_func = spherical_variogram

        try:
            popt, _ = curve_fit(fit_func, distances, semivars, p0=p0, bounds=bounds, maxfev=2000)
            a_range, sill, nugget = float(popt[0]), float(popt[1]), float(popt[2])
        except Exception as e:
            logger.warning(f"Variogram curve fitting ({model_type}) failed: {e}. Falling back to default parameters.")
            a_range = max_dist * 0.5
            sill = sill0
            nugget = 0.0

        h_fine = np.linspace(0, max_dist * 1.2, 150)
        gamma_fine = evaluate_variogram_model(h_fine, model_type, a_range, sill, nugget)
        model_curve = {"h": h_fine, "gamma": gamma_fine}

        pbar.update(len(distances))
        return a_range, sill, nugget, model_curve


def optimize_bss_variance(
    dem: np.ndarray,
    survey_points: np.ndarray,
    geometry: GridGeometry,
    kc_max: float | None = None,
    kc_min: float | None = None,
    d_kc: float | None = None,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    d_lambda: float | None = None,
    fft_filter_metric: str = "wavenumber",
    plots_dir: str | Path | None = None,
    prefix: str = "01_",
    stage_name: str = "stage1",
    interactive: bool = False,
    n_cores: int = -1,
    nrbins: int | None = None,
    show_progress: bool = True,
) -> OptimizationResult:
    """
    Iterative optimization process to determine optimum DEM surface slope smoothing degree.
    Supports filter parameterization by wavenumber (kc [rad/m]) or spatial wavelength (lambda [m]).
    """
    dx = geometry.dx
    dy = geometry.dy
    x_coords = geometry.x_coords
    y_coords = geometry.y_coords

    effective_n_cores = (os.cpu_count() or 1) if (n_cores == -1 or n_cores is None) else max(1, int(n_cores))

    cellsize_min = min(abs(dx), abs(dy))
    lambda_nyquist = 2.0 * cellsize_min
    k_nyquist = np.pi / cellsize_min

    logger.info(
        f"   DEM Spatial Resolution dx={dx:.1f}m, dy={dy:.1f}m -> Nyquist Limits: "
        f"k_Nyquist = {k_nyquist:.4f} rad/m, λ_Nyquist = {lambda_nyquist:.1f} m"
    )

    # Pre-filter valid survey points ONCE and compute distance matrix ONCE
    val_col = 3 if survey_points.shape[1] >= 4 else 2
    valid_pts_mask = ~np.isnan(survey_points[:, 0]) & ~np.isnan(survey_points[:, 1]) & (survey_points[:, val_col] > 0)
    pts_valid_coords = survey_points[valid_pts_mask, :2]
    survey_dists = pdist(pts_valid_coords) if len(pts_valid_coords) >= 2 else None

    # Pre-compute 2D Forward FFT and wavenumber grid ONCE on raw DEM elevation Z_surf
    A_shift_dem, k_grid_dem, k_max_grid = precompute_fft_grid(dem, dx=dx, dy=dy)
    base_slope = compute_slope_rad(dem, dx=dx, dy=dy)

    use_wavelength = str(fft_filter_metric).lower().strip() == "wavelength" or (lambda_min is not None or lambda_max is not None)

    if use_wavelength:
        lambda_min_val = float(lambda_min) if lambda_min is not None else lambda_nyquist
        if lambda_min_val < lambda_nyquist:
            logger.warning(
                f"   [Warning] Requested lambda_min ({lambda_min_val:.1f} m) is smaller than grid Nyquist wavelength "
                f"λ_Nyquist = {lambda_nyquist:.1f} m (for cellsize={cellsize_min:.1f} m). Clamping lambda_min to {lambda_nyquist:.1f} m."
            )
            lambda_min_val = lambda_nyquist

        lambda_max_val = float(lambda_max) if lambda_max is not None else max(1000.0, lambda_min_val * 10.0)
        d_lambda_val = float(d_lambda) if d_lambda is not None else 10.0
    else:
        kc_max_val = float(kc_max) if kc_max is not None else min(1.0, k_nyquist)
        if kc_max_val > k_nyquist:
            logger.warning(
                f"   [Warning] Requested kc_max ({kc_max_val:.4f} rad/m) exceeds grid Nyquist wavenumber "
                f"k_Nyquist = {k_nyquist:.4f} rad/m (for cellsize={cellsize_min:.1f} m). Clamping kc_max to {k_nyquist:.4f} rad/m."
            )
            kc_max_val = k_nyquist

        kc_min_val = float(kc_min) if kc_min is not None else 0.01
        d_kc_val = float(d_kc) if d_kc is not None else 0.01

    if survey_dists is not None and len(survey_dists) > 0:
        n_pairs = len(survey_dists)
        max_pair_dist = float(np.max(survey_dists))
        maxdist_calc = 0.5 * max_pair_dist
        max_bins_for_30_pairs = max(3, n_pairs // 30)

        if nrbins is None or nrbins <= 0:
            actual_nrbins_calc = max_bins_for_30_pairs
        else:
            actual_nrbins_calc = max(3, int(nrbins))

        bin_w = maxdist_calc / actual_nrbins_calc if actual_nrbins_calc > 0 else 0.0
        avg_pairs_p_bin = n_pairs / float(actual_nrbins_calc) if actual_nrbins_calc > 0 else 0.0

        bins_calc = np.linspace(0, maxdist_calc, actual_nrbins_calc + 1)
        bin_idxs = np.digitize(survey_dists, bins_calc) - 1
        valid_b_mask = (bin_idxs >= 0) & (bin_idxs < actual_nrbins_calc)
        counts_calc = np.bincount(bin_idxs[valid_b_mask], minlength=actual_nrbins_calc)
        min_p = int(np.min(counts_calc[counts_calc > 0])) if np.any(counts_calc > 0) else 0
        max_p = int(np.max(counts_calc)) if len(counts_calc) > 0 else 0

        logger.info(
            f"   Variogram Lag Distance Binning: {actual_nrbins_calc} bins (maxdist = {maxdist_calc:.2f} m, "
            f"bin width = {bin_w:.2f} m), relative distances per bin: avg = {avg_pairs_p_bin:.1f} pairs/bin "
            f"(range: {min_p} - {max_p} pairs, total pairs = {n_pairs})"
        )

    all_kc_variances = []
    all_smoothed_slopes = {}
    all_smoothed_dems = {}
    best_kc = None
    min_var = float("inf")
    best_slope_grid = base_slope.copy()
    best_dem_grid = dem.copy()
    range_fix = None

    def _eval_single_kc(kc_val: float):
        if kc_val < 0:
            return None

        kc_key = round(float(kc_val), 6)
        smoothed_dem = fft_gaussian_smooth_precomputed(A_shift_dem, k_grid_dem, kc=kc_val)
        smoothed_slope = compute_slope_rad(smoothed_dem, dx=dx, dy=dy)

        interp_slope = geometry.create_interpolator(smoothed_slope, fill_value=np.nan)
        pts_xy = np.column_stack((survey_points[:, 1], survey_points[:, 0]))
        slopes_pts = interp_slope(pts_xy)

        thickness_or_t = survey_points[:, val_col]
        bss_product = slopes_pts * thickness_or_t

        valid = ~np.isnan(bss_product) & (survey_points[:, val_col] > 0)
        if np.sum(valid) < 3:
            return None

        mean_product = float(np.mean(bss_product[valid]))
        use_precomputed_dists = survey_dists if (np.array_equal(valid, valid_pts_mask) and survey_dists is not None) else None

        var_result = calculate_variogram(
            survey_points[valid, :2],
            bss_product[valid],
            nrbins=nrbins,
            precomputed_dists=use_precomputed_dists,
            warn_low_pairs=False,
        )

        return (kc_val, kc_key, smoothed_dem, smoothed_slope, var_result, mean_product)

    go_on = True
    while go_on:
        if interactive:
            logger.info(f"--- BSS Filter Optimization (Grid k_Nyquist = {k_nyquist:.4f} rad/m, λ_Nyquist = {lambda_nyquist:.1f} m) ---")
            try:
                if use_wavelength:
                    val_min = input(f"Enter Minimum Spatial Wavelength (lambda_min in meters, default = {lambda_min_val:.1f}): ").strip()
                    if val_min:
                        lambda_min_val = max(float(val_min), lambda_nyquist)
                    val_max = input(f"Enter Maximum Spatial Wavelength (lambda_max in meters, default = {lambda_max_val:.1f}): ").strip()
                    if val_max:
                        lambda_max_val = float(val_max)
                    val_step = input(f"Enter Spatial Wavelength Stepwidth d_lambda (default = {d_lambda_val:.1f}): ").strip()
                    if val_step:
                        d_lambda_val = abs(float(val_step))
                else:
                    val_max = input(f"Enter Maximum Corner Frequency (kc_max in rad/m, default = {kc_max_val:.4f}): ").strip()
                    if val_max:
                        kc_max_val = min(float(val_max), k_nyquist)
                    val_min = input(f"Enter Minimum Corner Frequency (kc_min in rad/m, default = {kc_min_val:.4f}): ").strip()
                    if val_min:
                        kc_min_val = float(val_min)
                    val_step = input(f"Enter Corner Frequency Stepwidth d_kc (default = {d_kc_val:.4f}): ").strip()
                    if val_step:
                        d_kc_val = abs(float(val_step))
            except Exception as e:
                logger.warning(f"Input error, using defaults: {e}")

        if use_wavelength:
            eff_dlam = abs(d_lambda_val) if d_lambda_val > 0 else 10.0
            lambda_vals = np.arange(lambda_min_val, lambda_max_val + 1e-9, eff_dlam)
            if len(lambda_vals) > 0 and lambda_vals[-1] < lambda_max_val - 1e-6:
                lambda_vals = np.append(lambda_vals, lambda_max_val)
            kc_values = (2.0 * np.pi) / lambda_vals
        else:
            effective_dkc = abs(d_kc_val) if d_kc_val > 0 else 0.01
            kc_values = np.arange(kc_max_val, kc_min_val - 1e-9, -effective_dkc)
            if len(kc_values) > 0 and kc_values[-1] > kc_min_val + 1e-6:
                kc_values = np.append(kc_values, kc_min_val)

        step_variances = []
        evaluated_variograms = []

        valid_kc_list = [float(kc) for kc in kc_values if kc >= 0]
        if len(valid_kc_list) == 0:
            break

        # Step 1: Evaluate baseline/first kc if range_fix is not set yet
        if range_fix is None:
            first_eval = _eval_single_kc(valid_kc_list[0])
            if first_eval is not None:
                kc_val, kc_key, smoothed_dem, smoothed_slope, var_result, mean_product = first_eval
                if len(var_result["val"]) > 0:
                    a_range, sill, nugget, model_curve = fit_variogram_model(
                        var_result["distance"], var_result["val"], model_type="spherical"
                    )

                    if interactive or (plots_dir is not None):
                        from .plotting import plot_unfiltered_product_variogram
                        plot_unfiltered_product_variogram(
                            distances=var_result["distance"],
                            semivars=var_result["val"],
                            model_curve=model_curve,
                            a_range=a_range,
                            sill=sill,
                            nugget=nugget,
                            plots_dir=plots_dir,
                            prefix=prefix,
                            stage_name=stage_name,
                            interactive=interactive,
                        )
                        if interactive:
                            while True:
                                ans_range = input(f"\nCalculated Correlation Range = {a_range:.2f} m (using nrbins = {nrbins}). Accept these values? [Y/n]: ").strip().lower()
                                if ans_range in ["n", "no"]:
                                    val_bins = input(f"Enter custom number of distance bins (nrbins, current = {nrbins}): ").strip()
                                    if val_bins:
                                        try:
                                            nrbins = int(val_bins)
                                            # Recalculate baseline variogram and refit model with updated nrbins
                                            first_eval = _eval_single_kc(valid_kc_list[0])
                                            if first_eval is not None:
                                                _, _, _, _, var_result, _ = first_eval
                                                if len(var_result["val"]) > 0:
                                                    a_range, sill, nugget, model_curve = fit_variogram_model(
                                                        var_result["distance"], var_result["val"], model_type="spherical"
                                                    )
                                                    plot_unfiltered_product_variogram(
                                                        distances=var_result["distance"],
                                                        semivars=var_result["val"],
                                                        model_curve=model_curve,
                                                        a_range=a_range,
                                                        sill=sill,
                                                        nugget=nugget,
                                                        plots_dir=plots_dir,
                                                        prefix=prefix,
                                                        stage_name=stage_name,
                                                        interactive=interactive,
                                                    )
                                        except Exception as e:
                                            logger.warning(f"Invalid nrbins value: {e}")

                                    val_r = input(f"Enter custom correlation range [m] (calculated = {a_range:.2f} m): ").strip()
                                    range_fix = float(val_r) if val_r else a_range
                                    break
                                else:
                                    range_fix = a_range
                                    break
                        else:
                            range_fix = a_range
                    else:
                        range_fix = a_range

        if range_fix is None:
            range_fix = 1000.0

        # Step 2: Parallel execution across all kc_values using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=effective_n_cores) as executor:
            kc_eval_results = list(
                get_progress_bar(
                    executor.map(_eval_single_kc, valid_kc_list),
                    total=len(valid_kc_list),
                    desc="   [BSS Optimization] Searching k_c spectrum",
                    unit="kc",
                    disable=not show_progress,
                )
            )

        for res in kc_eval_results:
            if res is None:
                continue

            kc_val, kc_key, smoothed_dem, smoothed_slope, var_result, mean_product = res
            all_smoothed_dems[kc_key] = smoothed_dem.copy()
            all_smoothed_slopes[kc_key] = smoothed_slope.copy()

            if len(var_result["val"]) > 0:
                if mean_product != 0 and not np.isnan(mean_product):
                    gamma_norm = var_result["val"] / (mean_product**2)
                else:
                    gamma_norm = var_result["val"].copy()

                evaluated_variograms.append({
                    "kc": kc_val,
                    "distance": var_result["distance"],
                    "gamma_raw": var_result["val"],
                    "gamma_norm": gamma_norm,
                    "mean_product": mean_product,
                })

                effective_range = range_fix
                range_mask = var_result["distance"] <= effective_range
                if np.any(range_mask):
                    mean_variance = float(np.mean(gamma_norm[range_mask]))
                else:
                    mean_variance = float(np.mean(gamma_norm))

                step_variances.append([kc_val, mean_variance])
                all_kc_variances.append([kc_val, mean_variance])

                if mean_variance < min_var:
                    min_var = mean_variance
                    best_kc = kc_val
                    best_slope_grid = smoothed_slope
                    best_dem_grid = smoothed_dem

        if (interactive or (plots_dir is not None)) and len(evaluated_variograms) > 0:
            from .plotting import plot_bss_kc_optimization_variograms
            plot_bss_kc_optimization_variograms(
                evaluated_variograms=evaluated_variograms,
                best_kc=best_kc,
                all_kc_variances=all_kc_variances,
                plots_dir=plots_dir,
                prefix=prefix,
                stage_name=stage_name,
                interactive=interactive,
            )

        if interactive:
            logger.info("Iteration Results (Corner Frequency vs Mean Variance):")
            for kc_val, var_val in (step_variances if step_variances else all_kc_variances):
                logger.info(f"  k_c = {kc_val:.4f} --> Mean Variance = {var_val:.6f}")

            ans = input("\nContinue Optimization Process? [y/N]: ").strip().lower()
            if ans in ["y", "yes"]:
                go_on = True
            else:
                go_on = False
        else:
            go_on = False

    if best_kc is None:
        best_kc = float(kc_max)

    opt_wl = compute_cutoff_wavelength(best_kc, dx, dy)
    logger.info(f"   Optimal Corner Frequency k_c = {best_kc:.4f} (cutoff wavelength λ_c = {opt_wl:.2f} m)")

    kc_var_array = np.array(all_kc_variances)
    return OptimizationResult(
        optimal_kc=float(best_kc),
        optimal_slope_grid=best_slope_grid,
        all_kc_variances=kc_var_array,
        all_smoothed_slopes=all_smoothed_slopes,
        optimal_dem_grid=best_dem_grid,
        all_smoothed_dems=all_smoothed_dems,
        dx=float(dx),
        dy=float(dy),
    )
