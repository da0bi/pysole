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


def compute_cutoff_wavelength(kc: float) -> float:
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
    opt_variogram_params: dict[str, float] | None = None
    dx: float = 1.0
    dy: float = 1.0

    @property
    def optimal_wavelength(self) -> float:
        """
        Returns the physical spatial cutoff wavelength lambda_c [m] corresponding to optimal_kc.
        """
        return compute_cutoff_wavelength(self.optimal_kc)



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
        n_steps: int | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
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
            n_steps=n_steps,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
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

    # Dynamic bin calculation enforcing minimum 30 point pairs per bin threshold within maxdist
    in_range_pairs = int(np.sum(dists <= maxdist))
    max_bins_for_30_pairs = min(30, max(3, in_range_pairs // 30))

    if nrbins is None or nrbins <= 0:
        actual_nrbins = max_bins_for_30_pairs
    else:
        actual_nrbins = max(3, int(nrbins))
        avg_pairs = in_range_pairs / float(actual_nrbins) if actual_nrbins > 0 else 0
        if warn_low_pairs and avg_pairs < 30.0 and in_range_pairs >= 30:
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
    counts: np.ndarray | None = None,
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

        sigma_weights = None
        if counts is not None and len(counts) == len(distances):
            sigma_weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))

        try:
            popt, _ = curve_fit(fit_func, distances, semivars, p0=p0, bounds=bounds, sigma=sigma_weights, absolute_sigma=False, maxfev=2000)
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
    n_steps: int | None = None,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
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
    Uses Half-Domain domain-scaling for minimum frequency kc_min and dynamic discrete Fourier mode counting for n_steps.
    Supports wavelength metric with straight conversion to wavenumbers and fallback to kc_max / kc_min when null.
    """
    dx = geometry.dx
    dy = geometry.dy
    x_coords = geometry.x_coords
    y_coords = geometry.y_coords

    effective_n_cores = (os.cpu_count() or 1) if (n_cores == -1 or n_cores is None) else max(1, int(n_cores))

    M_rows, N_cols = dem.shape
    L_max = max(abs(dx) * N_cols, abs(dy) * M_rows)
    cellsize_min = min(abs(dx), abs(dy))
    lambda_nyquist = 2.0 * cellsize_min
    k_nyquist = np.pi / cellsize_min

    logger.info(
        f"   DEM Spatial Resolution dx={dx:.1f}m, dy={dy:.1f}m -> Nyquist Limits: "
        f"k_Nyquist = {k_nyquist:.4f} rad/m, λ_Nyquist = {lambda_nyquist:.1f} m"
    )

    use_wavelength = (
        str(fft_filter_metric).lower().strip() == "wavelength"
        or lambda_min is not None
        or lambda_max is not None
    )

    kc_max_default = float(kc_max) if kc_max is not None else k_nyquist
    kc_min_default = float(kc_min) if kc_min is not None else (4.0 * np.pi / L_max)

    kc_max_val = (2.0 * np.pi / float(lambda_min)) if lambda_min is not None else kc_max_default
    kc_min_val = (2.0 * np.pi / float(lambda_max)) if lambda_max is not None else kc_min_default

    kc_max_val = min(kc_max_val, k_nyquist)

    if kc_min_val >= kc_max_val:
        raise ValueError(
            f"Invalid wavenumber search range: minimum wavenumber k_c,min ({kc_min_val:.4f} rad/m) "
            f"must be strictly less than maximum wavenumber k_c,max ({kc_max_val:.4f} rad/m)."
        )

    if n_steps is None or n_steps <= 0:
        n_modes = int(np.floor((kc_max_val - kc_min_val) * L_max / (2.0 * np.pi)))
        n_steps_val = int(np.clip(n_modes, 10, 50))
    else:
        n_steps_val = max(3, int(n_steps))

    # Pre-filter valid survey points ONCE and compute distance matrix ONCE (only for N <= 5000 to avoid O(N^2) RAM footprint)
    val_col = 3 if survey_points.shape[1] >= 4 else 2
    valid_pts_mask = ~np.isnan(survey_points[:, 0]) & ~np.isnan(survey_points[:, 1]) & (survey_points[:, val_col] > 0)
    pts_valid_coords = survey_points[valid_pts_mask, :2]
    survey_dists = pdist(pts_valid_coords) if (len(pts_valid_coords) >= 2 and len(pts_valid_coords) <= 5000) else None

    # Pre-compute 2D Forward FFT and wavenumber grid ONCE on raw DEM elevation Z_surf with padding for lowest kc
    A_shift_dem, k_grid_dem, k_max_grid = precompute_fft_grid(dem, dx=dx, dy=dy, kc=kc_min_val)
    base_slope = compute_slope_rad(dem, dx=dx, dy=dy)

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
    fitted_var_params: dict[str, float] | None = None

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
            metric_label = "wavelength [m]" if use_wavelength else "wavenumber [rad/m]"
            logger.info(f"--- BSS Filter Optimization ({metric_label}, Grid k_Nyquist = {k_nyquist:.4f} rad/m, λ_Nyquist = {lambda_nyquist:.1f} m) ---")
            try:
                if use_wavelength:
                    cur_lmin = (2.0 * np.pi) / kc_max_val
                    cur_lmax = (2.0 * np.pi) / kc_min_val
                    val_lmin = input(f"Enter Minimum Cutoff Wavelength (lambda_min in m, default = {cur_lmin:.1f}): ").strip()
                    if val_lmin:
                        kc_max_val = min((2.0 * np.pi) / float(val_lmin), k_nyquist)
                    val_lmax = input(f"Enter Maximum Cutoff Wavelength (lambda_max in m, default = {cur_lmax:.1f}): ").strip()
                    if val_lmax:
                        kc_min_val = (2.0 * np.pi) / float(val_lmax)
                else:
                    val_max = input(f"Enter Maximum Corner Frequency (kc_max in rad/m, default = {kc_max_val:.4f}): ").strip()
                    if val_max:
                        kc_max_val = min(float(val_max), k_nyquist)
                    val_min = input(f"Enter Minimum Corner Frequency (kc_min in rad/m, default = {kc_min_val:.4f}): ").strip()
                    if val_min:
                        kc_min_val = float(val_min)
                val_steps = input(f"Enter Number of Evaluation Steps n_steps (default = {n_steps_val}): ").strip()
                if val_steps:
                    n_steps_val = max(3, int(val_steps))
            except Exception as e:
                logger.warning(f"Input error, using defaults: {e}")

        kc_values = np.linspace(kc_max_val, kc_min_val, n_steps_val)

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
                        var_result["distance"], var_result["val"], counts=var_result.get("np"), model_type="spherical"
                    )
                    fitted_var_params = {"range": float(a_range), "sill": float(sill), "nugget": float(nugget)}

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
                                                        var_result["distance"], var_result["val"], counts=var_result.get("np"), model_type="spherical"
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
                if not np.any(range_mask):
                    range_mask = np.ones_like(var_result["distance"], dtype=bool)

                counts_bin = var_result.get("np")
                if counts_bin is not None and np.sum(counts_bin[range_mask]) > 0:
                    mean_variance = float(np.average(gamma_norm[range_mask], weights=np.sqrt(np.maximum(counts_bin[range_mask], 1))))
                else:
                    mean_variance = float(np.mean(gamma_norm[range_mask]))

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
        fallback_kc = float(kc_max_val)
        logger.warning(f"BSS optimization search found no valid minimum variance. Defaulting k_c to {fallback_kc:.4f} rad/m.")
        best_kc = fallback_kc

    # Prune non-optimal smoothed DEM cache in non-interactive mode to minimize memory footprint
    if not interactive:
        best_key = round(float(best_kc), 6)
        all_smoothed_dems = {best_key: best_dem_grid} if best_dem_grid is not None else {}
        all_smoothed_slopes = {best_key: best_slope_grid} if best_slope_grid is not None else {}

    opt_wl = compute_cutoff_wavelength(best_kc)
    logger.info(f"   Optimal Corner Frequency k_c = {best_kc:.4f} (cutoff wavelength λ_c = {opt_wl:.2f} m)")

    kc_var_array = np.array(all_kc_variances)
    return OptimizationResult(
        optimal_kc=float(best_kc),
        optimal_slope_grid=best_slope_grid,
        all_kc_variances=kc_var_array,
        all_smoothed_slopes=all_smoothed_slopes,
        optimal_dem_grid=best_dem_grid,
        all_smoothed_dems=all_smoothed_dems,
        opt_variogram_params=fitted_var_params,
        dx=float(dx),
        dy=float(dy),
    )
