"""
Spatial Variogram Calculation, Theoretical Model Fitting, and Iterative BSS Optimization.
Ported from MATLAB scripts variogram.m, variogramfit.m, and FFTSmooth.m by Daniel Binder (2011).
Computes experimental isotropic variograms, fits theoretical models (Spherical, Exponential, Gaussian),
and determines optimum DEM surface slope smoothing corner frequency kc.
Saves normalized product variogram comparison plots across all evaluated kc to plots_dir.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from .smoothing import (
    compute_gradients,
    precompute_fft_grid,
    fft_gaussian_smooth_precomputed,
)
from .raster import GridGeometry
from .logging import logger


@dataclass
class OptimizationResult:
    """
    Typed data container storing BSS corner frequency slope optimization outputs.
    """
    optimal_kc: float
    optimal_slope_grid: np.ndarray
    all_kc_variances: np.ndarray
    all_smoothed_slopes: Dict[float, np.ndarray]


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
        kc_max: float = 10.0,
        kc_min: float = 0.01,
        d_kc: float = 0.1,
        plots_dir: Optional[str] = None,
        prefix: str = "01_",
        stage_name: str = "stage1",
        interactive: bool = False,
        n_cores: int = -1,
        nrbins: Optional[int] = None,
    ) -> OptimizationResult:
        """Executes BSS slope filter optimization across corner frequency spectrum."""
        return optimize_bss_variance(
            dem=self.dem,
            survey_points=survey_points,
            geometry=self.geometry,
            kc_max=kc_max,
            kc_min=kc_min,
            d_kc=d_kc,
            plots_dir=plots_dir,
            prefix=prefix,
            stage_name=stage_name,
            interactive=interactive,
            n_cores=n_cores,
            nrbins=nrbins,
        )


def calculate_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    maxdist: Optional[float] = None,
    nrbins: Optional[int] = None,
    precomputed_dists: Optional[np.ndarray] = None,
    warn_low_pairs: bool = True,
) -> Dict[str, np.ndarray]:
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
        Number of distance lag bins. If None, dynamically calculated to guarantee
        at least 30 point pairs per bin (nrbins = max(3, N_pairs // 30)).
    precomputed_dists : np.ndarray, optional
        Pre-calculated pdist(coords) array to avoid redundant distance calculations.
    warn_low_pairs : bool, optional
        If True, logs a warning if avg point pairs per bin is under 30. Defaults to False.

    Returns
    -------
    result : dict
        Dict with keys: 'distance' (lag distances h), 'val' (semivariance gamma(h)),
        'np' (number of point pairs per lag bin).
    """
    n_pts = len(coords)
    if n_pts < 2:
        return {"distance": np.array([]), "val": np.array([]), "np": np.array([])}

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
    gamma = np.full_like(h, n + c)
    mask = h <= a
    gamma[mask] = n + c * (1.5 * (h[mask] / a) - 0.5 * (h[mask] / a)**3)
    return gamma


def fit_variogram_model(
    distances: np.ndarray,
    semivars: np.ndarray,
    model_type: str = "spherical",
) -> Tuple[float, float, float, Dict[str, np.ndarray]]:
    """
    Fits a theoretical variogram model (Spherical) to experimental variogram data.
    Ported from variogramfit.m.

    Returns
    -------
    a_range : float
        Calculated spatial correlation range [m].
    sill : float
        Sill parameter C.
    nugget : float
        Nugget variance C0.
    model_curve : dict
        Dict with 'h' and 'gamma' fine curve points for plotting.
    """
    if len(distances) < 3 or len(semivars) < 3:
        a_default = float(np.max(distances)) if len(distances) > 0 else 1000.0
        sill_default = float(np.var(semivars)) if len(semivars) > 0 else 1.0
        h_fine = np.linspace(0, a_default * 1.5, 100)
        return a_default, sill_default, 0.0, {"h": h_fine, "gamma": np.full_like(h_fine, sill_default)}

    max_dist = float(np.max(distances))
    var_val = float(np.var(semivars)) if np.var(semivars) > 0 else float(np.mean(semivars))
    p0 = [max_dist * 0.5, var_val * 0.8, 0.0]
    bounds = ([1e-3, 1e-6, 0.0], [max_dist * 3.0, var_val * 10.0, var_val * 2.0])

    try:
        popt, _ = curve_fit(spherical_variogram, distances, semivars, p0=p0, bounds=bounds, maxfev=2000)
        a_range, sill, nugget = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        a_range = max_dist * 0.5
        sill = var_val
        nugget = 0.0

    h_fine = np.linspace(0, max_dist * 1.2, 150)
    gamma_fine = spherical_variogram(h_fine, a_range, sill, nugget)
    model_curve = {"h": h_fine, "gamma": gamma_fine}

    return a_range, sill, nugget, model_curve


def optimize_bss_variance(
    dem: np.ndarray,
    survey_points: np.ndarray,
    geometry: GridGeometry,
    kc_max: float = 10.0,
    kc_min: float = 0.01,
    d_kc: float = 0.1,
    plots_dir: Optional[str] = None,
    prefix: str = "01_",
    stage_name: str = "stage1",
    interactive: bool = False,
    n_cores: int = -1,
    nrbins: Optional[int] = None,
) -> OptimizationResult:
    """
    Iterative optimization process to determine optimum DEM surface slope smoothing degree kc.
    Ported from FFTSmooth.m.

    Plots the normalized variograms of the evaluated product for ALL tested corner frequencies (kc)
    in a single comparison plot, normalizing individual variograms with their individual product mean value squared.
    Saves all evaluated filtered DEM surface slope grids into a dictionary cache.
    Accelerated with multi-core CPU parallelization via n_cores.
    If interactive is True, enables an interactive CLI prompt allowing inspection of the BSS variance curve
    and adjustment of corner frequency search spectrum (kc_min, kc_max, d_kc), lag distance bin count (nrbins),
    and correlation range (a_range).
    """
    dx = geometry.dx
    dy = geometry.dy
    x_coords = geometry.x_coords
    y_coords = geometry.y_coords

    effective_n_cores = (os.cpu_count() or 1) if (n_cores == -1 or n_cores is None) else max(1, int(n_cores))

    # Pre-filter valid survey points ONCE and compute distance matrix ONCE
    valid_pts_mask = ~np.isnan(survey_points[:, 0]) & ~np.isnan(survey_points[:, 1]) & (survey_points[:, 3] > 0)
    pts_valid_coords = survey_points[valid_pts_mask, :2]
    survey_dists = pdist(pts_valid_coords) if len(pts_valid_coords) >= 2 else None

    # Compute baseline surface slope
    grads = compute_gradients(dem, dx=dx, dy=dy)
    base_slope = grads["slope_rad"]

    # Pre-compute 2D Forward FFT and wavenumber grid ONCE for the entire optimization loop
    A_shift_slope, k_grid_slope, k_max_grid = precompute_fft_grid(base_slope, dx=dx, dy=dy)

    kc_max = float(kc_max) if kc_max is not None else 10.0
    kc_min = float(kc_min) if kc_min is not None else 0.01
    d_kc = float(d_kc) if d_kc is not None else 0.1

    logger.info(f"   Input DEM Maximum Wavenumber (k_max_grid): {k_max_grid:.4f} rad/m (using kc_max = {kc_max:.4f}, kc_min = {kc_min:.4f}, d_kc = {d_kc:.4f}, n_cores = {effective_n_cores})")

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
    best_kc = None
    min_var = float("inf")
    best_slope_grid = base_slope.copy()
    range_fix = None

    def _eval_single_kc(kc_val: float):
        if kc_val < 0:
            return None

        kc_key = round(float(kc_val), 6)
        smoothed_slope = fft_gaussian_smooth_precomputed(A_shift_slope, k_grid_slope, kc=kc_val)

        interp_slope = geometry.create_interpolator(smoothed_slope, fill_value=np.nan)
        pts_xy = np.column_stack((survey_points[:, 1], survey_points[:, 0]))
        slopes_pts = interp_slope(pts_xy)

        thickness_or_t = survey_points[:, 3]
        bss_product = slopes_pts * thickness_or_t

        valid = ~np.isnan(bss_product) & (survey_points[:, 3] > 0)
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

        return (kc_val, kc_key, smoothed_slope, var_result, mean_product)

    go_on = True
    while go_on:
        if interactive:
            logger.info(f"--- BSS Filter Optimization (Grid k_max = {k_max_grid:.4f}) ---")
            try:
                curr_max_default = kc_max
                val_max = input(f"Enter Maximum Corner Frequency (kc_max, default = {curr_max_default:.4f}): ").strip()
                if val_max:
                    kc_max = float(val_max)

                val_min = input(f"Enter Minimum Corner Frequency (kc_min, default = {kc_min:.4f}): ").strip()
                if val_min:
                    kc_min = float(val_min)

                val_step = input(f"Enter Corner Frequency Stepwidth d_kc (default = {d_kc:.4f}): ").strip()
                if val_step:
                    d_kc = float(val_step)
            except Exception as e:
                logger.warning(f"Input error, using defaults: {e}")

        effective_dkc = abs(d_kc) if d_kc > 0 else 0.1
        kc_values = np.arange(kc_max, kc_min - 1e-9, -effective_dkc)
        if len(kc_values) > 0 and kc_values[-1] > kc_min + 1e-6:
            kc_values = np.append(kc_values, kc_min)

        step_variances = []
        evaluated_variograms = []

        valid_kc_list = [float(kc) for kc in kc_values if kc >= 0]
        if len(valid_kc_list) == 0:
            break

        # Step 1: Evaluate baseline/first kc if range_fix is not set yet
        if range_fix is None:
            first_eval = _eval_single_kc(valid_kc_list[0])
            if first_eval is not None:
                kc_val, kc_key, smoothed_slope, var_result, mean_product = first_eval
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
                                                _, _, _, var_result, _ = first_eval
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
            kc_eval_results = list(executor.map(_eval_single_kc, valid_kc_list))

        for res in kc_eval_results:
            if res is None:
                continue

            kc_val, kc_key, smoothed_slope, var_result, mean_product = res
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

    kc_var_array = np.array(all_kc_variances)
    return OptimizationResult(
        optimal_kc=float(best_kc),
        optimal_slope_grid=best_slope_grid,
        all_kc_variances=kc_var_array,
        all_smoothed_slopes=all_smoothed_slopes,
    )
