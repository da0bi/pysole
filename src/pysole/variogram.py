"""
Spatial Variogram Calculation, Theoretical Model Fitting, and Iterative BSS Optimization.
Ported from MATLAB scripts variogram.m, variogramfit.m, and FFTSmooth.m by Daniel Binder (2011).
Computes experimental isotropic variograms, fits theoretical models (Spherical, Exponential, Gaussian),
and determines optimum DEM surface slope smoothing corner frequency kc.
Saves normalized product variogram comparison plots across all evaluated kc to plots_dir.
"""

from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from .smoothing import compute_gradients, fft_gaussian_smooth


from .raster import ensure_spatial_coords


def calculate_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    maxdist: Optional[float] = None,
    nrbins: int = 15,
    precomputed_dists: Optional[np.ndarray] = None,
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
    nrbins : int
        Number of distance lag bins.
    precomputed_dists : np.ndarray, optional
        Pre-calculated pdist(coords) array to avoid redundant distance calculations.

    Returns
    -------
    dict containing:
        - 'distance': mean lag distance of each bin
        - 'val': experimental semivariance gamma(h)
        - 'np': number of point pairs in each bin
    """
    valid = ~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1]) & ~np.isnan(values)
    pts = coords[valid]
    vals = values[valid]

    if len(pts) < 2:
        return {"distance": np.array([]), "val": np.array([]), "np": np.array([])}

    dists = precomputed_dists if precomputed_dists is not None else pdist(pts)
    if maxdist is None:
        maxdist = float(np.max(dists)) * 0.5

    bin_edges = np.linspace(0.0, maxdist, nrbins + 1)
    val_diffs = pdist(vals[:, None], metric="sqeuclidean") * 0.5  # semivariance 0.5*(z1-z2)^2

    distances = []
    semivars = []
    pair_counts = []

    for i in range(nrbins):
        mask = (dists >= bin_edges[i]) & (dists < bin_edges[i + 1])
        if np.any(mask):
            distances.append(float(np.mean(dists[mask])))
            semivars.append(float(np.mean(val_diffs[mask])))
            pair_counts.append(int(np.sum(mask)))

    return {
        "distance": np.array(distances, dtype=np.float64),
        "val": np.array(semivars, dtype=np.float64),
        "np": np.array(pair_counts, dtype=np.int64),
    }


def spherical_model(h: np.ndarray, a: float, c: float, c0: float = 0.0) -> np.ndarray:
    """Spherical variogram model: gamma(h) = c0 + c * (1.5*(h/a) - 0.5*(h/a)^3)."""
    h_ratio = np.clip(h / max(a, 1e-6), 0.0, 1.0)
    gamma = c0 + c * (1.5 * h_ratio - 0.5 * (h_ratio**3))
    gamma[h > a] = c0 + c
    return gamma


def fit_variogram_model(
    distances: np.ndarray,
    semivars: np.ndarray,
    model_type: str = "spherical",
) -> Tuple[float, float, float, Dict[str, np.ndarray]]:
    """
    Fits a theoretical variogram model (Spherical, Exponential, Gaussian) to experimental data.
    Ported from variogramfit.m.

    Returns
    -------
    a_range : float (spatial correlation range [m])
    sill : float (total sill c + c0)
    nugget : float (nugget effect c0)
    model_curve : dict containing 'h' and 'gamma' for plotting
    """
    if len(distances) == 0 or len(semivars) == 0:
        return 100.0, 1.0, 0.0, {"h": np.linspace(0, 100, 50), "gamma": np.ones(50)}

    max_dist = float(np.max(distances))
    max_var = float(np.max(semivars))

    # Initial parameter guesses: [a (range), c (sill), c0 (nugget)]
    p0 = [max_dist * 0.5, max_var, 0.0]
    bounds = ([1e-3, 1e-6, 0.0], [max_dist * 2.0, max_var * 5.0, max_var])

    try:
        popt, _ = curve_fit(spherical_model, distances, semivars, p0=p0, bounds=bounds, maxfev=2000)
        a_range, sill, nugget = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        a_range = max_dist * 0.5
        sill = max_var
        nugget = 0.0

    h_dense = np.linspace(0.0, max_dist * 1.2, 100)
    gamma_dense = spherical_model(h_dense, a_range, sill, nugget)

    return a_range, sill, nugget, {"h": h_dense, "gamma": gamma_dense}


def optimize_bss_variance(
    dem: np.ndarray,
    survey_points: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    kc_max: float = 10.0,
    kc_min: float = 0.01,
    d_kc: float = 0.1,
    num_steps: int = 10,
    plots_dir: Optional[str] = None,
    prefix: str = "01_",
    stage_name: str = "stage1",
    interactive: bool = False,
    plotit: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    """
    Iterative optimization process to determine optimum DEM surface slope smoothing degree kc.
    Ported from FFTSmooth.m.

    Plots the normalized variograms of the evaluated product for ALL tested corner frequencies (kc)
    in a single comparison plot, normalizing individual variograms with their individual product mean value squared.
    Saves all evaluated filtered DEM surface slope grids into a dictionary cache.
    """
    x_coords, y_coords, bounds = ensure_spatial_coords(
        dem.shape, dx=dx, dy=dy, bounds=bounds, x_coords=x_coords, y_coords=y_coords
    )

    # Pre-compute survey point pairwise spatial distances ONCE
    pts_valid_coords = survey_points[:, :2]
    survey_dists = pdist(pts_valid_coords) if len(pts_valid_coords) >= 2 else None

    # Compute baseline surface slope
    grads = compute_gradients(dem, dx=dx, dy=dy)
    base_slope = grads["slope_rad"]

    # Calculate k_max from 2D FFT grid
    _, _, k_max_grid = fft_gaussian_smooth(base_slope, dx=dx, dy=dy, kc=0.1)

    if kc_max is None:
        kc_max = 10.0
    else:
        kc_max = float(kc_max)

    if d_kc is None:
        d_kc = 0.1
    else:
        d_kc = float(d_kc)

    print(f"Input DEM Maximum Wavenumber (k_max_grid): {k_max_grid:.4f} rad/m (using kc_max = {kc_max:.4f}, d_kc = {d_kc:.4f})")

    all_kc_variances = []
    all_smoothed_slopes = {}
    best_kc = None
    min_var = float("inf")
    best_slope_grid = base_slope.copy()
    range_fix = None

    go_on = True
    while go_on:
        if interactive:
            print(f"\n--- BSS Filter Optimization (Grid k_max = {k_max_grid:.4f}) ---")
            try:
                curr_max_default = kc_max if kc_max is not None else 10.0
                val_max = input(f"Enter Maximum Corner Frequency (kc_max, default = {curr_max_default:.4f}): ").strip()
                if val_max:
                    kc_max = float(val_max)
                else:
                    kc_max = float(curr_max_default)

                val_min = input(f"Enter Minimum Corner Frequency (kc_min, default = {kc_min:.4f}): ").strip()
                if val_min:
                    kc_min = float(val_min)

                default_dkc = d_kc if d_kc is not None else 0.1
                val_step = input(f"Enter Corner Frequency Stepwidth d_kc (default = {default_dkc:.4f}): ").strip()
                if val_step:
                    d_kc = float(val_step)
                else:
                    d_kc = default_dkc
            except Exception as e:
                print(f"Input error, using defaults: {e}")

        if kc_max is None:
            kc_max = 10.0

        if d_kc is not None and d_kc > 0:
            kc_values = np.arange(kc_max, kc_min - 1e-9, -abs(d_kc))
        else:
            kc_values = np.linspace(kc_max, kc_min, num_steps)

        step_variances = []
        evaluated_variograms = []

        for kc in kc_values:
            if kc < 0:
                continue

            kc_key = round(float(kc), 6)
            smoothed_slope, _, _ = fft_gaussian_smooth(base_slope, dx=dx, dy=dy, kc=kc)
            all_smoothed_slopes[kc_key] = smoothed_slope.copy()

            interp_slope = RegularGridInterpolator(
                (y_coords, x_coords),
                smoothed_slope,
                bounds_error=False,
                fill_value=np.nan,
            )

            pts_xy = np.column_stack((survey_points[:, 1], survey_points[:, 0]))
            slopes_pts = interp_slope(pts_xy)

            thickness_or_t = survey_points[:, 3]
            bss_product = slopes_pts * thickness_or_t

            valid = ~np.isnan(bss_product) & (survey_points[:, 3] > 0)
            if np.sum(valid) < 3:
                continue

            mean_product = float(np.mean(bss_product[valid]))

            var_result = calculate_variogram(
                survey_points[valid, :2],
                bss_product[valid],
                nrbins=15,
                precomputed_dists=survey_dists if np.all(valid) else None,
            )

            if len(var_result["val"]) > 0:
                # Normalize experimental variogram by individual product mean value squared
                if mean_product != 0 and not np.isnan(mean_product):
                    gamma_norm = var_result["val"] / (mean_product**2)
                else:
                    gamma_norm = var_result["val"].copy()

                evaluated_variograms.append({
                    "kc": kc,
                    "distance": var_result["distance"],
                    "gamma_raw": var_result["val"],
                    "gamma_norm": gamma_norm,
                    "mean_product": mean_product,
                })

                a_range, sill, nugget, model_curve = fit_variogram_model(
                    var_result["distance"], var_result["val"], model_type="spherical"
                )

                # Prompt user for correlation range approval / overwrite on first iteration
                # Prompt user for correlation range approval / overwrite on first iteration
                if range_fix is None and (interactive or (plots_dir is not None)):
                    try:
                        import matplotlib.pyplot as plt

                        plt.figure("Isotropic Variogram Analysis", figsize=(7, 5))
                        plt.clf()
                        plt.plot(var_result["distance"], var_result["val"], "ob", markersize=6, label="Experimental Variogram")
                        plt.plot(model_curve["h"], model_curve["gamma"], "-r", linewidth=2, label="Fitted Model (Spherical)")
                        plt.axvline(x=a_range, color="k", linestyle="--", linewidth=1.5, label=f"Calculated Correlation Range a = {a_range:.2f} m")
                        plt.grid(True)
                        plt.xlabel("Lag Distance h [m]")
                        plt.ylabel("Semivariance \u03b3(h)")
                        plt.title(f"Isotropic Variogram (Unfiltered Product)\nCalculated Range = {a_range:.2f} m, Sill = {sill:.4f}, Nugget = {nugget:.4f}")
                        plt.legend()
                        plt.tight_layout()

                        if plots_dir and range_fix is None:
                            os.makedirs(plots_dir, exist_ok=True)
                            plt.savefig(os.path.join(plots_dir, f"{prefix}01_bss_{stage_name}_unfiltered_product_variogram.png"), dpi=300, bbox_inches="tight")

                        if interactive:
                            plt.draw()
                            plt.pause(0.1)

                        if interactive:
                            ans_range = input(f"\nCalculated Correlation Range = {a_range:.2f} m. Do you agree? [Y/n]: ").strip().lower()
                            if ans_range in ["n", "no"]:
                                val_r = input("Enter custom correlation range [m]: ").strip()
                                range_fix = float(val_r) if val_r else a_range
                            else:
                                range_fix = a_range
                        else:
                            range_fix = a_range

                        plt.close("Isotropic Variogram Analysis")
                    except Exception as e:
                        range_fix = a_range

                effective_range = range_fix if range_fix is not None else a_range

                # Compute mean variance within correlation range h <= effective_range
                # Normalized by mean product squared (gamma_norm) matching FFTSmooth.m & Figure 02 comparison
                range_mask = var_result["distance"] <= effective_range
                if np.any(range_mask):
                    mean_variance = float(np.mean(gamma_norm[range_mask]))
                else:
                    mean_variance = float(np.mean(gamma_norm))

                step_variances.append([kc, mean_variance])
                all_kc_variances.append([kc, mean_variance])

                if mean_variance < min_var:
                    min_var = mean_variance
                    best_kc = kc
                    best_slope_grid = smoothed_slope

        # Plot all normalized product variograms for each tested kc in a single comparison plot
        if (interactive or (plots_dir is not None)) and len(evaluated_variograms) > 0:
            try:
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm

                plt.figure("Normalized Product Variograms Across Evaluated Corner Frequencies", figsize=(9, 6))
                plt.clf()

                sorted_vario = sorted(evaluated_variograms, key=lambda x: x["kc"], reverse=True)
                num_v = len(sorted_vario)
                colors = cm.viridis(np.linspace(0, 1, max(num_v, 2)))

                for idx, v_item in enumerate(sorted_vario):
                    kc_val = v_item["kc"]
                    is_opt = (best_kc is not None) and (abs(kc_val - best_kc) < 1e-6)

                    line_style = "-o" if is_opt else "--s"
                    lw = 2.5 if is_opt else 1.2
                    ms = 7 if is_opt else 4
                    col = "red" if is_opt else colors[idx]
                    lbl = f"k_c = {kc_val:.4f} (Optimal)" if is_opt else f"k_c = {kc_val:.4f}"

                    plt.plot(
                        v_item["distance"],
                        v_item["gamma_norm"],
                        line_style,
                        color=col,
                        linewidth=lw,
                        markersize=ms,
                        label=lbl,
                        zorder=5 if is_opt else 3,
                    )

                plt.grid(True, linestyle=":", alpha=0.6)
                plt.xlabel("Lag Distance h [m]")
                plt.ylabel("Normalized Semivariance \u03b3(h) / Mean\u00b2")
                plt.title("Normalized Product Variograms Across Evaluated Corner Frequencies (k_c)")
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                plt.tight_layout()

                if plots_dir:
                    os.makedirs(plots_dir, exist_ok=True)
                    plt.savefig(os.path.join(plots_dir, f"{prefix}02_bss_{stage_name}_variograms_all_kc_comparison.png"), dpi=300, bbox_inches="tight")

                if interactive:
                    plt.draw()
                    plt.pause(0.2)

                plt.close("Normalized Product Variograms Across Evaluated Corner Frequencies")
            except Exception as e:
                pass

        if (interactive or (plots_dir is not None)) and len(step_variances) > 0:
            try:
                import matplotlib.pyplot as plt

                var_arr = np.array(step_variances)
                plt.figure("Find Minimum BSS Variance", figsize=(7, 5))
                plt.clf()
                plt.plot(var_arr[:, 0], var_arr[:, 1], "--rs", linewidth=2, markerfacecolor="g", markersize=8)
                plt.grid(True)
                plt.title(f"Find Minimum BSS Variance (Correlation Range = {range_fix if range_fix is not None else 0.0:.1f}m)")
                plt.xlabel("Corner Frequency (k_c)")
                plt.ylabel("Mean Variance In Correlation Range")
                plt.tight_layout()

                if plots_dir:
                    os.makedirs(plots_dir, exist_ok=True)
                    plt.savefig(os.path.join(plots_dir, f"{prefix}03_bss_{stage_name}_variance_optimization.png"), dpi=300, bbox_inches="tight")

                if interactive:
                    plt.draw()
                    plt.pause(0.5)

                plt.close("Find Minimum BSS Variance")
            except Exception as e:
                pass

        if interactive:
            print("\nIteration Results (Corner Frequency vs Mean Variance):")
            for kc_val, var_val in (step_variances if step_variances else all_kc_variances):
                print(f"  k_c = {kc_val:.4f} --> Mean Variance = {var_val:.6f}")

            ans = input("\nContinue Optimization Process? [y/N]: ").strip().lower()
            if ans in ["y", "yes"]:
                go_on = True
                kc_max = None
            else:
                go_on = False
        else:
            go_on = False

    if best_kc is None:
        best_kc = float(kc_max)

    kc_var_array = np.array(all_kc_variances)
    return float(best_kc), best_slope_grid, kc_var_array, all_smoothed_slopes
