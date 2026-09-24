"""
Dedicated Visualization & Plotting Module for PySole.
Provides high-quality Matplotlib figures, contours, colorbars, and multi-panel plots for
variogram optimization, 3D ray migration vectors, Kriging uncertainty, and bedrock maps.
"""

import os
import warnings
from typing import Optional, List, Tuple, Dict, Any
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _get_transparent_cmap(cmap_name: str):
    """Returns a Matplotlib colormap with transparent NaN background."""
    cmap = plt.get_cmap(cmap_name).copy()
    if hasattr(cmap, "with_extremes"):
        return cmap.with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        cmap.set_bad(color="white", alpha=0.0)
    return cmap


def plot_unfiltered_product_variogram(
    distances: np.ndarray,
    semivars: np.ndarray,
    model_curve: Dict[str, np.ndarray],
    a_range: float,
    sill: float,
    nugget: float,
    plots_dir: Optional[str] = None,
    prefix: str = "01_",
    stage_name: str = "stage1",
    interactive: bool = False,
):
    """Plots the experimental isotropic variogram and fitted theoretical model for initial product."""
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure("Isotropic Variogram Analysis", figsize=(7, 5))
        plt.clf()
        plt.plot(distances, semivars, "ob", markersize=6, label="Experimental Variogram")
        plt.plot(model_curve["h"], model_curve["gamma"], "-r", linewidth=2, label="Fitted Model (Spherical)")
        plt.axvline(x=a_range, color="k", linestyle="--", linewidth=1.5, label=f"Calculated Correlation Range a = {a_range:.2f} m")
        plt.grid(True)
        plt.xlabel("Lag Distance h [m]")
        plt.ylabel("Semivariance \u03b3(h)")
        plt.title(f"Isotropic Variogram (Unfiltered Product)\nCalculated Range = {a_range:.2f} m, Sill = {sill:.4f}, Nugget = {nugget:.4f}")
        plt.legend()
        plt.tight_layout()

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, f"{prefix}01_bss_{stage_name}_unfiltered_product_variogram.png"), dpi=300, bbox_inches="tight")

        if interactive:
            plt.draw()
            plt.pause(0.1)

        plt.close("Isotropic Variogram Analysis")
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_bss_kc_optimization_variograms(
    evaluated_variograms: List[Dict[str, Any]],
    best_kc: float,
    all_kc_variances: List[Tuple[float, float]],
    plots_dir: Optional[str] = None,
    prefix: str = "01_",
    stage_name: str = "stage1",
    interactive: bool = False,
):
    """Plots normalized product variogram comparison for ALL evaluated corner frequencies kc."""
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        cmap = matplotlib.colormaps["viridis"].resampled(max(len(evaluated_variograms), 1))
        for idx, item in enumerate(evaluated_variograms):
            kc_val = item["kc"]
            dist = item["distance"]
            gamma_norm = item["gamma_norm"]

            if round(float(kc_val), 6) == round(float(best_kc), 6):
                ax1.plot(dist, gamma_norm, "-o", color="red", linewidth=2.5, markersize=6, label=f"Optimum kc = {kc_val:.3f}")
            else:
                ax1.plot(dist, gamma_norm, "--", color=cmap(idx), linewidth=1.0, alpha=0.7, label=f"kc = {kc_val:.3f}")

        ax1.set_xlabel("Lag Distance h [m]")
        ax1.set_ylabel("Normalized Semivariance \u03b3(h) / Mean\u00b2")
        ax1.set_title(f"Normalized Variograms across Corner Frequencies (kc)\nOptimum kc = {best_kc:.4f}")
        ax1.grid(True)
        if len(evaluated_variograms) <= 12:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        if len(all_kc_variances) > 0:
            kcs, vars_vals = zip(*all_kc_variances)
            ax2.plot(kcs, vars_vals, "s-b", linewidth=2, markersize=6)
            ax2.axvline(x=best_kc, color="r", linestyle="--", label=f"Min Variance kc = {best_kc:.3f}")
            ax2.set_xlabel("Corner Frequency kc [rad/m]")
            ax2.set_ylabel("Product Field Variance")
            ax2.set_title("BSS Variance vs Corner Frequency kc")
            ax2.grid(True)
            ax2.legend()

        plt.tight_layout()

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, f"{prefix}02_bss_{stage_name}_kc_optimization_variograms.png"), dpi=300, bbox_inches="tight")

        if interactive:
            plt.draw()
            plt.pause(0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_migration_displacement_vectors(
    travel_time_grid: np.ndarray,
    survey_points: np.ndarray,
    migrated_points: np.ndarray,
    dxi: np.ndarray,
    dyi: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    outline_mask: Optional[np.ndarray] = None,
    plots_dir: Optional[str] = None,
    interactive: bool = False,
):
    """Plots 2-panel figure matching MIG.m: traveltime field with horizontal ray displacement vectors and migrated survey points."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        plot_extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

        im1 = ax1.imshow(travel_time_grid, extent=plot_extent, origin="lower", cmap="viridis")
        min_tt = np.nanmin(travel_time_grid)
        max_tt = np.nanmax(travel_time_grid)
        if np.isfinite(min_tt) and np.isfinite(max_tt) and max_tt > min_tt:
            cnt1 = ax1.contour(
                travel_time_grid,
                extent=plot_extent,
                origin="lower",
                colors="white",
                linewidths=0.5,
                alpha=0.7,
            )
            ax1.clabel(cnt1, inline=True, fmt="%d ns", fontsize=8)

        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im1, cax=cax1, label="Traveltime [ns]")

        if outline_mask is not None:
            ax1.contour(
                outline_mask,
                levels=[0.5],
                extent=plot_extent,
                origin="lower",
                colors="black",
                linewidths=0.8,
                linestyles="solid",
            )

        # Downsample arrows for clean plot
        step = max(len(survey_points) // 80, 1)
        ax1.quiver(
            survey_points[::step, 0],
            survey_points[::step, 1],
            dxi[::step],
            dyi[::step],
            color="white",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            zorder=4,
        )
        ax1.set_title("Calculated Traveltime Field and 3D Migration Horizontal Displacement Vectors")
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")

        # Lower Subplot: Pre-migrated points (black dots) and Migrated points (colored by depth)
        x_mig = migrated_points[:, 0]
        y_mig = migrated_points[:, 1]
        d_mig = migrated_points[:, 3]

        ax2.scatter(
            survey_points[:, 0],
            survey_points[:, 1],
            c="black",
            s=1.5,
            alpha=0.7,
            label="Pre-migrated Points",
            zorder=2,
        )
        sc = ax2.scatter(x_mig, y_mig, c=d_mig, s=15, cmap="jet", label="Migrated Points", zorder=3)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(sc, cax=cax2, label="Migrated Depth [m]")

        if outline_mask is not None:
            ax2.contour(
                outline_mask,
                levels=[0.5],
                extent=plot_extent,
                origin="lower",
                colors="black",
                linewidths=0.8,
                linestyles="solid",
            )

        ax1.set_aspect("equal")
        ax2.set_aspect("equal")

        ax2.set_title("Migrated Survey Point Depths [m]")
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")
        ax2.legend(loc="upper right", fontsize=8)

        plt.tight_layout()

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(
                os.path.join(plots_dir, "02_01_eikonal_migration_displacement_vectors.png"),
                dpi=300,
                bbox_inches="tight",
            )

        if interactive:
            plt.draw()
            plt.pause(0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_kriging_bedrock_and_uncertainty(
    kriged_bedrock: np.ndarray,
    kriged_std: Optional[np.ndarray] = None,
    pts: Optional[np.ndarray] = None,
    plot_extent: Optional[List[float]] = None,
    kriged_variance: Optional[np.ndarray] = None,
    plots_dir: Optional[str] = None,
    interactive: bool = False,
):
    """Plots 2-panel figure showing interpolated bedrock elevation and Kriging standard uncertainty (± m)."""
    try:
        import matplotlib.pyplot as plt

        if kriged_std is None and kriged_variance is not None:
            unc_grid = np.sqrt(np.maximum(np.nan_to_num(kriged_variance, nan=0.0), 0.0))
            if np.isnan(kriged_variance).any():
                unc_grid[np.isnan(kriged_variance)] = np.nan
        elif kriged_std is not None:
            unc_grid = kriged_std
        else:
            unc_grid = np.zeros_like(kriged_bedrock)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        im1 = ax1.imshow(kriged_bedrock, extent=plot_extent, origin="lower", cmap="terrain")
        if pts is not None and len(pts) > 0:
            ax1.scatter(pts[:, 0], pts[:, 1], c="black", s=1.5, alpha=0.7, label="Survey Points")
            ax1.legend()
        ax1.set_title("Interpolated Bedrock Elevation")
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        fig.colorbar(im1, ax=ax1, label="Elevation [m]")

        cmap2 = _get_transparent_cmap("plasma")
        valid_unc = unc_grid[np.isfinite(unc_grid) & (unc_grid > 0)]
        if len(valid_unc) > 0:
            vmax2 = float(np.percentile(valid_unc, 98.5))
            vmax2 = max(vmax2, 1.0)
        else:
            vmax2 = None

        im2 = ax2.imshow(unc_grid, extent=plot_extent, origin="lower", cmap=cmap2, vmin=0, vmax=vmax2)
        ax2.set_title("Kriging Standard Uncertainty")
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")
        fig.colorbar(im2, ax=ax2, label="Standard Uncertainty [\u00b1 m]", extend="max" if vmax2 is not None else "neither")

        plt.tight_layout()

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, "04_01_kriging_bedrock_elevation_and_uncertainty.png"), dpi=300, bbox_inches="tight")

        if interactive:
            plt.draw()
            plt.pause(0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_calculated_bedrock_map(
    bedrock_grid: np.ndarray,
    outline_mask: Optional[np.ndarray],
    plot_extent: List[float],
    smooth_bedrock: bool = False,
    smoothing_sigma: float = 1.5,
    plots_dir: Optional[str] = None,
    interactive: bool = False,
):
    """Plots calculated bedrock elevation with surrounding terrain contours."""
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure("Calculated Bedrock Map", figsize=(8, 6))
        plt.clf()
        plt.imshow(bedrock_grid, extent=plot_extent, origin="lower", cmap="terrain")
        plt.colorbar(label="Elevation [m]")
        min_b = np.nanmin(bedrock_grid)
        max_b = np.nanmax(bedrock_grid)

        if smooth_bedrock:
            from scipy.ndimage import gaussian_filter
            contour_grid = gaussian_filter(bedrock_grid, sigma=max(smoothing_sigma, 2.5))
        else:
            contour_grid = bedrock_grid

        if np.isfinite(min_b) and np.isfinite(max_b) and max_b > min_b:
            levels_10m = np.arange(np.floor(min_b / 10.0) * 10.0, np.ceil(max_b / 10.0) * 10.0 + 10.0, 10.0)
            cnt = plt.contour(
                contour_grid,
                levels=levels_10m,
                extent=plot_extent,
                origin="lower",
                colors="black",
                linewidths=0.5,
                alpha=0.7,
            )
        else:
            cnt = plt.contour(
                contour_grid,
                extent=plot_extent,
                origin="lower",
                colors="black",
                linewidths=0.5,
                alpha=0.7,
            )
        plt.clabel(cnt, inline=True, fmt="%d m", fontsize=8)

        if outline_mask is not None:
            plt.contour(
                outline_mask,
                levels=[0.5],
                extent=plot_extent,
                origin="lower",
                colors="black",
                linewidths=1.0,
                linestyles="solid",
            )

        plt.title("Calculated Bedrock Elevation with Surrounding Terrain")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.tight_layout()

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, "05_01_calculated_bedrock_map.png"), dpi=300, bbox_inches="tight")

        if interactive:
            plt.draw()
            plt.pause(0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_final_blended_bedrock_map(
    blended_bedrock: np.ndarray,
    plot_extent: List[float],
    plots_dir: Optional[str] = None,
    interactive: bool = False,
):
    """Plots final geomorphologically blended bedrock elevation raster."""
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure("Final Blended Bedrock Map", figsize=(8, 6))
        plt.clf()
        plt.imshow(blended_bedrock, extent=plot_extent, origin="lower", cmap="terrain")
        plt.colorbar(label="Elevation [m]")
        plt.title("Final Geomorphologically Blended Bedrock Elevation")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.tight_layout()

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, "06_01_final_blended_bedrock_map.png"), dpi=300, bbox_inches="tight")

        if interactive:
            plt.draw()
            plt.pause(0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_final_ice_thickness_and_uncertainty(
    final_thickness: np.ndarray,
    kriged_std: np.ndarray,
    outline_mask: Optional[np.ndarray],
    plot_extent: List[float],
    mean_thick: float,
    mean_unc: float,
    plots_dir: Optional[str] = None,
    interactive: bool = False,
):
    """Plots final ice thickness map and Kriging standard uncertainty field (+- meters)."""
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        thick_draw = final_thickness.copy()
        if outline_mask is not None:
            thick_draw[~outline_mask] = np.nan

        cmap1 = _get_transparent_cmap("YlGnBu")
        im1 = ax1.imshow(thick_draw, extent=plot_extent, origin="lower", cmap=cmap1)
        fig.colorbar(im1, ax=ax1, label="Thickness [m]")
        if outline_mask is not None:
            ax1.contour(
                outline_mask,
                levels=[0.5],
                extent=plot_extent,
                origin="lower",
                colors="black",
                linewidths=1.5,
                linestyles="--",
            )
        ax1.set_title(f"Mean Thickness: {int(round(mean_thick))} m")
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")

        cmap2 = _get_transparent_cmap("magma")
        valid_unc = kriged_std[np.isfinite(kriged_std) & (kriged_std > 0)]
        if len(valid_unc) > 0:
            vmax2 = float(np.percentile(valid_unc, 98.5))
            vmax2 = max(vmax2, 1.0)
        else:
            vmax2 = None

        im2 = ax2.imshow(kriged_std, extent=plot_extent, origin="lower", cmap=cmap2, vmin=0, vmax=vmax2)
        fig.colorbar(im2, ax=ax2, label="Kriging Uncertainty [\u00b1 m]", extend="max" if vmax2 is not None else "neither")
        if outline_mask is not None:
            ax2.contour(
                outline_mask,
                levels=[0.5],
                extent=plot_extent,
                origin="lower",
                colors="black",
                linewidths=1.5,
                linestyles="--",
            )
        ax2.set_title(f"Mean Kriging Uncertainty: \u00b1{int(round(mean_unc))} m")
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")

        plt.tight_layout()

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, "07_01_final_ice_thickness_map.png"), dpi=300, bbox_inches="tight")

        if interactive:
            plt.draw()
            plt.pause(0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_final_ice_thickness_histogram(
    final_thickness: np.ndarray,
    outline_mask: Optional[np.ndarray],
    mean_thick: float,
    plots_dir: Optional[str] = None,
    interactive: bool = False,
):
    """Plots individual histogram of final ice depth distribution indicating mean ice depth."""
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure("Final Ice Thickness Histogram", figsize=(7, 5))
        plt.clf()

        thick_valid = final_thickness[np.isfinite(final_thickness) & (final_thickness > 0)]
        if outline_mask is not None and np.any(outline_mask):
            thick_valid = final_thickness[outline_mask & np.isfinite(final_thickness) & (final_thickness > 0)]

        if len(thick_valid) > 0:
            n_bins = max(15, min(50, len(thick_valid) // 100))
            plt.hist(thick_valid, bins=n_bins, color="#38bdf8", edgecolor="#1e293b", alpha=0.85, label="Ice Depth Frequency")
            median_thick = float(np.median(thick_valid))
            max_thick = float(np.max(thick_valid))
            std_thick = float(np.std(thick_valid))
        else:
            median_thick = mean_thick
            max_thick = 0.0
            std_thick = 0.0

        plt.axvline(x=mean_thick, color="#ef4444", linestyle="--", linewidth=2.0, label=f"Mean Ice Depth = {mean_thick:.1f} m")

        stats_text = (
            f"Mean: {mean_thick:.1f} m\n"
            f"Median: {median_thick:.1f} m\n"
            f"Std Dev: \u00b1{std_thick:.1f} m\n"
            f"Max Depth: {max_thick:.1f} m"
        )
        plt.gca().text(
            0.95, 0.95, stats_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="#cbd5e1")
        )

        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xlabel("Ice Depth / Thickness [m]")
        plt.ylabel("Pixel Frequency Count")
        plt.title(f"Final Ice Thickness Distribution\nMean = {mean_thick:.1f} m (Max = {max_thick:.1f} m)")
        plt.legend(loc="upper left")
        plt.tight_layout()

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, "07_02_final_ice_thickness_histogram.png"), dpi=300, bbox_inches="tight")

        if interactive:
            plt.draw()
            plt.pause(0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_final_basal_shear_stress_and_uncertainty(
    final_bss: np.ndarray,
    bss_std: np.ndarray,
    outline_mask: Optional[np.ndarray],
    plot_extent: List[float],
    mean_bss: float,
    mean_bss_unc: float,
    plots_dir: Optional[str] = None,
    interactive: bool = False,
):
    """Plots 2-panel figure showing Basal Shear Stress distribution (\u03c4b [kPa]) and Kriging BSS uncertainty (\u00b1 kPa)."""
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        bss_draw = final_bss.copy()
        if outline_mask is not None:
            bss_draw[~outline_mask] = np.nan

        cmap1 = _get_transparent_cmap("viridis")
        valid_bss = bss_draw[np.isfinite(bss_draw) & (bss_draw > 0)]
        vmax1 = float(np.percentile(valid_bss, 98.5)) if len(valid_bss) > 0 else None

        im1 = ax1.imshow(bss_draw, extent=plot_extent, origin="lower", cmap=cmap1, vmin=0, vmax=vmax1)
        fig.colorbar(im1, ax=ax1, label="Basal Shear Stress \u03c4b [kPa]", extend="max" if vmax1 is not None else "neither")
        if outline_mask is not None:
            ax1.contour(
                outline_mask,
                levels=[0.5],
                extent=plot_extent,
                origin="lower",
                colors="black",
                linewidths=1.5,
                linestyles="--",
            )
        ax1.set_title(f"Mean Basal Shear Stress: {mean_bss:.1f} kPa")
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")

        cmap2 = _get_transparent_cmap("plasma")
        valid_unc = bss_std[np.isfinite(bss_std) & (bss_std > 0)]
        if len(valid_unc) > 0:
            vmax2 = float(np.percentile(valid_unc, 98.5))
            vmax2 = max(vmax2, 0.1)
        else:
            vmax2 = None

        im2 = ax2.imshow(bss_std, extent=plot_extent, origin="lower", cmap=cmap2, vmin=0, vmax=vmax2)
        fig.colorbar(im2, ax=ax2, label="BSS Standard Uncertainty [\u00b1 kPa]", extend="max" if vmax2 is not None else "neither")
        if outline_mask is not None:
            ax2.contour(
                outline_mask,
                levels=[0.5],
                extent=plot_extent,
                origin="lower",
                colors="black",
                linewidths=1.5,
                linestyles="--",
            )
        ax2.set_title(f"Mean BSS Uncertainty: \u00b1{mean_bss_unc:.1f} kPa")
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")

        plt.tight_layout()

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, "08_01_final_basal_shear_stress_map.png"), dpi=300, bbox_inches="tight")

        if interactive:
            plt.draw()
            plt.pause(0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_final_basal_shear_stress_histogram(
    final_bss: np.ndarray,
    outline_mask: Optional[np.ndarray],
    mean_bss: float,
    plots_dir: Optional[str] = None,
    interactive: bool = False,
):
    """Plots individual histogram of Basal Shear Stress distribution (\u03c4b [kPa]) indicating mean BSS value."""
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure("Final Basal Shear Stress Histogram", figsize=(7, 5))
        plt.clf()

        bss_valid = final_bss[np.isfinite(final_bss) & (final_bss > 0)]
        if outline_mask is not None and np.any(outline_mask):
            bss_valid = final_bss[outline_mask & np.isfinite(final_bss) & (final_bss > 0)]

        if len(bss_valid) > 0:
            n_bins = max(15, min(50, len(bss_valid) // 100))
            plt.hist(bss_valid, bins=n_bins, color="#10b981", edgecolor="#1e293b", alpha=0.85, label="BSS Frequency")
            median_bss = float(np.median(bss_valid))
            max_bss = float(np.max(bss_valid))
            std_bss = float(np.std(bss_valid))
        else:
            median_bss = mean_bss
            max_bss = 0.0
            std_bss = 0.0

        plt.axvline(x=mean_bss, color="#ef4444", linestyle="--", linewidth=2.0, label=f"Mean BSS = {mean_bss:.1f} kPa")

        stats_text = (
            f"Mean: {mean_bss:.1f} kPa\n"
            f"Median: {median_bss:.1f} kPa\n"
            f"Std Dev: \u00b1{std_bss:.1f} kPa\n"
            f"Max BSS: {max_bss:.1f} kPa"
        )
        plt.gca().text(
            0.95, 0.95, stats_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="#cbd5e1")
        )

        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xlabel("Basal Shear Stress \u03c4b [kPa]")
        plt.ylabel("Pixel Frequency Count")
        plt.title(f"Final Basal Shear Stress Distribution\nMean = {mean_bss:.1f} kPa (Max = {max_bss:.1f} kPa)")
        plt.legend(loc="upper left")
        plt.tight_layout()

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, "08_02_final_basal_shear_stress_histogram.png"), dpi=300, bbox_inches="tight")

        if interactive:
            plt.draw()
            plt.pause(0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)
