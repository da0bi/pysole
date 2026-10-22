"""
Dedicated Visualization & Plotting Module for PySole.
Provides high-quality Matplotlib figures, contours, colorbars, and multi-panel plots for
variogram optimization, 3D ray migration vectors, Kriging uncertainty, and bedrock maps.
"""

import os
from pathlib import Path
import warnings
from typing import Any
import matplotlib
if not os.environ.get("DISPLAY") and matplotlib.get_backend().lower() not in ["agg", "module://matplotlib_inline.backend_inline"]:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import numpy as np
from .logging import logger


def _save_figure(plt_obj: Any, plots_dir: str | Path | os.PathLike | None, filename: str) -> Path | None:
    """
    Save an active Matplotlib figure to the specified directory.

    Creates parent directories if necessary and renders the figure at 300 DPI
    with tight bounding box cropping.

    Parameters
    ----------
    plt_obj : Any
        Matplotlib `pyplot` module or `Figure` object.
    plots_dir : str, Path, os.PathLike, or None
        Destination directory path. If None or empty, no file is saved.
    filename : str
        Target image filename (e.g., '01_unfiltered_variogram.png').

    Returns
    -------
    target_path : Path or None
        Resolved `Path` of the saved figure, or None if `plots_dir` was empty.
    """
    if not plots_dir:
        return None
    target_dir = Path(plots_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    plt_obj.savefig(target_path, dpi=300, bbox_inches="tight")
    logger.info(f"   Saved diagnostic figure to: {target_path}")
    return target_path


def _safe_interactive_pause(plt_obj: Any, duration: float = 0.5) -> None:
    """
    Safely trigger interactive figure rendering without emitting warnings on headless backends.

    Parameters
    ----------
    plt_obj : Any
        Matplotlib `pyplot` module.
    duration : float, default=0.5
        Pause duration in seconds for GUI event loop processing.

    Returns
    -------
    None
    """
    try:
        if str(matplotlib.get_backend()).lower() != "agg":
            plt_obj.draw()
            plt_obj.pause(duration)
    except Exception:
        pass


def _get_transparent_cmap(cmap_name: str) -> matplotlib.colors.Colormap:
    """
    Return a Matplotlib colormap configured with a fully transparent NaN background.

    Parameters
    ----------
    cmap_name : str
        Name of the base Matplotlib colormap (e.g., 'plasma', 'viridis', 'magma').

    Returns
    -------
    matplotlib.colors.Colormap
        Colormap copy with the 'bad' (NaN / NoData) color assigned an alpha of 0.0.
    """
    cmap = plt.get_cmap(cmap_name).copy()
    if hasattr(cmap, "with_extremes"):
        return cmap.with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        cmap.set_bad(color="white", alpha=0.0)
    return cmap


def _get_cell_edge_extent(x_coords: np.ndarray, y_coords: np.ndarray) -> list[float]:
    """
    Calculate cell-edge spatial extent [left, right, bottom, top] from cell centers.

    Parameters
    ----------
    x_coords : np.ndarray
        1D array of length N_cols with cell center X coordinates in meters.
    y_coords : np.ndarray
        1D array of length M_rows with cell center Y coordinates in meters.

    Returns
    -------
    list of float
        Four-element bounding extent [minx, maxx, miny, maxy] in meters for `plt.imshow`.
    """
    dx = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
    dy = abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0
    return [
        float(x_coords[0] - dx / 2),
        float(x_coords[-1] + dx / 2),
        float(y_coords[0] - dy / 2),
        float(y_coords[-1] + dy / 2),
    ]


def plot_unfiltered_product_variogram(
    distances: np.ndarray,
    semivars: np.ndarray,
    model_curve: dict[str, np.ndarray],
    a_range: float,
    sill: float,
    nugget: float,
    plots_dir: str | Path | None = None,
    prefix: str = "01_",
    stage_name: str = "stage1",
    interactive: bool = False,
    output_prefix: str = "",
) -> None:
    """
    Plot the experimental isotropic variogram and fitted theoretical model for the product field.

    Visualizes the empirical semivariance values across discrete lag distance bins
    alongside the fitted theoretical curve (e.g., spherical variogram model) and marks
    the calculated spatial correlation range.

    Parameters
    ----------
    distances : np.ndarray
        1D array of lag bin centers h in meters.
    semivars : np.ndarray
        1D array of empirical semivariance values gamma(h).
    model_curve : dict of str to np.ndarray
        Dictionary containing fine-resolution model evaluation vectors:
        - ``'h'``: Dense lag distance array in meters.
        - ``'gamma'``: Evaluated model semivariance array.
    a_range : float
        Fitted spatial correlation range parameter a in meters.
    sill : float
        Fitted structural sill variance parameter c.
    nugget : float
        Fitted nugget variance parameter c0.
    plots_dir : str, Path, or None, optional
        Target directory to save the output figure PNG. If None, figure is not saved.
    prefix : str, default="01_"
        File prefix for ordering diagnostic plots.
    stage_name : str, default="stage1"
        Stage name identifier included in the plot filename.
    interactive : bool, default=False
        If True and a GUI backend is active, displays the figure interactively.
    output_prefix : str, default=""
        Configured output prefix prepended to figure filename.

    Returns
    -------
    None
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure("Isotropic Variogram Analysis", figsize=(7, 5))
        plt.clf()
        plt.plot(distances, semivars, "ob", markersize=6, label="Experimental Variogram")
        plt.plot(model_curve["h"], model_curve["gamma"], "-r", linewidth=2, label="Fitted Model (Spherical)")
        plt.axvline(x=a_range, color="k", linestyle="--", linewidth=1.5, label=f"Calculated Correlation Range a = {a_range:.2f} m")
        plt.grid(True)
        plt.xlabel("Lag Distance h [m]")
        plt.ylabel("Semivariance \u03b3(h)")
        plt.title(f"Isotropic Variogram\nCalculated Range = {a_range:.2f} m, Sill = {sill:.4f}, Nugget = {nugget:.4f}")
        plt.legend()
        plt.tight_layout()

        pfx_str = f"{output_prefix}_" if output_prefix else ""
        if stage_name in ("stage2", "post_migration", "post") or str(prefix).startswith("03"):
            fig_name = f"04_{pfx_str}pass2_variogram.png"
        else:
            fig_name = f"01_{pfx_str}pass1_variogram.png"
        _save_figure(plt, plots_dir, fig_name)

        if interactive:
            _safe_interactive_pause(plt, 0.1)

        plt.close("Isotropic Variogram Analysis")
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_bss_kc_optimization_variograms(
    evaluated_variograms: list[dict[str, Any]],
    best_kc: float,
    all_kc_variances: list[tuple[float, float]],
    a_range: float | None = None,
    plots_dir: str | Path | None = None,
    prefix: str = "01_",
    stage_name: str = "stage1",
    interactive: bool = False,
    output_prefix: str = "",
) -> None:
    """
    Plot normalized product variograms and variance curve across candidate corner frequencies kc.

    Generates a 2-panel diagnostic figure:
    - Panel 1: Normalized semivariance curves gamma(h) / Mean^2 for all tested cutoff
      wavenumbers, highlighting the optimum minimum-variance kc.
    - Panel 2: Product field variance plotted as a function of corner frequency kc (rad/m),
      marking the global minimum identified by the spectral optimization.

    Parameters
    ----------
    evaluated_variograms : list of dict
        List of dictionaries for each tested corner frequency, each containing:
        - ``'kc'``: Corner wavenumber in rad/m.
        - ``'distance'``: Lag distances array in meters.
        - ``'gamma_norm'``: Normalized semivariance array.
    best_kc : float
        Optimal corner frequency wavenumber kc in rad/m minimizing product field variance.
    all_kc_variances : list of tuple of (float, float)
        List of (kc, variance) pairs evaluated during the frequency sweep.
    a_range : float, optional
        Active spatial correlation range parameter a in meters.
    plots_dir : str, Path, or None, optional
        Target directory to save the output figure PNG. If None, figure is not saved.
    prefix : str, default="01_"
        File prefix for ordering diagnostic plots.
    stage_name : str, default="stage1"
        Stage name identifier included in the plot filename.
    interactive : bool, default=False
        If True and a GUI backend is active, displays the figure interactively.
    output_prefix : str, default=""
        Configured output prefix prepended to figure filename.

    Returns
    -------
    None
    """
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        cmap = matplotlib.colormaps["viridis"].resampled(max(len(evaluated_variograms), 1))
        best_idx = 0
        min_diff = float("inf")
        for idx, item in enumerate(evaluated_variograms):
            diff = abs(float(item["kc"]) - float(best_kc))
            if diff < min_diff:
                min_diff = diff
                best_idx = idx

        for idx, item in enumerate(evaluated_variograms):
            kc_val = item["kc"]
            dist = item["distance"]
            gamma_norm = item["gamma_norm"]

            if idx == best_idx:
                ax1.plot(dist, gamma_norm, "-o", color="red", linewidth=2.5, markersize=6, label=f"Optimum $k_c$ = {kc_val:.4f}")
            else:
                ax1.plot(dist, gamma_norm, "--", color=cmap(idx), linewidth=1.0, alpha=0.7)

        if a_range is not None:
            ax1.axvline(x=a_range, color="k", linestyle="--", linewidth=1.5, label=f"Correlation Range a = {a_range:.2f} m")

        from .variogram import compute_cutoff_wavelength

        best_wl = compute_cutoff_wavelength(best_kc)
        wl_str = f" ($\\lambda_c$ = {best_wl:.2f} m)" if np.isfinite(best_wl) else ""
        ax1.set_xlabel("Lag Distance h [m]")
        ax1.set_ylabel("Normalized Semivariance \u03b3(h) / Mean\u00b2")
        ax1.set_title(
            f"Normalized Variograms across Corner Frequencies ($k_c$)\n"
            f"Optimum $k_c$ = {best_kc:.4f}{wl_str}"
        )
        ax1.grid(True)
        ax1.legend(loc="best", fontsize=8)

        if len(all_kc_variances) > 0:
            kcs, vars_vals = zip(*all_kc_variances)
            ax2.plot(kcs, vars_vals, "s-b", linewidth=2, markersize=6)
            ax2.axvline(x=best_kc, color="r", linestyle="--", label=f"Min Variance $k_c$ = {best_kc:.4f}")
            ax2.set_xlabel(r"Corner Frequency $k_c$ [rad/m]")
            ax2.set_ylabel("Product Field Variance")
            ax2.set_title(r"Mean BSS Variance in Correlation Range vs Corner Frequency $k_c$")
            ax2.grid(True)
            ax2.legend()

        plt.tight_layout()

        pfx_str = f"{output_prefix}_" if output_prefix else ""
        if stage_name in ("stage2", "post_migration", "post") or str(prefix).startswith("03"):
            fig_name = f"05_{pfx_str}pass2_kc_variogram_optimization.png"
        else:
            fig_name = f"02_{pfx_str}pass1_kc_variogram_optimization.png"
        _save_figure(plt, plots_dir, fig_name)

        if interactive:
            _safe_interactive_pause(plt, 0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def _compute_vector_scale_factor(
    dxi: np.ndarray,
    dyi: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    target_rel_len: float = 0.035,
) -> float:
    """
    Compute a clean, human-readable amplification scale factor for displacement quiver vectors.

    Ensures that in regional domains where displacements (10-50 m) are very small relative to the
    extent (e.g., 30 km), displacement vectors render as visible arrows rather than tiny dots.

    Parameters
    ----------
    dxi : np.ndarray
        Horizontal displacement along X [m].
    dyi : np.ndarray
        Horizontal displacement along Y [m].
    x_coords : np.ndarray
        Grid X coordinates [m].
    y_coords : np.ndarray
        Grid Y coordinates [m].
    target_rel_len : float, default 0.035
        Target visual arrow length as a fraction of the maximum domain dimension.

    Returns
    -------
    scale_factor : float
        Clean scalar multiplier (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, ...).
    """
    domain_span = max(
        float(np.nanmax(x_coords) - np.nanmin(x_coords)),
        float(np.nanmax(y_coords) - np.nanmin(y_coords)),
    )
    d_mag = np.hypot(dxi, dyi)
    finite_mag = d_mag[np.isfinite(d_mag) & (d_mag > 1e-6)]
    if len(finite_mag) == 0 or domain_span <= 0:
        return 1.0

    char_d = float(np.percentile(finite_mag, 90))
    if char_d <= 0:
        char_d = float(np.nanmean(finite_mag))
    if char_d <= 0:
        return 1.0

    target_len = target_rel_len * domain_span
    raw_factor = target_len / char_d
    if raw_factor <= 1.2:
        return 1.0

    exponent = 10.0 ** np.floor(np.log10(raw_factor))
    fraction = raw_factor / exponent
    if fraction < 1.5:
        factor = 1.0 * exponent
    elif fraction < 3.5:
        factor = 2.0 * exponent
    elif fraction < 7.5:
        factor = 5.0 * exponent
    else:
        factor = 10.0 * exponent
    return max(1.0, float(factor))


def plot_migration_displacement_vectors(
    travel_time_grid: np.ndarray,
    survey_points: np.ndarray,
    migrated_points: np.ndarray,
    dxi: np.ndarray,
    dyi: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    outline_mask: np.ndarray | None = None,
    plots_dir: str | Path | None = None,
    interactive: bool = False,
    output_prefix: str = "",
    horizontal_displacement_visualization: str = "arrows",
) -> None:
    """
    Plot 3D ray-tracing migration displacement vectors and migrated sounding depths.

    Generates a 2-panel glaciological diagnostic figure:
    - Upper Panel: Interpolated one_way_traveltime field (in nanoseconds) overlaid with
      horizontal displacement vectors (quiver arrows) illustrating the lateral shift
      from apex points to true basal reflection points driven by cross-profile slope.
    - Lower Panel: Spatial scatter plot comparing pre-migrated surface sounding locations
      (black dots) with migrated bed reflection points colored by true bedrock depth (in meters).

    Parameters
    ----------
    travel_time_grid : np.ndarray
        2D array of interpolated one_way_traveltimes in nanoseconds (ns) of shape (M_rows, N_cols).
    survey_points : np.ndarray
        Pre-migration survey point coordinates of shape (N, >=2) in projected meters.
    migrated_points : np.ndarray
        Migrated point array of shape (N, 4) containing [x_mig, y_mig, z_surf, depth_mig].
    dxi : np.ndarray
        Horizontal displacement components along the easting/X axis in meters.
    dyi : np.ndarray
        Horizontal displacement components along the northing/Y axis in meters.
    x_coords : np.ndarray
        1D array of cell center X coordinates in meters.
    y_coords : np.ndarray
        1D array of cell center Y coordinates in meters.
    outline_mask : np.ndarray, optional
        2D boolean active domain mask of shape (M_rows, N_cols).
    plots_dir : str, Path, or None, optional
        Target directory to save the output figure PNG. If None, figure is not saved.
    interactive : bool, default=False
        If True and a GUI backend is active, displays the figure interactively.

    Returns
    -------
    None
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        plot_extent = _get_cell_edge_extent(x_coords, y_coords)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

        im1 = ax1.imshow(travel_time_grid, extent=plot_extent, origin="lower", cmap="viridis")
        min_tt = np.nanmin(travel_time_grid)
        max_tt = np.nanmax(travel_time_grid)
        if np.isfinite(min_tt) and np.isfinite(max_tt) and max_tt > min_tt:
            cnt1 = ax1.contour(
                travel_time_grid,
                levels=8,
                extent=plot_extent,
                origin="lower",
                colors="white",
                linewidths=0.5,
                alpha=0.7,
            )
            ax1.clabel(cnt1, inline=True, fmt="%d ns", fontsize=6.0, inline_spacing=3)

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

        if horizontal_displacement_visualization == "grad_traveltime_grid":
            ax1.set_title("Calculated Traveltime Field", fontsize=9.5)
        else:
            # Downsample arrows for clean plot and dynamically amplify for visibility
            step = max(len(survey_points) // 80, 1)
            scale_factor = _compute_vector_scale_factor(dxi[::step], dyi[::step], x_coords, y_coords)
            ax1.quiver(
                survey_points[::step, 0],
                survey_points[::step, 1],
                dxi[::step] * scale_factor,
                dyi[::step] * scale_factor,
                color="white",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.003,
                zorder=4,
            )
            vec_label = f" (displacement vectors \u00d7{int(scale_factor) if scale_factor == int(scale_factor) else scale_factor:g})" if scale_factor > 1.0 else ""
            ax1.set_title(f"Calculated Traveltime Field and 3D Migration Horizontal Displacement Vectors{vec_label}", fontsize=9.5)
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")

        # Lower Subplot: Migrated points colored by depth
        x_mig = migrated_points[:, 0]
        y_mig = migrated_points[:, 1]
        d_mig = migrated_points[:, 3]

        n_pts = len(survey_points)
        s_mig = float(np.clip(180.0 / np.sqrt(n_pts), 1.5, 14.0))
        s_pre = float(np.clip(100.0 / np.sqrt(n_pts), 0.8, 8.0))

        if horizontal_displacement_visualization != "grad_traveltime_grid":
            ax2.scatter(
                survey_points[:, 0],
                survey_points[:, 1],
                c="black",
                s=s_pre,
                alpha=0.7,
                label="Pre-migrated Points",
                zorder=2,
            )
        sc = ax2.scatter(
            x_mig,
            y_mig,
            c=d_mig,
            s=s_mig,
            alpha=0.85,
            cmap="jet",
            label="Migrated Points",
            zorder=3,
        )

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

        ax2.set_title("Migrated Survey Point Depths [m]", fontsize=9.5)
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")
        if horizontal_displacement_visualization != "grad_traveltime_grid":
            ax2.legend(loc="upper right", fontsize=8)

        plt.tight_layout()

        pfx_str = f"{output_prefix}_" if output_prefix else ""
        _save_figure(plt, plots_dir, f"03_{pfx_str}3d_migration.png")

        if interactive:
            _safe_interactive_pause(plt, 0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_migration_horizontal_displacements(
    travel_time_grid: np.ndarray,
    survey_points: np.ndarray,
    migrated_points: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    dx: float,
    dy: float,
    outline_mask: np.ndarray | None = None,
    plots_dir: str | Path | None = None,
    interactive: bool = False,
    output_prefix: str = "",
) -> None:
    """
    Plot 3D ray migration horizontal displacements over the traveltime gradient magnitude field.

    Generates a standalone diagnostic figure (Figure 03 companion) displaying the magnitude of the
    spatial traveltime gradient |∇T| = sqrt((dT/dx)^2 + (dT/dy)^2) overlaid with the original
    survey points (grey) and the migrated reflection points (black).

    Parameters
    ----------
    travel_time_grid : np.ndarray
        2D array of interpolated one_way_traveltimes of shape (M_rows, N_cols).
    survey_points : np.ndarray
        Pre-migration survey point coordinates of shape (N, >=2) in projected meters.
    migrated_points : np.ndarray
        Migrated point array of shape (N, >=2) containing [x_mig, y_mig, ...].
    x_coords : np.ndarray
        1D array of cell center X coordinates in meters.
    y_coords : np.ndarray
        1D array of cell center Y coordinates in meters.
    dx : float
        Grid resolution along the X dimension in meters.
    dy : float
        Grid resolution along the Y dimension in meters.
    outline_mask : np.ndarray, optional
        2D boolean active domain mask of shape (M_rows, N_cols).
    plots_dir : str, Path, or None, optional
        Target directory to save the output figure PNG. If None, figure is not saved.
    interactive : bool, default=False
        If True and a GUI backend is active, displays the figure interactively.
    output_prefix : str, default=""
        Configured output prefix prepended to figure filename.

    Returns
    -------
    None
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.colors import LinearSegmentedColormap

        slope_y, slope_x = np.gradient(travel_time_grid, dy, dx)
        grad_mag = np.hypot(slope_x, slope_y)
        if outline_mask is not None:
            grad_mag = grad_mag.copy()
            grad_mag[~outline_mask] = np.nan

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_extent = _get_cell_edge_extent(x_coords, y_coords)

        cmap_gradient = LinearSegmentedColormap.from_list(
            "white_ylorrd", ["#ffffff", "#ffeda0", "#feb24c", "#f03b20", "#800026"]
        ).copy()
        if hasattr(cmap_gradient, "with_extremes"):
            cmap_gradient = cmap_gradient.with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
        else:
            cmap_gradient.set_bad(color="white", alpha=0.0)

        im = ax.imshow(
            grad_mag,
            extent=plot_extent,
            origin="lower",
            cmap=cmap_gradient,
            vmin=0.0,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, label="Traveltime Gradient Magnitude |\u2207T|")

        if outline_mask is not None:
            ax.contour(
                outline_mask,
                levels=[0.5],
                extent=plot_extent,
                origin="lower",
                colors="black",
                linewidths=0.8,
                linestyles="solid",
            )

        n_pts = len(survey_points)
        s_mig = float(np.clip(16.0 / np.sqrt(n_pts), 0.12, 1.8))
        s_orig = float(np.clip(22.0 / np.sqrt(n_pts), 0.18, 2.2))
        ax.scatter(
            survey_points[:, 0],
            survey_points[:, 1],
            c="#94a3b8",
            s=s_orig,
            alpha=0.7,
            linewidths=0,
            edgecolors="none",
            label="Original Points",
            zorder=3,
        )
        ax.scatter(
            migrated_points[:, 0],
            migrated_points[:, 1],
            c="black",
            s=s_mig,
            alpha=0.9,
            linewidths=0,
            edgecolors="none",
            label="Migrated Points",
            zorder=4,
        )

        ax.set_aspect("equal")
        ax.set_title("3D Ray Migration Horizontal Displacements", fontsize=10)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        markerscale = float(max(1.0, 2.5 / np.sqrt(s_mig)))
        ax.legend(loc="upper right", fontsize=8, markerscale=markerscale)

        plt.tight_layout()

        pfx_str = f"{output_prefix}_" if output_prefix else ""
        _save_figure(plt, plots_dir, f"03_{pfx_str}3d_migration_hor_displacements.png")

        if interactive:
            _safe_interactive_pause(plt, 0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


UNCERTAINTY_PLACEHOLDER_TEXT = "Uncertainty not available\n(see log)"


def _uncertainty_unavailable(grid: np.ndarray | None) -> bool:
    """
    Check if an uncertainty grid is missing or lacks finite numerical values.

    Parameters
    ----------
    grid : np.ndarray or None
        2D uncertainty array to inspect.

    Returns
    -------
    bool
        True if grid is None or contains no finite values; False otherwise.
    """
    return grid is None or not np.any(np.isfinite(grid))


def _draw_uncertainty_placeholder(ax: Any, plot_extent: list[float] | None, title: str) -> None:
    """
    Fill an uncertainty panel with a clean diagnostic placeholder when variance was skipped.

    Prevents rendering confusing all-zero uncertainty maps when variance computation
    was disabled for memory savings or speed.

    Parameters
    ----------
    ax : Any
        Matplotlib axis object.
    plot_extent : list of float or None
        Spatial extent [minx, maxx, miny, maxy] in meters.
    title : str
        Panel title string.

    Returns
    -------
    None
    """
    if plot_extent is not None and len(plot_extent) == 4:
        ax.set_xlim(plot_extent[0], plot_extent[1])
        ax.set_ylim(plot_extent[2], plot_extent[3])
    ax.set_facecolor("#f1f5f9")
    ax.text(
        0.5, 0.5, UNCERTAINTY_PLACEHOLDER_TEXT,
        transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#475569",
    )
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")


def plot_kriging_bedrock_and_uncertainty(
    kriged_bedrock: np.ndarray,
    kriged_std: np.ndarray | None = None,
    pts: np.ndarray | None = None,
    plot_extent: list[float] | None = None,
    kriged_variance: np.ndarray | None = None,
    plots_dir: str | Path | None = None,
    interactive: bool = False,
    output_prefix: str = "",
    model_suffix: str = "",
) -> None:
    """
    Plot 2-panel figure showing Kriging-interpolated bedrock elevation and standard uncertainty.

    Visualizes:
    - Left Panel: Predicted bedrock topography in meters with scattered survey point overlays.
    - Right Panel: Kriging standard error field (+- meters), representing spatial interpolation
      uncertainty conditioned on data spacing and variogram structure.

    Parameters
    ----------
    kriged_bedrock : np.ndarray
        2D array of interpolated bedrock elevations in meters of shape (M_rows, N_cols).
    kriged_std : np.ndarray, optional
        2D array of Kriging standard deviation (+- meters). If None, calculated from `kriged_variance`.
    pts : np.ndarray, optional
        Scattered survey points coordinates of shape (N, >=2) in meters.
    plot_extent : list of float, optional
        Four-element bounding box [minx, maxx, miny, maxy] in meters.
    kriged_variance : np.ndarray, optional
        2D array of Kriging estimation variance in meters^2.
    plots_dir : str, Path, or None, optional
        Target directory to save the output figure PNG. If None, figure is not saved.
    interactive : bool, default=False
        If True and a GUI backend is active, displays the figure interactively.
    output_prefix : str, default=""
        Configured output prefix prepended to figure filename.
    model_suffix : str, default=""
        Optional drift model identifier slug appended to the figure filename.

    Returns
    -------
    None
    """
    try:
        import matplotlib.pyplot as plt

        if kriged_std is None and kriged_variance is not None:
            unc_grid = np.sqrt(np.maximum(np.nan_to_num(kriged_variance, nan=0.0), 0.0))
            if np.isnan(kriged_variance).any():
                unc_grid[np.isnan(kriged_variance)] = np.nan
        elif kriged_std is not None:
            unc_grid = kriged_std
        else:
            unc_grid = np.full_like(kriged_bedrock, np.nan, dtype=float)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        im1 = ax1.imshow(kriged_bedrock, extent=plot_extent, origin="lower", cmap="terrain")
        if pts is not None and len(pts) > 0:
            ax1.scatter(pts[:, 0], pts[:, 1], c="black", s=0.5, alpha=0.7, label="Survey Points")
            ax1.legend()
        ax1.set_title("Interpolated Bedrock Elevation")
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        fig.colorbar(im1, ax=ax1, label="Elevation [m]")

        cmap2 = _get_transparent_cmap("plasma")
        if _uncertainty_unavailable(unc_grid):
            _draw_uncertainty_placeholder(ax2, plot_extent, "Kriging Standard Uncertainty")
        else:
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

        pfx_str = f"{output_prefix}_" if output_prefix else ""
        sfx = f"_{model_suffix.lstrip('_')}" if model_suffix else ""
        _save_figure(plt, plots_dir, f"06_{pfx_str}bedrock_elevation_and_uncertainty{sfx}.png")

        if interactive:
            _safe_interactive_pause(plt, 0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_final_blended_bedrock_map(
    blended_bedrock: np.ndarray,
    plot_extent: list[float],
    plots_dir: str | Path | None = None,
    interactive: bool = False,
    output_prefix: str = "",
    model_suffix: str = "",
) -> None:
    """
    Plot final geomorphologically blended bedrock elevation raster.

    Displays the seamlessly blended continuous bedrock topography where internal
    interpolated bed elevations taper smoothly into the ice-free proglacial DEM
    along the perimeter margins.

    Parameters
    ----------
    blended_bedrock : np.ndarray
        2D array of blended bedrock elevations in meters of shape (M_rows, N_cols).
    plot_extent : list of float
        Four-element bounding box [minx, maxx, miny, maxy] in meters.
    plots_dir : str, Path, or None, optional
        Target directory to save the output figure PNG. If None, figure is not saved.
    interactive : bool, default=False
        If True and a GUI backend is active, displays the figure interactively.
    output_prefix : str, default=""
        Configured output prefix prepended to figure filename.
    model_suffix : str, default=""
        Optional drift model identifier slug appended to the figure filename.

    Returns
    -------
    None
    """
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure("Final Blended Bedrock Map", figsize=(8, 6))
        plt.clf()
        plt.imshow(blended_bedrock, extent=plot_extent, origin="lower", cmap="terrain")
        plt.colorbar(label="Elevation [m]")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.tight_layout()

        pfx_str = f"{output_prefix}_" if output_prefix else ""
        sfx = f"_{model_suffix.lstrip('_')}" if model_suffix else ""
        _save_figure(plt, plots_dir, f"07_{pfx_str}final_embedded_bedrock{sfx}.png")

        if interactive:
            _safe_interactive_pause(plt, 0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_final_ice_thickness_and_uncertainty(
    final_thickness: np.ndarray,
    kriged_std: np.ndarray,
    outline_mask: np.ndarray | None,
    plot_extent: list[float],
    mean_thick: float,
    mean_unc: float,
    plots_dir: str | Path | None = None,
    interactive: bool = False,
    output_prefix: str = "",
    model_suffix: str = "",
) -> None:
    """
    Plot 2-panel figure showing final ice thickness map and Kriging standard uncertainty.

    Generates:
    - Left Panel: Continuous ice thickness distribution H(x, y) = S(x, y) - B(x, y) in meters,
      masked to the active glacier perimeter with reported domain-average mean thickness.
    - Right Panel: Kriging standard uncertainty field (+- meters), highlighting regions of
      dense sounding coverage (low uncertainty) versus unprobed tributary basins (high uncertainty).

    Parameters
    ----------
    final_thickness : np.ndarray
        2D array of ice thickness in meters of shape (M_rows, N_cols).
    kriged_std : np.ndarray
        2D array of Kriging standard uncertainty (+- meters) of shape (M_rows, N_cols).
    outline_mask : np.ndarray or None
        2D boolean active domain mask of shape (M_rows, N_cols).
    plot_extent : list of float
        Four-element bounding box [minx, maxx, miny, maxy] in meters.
    mean_thick : float
        Mean ice thickness across active glacier pixels in meters.
    mean_unc : float
        Mean Kriging uncertainty across active glacier pixels in meters.
    plots_dir : str, Path, or None, optional
        Target directory to save the output figure PNG. If None, figure is not saved.
    interactive : bool, default=False
        If True and a GUI backend is active, displays the figure interactively.
    output_prefix : str, default=""
        Configured output prefix prepended to figure filename.
    model_suffix : str, default=""
        Optional drift model identifier slug appended to the figure filename.

    Returns
    -------
    None
    """
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
                linewidths=0.8,
                linestyles="solid",
            )
        ax1.set_title(f"Mean Thickness: {int(round(mean_thick))} m")
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")

        cmap2 = _get_transparent_cmap("magma")
        if _uncertainty_unavailable(kriged_std):
            _draw_uncertainty_placeholder(ax2, plot_extent, "Mean Kriging Uncertainty: n/a")
        else:
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
                    linewidths=0.8,
                    linestyles="solid",
                )
            ax2.set_title(f"Mean Kriging Uncertainty: \u00b1{int(round(mean_unc))} m")
            ax2.set_xlabel("X [m]")
            ax2.set_ylabel("Y [m]")

        plt.tight_layout()

        pfx_str = f"{output_prefix}_" if output_prefix else ""
        sfx = f"_{model_suffix.lstrip('_')}" if model_suffix else ""
        _save_figure(plt, plots_dir, f"08_{pfx_str}final_thickness_and_uncertainty{sfx}.png")

        if interactive:
            _safe_interactive_pause(plt, 0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_final_ice_thickness_histogram(
    final_thickness: np.ndarray,
    outline_mask: np.ndarray | None,
    mean_thick: float,
    plots_dir: str | Path | None = None,
    interactive: bool = False,
    output_prefix: str = "",
    model_suffix: str = "",
    total_volume_km3: float | None = None,
    total_volume_unc_km3: float | None = None,
    dx: float | None = None,
    dy: float | None = None,
    kriged_std: np.ndarray | None = None,
) -> None:
    """
    Plot statistical frequency histogram of the final ice thickness distribution.

    Renders a binned frequency distribution of ice depth across active glacier pixels,
    marking the domain mean with a dashed line and providing summary statistics
    (mean, median, standard deviation, and maximum ice thickness in meters), as well as
    displaying the total calculated ice volume and uncertainty in the figure title.

    Parameters
    ----------
    final_thickness : np.ndarray
        2D array of ice thickness in meters of shape (M_rows, N_cols).
    outline_mask : np.ndarray or None
        2D boolean active domain mask of shape (M_rows, N_cols).
    mean_thick : float
        Mean ice thickness across active glacier pixels in meters.
    plots_dir : str, Path, or None, optional
        Target directory to save the output figure PNG. If None, figure is not saved.
    interactive : bool, default=False
        If True and a GUI backend is active, displays the figure interactively.
    output_prefix : str, default=""
        Configured output prefix prepended to figure filename.
    model_suffix : str, default=""
        Optional drift model identifier slug appended to the figure filename.
    total_volume_km3 : float or None, optional
        Total ice volume in km³. If None but dx and dy are provided, computed automatically.
    total_volume_unc_km3 : float or None, optional
        Total ice volume uncertainty in km³.
    dx : float or None, optional
        Grid resolution along X in meters.
    dy : float or None, optional
        Grid resolution along Y in meters.
    kriged_std : np.ndarray or None, optional
        2D array of Kriging thickness uncertainty in meters.

    Returns
    -------
    None
    """
    try:
        import matplotlib.pyplot as plt

        if total_volume_km3 is None and dx is not None and dy is not None:
            from .raster import calculate_ice_volume
            vol_res = calculate_ice_volume(final_thickness, dx, dy, outline_mask, kriged_std)
            total_volume_km3 = vol_res.volume_km3
            total_volume_unc_km3 = vol_res.uncertainty_km3

        fig = plt.figure("Final Ice Thickness Histogram", figsize=(7, 5))
        plt.clf()

        thick_valid = final_thickness[np.isfinite(final_thickness) & (final_thickness > 0)]
        if outline_mask is not None and np.any(outline_mask):
            thick_valid = final_thickness[outline_mask & np.isfinite(final_thickness) & (final_thickness > 0)]

        if len(thick_valid) > 0:
            n_bins = max(15, min(50, len(thick_valid) // 100))
            plt.hist(thick_valid, bins=n_bins, color="#38bdf8", edgecolor="#1e293b", alpha=0.85)
            mean_thick = float(np.mean(thick_valid))
            median_thick = float(np.median(thick_valid))
            max_thick = float(np.max(thick_valid))
            std_thick = float(np.std(thick_valid))
        else:
            mean_thick = 0.0 if mean_thick is None else mean_thick
            median_thick = mean_thick
            max_thick = 0.0
            std_thick = 0.0

        plt.axvline(x=mean_thick, color="#ef4444", linestyle="--", linewidth=2.0)

        stats_text = (
            f"Mean (red dashed): {mean_thick:.1f} m\n"
            f"Median: {median_thick:.1f} m\n"
            f"Std Dev: \u00b1{std_thick:.1f} m\n"
            f"Max: {max_thick:.1f} m"
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
        plt.xlim(left=0)
        plt.xlabel("Thickness [m]")
        plt.ylabel("Pixel Frequency Count")

        if total_volume_km3 is not None:
            if total_volume_unc_km3 is not None:
                vol_title_str = f"\nTotal Volume: {total_volume_km3:.4f} \u00b1 {total_volume_unc_km3:.4f} km\u00b3"
            else:
                vol_title_str = f"\nTotal Volume: {total_volume_km3:.4f} km\u00b3"
        else:
            vol_title_str = ""

        plt.title(f"Final Thickness Distribution{vol_title_str}")
        plt.tight_layout()

        pfx_str = f"{output_prefix}_" if output_prefix else ""
        sfx = f"_{model_suffix.lstrip('_')}" if model_suffix else ""
        _save_figure(plt, plots_dir, f"09_{pfx_str}final_thickness_histogram{sfx}.png")

        if interactive:
            _safe_interactive_pause(plt, 0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_final_basal_shear_stress_and_uncertainty(
    final_bss: np.ndarray,
    bss_std: np.ndarray,
    outline_mask: np.ndarray | None,
    plot_extent: list[float],
    mean_bss: float,
    mean_bss_unc: float,
    plots_dir: str | Path | None = None,
    interactive: bool = False,
    output_prefix: str = "",
    model_suffix: str = "",
) -> None:
    """
    Plot 2-panel figure showing Basal Shear Stress distribution and propagated Kriging uncertainty.

    Generates:
    - Left Panel: Basal shear stress field tau_b(x, y) = rho_ice * g * H * sin(alpha) in kPa
      across the glaciated body, with domain-average mean stress reported in the header.
    - Right Panel: Propagated standard uncertainty field (+- kPa) derived from Kriging bed
      elevation variance and local surface slope.

    Parameters
    ----------
    final_bss : np.ndarray
        2D array of basal shear stress in kPa of shape (M_rows, N_cols).
    bss_std : np.ndarray
        2D array of propagated BSS standard uncertainty (+- kPa) of shape (M_rows, N_cols).
    outline_mask : np.ndarray or None
        2D boolean active domain mask of shape (M_rows, N_cols).
    plot_extent : list of float
        Four-element bounding box [minx, maxx, miny, maxy] in meters.
    mean_bss : float
        Domain-average mean basal shear stress in kPa.
    mean_bss_unc : float
        Domain-average mean BSS standard uncertainty in kPa.
    plots_dir : str, Path, or None, optional
        Target directory to save the output figure PNG. If None, figure is not saved.
    interactive : bool, default=False
        If True and a GUI backend is active, displays the figure interactively.
    output_prefix : str, default=""
        Configured output prefix prepended to figure filename.
    model_suffix : str, default=""
        Optional drift model identifier slug appended to the figure filename.

    Returns
    -------
    None
    """
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
        fig.colorbar(im1, ax=ax1, label=r"Basal Shear Stress $\tau_b$ [kPa]", extend="max" if vmax1 is not None else "neither")
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
        ax1.set_title(f"Mean Basal Shear Stress: {mean_bss:.1f} kPa")
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")

        cmap2 = _get_transparent_cmap("plasma")
        if _uncertainty_unavailable(bss_std):
            _draw_uncertainty_placeholder(ax2, plot_extent, "Mean Basal Shear Stress Uncertainty: n/a")
        else:
            valid_unc = bss_std[np.isfinite(bss_std) & (bss_std > 0)]
            if len(valid_unc) > 0:
                vmax2 = float(np.percentile(valid_unc, 98.5))
                vmax2 = max(vmax2, 0.1)
            else:
                vmax2 = None

            im2 = ax2.imshow(bss_std, extent=plot_extent, origin="lower", cmap=cmap2, vmin=0, vmax=vmax2)
            fig.colorbar(im2, ax=ax2, label="Basal Shear Stress Uncertainty [\u00b1 kPa]", extend="max" if vmax2 is not None else "neither")
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
            ax2.set_title(f"Mean Basal Shear Stress Uncertainty: \u00b1{mean_bss_unc:.1f} kPa")
            ax2.set_xlabel("X [m]")
            ax2.set_ylabel("Y [m]")

        plt.tight_layout()

        pfx_str = f"{output_prefix}_" if output_prefix else ""
        sfx = f"_{model_suffix.lstrip('_')}" if model_suffix else ""
        _save_figure(plt, plots_dir, f"10_{pfx_str}final_basal_shear_stress_and_uncertainty{sfx}.png")

        if interactive:
            _safe_interactive_pause(plt, 0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)


def plot_final_basal_shear_stress_histogram(
    final_bss: np.ndarray,
    outline_mask: np.ndarray | None,
    mean_bss: float,
    plots_dir: str | Path | None = None,
    interactive: bool = False,
    output_prefix: str = "",
    model_suffix: str = "",
) -> None:
    """
    Plot statistical frequency histogram of the final Basal Shear Stress distribution.

    Renders a binned frequency distribution of basal shear stress values across active
    glacier pixels in kPa, marking the domain mean and providing key summary statistics
    (mean, median, standard deviation, and peak driving stress).

    Parameters
    ----------
    final_bss : np.ndarray
        2D array of basal shear stress in kPa of shape (M_rows, N_cols).
    outline_mask : np.ndarray or None
        2D boolean active domain mask of shape (M_rows, N_cols).
    mean_bss : float
        Mean basal shear stress across active glacier pixels in kPa.
    plots_dir : str, Path, or None, optional
        Target directory to save the output figure PNG. If None, figure is not saved.
    interactive : bool, default=False
        If True and a GUI backend is active, displays the figure interactively.
    output_prefix : str, default=""
        Configured output prefix prepended to figure filename.
    model_suffix : str, default=""
        Optional drift model identifier slug appended to the figure filename.

    Returns
    -------
    None
    """
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure("Final Basal Shear Stress Histogram", figsize=(7, 5))
        plt.clf()

        bss_valid = final_bss[np.isfinite(final_bss) & (final_bss > 0)]
        if outline_mask is not None and np.any(outline_mask):
            bss_valid = final_bss[outline_mask & np.isfinite(final_bss) & (final_bss > 0)]

        if len(bss_valid) > 0:
            n_bins = max(15, min(50, len(bss_valid) // 100))
            plt.hist(bss_valid, bins=n_bins, color="#10b981", edgecolor="#1e293b", alpha=0.85)
            mean_bss = float(np.mean(bss_valid))
            median_bss = float(np.median(bss_valid))
            max_bss = float(np.max(bss_valid))
            std_bss = float(np.std(bss_valid))
        else:
            mean_bss = 0.0 if mean_bss is None else mean_bss
            median_bss = mean_bss
            max_bss = 0.0
            std_bss = 0.0

        plt.axvline(x=mean_bss, color="#ef4444", linestyle="--", linewidth=2.0)

        stats_text = (
            f"Mean $\\tau_b$ (red dashed): {mean_bss:.1f} kPa\n"
            f"Median: {median_bss:.1f} kPa\n"
            f"Std Dev: \u00b1{std_bss:.1f} kPa\n"
            f"Max: {max_bss:.1f} kPa"
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
        plt.xlim(left=0)
        plt.xlabel(r"Basal Shear Stress $\tau_b$ [kPa]")
        plt.ylabel("Pixel Frequency Count")
        plt.title(r"Final Basal Shear Stress $\tau_b$ Distribution")
        plt.tight_layout()

        pfx_str = f"{output_prefix}_" if output_prefix else ""
        sfx = f"_{model_suffix.lstrip('_')}" if model_suffix else ""
        _save_figure(plt, plots_dir, f"11_{pfx_str}final_basal_shear_stress_histogram{sfx}.png")

        if interactive:
            _safe_interactive_pause(plt, 0.5)

        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Plotting skipped: {e}", UserWarning)
