"""
3D Eikonal Ray Migration Module for PySole.
Ported from MATLAB script MIG.m by Daniel Binder (2011).
Performs 3D Eikonal ray migration on zero-offset seismic / GPR travel times over complex surface topography,
computing non-orthogonal slowness vector components (sx, sy, sz) and horizontal/vertical ray displacements.
"""

from typing import Tuple, Optional, Union
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from .smoothing import compute_gradients


from .raster import ensure_spatial_coords


def migrate_eikonal_points(
    dem: np.ndarray,
    travel_time_grid: np.ndarray,
    survey_points: np.ndarray,
    velocity: float = 0.16,
    dx: float = 1.0,
    dy: float = 1.0,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
    outline_mask: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    plots_dir: Optional[str] = None,
    interactive: bool = False,
    plotit: bool = False,
) -> np.ndarray:
    """
    Migrates zero-offset GPR/seismic survey points into 3D space using the 3D Eikonal Ray Migration algorithm.
    Ported directly from MATLAB MIG.m (Binder, 2009, 2011).

    Parameters
    ----------
    dem : 2D np.ndarray
        Surface DEM grid Z(x, y) [m].
    travel_time_grid : 2D np.ndarray
        Continuous smoothed travel time grid T(x, y) [ns] or depth-equivalent product field.
    survey_points : np.ndarray
        Scattered input points [X, Y, Z_surface, depth_or_travel_time].
    velocity : float
        Signal propagation velocity [m/ns] (default = 0.16 m/ns for GPR in ice).
    dx : float
        Grid spacing along X.
    dy : float
        Grid spacing along Y.
    x_coords : 1D np.ndarray, optional
        Grid X coordinate vector.
    y_coords : 1D np.ndarray, optional
        Grid Y coordinate vector.
    outline_mask : 2D np.ndarray, optional
        Boolean creeping body boundary mask.
    bounds : tuple of float, optional
        (minx, miny, maxx, maxy) bounding box.
    plots_dir : str, optional
        Directory where generated displacement vector plots are saved.

    Returns
    -------
    migrated_points : np.ndarray
        Nx4 array of [x_mig, y_mig, z_surf_mig, depth_mig].
    """
    x_coords, y_coords, bounds = ensure_spatial_coords(
        dem.shape, dx=dx, dy=dy, bounds=bounds, x_coords=x_coords, y_coords=y_coords
    )

    # 1. Calculate horizontal slownesses s1 = dT/dx and s2 = dT/dy from continuous travel time field (gradient(INT))
    tt_grads = compute_gradients(travel_time_grid, dx=dx, dy=dy)
    s1_grid = tt_grads["slope_x"]  # \partial T / \partial x
    s2_grid = tt_grads["slope_y"]  # \partial T / \partial y

    # 2. Calculate surface DEM directional slope components (\partial Z / \partial x, \partial Z / \partial y)
    dem_grads = compute_gradients(dem, dx=dx, dy=dy)
    dz_dx = dem_grads["slope_x"]
    dz_dy = dem_grads["slope_y"]

    alpha_x = np.arctan(dz_dx)
    alpha_y = np.arctan(dz_dy)

    sin_alpha_x = np.sin(alpha_x)
    cos_alpha_x = np.cos(alpha_x)
    sin_alpha_y = np.sin(alpha_y)
    cos_alpha_y = np.cos(alpha_y)

    # 3. Non-orthogonal slowness coordinate transformation (matching MATLAB MIG.m)
    # s12_quadr = s1^2 + s2^2 + 2*s1*s2*sin(alpha_x)*sin(alpha_y)
    s12_quadr_grid = s1_grid**2 + s2_grid**2 + 2.0 * s1_grid * s2_grid * sin_alpha_x * sin_alpha_y

    # s3_grid over Eikonal equation: s3 = sqrt( (1/v)^2 - s12_quadr )
    inv_v_sq = (1.0 / max(velocity, 1e-4))**2
    s3_grid = np.sqrt(np.maximum(inv_v_sq - s12_quadr_grid, 0.0))

    # A_grid constant for each grid point
    A_grid = (cos_alpha_y**2) * (cos_alpha_x**2) + (sin_alpha_y**2) * (cos_alpha_x**2) + (sin_alpha_x**2) * (cos_alpha_y**2)
    A_grid = np.maximum(A_grid, 1e-6)

    # Calculate 3D surface-normal slowness components (sx_grid, sy_grid, sz_grid)
    sx1_grid = cos_alpha_x * ((cos_alpha_y**2 + sin_alpha_y**2) / A_grid) * s1_grid
    sx2_grid = -sin_alpha_x * cos_alpha_x * (sin_alpha_y / A_grid) * s2_grid
    sx3_grid = -sin_alpha_x * (cos_alpha_y / np.sqrt(A_grid)) * s3_grid
    sx_grid = sx1_grid + sx2_grid + sx3_grid

    sy1_grid = -sin_alpha_y * sin_alpha_x * (cos_alpha_y / A_grid) * s1_grid
    sy2_grid = cos_alpha_y * ((cos_alpha_x**2 + sin_alpha_x**2) / A_grid) * s2_grid
    sy3_grid = -cos_alpha_x * (sin_alpha_y / np.sqrt(A_grid)) * s3_grid
    sy_grid = sy1_grid + sy2_grid + sy3_grid

    sz1_grid = (cos_alpha_y**2) * (sin_alpha_x / A_grid) * s1_grid
    sz2_grid = (cos_alpha_x**2) * (sin_alpha_y / A_grid) * s2_grid
    sz3_grid = cos_alpha_x * (cos_alpha_y / np.sqrt(A_grid)) * s3_grid
    sz_grid = sz1_grid + sz2_grid + sz3_grid

    # 4. Ray displacement vector grids: dx_grid, dy_grid, dz_grid (MIG.m)
    # dx_grid = -T * v^2 * sx
    # dy_grid = -T * v^2 * sy
    # dz_grid = -T * v^2 * sz
    v_sq = velocity**2
    dx_grid = -travel_time_grid * v_sq * sx_grid
    dy_grid = -travel_time_grid * v_sq * sy_grid
    dz_grid = -travel_time_grid * v_sq * sz_grid

    # 5. Interpolate 3D ray displacement vectors at scattered survey locations (interp2)
    interp_dx = RegularGridInterpolator((y_coords, x_coords), dx_grid, bounds_error=False, fill_value=0.0)
    interp_dy = RegularGridInterpolator((y_coords, x_coords), dy_grid, bounds_error=False, fill_value=0.0)
    interp_dz = RegularGridInterpolator((y_coords, x_coords), dz_grid, bounds_error=False, fill_value=0.0)
    interp_dem = RegularGridInterpolator((y_coords, x_coords), dem, bounds_error=False, fill_value=np.nan)

    pts_xy = np.column_stack((survey_points[:, 1], survey_points[:, 0]))  # (Y, X)

    dxi = interp_dx(pts_xy)
    dyi = interp_dy(pts_xy)
    dzi = interp_dz(pts_xy)

    dxi = np.nan_to_num(dxi, nan=0.0)
    dyi = np.nan_to_num(dyi, nan=0.0)

    # Migrated coordinates & depth d_mig = -dz
    x_mig = survey_points[:, 0] + dxi
    y_mig = survey_points[:, 1] + dyi
    d_mig = np.maximum(-dzi, 0.0)

    # Fallback to unmigrated depth if dzi interpolation is zero
    unmig_d = survey_points[:, 3] * velocity
    valid_d = d_mig > 0
    d_mig[~valid_d] = unmig_d[~valid_d]

    # Sample surface DEM elevation at migrated (x_mig, y_mig)
    mig_pts_xy = np.column_stack((y_mig, x_mig))
    z_surf_mig = interp_dem(mig_pts_xy)

    # Clean NaNs in z_surf_mig using original surface z
    nan_z = np.isnan(z_surf_mig)
    z_surf_mig[nan_z] = survey_points[nan_z, 2]

    migrated_points = np.column_stack((x_mig, y_mig, z_surf_mig, d_mig))

    # 6. Plot 3D Migration Displacement Vectors matching MIG.m subplot(2,1,1) & subplot(2,1,2)
    should_plot = interactive or (plots_dir is not None)
    if should_plot:
        try:
            import matplotlib.pyplot as plt

            plot_extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

            im1 = ax1.imshow(travel_time_grid, extent=plot_extent, origin="lower", cmap="viridis")
            cnt1 = ax1.contour(travel_time_grid, extent=plot_extent, origin="lower", colors="white", linewidths=0.5, alpha=0.7)
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
            ax1.set_title("Calculated Traveltime Field and 3D Migration Horizontal Displacemant Vectors")
            ax1.set_xlabel("X [m]")
            ax1.set_ylabel("Y [m]")

            # Lower Subplot: Pre-migrated points (small black dots) and Migrated points (colored by depth)
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
                plt.savefig(os.path.join(plots_dir, "02_01_eikonal_migration_displacement_vectors.png"), dpi=300, bbox_inches="tight")

            if interactive:
                plt.draw()
                plt.pause(0.5)

            plt.close(fig)
        except Exception:
            pass

    return migrated_points
