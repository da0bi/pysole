"""
3D Eikonal Ray Migration Module for PySole.
Ported from MATLAB script MIG.m by Daniel Binder (2011).
Performs 3D Eikonal ray migration on zero-offset seismic / GPR travel times over complex surface topography,
computing non-orthogonal slowness vector components (sx, sy, sz) and horizontal/vertical ray displacements.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .config import validate_max_ray_angle, validate_ray_angle_method
from .smoothing import compute_gradients
from .raster import GridGeometry
from .logging import logger, get_progress_bar


@dataclass
class MigrationResult:
    """
    Data container storing 3D Eikonal ray migration results, ray metrics, and displacement fields.

    Attributes
    ----------
    migrated_points : np.ndarray
        2D array of retained relocated survey points (``ray_angle <= max_ray_angle_deg``) with shape
        ``(N_valid, 4)`` or ``(N_valid, 5)``: ``[X_mig, Y_mig, Z_surface, Depth_mig, (profile_id)]``.
    all_migrated_points : np.ndarray
        2D array of all ``N`` relocated survey points (including picks exceeding ``max_ray_angle_deg``)
        with shape ``(N, 7)`` or ``(N, 8)``:
        ``[X_mig, Y_mig, Z_surface, Depth_mig, horizontal_ds, azimuth, ray_angle, (profile_id)]``.
    dx_grid : np.ndarray
        2D raster grid of horizontal ray displacements along the X dimension [m].
    dy_grid : np.ndarray
        2D raster grid of horizontal ray displacements along the Y dimension [m].
    dz_grid : np.ndarray
        2D raster grid of vertical ray displacements along the Z dimension [m].
    horizontal_ds : np.ndarray
        1D array of horizontal shift distances [m] for all ``N`` survey picks.
    azimuth : np.ndarray
        1D array of horizontal shift compass bearings [deg] in ``[0, 360)`` (0° = North, 90° = East).
    ray_angle : np.ndarray
        1D array of 3D ray path angles from vertical/nadir [deg] in ``[0, 90]`` (0° = vertical, 90° = horizontal),
        measured as set by ``ray_angle_method``. Picks whose migrated depth collapsed to <= 0 (near-horizontal
        rays) are kept at their unmigrated position and get 90°.
    valid_ray_mask : np.ndarray
        1D boolean mask indicating which survey picks satisfied ``ray_angle <= max_ray_angle_deg``.
    """
    migrated_points: np.ndarray
    all_migrated_points: np.ndarray
    dx_grid: np.ndarray
    dy_grid: np.ndarray
    dz_grid: np.ndarray
    horizontal_ds: np.ndarray
    azimuth: np.ndarray
    ray_angle: np.ndarray
    valid_ray_mask: np.ndarray


class EikonalMigrator:
    """
    Engine executing 3D Eikonal Ray Migration over complex surface topography.

    In steep alpine terrain, radar signals reflected from subsurface bedrock reflectors
    do not propagate along vertical raypaths. Instead, rays originate perpendicular to
    the reflector and arrive obliquely at the glacier surface. This engine solves the
    3D Eikonal slowness equations in a non-orthogonal coordinate system bounded by the
    sloping surface DEM, computing the horizontal and vertical vector displacements
    required to relocate scattered survey picks to their true 3D reflection points.

    Parameters
    ----------
    dem : np.ndarray
        2D raster grid of surface elevations [m] with shape ``(height, width)``.
    geometry : GridGeometry
        Standardized spatial coordinate container defining cell size and spatial extents.
    outline_mask : np.ndarray, optional
        Boolean mask of active glacier area.
    dem_grads : dict of str to np.ndarray, optional
        Pre-computed spatial gradient dictionary from :func:`~pysole.smoothing.compute_gradients`.
    """

    def __init__(
        self,
        dem: np.ndarray,
        geometry: GridGeometry,
        outline_mask: np.ndarray | None = None,
        dem_grads: dict[str, np.ndarray] | None = None,
    ):
        self.dem = dem
        self.geometry = geometry
        self.outline_mask = outline_mask
        self.dem_grads = dem_grads

    def migrate(
        self,
        travel_time_grid: np.ndarray,
        survey_points: np.ndarray,
        velocity: float = 0.16,
        max_ray_angle_deg: float | None = 60.0,
        plots_dir: str | Path | None = None,
        interactive: bool = False,
        show_progress: bool = True,
        output_prefix: str = "",
        horizontal_displacement_visualization: str = "arrows",
        ray_angle_method: str = "ray",
    ) -> MigrationResult:
        """
        Executes 3D Eikonal ray migration for a continuous traveltime field and scattered picks.

        Parameters
        ----------
        travel_time_grid : np.ndarray
            Continuous 2D grid of one-way traveltimes [ns] matching the DEM dimensions.
        survey_points : np.ndarray
            2D array of scattered survey sounding points ``[X, Y, Z_surf, travel_time, ...]``.
        velocity : float, default 0.16
            Electromagnetic radar signal propagation velocity in ice [m/ns]
            (typically ~0.160 m/ns in temperate ice/firn, up to ~0.168 m/ns in cold ice).
        max_ray_angle_deg : float or None, default 60.0
            Maximum allowed 3D ray angle from vertical [deg] in ``(0, 90]``. Picks exceeding
            this angle are discarded from ``result.migrated_points``. Set to ``None`` to disable.
        plots_dir : str or Path, optional
            Output folder for diagnostic migration displacement vector figures.
        interactive : bool, default False
            If True, displays interactive Matplotlib figures during migration.
        show_progress : bool, default True
            If True, displays a progress bar during point relocation.
        output_prefix : str, default ""
            Output prefix for diagnostic figures.
        horizontal_displacement_visualization : str, default "arrows"
            Visualization style for horizontal displacements: 'arrows' or 'grad_traveltime_grid'.
        ray_angle_method : str, default "ray"
            How ``ray_angle`` is measured: ``"ray"`` (ray displacement vector) or ``"depth"``
            (horizontal shift over migrated depth). See :func:`migrate_eikonal_points`.

        Returns
        -------
        result : MigrationResult
            Migration result container storing retained picks, full export table, and displacement grids.
        """
        if self.dem_grads is None:
            self.dem_grads = compute_gradients(self.dem, dx=self.geometry.dx, dy=self.geometry.dy)
        return migrate_eikonal_points(
            dem=self.dem,
            travel_time_grid=travel_time_grid,
            survey_points=survey_points,
            geometry=self.geometry,
            velocity=velocity,
            max_ray_angle_deg=max_ray_angle_deg,
            outline_mask=self.outline_mask,
            plots_dir=plots_dir,
            interactive=interactive,
            dem_grads=self.dem_grads,
            show_progress=show_progress,
            output_prefix=output_prefix,
            horizontal_displacement_visualization=horizontal_displacement_visualization,
            ray_angle_method=ray_angle_method,
        )

    def migrate_points(
        self,
        points: np.ndarray,
        velocity: float = 0.16,
        max_ray_angle_deg: float | None = 60.0,
    ) -> np.ndarray:
        """
        Convenience method to migrate a survey point set using a homogeneous traveltime field.

        Parameters
        ----------
        points : np.ndarray
            Input sounding points array ``[X, Y, Z, travel_time, ...]``.
        velocity : float, default 0.16
            Radar wave velocity in glacier ice [m/ns].
        max_ray_angle_deg : float or None, default 60.0
            Maximum allowed 3D ray angle from vertical [deg].

        Returns
        -------
        migrated_points : np.ndarray
            Relocated points array ``[X_mig, Y_mig, Z_surf_mig, Depth_mig, ...]``.
        """
        if len(points) > 0 and points.shape[1] >= 4 and not np.all(np.isnan(points[:, 3])):
            mean_tt = float(np.nanmean(points[:, 3]))
            tt_grid = np.full_like(self.dem, mean_tt)
        else:
            tt_grid = np.zeros_like(self.dem)
        res = self.migrate(
            travel_time_grid=tt_grid,
            survey_points=points,
            velocity=velocity,
            max_ray_angle_deg=max_ray_angle_deg,
            show_progress=False,
        )
        return res.migrated_points


def migrate_eikonal_points(
    dem: np.ndarray,
    travel_time_grid: np.ndarray,
    survey_points: np.ndarray,
    geometry: GridGeometry,
    velocity: float = 0.16,
    max_ray_angle_deg: float | None = 60.0,
    outline_mask: np.ndarray | None = None,
    plots_dir: str | Path | None = None,
    interactive: bool = False,
    dem_grads: dict[str, np.ndarray] | None = None,
    show_progress: bool = True,
    output_prefix: str = "",
    horizontal_displacement_visualization: str = "arrows",
    ray_angle_method: str = "ray",
) -> MigrationResult:
    """
    Relocates scattered radar soundings into 3D space using 3D Eikonal ray normal migration.

    In complex alpine topography, zero-offset radar reflections originate perpendicular
    to bedrock surfaces. If unmigrated, these reflections are erroneously positioned
    vertically beneath the receiver antenna, exaggerating ice thickness in U-shaped
    valleys and placing valley walls incorrectly.

    This function solves the 3D Eikonal slowness equations over an arbitrary sloping
    surface boundary:
    1. Computes apparent horizontal slownesses:
       :math:`s_1 = -\\partial T / \\partial x`, :math:`s_2 = -\\partial T / \\partial y`.
    2. Incorporates surface topographic slopes:
       :math:`z_x = \\partial Z / \\partial x`, :math:`z_y = \\partial Z / \\partial y`.
    3. Solves the quadratic Eikonal condition for true vertical slowness :math:`p_z`
       and horizontal slownesses :math:`(p_x, p_y)`.
    4. Evaluates 3D ray displacement vectors:
       :math:`(\\Delta x, \\Delta y, \\Delta z) = T \\cdot v^2 \\cdot (p_x, p_y, p_z)`.
    5. Interpolates displacement vectors at scattered sounding coordinates to yield
       relocated positions :math:`(X_{\\text{mig}}, Y_{\\text{mig}})`, true perpendicular ice
       thickness, horizontal shift distance ``horizontal_ds``, compass ``azimuth``, and
       ``ray_angle`` from vertical, filtering picks that exceed ``max_ray_angle_deg``.

    Parameters
    ----------
    dem : np.ndarray
        2D surface DEM grid :math:`Z(x, y)` [m].
    travel_time_grid : np.ndarray
        Continuous smoothed one-way traveltime grid :math:`T(x, y)` [ns].
    survey_points : np.ndarray
        Input survey sounding points ``[X, Y, Z_surf, travel_time, ...]``.
    geometry : GridGeometry
        Standardized spatial geometry container for raster grids.
    velocity : float, default 0.16
        Electromagnetic signal propagation velocity [m/ns]
        (~0.160 m/ns in temperate ice/firn, ~0.168 m/ns in cold pure ice).
    max_ray_angle_deg : float or None, default 60.0
        Maximum allowed 3D ray path angle from vertical [deg] in ``(0, 90]``.
        Migrated picks with ``ray_angle > max_ray_angle_deg`` are discarded from
        ``result.migrated_points`` while preserved in ``result.all_migrated_points``.
    outline_mask : np.ndarray, optional
        Boolean glacier outline mask.
    plots_dir : str or Path, optional
        Directory where generated displacement vector figures are saved.
    interactive : bool, default False
        Whether to display figures interactively.
    dem_grads : dict of str to np.ndarray, optional
        Pre-computed surface DEM gradients dictionary.
    show_progress : bool, default True
        Whether to show terminal progress bar during ray relocation.
    output_prefix : str, default ""
        Output prefix for diagnostic figures.
    horizontal_displacement_visualization : str, default "arrows"
        Visualization style for horizontal displacements: 'arrows' or 'grad_traveltime_grid'.
    ray_angle_method : str, default "ray"
        How ``ray_angle`` (and thus the ``max_ray_angle_deg`` filter) is measured:

        - ``"ray"``: angle of the ray displacement vector from vertical,
          ``atan(horizontal_ds / vertical ray displacement)``.
        - ``"depth"``: ``atan(horizontal_ds / migrated depth)``, with the depth measured below the
          surface at the relocated position (includes the surface elevation change between pick
          and relocated position).

        In both cases picks whose migrated depth collapsed to <= 0 (near-horizontal rays) get 90°.

    Returns
    -------
    result : MigrationResult
        Container holding retained migrated points, full export table, ray metrics, and 2D displacement grids.
    """
    eff_max_angle = validate_max_ray_angle(max_ray_angle_deg, "max_ray_angle_deg")
    angle_method = validate_ray_angle_method(ray_angle_method, "ray_angle_method")

    dx = geometry.dx
    dy = geometry.dy
    x_coords = geometry.x_coords
    y_coords = geometry.y_coords

    # 1. Calculate horizontal slownesses s1 = -\partial T / \partial x and s2 = -\partial T / \partial y
    tt_grads = compute_gradients(travel_time_grid, dx=dx, dy=dy)
    s1_grid = -tt_grads["slope_x"]
    s2_grid = -tt_grads["slope_y"]

    # 2. Retrieve surface DEM directional slope components zx = \partial Z / \partial x, zy = \partial Z / \partial y
    if dem_grads is None:
        dem_grads = compute_gradients(dem, dx=dx, dy=dy)

    zx_grid = dem_grads["slope_x"]
    zy_grid = dem_grads["slope_y"]

    # 3. Closed-form Eikonal 3D ray slowness vector formulation (exact for arbitrary oblique slopes zx, zy)
    # Ref: p_z = (B - \sqrt{B^2 - A C}) / A, where A = 1 + zx^2 + zy^2, B = s1*zx + s2*zy, C = s1^2 + s2^2 - 1/v^2
    v_inv = 1.0 / max(velocity, 1e-4)
    v_inv_sq = v_inv**2

    A_grid = 1.0 + zx_grid**2 + zy_grid**2
    B_grid = s1_grid * zx_grid + s2_grid * zy_grid
    C_grid = s1_grid**2 + s2_grid**2 - v_inv_sq

    disc_grid = B_grid**2 - A_grid * C_grid
    evanescent_count = int(np.sum(disc_grid < 0.0))
    if evanescent_count > 0:
        logger.info(
            f"   [3D Ray Migration] {evanescent_count} grid cells triggered evanescent wave condition "
            f"(|s_h| > 1/v); vertical slowness s3 clamped to 0.0."
        )

    disc_clamped = np.maximum(disc_grid, 0.0)
    pz_grid = (B_grid - np.sqrt(disc_clamped)) / A_grid
    px_grid = s1_grid - pz_grid * zx_grid
    py_grid = s2_grid - pz_grid * zy_grid

    # 4. Ray displacement vector grids: dx_grid, dy_grid, dz_grid
    v_sq = velocity**2
    dx_grid = travel_time_grid * v_sq * px_grid
    dy_grid = travel_time_grid * v_sq * py_grid
    dz_grid = travel_time_grid * v_sq * pz_grid

    # 5. Vectorized multi-channel interpolation of 3D ray displacement vectors at scattered survey locations
    with get_progress_bar(
        total=len(survey_points),
        desc="   [3D Ray Migration] Relocating survey picks",
        unit="picks",
        disable=not show_progress,
    ) as pbar:
        pts_xy = np.column_stack((survey_points[:, 1], survey_points[:, 0]))  # (Y, X)

        displacement_stack = np.stack([dx_grid, dy_grid, dz_grid], axis=-1)
        interp_vectors = geometry.create_interpolator(displacement_stack, fill_value=0.0)
        interpolated_disp = interp_vectors(pts_xy)

        pbar.update(len(survey_points))

    dzi = interpolated_disp[:, 2]
    nan_disp = np.isnan(dzi)
    dxi = np.where(nan_disp, 0.0, np.nan_to_num(interpolated_disp[:, 0], nan=0.0))
    dyi = np.where(nan_disp, 0.0, np.nan_to_num(interpolated_disp[:, 1], nan=0.0))

    # Relocated horizontal coordinates
    x_mig = survey_points[:, 0] + dxi
    y_mig = survey_points[:, 1] + dyi

    # Single surface DEM evaluation at source and relocated coordinates
    interp_dem = geometry.create_interpolator(dem, fill_value=np.nan)
    z_s0 = interp_dem(pts_xy)
    z_surf_mig = interp_dem(np.column_stack((y_mig, x_mig)))
    nan_z = np.isnan(z_surf_mig)
    z_surf_mig[nan_z] = survey_points[nan_z, 2]

    delta_zs = np.where(np.isnan(z_surf_mig - z_s0), 0.0, z_surf_mig - z_s0)
    unmig_d = survey_points[:, 3] * velocity
    d_mig = np.where(nan_disp, unmig_d, np.maximum(-dzi + delta_zs, 0.0))

    # Fallback to unmigrated coordinates if dzi interpolation hit boundary NaNs or non-positive depth.
    # Non-positive depth of a pick with a positive traveltime means the ray became (near-)horizontal:
    # these "collapsed" picks get ray_angle = 90° below, so max_ray_angle_deg removes them.
    # Zero-traveltime picks (ice-free, depth 0) are no collapse and stay vertical.
    valid_d = (~nan_disp) & (d_mig > 0)
    collapsed = (~nan_disp) & (d_mig <= 0) & (unmig_d > 0)
    boundary_count = int(np.sum(nan_disp))
    collapsed_count = int(np.sum(collapsed))
    if boundary_count > 0:
        logger.info(
            f"   [3D Ray Migration] {boundary_count} survey picks fell back to unmigrated coordinates "
            f"due to boundary NaN gradients."
        )
    if collapsed_count > 0:
        logger.info(
            f"   [3D Ray Migration] {collapsed_count} survey picks with non-positive migrated depth "
            f"(near-horizontal rays) fell back to unmigrated coordinates; ray_angle set to 90°."
        )
    d_mig[~valid_d] = unmig_d[~valid_d]
    x_mig[~valid_d] = survey_points[~valid_d, 0]
    y_mig[~valid_d] = survey_points[~valid_d, 1]
    z_surf_mig[~valid_d] = np.where(np.isnan(z_s0[~valid_d]), survey_points[~valid_d, 2], z_s0[~valid_d])
    dxi[~valid_d] = 0.0
    dyi[~valid_d] = 0.0

    # Compute ray shift metrics: horizontal_ds [m], azimuth [0°, 360°) (0°=N, 90°=E), ray_angle [0°, 90°] (0°=nadir)
    horizontal_ds = np.hypot(dxi, dyi)
    azimuth = np.where(horizontal_ds > 0.0, np.mod(np.degrees(np.arctan2(dxi, dyi)), 360.0), 0.0)
    azimuth = np.where(np.isclose(azimuth, 360.0, atol=1e-9), 0.0, azimuth)
    if angle_method == "depth":
        vertical_extent = d_mig
    else:
        vertical_extent = np.where(~valid_d, unmig_d, np.maximum(-dzi, 0.0))
    ray_angle = np.where(
        horizontal_ds > 0.0,
        np.clip(np.degrees(np.arctan2(horizontal_ds, vertical_extent)), 0.0, 90.0),
        0.0,
    )
    ray_angle[collapsed] = 90.0

    if survey_points.shape[1] >= 5:
        all_migrated_points = np.column_stack(
            (x_mig, y_mig, z_surf_mig, d_mig, horizontal_ds, azimuth, ray_angle, survey_points[:, 4])
        )
        full_pass2_points = np.column_stack((x_mig, y_mig, z_surf_mig, d_mig, survey_points[:, 4]))
    else:
        all_migrated_points = np.column_stack(
            (x_mig, y_mig, z_surf_mig, d_mig, horizontal_ds, azimuth, ray_angle)
        )
        full_pass2_points = np.column_stack((x_mig, y_mig, z_surf_mig, d_mig))

    # Filter points exceeding max_ray_angle_deg for subsequent depth grid interpolation
    if eff_max_angle is not None and len(survey_points) > 0:
        valid_ray_mask = ray_angle <= eff_max_angle
        discarded_count = int(np.sum(~valid_ray_mask))
        max_obs_angle = float(np.nanmax(ray_angle))
        logger.info(
            f"   [3D Ray Migration] Discarded {discarded_count} of {len(survey_points)} survey picks "
            f"exceeding max_ray_angle_deg = {eff_max_angle:.1f}\u00b0 (max observed ray angle = {max_obs_angle:.1f}\u00b0)."
        )
        if np.sum(valid_ray_mask) == 0:
            raise ValueError(
                f"All {len(survey_points)} migrated survey picks exceeded max_ray_angle_deg = {eff_max_angle:.1f}\u00b0 "
                f"(max observed ray angle = {max_obs_angle:.1f}\u00b0). Increase 'migration_parameters.max_ray_angle_deg'."
            )
        migrated_points = full_pass2_points[valid_ray_mask]
        discarded_points = full_pass2_points[~valid_ray_mask]
    else:
        valid_ray_mask = np.ones(len(survey_points), dtype=bool)
        migrated_points = full_pass2_points
        discarded_points = full_pass2_points[:0]

    # 6. Generate 3D Migration Displacement Vector diagnostic figures
    should_plot = interactive or (plots_dir is not None)
    if should_plot:
        from .plotting import plot_migration_displacement_vectors
        plot_migration_displacement_vectors(
            travel_time_grid=travel_time_grid,
            survey_points=survey_points,
            migrated_points=migrated_points,
            dxi=dxi[valid_ray_mask],
            dyi=dyi[valid_ray_mask],
            x_coords=x_coords,
            y_coords=y_coords,
            outline_mask=outline_mask,
            plots_dir=plots_dir,
            interactive=interactive,
            output_prefix=output_prefix,
            horizontal_displacement_visualization=horizontal_displacement_visualization,
            discarded_points=discarded_points,
            max_ray_angle_deg=eff_max_angle,
            quiver_origins=survey_points[valid_ray_mask],
        )
        if horizontal_displacement_visualization == "grad_traveltime_grid":
            from .plotting import plot_migration_horizontal_displacements
            plot_migration_horizontal_displacements(
                travel_time_grid=travel_time_grid,
                survey_points=survey_points,
                migrated_points=migrated_points,
                x_coords=x_coords,
                y_coords=y_coords,
                dx=dx,
                dy=dy,
                outline_mask=outline_mask,
                plots_dir=plots_dir,
                interactive=interactive,
                output_prefix=output_prefix,
            )

    return MigrationResult(
        migrated_points=migrated_points,
        all_migrated_points=all_migrated_points,
        dx_grid=dx_grid,
        dy_grid=dy_grid,
        dz_grid=dz_grid,
        horizontal_ds=horizontal_ds,
        azimuth=azimuth,
        ray_angle=ray_angle,
        valid_ray_mask=valid_ray_mask,
    )

