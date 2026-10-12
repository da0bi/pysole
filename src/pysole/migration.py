"""
3D Eikonal Ray Migration Module for PySole.
Ported from MATLAB script MIG.m by Daniel Binder (2011).
Performs 3D Eikonal ray migration on zero-offset seismic / GPR travel times over complex surface topography,
computing non-orthogonal slowness vector components (sx, sy, sz) and horizontal/vertical ray displacements.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .smoothing import compute_gradients
from .raster import GridGeometry
from .logging import logger, get_progress_bar


@dataclass
class MigrationResult:
    """
    Data container storing 3D Eikonal ray migration results and displacement fields.

    Attributes
    ----------
    migrated_points : np.ndarray
        2D array of relocated survey points with shape ``(N, >=4)``. Columns correspond to:
        - Column 0: Relocated X coordinate (easting) [m].
        - Column 1: Relocated Y coordinate (northing) [m].
        - Column 2: Surface DEM elevation Z_surface at the relocated position [m].
        - Column 3: Migrated true perpendicular ice thickness / depth [m].
        - Additional columns (e.g., survey profile ID) are preserved if present.
    dx_grid : np.ndarray
        2D raster grid of horizontal ray displacements along the X dimension [m].
    dy_grid : np.ndarray
        2D raster grid of horizontal ray displacements along the Y dimension [m].
    dz_grid : np.ndarray
        2D raster grid of vertical ray displacements along the Z dimension [m].
    """
    migrated_points: np.ndarray
    dx_grid: np.ndarray
    dy_grid: np.ndarray
    dz_grid: np.ndarray


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
        self.dem_grads = (
            dem_grads
            if dem_grads is not None
            else compute_gradients(dem, dx=self.geometry.dx, dy=self.geometry.dy)
        )

    def migrate(
        self,
        travel_time_grid: np.ndarray,
        survey_points: np.ndarray,
        velocity: float = 0.16,
        plots_dir: str | Path | None = None,
        interactive: bool = False,
        show_progress: bool = True,
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
        plots_dir : str or Path, optional
            Output folder for diagnostic migration displacement vector figures.
        interactive : bool, default False
            If True, displays interactive Matplotlib figures during migration.
        show_progress : bool, default True
            If True, displays a progress bar during point relocation.

        Returns
        -------
        result : MigrationResult
            Migration result container storing relocated picks and displacement grids.
        """
        return migrate_eikonal_points(
            dem=self.dem,
            travel_time_grid=travel_time_grid,
            survey_points=survey_points,
            geometry=self.geometry,
            velocity=velocity,
            outline_mask=self.outline_mask,
            plots_dir=plots_dir,
            interactive=interactive,
            dem_grads=self.dem_grads,
            show_progress=show_progress,
        )

    def migrate_points(self, points: np.ndarray, velocity: float = 0.16) -> np.ndarray:
        """
        Convenience method to migrate a survey point set using a homogeneous traveltime field.

        Parameters
        ----------
        points : np.ndarray
            Input sounding points array ``[X, Y, Z, travel_time, ...]``.
        velocity : float, default 0.16
            Radar wave velocity in glacier ice [m/ns].

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
        res = self.migrate(travel_time_grid=tt_grid, survey_points=points, velocity=velocity, show_progress=False)
        return res.migrated_points


def migrate_eikonal_points(
    dem: np.ndarray,
    travel_time_grid: np.ndarray,
    survey_points: np.ndarray,
    geometry: GridGeometry,
    velocity: float = 0.16,
    outline_mask: np.ndarray | None = None,
    plots_dir: str | Path | None = None,
    interactive: bool = False,
    dem_grads: dict[str, np.ndarray] | None = None,
    show_progress: bool = True,
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
       relocated positions :math:`(X_{\\text{mig}}, Y_{\\text{mig}})` and true
       perpendicular ice thickness.

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

    Returns
    -------
    result : MigrationResult
        Container holding migrated points array and 2D displacement vector grids.
    """
    dx = geometry.dx
    dy = geometry.dy
    x_coords = geometry.x_coords
    y_coords = geometry.y_coords

    # 1. Calculate horizontal slownesses s1 = -\partial T / \partial x and s2 = -\partial T / \partial y
    tt_grads = compute_gradients(travel_time_grid, dx=dx, dy=dy)
    s1_grid = -tt_grads["slope_x"]  # -\partial T / \partial x
    s2_grid = -tt_grads["slope_y"]  # -\partial T / \partial y

    # 2. Retrieve pre-computed surface DEM directional slope components zx = \partial Z / \partial x, zy = \partial Z / \partial y
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
    evanescent_mask = disc_grid < 0.0
    evanescent_count = int(np.sum(evanescent_mask))
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

    dxi = np.nan_to_num(interpolated_disp[:, 0], nan=0.0)
    dyi = np.nan_to_num(interpolated_disp[:, 1], nan=0.0)
    dzi = interpolated_disp[:, 2]

    interp_dem = geometry.create_interpolator(dem, fill_value=np.nan)

    # Migrated coordinates & thickness adjustment for surface elevation delta between source and relocated point
    x_mig = survey_points[:, 0] + dxi
    y_mig = survey_points[:, 1] + dyi

    z_s0 = interp_dem(pts_xy)
    pts_mig_xy = np.column_stack((y_mig, x_mig))
    z_smig = interp_dem(pts_mig_xy)

    delta_zs = np.where(np.isnan(z_smig - z_s0), 0.0, z_smig - z_s0)
    d_mig = np.maximum(-dzi + delta_zs, 0.0)

    # Fallback to unmigrated depth if dzi interpolation is zero or invalid
    unmig_d = survey_points[:, 3] * velocity
    valid_d = (~np.isnan(dzi)) & (d_mig > 0)
    fallback_count = int(np.sum(~valid_d))
    if fallback_count > 0:
        logger.info(
            f"   [3D Ray Migration] {fallback_count} survey picks fell back to unmigrated coordinates "
            f"due to boundary NaN gradients."
        )
    d_mig[~valid_d] = unmig_d[~valid_d]

    # Sample surface DEM elevation at migrated (x_mig, y_mig)
    mig_pts_xy = np.column_stack((y_mig, x_mig))
    z_surf_mig = interp_dem(mig_pts_xy)

    # Clean NaNs in z_surf_mig using original surface z
    nan_z = np.isnan(z_surf_mig)
    z_surf_mig[nan_z] = survey_points[nan_z, 2]

    if survey_points.shape[1] >= 5:
        migrated_points = np.column_stack((x_mig, y_mig, z_surf_mig, d_mig, survey_points[:, 4]))
    else:
        migrated_points = np.column_stack((x_mig, y_mig, z_surf_mig, d_mig))

    # 6. Generate 3D Migration Displacement Vector diagnostic figures
    should_plot = interactive or (plots_dir is not None)
    if should_plot:
        from .plotting import plot_migration_displacement_vectors
        plot_migration_displacement_vectors(
            travel_time_grid=travel_time_grid,
            survey_points=survey_points,
            migrated_points=migrated_points,
            dxi=dxi,
            dyi=dyi,
            x_coords=x_coords,
            y_coords=y_coords,
            outline_mask=outline_mask,
            plots_dir=plots_dir,
            interactive=interactive,
        )

    return MigrationResult(
        migrated_points=migrated_points,
        dx_grid=dx_grid,
        dy_grid=dy_grid,
        dz_grid=dz_grid,
    )
