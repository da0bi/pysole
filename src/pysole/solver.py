"""
Main Solver API for PySole package.
Implements the workflow for physically-informed bedrock interpolation & 3D migration,
caching surface gradients and smoothed slope grids for maximum computational efficiency.
"""

from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from .raster import BedrockMap, load_dem, load_outline, ensure_spatial_coords
from .migration import migrate_eikonal_points
from .variogram import optimize_bss_variance, calculate_variogram, fit_variogram_model
from .interpolation import blend_margin_topography, kriging_interpolation, random_forest_hole_filling
from .smoothing import compute_gradients, fft_gaussian_smooth


class Solver:
    """
    Physically-Informed Bedrock Interpolation & 3D Migration Solver.
    """

    def __init__(
        self,
        dem: Union[str, np.ndarray],
        outline: Union[str, np.ndarray, None] = None,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        kriging_method: str = "universal",
        variogram_model: str = "spherical",
        pre_kriging_method: Optional[str] = None,
        pre_drift_terms: Optional[List[str]] = None,
        pre_variogram_model: Optional[str] = None,
        pre_zero_boundary: bool = False,
        post_kriging_method: Optional[str] = None,
        post_drift_terms: Optional[List[str]] = None,
        post_variogram_model: Optional[str] = None,
        post_zero_boundary: bool = False,
        perform_migration: bool = True,
        survey_data_type: str = "one_way_travel_time",
        plots_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        dem : str or np.ndarray
            Path to surface DEM GeoTIFF/ASCII file, or 2D numpy array.
        outline : str or np.ndarray, optional
            Path to creeping body boundary polygon (Shapefile/GeoJSON) or boolean mask.
        dx : float, optional
            Pixel resolution along X. If None, derived directly from actual DEM metadata.
        dy : float, optional
            Pixel resolution along Y. If None, derived directly from actual DEM metadata.
        bounds : tuple of float, optional
            Spatial bounding box (minx, miny, maxx, maxy).
        pre_kriging_method : str, optional
            1st-pass pre-migration Kriging approach ('universal', 'ordinary', 'regression'). Default 'universal'.
        pre_drift_terms : list of str, optional
            1st-pass drift terms (e.g. ['quadratic'], ['regional_linear'], ['sia_thickness']). Default ['quadratic'].
        pre_variogram_model : str, optional
            1st-pass variogram model ('spherical', 'exponential', 'gaussian', 'linear'). Default 'spherical'.
        pre_zero_boundary : bool, optional
            If True, enforces a zero traveltime boundary condition (T=0) on the glacier margin outline.
        post_kriging_method : str, optional
            2nd-pass post-migration Kriging approach ('universal', 'ordinary', 'regression'). Default 'universal'.
        post_drift_terms : list of str, optional
            2nd-pass drift terms (e.g. ['sia_thickness'], ['quadratic'], ['regional_linear']). Default ['sia_thickness'].
        post_variogram_model : str, optional
            2nd-pass variogram model ('spherical', 'exponential', 'gaussian', 'linear'). Default 'spherical'.
        post_zero_boundary : bool, optional
            If True, enforces a zero thickness boundary condition (H=0) on the glacier margin outline.
        perform_migration : bool
            If True (default), performs 3D Eikonal ray migration on travel times. If False, skips migration.
        survey_data_type : str
            Type of input survey data defined in inputs section: 'one_way_travel_time' (default),
            'two_way_travel_time' (converts TWT/2), or 'thickness' / 'ice_thickness' (direct depth/thickness measurements, skips migration).
        plots_dir : str
            Directory where generated plots are automatically saved.
        """
        self.dem_grid, self.meta = load_dem(dem, dx=dx, dy=dy, bounds=bounds)
        self.outline_mask = load_outline(outline, self.dem_grid, self.meta)

        self.dx = float(self.meta.get("dx", 1.0))
        self.dy = float(self.meta.get("dy", 1.0))
        self.x_coords, self.y_coords, self.bounds = ensure_spatial_coords(
            self.dem_grid.shape,
            dx=self.dx,
            dy=self.dy,
            bounds=self.meta.get("bounds"),
        )

        # Migration & Kriging options (Pre-migration & Post-migration)
        self.pre_kriging_method = pre_kriging_method or kriging_method or "universal"
        self.pre_drift_terms = pre_drift_terms if pre_drift_terms is not None else ["quadratic"]
        self.pre_variogram_model = pre_variogram_model or variogram_model or "spherical"
        self.pre_zero_boundary = bool(pre_zero_boundary)

        self.post_kriging_method = post_kriging_method or kriging_method or "universal"
        if post_drift_terms is not None:
            self.post_drift_terms = post_drift_terms
        elif self.post_kriging_method in ["sia_thickness", "sia"]:
            self.post_drift_terms = ["sia_thickness"]
        else:
            self.post_drift_terms = ["sia_thickness"]
        self.post_variogram_model = post_variogram_model or variogram_model or "spherical"
        self.post_zero_boundary = bool(post_zero_boundary)

        # Legacy properties for backward compatibility
        self.kriging_method = self.post_kriging_method
        self.variogram_model = self.post_variogram_model
        self.drift_terms = self.post_drift_terms
        self.perform_migration = perform_migration
        self.survey_data_type = survey_data_type
        self.plots_dir = plots_dir

        # Gradient and Smoothed Slope Caches (saving filtered DEM surface slopes for each kc)
        self._gradient_cache: Optional[Dict[str, np.ndarray]] = None
        self._smoothed_slopes_cache: Dict[float, np.ndarray] = {}

        # Internal state
        self.survey_points: Optional[np.ndarray] = None
        self.migrated_points: Optional[np.ndarray] = None
        self.opt_kc: Optional[float] = None
        self.opt_slope: Optional[np.ndarray] = None
        self.kriged_bedrock: Optional[np.ndarray] = None
        self.kriged_variance: Optional[np.ndarray] = None
        self.rf_filled_bedrock: Optional[np.ndarray] = None
        self.blended_bedrock: Optional[np.ndarray] = None
        self.final_thickness: Optional[np.ndarray] = None
        self.final_grid: Optional[np.ndarray] = None

    @property
    def plot_extent(self) -> List[float]:
        """Returns Matplotlib plot extent [minx, maxx, miny, maxy]."""
        return [self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3]]

    def _get_dem_gradients(self) -> Dict[str, np.ndarray]:
        """
        Retrieves or computes surface DEM slope gradients, caching the result.
        """
        if self._gradient_cache is None:
            self._gradient_cache = compute_gradients(self.dem_grid, dx=self.dx, dy=self.dy)
        return self._gradient_cache

    def get_smoothed_slope(self, kc: float) -> np.ndarray:
        """
        Retrieves the saved filtered DEM surface slope grid for a given corner frequency kc.
        If not yet cached, computes and caches it.
        """
        kc_key = round(float(kc), 6)
        if kc_key not in self._smoothed_slopes_cache:
            grads = self._get_dem_gradients()
            base_slope = grads["slope_rad"]
            smoothed_slope, _, _ = fft_gaussian_smooth(base_slope, dx=self.dx, dy=self.dy, kc=kc)
            self._smoothed_slopes_cache[kc_key] = smoothed_slope
        return self._smoothed_slopes_cache[kc_key]

    @classmethod
    def from_config(cls, config_path: Union[str, os.PathLike] = "pysole.json") -> "Solver":
        """
        Initializes Solver instance using inputs defined in pysole.json configuration file.
        """
        from .config import load_config
        cfg = load_config(config_path)
        inputs = cfg.get("inputs", {})
        spatial = cfg.get("spatial_parameters", {})
        migration_cfg = cfg.get("migration_parameters", {})
        kriging_cfg = cfg.get("kriging_parameters", {})
        outputs = cfg.get("outputs", {})

        survey_dtype = inputs.get("survey_data_type") or migration_cfg.get("survey_data_type", "one_way_travel_time")

        pre_krig_cfg = kriging_cfg.get("pre_migration", {}) if isinstance(kriging_cfg.get("pre_migration"), dict) else {}
        post_krig_cfg = kriging_cfg.get("post_migration", {}) if isinstance(kriging_cfg.get("post_migration"), dict) else {}

        pre_method = pre_krig_cfg.get("method") or kriging_cfg.get("method", "universal")
        pre_drifts = pre_krig_cfg.get("drift_terms") or kriging_cfg.get("drift_terms", ["quadratic"])
        pre_var_model = pre_krig_cfg.get("variogram_model") or kriging_cfg.get("variogram_model", "spherical")
        pre_zero_boundary = pre_krig_cfg.get("include_zero_boundary_condition", False)

        post_method = post_krig_cfg.get("method") or kriging_cfg.get("method", "universal")
        post_drifts = post_krig_cfg.get("drift_terms") or kriging_cfg.get("drift_terms", ["sia_thickness"])
        post_var_model = post_krig_cfg.get("variogram_model") or kriging_cfg.get("variogram_model", "spherical")
        post_zero_boundary = post_krig_cfg.get("include_zero_boundary_condition", False)

        solver = cls(
            dem=inputs.get("dem_path"),
            outline=inputs.get("outline_path"),
            dx=spatial.get("dx"),
            dy=spatial.get("dy"),
            bounds=spatial.get("bounds"),
            pre_kriging_method=pre_method,
            pre_drift_terms=pre_drifts,
            pre_variogram_model=pre_var_model,
            pre_zero_boundary=pre_zero_boundary,
            post_kriging_method=post_method,
            post_drift_terms=post_drifts,
            post_variogram_model=post_var_model,
            post_zero_boundary=post_zero_boundary,
            perform_migration=migration_cfg.get("perform_migration", True),
            survey_data_type=survey_dtype,
            plots_dir=outputs.get("plots_dir", None),
        )
        return solver

    def migrate_eikonal(
        self,
        travel_times: Union[str, np.ndarray],
        velocity: Optional[float] = None,
        interactive: bool = False,
        plotit: bool = False,
    ) -> np.ndarray:
        """
        Migrates zero-offset GPR or seismic travel times into 3D space using the Eikonal equation.
        Supports survey_data_type:
          - 'one_way_travel_time' (default)
          - 'two_way_travel_time' (converts TWT / 2 to OWTT)
          - 'thickness' / 'ice_thickness' (direct depth/thickness measurements, skips migration)
        If interactive (interactive_migration=true), allows iterative velocity testing.
        """
        if isinstance(travel_times, str):
            try:
                pts = np.loadtxt(travel_times, delimiter="," if travel_times.endswith(".csv") else None)
            except ValueError:
                import pandas as pd
                df_tmp = pd.read_csv(travel_times)
                pts = df_tmp.select_dtypes(include=[np.number]).to_numpy()
        else:
            pts = np.array(travel_times, dtype=np.float64)

        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        # Check spatial coordinate alignment between survey points and DEM bounds
        minx, miny, maxx, maxy = self.bounds
        px_min, py_min = pts[:, 0].min(), pts[:, 1].min()
        px_max, py_max = pts[:, 0].max(), pts[:, 1].max()

        buf_x = max((maxx - minx) * 0.1, 1.0)
        buf_y = max((maxy - miny) * 0.1, 1.0)

        if (px_max < minx - buf_x) or (px_min > maxx + buf_x) or (py_max < miny - buf_y) or (py_min > maxy + buf_y):
            err_msg = (
                f"\n[Spatial Coordinate System Error] Survey profile coordinates do not match DEM spatial bounds!\n"
                f"  - Survey Points Extent: X=[{px_min:.2f}, {px_max:.2f}], Y=[{py_min:.2f}, {py_max:.2f}]\n"
                f"  - Surface DEM Bounds:  X=[{minx:.2f}, {maxx:.2f}], Y=[{miny:.2f}, {maxy:.2f}]\n"
                f"All input datasets (DEM, outline, survey profiles) must use the exact same Coordinate Reference System."
            )
            print(err_msg)
            raise ValueError(err_msg)

        if pts.shape[1] == 3:
            interp_z = self._sample_dem(pts[:, 0], pts[:, 1])
            pts = np.column_stack((pts[:, 0], pts[:, 1], interp_z, pts[:, 2]))

        # Handle survey_data_type conversion
        dtype_str = str(self.survey_data_type).lower().strip()

        if dtype_str in ["two_way_travel_time", "twt", "two_way"]:
            print("\n[Survey Data Type: TWT] Two-Way Traveltimes detected. Converting to One-Way Traveltimes (OWTT = TWT / 2.0).")
            pts[:, 3] = pts[:, 3] / 2.0
        elif dtype_str in ["one_way_travel_time", "owtt", "one_way"]:
            print("\n[Survey Data Type: OWTT] One-Way Traveltimes detected.")

        self.survey_points = pts

        # Handle Ice Thickness survey points -> skip 3D ray migration
        if dtype_str in ["thickness", "ice_thickness", "depth"]:
            print("\n[Survey Data Type: Ice Thickness] Input data represents direct ice thickness measurements. Skipping 3D Eikonal ray migration.")
            self.migrated_points = pts.copy()
            return self.migrated_points

        # Handle perform_migration = False
        if not self.perform_migration:
            print("\n[Migration Skipped] 'perform_migration' is set to False in configuration. Using unmigrated survey points directly.")
            vel = velocity if velocity is not None else 0.16
            unmig_depths = pts[:, 3] * vel if pts[:, 3].max() > 15.0 else pts[:, 3]
            self.migrated_points = np.column_stack((pts[:, 0], pts[:, 1], pts[:, 2], unmig_depths))
            return self.migrated_points

        if velocity is None:
            if interactive:
                try:
                    val = input("Signal Propagation Velocity [m/ns]? (e.g. 0.16): ").strip()
                    velocity = float(val) if val else 0.16
                except Exception:
                    velocity = 0.16
            else:
                velocity = 0.16

        # 1st Pass BSS surface slope optimization for pre-migration traveltimes (T_i * sin(alpha_opt1))
        self.optimize_bss(interactive=interactive, plotit=plotit or (self.plots_dir is not None))

        opt_slope_sin1 = np.sin(self.opt_slope)
        interp_slope1 = RegularGridInterpolator(
            (self.y_coords, self.x_coords),
            opt_slope_sin1,
            bounds_error=False,
            fill_value=np.nan,
        )

        pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))  # (Y, X)
        slopes_pts1 = interp_slope1(pts_xy)
        slopes_pts1 = np.maximum(np.nan_to_num(slopes_pts1, nan=0.1), 1e-4)

        # 1st Pass Point products P_i1 = T_i * sin(alpha_opt1)
        tt_pts = pts[:, 3]  # traveltimes
        product_pts1 = tt_pts * slopes_pts1

        sample_prod_pts1 = np.column_stack((pts[:, 0], pts[:, 1], product_pts1))

        # 1st Pass Product field Kriging interpolation P_i1 -> P1(x,y)
        # 1st pass uses Universal Kriging with quadratic spatial drift by default to produce a smooth, stable traveltime wavefront T(x,y)
        prod_grid1, _ = kriging_interpolation(
            sample_points=sample_prod_pts1,
            target_grid_shape=self.dem_grid.shape,
            bounds=self.bounds,
            method=self.pre_kriging_method,
            variogram_model=self.pre_variogram_model,
            dem_grid=self.dem_grid,
            drift_terms=self.pre_drift_terms,
            outline_mask=self.outline_mask,
            include_zero_boundary_condition=self.pre_zero_boundary,
        )

        # Retrieve continuous traveltime field T(x,y) = P1(x,y) / sin(alpha_opt1)
        min_slope_sin1 = np.sin(np.radians(2.0))
        safe_slope_grid1 = np.maximum(opt_slope_sin1, min_slope_sin1)
        tt_grid = np.maximum(prod_grid1 / safe_slope_grid1, 0.0)

        # Perform 3D Eikonal ray migration using retrieved traveltime field T(x,y)
        self._get_dem_gradients()

        self.migrated_points = migrate_eikonal_points(
            dem=self.dem_grid,
            travel_time_grid=tt_grid,
            survey_points=pts,
            velocity=velocity,
            dx=self.dx,
            dy=self.dy,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
            outline_mask=self.outline_mask,
            bounds=self.bounds,
            plots_dir=self.plots_dir,
            interactive=interactive,
            plotit=plotit or interactive or (self.plots_dir is not None),
        )

        # Interactive iterative migration velocity testing loop
        if interactive:
            while True:
                try:
                    ans = input(f"\nCurrent migration velocity v = {velocity:.4f} m/ns. Test another migration velocity? [y/N]: ").strip().lower()
                    if ans in ["y", "yes"]:
                        val = input("Enter new signal propagation velocity [m/ns] (e.g. 0.15): ").strip()
                        if val:
                            velocity = float(val)
                            print(f"Re-running 3D Eikonal ray migration with v = {velocity:.4f} m/ns...")
                            self.migrated_points = migrate_eikonal_points(
                                dem=self.dem_grid,
                                travel_time_grid=tt_grid,
                                survey_points=pts,
                                velocity=velocity,
                                dx=self.dx,
                                dy=self.dy,
                                x_coords=self.x_coords,
                                y_coords=self.y_coords,
                                outline_mask=self.outline_mask,
                                bounds=self.bounds,
                                plots_dir=self.plots_dir,
                                interactive=interactive,
                                plotit=True,
                            )
                    else:
                        break
                except Exception:
                    break

        return self.migrated_points

    def optimize_bss(
        self,
        kc_max: float = 10.0,
        kc_min: float = 0.01,
        d_kc: float = 0.1,
        num_steps: int = 10,
        prefix: Optional[str] = None,
        interactive: bool = False,
        plotit: bool = False,
    ) -> float:
        """
        Iterative optimization process to determine optimum surface slope smoothing degree kc
        matching FFTSmooth.m, saving filtered DEM surface slope grids for ALL evaluated kc values
        and saving variogram/variance plots to plots_dir.
        """
        pts = self.migrated_points if self.migrated_points is not None else self.survey_points
        if pts is None:
            xx, yy = np.meshgrid(self.x_coords[::5], self.y_coords[::5])
            pts = np.column_stack((xx.ravel(), yy.ravel(), self.dem_grid[::5, ::5].ravel(), np.ones(xx.size) * 10.0))

        if prefix is None:
            prefix = "03_" if self.migrated_points is not None else "01_"

        stage_name = "stage2" if prefix == "03_" else "stage1"

        should_plot = plotit or interactive or (self.plots_dir is not None)

        self.opt_kc, self.opt_slope, _, all_slopes = optimize_bss_variance(
            dem=self.dem_grid,
            survey_points=pts,
            dx=self.dx,
            dy=self.dy,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
            bounds=self.bounds,
            kc_max=kc_max,
            kc_min=kc_min,
            d_kc=d_kc,
            num_steps=num_steps,
            plots_dir=self.plots_dir,
            prefix=prefix,
            stage_name=stage_name,
            interactive=interactive,
            plotit=should_plot,
        )

        # Save all evaluated filtered DEM surface slope grids into cache
        self._smoothed_slopes_cache.update(all_slopes)
        return self.opt_kc

    def interpolate_kriging(
        self,
        method: Optional[str] = None,
        variogram_model: Optional[str] = None,
        plotit: bool = False,
        interactive: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Default primary step: Performs BSS surface slope optimization on survey/migrated points
        applying the product of depths d_i x sin(surface slopes).
        Kriging interpolates product field P(x,y) using configured Kriging approach,
        and reconstructs continuous bedrock thickness H(x,y) = P(x,y) / sin(alpha_opt) and elevation Z_bedrock(x,y).
        Saves bedrock & uncertainty plots to plots_dir.
        """
        pts = self.migrated_points if self.migrated_points is not None else self.survey_points
        if pts is None:
            raise ValueError("No survey or migrated points available. Run migrate_eikonal() first.")

        should_plot = interactive or (self.plots_dir is not None)

        # BSS surface slope optimization on depths (d_i x sin(surface slopes))
        self.optimize_bss(interactive=interactive, plotit=should_plot)

        opt_slope_sin = np.sin(self.opt_slope)
        interp_slope = RegularGridInterpolator(
            (self.y_coords, self.x_coords),
            opt_slope_sin,
            bounds_error=False,
            fill_value=np.nan,
        )

        pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))  # (Y, X)
        slopes_pts = interp_slope(pts_xy)
        slopes_pts = np.maximum(np.nan_to_num(slopes_pts, nan=0.1), 1e-4)

        thickness_pts = pts[:, 3]  # depth or thickness
        product_pts = thickness_pts * slopes_pts

        sample_prod_pts = np.column_stack((pts[:, 0], pts[:, 1], product_pts))

        krig_method = method if method is not None else self.kriging_method
        var_model = variogram_model if variogram_model is not None else self.variogram_model

        # Perform Kriging interpolation on product points P_i -> P(x,y)
        prod_grid, prod_var = kriging_interpolation(
            sample_points=sample_prod_pts,
            target_grid_shape=self.dem_grid.shape,
            bounds=self.bounds,
            method=krig_method,
            variogram_model=var_model,
            dem_grid=self.dem_grid,
            opt_slope_grid=self.opt_slope,
            drift_terms=self.drift_terms,
            outline_mask=self.outline_mask,
            include_zero_boundary_condition=self.post_zero_boundary,
        )

        # Reconstruct thickness H(x,y) = P(x,y) / sin(alpha_opt)
        # Apply physical safeguards:
        # 1. Minimum physical surface slope threshold (2.0 degrees)
        min_slope_sin = np.sin(np.radians(2.0))
        safe_slope_grid = np.maximum(opt_slope_sin, min_slope_sin)

        # 2. Maximum physical ice thickness cap (1.5 * max survey depth, or minimum 500m)
        max_thickness = max(float(np.max(thickness_pts)) * 1.5, 500.0)
        thickness_grid = np.clip(prod_grid / safe_slope_grid, 0.0, max_thickness)

        # 3. Mask out non-glacier / creeping body areas outside outline (H = 0 outside boundary)
        if self.outline_mask is not None:
            thickness_grid[~self.outline_mask] = 0.0

        self.kriged_bedrock = self.dem_grid - thickness_grid
        self.kriged_variance = prod_var / (safe_slope_grid**2)
        if self.outline_mask is not None:
            self.kriged_variance[~self.outline_mask] = 0.0

        plot_extent = self.plot_extent

        if should_plot:
            try:
                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                im1 = ax1.imshow(self.kriged_bedrock, extent=plot_extent, origin="lower", cmap="terrain")
                ax1.scatter(pts[:, 0], pts[:, 1], c="black", s=1.5, alpha=0.7, label="Survey Points")
                ax1.set_title("Interpolated Bedrock")
                ax1.set_xlabel("X [m]")
                ax1.set_ylabel("Y [m]")
                fig.colorbar(im1, ax=ax1, label="Elevation [m]")
                ax1.legend()

                im2 = ax2.imshow(self.kriged_variance, extent=plot_extent, origin="lower", cmap="plasma")
                ax2.set_title("Kriging Uncertainty")
                ax2.set_xlabel("X [m]")
                ax2.set_ylabel("Y [m]")
                fig.colorbar(im2, ax=ax2, label="Variance [\u03c3\u00b2]")

                plt.tight_layout()

                if self.plots_dir:
                    os.makedirs(self.plots_dir, exist_ok=True)
                    plt.savefig(os.path.join(self.plots_dir, "04_01_kriging_bedrock_elevation_and_uncertainty.png"), dpi=300, bbox_inches="tight")

                if interactive:
                    plt.draw()
                    plt.pause(0.5)

                plt.close(fig)
            except Exception:
                pass

        return self.kriged_bedrock, self.kriged_variance

    def fill_holes_rf(self) -> np.ndarray:
        """
        Trains a Random Forest ML model on Kriging interpolated data to fill remaining bedrock holes.
        """
        if self.kriged_bedrock is None:
            self.interpolate_kriging()

        self.rf_filled_bedrock = random_forest_hole_filling(
            dem=self.dem_grid,
            bedrock_grid=self.kriged_bedrock,
            boundary_mask=self.outline_mask,
            dx=self.dx,
            dy=self.dy,
        )
        return self.rf_filled_bedrock

    def apply_geomorph_smoothing(self, min_gap_dist: Optional[float] = None) -> np.ndarray:
        """
        Final step towards continuous bedrock: Geomorphological margin blending as the final step
        to assure a smooth transition from calculated bedrock to the surrounding surface DEM,
        pruning bedrock within min_gap_dist from the margin.

        Parameters
        ----------
        min_gap_dist : float, optional
            Minimum gap distance / margin blend zone width [m]. Prunes bedrock within this distance.
        """
        base_grid = self.rf_filled_bedrock if self.rf_filled_bedrock is not None else self.kriged_bedrock
        if base_grid is None:
            base_grid = self.fill_holes_rf()

        self.blended_bedrock = blend_margin_topography(
            dem=self.dem_grid,
            bedrock_input=base_grid,
            boundary_mask=self.outline_mask,
            dx=self.dx,
            dy=self.dy,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
            min_gap_dist=min_gap_dist,
        )
        return self.blended_bedrock

    def smooth_bedrock_dem(
        self,
        grid: np.ndarray,
        method: str = "gaussian",
        sigma: float = 1.5,
        kernel_size: int = 3,
        kc_cutoff: Optional[float] = None,
    ) -> np.ndarray:
        """
        Applies spatial smoothing to the calculated bedrock DEM grid.
        Supports 'gaussian', 'median', and 'fft_lowpass'.
        """
        method = method.lower().strip()
        out_grid = grid.copy()

        if method == "gaussian":
            from scipy.ndimage import gaussian_filter

            valid_mask = ~np.isnan(out_grid)
            if np.any(valid_mask):
                filled = np.where(valid_mask, out_grid, np.nanmean(out_grid))
                smoothed = gaussian_filter(filled, sigma=sigma)
                out_grid[valid_mask] = smoothed[valid_mask]

        elif method == "median":
            from scipy.ndimage import median_filter

            out_grid = median_filter(out_grid, size=kernel_size)

        elif method == "fft_lowpass":
            M, N = out_grid.shape
            kc = kc_cutoff if kc_cutoff is not None else (self.best_kc if hasattr(self, "best_kc") and self.best_kc else 1.0)

            kx = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)
            ky = 2.0 * np.pi * np.fft.fftfreq(M, d=self.dy)
            KX, KY = np.meshgrid(kx, ky)
            KR = np.sqrt(KX**2 + KY**2)

            H_filter = np.exp(-(KR**2) / (2.0 * (kc**2)))

            valid_mask = ~np.isnan(out_grid)
            filled = np.where(valid_mask, out_grid, np.nanmean(out_grid))

            F_grid = np.fft.fft2(filled)
            F_filtered = F_grid * H_filter
            smoothed = np.real(np.fft.ifft2(F_filtered))
            out_grid[valid_mask] = smoothed[valid_mask]

        return out_grid

    def finalize_topography(
        self,
        interactive: bool = True,
        plotit: bool = True,
        random_forest_gap_filling: Optional[bool] = None,
        apply_margin_blend: Optional[bool] = None,
        min_gap_dist: Optional[float] = None,
        smooth_bedrock: bool = False,
        smoothing_method: str = "gaussian",
        smoothing_sigma: float = 1.5,
        smoothing_kernel_size: int = 3,
        smoothing_kc_cutoff: Optional[float] = None,
    ) -> BedrockMap:
        """
        Executes the full final sequence towards the continuous bedrock result:
        1. Kriging product interpolation (performed by default)
        2. Optional spatial DEM smoothing (Gaussian, Median, or FFT low-pass)
        3. Plot interpolated bedrock field and Kriging uncertainty field
        4. Operator inspects plots and decides in terminal whether to apply Random Forest gap filling
        5. Plot calculated bedrock with surrounding terrain as last step before margin blending
        6. Prompt user if blend_margin_topography should be applied or not
        7. If yes, prompt for minimum gap distance and prune bedrock within this gap value.
        8. Save final ice thickness map figure.
        Saves generated figures into plots_dir directory.
        """
        should_plot = interactive or (self.plots_dir is not None)

        # 1. Default Kriging product interpolation
        self.interpolate_kriging(plotit=should_plot, interactive=interactive)

        # 2. Apply bedrock DEM spatial smoothing if enabled
        if smooth_bedrock:
            print(f"Applying bedrock DEM spatial smoothing ({smoothing_method})...")
            thick_raw = np.maximum(self.dem_grid - self.kriged_bedrock, 0.0)
            if self.outline_mask is not None:
                thick_raw[~self.outline_mask] = 0.0

            thick_smoothed = self.smooth_bedrock_dem(
                thick_raw,
                method=smoothing_method,
                sigma=smoothing_sigma,
                kernel_size=smoothing_kernel_size,
                kc_cutoff=smoothing_kc_cutoff,
            )
            if self.outline_mask is not None:
                thick_smoothed[~self.outline_mask] = 0.0

            self.kriged_bedrock = self.dem_grid - thick_smoothed

        # 3. User inspects plots and decides on Random Forest gap filling
        if random_forest_gap_filling is None:
            if interactive:
                try:
                    ans = input("\nBased on Kriging plots, do you want to apply Random Forest gap filling? [y/N]: ").strip().lower()
                    random_forest_gap_filling = ans in ["y", "yes"]
                except Exception:
                    random_forest_gap_filling = False
            else:
                random_forest_gap_filling = False

        if random_forest_gap_filling:
            self.fill_holes_rf()
            current_bedrock = self.rf_filled_bedrock
        else:
            self.rf_filled_bedrock = self.kriged_bedrock.copy()
            current_bedrock = self.kriged_bedrock

        plot_extent = self.plot_extent
        pts = self.migrated_points if self.migrated_points is not None else self.survey_points

        # 4. Plot calculated bedrock with surrounding terrain as last step before margin blending
        if should_plot:
            try:
                import matplotlib.pyplot as plt

                plt.figure("Calculated Bedrock Map", figsize=(8, 6))
                plt.clf()
                plt.imshow(current_bedrock, extent=plot_extent, origin="lower", cmap="terrain")
                plt.colorbar(label="Elevation [m]")
                min_b = np.nanmin(current_bedrock)
                max_b = np.nanmax(current_bedrock)

                if smooth_bedrock:
                    from scipy.ndimage import gaussian_filter
                    contour_grid = gaussian_filter(current_bedrock, sigma=max(smoothing_sigma, 2.5))
                else:
                    contour_grid = current_bedrock

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

                if self.outline_mask is not None:
                    plt.contour(
                        self.outline_mask,
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

                if self.plots_dir:
                    os.makedirs(self.plots_dir, exist_ok=True)
                    plt.savefig(os.path.join(self.plots_dir, "05_01_calculated_bedrock_map.png"), dpi=300, bbox_inches="tight")

                if interactive:
                    plt.draw()
                    plt.pause(0.5)
                plt.close()
            except Exception:
                pass

        # 5. Prompt user if blend_margin_topography should be applied
        if apply_margin_blend is None:
            if interactive:
                try:
                    ans_blend = input("\nDo you want to apply margin blending to surrounding terrain (blend_margin_topography)? [Y/n]: ").strip().lower()
                    apply_margin_blend = ans_blend not in ["n", "no"]
                except Exception:
                    apply_margin_blend = True
            else:
                apply_margin_blend = False

        # 6. If applied, prompt for minimum gap distance and prune bedrock within min_gap_dist
        if apply_margin_blend:
            if min_gap_dist is None:
                if interactive:
                    try:
                        val_gap = input("Enter minimum gap distance / margin width [m] (e.g. 50.0): ").strip()
                        min_gap_dist = float(val_gap) if val_gap else 50.0
                    except Exception:
                        min_gap_dist = 50.0
                else:
                    min_gap_dist = 50.0

            print(f"Applying geomorphological margin blending (min_gap_dist = {min_gap_dist:.1f} m)...")
            self.blended_bedrock = blend_margin_topography(
                dem=self.dem_grid,
                bedrock_input=current_bedrock,
                boundary_mask=self.outline_mask,
                dx=self.dx,
                dy=self.dy,
                x_coords=self.x_coords,
                y_coords=self.y_coords,
                min_gap_dist=min_gap_dist,
            )
        else:
            self.blended_bedrock = current_bedrock.copy()

        # Save final blended bedrock plot
        if should_plot and apply_margin_blend:
            try:
                import matplotlib.pyplot as plt

                plt.figure("Final Blended Bedrock Map", figsize=(8, 6))
                plt.clf()
                plt.imshow(self.blended_bedrock, extent=plot_extent, origin="lower", cmap="terrain")
                plt.colorbar(label="Elevation [m]")
                plt.title("Final Geomorphologically Blended Bedrock Elevation")
                plt.xlabel("X [m]")
                plt.ylabel("Y [m]")
                plt.tight_layout()

                if self.plots_dir:
                    os.makedirs(self.plots_dir, exist_ok=True)
                    plt.savefig(os.path.join(self.plots_dir, "06_01_final_blended_bedrock_map.png"), dpi=300, bbox_inches="tight")

                if interactive:
                    plt.draw()
                    plt.pause(0.5)
                plt.close()
            except Exception:
                pass

        # 7. Save final calculated ice thickness map & Kriging uncertainty plot in +- meters (07_01_final_ice_thickness_map.png)
        self.final_thickness = np.maximum(self.dem_grid - self.blended_bedrock, 0.0)
        if smooth_bedrock:
            self.final_thickness = self.smooth_bedrock_dem(
                self.final_thickness,
                method=smoothing_method,
                sigma=smoothing_sigma,
                kernel_size=smoothing_kernel_size,
                kc_cutoff=smoothing_kc_cutoff,
            )
        if self.outline_mask is not None:
            self.final_thickness[~self.outline_mask] = 0.0

        # Calculate Kriging standard uncertainty field in +- meters
        if self.kriged_variance is not None:
            self.kriged_std = np.sqrt(np.maximum(self.kriged_variance, 0.0))
            if self.outline_mask is not None:
                self.kriged_std[~self.outline_mask] = 0.0
        else:
            self.kriged_std = np.zeros_like(self.final_thickness)

        # Calculate mean ice thickness and mean uncertainty inside boundary outline
        if self.outline_mask is not None and np.any(self.outline_mask):
            mean_thick = float(np.mean(self.final_thickness[self.outline_mask]))
            mean_unc = float(np.mean(self.kriged_std[self.outline_mask]))
        else:
            mean_thick = float(np.mean(self.final_thickness))
            mean_unc = float(np.mean(self.kriged_std))

        if should_plot:
            try:
                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Subplot 1: Final Calculated Ice Thickness Map
                im1 = ax1.imshow(self.final_thickness, extent=plot_extent, origin="lower", cmap="YlGnBu")
                fig.colorbar(im1, ax=ax1, label="Thickness [m]")
                if self.outline_mask is not None:
                    ax1.contour(
                        self.outline_mask,
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

                # Subplot 2: Kriging Thickness Uncertainty Field (+- meters)
                im2 = ax2.imshow(self.kriged_std, extent=plot_extent, origin="lower", cmap="magma")
                fig.colorbar(im2, ax=ax2, label="Kriging Uncertainty [± m]")
                if self.outline_mask is not None:
                    ax2.contour(
                        self.outline_mask,
                        levels=[0.5],
                        extent=plot_extent,
                        origin="lower",
                        colors="white",
                        linewidths=1.5,
                        linestyles="--",
                    )
                ax2.set_title(f"Mean Kriging Uncertainty: ±{int(round(mean_unc))} m")
                ax2.set_xlabel("X [m]")
                ax2.set_ylabel("Y [m]")

                plt.tight_layout()

                if self.plots_dir:
                    os.makedirs(self.plots_dir, exist_ok=True)
                    plt.savefig(os.path.join(self.plots_dir, "07_01_final_ice_thickness_map.png"), dpi=300, bbox_inches="tight")

                if interactive:
                    plt.draw()
                    plt.pause(0.5)
                plt.close(fig)
            except Exception:
                pass

        self.final_grid = self.blended_bedrock
        return BedrockMap(
            grid=self.final_grid,
            bounds=self.bounds,
            crs=self.meta.get("crs"),
            transform=self.meta.get("transform"),
            name="final_bedrock",
        )

    def _sample_dem(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (self.y_coords, self.x_coords),
            self.dem_grid,
            bounds_error=False,
            fill_value=np.nan,
        )
        return interp(np.column_stack((y, x)))
