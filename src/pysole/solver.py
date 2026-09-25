"""
Main Solver API for PySole package.
Implements the workflow for physically-informed bedrock interpolation & 3D migration,
caching surface gradients and smoothed slope grids for maximum computational efficiency.
"""

from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from .raster import BedrockMap, load_dem, load_outline, ensure_spatial_coords, GridGeometry, load_survey_points
from .migration import migrate_eikonal_points, EikonalMigrator, MigrationResult
from .variogram import BSSOptimizer, OptimizationResult
from .interpolation import blend_margin_topography, kriging_interpolation, random_forest_hole_filling, KrigingEngine, BedrockFinalizer, KrigingResult
from .smoothing import compute_gradients, fft_gaussian_smooth
from .logging import logger


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
        pre_kriging_method: str = "universal",
        pre_drift_terms: Optional[List[str]] = None,
        pre_variogram_model: str = "spherical",
        pre_zero_boundary: bool = True,
        post_kriging_method: str = "universal",
        post_drift_terms: Optional[List[str]] = None,
        post_variogram_model: str = "spherical",
        post_zero_boundary: bool = True,
        perform_migration: bool = True,
        survey_data_type: str = "one_way_travel_time",
        plots_dir: Optional[str] = None,
        n_cores: int = -1,
        built_in_kriging: bool = True,
        nrbins: Optional[int] = None,
        ice_density: float = 900.0,
        g: float = 9.81,
        base_dir: Optional[str] = None,
        survey_data_path: Optional[str] = None,
        show_progress: bool = True,
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
        pre_kriging_method : str
            1st-pass pre-migration Kriging approach ('universal', 'ordinary', 'regression'). Default 'universal'.
        pre_drift_terms : list of str, optional
            1st-pass drift terms (e.g. ['sia_thickness'], ['quadratic'], ['regional_linear']). Default ['sia_thickness'].
        pre_variogram_model : str
            1st-pass variogram model ('spherical', 'exponential', 'gaussian', 'linear'). Default 'spherical'.
        pre_zero_boundary : bool
            If True, enforces a zero traveltime boundary condition (T=0) on the glacier margin outline.
        post_kriging_method : str
            2nd-pass post-migration Kriging approach ('universal', 'ordinary', 'regression'). Default 'universal'.
        post_drift_terms : list of str, optional
            2nd-pass drift terms (e.g. ['sia_thickness'], ['quadratic'], ['regional_linear']). Default ['sia_thickness'].
        post_variogram_model : str
            2nd-pass variogram model ('spherical', 'exponential', 'gaussian', 'linear'). Default 'spherical'.
        post_zero_boundary : bool
            If True, enforces a zero thickness boundary condition (H=0) on the glacier margin outline.
        perform_migration : bool
            If True (default), performs 3D Eikonal ray migration on travel times. If False, skips migration.
        survey_data_type : str
            Type of input survey data defined in inputs section: 'one_way_travel_time' (default),
            'two_way_travel_time' (converts TWT/2), or 'thickness' / 'ice_thickness' (direct depth/thickness measurements, skips migration).
        plots_dir : str, optional
            Directory where generated plots are automatically saved.
        n_cores : int
            Number of CPU cores for multi-threading/processing (-1 for all available cores).
        base_dir : str, optional
            General workspace directory for all output files. If None, defaults to parent directory of survey_data_path.
        survey_data_path : str, optional
            Path to survey data file (used for base_dir resolution fallback if base_dir is None).
        show_progress : bool
            If True (default), displays terminal progress bars during heavy processing steps.
        """
        self.dem_grid, self.meta = load_dem(dem, dx=dx, dy=dy, bounds=bounds)
        self.outline_mask = load_outline(outline, self.dem_grid, self.meta)

        self.geometry = GridGeometry.create(
            self.dem_grid.shape,
            dx=float(self.meta.get("dx", 1.0)),
            dy=float(self.meta.get("dy", 1.0)),
            bounds=self.meta.get("bounds"),
        )
        self.dx = self.geometry.dx
        self.dy = self.geometry.dy
        self.x_coords = self.geometry.x_coords
        self.y_coords = self.geometry.y_coords
        self.bounds = self.geometry.bounds

        # Migration & Kriging options (Pre-migration & Post-migration)
        self.pre_kriging_method = pre_kriging_method
        self.pre_drift_terms = pre_drift_terms if pre_drift_terms is not None else ["sia_thickness"]
        self.pre_variogram_model = pre_variogram_model
        self.pre_zero_boundary = bool(pre_zero_boundary)

        self.post_kriging_method = post_kriging_method
        self.post_drift_terms = post_drift_terms if post_drift_terms is not None else ["sia_thickness"]
        self.post_variogram_model = post_variogram_model
        self.post_zero_boundary = bool(post_zero_boundary)

        self.perform_migration = perform_migration
        self.survey_data_type = survey_data_type
        self.base_dir = base_dir
        self.survey_data_path = survey_data_path
        self._raw_plots_dir = plots_dir
        self.n_cores = n_cores
        self.built_in_kriging = bool(built_in_kriging)
        self.nrbins = int(nrbins) if nrbins is not None else None
        self.ice_density = float(ice_density)
        self.g = float(g)
        self.show_progress = bool(show_progress)

        # Gradient and Smoothed Slope Caches
        self._gradient_cache: Optional[Dict[str, np.ndarray]] = None
        self._smoothed_slopes_cache: Dict[float, np.ndarray] = {}

        # Decoupled sub-engines strictly bound to GridGeometry
        self.migrator = EikonalMigrator(self.dem_grid, geometry=self.geometry, outline_mask=self.outline_mask)
        self.bss_optimizer = BSSOptimizer(self.dem_grid, geometry=self.geometry)
        self.kriging_engine = KrigingEngine(self.dem_grid, geometry=self.geometry, outline_mask=self.outline_mask)
        self.finalizer = BedrockFinalizer(self.dem_grid, geometry=self.geometry, outline_mask=self.outline_mask)

        # Internal state
        self.survey_points: Optional[np.ndarray] = None
        self.migrated_points: Optional[np.ndarray] = None
        self.opt_kc: Optional[float] = None
        self.opt_slope: Optional[np.ndarray] = None
        self.kriged_thickness: Optional[np.ndarray] = None
        self.kriged_bedrock: Optional[np.ndarray] = None
        self.kriged_variance: Optional[np.ndarray] = None
        self.rf_filled_bedrock: Optional[np.ndarray] = None
        self.blended_bedrock: Optional[np.ndarray] = None
        self.final_thickness: Optional[np.ndarray] = None
        self.kriged_std: Optional[np.ndarray] = None
        self.final_bss: Optional[np.ndarray] = None
        self.bss_std: Optional[np.ndarray] = None
        self.final_grid: Optional[np.ndarray] = None
        self.config: Dict[str, Any] = {}
        self.config_path: Optional[str] = None

    @property
    def effective_base_dir(self) -> str:
        """Returns the effective base directory for output file resolution."""
        if self.base_dir:
            return os.path.expanduser(self.base_dir)
        if self.survey_data_path and isinstance(self.survey_data_path, (str, os.PathLike)):
            return os.path.dirname(os.path.expanduser(str(self.survey_data_path)))
        return ""

    def resolve_path(self, path: Optional[str]) -> Optional[str]:
        """Resolves a target file or directory path relative to effective_base_dir if relative."""
        if not path:
            return path
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            return expanded
        eff_base = self.effective_base_dir
        if eff_base:
            return os.path.join(eff_base, expanded)
        return expanded

    @property
    def plots_dir(self) -> str:
        """Directory where generated plots are automatically saved."""
        target = self._raw_plots_dir if self._raw_plots_dir is not None else "figures"
        return self.resolve_path(target)

    @plots_dir.setter
    def plots_dir(self, value: Optional[str]) -> None:
        self._raw_plots_dir = value

    @property
    def plot_extent(self) -> List[float]:
        """Returns Matplotlib plot extent [minx, maxx, miny, maxy]."""
        return self.geometry.extent

    def _get_dem_gradients(self) -> Dict[str, np.ndarray]:
        if self._gradient_cache is None:
            self._gradient_cache = compute_gradients(self.dem_grid, dx=self.dx, dy=self.dy)
        return self._gradient_cache

    def get_smoothed_slope(self, kc: float) -> np.ndarray:
        kc_key = round(float(kc), 6)
        if kc_key not in self._smoothed_slopes_cache:
            grads = self._get_dem_gradients()
            base_slope = grads["slope_rad"]
            smoothed_slope, _, _ = fft_gaussian_smooth(base_slope, dx=self.dx, dy=self.dy, kc=kc)
            self._smoothed_slopes_cache[kc_key] = smoothed_slope
        return self._smoothed_slopes_cache[kc_key]

    @classmethod
    def from_config(cls, config_path: Union[str, os.PathLike] = "pysole.json") -> "Solver":
        """Initializes Solver instance using inputs defined in pysole.json configuration file."""
        from .config import load_config
        cfg = load_config(config_path)
        inputs = cfg.get("inputs", {})
        spatial = cfg.get("spatial_parameters", {})
        migration_cfg = cfg.get("migration_parameters", {})
        kriging_cfg = cfg.get("kriging_parameters", {})
        outputs = cfg.get("outputs", {})

        survey_dtype = inputs.get("survey_data_type", "one_way_travel_time")

        pre_krig_cfg = kriging_cfg.get("pre_migration", {}) if isinstance(kriging_cfg.get("pre_migration"), dict) else {}
        post_krig_cfg = kriging_cfg.get("post_migration", {}) if isinstance(kriging_cfg.get("post_migration"), dict) else {}

        pre_method = pre_krig_cfg.get("method", "universal")
        pre_drifts = pre_krig_cfg.get("drift_terms", ["sia_thickness"])
        pre_var_model = pre_krig_cfg.get("variogram_model", "spherical")
        pre_zero_boundary = pre_krig_cfg.get("include_zero_boundary_condition", True)

        post_method = post_krig_cfg.get("method", "universal")
        post_drifts = post_krig_cfg.get("drift_terms", ["sia_thickness"])
        post_var_model = post_krig_cfg.get("variogram_model", "spherical")
        post_zero_boundary = post_krig_cfg.get("include_zero_boundary_condition", True)

        opt_cfg = cfg.get("optimization_parameters", {})
        n_cores = inputs.get("n_cores", -1)
        nrbins = opt_cfg.get("nrbins", None)
        built_in_kriging = kriging_cfg.get("built_in_kriging", True)
        ice_density = inputs.get("ice_density", 900.0)
        g_val = inputs.get("g", 9.81)

        show_progress_val = inputs.get("show_progress", True)

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
            n_cores=n_cores,
            built_in_kriging=built_in_kriging,
            nrbins=nrbins,
            ice_density=ice_density,
            g=g_val,
            base_dir=inputs.get("base_dir"),
            survey_data_path=inputs.get("survey_data_path"),
            show_progress=show_progress_val,
        )
        solver.config = cfg
        solver.config_path = str(config_path)
        return solver

    def migrate_eikonal(
        self,
        travel_times: Union[str, np.ndarray],
        velocity: Optional[float] = None,
        interactive: bool = False,
        plotit: bool = False,
    ) -> np.ndarray:
        """Migrates zero-offset GPR or seismic travel times into 3D space using the Eikonal equation."""
        if not self.survey_data_path and isinstance(travel_times, (str, os.PathLike)):
            self.survey_data_path = str(travel_times)

        pts = load_survey_points(
            travel_times,
            bounds=self.bounds,
            dem_grid=self.dem_grid,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
        )

        dtype_str = str(self.survey_data_type).lower().strip()

        if dtype_str in ["two_way_travel_time", "twt", "two_way"]:
            logger.info("   [Survey Data Type: TWT] Two-Way Traveltimes detected. Converting to One-Way Traveltimes (OWTT = TWT / 2.0).")
            pts[:, 3] = pts[:, 3] / 2.0
        elif dtype_str in ["one_way_travel_time", "owtt", "one_way"]:
            logger.info("   [Survey Data Type: OWTT] One-Way Traveltimes detected.")

        self.survey_points = pts

        if dtype_str in ["thickness", "ice_thickness", "depth"]:
            logger.info("   [Survey Data Type: Ice Thickness] Input data represents direct ice thickness measurements. Skipping 3D Eikonal ray migration.")
            self.migrated_points = pts.copy()
            return self.migrated_points

        if not self.perform_migration:
            logger.info("   [Migration Skipped] 'perform_migration' is set to False in configuration. Using unmigrated survey points directly.")
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

        # 1st Pass BSS surface slope optimization (Pre-migration traveltimes)
        logger.info("   [Pass 1: Pre-Migration] Evaluating BSS Surface Slope Optimization for Traveltime Field T(x,y)...")
        self.optimize_bss(prefix="01_", interactive=interactive)
        logger.info(f"   [Pass 1: Pre-Migration] Optimal Traveltime Corner Frequency k_c = {self.opt_kc:.4f} rad/m")

        opt_slope_sin1 = np.sin(self.opt_slope)
        interp_slope1 = self.geometry.create_interpolator(opt_slope_sin1, fill_value=np.nan)

        pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))  # (Y, X)
        slopes_pts1 = interp_slope1(pts_xy)
        slopes_pts1 = np.maximum(np.nan_to_num(slopes_pts1, nan=0.1), 1e-4)

        tt_pts = pts[:, 3]
        product_pts1 = tt_pts * slopes_pts1

        sample_prod_pts1 = np.column_stack((pts[:, 0], pts[:, 1], product_pts1))

        logger.info(f"   [Pass 1: Pre-Migration] Performing {self.pre_kriging_method.capitalize()} Kriging Interpolation for Traveltimes...")
        krig1_res = kriging_interpolation(
            sample_points=sample_prod_pts1,
            geometry=self.geometry,
            method=self.pre_kriging_method,
            variogram_model=self.pre_variogram_model,
            dem_grid=self.dem_grid,
            opt_slope_grid=self.opt_slope,
            drift_terms=self.pre_drift_terms,
            outline_mask=self.outline_mask,
            include_zero_boundary_condition=self.pre_zero_boundary,
            n_cores=self.n_cores,
            built_in_kriging=self.built_in_kriging,
            show_progress=self.show_progress,
        )
        prod_grid1 = krig1_res.bedrock_grid

        slope_floor_deg = float(self.config.get("optimization_parameters", {}).get("slope_floor_deg", 5.0))
        min_slope_sin1 = np.sin(np.radians(slope_floor_deg))
        safe_slope_grid1 = np.maximum(opt_slope_sin1, min_slope_sin1)
        tt_grid = np.maximum(prod_grid1 / safe_slope_grid1, 0.0)

        cached_dem_grads = self._get_dem_gradients()

        logger.info(f"   [Ray Migration] Executing 3D Eikonal Ray Displacement (velocity v = {velocity:.4f} m/ns)...")
        mig_res = migrate_eikonal_points(
            dem=self.dem_grid,
            travel_time_grid=tt_grid,
            survey_points=pts,
            geometry=self.geometry,
            velocity=velocity,
            outline_mask=self.outline_mask,
            plots_dir=self.plots_dir,
            interactive=interactive,
            dem_grads=cached_dem_grads,
            show_progress=self.show_progress,
        )
        self.migrated_points = mig_res.migrated_points

        if interactive:
            while True:
                try:
                    ans = input(f"\nCurrent migration velocity v = {velocity:.4f} m/ns. Test another migration velocity? [y/N]: ").strip().lower()
                    if ans in ["y", "yes"]:
                        val = input("Enter new signal propagation velocity [m/ns] (e.g. 0.15): ").strip()
                        if val:
                            velocity = float(val)
                            logger.info(f"Re-running 3D Eikonal ray migration with v = {velocity:.4f} m/ns...")
                            mig_res2 = migrate_eikonal_points(
                                dem=self.dem_grid,
                                travel_time_grid=tt_grid,
                                survey_points=pts,
                                geometry=self.geometry,
                                velocity=velocity,
                                outline_mask=self.outline_mask,
                                plots_dir=self.plots_dir,
                                interactive=interactive,
                                dem_grads=cached_dem_grads,
                                show_progress=self.show_progress,
                            )
                            self.migrated_points = mig_res2.migrated_points
                    else:
                        break
                except Exception:
                    break

        return self.migrated_points

    def optimize_bss(
        self,
        kc_max: Optional[float] = None,
        kc_min: Optional[float] = None,
        d_kc: Optional[float] = None,
        nrbins: Optional[int] = None,
        prefix: Optional[str] = None,
        interactive: bool = False,
        plotit: bool = False,
    ) -> float:
        """
        Iterative optimization process to determine optimum surface slope smoothing degree kc.
        When interactive is True, enables CLI prompts to adjust kc_min, kc_max, d_kc, nrbins, and a_range.
        """
        pts = self.migrated_points if self.migrated_points is not None else self.survey_points
        if pts is None:
            xx, yy = np.meshgrid(self.x_coords[::5], self.y_coords[::5])
            pts = np.column_stack((xx.ravel(), yy.ravel(), self.dem_grid[::5, ::5].ravel(), np.ones(xx.size) * 10.0))

        if hasattr(self, "config") and isinstance(self.config, dict):
            opt_cfg = self.config.get("optimization_parameters", {})
            if kc_max is None and "kc_max" in opt_cfg:
                kc_max = opt_cfg["kc_max"]
            if kc_min is None and "kc_min" in opt_cfg:
                kc_min = opt_cfg["kc_min"]
            if d_kc is None and "d_kc" in opt_cfg:
                d_kc = opt_cfg["d_kc"]
            if nrbins is None and "nrbins" in opt_cfg:
                nrbins = opt_cfg["nrbins"]

        if nrbins is None:
            nrbins = self.nrbins

        if prefix is None:
            prefix = "03_" if self.migrated_points is not None else "01_"

        stage_name = "stage2" if prefix == "03_" else "stage1"

        opt_res = self.bss_optimizer.optimize(
            survey_points=pts,
            kc_max=kc_max,
            kc_min=kc_min,
            d_kc=d_kc,
            plots_dir=self.plots_dir,
            prefix=prefix,
            stage_name=stage_name,
            interactive=interactive,
            n_cores=self.n_cores,
            nrbins=nrbins,
            show_progress=self.show_progress,
        )

        self.opt_kc = opt_res.optimal_kc
        self.opt_slope = opt_res.optimal_slope_grid
        self._smoothed_slopes_cache.update(opt_res.all_smoothed_slopes)
        return self.opt_kc

    def interpolate_kriging(
        self,
        method: Optional[str] = None,
        variogram_model: Optional[str] = None,
        interactive: bool = False,
        plotit: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pts = self.migrated_points if self.migrated_points is not None else self.survey_points
        if pts is None:
            raise ValueError("No survey or migrated points available. Run migrate_eikonal() first.")

        should_plot = plotit or interactive or (self.plots_dir is not None)

        if self.opt_slope is None:
            logger.info("   [Pass 2: Post-Migration] Evaluating BSS Surface Slope Optimization for Depth Field D(x,y)...")
            self.optimize_bss(interactive=interactive)

        opt_slope_sin = np.sin(self.opt_slope)
        interp_slope = self.geometry.create_interpolator(opt_slope_sin, fill_value=np.nan)

        pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))  # (Y, X)
        slopes_pts = interp_slope(pts_xy)
        slopes_pts = np.maximum(np.nan_to_num(slopes_pts, nan=0.1), 1e-4)

        thickness_pts = pts[:, 3]
        product_pts = thickness_pts * slopes_pts

        sample_prod_pts = np.column_stack((pts[:, 0], pts[:, 1], product_pts))

        krig_method = method if method is not None else self.post_kriging_method
        var_model = variogram_model if variogram_model is not None else self.post_variogram_model

        logger.info(f"   [Pass 2: Post-Migration] Performing {krig_method.capitalize()} Kriging Interpolation for Depth Field D(x,y)...")
        krig_res = kriging_interpolation(
            sample_points=sample_prod_pts,
            geometry=self.geometry,
            method=krig_method,
            variogram_model=var_model,
            dem_grid=self.dem_grid,
            opt_slope_grid=self.opt_slope,
            drift_terms=self.post_drift_terms,
            outline_mask=self.outline_mask,
            include_zero_boundary_condition=self.post_zero_boundary,
            n_cores=self.n_cores,
            built_in_kriging=self.built_in_kriging,
            show_progress=self.show_progress,
        )
        prod_grid = krig_res.bedrock_grid
        prod_var = krig_res.variance_grid

        slope_floor_deg = float(self.config.get("optimization_parameters", {}).get("slope_floor_deg", 5.0))
        min_slope_sin = np.sin(np.radians(slope_floor_deg))
        safe_slope_grid = np.maximum(opt_slope_sin, min_slope_sin)

        max_thickness = max(float(np.max(thickness_pts)) * 1.5, 500.0)
        thickness_grid = np.clip(prod_grid / safe_slope_grid, 0.0, max_thickness)

        if self.outline_mask is not None:
            thickness_grid[~self.outline_mask] = 0.0

        self.kriged_thickness = thickness_grid.copy()
        self.kriged_bedrock = self.dem_grid - thickness_grid
        self.kriged_variance = prod_var / (safe_slope_grid**2)
        if self.outline_mask is not None:
            self.kriged_variance[~self.outline_mask] = np.nan

        self.kriged_std = np.sqrt(np.maximum(np.nan_to_num(self.kriged_variance, nan=0.0), 0.0))
        if self.outline_mask is not None:
            self.kriged_std[~self.outline_mask] = np.nan

        plot_extent = self.plot_extent

        if should_plot:
            from .plotting import plot_kriging_bedrock_and_uncertainty
            plot_kriging_bedrock_and_uncertainty(
                kriged_bedrock=self.kriged_bedrock,
                kriged_std=self.kriged_std,
                pts=pts,
                plot_extent=plot_extent,
                plots_dir=self.plots_dir,
                interactive=interactive,
            )

        return self.kriged_bedrock, self.kriged_variance

    def fill_holes_rf(self) -> np.ndarray:
        """Trains Random Forest ML model to fill remaining bedrock holes."""
        if self.kriged_bedrock is None:
            self.interpolate_kriging()

        self.rf_filled_bedrock = self.finalizer.fill_holes(self.kriged_bedrock, n_cores=self.n_cores, show_progress=self.show_progress)
        return self.rf_filled_bedrock

    def apply_geomorph_smoothing(self, min_gap_dist: Optional[float] = None) -> np.ndarray:
        """Applies geomorphological margin blending to surrounding surface DEM."""
        base_grid = self.rf_filled_bedrock if self.rf_filled_bedrock is not None else self.kriged_bedrock
        if base_grid is None:
            base_grid = self.fill_holes_rf()

        self.blended_bedrock = self.finalizer.blend_margin(base_grid, min_gap_dist=min_gap_dist)
        return self.blended_bedrock

    def smooth_bedrock_dem(
        self,
        grid: np.ndarray,
        method: str = "gaussian",
        sigma: float = 1.5,
        kernel_size: int = 3,
        kc_cutoff: Optional[float] = None,
    ) -> np.ndarray:
        """Applies spatial smoothing to calculated bedrock DEM grid ('gaussian', 'median', 'fft_lowpass')."""
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
            kc = kc_cutoff if kc_cutoff is not None else 1.0

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
        Executes full final sequence towards continuous bedrock topography result.

        Parameters
        ----------
        smoothing_sigma : float
            Smoothing strength (radius in pixels) for Gaussian filtering (default 1.5).
            Higher values produce smoother bedrock terrain.
        smoothing_kernel_size : int
            Window kernel size (k x k) for median filtering (must be an odd integer, default 3).
            Higher values produce smoother bedrock terrain.
        smoothing_kc_cutoff : float, optional
            Corner frequency cutoff wavenumber (k_c,smooth) for FFT low-pass filtering.
            If None, defaults to optimal k_c. Lower values produce smoother bedrock terrain.
        """
        should_plot = plotit or interactive or (self.plots_dir is not None)

        if self.kriged_thickness is None:
            if self.kriged_bedrock is not None:
                self.kriged_thickness = np.maximum(self.dem_grid - self.kriged_bedrock, 0.0)
                if self.outline_mask is not None:
                    self.kriged_thickness[~self.outline_mask] = 0.0
            else:
                self.interpolate_kriging(interactive=interactive, plotit=should_plot)

        # Step 4 Topography Finalization: Takes depth field D(x,y) from Step 3
        if smooth_bedrock:
            logger.info(f"   Applying depth field spatial smoothing ({smoothing_method})...")
            thick_raw = self.kriged_thickness.copy()
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
        else:
            self.kriged_bedrock = self.dem_grid - self.kriged_thickness
            if self.outline_mask is not None:
                self.kriged_bedrock[~self.outline_mask] = self.dem_grid[~self.outline_mask]

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

        if should_plot:
            from .plotting import plot_calculated_bedrock_map
            plot_calculated_bedrock_map(
                bedrock_grid=current_bedrock,
                outline_mask=self.outline_mask,
                plot_extent=plot_extent,
                smooth_bedrock=smooth_bedrock,
                smoothing_sigma=smoothing_sigma,
                plots_dir=self.plots_dir,
                interactive=interactive,
            )

        if apply_margin_blend is None:
            if interactive:
                try:
                    ans_blend = input("\nDo you want to apply margin blending to surrounding terrain (blend_margin_topography)? [Y/n]: ").strip().lower()
                    apply_margin_blend = ans_blend not in ["n", "no"]
                except Exception:
                    apply_margin_blend = True
            else:
                apply_margin_blend = False

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

            logger.info(f"   Applying geomorphological margin blending (min_gap_dist = {min_gap_dist:.1f} m)...")
            self.blended_bedrock = blend_margin_topography(
                dem=self.dem_grid,
                bedrock_input=current_bedrock,
                boundary_mask=self.outline_mask,
                geometry=self.geometry,
                min_gap_dist=min_gap_dist,
            )
        else:
            self.blended_bedrock = current_bedrock.copy()

        if should_plot and apply_margin_blend:
            from .plotting import plot_final_blended_bedrock_map
            plot_final_blended_bedrock_map(
                blended_bedrock=self.blended_bedrock,
                plot_extent=plot_extent,
                plots_dir=self.plots_dir,
                interactive=interactive,
            )

        self.final_thickness = np.maximum(self.dem_grid - self.blended_bedrock, 0.0)
        if self.outline_mask is not None:
            self.final_thickness[~self.outline_mask] = 0.0

        if self.kriged_variance is not None:
            self.kriged_std = np.sqrt(np.maximum(np.nan_to_num(self.kriged_variance, nan=0.0), 0.0))
            if self.outline_mask is not None:
                self.kriged_std[~self.outline_mask] = np.nan
        else:
            self.kriged_std = np.full_like(self.final_thickness, np.nan)
            if self.outline_mask is not None:
                self.kriged_std[~self.outline_mask] = np.nan

        if self.outline_mask is not None and np.any(self.outline_mask):
            mean_thick = float(np.nanmean(self.final_thickness[self.outline_mask]))
            mean_unc = float(np.nanmean(self.kriged_std[self.outline_mask]))
        else:
            mean_thick = float(np.nanmean(self.final_thickness))
            mean_unc = float(np.nanmean(self.kriged_std))

        # Basal Shear Stress Calculation (tb = rho_ice * g * D * sin(alpha) in kPa)
        if self.opt_slope is not None and self.opt_slope.shape == self.dem_grid.shape:
            sin_alpha_opt = np.sin(self.opt_slope)
        else:
            sin_alpha_opt = np.sin(self._get_dem_gradients()["slope_rad"])

        self.final_bss = (self.ice_density * self.g * self.final_thickness * sin_alpha_opt) / 1000.0
        self.bss_std = (self.ice_density * self.g * np.nan_to_num(self.kriged_std, nan=0.0) * sin_alpha_opt) / 1000.0

        if self.outline_mask is not None:
            self.final_bss[~self.outline_mask] = np.nan
            self.bss_std[~self.outline_mask] = np.nan

        if self.outline_mask is not None and np.any(self.outline_mask):
            mean_bss = float(np.nanmean(self.final_bss[self.outline_mask]))
            mean_bss_unc = float(np.nanmean(self.bss_std[self.outline_mask]))
        else:
            mean_bss = float(np.nanmean(self.final_bss))
            mean_bss_unc = float(np.nanmean(self.bss_std))

        if should_plot:
            from .plotting import (
                plot_final_ice_thickness_and_uncertainty,
                plot_final_ice_thickness_histogram,
                plot_final_basal_shear_stress_and_uncertainty,
                plot_final_basal_shear_stress_histogram,
            )
            plot_final_ice_thickness_and_uncertainty(
                final_thickness=self.final_thickness,
                kriged_std=self.kriged_std,
                outline_mask=self.outline_mask,
                plot_extent=plot_extent,
                mean_thick=mean_thick,
                mean_unc=mean_unc,
                plots_dir=self.plots_dir,
                interactive=interactive,
            )
            plot_final_ice_thickness_histogram(
                final_thickness=self.final_thickness,
                outline_mask=self.outline_mask,
                mean_thick=mean_thick,
                plots_dir=self.plots_dir,
                interactive=interactive,
            )
            plot_final_basal_shear_stress_and_uncertainty(
                final_bss=self.final_bss,
                bss_std=self.bss_std,
                outline_mask=self.outline_mask,
                plot_extent=plot_extent,
                mean_bss=mean_bss,
                mean_bss_unc=mean_bss_unc,
                plots_dir=self.plots_dir,
                interactive=interactive,
            )
            plot_final_basal_shear_stress_histogram(
                final_bss=self.final_bss,
                outline_mask=self.outline_mask,
                mean_bss=mean_bss,
                plots_dir=self.plots_dir,
                interactive=interactive,
            )

        self.final_grid = self.blended_bedrock
        return BedrockMap(
            grid=self.final_grid,
            bounds=self.bounds,
            crs=self.meta.get("crs"),
            transform=self.meta.get("transform"),
            name="final_bedrock",
        )

    def run_pipeline(self) -> BedrockMap:
        """
        Executes the complete 5-step PySole workflow using loaded configuration settings.

        Returns
        -------
        bedrock_map : BedrockMap
            Final predicted bedrock elevation grid.
        """
        cfg = getattr(self, "config", {})
        inputs = cfg.get("inputs", {})
        migration = cfg.get("migration_parameters", {})
        opt = cfg.get("optimization_parameters", {})
        fin_cfg = cfg.get("finalization_parameters", {})
        outputs = cfg.get("outputs", {})

        survey_data_path = inputs.get("survey_data_path")
        if not survey_data_path:
            raise ValueError("Configuration 'inputs.survey_data_path' must be specified.")

        cfg_name = getattr(self, "config_path", "pysole.json")
        logger.info("================================================================================")
        logger.info(f"       STARTING NEW PYSOLE BEDROCK TOPOGRAPHY CALCULATION ({cfg_name})")
        logger.info("================================================================================")
        logger.info(f"1. Initializing PySole Solver from '{cfg_name}'...")
        logger.info(f"   DEM Resolution: dx = {self.dx} m, dy = {self.dy} m")
        logger.info(f"   Survey Data Type: {self.survey_data_type}")
        logger.info(f"   Pre-Migration Kriging: {self.pre_kriging_method} (drift = {self.pre_drift_terms}, variogram = {self.pre_variogram_model})")
        logger.info(f"   Post-Migration Kriging: {self.post_kriging_method} (drift = {self.post_drift_terms}, variogram = {self.post_variogram_model})")
        logger.info(f"   Perform Migration: {self.perform_migration}")
        logger.info(f"   Interactive Migration: {migration.get('interactive_migration', False)}")
        logger.info(f"   Plots Output Directory: {self.plots_dir}")

        if self.perform_migration and str(self.survey_data_type).lower().strip() not in ["thickness", "ice_thickness", "depth"]:
            logger.info("2. Performing 3D Eikonal Ray Migration...")
        else:
            logger.info("2. Skipping 3D Eikonal Ray Migration...")

        self.migrate_eikonal(
            travel_times=survey_data_path,
            velocity=migration.get("velocity"),
            interactive=migration.get("interactive_migration", False),
            plotit=True,
        )

        logger.info("3. Performing Post-Migration BSS Slope Optimization & Depth Interpolation (Field D(x,y))...")
        opt_kc = self.optimize_bss(
            interactive=opt.get("interactive_optimization", False),
            plotit=True,
        )
        logger.info(f"   Optimal Post-Migration Corner Frequency k_c = {opt_kc:.4f} rad/m")
        self.interpolate_kriging(interactive=opt.get("interactive_optimization", False), plotit=True)

        logger.info("4. Finalizing Bedrock Topography...")
        bedrock_map = self.finalize_topography(
            interactive=opt.get("interactive_optimization", False),
            plotit=True,
            random_forest_gap_filling=fin_cfg.get("random_forest_gap_filling", False),
            apply_margin_blend=fin_cfg.get("apply_margin_blend", False),
            min_gap_dist=fin_cfg.get("min_gap_dist", 50.0),
            smooth_bedrock=fin_cfg.get("smooth_bedrock", False),
            smoothing_method=fin_cfg.get("smoothing_method", "gaussian"),
            smoothing_sigma=fin_cfg.get("smoothing_sigma", 1.5),
            smoothing_kernel_size=fin_cfg.get("smoothing_kernel_size", 3),
            smoothing_kc_cutoff=fin_cfg.get("smoothing_kc_cutoff", None),
        )

        output_name = outputs.get("output_name")
        output_format = outputs.get("output_format")
        if output_name:
            stem, _ = os.path.splitext(output_name)
            resolved_output_stem = self.resolve_path(stem if stem else output_name)
            saved_res = bedrock_map.save(resolved_output_stem, formats=output_format)
            if isinstance(saved_res, list):
                for sf in saved_res:
                    logger.info(f"5. Saved predicted bedrock map to: {sf}")
            else:
                logger.info(f"5. Saved predicted bedrock map to: {saved_res}")

        return bedrock_map
