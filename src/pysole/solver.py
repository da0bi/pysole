"""
Main Solver API for PySole package.
Implements the workflow for physically-informed bedrock interpolation & 3D migration,
caching surface gradients and smoothed slope grids for maximum computational efficiency.
"""

from pathlib import Path
from typing import Any
import numpy as np
import os
from .raster import BedrockMap, load_dem, load_outline, GridGeometry, load_survey_points, save_points_csv
from .migration import migrate_eikonal_points, EikonalMigrator, MigrationResult
from .variogram import BSSOptimizer, OptimizationResult, compute_cutoff_wavelength
from .interpolation import blend_margin_topography, kriging_interpolation, KrigingEngine, BedrockFinalizer, KrigingResult
from .smoothing import compute_gradients, fft_gaussian_smooth, precompute_fft_grid, fft_gaussian_smooth_precomputed
from .config import OutputsConfig, resolve_path as resolve_config_path, resolve_input_path as resolve_input_config_path, resolve_output_dir
from .logging import logger
from .drift_analyzer import DriftAnalyzer
from .survey_planner import SurveyPlanner
import sys


class Solver:
    """
    Physically-Informed Bedrock Interpolation & 3D Migration Solver.
    """

    def __init__(
        self,
        dem: str | Path | np.ndarray,
        outline: str | Path | np.ndarray | None = None,
        dx: float | None = None,
        dy: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        origin: tuple[float, float] | None = None,
        crs: Any = None,
        pre_kriging_method: str = "universal",
        pre_drift_terms: list[str] | None = None,
        pre_variogram_model: str = "spherical",
        pre_zero_boundary: bool = True,
        pre_interpolation_target: str = "P",
        post_kriging_method: str = "universal",
        post_drift_terms: list[str] | None = None,
        post_variogram_model: str = "spherical",
        post_zero_boundary: bool = True,
        post_interpolation_target: str = "P",
        perform_migration: bool = True,
        survey_data_type: str = "one_way_traveltime",
        plots_dir: str | Path | None = None,
        n_cores: int = -1,
        kriging_engine: str = "native",
        nrbins: int | None = None,
        ice_density: float = 900.0,
        g: float = 9.81,
        output_dir: str | Path | None = None,
        survey_data_path: str | Path | None = None,
        survey_profile_column: str | None = None,
        config_path: str | Path | None = None,
        show_progress: bool = True,
    ):
        """
        Parameters
        ----------
        dem : str, Path, or np.ndarray
            Path to surface DEM GeoTIFF/ASCII file, or 2D numpy array.
        outline : str, Path, or np.ndarray, optional
            Path to creeping body boundary polygon (Shapefile/GeoJSON) or boolean mask.
        dx : float, optional
            Pixel resolution along X. If None, derived directly from actual DEM metadata.
        dy : float, optional
            Pixel resolution along Y. If None, derived directly from actual DEM metadata.
        bounds : tuple of float, optional
            Spatial bounding box (minx, miny, maxx, maxy).
        origin : tuple of float, optional
            Lower-left coordinate origin (xll, yll) for headerless DEM formats (CSV, NPY, ndarray).
        crs : str or int, optional
            Coordinate Reference System (e.g. 'EPSG:32633'). Enforces projected metric system checks.
        pre_kriging_method : str
            1st-pass pre-migration Kriging approach ('universal', 'ordinary', 'regression'). Default 'universal'.
        pre_drift_terms : list of str, optional
            1st-pass drift terms (e.g. ['sia'], ['quadratic_xy'], ['linear_xy']). Default ['sia'].
        pre_variogram_model : str
            1st-pass variogram model ('spherical', 'exponential', 'gaussian', 'linear'). Default 'spherical'.
        pre_zero_boundary : bool
            If True, enforces a zero traveltime boundary condition (T=0) on the glacier margin outline.
        post_kriging_method : str
            2nd-pass post-migration Kriging approach ('universal', 'ordinary', 'regression'). Default 'universal'.
        post_drift_terms : list of str, optional
            2nd-pass drift terms (e.g. ['sia'], ['quadratic_xy'], ['linear_xy']). Default ['sia'].
        post_variogram_model : str
            2nd-pass variogram model ('spherical', 'exponential', 'gaussian', 'linear'). Default 'spherical'.
        post_zero_boundary : bool
            If True, enforces a zero thickness boundary condition (H=0) on the glacier margin outline.
        perform_migration : bool
            If True (default), performs 3D Eikonal ray migration on travel times. If False, skips migration.
        survey_data_type : str
            Type of input survey data defined in inputs section: 'one_way_traveltime' (default),
            'two_way_traveltime' (converts TWT/2), or 'depth' (direct depth/thickness measurements, skips migration).
        plots_dir : str or Path, optional
            Directory where generated plots are automatically saved.
        n_cores : int
            Number of CPU cores for multi-threading/processing (-1 for all available cores).
        kriging_engine : str
            Kriging solver engine: 'native' (default, high-performance solver) or 'pykrige'.
        output_dir : str or Path, optional
            General workspace directory for all output files. If None, defaults to parent directory of survey_data_path.
        survey_profile_column : str, optional
            Column name identifying individual survey profiles in survey CSV data for LOPO spatial CV.
        config_path : str or Path, optional
            Path to configuration file used for resolving relative paths.
        show_progress : bool
            If True (default), displays terminal progress bars during heavy processing steps.
        """
        self.output_dir = output_dir
        self.survey_data_path = survey_data_path
        self.survey_profile_column = survey_profile_column
        self.config_path = str(config_path) if config_path else None
        self._raw_plots_dir = plots_dir

        self.dem_grid, self.meta = load_dem(
            self.resolve_input_path(dem),
            dx=dx,
            dy=dy,
            bounds=bounds,
            origin=origin,
            crs=crs,
        )
        self.outline_mask = load_outline(self.resolve_input_path(outline), self.dem_grid, self.meta)

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
        self.pre_drift_terms = pre_drift_terms if pre_drift_terms is not None else ["sia"]
        self.pre_variogram_model = pre_variogram_model
        self.pre_zero_boundary = bool(pre_zero_boundary)
        self.pre_interpolation_target = str(pre_interpolation_target).upper().strip()

        self.post_kriging_method = post_kriging_method
        self.post_drift_terms = post_drift_terms if post_drift_terms is not None else ["sia"]
        self.post_variogram_model = post_variogram_model
        self.post_zero_boundary = bool(post_zero_boundary)
        self.post_interpolation_target = str(post_interpolation_target).upper().strip()

        self.perform_migration = perform_migration
        self.survey_data_type = survey_data_type
        self.n_cores = n_cores
        self.engine_type = str(kriging_engine).lower().strip()
        self.nrbins = int(nrbins) if nrbins is not None else None
        self.ice_density = float(ice_density)
        self.g = float(g)
        self.show_progress = bool(show_progress)

        # Gradient, DEM, Feature, and Sample Points Caches
        self._gradient_cache: dict[str, np.ndarray] | None = None
        self._fft_dem_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._smoothed_dem_cache: dict[float, np.ndarray] = {}
        self._sample_pts_cache: dict[tuple[str, str], np.ndarray] = {}

        # Decoupled sub-engines strictly bound to GridGeometry
        self.migrator = EikonalMigrator(self.dem_grid, geometry=self.geometry, outline_mask=self.outline_mask)
        self.bss_optimizer = BSSOptimizer(self.dem_grid, geometry=self.geometry)
        self.kriging_engine = KrigingEngine(self.dem_grid, geometry=self.geometry, outline_mask=self.outline_mask)
        self.finalizer = BedrockFinalizer(self.dem_grid, geometry=self.geometry, outline_mask=self.outline_mask)

        # Internal state
        self.survey_points: np.ndarray | None = None
        self.migrated_points: np.ndarray | None = None
        self._traveltime_grid: np.ndarray | None = None
        self.opt_kc: float | None = None
        self.opt_slope: np.ndarray | None = None
        self.kriged_thickness: np.ndarray | None = None
        self.kriged_bedrock: np.ndarray | None = None
        self.kriged_variance: np.ndarray | None = None
        self.rf_filled_bedrock: np.ndarray | None = None
        self.blended_bedrock: np.ndarray | None = None
        self.final_grid: np.ndarray | None = None
        self.final_thickness: np.ndarray | None = None
        self.kriged_std: np.ndarray | None = None
        self.final_bss: np.ndarray | None = None
        self.bss_std: np.ndarray | None = None
        self.opt_variogram_params: dict[str, float] | None = None
        self.config: dict[str, Any] = {}
        self.config_path: str | None = None

    @property
    def traveltime_grid(self) -> np.ndarray | None:
        """Reconstructed pre-migration signal traveltime grid T(x,y)."""
        return self._traveltime_grid

    @traveltime_grid.setter
    def traveltime_grid(self, value: np.ndarray | None) -> None:
        self._traveltime_grid = value

    @property
    def opt_wavelength(self) -> float | None:
        """Optimal physical spatial cutoff wavelength lambda_c [m]."""
        if self.opt_kc is None:
            return None
        return compute_cutoff_wavelength(self.opt_kc, self.dx, self.dy)

    @property
    def thickness_grid(self) -> np.ndarray | None:
        """Final predicted ice thickness grid D(x,y) [m]."""
        return self.final_thickness if self.final_thickness is not None else self.kriged_thickness

    @property
    def thickness_std_grid(self) -> np.ndarray | None:
        """Final ice thickness Kriging standard error uncertainty grid [m]."""
        return self.kriged_std

    @property
    def basal_shear_stress_grid(self) -> np.ndarray | None:
        """Final basal shear stress grid tau_b(x,y) [kPa]."""
        return self.final_bss

    @property
    def basal_shear_stress_std_grid(self) -> np.ndarray | None:
        """Final basal shear stress Kriging standard error uncertainty grid sigma_tau_b(x,y) [kPa]."""
        return self.bss_std

    @property
    def outputs_config(self) -> dict[str, Any]:
        """Returns the outputs configuration dictionary."""
        return self.config.get("outputs", {})

    @property
    def outputs_config_obj(self) -> OutputsConfig:
        """Returns structured OutputsConfig instance."""
        return OutputsConfig.from_dict(self.outputs_config)

    @property
    def optimization_config(self) -> dict[str, Any]:
        """Returns the optimization parameters configuration dictionary."""
        return self.config.get("optimization_parameters", {})

    @property
    def slope_floor_deg(self) -> float:
        """Minimum surface slope angle threshold in degrees [°]."""
        return float(self.optimization_config.get("slope_floor_deg", 5.0))

    def _cfg_get(self, section: str, key: str, default: Any = None) -> Any:
        """Helper method for safe configuration parameter lookup."""
        sec = self.config.get(section, {})
        if isinstance(sec, dict):
            val = sec.get(key)
            return val if val is not None else default
        return default

    @property
    def bedrock_map(self) -> BedrockMap | None:
        """Returns the final bedrock elevation grid as a BedrockMap object."""
        if self.final_grid is None:
            return None
        return BedrockMap(
            grid=self.final_grid,
            bounds=self.bounds,
            crs=self.meta.get("crs"),
            transform=self.meta.get("transform"),
            name="Bedrock Elevation",
        )

    @property
    def results(self) -> dict[str, Any]:
        """Returns a dictionary containing all computed pipeline outputs."""
        return {
            "bedrock_map": self.bedrock_map,
            "thickness_grid": self.thickness_grid,
            "thickness_std_grid": self.thickness_std_grid,
            "basal_shear_stress_grid": self.basal_shear_stress_grid,
            "basal_shear_stress_std_grid": self.basal_shear_stress_std_grid,
            "migrated_points": self.migrated_points,
            "opt_kc": self.opt_kc,
            "opt_wavelength": self.opt_wavelength,
            "kriged_bedrock": self.kriged_bedrock,
            "rf_filled_bedrock": self.rf_filled_bedrock,
            "blended_bedrock": self.blended_bedrock,
        }

    def summary(self) -> str:
        """Prints and returns a formatted execution summary report of the Solver state and computed results."""
        lines = [
            "================================================================================",
            "                        PYSOLE SOLVER EXECUTION SUMMARY                         ",
            "================================================================================",
            f"  DEM Shape Grid       : {self.dem_grid.shape[0]} x {self.dem_grid.shape[1]} (dx={self.dx:.2f} m, dy={self.dy:.2f} m)",
            f"  Spatial Bounds [m]   : minx={self.bounds[0]:.1f}, miny={self.bounds[1]:.1f}, maxx={self.bounds[2]:.1f}, maxy={self.bounds[3]:.1f}",
            f"  CRS Metadata         : {self.meta.get('crs') or 'Not defined (Local Meters)'}",
            f"  Execution Engine     : {self.engine_type.upper()} Dual Kriging",
            f"  3D Ray Migration     : {'ENABLED' if self.perform_migration else 'DISABLED/SKIPPED'} ({self.survey_data_type})",
            f"  Optimal Corner Freq  : {f'kc = {self.opt_kc:.4f} rad/m (lambda_c = {self.opt_wavelength:.1f} m)' if self.opt_kc is not None and self.opt_wavelength is not None else 'Not optimized'}",
            f"  Pre-Migration Target : {self.pre_kriging_method.capitalize()} Kriging (Target: '{self.pre_interpolation_target}', Drifts: {self.pre_drift_terms})",
            f"  Post-Migration Target: {self.post_kriging_method.capitalize()} Kriging (Target: '{self.post_interpolation_target}', Drifts: {self.post_drift_terms})",
        ]
        if self.final_thickness is not None:
            valid_h = self.final_thickness[self.outline_mask]
            if len(valid_h) > 0:
                lines.append(f"  Ice Thickness Range  : min={np.nanmin(valid_h):.1f} m, mean={np.nanmean(valid_h):.1f} m, max={np.nanmax(valid_h):.1f} m")
        if self.final_bss is not None:
            valid_tau = self.final_bss[self.outline_mask]
            if len(valid_tau) > 0:
                lines.append(f"  Basal Shear Stress   : min={np.nanmin(valid_tau):.1f} kPa, mean={np.nanmean(valid_tau):.1f} kPa, max={np.nanmax(valid_tau):.1f} kPa")
        lines.append("================================================================================")
        summary_text = "\n".join(lines)
        logger.info(summary_text)
        return summary_text

    def _get_resolved_output_prefix(self) -> str:
        raw_prefix = self.outputs_config_obj.output_prefix or "final"
        prefix_path = Path(raw_prefix)

        # Strip extension if user accidentally provided one (e.g. "final.tif")
        if prefix_path.suffix.lower() in [".tif", ".tiff", ".asc", ".csv", ".npy", ".geotiff"]:
            prefix_path = prefix_path.with_suffix("")

        return self.resolve_path(str(prefix_path))

    def _export_optional_raster(self, grid: np.ndarray | None, suffix: str, name: str) -> str | list[str] | None:
        if grid is None:
            return None
        base_prefix = self._get_resolved_output_prefix()
        filepath = f"{base_prefix}_{suffix}"
        fmt = self.outputs_config_obj.output_format
        raster = BedrockMap(grid=grid, bounds=self.bounds, crs=self.meta.get("crs"), transform=self.meta.get("transform"), name=name)
        saved = raster.save(filepath, formats=fmt)
        if isinstance(saved, list):
            for sf in saved:
                logger.info(f"   Saved optional {name} map to: {sf}")
        else:
            logger.info(f"   Saved optional {name} map to: {saved}")
        return saved

    def _export_optional_points_csv(self, points: np.ndarray | None, suffix: str) -> str | None:
        if points is None:
            return None
        base_prefix = self._get_resolved_output_prefix()
        filepath = f"{base_prefix}_{suffix}.csv"
        saved = save_points_csv(points, filepath)
        logger.info(f"   Saved optional migrated survey points to: {saved}")
        return saved

    def export_outputs(self, stage: str | None = None) -> list[str]:
        """
        Exports optional spatial datasets according to configured boolean output flags.

        Parameters
        ----------
        stage : str, optional
            Workflow stage name ('migration', 'finalization', or None to export all stages).

        Returns
        -------
        saved_files : list of str
            Paths of saved output files.
        """
        saved: list[str] = []
        cfg_out = self.outputs_config_obj

        if stage is None or stage == "migration":
            if cfg_out.save_traveltime_grid and self._traveltime_grid is not None:
                res = self._export_optional_raster(self._traveltime_grid, suffix="traveltime", name="traveltime")
                if res:
                    saved.extend([res] if isinstance(res, str) else res)
            if cfg_out.save_migrated_points and self.migrated_points is not None:
                res = self._export_optional_points_csv(self.migrated_points, suffix="migrated_points")
                if res:
                    saved.append(res)

        if stage is None or stage == "finalization":
            if cfg_out.save_thickness_grid and self.final_thickness is not None:
                res = self._export_optional_raster(self.final_thickness, suffix="thickness", name="thickness")
                if res:
                    saved.extend([res] if isinstance(res, str) else res)
            if cfg_out.save_thickness_uncertainty and self.kriged_std is not None:
                res = self._export_optional_raster(self.kriged_std, suffix="thickness_uncertainty", name="thickness uncertainty")
                if res:
                    saved.extend([res] if isinstance(res, str) else res)
            if cfg_out.save_basal_shear_stress and self.final_bss is not None:
                res = self._export_optional_raster(self.final_bss, suffix="basal_shear_stress", name="basal shear stress")
                if res:
                    saved.extend([res] if isinstance(res, str) else res)
            if cfg_out.save_basal_shear_stress_uncertainty and self.bss_std is not None:
                res = self._export_optional_raster(self.bss_std, suffix="basal_shear_stress_uncertainty", name="basal shear stress uncertainty")

        return saved

    def clear_intermediate_grids(self) -> None:
        """Clears intermediate 2D array grids from memory to optimize footprint for large datasets."""
        self._traveltime_grid = None
        self.kriged_thickness = None
        self.kriged_variance = None
        self.kriged_std = None
        self.final_bss = None
        self.bss_std = None

    @property
    def effective_output_dir(self) -> str:
        """Returns the effective output directory for file resolution."""
        cfg_p = getattr(self, "config_path", None)
        eff_dir = resolve_output_dir(
            output_dir=self.output_dir,
            survey_data_path=self.survey_data_path,
            config_path=cfg_p,
        )
        return str(eff_dir)

    def resolve_input_path(self, path: str | Path | os.PathLike | None) -> str | None:
        """Resolves an input file path relative to config_path parent or working directory."""
        cfg_p = getattr(self, "config_path", None)
        return resolve_input_config_path(path, config_path=cfg_p)

    def resolve_path(self, path: str | Path | os.PathLike | None) -> str | None:
        """Resolves a target file or directory path relative to effective_output_dir if relative."""
        cfg_p = getattr(self, "config_path", None)
        return resolve_config_path(path, output_dir=self.output_dir, survey_data_path=self.survey_data_path, config_path=cfg_p)

    @property
    def plots_dir(self) -> str:
        """Directory where generated plots are automatically saved."""
        target = self._raw_plots_dir if self._raw_plots_dir is not None else "figures"
        return self.resolve_path(target)

    @plots_dir.setter
    def plots_dir(self, value: str | Path | None) -> None:
        self._raw_plots_dir = value

    @property
    def plot_extent(self) -> list[float]:
        """Returns Matplotlib plot extent [minx, maxx, miny, maxy]."""
        return self.geometry.extent

    def _get_dem_gradients(self) -> dict[str, np.ndarray]:
        if self._gradient_cache is None:
            self._gradient_cache = compute_gradients(self.dem_grid, dx=self.dx, dy=self.dy)
        return self._gradient_cache

    def _get_fft_dem_grids(self) -> tuple[np.ndarray, np.ndarray]:
        if self._fft_dem_cache is None:
            A_shift_dem, k_grid_dem, _ = precompute_fft_grid(self.dem_grid, dx=self.dx, dy=self.dy)
            self._fft_dem_cache = (A_shift_dem, k_grid_dem)
        return self._fft_dem_cache

    def get_smoothed_dem(self, kc: float) -> np.ndarray:
        kc_key = round(float(kc), 6)
        if kc_key not in self._smoothed_dem_cache:
            A_shift_dem, k_grid_dem = self._get_fft_dem_grids()
            smoothed_dem = fft_gaussian_smooth_precomputed(A_shift_dem, k_grid_dem, kc=kc)
            self._smoothed_dem_cache[kc_key] = smoothed_dem
        return self._smoothed_dem_cache[kc_key]

    def get_smoothed_slope(self, kc: float) -> np.ndarray:
        smoothed_dem = self.get_smoothed_dem(kc)
        return compute_gradients(smoothed_dem, dx=self.dx, dy=self.dy)["slope_rad"]

    def get_smoothed_curvature(self, kc: float) -> np.ndarray:
        from .smoothing import compute_surface_curvature

        smoothed_dem = self.get_smoothed_dem(kc)
        return compute_surface_curvature(smoothed_dem, dx=self.dx, dy=self.dy)

    def _normalize_survey_data_type(self) -> str:
        """
        Normalizes and validates survey_data_type into one of 3 canonical types:
          - 'one_way_traveltime'
          - 'two_way_traveltime'
          - 'depth'
        """
        dtype_str = str(self.survey_data_type).lower().strip()
        if dtype_str in ("one_way_traveltime", "two_way_traveltime", "depth"):
            return dtype_str
        raise ValueError(
            f"Invalid survey_data_type '{self.survey_data_type}'. "
            f"Must be one of ['one_way_traveltime', 'two_way_traveltime', 'depth']."
        )

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | os.PathLike | dict[str, Any] = "pysole.json",
        log_level: str | None = None,
    ) -> "Solver":
        """Initializes Solver instance using inputs defined in pysole.json configuration file."""
        from .config import load_config
        cfg = load_config(config_path, log_level=log_level)
        inputs = cfg.get("inputs", {})
        
        dem_path = inputs.get("dem_path")
        if dem_path is None:
            raise ValueError(
                "Missing required 'dem_path' parameter in configuration. "
                "Please specify a valid surface DEM file path (e.g. '.tif', '.asc', '.csv', '.npy') under the 'inputs' section."
            )

        spatial = cfg.get("spatial_parameters", {})
        migration_cfg = cfg.get("migration_parameters", {})
        kriging_cfg = cfg.get("kriging_parameters", {})
        outputs = cfg.get("outputs", {})

        survey_dtype = inputs.get("survey_data_type", "one_way_traveltime")

        pre_krig_cfg = kriging_cfg.get("pre_migration", {}) if isinstance(kriging_cfg.get("pre_migration"), dict) else {}
        post_krig_cfg = kriging_cfg.get("post_migration", {}) if isinstance(kriging_cfg.get("post_migration"), dict) else {}

        pre_target = pre_krig_cfg.get("interpolation_target", "P")
        pre_method = pre_krig_cfg.get("method", "universal" if str(pre_target).upper() == "T" else "ordinary")
        pre_drifts = pre_krig_cfg.get("drift_terms", ["sia"] if str(pre_target).upper() == "T" else [])
        pre_var_model = pre_krig_cfg.get("variogram_model", "spherical")
        pre_zero_boundary = pre_krig_cfg.get("include_zero_boundary_condition", True)

        post_target = post_krig_cfg.get("interpolation_target", "P")
        post_method = post_krig_cfg.get("method", "universal" if str(post_target).upper() == "D" else "ordinary")
        post_drifts = post_krig_cfg.get("drift_terms", ["sia"] if str(post_target).upper() == "D" else [])
        post_var_model = post_krig_cfg.get("variogram_model", "spherical")
        post_zero_boundary = post_krig_cfg.get("include_zero_boundary_condition", True)

        opt_cfg = cfg.get("optimization_parameters", {})
        n_cores = inputs.get("n_cores", -1)
        kriging_engine = kriging_cfg.get("engine", "native")
        nrbins = opt_cfg.get("nrbins", None)
        ice_density = inputs.get("ice_density", 900.0)
        g_val = inputs.get("g", 9.81)

        show_progress_val = inputs.get("show_progress", True)

        solver = cls(
            dem=inputs.get("dem_path"),
            outline=inputs.get("outline_path"),
            dx=spatial.get("dx"),
            dy=spatial.get("dy"),
            bounds=spatial.get("bounds"),
            origin=spatial.get("origin"),
            crs=spatial.get("crs"),
            pre_kriging_method=pre_method,
            pre_drift_terms=pre_drifts,
            pre_variogram_model=pre_var_model,
            pre_zero_boundary=pre_zero_boundary,
            pre_interpolation_target=pre_target,
            post_kriging_method=post_method,
            post_drift_terms=post_drifts,
            post_variogram_model=post_var_model,
            post_zero_boundary=post_zero_boundary,
            post_interpolation_target=post_target,
            perform_migration=migration_cfg.get("perform_migration", True),
            survey_data_type=survey_dtype,
            plots_dir=outputs.get("plots_dir", None),
            n_cores=n_cores,
            kriging_engine=kriging_engine,
            nrbins=nrbins,
            ice_density=ice_density,
            g=g_val,
            output_dir=outputs.get("output_dir"),
            survey_data_path=inputs.get("survey_data_path"),
            survey_profile_column=inputs.get("survey_profile_column"),
            config_path=config_path if not isinstance(config_path, dict) else None,
            show_progress=show_progress_val,
        )
        solver.config = cfg
        solver.config_path = str(config_path)
        return solver

    def get_sample_points(self, stage: str, target_type: str = "P") -> np.ndarray:
        """
        Retrieves or constructs cached 3-column sample points matrix [X, Y, Target]
        for the specified stage ('pre_migration' or 'post_migration') and target_type ('P', 'T', 'D').
        Guarantees that slope interpolation and multiplication for Product 'P' (T*sin(alpha) or D*sin(alpha))
        is evaluated ONCE and cached.
        """
        stage_key = "pre_migration" if "pre" in stage.lower() else "post_migration"
        target_upper = str(target_type).upper().strip()
        cache_key = (stage_key, target_upper)

        if not hasattr(self, "_sample_pts_cache"):
            self._sample_pts_cache = {}

        if cache_key in self._sample_pts_cache:
            return self._sample_pts_cache[cache_key]

        pts = self.migrated_points if (stage_key == "post_migration" and self.migrated_points is not None) else getattr(self, "pre_kriging_points", None)
        if pts is None:
            if stage_key == "post_migration" and self.migrated_points is not None:
                pts = self.migrated_points
            else:
                pts = getattr(self, "survey_points", None)

        if pts is None:
            raise ValueError(f"No survey/migrated points available to create sample points for stage '{stage_key}'.")

        if self.opt_slope is None:
            prefix = "01_" if stage_key == "pre_migration" else "03_"
            self.optimize_bss(prefix=prefix)

        val_col = 3 if pts.shape[1] >= 4 else 2

        if target_upper == "P":
            opt_slope_sin = np.sin(self.opt_slope)
            interp_slope = self.geometry.create_interpolator(opt_slope_sin, fill_value=np.nan)
            pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))  # (Y, X)
            slopes_pts = interp_slope(pts_xy)
            min_slope_sin = np.sin(np.radians(self.slope_floor_deg))
            slopes_pts = np.maximum(np.nan_to_num(slopes_pts, nan=min_slope_sin), min_slope_sin)

            product_values = pts[:, val_col] * slopes_pts
            sample_pts = np.column_stack((pts[:, 0], pts[:, 1], product_values))
        else:
            sample_pts = np.column_stack((pts[:, 0], pts[:, 1], pts[:, val_col]))

        self._sample_pts_cache[cache_key] = sample_pts
        return sample_pts

    def _execute_kriging_pass(
        self,
        target_type: str,
        points: np.ndarray,
        krig_method: str,
        drift_terms: list[str] | None,
        var_model: str,
        zero_boundary: bool,
        pass_name: str = "Pass 1: Pre-Migration",
    ) -> KrigingResult:
        """
        Unified Pass Dispatcher executing Kriging interpolation for direct targets ('T' / 'D')
        or BSS product targets ('P').
        """
        target_upper = str(target_type).upper().strip()
        is_pre = "Pass 1" in pass_name
        stage_key = "pre_migration" if is_pre else "post_migration"

        sample_pts = self.get_sample_points(stage_key, target_upper)

        if target_upper == "P":
            logger.info(f"   [{pass_name}] Performing {krig_method.capitalize()} Kriging Interpolation for BSS Product P(x,y)...")
            if drift_terms and "sia" in drift_terms:
                logger.warning(
                    f"   [{pass_name} Warning] 'sia' drift is active during BSS product P(x,y) interpolation. "
                    "This can cause 1/sin^2(alpha) double-scaling artifacts at low-slope margins. "
                    "Recommendation: Use 'ordinary' Kriging or non-slope spatial drifts (e.g. ['z_dem']) for product targets."
                )

        pass_cfg = self.config.get("kriging_parameters", {}).get(stage_key, {})
        run_analyzer = pass_cfg.get("drift_analyzer", False)

        if krig_method.lower() == "ordinary":
            if run_analyzer:
                logger.info("   [INFO] Ordinary Kriging detected: drift_analyzer automatically set to False.")
            run_analyzer = False

        if run_analyzer:
            is_batch = getattr(self, "is_batch_mode", False) or not sys.stdin.isatty()
            if is_batch:
                logger.info("   [INFO] Non-interactive environment detected (TTY disabled or --batch flag set). Automatically overriding interactive options to False.")

            prof_col = self.survey_profile_column or self.config.get("inputs", {}).get("survey_profile_column")
            analyzer = DriftAnalyzer(
                mode=stage_key,
                target_name=self.survey_data_type,
                interpolation_target=target_upper,
                include_zero_boundary=zero_boundary,
                survey_profile_column=prof_col,
            )

            a_0 = 100.0
            if hasattr(self, "opt_variogram_params") and self.opt_variogram_params:
                a_0 = float(self.opt_variogram_params.get("range", 100.0))

            var_params = {"nugget": 0.0, "sill": 1.0, "range": a_0}
            alpha_deg = np.degrees(self.opt_slope) if self.opt_slope is not None else np.zeros_like(self.dem_grid)
            prof_data = sample_pts[:, 4] if (prof_col and sample_pts is not None and sample_pts.shape[1] >= 5) else None

            diag = analyzer.run_diagnostics(
                x_pts=sample_pts[:, 0],
                y_pts=sample_pts[:, 1],
                z_values=sample_pts[:, 2],
                dem_grid=self.dem_grid,
                dx=self.dx,
                dy=self.dy,
                bounds=self.bounds,
                alpha_opt_deg=alpha_deg,
                variogram_model=var_model,
                variogram_params=var_params,
                profile_data=prof_data,
                interactive=not is_batch,
            )
            if diag:
                drift_terms = diag[0].terms
                krig_method = "universal"

        ext_drifts: dict[str, np.ndarray] = {}
        if drift_terms and "curvature_dem" in drift_terms:
            kc_use = self.opt_kc if self.opt_kc is not None else 0.05
            ext_drifts["curvature_dem"] = self.get_smoothed_curvature(kc_use)

        krig_res = kriging_interpolation(
            sample_points=sample_pts,
            geometry=self.geometry,
            method=krig_method,
            variogram_model=var_model,
            variogram_params=getattr(self, "opt_variogram_params", None),
            dem_grid=self.dem_grid,
            opt_slope_grid=self.opt_slope,
            drift_terms=drift_terms,
            outline_mask=self.outline_mask,
            include_zero_boundary_condition=zero_boundary,
            n_cores=self.n_cores,
            engine=self.kriging_engine,
            slope_floor_deg=self.slope_floor_deg,
            show_progress=self.show_progress,
            external_drift_grid=ext_drifts if len(ext_drifts) > 0 else None,
        )

        if target_upper == "P":
            opt_slope_sin = np.sin(self.opt_slope)
            min_slope_sin = np.sin(np.radians(self.slope_floor_deg))
            safe_slope_grid = np.maximum(opt_slope_sin, min_slope_sin)
            grid = krig_res.bedrock_grid / safe_slope_grid
            var = krig_res.variance_grid / (safe_slope_grid**2)
            return KrigingResult(bedrock_grid=grid, variance_grid=var)

        return krig_res

    def migrate_eikonal(
        self,
        travel_times: str | Path | os.PathLike | np.ndarray | None = None,
        velocity: float | None = None,
        interactive: bool = False,
        plotit: bool = False,
    ) -> np.ndarray:
        """Migrates zero-offset GPR or seismic travel times into 3D space using the Eikonal equation."""
        if travel_times is None:
            travel_times = self.survey_data_path
        if travel_times is None:
            raise ValueError("No survey_data_path or travel_times provided for Eikonal migration.")

        if not self.survey_data_path and isinstance(travel_times, (str, Path, os.PathLike)):
            self.survey_data_path = str(travel_times)

        pts = load_survey_points(
            self.resolve_input_path(travel_times),
            bounds=self.bounds,
            dem_grid=self.dem_grid,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
        )

        pts_minx, pts_miny = float(pts[:, 0].min()), float(pts[:, 1].min())
        pts_maxx, pts_maxy = float(pts[:, 0].max()), float(pts[:, 1].max())
        if (self.bounds[0] == 0.0 and self.bounds[1] == 0.0) and (pts_minx > 10000.0 or pts_miny > 10000.0):
            err_msg = (
                f"\n[Spatial Coordinate Origin Mismatch Error] Survey points use projected metric coordinates "
                f"(X in [{pts_minx:.1f}, {pts_maxx:.1f}], Y in [{pts_miny:.1f}, {pts_maxy:.1f}]), "
                f"but DEM bounding box defaults to origin (0.0, 0.0)!\n"
                f"For headerless CSV/NPY DEMs, you must define 'spatial_parameters.origin': [xll, yll] or "
                f"'spatial_parameters.bounds': [minx, miny, maxx, maxy] in pysole.json so the DEM aligns with survey points."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        dtype_str = self._normalize_survey_data_type()

        if dtype_str == "two_way_traveltime":
            logger.info("   [Survey Data Type: TWT] Two-Way Traveltimes detected. Converting to One-Way Traveltimes (OWTT = TWT / 2.0).")
            pts[:, 3] = pts[:, 3] / 2.0
        elif dtype_str == "one_way_traveltime":
            logger.info("   [Survey Data Type: OWTT] One-Way Traveltimes detected.")

        self.survey_points = pts

        if dtype_str == "depth":
            logger.info("   [Survey Data Type: Depth] Input data represents direct depth/thickness measurements. Skipping 3D Eikonal ray migration.")
            self.migrated_points = pts.copy()
            if self.outputs_config.get("save_migrated_points", False):
                logger.info("   [Migration Skipped] 3D Eikonal Ray Migration is disabled. Skipping 'save_migrated_points' output.")
            return self.migrated_points

        v_eff = float(velocity) if velocity is not None else 0.16
        if velocity is None and interactive:
            try:
                val = input("Signal Propagation Velocity [m/ns]? (e.g. 0.16): ").strip()
                if val:
                    v_eff = float(val)
            except Exception:
                pass

        if not self.perform_migration:
            logger.info("   [Migration Skipped] 'perform_migration' is set to False in configuration. Using unmigrated survey points directly.")
            unmig_depths = pts[:, 3] * v_eff if pts[:, 3].max() > 15.0 else pts[:, 3]
            self.migrated_points = np.column_stack((pts[:, 0], pts[:, 1], pts[:, 2], unmig_depths))
            if self.outputs_config.get("save_migrated_points", False):
                logger.info("   [Migration Skipped] 3D Eikonal Ray Migration is disabled. Skipping 'save_migrated_points' output.")
            return self.migrated_points

        krig1_res = self._execute_kriging_pass(
            target_type=self.pre_interpolation_target,
            points=pts,
            krig_method=self.pre_kriging_method,
            drift_terms=self.pre_drift_terms,
            var_model=self.pre_variogram_model,
            zero_boundary=self.pre_zero_boundary,
            pass_name="Pass 1: Pre-Migration",
        )
        tt_grid = np.maximum(krig1_res.bedrock_grid, 0.0)
        if self.outline_mask is not None:
            tt_grid[~self.outline_mask] = np.nan
        self._traveltime_grid = tt_grid

        if self.outputs_config_obj.save_traveltime_grid:
            self._export_optional_raster(self._traveltime_grid, suffix="traveltime", name="traveltime")

        cached_dem_grads = self._get_dem_gradients()

        logger.info(f"   [Ray Migration] Executing 3D Eikonal Ray Displacement (velocity v = {v_eff:.4f} m/ns)...")
        mig_res = migrate_eikonal_points(
            dem=self.dem_grid,
            travel_time_grid=tt_grid,
            survey_points=pts,
            geometry=self.geometry,
            velocity=v_eff,
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
                    ans = input(f"\nCurrent migration velocity v = {v_eff:.4f} m/ns. Test another migration velocity? [y/N]: ").strip().lower()
                    if ans in ["y", "yes"]:
                        val = input("Enter new signal propagation velocity [m/ns] (e.g. 0.15): ").strip()
                        if val:
                            v_eff = float(val)
                            logger.info(f"Re-running 3D Eikonal ray migration with v = {v_eff:.4f} m/ns...")
                            mig_res2 = migrate_eikonal_points(
                                dem=self.dem_grid,
                                travel_time_grid=tt_grid,
                                survey_points=pts,
                                geometry=self.geometry,
                                velocity=v_eff,
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

        if self.outputs_config_obj.save_migrated_points and self.migrated_points is not None:
            self._export_optional_points_csv(self.migrated_points, suffix="migrated_points")

        return self.migrated_points

    def optimize_bss(
        self,
        kc_max: float | None = None,
        kc_min: float | None = None,
        d_kc: float | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        d_lambda: float | None = None,
        fft_filter_metric: str | None = None,
        nrbins: int | None = None,
        prefix: str | None = None,
        interactive: bool = False,
        plotit: bool = False,
    ) -> float:
        """
        Iterative optimization process to determine optimum surface slope smoothing degree.
        When interactive is True, enables CLI prompts to adjust filter spectrum parameters.
        """
        pts = self.migrated_points if self.migrated_points is not None else self.survey_points
        if pts is None:
            xx, yy = np.meshgrid(self.x_coords[::5], self.y_coords[::5])
            pts = np.column_stack((xx.ravel(), yy.ravel(), self.dem_grid[::5, ::5].ravel(), np.ones(xx.size) * 10.0))

        if hasattr(self, "config") and isinstance(self.config, dict):
            opt_cfg = self.config.get("optimization_parameters", {})
            if fft_filter_metric is None:
                fft_filter_metric = opt_cfg.get("fft_filter_metric", "wavenumber")
            if kc_max is None:
                kc_max = opt_cfg.get("kc_max")
            if kc_min is None:
                kc_min = opt_cfg.get("kc_min")
            if d_kc is None:
                d_kc = opt_cfg.get("d_kc")
            if lambda_min is None:
                lambda_min = opt_cfg.get("lambda_min")
            if lambda_max is None:
                lambda_max = opt_cfg.get("lambda_max")
            if d_lambda is None:
                d_lambda = opt_cfg.get("d_lambda")
            if nrbins is None:
                nrbins = opt_cfg.get("nrbins")

        if fft_filter_metric is None:
            fft_filter_metric = "wavenumber"

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
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            d_lambda=d_lambda,
            fft_filter_metric=fft_filter_metric,
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
        self.opt_variogram_params = opt_res.opt_variogram_params
        if opt_res.all_smoothed_dems:
            self._smoothed_dem_cache.update(opt_res.all_smoothed_dems)
        return self.opt_kc

    def calculate_bedrock(
        self,
        method: str | None = None,
        variogram_model: str | None = None,
        interactive: bool = False,
        plotit: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates bedrock elevation grid using Dual Kriging interpolation pass.
        """
        pts = self.migrated_points if self.migrated_points is not None else self.survey_points
        if pts is None:
            raise ValueError("No survey or migrated points available. Run migrate_eikonal() first.")

        should_plot = plotit or interactive or (self.plots_dir is not None)

        krig_method = method if method is not None else self.post_kriging_method
        var_model = variogram_model if variogram_model is not None else self.post_variogram_model

        krig_res = self._execute_kriging_pass(
            target_type=self.post_interpolation_target,
            points=pts,
            krig_method=krig_method,
            drift_terms=self.post_drift_terms,
            var_model=var_model,
            zero_boundary=self.post_zero_boundary,
            pass_name="Pass 2: Post-Migration",
        )
        grid_raw = krig_res.bedrock_grid
        prod_var = krig_res.variance_grid

        val_col = 3 if pts.shape[1] >= 4 else 2
        max_thickness = max(float(np.max(pts[:, val_col])) * 1.5, 500.0)
        thickness_grid = np.clip(grid_raw, 0.0, max_thickness)

        if self.outline_mask is not None:
            thickness_grid[~self.outline_mask] = 0.0

        self.kriged_thickness = thickness_grid.copy()
        self.kriged_bedrock = self.dem_grid - thickness_grid
        self.kriged_variance = prod_var
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

    def interpolate_kriging(self, *args, **kwargs):
        """Alias for calculate_bedrock()."""
        return self.calculate_bedrock(*args, **kwargs)

    def fill_holes_rf(self) -> np.ndarray:
        """Trains Random Forest ML model to fill remaining bedrock holes."""
        if self.kriged_bedrock is None:
            self.calculate_bedrock()

        self.rf_filled_bedrock = self.finalizer.fill_holes(self.kriged_bedrock, n_cores=self.n_cores, show_progress=self.show_progress)
        return self.rf_filled_bedrock

    def apply_geomorph_smoothing(self, min_gap_dist: float | None = None) -> np.ndarray:
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
        kc_cutoff: float | None = None,
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
            kc = kc_cutoff if kc_cutoff is not None else (self.opt_kc if self.opt_kc is not None else 1.0)

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

    def finalize_bedrock(
        self,
        interactive: bool = True,
        plotit: bool = True,
        random_forest_gap_filling: bool | None = None,
        apply_margin_blend: bool | None = None,
        min_gap_dist: float | None = None,
        smooth_bedrock: bool = False,
        smoothing_method: str = "gaussian",
        smoothing_sigma: float = 1.5,
        smoothing_kernel_size: int = 3,
        smoothing_kc_cutoff: float | None = None,
    ) -> BedrockMap:
        """
        Finalizes bedrock topography by executing optional Random Forest gap filling,
        margin blending, and spatial DEM smoothing. Returns completed BedrockMap.
        """
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

        self.export_outputs("finalization")

        self.final_grid = self.blended_bedrock
        return BedrockMap(
            grid=self.final_grid,
            bounds=self.bounds,
            crs=self.meta.get("crs"),
            transform=self.meta.get("transform"),
            name="final_bedrock",
        )

    def finalize_topography(self, *args, **kwargs) -> BedrockMap:
        """Alias for finalize_bedrock()."""
        return self.finalize_bedrock(*args, **kwargs)

    def recommend_drift_model(
        self,
        stage: str = "post_migration",
        interactive: bool = False,
    ) -> list[Any]:
        """
        Runs Universal Kriging Drift Analyzer diagnostics to recommend optimal drift models.
        """
        target_name = "traveltime" if stage == "pre_migration" else "depth"
        stage_prefix = stage.split("_")[0]
        interp_target = getattr(self, f"{stage_prefix}_interpolation_target", "P")
        inc_zero = getattr(self, f"{stage_prefix}_zero_boundary", True)

        analyzer = DriftAnalyzer(
            mode=stage,
            target_name=target_name,
            interpolation_target=interp_target,
            include_zero_boundary=inc_zero,
            survey_profile_column=getattr(self, "survey_profile_column", None),
        )

        sample_pts = self.get_sample_points(stage, interp_target)
        x_pts, y_pts, z_vals = sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2]
        opt_alpha = np.degrees(self.opt_slope) if self.opt_slope is not None else np.zeros_like(self.dem_grid)

        return analyzer.run_diagnostics(
            x_pts=x_pts,
            y_pts=y_pts,
            z_values=z_vals,
            dem_grid=self.dem_grid,
            dx=self.dx,
            dy=self.dy,
            bounds=self.bounds,
            alpha_opt_deg=opt_alpha,
            variogram_model="spherical",
            variogram_params={"range": 100.0, "sill": 1.0, "nugget": 0.0},
            interactive=interactive,
        )

    def plan_survey(
        self,
        kc: float = 0.5,
        tau_0: float = 100e3,
        max_length_km: float = 5.0,
        output_prefix: str | None = None,
        output_dir: str | Path | None = None,
        plots_dir: str | Path | None = None,
        output_format: str | None = None,
    ) -> dict[str, Any]:
        """
        Executes forward survey planning for unprobed glaciers using synthetic SIA modeling.
        """
        cfg = getattr(self, "config", {})
        outputs = cfg.get("outputs", {})

        prefix = output_prefix or outputs.get("output_prefix", "final")
        out_d = output_dir or self.output_dir or resolve_output_dir(outputs.get("output_dir"), config_path=self.config_path)
        plots_d = plots_dir or self.plots_dir
        fmt = output_format or outputs.get("output_format", "tif")
        if isinstance(fmt, list):
            fmt = fmt[0]

        planner = SurveyPlanner(
            dem=self.dem_grid,
            outline=self.outline_mask,
            dx=self.dx,
            dy=self.dy,
            bounds=self.bounds,
            crs=self.meta.get("crs"),
            ice_density=float(self.config.get("inputs", {}).get("ice_density", 900.0)),
            g=float(self.config.get("inputs", {}).get("g", 9.81)),
        )
        return planner.plan_survey(
            kc=kc,
            tau_0=tau_0,
            max_length_km=max_length_km,
            output_prefix=prefix,
            output_dir=out_d,
            plots_dir=plots_d,
            output_format=fmt,
        )

    def run_pipeline(
        self,
        survey_data_path: str | Path | os.PathLike | None = None,
    ) -> BedrockMap:
        """
        Executes the complete 5-step PySole workflow using loaded configuration settings.

        Parameters
        ----------
        survey_data_path : str, Path, or os.PathLike, optional
            Path to survey points dataset. Overrides solver's survey_data_path attribute.

        Returns
        -------
        bedrock_map : BedrockMap
            Final predicted bedrock elevation grid.
        """
        if survey_data_path is not None:
            self.survey_data_path = str(survey_data_path)

        cfg = getattr(self, "config", {})
        inputs = cfg.get("inputs", {})
        migration = cfg.get("migration_parameters", {})
        opt = cfg.get("optimization_parameters", {})
        fin_cfg = cfg.get("finalization_parameters", {})
        outputs = cfg.get("outputs", {})

        survey_data_path = self.survey_data_path or inputs.get("survey_data_path")
        if not survey_data_path:
            logger.info("[INFO] No survey_data_path provided. Automatically switching execution mode to Unprobed Glacier Survey Planner.")
            res = self.plan_survey()
            return BedrockMap(grid=res["sia_modelled_depth"], bounds=self.bounds, crs=self.meta.get("crs"), transform=self.meta.get("transform"), name="sia_modelled_depth")

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

        if self.perform_migration and self._normalize_survey_data_type() != "depth":
            logger.info("2. Performing 3D Eikonal Ray Migration...")
        else:
            logger.info("2. Skipping 3D Eikonal Ray Migration...")

        is_batch = (not self.show_progress) or (not sys.stdin.isatty())
        opt_interactive = False if is_batch else opt.get("interactive_optimization", False)
        mig_interactive = False if is_batch else migration.get("interactive_migration", False)

        self.migrate_eikonal(
            travel_times=survey_data_path,
            velocity=migration.get("velocity"),
            interactive=mig_interactive,
            plotit=True,
        )

        logger.info("3. Performing Post-Migration BSS Slope Optimization & Depth Interpolation (Field D(x,y))...")
        opt_kc = self.optimize_bss(
            interactive=opt_interactive,
            plotit=True,
        )
        opt_wl = compute_cutoff_wavelength(opt_kc, self.dx, self.dy)
        logger.info(f"   Optimal Post-Migration Corner Frequency k_c = {opt_kc:.4f} (cutoff wavelength λ_c = {opt_wl:.2f} m)")
        self.interpolate_kriging(interactive=opt_interactive, plotit=True)

        logger.info("4. Finalizing Bedrock Topography...")
        bedrock_map = self.finalize_topography(
            interactive=opt_interactive,
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

        output_format = self.outputs_config_obj.output_format
        resolved_prefix = self._get_resolved_output_prefix()
        bedrock_filepath = f"{resolved_prefix}_bedrock"
        saved_res = bedrock_map.save(bedrock_filepath, formats=output_format)
        if isinstance(saved_res, list):
            for sf in saved_res:
                logger.info(f"5. Saved predicted bedrock map to: {sf}")
        else:
            logger.info(f"5. Saved predicted bedrock map to: {saved_res}")

        return bedrock_map
