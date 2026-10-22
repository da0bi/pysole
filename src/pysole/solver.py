"""
Main Solver API for PySole.

Provides the high-level orchestration interface for physically-informed bedrock
topography reconstruction, 3D Eikonal ray normal migration, Basal Shear Stress (BSS)
spectral slope optimization, Dual Kriging interpolation with rigorous uncertainty
estimation, and geomorphological post-processing.
"""

from pathlib import Path
from typing import Any
import numpy as np
import os
from .raster import BedrockMap, load_dem, load_outline, load_aligned_raster, GridGeometry, load_survey_points, save_points_csv, calculate_ice_volume, IceVolumeResult
from .migration import migrate_eikonal_points, EikonalMigrator
from .variogram import BSSOptimizer, OptimizationResult, compute_cutoff_wavelength, resolve_native_variogram_model
from .interpolation import (
    blend_margin_topography,
    build_drift_grids,
    kriging_interpolation,
    KrigingEngine,
    BedrockFinalizer,
    KrigingResult,
)
from .drift import (
    OPTIONAL_RASTER_DRIFT_TERMS,
    RASTER_DRIFT_TERMS,
    DriftPlan,
    build_drift_suite,
    default_model,
    format_drift_name,
    normalize_engine,
    normalize_method,
    resolve_available_optional_terms,
    resolve_drift_plan,
    slugify_drift_name,
    validate_required_drift_rasters,
)
from .smoothing import compute_gradients, precompute_fft_grid, fft_gaussian_smooth_precomputed, compute_surface_curvature
from .config import OutputsConfig, resolve_path as resolve_config_path, resolve_input_path as resolve_input_config_path, resolve_output_dir, resolve_plots_dir, configured_output_dir, sanitize_config, warn_unknown_config_keys, validate_memory_fraction
from .logging import logger, log_section_header, end_terminal_block
from .drift_analyzer import (
    FAILED_AICC,
    CandidateDriftResult,
    DriftAnalyzer,
    DriftComparisonRecord,
    format_drift_comparison_table,
    save_drift_comparison_files,
)
from .survey_planner import SurveyPlanner
import sys


def _compute_distribution_stats(
    grid: np.ndarray, outline_mask: np.ndarray | None = None
) -> tuple[float, float, float, float]:
    """Computes (mean, median, std, max) across strictly positive finite cells in the active domain."""
    finite = np.isfinite(grid) & (grid > 0.0)
    if outline_mask is not None and np.any(outline_mask):
        valid = grid[outline_mask & finite]
    else:
        valid = grid[finite]
    if valid.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(valid)),
        float(np.median(valid)),
        float(np.std(valid)),
        float(np.max(valid)),
    )


def format_final_result_summary(
    volume_res: IceVolumeResult,
    thick_stats: tuple[float, float, float, float],
    bss_stats: tuple[float, float, float, float],
    model_name: str | None = None,
    model_slug: str | None = None,
    model_terms: list[str] | None = None,
    model_rank: int | None = None,
    candidate_index: int | None = None,
    candidate_total: int | None = None,
    bss_mean_unc: float | None = None,
) -> str:
    """Formats a prominent terminal card summarizing glacier domain area, volume, thickness, and basal shear stress.

    Parameters
    ----------
    volume_res : IceVolumeResult
        Glacier domain area, volume, and propagated uncertainty result.
    thick_stats : tuple of 4 floats
        Ice thickness (mean, median, std_dev, max) in meters across active domain cells.
    bss_stats : tuple of 4 floats
        Basal shear stress (mean, median, std_dev, max) in kPa across active domain cells.
    model_name : str or None, optional
        Human-readable name of the evaluated drift model.
    model_slug : str or None, optional
        Filesystem identifier slug for the drift model.
    model_terms : list of str or None, optional
        List of drift component terms.
    model_rank : int or None, optional
        Rank of candidate model from Drift Analyzer.
    candidate_index : int or None, optional
        1-based candidate index if part of a multi-model evaluation suite.
    candidate_total : int or None, optional
        Total number of candidate models in the suite.
    bss_mean_unc : float or None, optional
        Domain-average Kriging uncertainty in basal shear stress in kPa.

    Returns
    -------
    summary_card : str
        Structured text block ready for terminal printing.
    """
    is_multi = candidate_total is not None and candidate_total > 1 and candidate_index is not None
    if is_multi:
        win_str = " (PRIMARY WINNER)" if candidate_index == 1 else ""
        title_line = f"FINAL RESULT SUMMARY — CANDIDATE {candidate_index} OF {candidate_total}{win_str}"
        disp_name = model_name or (format_drift_name(model_terms) if model_terms is not None else "Ordinary Kriging")
        rank_str = f"#{model_rank or candidate_index}"
        rec_str = " (Recommended)" if candidate_index == 1 else ""
        sub_title = f"Model: {disp_name} | Rank: {rank_str}{rec_str}"
        lines = [
            "=" * 80,
            title_line.center(80),
            sub_title.center(80),
            "=" * 80,
        ]
        terms_repr = str(model_terms) if model_terms is not None else "[]"
        slug_repr = model_slug or "ordinary"
        lines.append(f"   Drift Model Terms:            {terms_repr} (Slug: {slug_repr})")
    else:
        title_line = "FINAL RESULT SUMMARY"
        disp_name = model_name or (format_drift_name(model_terms) if model_terms is not None else "Ordinary Kriging")
        lines = [
            "=" * 80,
            title_line.center(80),
            "=" * 80,
            f"   Drift Model:                  {disp_name}",
        ]

    lines.append(f"   Glacier Domain Area:          {volume_res.area_km2:.3f} km\u00b2 ({volume_res.area_m2:,.0f} m\u00b2)")

    if volume_res.uncertainty_km3 is not None and volume_res.uncertainty_m3 is not None:
        lines.append(
            f"   Total Ice Volume:             {volume_res.volume_km3:.4f} \u00b1 {volume_res.uncertainty_km3:.4f} km\u00b3 "
            f"({volume_res.volume_m3 / 1e6:.2f} \u00b1 {volume_res.uncertainty_m3 / 1e6:.2f} \u00d7 10\u2076 m\u00b3)"
        )
    else:
        lines.append(
            f"   Total Ice Volume:             {volume_res.volume_km3:.4f} km\u00b3 "
            f"({volume_res.volume_m3 / 1e6:.2f} \u00d7 10\u2076 m\u00b3)"
        )

    lines.append("   " + "-" * 77)
    lines.append("   Final Ice Thickness Distribution (Figure 09):")
    if volume_res.mean_uncertainty is not None:
        lines.append(f"     Mean Ice Thickness:         {thick_stats[0]:.1f} \u00b1 {volume_res.mean_uncertainty:.1f} m")
    else:
        lines.append(f"     Mean Ice Thickness:         {thick_stats[0]:.1f} m")
    lines.append(f"     Median Ice Thickness:       {thick_stats[1]:.1f} m")
    lines.append(f"     Ice Thickness Std Dev:      \u00b1{thick_stats[2]:.1f} m")
    lines.append(f"     Maximum Ice Thickness:      {thick_stats[3]:.1f} m")

    lines.append("   " + "-" * 77)
    lines.append("   Final Basal Shear Stress Distribution (Figure 11):")
    if bss_mean_unc is not None:
        lines.append(f"     Mean Basal Shear Stress:    {bss_stats[0]:.1f} \u00b1 {bss_mean_unc:.1f} kPa")
    else:
        lines.append(f"     Mean Basal Shear Stress:    {bss_stats[0]:.1f} kPa")
    lines.append(f"     Median Basal Shear Stress:  {bss_stats[1]:.1f} kPa")
    lines.append(f"     Basal Shear Stress Std Dev: \u00b1{bss_stats[2]:.1f} kPa")
    lines.append(f"     Maximum Basal Shear Stress: {bss_stats[3]:.1f} kPa")
    lines.append("=" * 80)

    return "\n".join(lines)


def _prompt_finalization_bool(prompt_text: str, default: bool) -> bool:
    tag = "Y/n" if default else "y/N"
    try:
        raw = input(f"{prompt_text} [{tag}]: ").strip().lower()
        if not raw:
            return default
        return raw in ["y", "yes", "true", "1"]
    except Exception:
        return default


def _prompt_finalization_float(prompt_text: str, default: float, min_val: float | None = None) -> float:
    while True:
        try:
            raw = input(f"{prompt_text} [{default}]: ").strip()
            if not raw:
                return default
            val = float(raw)
            if min_val is not None and val < min_val:
                sys.stdout.write(f"Value must be >= {min_val}. Please try again.\n")
                sys.stdout.flush()
                continue
            return val
        except ValueError:
            sys.stdout.write("Invalid numeric value. Please try again.\n")
            sys.stdout.flush()
        except Exception:
            return default


def _prompt_finalization_choice(prompt_text: str, choices: list[str], default: str) -> str:
    choices_str = "/".join(choices)
    while True:
        try:
            raw = input(f"{prompt_text} ({choices_str}) [{default}]: ").strip().lower()
            if not raw:
                return default
            if raw in choices:
                return raw
            sys.stdout.write(f"Invalid choice '{raw}'. Choose from: {choices_str}\n")
            sys.stdout.flush()
        except Exception:
            return default


def _prompt_finalization_odd_int(prompt_text: str, default: int) -> int:
    while True:
        try:
            raw = input(f"{prompt_text} [{default}]: ").strip()
            if not raw:
                return default
            val = int(raw)
            if val < 1 or val % 2 == 0:
                sys.stdout.write("Kernel size must be a positive odd integer (e.g. 3, 5, 7). Please try again.\n")
                sys.stdout.flush()
                continue
            return val
        except ValueError:
            sys.stdout.write("Invalid integer. Please try again.\n")
            sys.stdout.flush()
        except Exception:
            return default


class Solver:
    """
    Physically-Informed Bedrock Interpolation & 3D Migration Solver.

    Coordinates the complete multi-stage scientific workflow:
      1. Surface DEM ingestion, coordinate validation, and gradient precomputations.
      2. Pre-migration traveltime interpolation and 3D Eikonal normal ray migration.
      3. Spectral low-pass filtering of surface slope and Basal Shear Stress (BSS)
         variance minimization to find the optimal cutoff wavenumber k_c [rad/m].
      4. Post-migration Dual Kriging (Ordinary, Universal, or Regression Kriging)
         to interpolate ice thickness D(x, y) [m] and estimate prediction uncertainty.
      5. Topography finalization through optional depth field smoothing, Random Forest
         gap filling in unconstrained areas, and geomorphological boundary margin blending.
      6. Forward modeling of unprobed glacier survey paths via Shallow Ice Approximation (SIA).
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
        pre_kriging_method: str | None = None,
        pre_drift_terms: list | None = None,
        pre_variogram_model: str = "spherical",
        pre_zero_boundary: bool = True,
        pre_interpolation_target: str = "P",
        pre_drift_analyzer: bool = False,
        pre_calculate_uncertainty: bool = False,
        post_kriging_method: str | None = None,
        post_drift_terms: list | None = None,
        post_variogram_model: str = "spherical",
        post_zero_boundary: bool = True,
        post_interpolation_target: str = "P",
        post_drift_analyzer: bool = False,
        post_calculate_uncertainty: bool = False,
        perform_migration: bool = True,
        survey_data_type: str = "one_way_traveltime",
        horizontal_displacement_visualization: str = "arrows",
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
        max_memory_fraction: float | None = None,
        config: dict[str, Any] | None = None,
        mass_balance: str | Path | np.ndarray | BedrockMap | None = None,
        velocity: str | Path | np.ndarray | BedrockMap | None = None,
    ):
        """
        Parameters
        ----------
        dem : str, Path, or np.ndarray
            Path to surface DEM GeoTIFF/ASCII file, or 2D NumPy array of surface elevations [m].
        outline : str, Path, or np.ndarray, optional
            Path to glacier boundary polygon (Shapefile/GeoJSON) or 2D boolean mask.
        dx : float, optional
            Spatial grid spacing along the X-axis [m]. If None, derived from DEM metadata.
        dy : float, optional
            Spatial grid spacing along the Y-axis [m]. If None, derived from DEM metadata.
        bounds : tuple of float, optional
            Projected metric spatial bounding box (minx, miny, maxx, maxy) [m].
        origin : tuple of float, optional
            Lower-left coordinate origin (xll, yll) [m] for headerless DEM formats (CSV, NPY, ndarray).
        crs : str, int, or pyproj.CRS, optional
            Coordinate Reference System (e.g. 'EPSG:32633'). Enforces projected metric units.
        pre_kriging_method : str, optional
            First-pass pre-migration Kriging method: 'ordinary', 'universal' or 'regression'. If omitted: 'universal'
            for the direct target 'T', otherwise 'ordinary' (see :func:`pysole.drift.default_model`).
        pre_drift_terms : list, optional
            First-pass drift terms: None (omitted) or a list of 'sia', 'z_dem', 'curvature_dem', 'mass_balance',
            'velocity', 'velocity_sia', 'linear_xy', 'quadratic_xy' (inside an analyzer suite one nested [] adds
            the Ordinary candidate). 'universal' without analyzer needs the list (one multi-drift model); an omitted
            method with target 'T' and omitted terms uses ['sia']; 'ordinary' ignores terms and analyzer (with a
            warning); see :mod:`pysole.drift` for all rules.
        pre_variogram_model : str, default 'spherical'
            First-pass spatial variogram model: 'spherical', 'exponential', or 'gaussian'.
            Note: 'linear' is supported under engine='pykrige'; engine='native' falls back to 'spherical'.
        pre_zero_boundary : bool, default True
            If True, enforces a zero traveltime boundary condition (T = 0 ns) along the glacier margin.
        pre_interpolation_target : str, default 'P'
            Target quantity interpolated in the first pass: 'P' (BSS product P = T * sin(alpha_opt))
            or 'T' (direct traveltime in ns).
        pre_drift_analyzer : bool, default False
            If True, ranks the drift candidates of the first pass with the Drift Analyzer (needs 'universal').
        pre_calculate_uncertainty : bool, default False
            If True, calculates estimation variance during Pass 1 pre-migration Kriging.
        post_kriging_method : str, optional
            Second-pass post-migration Kriging method: 'ordinary', 'universal' or 'regression'. If omitted:
            'universal' for the direct target 'D', otherwise 'ordinary'.
        post_drift_terms : list, optional
            Second-pass drift terms; same rules as ``pre_drift_terms``.
        post_variogram_model : str, default 'spherical'
            Second-pass spatial variogram model: 'spherical', 'exponential', or 'gaussian'.
        post_zero_boundary : bool, default True
            If True, enforces a zero ice thickness boundary condition (H = 0 m) along the glacier margin.
        post_interpolation_target : str, default 'P'
            Target quantity interpolated in the second pass: 'P' (BSS product P = D * sin(alpha_opt))
            or 'D' (direct migrated ice thickness in m).
        post_drift_analyzer : bool, default False
            If True, ranks the drift candidates of the second pass with the Drift Analyzer (needs 'universal').
        post_calculate_uncertainty : bool, default False
            If True, calculates estimation variance during Pass 2 post-migration Kriging.
        perform_migration : bool, default True
            If True, performs 3D Eikonal normal ray migration on radar traveltimes. If False, skips migration.
        survey_data_type : str, default 'one_way_traveltime'
            Type of input survey data: 'one_way_traveltime' [ns], 'two_way_traveltime' [ns]
            (automatically converted via two_way_traveltime / 2.0), or 'depth' [m] (direct ice thickness measurements).
        plots_dir : str or Path, optional
            Directory where diagnostic figures and maps are saved.
        n_cores : int, default -1
            Number of CPU cores for parallelized operations (-1 uses all available CPU cores).
        kriging_engine : str, default 'native'
            Kriging solver engine: 'native' (high-performance Dual Kriging) or 'pykrige'.
        nrbins : int, optional
            Number of distance lag bins for experimental variogram calculation. If None, chosen automatically.
        ice_density : float, default 900.0
            Bulk ice density [kg/m^3] used for basal shear stress calculation and SIA depth modeling.
        g : float, default 9.81
            Gravitational acceleration [m/s^2].
        output_dir : str or Path, optional
            General workspace directory for all output raster and vector files.
        survey_data_path : str or Path, optional
            Path to survey points file; used to resolve default output paths if output_dir is unspecified.
        survey_profile_column : str, optional
            Column name identifying individual survey profiles in CSV data for LOPO spatial cross-validation.
        config_path : str or Path, optional
            Path to configuration file used for resolving relative paths.
        show_progress : bool, default True
            If True, displays terminal progress bars during computational stages.
        max_memory_fraction : float, optional
            RAM budget of the native Kriging engine as a fraction of available memory (0 < f <= 0.9).
        config : dict of str to Any, optional
            Parsed configuration dictionary if already loaded.
        mass_balance : str, Path, np.ndarray, or BedrockMap, optional
            Path to or array of decadal-mean surface mass balance raster (used by the ``'mass_balance'`` drift term
            and Random Forest gap filling).
        velocity : str, Path, np.ndarray, or BedrockMap, optional
            Path to or array of decadal-mean surface flow velocity magnitude raster (used by the ``'velocity'`` and
            ``'velocity_sia'`` drift terms and Random Forest gap filling).

        Raises
        ------
        ConfigError
            If the Kriging method, engine or drift settings are invalid or contradictory; raised before any
            file is read.
        """
        self.config = dict(config) if isinstance(config, dict) else {}
        if self.config:
            inputs_cfg = self.config.get("inputs", {}) if isinstance(self.config.get("inputs"), dict) else {}
            if mass_balance is None and inputs_cfg.get("mass_balance_path") is not None:
                mass_balance = inputs_cfg.get("mass_balance_path")
            if velocity is None and inputs_cfg.get("velocity_path") is not None:
                velocity = inputs_cfg.get("velocity_path")
            krig_cfg = self.config.get("kriging_parameters", {})
            if pre_drift_terms is None and "drift_terms" in krig_cfg.get("pre_migration", {}):
                pre_drift_terms = krig_cfg.get("pre_migration", {}).get("drift_terms")
            if post_drift_terms is None and "drift_terms" in krig_cfg.get("post_migration", {}):
                post_drift_terms = krig_cfg.get("post_migration", {}).get("drift_terms")
            if pre_kriging_method is None and "method" in krig_cfg.get("pre_migration", {}):
                pre_kriging_method = krig_cfg.get("pre_migration", {}).get("method")
            if post_kriging_method is None and "method" in krig_cfg.get("post_migration", {}):
                post_kriging_method = krig_cfg.get("post_migration", {}).get("method")
            if not pre_drift_analyzer and "drift_analyzer" in krig_cfg.get("pre_migration", {}):
                pre_drift_analyzer = krig_cfg.get("pre_migration", {}).get("drift_analyzer", False)
            if not post_drift_analyzer and "drift_analyzer" in krig_cfg.get("post_migration", {}):
                post_drift_analyzer = krig_cfg.get("post_migration", {}).get("drift_analyzer", False)
            if not pre_calculate_uncertainty and "calculate_uncertainty" in krig_cfg.get("pre_migration", {}):
                pre_calculate_uncertainty = krig_cfg.get("pre_migration", {}).get("calculate_uncertainty", False)
            if not post_calculate_uncertainty and "calculate_uncertainty" in krig_cfg.get("post_migration", {}):
                post_calculate_uncertainty = krig_cfg.get("post_migration", {}).get("calculate_uncertainty", False)

        self.mass_balance_grid: np.ndarray | None = mass_balance  # type: ignore[assignment]
        self.velocity_grid: np.ndarray | None = velocity  # type: ignore[assignment]

        # Validate the Kriging drift configuration first: nothing is read or computed for an invalid setup.
        self.engine_type = normalize_engine(kriging_engine)
        self.pre_interpolation_target = str(pre_interpolation_target).upper().strip()
        self.post_interpolation_target = str(post_interpolation_target).upper().strip()
        self.pre_drift_analyzer = bool(pre_drift_analyzer)
        self.post_drift_analyzer = bool(post_drift_analyzer)
        pre_method, pre_terms = default_model(
            pre_kriging_method, pre_drift_terms, self.pre_interpolation_target, "T", self.pre_drift_analyzer
        )
        post_method, post_terms = default_model(
            post_kriging_method, post_drift_terms, self.post_interpolation_target, "D", self.post_drift_analyzer
        )
        self.pre_kriging_method = normalize_method(pre_method)
        self.pre_drift_terms = pre_terms
        self.post_kriging_method = normalize_method(post_method)
        self.post_drift_terms = post_terms
        pre_plan = self._drift_plan("pre_migration")
        post_plan = self._drift_plan("post_migration")
        self._log_drift_plan("Pass 1 - Constructing Traveltime Grid", pre_plan)
        self._log_drift_plan("Pass 2 - Constructing Depth Grid", post_plan)
        if self.pre_kriging_method == "universal" and not pre_plan.analyze:
            self.pre_drift_analyzer = False
        if self.post_kriging_method == "universal" and not post_plan.analyze:
            self.post_drift_analyzer = False

        self.output_dir = output_dir
        self.survey_data_path = survey_data_path
        self.survey_profile_column = survey_profile_column
        self.is_batch_mode = False  # set by the pipeline for `--batch`: no interactive prompts
        self.config_path = str(config_path) if config_path else None
        self._raw_plots_dir = plots_dir
        self._max_memory_fraction = None if max_memory_fraction is None else validate_memory_fraction(max_memory_fraction, "max_memory_fraction")

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

        self.mass_balance_grid = None
        if mass_balance is not None:
            mb_input = self.resolve_input_path(mass_balance) if isinstance(mass_balance, (str, Path, os.PathLike)) else mass_balance
            self.mass_balance_grid = load_aligned_raster(
                mb_input,
                geometry=self.geometry,
                ref_meta=self.meta,
                name="mass_balance",
            )
        self.velocity_grid = None
        if velocity is not None:
            vel_input = self.resolve_input_path(velocity) if isinstance(velocity, (str, Path, os.PathLike)) else velocity
            vel_grid = load_aligned_raster(
                vel_input,
                geometry=self.geometry,
                ref_meta=self.meta,
                name="velocity",
            )
            vel_grid = np.maximum(vel_grid, 0.0)
            if self.outline_mask is not None:
                vel_grid = vel_grid.copy()
                vel_grid[~self.outline_mask] = 0.0
            self.velocity_grid = vel_grid

        # Migration & Kriging options (Pre-migration & Post-migration)
        self.pre_variogram_model = pre_variogram_model
        self.pre_zero_boundary = bool(pre_zero_boundary)
        self.pre_calculate_uncertainty = bool(pre_calculate_uncertainty)

        self.post_variogram_model = post_variogram_model
        self.post_zero_boundary = bool(post_zero_boundary)
        self.post_calculate_uncertainty = bool(post_calculate_uncertainty)

        self.perform_migration = perform_migration
        self.survey_data_type = survey_data_type
        self.horizontal_displacement_visualization = horizontal_displacement_visualization
        self.n_cores = n_cores

        # "linear" is a 1-D model: under the native engine it is replaced by "spherical" once, here,
        # so the warning is logged a single time and every downstream consumer (kriging passes and
        # the drift analyzer) sees a valid 2-D model. engine="pykrige" keeps native linear support.
        pre_was_linear = str(self.pre_variogram_model).lower().strip() == "linear"
        self.pre_variogram_model = self._resolve_variogram_for_engine(
            self.pre_variogram_model, self.pre_kriging_method
        )
        self.post_variogram_model = self._resolve_variogram_for_engine(
            self.post_variogram_model, self.post_kriging_method, warn=not pre_was_linear
        )
        self.nrbins = int(nrbins) if nrbins is not None else None
        self.ice_density = float(ice_density)
        self.g = float(g)
        self.show_progress = bool(show_progress)

        # Gradient, DEM, Velocity, Feature, and Sample Points Caches
        self._gradient_cache: dict[str, np.ndarray] | None = None
        self._fft_dem_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._fft_dem_pad_kc: float = float("inf")  # k_c the cached FFT padding was sized for
        self._smoothed_dem_cache: dict[float, np.ndarray] = {}
        self._fft_vel_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._fft_vel_pad_kc: float = float("inf")
        self._smoothed_vel_cache: dict[float, np.ndarray] = {}
        self._sample_pts_cache: dict[tuple[str, str], np.ndarray] = {}

        # Decoupled sub-engines strictly bound to GridGeometry
        self.migrator = EikonalMigrator(self.dem_grid, geometry=self.geometry, outline_mask=self.outline_mask)
        self.bss_optimizer = BSSOptimizer(self.dem_grid, geometry=self.geometry)
        self.kriging_engine = KrigingEngine(self.dem_grid, geometry=self.geometry, outline_mask=self.outline_mask)
        self.finalizer = BedrockFinalizer(self.dem_grid, geometry=self.geometry, outline_mask=self.outline_mask)

        # Internal state
        self.survey_points: np.ndarray | None = None
        self.pre_kriging_points: np.ndarray | None = None
        self.migrated_points: np.ndarray | None = None
        self._traveltime_grid: np.ndarray | None = None
        self._traveltime_std: np.ndarray | None = None
        self.opt_kc: float | None = None
        self.opt_slope: np.ndarray | None = None
        self.pre_opt_res: OptimizationResult | None = None
        self.post_opt_res: OptimizationResult | None = None
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
        self.volume_result: IceVolumeResult | None = None
        self.glacier_area_m2: float | None = None
        self.glacier_area_km2: float | None = None
        self.total_volume_m3: float | None = None
        self.total_volume_km3: float | None = None
        self.total_volume_unc_m3: float | None = None
        self.total_volume_unc_km3: float | None = None
        self.opt_variogram_params: dict[str, float] | None = None
        self.pre_selected_drift_models: list[CandidateDriftResult] = []
        self.post_selected_drift_models: list[CandidateDriftResult] = []
        self.pre_drift_comparison_records: list[DriftComparisonRecord] = []
        self.post_drift_comparison_records: list[DriftComparisonRecord] = []
        if config is not None:
            # Unknown (e.g. outdated) keys are dropped with one warning; see config.sanitize_config
            self.config, _unknown_keys = sanitize_config(config)
            warn_unknown_config_keys(_unknown_keys)
        elif not hasattr(self, "config") or self.config is None:
            self.config = {}
        if self.config_path and Path(self.config_path).exists() and not self.config:
            try:
                from .config import load_config
                # Keep an already-configured logger level (e.g. CLI --verbose/--debug set by run_from_config)
                active_level = logger.level if (logger.handlers and logger.level) else None
                self.config = load_config(self.config_path, log_level=active_level)
            except Exception:
                pass

    @property
    def traveltime_grid(self) -> np.ndarray | None:
        """Reconstructed pre-migration one-way radar traveltime grid T(x, y) [ns]."""
        return self._traveltime_grid

    @traveltime_grid.setter
    def traveltime_grid(self, value: np.ndarray | None) -> None:
        self._traveltime_grid = value

    @property
    def traveltime_std_grid(self) -> np.ndarray | None:
        """Pre-migration traveltime Kriging standard error uncertainty grid sigma_T(x, y) [ns]."""
        return self._traveltime_std

    @property
    def opt_wavelength(self) -> float | None:
        """Optimal physical spatial cutoff wavelength lambda_c = 2*pi / k_c [m]."""
        if self.opt_kc is None:
            return None
        return compute_cutoff_wavelength(self.opt_kc)

    @property
    def thickness_grid(self) -> np.ndarray | None:
        """Final predicted ice thickness grid D(x, y) [m]."""
        return self.final_thickness if self.final_thickness is not None else self.kriged_thickness

    @property
    def thickness_std_grid(self) -> np.ndarray | None:
        """Final ice thickness Kriging standard error uncertainty grid sigma_D(x, y) [m]."""
        return self.kriged_std

    @property
    def basal_shear_stress_grid(self) -> np.ndarray | None:
        """Final basal shear stress grid tau_b(x, y) = rho_ice * g * D * sin(alpha_opt) / 1000 [kPa]."""
        return self.final_bss

    @property
    def basal_shear_stress_std_grid(self) -> np.ndarray | None:
        """Final basal shear stress Kriging standard error uncertainty grid sigma_tau_b(x, y) [kPa]."""
        return self.bss_std

    @property
    def outputs_config(self) -> dict[str, Any]:
        """Dictionary of output export configuration parameters."""
        return self.config.get("outputs", {})

    @property
    def outputs_config_obj(self) -> OutputsConfig:
        """Structured OutputsConfig dataclass instance for type-safe export options."""
        return OutputsConfig.from_dict(self.outputs_config)

    @property
    def optimization_config(self) -> dict[str, Any]:
        """Dictionary of spectral slope optimization parameters."""
        return self.config.get("optimization_parameters", {})

    @property
    def slope_floor_deg(self) -> float:
        """Minimum surface slope angle threshold in degrees [°] for stabilizing numerical divisions."""
        return float(self.optimization_config.get("slope_floor_deg", 5.0))

    def _available_optional_terms(self) -> list[str]:
        """Returns the list of optional raster drift terms available on this Solver instance."""
        return resolve_available_optional_terms(
            has_mass_balance=self.mass_balance_grid is not None,
            has_velocity=self.velocity_grid is not None,
        )

    def _drift_plan(self, stage: str, method: str | None = None) -> DriftPlan:
        """Resolves the drift plan of a stage from the current method / drift_terms / drift_analyzer settings.

        Parameters
        ----------
        stage : str
            ``'pre_migration'`` or ``'post_migration'``.
        method : str, optional
            Method that replaces the configured one of the stage (e.g. from ``calculate_bedrock(method=...)``).

        Returns
        -------
        plan : DriftPlan
            Resolved plan; its ``notices`` are not logged here (see :meth:`_log_drift_plan`).

        Raises
        ------
        ConfigError
            If the combination of method, drift_terms and drift_analyzer is invalid, or if a requested
            optional raster drift term lacks its input raster.
        """
        prefix = "pre" if stage == "pre_migration" else "post"
        avail_opt = self._available_optional_terms()
        plan = resolve_drift_plan(
            method if method is not None else getattr(self, f"{prefix}_kriging_method"),
            getattr(self, f"{prefix}_drift_analyzer"),
            getattr(self, f"{prefix}_drift_terms"),
            self.engine_type,
            available_optional_terms=avail_opt,
        )
        if plan.method != "ordinary":
            validate_required_drift_rasters(
                plan.terms if plan.terms else plan.suite,
                has_mass_balance="mass_balance" in avail_opt,
                has_velocity="velocity" in avail_opt,
            )
        return plan

    @staticmethod
    def _log_drift_plan(stage_label: str, plan: DriftPlan) -> None:
        """Logs the notices (overrides, skipped analyzer) of a drift plan once."""
        for level, message in plan.notices:
            getattr(logger, level)(f"   [Dual Kriging Engine] [{stage_label}] {message}")

    def _resolve_variogram_for_engine(self, model: str, method: str, warn: bool = True) -> str:
        """Resolves variogram model compatibility for the selected Kriging engine.

        Replaces 1D 'linear' variograms with 'spherical' when using the native 2D Dual
        Kriging engine, logging a single notification. PyKrige supports 'linear' natively.

        Parameters
        ----------
        model : str
            Requested variogram model ('spherical', 'exponential', 'gaussian', 'linear').
        method : str
            Kriging method ('ordinary', 'universal', 'regression').
        warn : bool, default True
            If True, logs a warning when falling back from linear to spherical model.

        Returns
        -------
        resolved_model : str
            Engine-compatible variogram model name.
        """
        engine = self.engine_type.replace("_kriging", "")
        is_regression = str(method).lower().strip() in ("regression", "regression_kriging")
        if engine == "native" and not is_regression:
            return resolve_native_variogram_model(model, warn=warn)
        return model

    def _slope_source(self) -> np.ndarray:
        """Returns the active surface slope angle grid [rad].

        Uses the spectrally optimized surface slope if optimization has been performed;
        otherwise falls back to raw numerical surface DEM gradients.

        Returns
        -------
        slope_grid : np.ndarray
            2D array of surface slope angles in radians [rad].
        """
        if self.opt_slope is not None:
            return self.opt_slope
        return self._get_dem_gradients()["slope_rad"]

    @property
    def slope_sin(self) -> np.ndarray:
        """Raw sin(alpha_opt) grid without lower threshold bounding.

        Used strictly for basal shear stress calculation (tau_b = rho * g * D * sin(alpha)),
        where artificially flooring slopes in flat valley floors would introduce a positive bias.

        Returns
        -------
        np.ndarray
            2D array of unfloored sine values sin(alpha_opt).
        """
        return np.sin(self._slope_source())

    @property
    def safe_slope_sin(self) -> np.ndarray:
        """Stabilized sin(alpha_opt) grid bounded below by sin(slope_floor_deg).

        Required for conversions between basal shear stress product P(x, y) and physical
        traveltime T(x, y) or ice thickness D(x, y) (which require division by sin(alpha)).
        Cached and invalidated only when the underlying slope array changes.

        Returns
        -------
        np.ndarray
            2D array of floored sine values max(sin(alpha), sin(slope_floor_deg)).
        """
        src = self._slope_source()
        key = (id(src), self.slope_floor_deg)
        cache = getattr(self, "_safe_slope_sin_cache", None)
        if cache is not None and cache[0] == key and cache[1] is src:
            return cache[2]
        min_slope_sin = np.sin(np.radians(self.slope_floor_deg))
        result = np.maximum(np.sin(src), min_slope_sin)
        self._safe_slope_sin_cache = (key, src, result)
        return result

    def _cfg_get(self, section: str, key: str, default: Any = None) -> Any:
        """Safely retrieves a configuration parameter from a nested dictionary section.

        Parameters
        ----------
        section : str
            Top-level configuration section key (e.g. 'inputs', 'spatial_parameters').
        key : str
            Parameter name inside the section.
        default : Any, optional
            Fallback value if the section or key does not exist.

        Returns
        -------
        value : Any
            Configured parameter value or fallback default.
        """
        sec = self.config.get(section, {})
        if isinstance(sec, dict):
            val = sec.get(key)
            return val if val is not None else default
        return default

    @property
    def bedrock_map(self) -> BedrockMap | None:
        """Returns the final predicted bedrock elevation grid wrapped as a BedrockMap object.

        Returns
        -------
        BedrockMap or None
            Bedrock elevation map [m] with spatial georeferencing metadata, or None if not yet computed.
        """
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
        """Dictionary containing all final and intermediate spatial grids and parameters.

        Returns
        -------
        results_dict : dict of str to Any
            Dictionary containing:
              - 'bedrock_map' : BedrockMap of final bedrock elevation [m].
              - 'thickness_grid' : 2D array of final ice thickness [m].
              - 'thickness_std_grid' : 2D array of ice thickness standard error [m].
              - 'basal_shear_stress_grid' : 2D array of basal shear stress [kPa].
              - 'basal_shear_stress_std_grid' : 2D array of BSS standard error [kPa].
              - 'migrated_points' : (N, 4+) array of 3D migrated coordinates [m].
              - 'opt_kc' : Optimal smoothing corner wavenumber [rad/m].
              - 'opt_wavelength' : Optimal smoothing cutoff wavelength [m].
              - 'kriged_bedrock' : 2D array of pre-blended Kriged bedrock elevation [m].
              - 'rf_filled_bedrock' : 2D array of Random Forest gap-filled bedrock [m].
              - 'blended_bedrock' : 2D array of margin-blended bedrock [m].
        """
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
        """Formats and logs a comprehensive execution summary report of the Solver state.

        Returns
        -------
        summary_text : str
            Formatted multi-line execution summary report string.
        """
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
            f"  Pre-Migration Target : {self._drift_plan('pre_migration').describe()} (Target: '{self.pre_interpolation_target}')",
            f"  Post-Migration Target: {self._drift_plan('post_migration').describe()} (Target: '{self.post_interpolation_target}')",
        ]
        if self.final_thickness is not None:
            mask_h = (self.outline_mask & (self.final_thickness > 0)) if (self.outline_mask is not None and np.any(self.outline_mask)) else (self.final_thickness > 0)
            valid_h = self.final_thickness[mask_h & np.isfinite(self.final_thickness)]
            if len(valid_h) > 0:
                lines.append(f"  Ice Thickness Range  : min={np.min(valid_h):.1f} m, mean={np.mean(valid_h):.1f} m, max={np.max(valid_h):.1f} m")
        if self.final_bss is not None:
            mask_tau = (self.outline_mask & (self.final_bss > 0)) if (self.outline_mask is not None and np.any(self.outline_mask)) else (self.final_bss > 0)
            valid_tau = self.final_bss[mask_tau & np.isfinite(self.final_bss)]
            if len(valid_tau) > 0:
                lines.append(f"  Basal Shear Stress   : min={np.min(valid_tau):.1f} kPa, mean={np.mean(valid_tau):.1f} kPa, max={np.max(valid_tau):.1f} kPa")
        lines.append("================================================================================")
        summary_text = "\n".join(lines)
        logger.info(summary_text)
        return summary_text

    @property
    def figure_prefix(self) -> str:
        """Returns clean alphanumeric prefix stem for diagnostic figures (e.g. 'apo' or '')."""
        raw = (self.outputs_config_obj.output_prefix or "").strip()
        if not raw:
            return ""
        stem = Path(raw).name
        # Strip common extensions if entered by user
        for ext in [".tif", ".tiff", ".asc", ".csv", ".npy", ".geotiff"]:
            if stem.lower().endswith(ext):
                stem = stem[:-len(ext)]
        return stem

    def _get_applied_drift_slug(self, stage_key: str) -> str:
        """Resolves the canonical filesystem slug for the applied drift model of a stage."""
        stage_norm = "pre_migration" if ("pre" in stage_key.lower() or "1" in stage_key) else "post_migration"
        selected = self.pre_selected_drift_models if stage_norm == "pre_migration" else self.post_selected_drift_models
        if selected:
            return slugify_drift_name(selected[0].terms)
        plan = self._drift_plan(stage_norm)
        if plan.method == "universal":
            return slugify_drift_name(plan.terms)
        return "ordinary"

    def _get_output_filepath(self, suffix: str, ext: str = "", model_slug: str | None = None) -> str:
        """Constructs an output filepath incorporating optional prefix, clean suffix, and applied drift model slug.

        Parameters
        ----------
        suffix : str
            Dataset identifier (e.g. 'bedrock', 'thickness', 'migrated_points').
        ext : str, optional
            File extension (e.g. '.csv'). If omitted, no extension is appended (suitable for raster save).
        model_slug : str or None, optional
            Applied drift model slug appended to the filename (e.g. 'sia', 'z_dem', 'ordinary').

        Returns
        -------
        filepath : str
            Resolved filesystem path for output export.
        """
        raw_prefix = (self.outputs_config_obj.output_prefix or "").strip()
        ext_clean = f".{ext.lstrip('.')}" if ext else ""
        full_suffix = f"{suffix}_{model_slug}" if model_slug else suffix

        if not raw_prefix:
            return self.resolve_path(f"{full_suffix}{ext_clean}")

        prefix_path = Path(raw_prefix)
        for e in [".tif", ".tiff", ".asc", ".csv", ".npy", ".geotiff"]:
            if prefix_path.suffix.lower() == e:
                prefix_path = prefix_path.with_suffix("")
                break

        parent = prefix_path.parent
        stem = prefix_path.name.rstrip("_")
        filename = f"{stem}_{full_suffix}{ext_clean}" if stem else f"{full_suffix}{ext_clean}"
        target = parent / filename if str(parent) != "." else Path(filename)
        return self.resolve_path(str(target))

    def _get_resolved_output_prefix(self) -> str:
        """Resolves the base file path prefix for exporting raster and tabular outputs.

        Returns
        -------
        resolved_prefix : str
            Base filesystem path prefix without extension.
        """
        raw_prefix = (self.outputs_config_obj.output_prefix or "").strip()
        if not raw_prefix:
            eff_dir = Path(self.resolve_path(".") or ".").resolve()
            return str(eff_dir) + os.sep

        prefix_path = Path(raw_prefix)
        for ext in [".tif", ".tiff", ".asc", ".csv", ".npy", ".geotiff"]:
            if prefix_path.suffix.lower() == ext:
                prefix_path = prefix_path.with_suffix("")
                break

        return self.resolve_path(str(prefix_path))

    def _export_optional_raster(
        self, grid: np.ndarray | None, suffix: str, name: str, model_slug: str | None = None
    ) -> str | list[str] | None:
        """Exports a 2D spatial grid to raster files if the grid is populated.

        Parameters
        ----------
        grid : np.ndarray or None
            2D array to export.
        suffix : str
            Filename suffix appended to base output prefix (e.g. 'thickness', 'bedrock').
        name : str
            Human-readable dataset name for logging.
        model_slug : str or None, optional
            Applied drift model slug appended to the filename.

        Returns
        -------
        saved_paths : str, list of str, or None
            Path(s) to saved raster file(s), or None if grid is empty.
        """
        if grid is None:
            return None
        filepath = self._get_output_filepath(suffix, model_slug=model_slug)
        fmt = self.outputs_config_obj.output_format
        raster = BedrockMap(grid=grid, bounds=self.bounds, crs=self.meta.get("crs"), transform=self.meta.get("transform"), name=name)
        saved = raster.save(filepath, formats=fmt)
        if isinstance(saved, list):
            for sf in saved:
                logger.info(f"   Saved {name} grid to: {sf}")
        else:
            logger.info(f"   Saved {name} grid to: {saved}")
        return saved

    def _export_optional_points_csv(self, points: np.ndarray | None, suffix: str) -> str | None:
        """Exports point coordinate data to CSV format.

        Parameters
        ----------
        points : np.ndarray or None
            Coordinate matrix (N, M) to export.
        suffix : str
            Filename suffix appended to base output prefix.

        Returns
        -------
        saved_path : str or None
            Path to exported CSV file, or None if points array is empty.
        """
        if points is None:
            return None
        filepath = self._get_output_filepath(suffix, ext=".csv")
        saved = save_points_csv(points, filepath)
        logger.info(f"   Saved migrated survey points to: {saved}")
        return saved

    @staticmethod
    def _uncertainty_exportable(grid: np.ndarray | None, name: str) -> bool:
        """Determines whether an uncertainty grid contains valid finite numerical values.

        Parameters
        ----------
        grid : np.ndarray or None
            Uncertainty grid to test.
        name : str
            Dataset name for user notification logging.

        Returns
        -------
        is_exportable : bool
            True if the grid exists and contains at least one finite number; False otherwise.
        """
        if grid is not None and np.any(np.isfinite(grid)):
            return True
        logger.warning(
            f"   [Export Skipped] The {name} raster is not exported: Kriging uncertainty was not calculated "
            "(kriging_parameters.*.calculate_uncertainty=false, skipped by the memory guard, or not supported by method)."
        )
        return False

    def export_outputs(
        self, stage: str | None = None, model_slug: str | None = None, export_points: bool = True
    ) -> list[str]:
        """
        Exports optional spatial datasets according to configured boolean output flags.

        Parameters
        ----------
        stage : str, optional
            Workflow stage name ('migration', 'finalization', or None to export all stages).
        model_slug : str or None, optional
            Applied drift model slug appended to the filename. If None, automatically resolved
            from the active stage's applied drift model.
        export_points : bool, optional
            If True (default), exports optional point datasets (such as migrated survey points).
            If False, point dataset export is skipped (e.g. during secondary candidate comparison runs).

        Returns
        -------
        saved_files : list of str
            Paths of saved output files.
        """
        saved: list[str] = []
        cfg_out = self.outputs_config_obj

        if stage is None or stage == "migration":
            m_slug = model_slug if model_slug is not None else self._get_applied_drift_slug("pre_migration")
            if cfg_out.save_traveltime_grid and self._traveltime_grid is not None:
                res = self._export_optional_raster(
                    self._traveltime_grid, suffix="traveltime", name="traveltime", model_slug=m_slug
                )
                if res:
                    saved.extend([res] if isinstance(res, str) else res)
            if cfg_out.save_traveltime_uncertainty:
                if self._uncertainty_exportable(self._traveltime_std, "traveltime uncertainty"):
                    res = self._export_optional_raster(
                        self._traveltime_std, suffix="traveltime_uncertainty", name="traveltime uncertainty", model_slug=m_slug
                    )
                    if res:
                        saved.extend([res] if isinstance(res, str) else res)
            if export_points and cfg_out.save_migrated_points and self.migrated_points is not None:
                # Migrated points are always saved to root results/ without model slug
                res = self._export_optional_points_csv(self.migrated_points, suffix="migrated_points")
                if res:
                    saved.append(res)

        if stage is None or stage == "finalization":
            f_slug = model_slug if model_slug is not None else self._get_applied_drift_slug("post_migration")
            if cfg_out.save_thickness_grid and self.final_thickness is not None:
                res = self._export_optional_raster(
                    self.final_thickness, suffix="thickness", name="thickness", model_slug=f_slug
                )
                if res:
                    saved.extend([res] if isinstance(res, str) else res)
            if cfg_out.save_thickness_uncertainty:
                if self._uncertainty_exportable(self.kriged_std, "thickness uncertainty"):
                    res = self._export_optional_raster(
                        self.kriged_std, suffix="thickness_uncertainty", name="thickness uncertainty", model_slug=f_slug
                    )
                    if res:
                        saved.extend([res] if isinstance(res, str) else res)
            if cfg_out.save_basal_shear_stress and self.final_bss is not None:
                res = self._export_optional_raster(
                    self.final_bss, suffix="basal_shear_stress", name="basal shear stress", model_slug=f_slug
                )
                if res:
                    saved.extend([res] if isinstance(res, str) else res)
            if cfg_out.save_basal_shear_stress_uncertainty:
                if self._uncertainty_exportable(self.bss_std, "basal shear stress uncertainty"):
                    res = self._export_optional_raster(
                        self.bss_std, suffix="basal_shear_stress_uncertainty", name="basal shear stress uncertainty", model_slug=f_slug
                    )
                    if res:
                        saved.extend([res] if isinstance(res, str) else res)
            if getattr(cfg_out, "save_bedrock_grid", True) and self.final_grid is not None:
                res = self._export_optional_raster(
                    self.final_grid, suffix="bedrock", name="bedrock", model_slug=f_slug
                )
                if res:
                    saved.extend([res] if isinstance(res, str) else res)

        return saved

    def clear_intermediate_grids(self) -> None:
        """Clears intermediate 2D memory arrays to reduce RAM footprint.

        Frees intermediate traveltime, Kriged thickness, variance, and basal shear stress grids
        when processing exceptionally large spatial grids.
        """
        self._traveltime_grid = None
        self._traveltime_std = None
        self.kriged_thickness = None
        self.kriged_variance = None
        self.kriged_std = None
        self.final_bss = None
        self.bss_std = None

    @property
    def effective_output_dir(self) -> str:
        """Returns the resolved filesystem directory where output files are saved.

        Returns
        -------
        output_directory : str
            Resolved absolute directory path string.
        """
        cfg_p = getattr(self, "config_path", None)
        eff_dir = resolve_output_dir(
            output_dir=self.output_dir,
            survey_data_path=self.survey_data_path,
            config_path=cfg_p,
        )
        return str(eff_dir)

    def resolve_input_path(self, path: str | Path | os.PathLike | None) -> str | None:
        """Resolves an input file path relative to the configuration file or working directory.

        Parameters
        ----------
        path : str, Path, os.PathLike, or None
            Raw input path specification.

        Returns
        -------
        resolved_path : str or None
            Absolute path string, or None if path is None.
        """
        cfg_p = getattr(self, "config_path", None)
        return resolve_input_config_path(path, config_path=cfg_p)

    def resolve_path(self, path: str | Path | os.PathLike | None) -> str | None:
        """Resolves an output file or directory path relative to effective_output_dir.

        Parameters
        ----------
        path : str, Path, os.PathLike, or None
            Raw target output path specification.

        Returns
        -------
        resolved_path : str or None
            Absolute path string, or None if path is None.
        """
        cfg_p = getattr(self, "config_path", None)
        return resolve_config_path(path, output_dir=self.output_dir, survey_data_path=self.survey_data_path, config_path=cfg_p)

    @property
    def plots_dir(self) -> str:
        """Directory where diagnostic figures and maps are saved.

        Absolute paths are preserved as-is; relative paths are resolved next to the primary
        data outputs in a 'figures' subfolder.

        Returns
        -------
        directory_path : str
            Resolved absolute path to figures directory.
        """
        out_cfg = (getattr(self, "config", None) or {}).get("outputs", {})
        return resolve_plots_dir(
            self._raw_plots_dir,
            output_prefix=out_cfg.get("output_prefix"),
            output_dir=self.output_dir,
            survey_data_path=self.survey_data_path,
            config_path=getattr(self, "config_path", None),
        )

    @plots_dir.setter
    def plots_dir(self, value: str | Path | None) -> None:
        self._raw_plots_dir = value

    @property
    def plot_extent(self) -> list[float]:
        """Returns Matplotlib bounding box extent [minx, maxx, miny, maxy] in projected meters [m].

        Returns
        -------
        extent : list of float
            Four-element list [minx, maxx, miny, maxy] in projected meters [m].
        """
        return self.geometry.extent

    def _get_dem_gradients(self) -> dict[str, np.ndarray]:
        """Calculates and caches spatial derivatives and slope angles of the raw surface DEM.

        Returns
        -------
        gradient_dict : dict of str to np.ndarray
            Dictionary containing:
              - 'dz_dx' : Partial derivative along X [m/m].
              - 'dz_dy' : Partial derivative along Y [m/m].
              - 'slope_rad' : Surface slope angle [rad].
              - 'aspect_rad' : Aspect direction [rad].
        """
        if self._gradient_cache is None:
            self._gradient_cache = compute_gradients(self.dem_grid, dx=self.dx, dy=self.dy)
        return self._gradient_cache

    def _get_fft_smoothed_grid(
        self,
        grid: np.ndarray,
        kc: float,
        fft_cache_attr: str,
        pad_kc_attr: str,
        smoothed_cache: dict[float, np.ndarray],
        clamp_non_negative: bool = False,
    ) -> np.ndarray:
        """Returns a cached 2D Gaussian low-pass filtered raster grid at cutoff wavenumber ``kc`` [rad/m]."""
        kc_key = round(float(kc), 6)
        if kc_key not in smoothed_cache:
            kc_pad = 0.01 if (kc is None or not kc > 0) else min(float(kc), 0.01)
            fft_cache = getattr(self, fft_cache_attr)
            pad_kc = getattr(self, pad_kc_attr)
            if fft_cache is None or kc_pad < pad_kc:
                A_shift, k_grid, _ = precompute_fft_grid(grid, dx=self.dx, dy=self.dy, kc=kc_pad)
                fft_cache = (A_shift, k_grid)
                setattr(self, fft_cache_attr, fft_cache)
                setattr(self, pad_kc_attr, kc_pad)
            A_shift, k_grid = fft_cache
            smoothed = fft_gaussian_smooth_precomputed(A_shift, k_grid, kc=kc)
            if clamp_non_negative:
                smoothed = np.maximum(smoothed, 0.0)
            smoothed_cache[kc_key] = smoothed
        return smoothed_cache[kc_key]

    def get_smoothed_dem(self, kc: float) -> np.ndarray:
        """Returns the 2D surface DEM smoothed with a 2D Gaussian low-pass filter at cutoff k_c.

        Caches filtered elevation grids in memory to avoid redundant inverse FFT operations.

        Parameters
        ----------
        kc : float
            Spatial cutoff wavenumber k_c [rad/m] (lambda_c = 2*pi / k_c).

        Returns
        -------
        smoothed_dem : np.ndarray
            2D array of smoothed surface elevations [m].
        """
        return self._get_fft_smoothed_grid(
            self.dem_grid, kc, "_fft_dem_cache", "_fft_dem_pad_kc", self._smoothed_dem_cache
        )

    def get_smoothed_slope(self, kc: float) -> np.ndarray:
        """Calculates the surface slope angle grid from a DEM smoothed at cutoff wavenumber k_c.

        Parameters
        ----------
        kc : float
            Spatial cutoff wavenumber k_c [rad/m].

        Returns
        -------
        slope_rad : np.ndarray
            2D array of filtered surface slope angles in radians [rad].
        """
        smoothed_dem = self.get_smoothed_dem(kc)
        return compute_gradients(smoothed_dem, dx=self.dx, dy=self.dy)["slope_rad"]

    def get_smoothed_curvature(self, kc: float) -> np.ndarray:
        """Calculates surface Laplacian curvature from a DEM smoothed at cutoff wavenumber k_c.

        Parameters
        ----------
        kc : float
            Spatial cutoff wavenumber k_c [rad/m].

        Returns
        -------
        curvature : np.ndarray
            2D array of surface Laplacian curvature values [m^-1].
        """
        smoothed_dem = self.get_smoothed_dem(kc)
        return compute_surface_curvature(smoothed_dem, dx=self.dx, dy=self.dy)

    def get_smoothed_velocity(self, kc: float) -> np.ndarray:
        """Returns the 2D surface flow velocity magnitude smoothed with a 2D Gaussian low-pass filter at cutoff k_c.

        Parameters
        ----------
        kc : float
            Spatial cutoff wavenumber k_c [rad/m] (lambda_c = 2*pi / k_c).

        Returns
        -------
        smoothed_vel : np.ndarray
            2D array of non-negative spectrally smoothed surface flow velocities.
        """
        if self.velocity_grid is None:
            raise ValueError("Cannot compute smoothed velocity: no velocity raster was provided.")
        return self._get_fft_smoothed_grid(
            self.velocity_grid, kc, "_fft_vel_cache", "_fft_vel_pad_kc", self._smoothed_vel_cache, clamp_non_negative=True
        )

    def get_drift_grids(self, drift_terms: list[str] | tuple[str, ...] | set[str] | None) -> dict[str, np.ndarray]:
        """Builds the raster covariates of the requested drift terms (single source for analyzer and production).

        ``sia`` uses the optimised smoothed slope floored at ``slope_floor_deg``, ``z_dem`` the surface DEM,
        ``curvature_dem`` the Laplacian curvature of the DEM smoothed at the optimal ``k_c``, ``mass_balance``
        the unsmoothed decadal-mean surface mass balance raster, and ``velocity`` / ``velocity_sia`` the
        off-glacier zero-clamped surface flow velocity magnitude smoothed at the optimal ``k_c``.

        Parameters
        ----------
        drift_terms : list, tuple or set of str, optional
            Primitive drift terms; only the raster terms produce a grid.

        Returns
        -------
        grids : dict of str to np.ndarray
            Raster covariate grids keyed by term name (empty if no raster term is requested).
        """
        provided: dict[str, np.ndarray] = {}
        kc_use = self.opt_kc if self.opt_kc is not None else 0.05
        if drift_terms and "curvature_dem" in drift_terms:
            provided["curvature_dem"] = self.get_smoothed_curvature(kc_use)
        vel_grid_for_drift: np.ndarray | None = None
        if drift_terms and ("velocity" in drift_terms or "velocity_sia" in drift_terms):
            if self.velocity_grid is not None:
                vel_grid_for_drift = self.get_smoothed_velocity(kc_use)
        return build_drift_grids(
            drift_terms,
            geometry=self.geometry,
            dem_grid=self.dem_grid,
            opt_slope_grid=self.opt_slope,
            slope_floor_deg=self.slope_floor_deg,
            mass_balance_grid=self.mass_balance_grid,
            velocity_grid=vel_grid_for_drift,
            provided=provided,
        )

    def _normalize_survey_data_type(self) -> str:
        """Normalizes and validates survey_data_type into one of 3 canonical formats.

        Returns
        -------
        normalized_type : str
            One of 'one_way_traveltime', 'two_way_traveltime', or 'depth'.

        Raises
        ------
        ValueError
            If survey_data_type is not recognized.
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
        drift_analyzer: bool | None = None,
        config_file: str | Path | os.PathLike | None = None,
    ) -> "Solver":
        """Instantiates and configures a Solver pipeline from a JSON configuration file.

        Parses all input paths, spatial parameters, physical constants, migration options,
        variogram specifications, and output preferences defined in pysole.json.

        Parameters
        ----------
        config_path : str, Path, os.PathLike, or dict of str to Any, default 'pysole.json'
            Filesystem path to the JSON configuration file, or pre-loaded dictionary.
        log_level : str, optional
            Logging level override ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        drift_analyzer : bool, optional
            If True, enables the drift analyzer of both Kriging stages (the ``--drift-analyzer`` command-line
            flag); it needs method 'universal'. None keeps the configured values.
        config_file : str, Path, or os.PathLike, optional
            Explicit filesystem path to configuration file when config_path is a preloaded dictionary.

        Returns
        -------
        solver : Solver
            Fully initialized Solver instance ready for execution via `run_pipeline()`
            or individual stage method calls.

        Raises
        ------
        ValueError
            If required configuration keys (such as 'dem_path') are missing.
        ConfigError
            If the configuration is invalid, e.g. contradictory Kriging method / drift_terms settings.
        """
        from .config import load_config
        cfg = load_config(config_path, log_level=log_level, drift_analyzer=drift_analyzer)
        resolved_cfg_file = config_file or (config_path if not isinstance(config_path, dict) else None)
        cfg_path_str = str(resolved_cfg_file) if resolved_cfg_file is not None else None
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

        # Method, drift_terms and drift_analyzer arrive resolved from load_config (defaults applied, validated)
        pre_target = pre_krig_cfg.get("interpolation_target", "P")
        pre_method = pre_krig_cfg.get("method")
        pre_drifts = pre_krig_cfg.get("drift_terms")
        pre_var_model = pre_krig_cfg.get("variogram_model", "spherical")
        pre_zero_boundary = pre_krig_cfg.get("include_zero_boundary_condition", True)
        pre_drift_analyzer = pre_krig_cfg.get("drift_analyzer", False)
        pre_calculate_uncertainty = pre_krig_cfg.get("calculate_uncertainty", False)

        post_target = post_krig_cfg.get("interpolation_target", "P")
        post_method = post_krig_cfg.get("method")
        post_drifts = post_krig_cfg.get("drift_terms")
        post_var_model = post_krig_cfg.get("variogram_model", "spherical")
        post_zero_boundary = post_krig_cfg.get("include_zero_boundary_condition", True)
        post_drift_analyzer = post_krig_cfg.get("drift_analyzer", False)
        post_calculate_uncertainty = post_krig_cfg.get("calculate_uncertainty", False)

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
            pre_drift_analyzer=pre_drift_analyzer,
            pre_calculate_uncertainty=pre_calculate_uncertainty,
            post_kriging_method=post_method,
            post_drift_terms=post_drifts,
            post_variogram_model=post_var_model,
            post_zero_boundary=post_zero_boundary,
            post_interpolation_target=post_target,
            post_drift_analyzer=post_drift_analyzer,
            post_calculate_uncertainty=post_calculate_uncertainty,
            perform_migration=migration_cfg.get("perform_migration", True),
            survey_data_type=survey_dtype,
            horizontal_displacement_visualization=migration_cfg.get("horizontal_displacement_visualization", "arrows"),
            plots_dir=outputs.get("plots_dir", None),
            n_cores=n_cores,
            kriging_engine=kriging_engine,
            nrbins=nrbins,
            ice_density=ice_density,
            g=g_val,
            output_dir=configured_output_dir(cfg),
            survey_data_path=inputs.get("survey_data_path"),
            survey_profile_column=inputs.get("survey_profile_column"),
            config_path=cfg_path_str,
            show_progress=show_progress_val,
            config=cfg,
            mass_balance=inputs.get("mass_balance_path"),
            velocity=inputs.get("velocity_path"),
        )
        solver.config = cfg
        solver.config_path = cfg_path_str
        if solver.survey_data_path:
            try:
                solver.load_survey_points()
            except Exception as e:
                logger.debug(f"Deferred survey points loading: {e}")
        return solver

    def get_sample_points(
        self,
        stage: str,
        target_type: str = "P",
        interactive: bool | None = None,
    ) -> np.ndarray:
        """Retrieves or builds cached observation points for Kriging interpolation.

        Guarantees that surface slope interpolation and multiplication for the basal
        shear stress product P = T * sin(alpha_opt) or P = D * sin(alpha_opt) is evaluated
        once and cached in memory.

        Parameters
        ----------
        stage : str
            Workflow stage ('pre_migration' or 'post_migration').
        target_type : str, default 'P'
            Interpolation target variable:
              - 'P' : Basal shear stress product (T * sin(alpha) or D * sin(alpha)).
              - 'T' : Direct signal traveltime [ns].
              - 'D' : Direct ice thickness [m].
        interactive : bool or None, optional
            Whether to run BSS optimization interactively if slopes have not yet been evaluated.
            If None, inferred from configuration and terminal mode.

        Returns
        -------
        sample_points : np.ndarray
            (N, 3) or (N, 4) NumPy array of observation points [X, Y, Target]
            (with optional profile ID in column 3).

        Raises
        ------
        ValueError
            If no survey or migrated points are loaded.
        """
        stage_key = "pre_migration" if "pre" in stage.lower() else "post_migration"
        target_upper = str(target_type).upper().strip()
        cache_key = (stage_key, target_upper)

        if not hasattr(self, "_sample_pts_cache"):
            self._sample_pts_cache = {}

        if cache_key in self._sample_pts_cache:
            return self._sample_pts_cache[cache_key]

        pts = self.migrated_points if (stage_key == "post_migration" and self.migrated_points is not None) else self.pre_kriging_points
        if pts is None:
            if stage_key == "post_migration" and self.migrated_points is not None:
                pts = self.migrated_points
            else:
                pts = self.survey_points

        if pts is None:
            if self.survey_points is None and self.survey_data_path and str(self.survey_data_path).strip():
                self.load_survey_points()
                pts = self.survey_points
        if pts is None:
            raise ValueError(
                "Cannot run BSS variogram optimization or Kriging, because no survey data was configured. "
                "Call solver.plan_survey() to model synthetic SIA thickness and design a survey track layout for an unprobed glacier."
            )

        val_col = 3 if pts.shape[1] >= 4 else 2
        has_prof = (pts.shape[1] >= 5)

        if target_upper == "P":
            interp_slope = self.geometry.create_interpolator(self.safe_slope_sin, fill_value=np.nan)
            pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))  # (Y, X)
            slopes_pts = interp_slope(pts_xy)
            min_slope_sin = np.sin(np.radians(self.slope_floor_deg))
            slopes_pts = np.maximum(np.nan_to_num(slopes_pts, nan=min_slope_sin), min_slope_sin)

            product_values = pts[:, val_col] * slopes_pts
            if has_prof:
                sample_pts = np.column_stack((pts[:, 0], pts[:, 1], product_values, pts[:, 4]))
            else:
                sample_pts = np.column_stack((pts[:, 0], pts[:, 1], product_values))
        else:
            if has_prof:
                sample_pts = np.column_stack((pts[:, 0], pts[:, 1], pts[:, val_col], pts[:, 4]))
            else:
                sample_pts = np.column_stack((pts[:, 0], pts[:, 1], pts[:, val_col]))

        self._sample_pts_cache[cache_key] = sample_pts
        return sample_pts

    @property
    def max_memory_fraction(self) -> float:
        """RAM budget fraction for the native Dual Kriging engine (0 < f <= 0.9).

        Returns
        -------
        fraction : float
            Fraction of available host physical RAM allowed for solver allocation.
        """
        if self._max_memory_fraction is not None:
            return self._max_memory_fraction
        cfg_val = self._cfg_get("kriging_parameters", "max_memory_fraction", 0.5)
        return validate_memory_fraction(cfg_val)

    def _run_drift_analysis(
        self,
        stage_key: str,
        target_type: str,
        sample_pts: np.ndarray,
        zero_boundary: bool,
        var_model: str,
        suite: list[list[str]],
        interactive: bool,
        variogram_params: dict[str, float] | None = None,
    ) -> list[CandidateDriftResult]:
        """Runs the Drift Analyzer of one stage on the production sample points, drift grids and fitted variogram.

        The one place that builds the analyzer; both the kriging passes and :meth:`recommend_drift_model` call it.
        The analyzer evaluates the candidates with the variogram model of the stage and the fitted
        ``opt_variogram_params`` of the production run (engine defaults if no variogram was fitted yet).

        Parameters
        ----------
        stage_key : str
            ``'pre_migration'`` or ``'post_migration'``.
        target_type : str
            Interpolation target variable of the stage ('P', 'T' or 'D').
        sample_pts : np.ndarray
            Sample points of the stage; columns ``[x, y, value]`` plus an optional profile ID column.
        zero_boundary : bool
            Whether zero-value boundary constraints are active (informational for the analyzer).
        var_model : str
            Variogram model name of the stage (the 1-D ``"linear"`` model is replaced by ``"spherical"``).
        suite : list of list of str
            Candidate suite (see :func:`pysole.drift.build_drift_suite`).
        interactive : bool
            Whether the analyzer may prompt on the terminal.
        variogram_params : dict of str to float, optional
            Fitted variogram parameters (if None, uses ``self.opt_variogram_params``).

        Returns
        -------
        results : list of CandidateDriftResult
            Ranked candidates, best first.
        """
        mig_cfg = self.config.get("migration_parameters", {}) if hasattr(self, "config") and isinstance(self.config, dict) else {}
        velocity = float(mig_cfg.get("velocity", 0.16)) if mig_cfg.get("velocity") is not None else 0.16
        analyzer = DriftAnalyzer(
            mode=stage_key,
            target_name="traveltime" if stage_key == "pre_migration" else "depth",
            interpolation_target=target_type,
            include_zero_boundary=zero_boundary,
            survey_profile_column=self.survey_profile_column or self.config.get("inputs", {}).get("survey_profile_column"),
            max_memory_fraction=self.max_memory_fraction,
            show_progress=self.show_progress,
            survey_data_type=self.survey_data_type,
            velocity=velocity,
        )
        use_var_params = variogram_params if variogram_params is not None else self.opt_variogram_params
        krig_cfg = self.config.get("kriging_parameters", {}).get(stage_key, {}) if hasattr(self, "config") and isinstance(self.config, dict) else {}
        stage_norm = "pre_migration" if ("pre" in stage_key.lower() or "1" in stage_key) else "post_migration"
        terms = self.pre_drift_terms if stage_norm == "pre_migration" else self.post_drift_terms
        is_predefined = bool(krig_cfg.get("drift_terms")) or (
            isinstance(terms, (list, tuple)) and any(isinstance(elem, (list, tuple)) for elem in terms)
        )
        avail_opt = self._available_optional_terms()
        suite_raster_terms = {t for cand in suite for t in cand if t in RASTER_DRIFT_TERMS}
        active_raster_terms = tuple(
            t for t in RASTER_DRIFT_TERMS
            if t not in OPTIONAL_RASTER_DRIFT_TERMS or t in avail_opt or t in suite_raster_terms
        )
        eval_results = analyzer.run_diagnostics(
            x_pts=sample_pts[:, 0],
            y_pts=sample_pts[:, 1],
            z_values=sample_pts[:, 2],
            geometry=self.geometry,
            drift_grids=self.get_drift_grids(active_raster_terms),
            candidates=suite,
            variogram_model=resolve_native_variogram_model(var_model, warn=False),
            variogram_params=use_var_params,
            profile_data=sample_pts[:, 3] if sample_pts.shape[1] >= 4 else None,
            interactive=interactive,
            is_predefined_suite=is_predefined,
        )
        if stage_key == "pre_migration":
            self.pre_selected_drift_models = analyzer.selected_models
        else:
            self.post_selected_drift_models = analyzer.selected_models
        return eval_results

    def _ensure_optimized(self, stage_key: str, interactive: bool | None = None) -> None:
        """Guarantees BSS slope optimization and variogram parameters have been resolved for the stage."""
        stage_norm = "pre_migration" if ("pre" in stage_key.lower() or "1" in stage_key) else "post_migration"
        is_inter = bool(interactive) and not self.is_batch_mode and sys.stdin.isatty()

        if stage_norm == "pre_migration":
            if self.pre_opt_res is None and self.opt_slope is None:
                self.optimize_bss(stage="pre_migration", interactive=is_inter, prefix="01_")
        else:
            if self.post_opt_res is None:
                if self.opt_slope is None or (self.pre_opt_res is not None and self.migrated_points is not None):
                    self.optimize_bss(stage="post_migration", interactive=is_inter, prefix="03_")

    def _effective_variogram_params(
        self, sample_pts: np.ndarray, target_upper: str
    ) -> dict[str, float] | None:
        """Builds effective variogram parameters, scaling sill to target variance for direct targets."""
        if target_upper == "P" or self.opt_variogram_params is None:
            return self.opt_variogram_params
        z_var = float(np.var(sample_pts[:, 2])) if len(sample_pts) > 1 else 1.0
        opt_sill = float(self.opt_variogram_params.get("sill", z_var))
        nugget_ratio = (
            float(self.opt_variogram_params.get("nugget", 0.0)) / opt_sill
            if opt_sill > 0
            else 0.0
        )
        return {
            "range": float(self.opt_variogram_params["range"]),
            "sill": z_var if z_var > 0 else 1.0,
            "nugget": nugget_ratio * z_var,
        }

    def _execute_kriging_pass(
        self,
        target_type: str,
        points: np.ndarray,
        krig_method: str,
        var_model: str,
        zero_boundary: bool,
        pass_name: str = "Pass 1 - Constructing Traveltime Grid",
        return_variance: bool = True,
        interactive_optimization: bool | None = None,
    ) -> KrigingResult:
        """Executes a unified Kriging interpolation pass for direct or BSS product targets.

        Dispatches Dual Kriging interpolation for direct physical targets ('T' in ns, 'D' in m)
        or basal shear stress product targets ('P' = T * sin(alpha) or D * sin(alpha)).
        Handles drift analysis diagnostics, external curvature grids, and memory guarding.

        Parameters
        ----------
        target_type : str
            Interpolation target variable ('P', 'T', or 'D').
        points : np.ndarray
            Observation points coordinate matrix.
        krig_method : str
            Kriging method ('ordinary', 'universal', 'regression'). Together with the drift_terms and
            drift_analyzer settings of the stage it is resolved by :func:`pysole.drift.resolve_drift_plan`:
            'ordinary' ignores drift terms and analyzer, 'universal' applies the configured drift model or
            the winner of the drift analyzer suite.
        var_model : str
            Theoretical variogram model name ('spherical', 'exponential', 'gaussian').
        zero_boundary : bool
            If True, enforces zero values along the glacier perimeter outline.
        pass_name : str, default 'Pass 1 - Constructing Traveltime Grid'
            Descriptive pass label for logging.
        return_variance : bool, default True
            If True, computes formal Kriging estimation variance.
        interactive_optimization : bool or None, optional
            Whether to run BSS optimization interactively if slopes have not yet been evaluated.

        Returns
        -------
        krig_result : KrigingResult
            Result container holding the interpolated grid and uncertainty variance grid.
        """
        target_upper = str(target_type).upper().strip()
        is_pre = "Pass 1" in pass_name
        stage_key = "pre_migration" if is_pre else "post_migration"

        self._ensure_optimized(stage_key, interactive=interactive_optimization)
        sample_pts = self.get_sample_points(stage_key, target_upper, interactive=interactive_optimization)
        effective_var_params = self._effective_variogram_params(sample_pts, target_upper)

        # The plan was validated (and its notices logged) when the Solver was built; it is re-resolved here
        # silently because ``calculate_bedrock(method=...)`` may replace the method of the stage.
        plan = self._drift_plan(stage_key, method=krig_method)
        krig_method = plan.method
        drift_terms = plan.terms

        if target_upper == "P":
            target_desc = "BSS Product P(x, y)"
        elif target_upper == "T":
            target_desc = "Traveltime T(x, y)"
        else:
            target_desc = "Depth D(x, y)"
        logger.info(f"   [Dual Kriging Engine] [{pass_name}] Performing {krig_method.capitalize()} Kriging Interpolation for {target_desc}...")

        if plan.analyze:
            is_interactive = plan.interactive and not self.is_batch_mode and sys.stdin.isatty()
            if not plan.interactive:
                logger.info(
                    f"   [Drift Analyzer] [{pass_name}] Non-interactive multi-model suite run (drift_analyzer: false). "
                    "Using default spatial exclusion buffer radius."
                )
            elif not is_interactive:
                logger.info("   [INFO] Non-interactive environment detected (TTY disabled or --batch flag set). Automatically overriding interactive options to False.")

            diag = self._run_drift_analysis(
                stage_key=stage_key,
                target_type=target_upper,
                sample_pts=sample_pts,
                zero_boundary=zero_boundary,
                var_model=var_model,
                suite=plan.suite,
                interactive=is_interactive,
                variogram_params=effective_var_params,
            )
            # The winner decides the final method: Ordinary (no drift term) or Universal with its terms
            if not diag or not np.isfinite(diag[0].aicc) or diag[0].aicc >= FAILED_AICC:
                logger.warning(
                    f"   [Dual Kriging Engine] [{pass_name}] The Drift Analyzer could not cross-validate any candidate; "
                    "falling back to Ordinary Kriging."
                )
                drift_terms = []
            else:
                drift_terms = list(diag[0].terms)
            krig_method = "universal" if drift_terms else "ordinary"

        if target_upper == "P" and krig_method == "universal" and ("sia" in drift_terms or "velocity_sia" in drift_terms):
            active_sia_term = "velocity_sia" if "velocity_sia" in drift_terms else "sia"
            logger.warning(
                f"   [Dual Kriging Engine] [{pass_name} Warning] '{active_sia_term}' drift is active during BSS product P(x,y) interpolation. "
                "This can cause 1/sin^2(alpha) double-scaling artifacts at low-slope margins. "
                "Recommendation: Use 'ordinary' Kriging or non-slope spatial drifts (e.g. ['z_dem']) for product targets."
            )

        # The production run samples exactly the grids the analyzer evaluated (k_c-smoothed curvature, floored SIA).
        ext_drifts = self.get_drift_grids(drift_terms)

        krig_res = kriging_interpolation(
            sample_points=sample_pts,
            geometry=self.geometry,
            method=krig_method,
            variogram_model=var_model,
            variogram_params=effective_var_params,
            dem_grid=self.dem_grid,
            opt_slope_grid=self.opt_slope,
            drift_terms=drift_terms,
            outline_mask=self.outline_mask,
            include_zero_boundary_condition=zero_boundary,
            n_cores=self.n_cores,
            engine=self.engine_type,
            slope_floor_deg=self.slope_floor_deg,
            show_progress=self.show_progress,
            external_drift_grid=ext_drifts if len(ext_drifts) > 0 else None,
            return_variance=return_variance,
            max_memory_fraction=self.max_memory_fraction,
        )

        if target_upper == "P":
            safe_slope_grid = self.safe_slope_sin
            grid = krig_res.bedrock_grid / safe_slope_grid
            var = krig_res.variance_grid / (safe_slope_grid**2)
            return KrigingResult(bedrock_grid=grid, variance_grid=var)

        return krig_res

    def _warn_traveltime_exports_unavailable(self, reason: str) -> None:
        """Warns once if traveltime exports are requested although no traveltime grid is computed.

        Parameters
        ----------
        reason : str
            Explanation of why traveltime calculation was skipped.
        """
        cfg = self.outputs_config_obj
        flags = [
            name for name in ("save_traveltime_grid", "save_traveltime_uncertainty") if getattr(cfg, name)
        ]
        if flags:
            logger.warning(
                f"   [3D Ray Migration] {', '.join(flags)} requested, but no traveltime grid is computed "
                f"because {reason}; nothing will be exported for it."
            )

    def load_survey_points(
        self,
        survey_data: str | Path | os.PathLike | np.ndarray | None = None,
    ) -> np.ndarray:
        """Loads, validates, and normalizes survey soundings from file or array.

        Parameters
        ----------
        survey_data : str, Path, os.PathLike, np.ndarray, or None, optional
            Path to survey points file (CSV) or pre-loaded coordinate array.
            If None, defaults to ``self.survey_data_path``.

        Returns
        -------
        survey_points : np.ndarray
            (N, 4+) coordinate array [X, Y, Z_dem, Traveltime_or_Depth, ...] in projected metric units.
        """
        if survey_data is None:
            survey_data = self.survey_data_path
        if survey_data is None:
            raise ValueError("No survey_data_path or survey_data provided to load survey points.")

        if isinstance(survey_data, np.ndarray):
            pts = survey_data.copy()
        else:
            if not self.survey_data_path:
                self.survey_data_path = str(survey_data)
            pts = load_survey_points(
                self.resolve_input_path(survey_data),
                bounds=self.bounds,
                dem_grid=self.dem_grid,
                x_coords=self.x_coords,
                y_coords=self.y_coords,
                profile_column=self.survey_profile_column or self._cfg_get("inputs", "survey_profile_column"),
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
            logger.info("   [Survey Data Type] two_way_traveltime detected. Converting to one_way_traveltime (one_way_traveltime = two_way_traveltime / 2.0)...")
            pts[:, 3] = pts[:, 3] / 2.0
        elif dtype_str == "one_way_traveltime":
            logger.info("   [Survey Data Type] one_way_traveltime detected.")
        elif dtype_str == "depth":
            logger.info("   [Survey Data Type] depth detected. Skipping 3D Eikonal ray migration.")

        self.survey_points = pts
        self._sample_pts_cache.clear()
        return pts

    def migrate_eikonal(
        self,
        travel_times: str | Path | os.PathLike | np.ndarray | None = None,
        velocity: float | None = None,
        interactive: bool = False,
        interactive_optimization: bool | None = None,
        plotit: bool = False,
    ) -> np.ndarray:
        """Migrates zero-offset radar or seismic traveltimes into 3D space via the Eikonal equation.

        Corrects geometric sounding positions for non-vertical wave propagation caused by dipping
        basal topography. Radar echoes recorded at the glacier surface are displaced back along
        the 3D wavefront normal vector defined by surface traveltime gradients:
          dx = -v^2 * T * (dT/dx), dy = -v^2 * T * (dT/dy),
          z_mig = z_surf - v * T * sqrt(1 - v^2 * |grad T|^2)

        Parameters
        ----------
        travel_times : str, Path, os.PathLike, or np.ndarray, optional
            Path to survey points file or array of survey coordinates and traveltimes.
            If None, uses survey_data_path attribute.
        velocity : float, optional
            Effective electromagnetic wave propagation velocity [m/ns] (default 0.16 m/ns in cold ice).
        interactive : bool, default False
            If True, enables interactive command-line prompts to inspect and adjust velocity.
        interactive_optimization : bool or None, optional
            If True, enables interactive prompts during Stage 1 BSS variogram optimization if required
            for pre-migration slope filtering. If None, uses configuration setting.
        plotit : bool, default False
            If True, generates visual diagnostic figures of displacement vectors.

        Returns
        -------
        migrated_points : np.ndarray
            (N, 4+) NumPy array of migrated sounding coordinates [X_mig, Y_mig, Z_mig, Depth_mig]
            in projected metric coordinates [m].
        """
        if travel_times is not None:
            if isinstance(travel_times, np.ndarray):
                pts = self.load_survey_points(travel_times)
            else:
                resolved_tt = str(travel_times)
                if self.survey_points is None or self.survey_data_path != resolved_tt:
                    pts = self.load_survey_points(travel_times)
                else:
                    pts = self.survey_points
        else:
            if self.survey_points is None:
                pts = self.load_survey_points()
            else:
                pts = self.survey_points

        dtype_str = self._normalize_survey_data_type()

        if dtype_str == "depth":
            self.migrated_points = pts.copy()
            if self.outputs_config.get("save_migrated_points", False):
                logger.info("   [3D Ray Migration] Migration is disabled. Skipping 'save_migrated_points' output.")
            self._warn_traveltime_exports_unavailable("survey_data_type is 'depth'")
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
            logger.info("   [3D Ray Migration] 'perform_migration' is set to False in configuration. Using unmigrated survey coordinates.")
            unmig_depths = pts[:, 3] * v_eff if pts[:, 3].max() > 15.0 else pts[:, 3]
            # Preserve optional profile-ID column (index 4) so LOPO / drift CV stays available
            self.migrated_points = np.column_stack((pts[:, 0], pts[:, 1], pts[:, 2], unmig_depths, pts[:, 4:]))
            if self.outputs_config.get("save_migrated_points", False):
                logger.info("   [3D Ray Migration] Migration is disabled. Skipping 'save_migrated_points' output.")
            self._warn_traveltime_exports_unavailable("'perform_migration' is False")
            return self.migrated_points

        if interactive_optimization is None:
            effective_opt_interactive = bool(interactive)
        else:
            effective_opt_interactive = bool(interactive_optimization) and not self.is_batch_mode and sys.stdin.isatty()

        krig1_res = self._execute_kriging_pass(
            target_type=self.pre_interpolation_target,
            points=pts,
            krig_method=self.pre_kriging_method,
            var_model=self.pre_variogram_model,
            zero_boundary=self.pre_zero_boundary,
            pass_name="Pass 1 - Constructing Traveltime Grid",
            return_variance=self.pre_calculate_uncertainty,
            interactive_optimization=effective_opt_interactive,
        )
        tt_grid = np.maximum(krig1_res.bedrock_grid, 0.0)
        if self.outline_mask is not None:
            tt_grid[~self.outline_mask] = np.nan
        self._traveltime_grid = tt_grid

        if krig1_res.variance_grid is not None and not np.all(np.isnan(krig1_res.variance_grid)):
            self._traveltime_std = np.sqrt(np.maximum(krig1_res.variance_grid, 0.0))
            if self.outline_mask is not None:
                self._traveltime_std[~self.outline_mask] = np.nan
        else:
            self._traveltime_std = None

        cached_dem_grads = self._get_dem_gradients()

        logger.info(f"   [3D Ray Migration] Executing 3D Eikonal Ray Displacement (velocity v = {v_eff:.4f} m/ns)...")
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
            output_prefix=self.figure_prefix,
            horizontal_displacement_visualization=self.horizontal_displacement_visualization,
        )
        self.migrated_points = mig_res.migrated_points

        if interactive:
            log_section_header("INTERACTIVE 3D EIKONAL RAY MIGRATION")
            while True:
                try:
                    ans = input(f"Current migration velocity v = {v_eff:.4f} m/ns. Test another migration velocity? [y/N]: ").strip().lower()
                    if ans in ["y", "yes"]:
                        val = input("Enter new signal propagation velocity [m/ns] (e.g. 0.15): ").strip()
                        if val:
                            v_eff = float(val)
                            logger.info(f"   [3D Ray Migration] Re-running with custom velocity v = {v_eff:.4f} m/ns...")
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
                                output_prefix=self.figure_prefix,
                                horizontal_displacement_visualization=self.horizontal_displacement_visualization,
                            )
                            self.migrated_points = mig_res2.migrated_points
                    else:
                        break
                except Exception:
                    break
            end_terminal_block()

        if self.pre_selected_drift_models:
            records: list[DriftComparisonRecord] = []
            target_upper = self.pre_interpolation_target.upper().strip()
            tt_unit = "m" if self.survey_data_type == "depth" else "ns"
            sample_pts = self.get_sample_points("pre_migration", target_upper, interactive=effective_opt_interactive)
            eff_var = self._effective_variogram_params(sample_pts, target_upper)

            for idx, cand in enumerate(self.pre_selected_drift_models):
                cand_slug = slugify_drift_name(cand.terms)
                cand_rank = cand.rank or (idx + 1)
                if idx == 0:
                    cand_tt = tt_grid
                    cand_std = self._traveltime_std
                    self.export_outputs("migration", model_slug=cand_slug)
                    t_mean, t_median, t_std, t_max = _compute_distribution_stats(cand_tt, self.outline_mask)
                    failed = False
                else:
                    logger.info(f"   [Drift Comparison] [Pass 1] Evaluating candidate model '{cand.name}'...")
                    ext_drifts = self.get_drift_grids(cand.terms)
                    cand_method = "universal" if cand.terms else "ordinary"
                    try:
                        k_cand = kriging_interpolation(
                            sample_points=sample_pts,
                            geometry=self.geometry,
                            method=cand_method,
                            variogram_model=self.pre_variogram_model,
                            variogram_params=eff_var,
                            dem_grid=self.dem_grid,
                            opt_slope_grid=self.opt_slope,
                            drift_terms=cand.terms,
                            outline_mask=self.outline_mask,
                            include_zero_boundary_condition=self.pre_zero_boundary,
                            n_cores=self.n_cores,
                            engine=self.engine_type,
                            slope_floor_deg=self.slope_floor_deg,
                            show_progress=self.show_progress,
                            external_drift_grid=ext_drifts if len(ext_drifts) > 0 else None,
                            return_variance=self.pre_calculate_uncertainty,
                            max_memory_fraction=self.max_memory_fraction,
                        )
                        if target_upper == "P":
                            cand_tt = np.maximum(k_cand.bedrock_grid / self.safe_slope_sin, 0.0)
                            cand_var = k_cand.variance_grid / (self.safe_slope_sin**2) if k_cand.variance_grid is not None else None
                        else:
                            cand_tt = np.maximum(k_cand.bedrock_grid, 0.0)
                            cand_var = k_cand.variance_grid

                        if self.outline_mask is not None:
                            cand_tt[~self.outline_mask] = np.nan
                        cand_std = np.sqrt(np.maximum(cand_var, 0.0)) if (cand_var is not None and not np.all(np.isnan(cand_var))) else None
                        if cand_std is not None and self.outline_mask is not None:
                            cand_std[~self.outline_mask] = np.nan

                        orig_tt = self._traveltime_grid
                        orig_std = self._traveltime_std
                        self._traveltime_grid = cand_tt
                        self._traveltime_std = cand_std
                        self.export_outputs("migration", model_slug=cand_slug, export_points=False)
                        self._traveltime_grid = orig_tt
                        self._traveltime_std = orig_std

                        t_mean, t_median, t_std, t_max = _compute_distribution_stats(cand_tt, self.outline_mask)
                        failed = False
                    except Exception as e:
                        logger.warning(f"   [Drift Comparison] Interpolation failed for '{cand.name}': {e}")
                        t_mean, t_median, t_std, t_max = 0.0, 0.0, 0.0, 0.0
                        failed = True

                if self.pre_calculate_uncertainty and cand_std is not None and not failed:
                    mask_tt = (self.outline_mask & (cand_tt > 0)) if (self.outline_mask is not None and np.any(self.outline_mask)) else (cand_tt > 0)
                    v_unc = cand_std[mask_tt & np.isfinite(cand_std)]
                    t_unc_mean = float(np.mean(v_unc)) if v_unc.size > 0 else 0.0
                else:
                    t_unc_mean = None

                records.append(
                    DriftComparisonRecord(
                        pass_label="Pass 1",
                        rank=cand_rank,
                        model_name=cand.name,
                        model_slug=cand_slug,
                        terms=cand.terms,
                        cv_rmse=cand.cv_rmse,
                        delta_rmse_pct=cand.delta_rmse_pct,
                        aicc=cand.aicc,
                        delta_aicc=cand.delta_aicc,
                        vif_max=cand.vif_max,
                        status="FAILED" if failed else ("PENALIZED" if cand.is_penalized else ("WARNING" if cand.warning_msg else "VALID")),
                        rmse_unit=tt_unit,
                        target_mean=t_mean,
                        target_median=t_median,
                        target_std=t_std,
                        target_max=t_max,
                        target_unit=tt_unit,
                        unc_mean=t_unc_mean,
                        is_failed=failed,
                    )
                )

            self.pre_drift_comparison_records = records
            target_out_dir = self.effective_output_dir
            if target_out_dir:
                save_drift_comparison_files(
                    target_out_dir,
                    pre_records=self.pre_drift_comparison_records,
                    post_records=self.post_drift_comparison_records,
                )
            print(format_drift_comparison_table(records, "Pass 1"))
        else:
            self.export_outputs("migration")
        return self.migrated_points

    def optimize_bss(
        self,
        stage: str | None = None,
        kc_max: float | None = None,
        kc_min: float | None = None,
        n_steps: int | None = None,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        fft_filter_metric: str | None = None,
        nrbins: int | None = None,
        prefix: str | None = None,
        interactive: bool = False,
        plotit: bool = False,
    ) -> float:
        """Determines the physically optimal surface slope smoothing degree via BSS variance minimization.

        Implements iterative spectral optimization based on glaciological principles: local basal
        shear stress (BSS) tau_b = rho_ice * g * D * sin(alpha) is driven by regional ice surface
        slope rather than micro-topographic roughness. High-frequency DEM slope noise causes
        unrealistic local BSS spikes, whereas over-smoothing eliminates legitimate valley gradients.
        At each trial cutoff wavenumber k_c (corresponding to spatial wavelength lambda_c = 2*pi / k_c),
        the surface DEM is smoothed with a 2D Gaussian FFT low-pass filter, and the spatial variance
        of derived basal shear stress is computed. The optimal wavenumber k_c minimizes this variance.

        Parameters
        ----------
        stage : str, optional
            Stage to optimize: 'pre_migration' (or 'pass1') using unmigrated survey soundings,
            or 'post_migration' (or 'pass2') using 3D migrated soundings. If None, automatically
            inferred as 'post_migration' if migrated soundings exist, else 'pre_migration'.
        kc_max : float, optional
            Upper wavenumber bound [rad/m] for trial search. If None, derived from DEM resolution (Nyquist).
        kc_min : float, optional
            Lower wavenumber bound [rad/m] for trial search. If None, derived from DEM domain extent.
        n_steps : int, optional
            Number of discrete trial evaluation steps across the spectral bandwidth.
        lambda_min : float, optional
            Minimum physical spatial cutoff wavelength [m] (alternative parameterization to kc_max).
        lambda_max : float, optional
            Maximum physical spatial cutoff wavelength [m] (alternative parameterization to kc_min).
        fft_filter_metric : str, optional
            Evaluation metric domain: 'wavenumber' [rad/m] or 'wavelength' [m]. Default 'wavenumber'.
        nrbins : int, optional
            Number of distance lag bins for experimental variogram calculation.
        prefix : str, optional
            Output filename prefix for diagnostic optimization figures (e.g. '01_' or '03_').
        interactive : bool, default False
            If True, prompts in the terminal to inspect and adjust the trial parameter spectrum.
        plotit : bool, default False
            If True, generates visual diagnostic figures of BSS variance curves.

        Returns
        -------
        opt_kc : float
            Optimal spatial cutoff wavenumber k_c [rad/m] (lambda_c = 2*pi / opt_kc [m]).
        """
        if stage is None:
            stage_key = "post_migration" if self.migrated_points is not None else "pre_migration"
        else:
            stage_key = "pre_migration" if ("pre" in str(stage).lower() or "1" in str(stage)) else "post_migration"

        if stage_key == "pre_migration":
            if self.survey_points is None and self.survey_data_path is not None:
                self.load_survey_points()
            pts = self.survey_points if self.survey_points is not None else self.pre_kriging_points
        else:
            if self.migrated_points is not None:
                pts = self.migrated_points
            else:
                if self.survey_points is None and self.survey_data_path is not None:
                    self.load_survey_points()
                pts = self.survey_points

        if pts is None:
            raise ValueError(
                "Cannot run BSS variogram optimization or Kriging, because no survey data was configured. "
                "Call solver.plan_survey() to model synthetic SIA thickness and design a survey track layout for an unprobed glacier."
            )

        opt_cfg = self.config.get("optimization_parameters", {}) if isinstance(self.config, dict) else {}
        if kc_max is None:
            kc_max = opt_cfg.get("kc_max")
        if kc_min is None:
            kc_min = opt_cfg.get("kc_min")
        if n_steps is None:
            n_steps = opt_cfg.get("n_steps")
        if lambda_min is None:
            lambda_min = opt_cfg.get("lambda_min")
        if lambda_max is None:
            lambda_max = opt_cfg.get("lambda_max")
        if fft_filter_metric is None:
            fft_filter_metric = opt_cfg.get("fft_filter_metric", "wavenumber")
        if nrbins is None:
            nrbins = opt_cfg.get("nrbins")

        if fft_filter_metric is None:
            fft_filter_metric = "wavenumber"

        if nrbins is None:
            nrbins = self.nrbins

        if prefix is None:
            prefix = "01_" if stage_key == "pre_migration" else "03_"

        stage_name = "stage1" if prefix.startswith("01") else "stage2"

        opt_res = self.bss_optimizer.optimize(
            survey_points=pts,
            kc_max=kc_max,
            kc_min=kc_min,
            n_steps=n_steps,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            fft_filter_metric=fft_filter_metric,
            plots_dir=self.plots_dir,
            prefix=prefix,
            stage_name=stage_name,
            interactive=interactive,
            n_cores=self.n_cores,
            nrbins=nrbins,
            show_progress=self.show_progress,
            output_prefix=self.figure_prefix,
        )

        if stage_key == "pre_migration":
            self.pre_opt_res = opt_res
        else:
            self.post_opt_res = opt_res

        self.opt_kc = opt_res.optimal_kc
        self.opt_slope = opt_res.optimal_slope_grid
        self.opt_variogram_params = opt_res.opt_variogram_params
        self._sample_pts_cache.clear()
        # Retain strictly the optimal smoothed DEM in the solver cache to minimize memory footprint
        best_key = round(float(self.opt_kc), 6)
        if opt_res.optimal_dem_grid is not None:
            self._smoothed_dem_cache = {best_key: opt_res.optimal_dem_grid}
        else:
            self._smoothed_dem_cache.clear()
        return self.opt_kc

    def calculate_bedrock(
        self,
        method: str | None = None,
        variogram_model: str | None = None,
        interactive: bool = False,
        calculate_uncertainty: bool | None = None,
        interactive_optimization: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolates post-migration bedrock elevation and estimation uncertainty via Dual Kriging.

        Executes the second-pass geostatistical interpolation using migrated observation soundings.
        Interpolates either ice thickness D(x, y) [m] or basal shear stress product P(x, y) = D * sin(alpha_opt),
        enforces non-negative thickness bounds, clamps extreme extrapolation artifacts, subtracts
        thickness from surface DEM elevation to produce bedrock elevation Z_bed(x, y) = Z_surf(x, y) - D(x, y),
        and extracts Kriging estimation variance sigma^2_D(x, y) [m^2].

        Parameters
        ----------
        method : str, optional
            Kriging interpolation method ('universal', 'ordinary', 'regression').
            If None, uses post_kriging_method.
        variogram_model : str, optional
            Theoretical variogram model ('spherical', 'exponential', 'gaussian').
            If None, uses post_variogram_model.
        interactive : bool, default False
            If True, generates interactive diagnostic plots.
        calculate_uncertainty : bool or None, optional
            If True, calculates Kriging estimation variance during Pass 2. If None, uses
            post_calculate_uncertainty configuration setting.
        interactive_optimization : bool or None, optional
            Whether to run BSS optimization interactively if slopes have not yet been evaluated.

        Returns
        -------
        kriged_bedrock : np.ndarray
            2D array of interpolated bedrock elevation values [m].
        kriged_variance : np.ndarray
            2D array of Kriging estimation variance [m^2] (or NaN if variance calculation was skipped).

        Raises
        ------
        ValueError
            If no survey or migrated points are available.
        """
        pts = self.migrated_points if self.migrated_points is not None else self.survey_points
        if pts is None:
            raise ValueError("No survey or migrated points available. Run migrate_eikonal() first.")

        should_plot = interactive or (self.plots_dir is not None)

        if method is not None:
            self._log_drift_plan("Pass 2 - Constructing Depth Grid", self._drift_plan("post_migration", method=method))
            krig_method = normalize_method(method)
        else:
            krig_method = self.post_kriging_method
        if variogram_model is not None:
            var_model = self._resolve_variogram_for_engine(variogram_model, krig_method)
        else:
            var_model = self.post_variogram_model

        if interactive_optimization is None:
            eff_opt_inter = bool(interactive) and not self.is_batch_mode and sys.stdin.isatty()
        else:
            eff_opt_inter = bool(interactive_optimization) and not self.is_batch_mode and sys.stdin.isatty()

        eff_calc_unc = self.post_calculate_uncertainty if calculate_uncertainty is None else bool(calculate_uncertainty)
        krig_res = self._execute_kriging_pass(
            target_type=self.post_interpolation_target,
            points=pts,
            krig_method=krig_method,
            var_model=var_model,
            zero_boundary=self.post_zero_boundary,
            pass_name="Pass 2 - Constructing Depth Grid",
            return_variance=eff_calc_unc,
            interactive_optimization=eff_opt_inter,
        )
        grid_raw = krig_res.bedrock_grid
        prod_var = krig_res.variance_grid

        val_col = 3 if pts.shape[1] >= 4 else 2
        valid_pts_val = pts[~np.isnan(pts[:, val_col]), val_col]
        max_thickness = max(float(np.max(valid_pts_val)) * 1.5, 500.0) if len(valid_pts_val) > 0 else 500.0

        eval_mask = self.outline_mask if self.outline_mask is not None else np.ones_like(grid_raw, dtype=bool)
        n_clipped_neg = int(np.sum((grid_raw < 0.0) & eval_mask))
        n_clipped_max = int(np.sum((grid_raw > max_thickness) & eval_mask))
        if n_clipped_neg > 0 or n_clipped_max > 0:
            logger.info(
                f"   [Thickness Clipping] Clipped {n_clipped_neg} negative cells and {n_clipped_max} excessive cells "
                f"(> {max_thickness:.1f}m) to valid thickness range [0, {max_thickness:.1f}m]."
            )
        thickness_grid = np.clip(grid_raw, 0.0, max_thickness)

        if self.outline_mask is not None:
            thickness_grid[~self.outline_mask] = 0.0

        self.kriged_thickness = thickness_grid.copy()
        self.kriged_bedrock = self.dem_grid - thickness_grid
        self.kriged_variance = prod_var
        if self.outline_mask is not None:
            self.kriged_variance[~self.outline_mask] = np.nan

        # NaN variance (variance evaluation skipped) propagates to NaN std, never to a spurious 0 uncertainty
        self.kriged_std = np.sqrt(np.maximum(self.kriged_variance, 0.0))
        if self.outline_mask is not None:
            self.kriged_std[~self.outline_mask] = np.nan

        plot_extent = self.plot_extent

        if should_plot:
            from .plotting import plot_kriging_bedrock_and_uncertainty
            if self.post_selected_drift_models and self.plots_dir:
                p_slug = slugify_drift_name(self.post_selected_drift_models[0].terms)
                p_dir = Path(self.plots_dir) / p_slug
                sfx = p_slug
            else:
                p_dir = self.plots_dir
                sfx = ""
            plot_kriging_bedrock_and_uncertainty(
                kriged_bedrock=self.kriged_bedrock,
                kriged_std=self.kriged_std,
                pts=pts,
                plot_extent=plot_extent,
                plots_dir=p_dir,
                interactive=interactive,
                output_prefix=self.figure_prefix,
                model_suffix=sfx,
            )

        return self.kriged_bedrock, self.kriged_variance

    def _extra_rf_feature_grids(self) -> dict[str, np.ndarray] | None:
        """Builds optional auxiliary feature grids (mass_balance, velocity) for Random Forest gap filling."""
        extra: dict[str, np.ndarray] = {}
        if self.mass_balance_grid is not None:
            extra["mass_balance"] = self.mass_balance_grid
        if self.velocity_grid is not None:
            kc_use = self.opt_kc if self.opt_kc is not None else 0.05
            extra["velocity"] = self.get_smoothed_velocity(kc_use)
        return extra if extra else None

    def fill_holes_rf(self) -> np.ndarray:
        """Fills unconstrained bedrock voids and gaps using Random Forest machine learning regression.

        Trains a multi-feature Random Forest regressor on Kriged bedrock elevations using terrain
        covariates (surface elevation, slope, aspect, distance to boundary outline, surface curvature,
        plus optional mass balance and smoothed velocity fields) and predicts bedrock elevation in
        areas where Kriging uncertainty is high or data gaps exist.

        Returns
        -------
        rf_filled_bedrock : np.ndarray
            2D array of gap-filled bedrock elevations [m].
        """
        if self.kriged_bedrock is None:
            self.calculate_bedrock()

        self.rf_filled_bedrock = self.finalizer.fill_holes(
            self.kriged_bedrock,
            n_cores=self.n_cores,
            show_progress=self.show_progress,
            extra_feature_grids=self._extra_rf_feature_grids(),
        )
        return self.rf_filled_bedrock

    def apply_geomorph_smoothing(self, min_gap_dist: float | None = None) -> np.ndarray:
        """Applies geomorphological boundary margin blending to surrounding surface terrain.

        Smoothly tapers ice thickness to zero along the glacier boundary perimeter using
        a distance-weighted sigmoid transition, ensuring that predicted bedrock elevation
        flawlessly matches surrounding unglaciated surface topography.

        Parameters
        ----------
        min_gap_dist : float, optional
            Transition blending buffer width [m] inward from the ice margin boundary.
            If None, uses 50.0 m.

        Returns
        -------
        blended_bedrock : np.ndarray
            2D array of margin-blended bedrock elevations [m].
        """
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
        """Applies spatial filter smoothing to calculated bedrock or ice thickness fields.

        Parameters
        ----------
        grid : np.ndarray
            2D input array of elevation or thickness values [m].
        method : str, default 'gaussian'
            Spatial filtering algorithm: 'gaussian', 'median', or 'fft_lowpass'.
        sigma : float, default 1.5
            Gaussian smoothing kernel standard deviation in pixels [px].
        kernel_size : int, default 3
            Median filter window size (must be an odd integer, e.g. 3, 5).
        kc_cutoff : float, optional
            Corner frequency cutoff wavenumber k_c [rad/m] for 2D FFT lowpass filtering.
            If None, defaults to optimal k_c from BSS optimization.

        Returns
        -------
        smoothed_grid : np.ndarray
            2D array of spatially smoothed elevations [m].
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
            from .smoothing import fft_gaussian_smooth

            kc = kc_cutoff if kc_cutoff is not None else (self.opt_kc if self.opt_kc is not None else 0.05)
            smoothed_grid, _, _ = fft_gaussian_smooth(out_grid, dx=self.dx, dy=self.dy, kc=kc)
            valid_mask = ~np.isnan(out_grid)
            out_grid[valid_mask] = smoothed_grid[valid_mask]

        return out_grid

    def finalize_bedrock(
        self,
        interactive: bool = False,
        random_forest_gap_filling: bool | None = None,
        apply_margin_blend: bool | None = None,
        min_gap_dist: float | None = None,
        smooth_bedrock: bool = False,
        smoothing_method: str = "gaussian",
        smoothing_sigma: float = 1.5,
        smoothing_kernel_size: int = 3,
        smoothing_kc_cutoff: float | None = None,
    ) -> BedrockMap:
        """Finalizes continuous bedrock topography, ice thickness, basal shear stress, and uncertainty.

        Executes the post-processing pipeline following Dual Kriging:
          1. Optional spatial smoothing of the Kriged ice thickness field D(x, y) [m].
          2. Bedrock calculation Z_bed(x, y) = Z_surf(x, y) - D(x, y) [m].
          3. Optional Random Forest machine learning gap filling in unconstrained areas.
          4. Optional geomorphological boundary margin blending to surrounding surface DEM.
          5. Derivation of final ice thickness, basal shear stress tau_b = rho * g * D * sin(alpha) [kPa],
             and their standard error uncertainty fields.
          6. Automatic generation of diagnostic figures and configured raster/vector file exports.

        Parameters
        ----------
        interactive : bool, default False
            If True and running in an interactive terminal (non-batch), prompts to inspect and adjust
            finalization options and refreshes diagnostic figure 07 in real time.
        random_forest_gap_filling : bool, optional
            If True, applies Random Forest ML regression to fill unconstrained voids. If None, asks
            when interactive, otherwise False.
        apply_margin_blend : bool, optional
            If True, smoothly transitions ice thickness to 0 m at glacier perimeter boundary. If None,
            asks when interactive, otherwise False.
        min_gap_dist : float, optional
            Transition margin buffer width [m] inward from outline boundary (default 50.0 m).
        smooth_bedrock : bool, default False
            If True, filters the Kriged depth field prior to bedrock elevation calculation.
        smoothing_method : str, default 'gaussian'
            Spatial filtering algorithm: 'gaussian', 'median', or 'fft_lowpass'.
        smoothing_sigma : float, default 1.5
            Smoothing strength (Gaussian kernel radius in grid pixels [px]).
        smoothing_kernel_size : int, default 3
            Window kernel size (k x k pixels) for median filtering (must be an odd integer).
        smoothing_kc_cutoff : float, optional
            Cutoff wavenumber k_c [rad/m] for 2D FFT lowpass filtering. If None, uses optimal k_c.

        Returns
        -------
        bedrock_map : BedrockMap
            Final continuous bedrock elevation map [m] wrapped with spatial georeferencing metadata.
        """
        is_interactive = bool(interactive and sys.stdin.isatty() and not getattr(self, "is_batch_mode", False))
        should_plot = interactive or (self.plots_dir is not None)
        plot_extent = self.plot_extent

        if self.post_selected_drift_models and self.plots_dir:
            primary_slug = slugify_drift_name(self.post_selected_drift_models[0].terms)
            primary_plots_dir = Path(self.plots_dir) / primary_slug
            primary_suffix = primary_slug
        else:
            primary_slug = slugify_drift_name(self.post_selected_drift_models[0].terms) if self.post_selected_drift_models else ""
            primary_plots_dir = self.plots_dir
            primary_suffix = ""

        if self.kriged_thickness is None:
            if self.kriged_bedrock is not None:
                self.kriged_thickness = np.maximum(self.dem_grid - self.kriged_bedrock, 0.0)
                if self.outline_mask is not None:
                    self.kriged_thickness[~self.outline_mask] = 0.0
            else:
                self.calculate_bedrock(interactive=False)

        cur_smooth_bedrock = bool(smooth_bedrock) if smooth_bedrock is not None else False
        cur_smoothing_method = str(smoothing_method or "gaussian")
        cur_smoothing_sigma = float(smoothing_sigma if smoothing_sigma is not None else 1.5)
        cur_smoothing_kernel_size = int(smoothing_kernel_size if smoothing_kernel_size is not None else 3)
        cur_smoothing_kc_cutoff = smoothing_kc_cutoff
        cur_rf = bool(random_forest_gap_filling) if random_forest_gap_filling is not None else False
        cur_blend = bool(apply_margin_blend) if apply_margin_blend is not None else False
        cur_min_gap_dist = float(min_gap_dist if min_gap_dist is not None else 50.0)

        if is_interactive:
            log_section_header("INTERACTIVE BEDROCK FINALIZATION")
            sys.stdout.write("Post-migration Kriging has produced the initial bedrock elevation map (figure 06).\n")
            sys.stdout.write("Configure finalization options below to refine the final embedded bedrock terrain.\n\n")
            sys.stdout.flush()

            while True:
                # Prompt 1: Spatial Bedrock Smoothing
                sys.stdout.write("--- Finalization Option 1: Bedrock / Depth Field Spatial Smoothing ---\n")
                sys.stdout.flush()
                cur_smooth_bedrock = _prompt_finalization_bool("Apply spatial smoothing to kriged depth/bedrock?", cur_smooth_bedrock)
                if cur_smooth_bedrock:
                    cur_smoothing_method = _prompt_finalization_choice(
                        "Smoothing method", ["gaussian", "median", "fft_lowpass"], cur_smoothing_method
                    )
                    if cur_smoothing_method == "gaussian":
                        cur_smoothing_sigma = _prompt_finalization_float("Gaussian smoothing sigma [px]", cur_smoothing_sigma, min_val=0.1)
                    elif cur_smoothing_method == "median":
                        cur_smoothing_kernel_size = _prompt_finalization_odd_int("Median kernel size (odd int)", cur_smoothing_kernel_size)
                    elif cur_smoothing_method == "fft_lowpass":
                        default_kc = float(self.opt_kc) if self.opt_kc is not None else 0.03
                        cur_kc_str = f"{cur_smoothing_kc_cutoff:.4f}" if cur_smoothing_kc_cutoff is not None else f"{default_kc:.4f}"
                        try:
                            ans_kc = input(f"Cutoff wavenumber k_c [rad/m] [{cur_kc_str}]: ").strip()
                            cur_smoothing_kc_cutoff = float(ans_kc) if ans_kc else (cur_smoothing_kc_cutoff or default_kc)
                        except Exception:
                            cur_smoothing_kc_cutoff = cur_smoothing_kc_cutoff or default_kc

                # Prompt 2: Machine Learning Gap Filling
                sys.stdout.write("\n--- Finalization Option 2: Machine Learning Gap Filling ---\n")
                sys.stdout.flush()
                cur_rf = _prompt_finalization_bool("Based on Kriging plots, apply Random Forest gap filling across voids?", cur_rf)

                # Prompt 3: Geomorphological Boundary Margin Blending
                sys.stdout.write("\n--- Finalization Option 3: Geomorphological Boundary Margin Blending ---\n")
                sys.stdout.flush()
                cur_blend = _prompt_finalization_bool("Apply margin blending to surrounding terrain (blend_margin_topography)?", cur_blend)
                if cur_blend:
                    cur_min_gap_dist = _prompt_finalization_float("Margin blending transition buffer width [m]", cur_min_gap_dist, min_val=1.0)

                sys.stdout.write("\n>> Computing candidate embedded bedrock topography...\n")
                sys.stdout.flush()

                # Execute candidate finalization pipeline:
                if cur_smooth_bedrock:
                    thick_raw = self.kriged_thickness.copy()
                    if self.outline_mask is not None:
                        thick_raw[~self.outline_mask] = 0.0
                    thick_smoothed = self.smooth_bedrock_dem(
                        thick_raw,
                        method=cur_smoothing_method,
                        sigma=cur_smoothing_sigma,
                        kernel_size=cur_smoothing_kernel_size,
                        kc_cutoff=cur_smoothing_kc_cutoff,
                    )
                    if self.outline_mask is not None:
                        thick_smoothed[~self.outline_mask] = 0.0
                    cand_bedrock = self.dem_grid - thick_smoothed
                else:
                    cand_bedrock = self.dem_grid - self.kriged_thickness
                    if self.outline_mask is not None:
                        cand_bedrock[~self.outline_mask] = self.dem_grid[~self.outline_mask]

                if cur_rf:
                    self.fill_holes_rf()
                    cand_bedrock = self.rf_filled_bedrock
                else:
                    self.rf_filled_bedrock = cand_bedrock.copy()

                if cur_blend:
                    if self.outline_mask is not None:
                        cand_blended = blend_margin_topography(
                            dem=self.dem_grid,
                            bedrock_input=cand_bedrock,
                            boundary_mask=self.outline_mask,
                            geometry=self.geometry,
                            min_gap_dist=cur_min_gap_dist,
                        )
                    else:
                        cand_blended = cand_bedrock.copy()
                else:
                    cand_blended = cand_bedrock.copy()

                self.blended_bedrock = cand_blended

                # Immediately update diagnostic plot 07 on disk
                if should_plot:
                    from .plotting import plot_final_blended_bedrock_map
                    plot_final_blended_bedrock_map(
                        blended_bedrock=self.blended_bedrock,
                        plot_extent=plot_extent,
                        plots_dir=primary_plots_dir,
                        interactive=interactive,
                        output_prefix=self.figure_prefix,
                        model_suffix=primary_suffix,
                    )

                cand_thickness = np.maximum(self.dem_grid - self.blended_bedrock, 0.0)
                if self.outline_mask is not None:
                    cand_thickness[~self.outline_mask] = 0.0
                finite_bed = self.blended_bedrock[np.isfinite(self.blended_bedrock)]
                min_bed = float(np.min(finite_bed)) if finite_bed.size > 0 else 0.0
                max_bed = float(np.max(finite_bed)) if finite_bed.size > 0 else 0.0
                active_thick = cand_thickness[self.outline_mask] if (self.outline_mask is not None and np.any(self.outline_mask)) else cand_thickness
                finite_thick = active_thick[np.isfinite(active_thick) & (active_thick > 0)]
                mean_th = float(np.mean(finite_thick)) if finite_thick.size > 0 else 0.0
                max_th = float(np.max(finite_thick)) if finite_thick.size > 0 else 0.0

                sys.stdout.write("Candidate Bedrock Summary:\n")
                sys.stdout.write(f"  - Bedrock elevation range : {min_bed:.1f} m to {max_bed:.1f} m\n")
                sys.stdout.write(f"  - Mean ice thickness      : {mean_th:.1f} m (max: {max_th:.1f} m)\n")
                sys.stdout.write(f"  - Spatial smoothing       : {cur_smooth_bedrock} ({cur_smoothing_method})\n")
                sys.stdout.write(f"  - Random Forest gap fill  : {cur_rf}\n")
                sys.stdout.write(f"  - Margin blend applied    : {cur_blend} (buffer: {cur_min_gap_dist:.1f} m)\n")
                if primary_plots_dir:
                    pfx_str = f"{self.figure_prefix}_" if self.figure_prefix else ""
                    sfx_str = f"_{primary_suffix.lstrip('_')}" if primary_suffix else ""
                    fig_path = Path(primary_plots_dir).resolve() / f"07_{pfx_str}final_embedded_bedrock{sfx_str}.png"
                    sys.stdout.write(f"  - Diagnostic figure 07    : {fig_path}\n")
                sys.stdout.flush()

                try:
                    ans_accept = input("\nAccept final embedded bedrock? [Y/n] (or 'r' to re-adjust parameters): ").strip().lower()
                except Exception:
                    ans_accept = "y"

                if ans_accept in ["", "y", "yes"]:
                    self.kriged_bedrock = cand_bedrock
                    end_terminal_block()
                    break
        else:
            # Single-pass deterministic execution using configured/passed arguments
            if cur_smooth_bedrock:
                logger.info(f"   [Spatial Smoothing] Applying depth field spatial smoothing ({cur_smoothing_method})...")
                thick_raw = self.kriged_thickness.copy()
                if self.outline_mask is not None:
                    thick_raw[~self.outline_mask] = 0.0

                thick_smoothed = self.smooth_bedrock_dem(
                    thick_raw,
                    method=cur_smoothing_method,
                    sigma=cur_smoothing_sigma,
                    kernel_size=cur_smoothing_kernel_size,
                    kc_cutoff=cur_smoothing_kc_cutoff,
                )
                if self.outline_mask is not None:
                    thick_smoothed[~self.outline_mask] = 0.0

                self.kriged_bedrock = self.dem_grid - thick_smoothed
            else:
                self.kriged_bedrock = self.dem_grid - self.kriged_thickness
                if self.outline_mask is not None:
                    self.kriged_bedrock[~self.outline_mask] = self.dem_grid[~self.outline_mask]

            if cur_rf:
                self.fill_holes_rf()
                current_bedrock = self.rf_filled_bedrock
            else:
                self.rf_filled_bedrock = self.kriged_bedrock.copy()
                current_bedrock = self.kriged_bedrock

            if cur_blend:
                logger.info(f"   [Margin Blending] Applying geomorphological margin blending (min_gap_dist = {cur_min_gap_dist:.1f} m)...")
                self.blended_bedrock = blend_margin_topography(
                    dem=self.dem_grid,
                    bedrock_input=current_bedrock,
                    boundary_mask=self.outline_mask,
                    geometry=self.geometry,
                    min_gap_dist=cur_min_gap_dist,
                )
            else:
                self.blended_bedrock = current_bedrock.copy()

            if should_plot:
                from .plotting import plot_final_blended_bedrock_map
                plot_final_blended_bedrock_map(
                    blended_bedrock=self.blended_bedrock,
                    plot_extent=plot_extent,
                    plots_dir=primary_plots_dir,
                    interactive=interactive,
                    output_prefix=self.figure_prefix,
                    model_suffix=primary_suffix,
                )

        self.final_thickness = np.maximum(self.dem_grid - self.blended_bedrock, 0.0)
        if self.outline_mask is not None:
            self.final_thickness[~self.outline_mask] = 0.0

        if self.kriged_variance is not None:
            self.kriged_std = np.sqrt(np.maximum(self.kriged_variance, 0.0))
            if self.outline_mask is not None:
                self.kriged_std[~self.outline_mask] = np.nan
        else:
            self.kriged_std = np.full_like(self.final_thickness, np.nan)
            if self.outline_mask is not None:
                self.kriged_std[~self.outline_mask] = np.nan

        # Domain & histogram means (strictly evaluated over positive glaciated cells H > 0 and tau_b > 0)
        mask_thick = (self.outline_mask & (self.final_thickness > 0)) if (self.outline_mask is not None and np.any(self.outline_mask)) else (self.final_thickness > 0)
        thick_valid = self.final_thickness[mask_thick & np.isfinite(self.final_thickness)]
        unc_valid = self.kriged_std[mask_thick & np.isfinite(self.kriged_std)]
        mean_thick = float(np.mean(thick_valid)) if thick_valid.size > 0 else 0.0
        mean_unc = float(np.mean(unc_valid)) if unc_valid.size > 0 else 0.0

        # Basal Shear Stress Calculation (tb = rho_ice * g * D * sin(alpha) in kPa).
        # Deliberately the UNFLOORED slope: the slope floor only stabilizes the P <-> D/T
        # conversions; applying it here would bias tb high in cells flatter than the floor.
        sin_alpha_opt = self.slope_sin
        self.final_bss = (self.ice_density * self.g * self.final_thickness * sin_alpha_opt) / 1000.0
        self.bss_std = (self.ice_density * self.g * self.kriged_std * sin_alpha_opt) / 1000.0

        if self.outline_mask is not None:
            self.final_bss[~self.outline_mask] = np.nan
            self.bss_std[~self.outline_mask] = np.nan

        mask_bss = (self.outline_mask & (self.final_bss > 0)) if (self.outline_mask is not None and np.any(self.outline_mask)) else (self.final_bss > 0)
        bss_valid = self.final_bss[mask_bss & np.isfinite(self.final_bss)]
        bss_unc_valid = self.bss_std[mask_bss & np.isfinite(self.bss_std)]
        mean_bss = float(np.mean(bss_valid)) if bss_valid.size > 0 else 0.0
        mean_bss_unc = float(np.mean(bss_unc_valid)) if bss_unc_valid.size > 0 else 0.0

        # Ice volume, glacier domain area, and propagated uncertainty
        prim_vol = calculate_ice_volume(
            self.final_thickness,
            self.dx,
            self.dy,
            self.outline_mask,
            self.kriged_std if self.post_calculate_uncertainty else None,
        )
        self.volume_result = prim_vol
        self.glacier_area_m2 = prim_vol.area_m2
        self.glacier_area_km2 = prim_vol.area_km2
        self.total_volume_m3 = prim_vol.volume_m3
        self.total_volume_km3 = prim_vol.volume_km3
        self.total_volume_unc_m3 = prim_vol.uncertainty_m3
        self.total_volume_unc_km3 = prim_vol.uncertainty_km3

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
                plots_dir=primary_plots_dir,
                interactive=interactive,
                output_prefix=self.figure_prefix,
                model_suffix=primary_suffix,
            )
            plot_final_ice_thickness_histogram(
                final_thickness=self.final_thickness,
                outline_mask=self.outline_mask,
                mean_thick=mean_thick,
                plots_dir=primary_plots_dir,
                interactive=interactive,
                output_prefix=self.figure_prefix,
                model_suffix=primary_suffix,
                total_volume_km3=prim_vol.volume_km3,
                total_volume_unc_km3=prim_vol.uncertainty_km3,
                kriged_std=self.kriged_std if self.post_calculate_uncertainty else None,
                mean_unc=mean_unc if self.post_calculate_uncertainty else None,
            )
            plot_final_basal_shear_stress_and_uncertainty(
                final_bss=self.final_bss,
                bss_std=self.bss_std,
                outline_mask=self.outline_mask,
                plot_extent=plot_extent,
                mean_bss=mean_bss,
                mean_bss_unc=mean_bss_unc,
                plots_dir=primary_plots_dir,
                interactive=interactive,
                output_prefix=self.figure_prefix,
                model_suffix=primary_suffix,
            )
            plot_final_basal_shear_stress_histogram(
                final_bss=self.final_bss,
                outline_mask=self.outline_mask,
                mean_bss=mean_bss,
                plots_dir=primary_plots_dir,
                interactive=interactive,
                output_prefix=self.figure_prefix,
                model_suffix=primary_suffix,
            )

        self.final_grid = self.blended_bedrock

        if self.post_selected_drift_models:
            self.export_outputs("finalization", model_slug=primary_slug)
            records: list[DriftComparisonRecord] = []
            summary_cards: list[str] = []
            prim_cand = self.post_selected_drift_models[0]
            prim_rank = prim_cand.rank or 1
            h_mean, h_med, h_std, h_max = _compute_distribution_stats(self.final_thickness, self.outline_mask)
            b_mean, b_med, b_std, b_max = _compute_distribution_stats(self.final_bss, self.outline_mask)
            total_cands = len(self.post_selected_drift_models)
            if total_cands > 1:
                summary_cards.append(
                    format_final_result_summary(
                        volume_res=prim_vol,
                        thick_stats=(h_mean, h_med, h_std, h_max),
                        bss_stats=(b_mean, b_med, b_std, b_max),
                        model_name=prim_cand.name,
                        model_slug=primary_slug,
                        model_terms=prim_cand.terms,
                        model_rank=prim_rank,
                        candidate_index=1,
                        candidate_total=total_cands,
                        bss_mean_unc=mean_bss_unc if self.post_calculate_uncertainty else None,
                    )
                )
            else:
                summary_cards.append(
                    format_final_result_summary(
                        volume_res=prim_vol,
                        thick_stats=(h_mean, h_med, h_std, h_max),
                        bss_stats=(b_mean, b_med, b_std, b_max),
                        model_name=prim_cand.name,
                        model_slug=primary_slug,
                        model_terms=prim_cand.terms,
                        model_rank=prim_rank,
                        bss_mean_unc=mean_bss_unc if self.post_calculate_uncertainty else None,
                    )
                )
            records.append(
                DriftComparisonRecord(
                    pass_label="Pass 2",
                    rank=prim_rank,
                    model_name=prim_cand.name,
                    model_slug=primary_slug,
                    terms=prim_cand.terms,
                    cv_rmse=prim_cand.cv_rmse,
                    delta_rmse_pct=prim_cand.delta_rmse_pct,
                    aicc=prim_cand.aicc,
                    delta_aicc=prim_cand.delta_aicc,
                    vif_max=prim_cand.vif_max,
                    status="PENALIZED" if prim_cand.is_penalized else ("WARNING" if prim_cand.warning_msg else "VALID"),
                    rmse_unit="m",
                    thick_mean=h_mean,
                    thick_median=h_med,
                    thick_std=h_std,
                    thick_max=h_max,
                    bss_mean=b_mean,
                    bss_median=b_med,
                    bss_std=b_std,
                    bss_max=b_max,
                    unc_mean=mean_unc if self.post_calculate_uncertainty else None,
                    is_failed=False,
                )
            )

            pts = self.migrated_points if self.migrated_points is not None else self.survey_points
            val_col = 3 if pts.shape[1] >= 4 else 2
            valid_pts_val = pts[~np.isnan(pts[:, val_col]), val_col]
            max_thickness = max(float(np.max(valid_pts_val)) * 1.5, 500.0) if len(valid_pts_val) > 0 else 500.0

            target_upper = self.post_interpolation_target.upper().strip()
            sample_pts = self.get_sample_points("post_migration", target_upper)
            eff_var = self._effective_variogram_params(sample_pts, target_upper)

            orig_thickness = self.final_thickness
            orig_bss = self.final_bss
            orig_grid = self.final_grid
            orig_std = self.kriged_std
            orig_bss_std = self.bss_std

            for idx, cand in enumerate(self.post_selected_drift_models[1:], start=2):
                cand_slug = slugify_drift_name(cand.terms)
                cand_rank = cand.rank or idx
                logger.info(f"   [Drift Comparison] [Pass 2] Evaluating candidate model '{cand.name}'...")
                ext_drifts = self.get_drift_grids(cand.terms)
                cand_method = "universal" if cand.terms else "ordinary"
                try:
                    k_cand = kriging_interpolation(
                        sample_points=sample_pts,
                        geometry=self.geometry,
                        method=cand_method,
                        variogram_model=self.post_variogram_model,
                        variogram_params=eff_var,
                        dem_grid=self.dem_grid,
                        opt_slope_grid=self.opt_slope,
                        drift_terms=cand.terms,
                        outline_mask=self.outline_mask,
                        include_zero_boundary_condition=self.post_zero_boundary,
                        n_cores=self.n_cores,
                        engine=self.engine_type,
                        slope_floor_deg=self.slope_floor_deg,
                        show_progress=self.show_progress,
                        external_drift_grid=ext_drifts if len(ext_drifts) > 0 else None,
                        return_variance=self.post_calculate_uncertainty,
                        max_memory_fraction=self.max_memory_fraction,
                    )
                    if target_upper == "P":
                        cand_raw = np.maximum(k_cand.bedrock_grid / self.safe_slope_sin, 0.0)
                        cand_var = k_cand.variance_grid / (self.safe_slope_sin**2) if k_cand.variance_grid is not None else None
                    else:
                        cand_raw = k_cand.bedrock_grid
                        cand_var = k_cand.variance_grid

                    cand_thick = np.clip(cand_raw, 0.0, max_thickness)
                    if self.outline_mask is not None:
                        cand_thick[~self.outline_mask] = 0.0

                    cand_bed = self.dem_grid - cand_thick
                    if self.outline_mask is not None:
                        cand_bed[~self.outline_mask] = self.dem_grid[~self.outline_mask]

                    cand_var_masked = cand_var.copy() if cand_var is not None else None
                    if cand_var_masked is not None and self.outline_mask is not None:
                        cand_var_masked[~self.outline_mask] = np.nan
                    cand_kriged_std = np.sqrt(np.maximum(cand_var_masked, 0.0)) if (cand_var_masked is not None and not np.all(np.isnan(cand_var_masked))) else np.full_like(cand_thick, np.nan)
                    if self.outline_mask is not None:
                        cand_kriged_std[~self.outline_mask] = np.nan

                    if cur_smooth_bedrock:
                        th_raw = cand_thick.copy()
                        if self.outline_mask is not None:
                            th_raw[~self.outline_mask] = 0.0
                        th_smooth = self.smooth_bedrock_dem(
                            th_raw,
                            method=cur_smoothing_method,
                            sigma=cur_smoothing_sigma,
                            kernel_size=cur_smoothing_kernel_size,
                            kc_cutoff=cur_smoothing_kc_cutoff,
                        )
                        if self.outline_mask is not None:
                            th_smooth[~self.outline_mask] = 0.0
                        cand_bed_post = self.dem_grid - th_smooth
                    else:
                        cand_bed_post = cand_bed

                    if cur_rf:
                        cand_rf_bed = self.finalizer.fill_holes(
                            cand_bed_post,
                            n_cores=self.n_cores,
                            show_progress=False,
                            extra_feature_grids=self._extra_rf_feature_grids(),
                        )
                    else:
                        cand_rf_bed = cand_bed_post

                    if cur_blend:
                        cand_blended = blend_margin_topography(
                            dem=self.dem_grid,
                            bedrock_input=cand_rf_bed,
                            boundary_mask=self.outline_mask,
                            geometry=self.geometry,
                            min_gap_dist=cur_min_gap_dist,
                        )
                    else:
                        cand_blended = cand_rf_bed.copy()

                    cand_final_thick = np.maximum(self.dem_grid - cand_blended, 0.0)
                    if self.outline_mask is not None:
                        cand_final_thick[~self.outline_mask] = 0.0

                    cand_final_bss = (self.ice_density * self.g * cand_final_thick * self.slope_sin) / 1000.0
                    cand_bss_std = (self.ice_density * self.g * cand_kriged_std * self.slope_sin) / 1000.0
                    if self.outline_mask is not None:
                        cand_final_bss[~self.outline_mask] = np.nan
                        cand_bss_std[~self.outline_mask] = np.nan

                    cand_mask_thick = (self.outline_mask & (cand_final_thick > 0)) if (self.outline_mask is not None and np.any(self.outline_mask)) else (cand_final_thick > 0)
                    c_th_valid = cand_final_thick[cand_mask_thick & np.isfinite(cand_final_thick)]
                    c_unc_valid = cand_kriged_std[cand_mask_thick & np.isfinite(cand_kriged_std)]
                    c_mean_thick = float(np.mean(c_th_valid)) if c_th_valid.size > 0 else 0.0
                    c_mean_unc = float(np.mean(c_unc_valid)) if c_unc_valid.size > 0 else 0.0

                    cand_mask_bss = (self.outline_mask & (cand_final_bss > 0)) if (self.outline_mask is not None and np.any(self.outline_mask)) else (cand_final_bss > 0)
                    c_bss_valid = cand_final_bss[cand_mask_bss & np.isfinite(cand_final_bss)]
                    c_bss_unc_valid = cand_bss_std[cand_mask_bss & np.isfinite(cand_bss_std)]
                    c_mean_bss = float(np.mean(c_bss_valid)) if c_bss_valid.size > 0 else 0.0
                    c_mean_bss_unc = float(np.mean(c_bss_unc_valid)) if c_bss_unc_valid.size > 0 else 0.0

                    cand_vol = calculate_ice_volume(
                        cand_final_thick,
                        self.dx,
                        self.dy,
                        self.outline_mask,
                        cand_kriged_std if self.post_calculate_uncertainty else None,
                    )

                    cand_plots_dir = Path(self.plots_dir) / cand_slug if self.plots_dir else None
                    if cand_plots_dir:
                        from .plotting import (
                            plot_kriging_bedrock_and_uncertainty,
                            plot_final_blended_bedrock_map,
                            plot_final_ice_thickness_and_uncertainty,
                            plot_final_ice_thickness_histogram,
                            plot_final_basal_shear_stress_and_uncertainty,
                            plot_final_basal_shear_stress_histogram,
                        )
                        plot_kriging_bedrock_and_uncertainty(
                            kriged_bedrock=cand_bed,
                            kriged_std=cand_kriged_std,
                            pts=pts,
                            plot_extent=plot_extent,
                            plots_dir=cand_plots_dir,
                            interactive=False,
                            output_prefix=self.figure_prefix,
                            model_suffix=cand_slug,
                        )
                        plot_final_blended_bedrock_map(
                            blended_bedrock=cand_blended,
                            plot_extent=plot_extent,
                            plots_dir=cand_plots_dir,
                            interactive=False,
                            output_prefix=self.figure_prefix,
                            model_suffix=cand_slug,
                        )
                        plot_final_ice_thickness_and_uncertainty(
                            final_thickness=cand_final_thick,
                            kriged_std=cand_kriged_std,
                            outline_mask=self.outline_mask,
                            plot_extent=plot_extent,
                            mean_thick=c_mean_thick,
                            mean_unc=c_mean_unc,
                            plots_dir=cand_plots_dir,
                            interactive=False,
                            output_prefix=self.figure_prefix,
                            model_suffix=cand_slug,
                        )
                        plot_final_ice_thickness_histogram(
                            final_thickness=cand_final_thick,
                            outline_mask=self.outline_mask,
                            mean_thick=c_mean_thick,
                            plots_dir=cand_plots_dir,
                            interactive=False,
                            output_prefix=self.figure_prefix,
                            model_suffix=cand_slug,
                            total_volume_km3=cand_vol.volume_km3,
                            total_volume_unc_km3=cand_vol.uncertainty_km3,
                            kriged_std=cand_kriged_std if self.post_calculate_uncertainty else None,
                            mean_unc=c_mean_unc if self.post_calculate_uncertainty else None,
                        )
                        plot_final_basal_shear_stress_and_uncertainty(
                            final_bss=cand_final_bss,
                            bss_std=cand_bss_std,
                            outline_mask=self.outline_mask,
                            plot_extent=plot_extent,
                            mean_bss=c_mean_bss,
                            mean_bss_unc=c_mean_bss_unc,
                            plots_dir=cand_plots_dir,
                            interactive=False,
                            output_prefix=self.figure_prefix,
                            model_suffix=cand_slug,
                        )
                        plot_final_basal_shear_stress_histogram(
                            final_bss=cand_final_bss,
                            outline_mask=self.outline_mask,
                            mean_bss=c_mean_bss,
                            plots_dir=cand_plots_dir,
                            interactive=False,
                            output_prefix=self.figure_prefix,
                            model_suffix=cand_slug,
                        )

                    self.final_thickness = cand_final_thick
                    self.final_bss = cand_final_bss
                    self.final_grid = cand_blended
                    self.kriged_std = cand_kriged_std
                    self.bss_std = cand_bss_std
                    self.export_outputs("finalization", model_slug=cand_slug)

                    ch_mean, ch_med, ch_std, ch_max = _compute_distribution_stats(cand_final_thick, self.outline_mask)
                    cb_mean, cb_med, cb_std, cb_max = _compute_distribution_stats(cand_final_bss, self.outline_mask)
                    cand_unc_mean = c_mean_unc if self.post_calculate_uncertainty else None
                    failed = False
                    summary_cards.append(
                        format_final_result_summary(
                            volume_res=cand_vol,
                            thick_stats=(ch_mean, ch_med, ch_std, ch_max),
                            bss_stats=(cb_mean, cb_med, cb_std, cb_max),
                            model_name=cand.name,
                            model_slug=cand_slug,
                            model_terms=cand.terms,
                            model_rank=cand_rank,
                            candidate_index=idx,
                            candidate_total=total_cands,
                            bss_mean_unc=c_mean_bss_unc if self.post_calculate_uncertainty else None,
                        )
                    )
                except Exception as e:
                    logger.warning(f"   [Drift Comparison] Interpolation failed for '{cand.name}': {e}")
                    ch_mean, ch_med, ch_std, ch_max = 0.0, 0.0, 0.0, 0.0
                    cb_mean, cb_med, cb_std, cb_max = 0.0, 0.0, 0.0, 0.0
                    cand_unc_mean = None
                    failed = True

                records.append(
                    DriftComparisonRecord(
                        pass_label="Pass 2",
                        rank=cand_rank,
                        model_name=cand.name,
                        model_slug=cand_slug,
                        terms=cand.terms,
                        cv_rmse=cand.cv_rmse,
                        delta_rmse_pct=cand.delta_rmse_pct,
                        aicc=cand.aicc,
                        delta_aicc=cand.delta_aicc,
                        vif_max=cand.vif_max,
                        status="FAILED" if failed else ("PENALIZED" if cand.is_penalized else ("WARNING" if cand.warning_msg else "VALID")),
                        rmse_unit="m",
                        thick_mean=ch_mean,
                        thick_median=ch_med,
                        thick_std=ch_std,
                        thick_max=ch_max,
                        bss_mean=cb_mean,
                        bss_median=cb_med,
                        bss_std=cb_std,
                        bss_max=cb_max,
                        unc_mean=cand_unc_mean,
                        is_failed=failed,
                    )
                )

            self.final_thickness = orig_thickness
            self.final_bss = orig_bss
            self.final_grid = orig_grid
            self.kriged_std = orig_std
            self.bss_std = orig_bss_std

            self.post_drift_comparison_records = records
            target_out_dir = self.effective_output_dir
            if target_out_dir:
                save_drift_comparison_files(
                    target_out_dir,
                    pre_records=self.pre_drift_comparison_records,
                    post_records=self.post_drift_comparison_records,
                )
            for card in summary_cards:
                print(f"\n{card}")
            print(format_drift_comparison_table(records, "Pass 2"))
        else:
            self.export_outputs("finalization")
            h_stats = _compute_distribution_stats(self.final_thickness, self.outline_mask)
            b_stats = _compute_distribution_stats(self.final_bss, self.outline_mask)
            disp_model_name = format_drift_name(self.post_drift_terms) if self.post_drift_terms is not None else "Ordinary Kriging"
            summary_card = format_final_result_summary(
                volume_res=prim_vol,
                thick_stats=h_stats,
                bss_stats=b_stats,
                model_name=disp_model_name,
                model_terms=self.post_drift_terms,
                bss_mean_unc=mean_bss_unc if self.post_calculate_uncertainty else None,
            )
            print(f"\n{summary_card}")

        return BedrockMap(
            grid=self.final_grid,
            bounds=self.bounds,
            crs=self.meta.get("crs"),
            transform=self.meta.get("transform"),
            name="final_bedrock",
        )

    def recommend_drift_model(
        self,
        stage: str = "post_migration",
        interactive: bool = False,
    ) -> list[CandidateDriftResult]:
        """Runs Universal Kriging Drift Analyzer diagnostics to recommend optimal drift functions.

        Evaluates the complete dynamic default suite - Ordinary Kriging plus all non-redundant drift models
        built from the active primitives (`sia`, `z_dem`, `curvature_dem`, `linear_xy`, `quadratic_xy`, plus
        `mass_balance`, `velocity`, and `velocity_sia` when configured; 24, 48, 60, or 120 models, see
        :func:`pysole.drift.build_drift_suite`) - independent of the configured method and drift_terms, using
        Variance Inflation Factor (VIF), Akaike Information Criterion (AICc), and Leave-One-Profile-Out
        (LOPO) or spatial Buffer Leave-One-Out Cross-Validation.

        Parameters
        ----------
        stage : str, default 'post_migration'
            Workflow stage to evaluate ('pre_migration' or 'post_migration').
        interactive : bool, default False
            If True, generates visual diagnostic figures and prompt dialogues.

        Returns
        -------
        diagnostics : list of CandidateDriftResult
            Evaluated drift candidates ranked from best to worst (not penalized first, then by AICc), with the
            ΔAICc and ΔRMSE % columns. The stage's variogram model and the fitted variogram parameters of the
            production run are used.
        """
        stage_prefix = stage.split("_")[0]
        stage_key = "pre_migration" if ("pre" in stage.lower() or "1" in stage) else "post_migration"
        self._ensure_optimized(stage_key, interactive=interactive)
        interp_target = getattr(self, f"{stage_prefix}_interpolation_target", "P")
        sample_pts = self.get_sample_points(stage_key, interp_target)

        effective_var_params = self._effective_variogram_params(sample_pts, interp_target.upper())

        return self._run_drift_analysis(
            stage_key=stage_key,
            target_type=interp_target,
            sample_pts=sample_pts,
            zero_boundary=getattr(self, f"{stage_prefix}_zero_boundary", True),
            var_model=getattr(self, f"{stage_prefix}_variogram_model", "spherical"),
            suite=build_drift_suite(None, available_optional_terms=self._available_optional_terms()),
            interactive=interactive,
            variogram_params=effective_var_params,
        )

    def plan_survey(
        self,
        kc: float | None = None,
        lambda_c: float | None = None,
        tau_0: float | None = None,
        max_length_km: float | None = None,
        output_prefix: str | None = None,
        output_dir: str | Path | None = None,
        plots_dir: str | Path | None = None,
        output_format: str | None = None,
        interactive: bool | None = None,
    ) -> dict[str, Any]:
        """Executes forward survey planning for unprobed glaciers using synthetic SIA modeling.

        Calculates theoretical ice thickness via the Shallow Ice Approximation:
          D_SIA(x, y) = tau_0 / (rho_ice * g * sin(alpha_opt))
        and automatically places optimal radar survey profiles (transverse cross-glacier lines
        and longitudinal centerlines) oriented to maximize basal topography resolution while
        respecting maximum path length constraints.

        Parameters
        ----------
        kc : float, optional
            Spatial cutoff wavenumber k_c [rad/m] for filtering surface slopes.
        lambda_c : float, optional
            Spatial cutoff wavelength lambda_c [m] for filtering surface slopes (default: 200.0 m).
        tau_0 : float, optional
            Assumed target basal shear stress threshold [Pa] (default: 100 kPa = 100,000 Pa).
        max_length_km : float, optional
            Maximum cumulative distance per survey loop [km] (default: 5.0 km).
        output_prefix : str, optional
            Filename prefix for exported survey track files (GPX, GeoJSON, CSV).
        output_dir : str or Path, optional
            Output filesystem directory for survey tracks.
        plots_dir : str or Path, optional
            Output filesystem directory for diagnostic planning maps.
        output_format : str, optional
            Format for synthetic raster export ('tif', 'asc').
        interactive : bool, optional
            If True and running in an interactive terminal, prompts for any omitted parameters.
            If None, defaults to False in batch mode and True otherwise.

        Returns
        -------
        survey_results : dict of str to Any
            Dictionary containing:
              - 'sia_modelled_depth' : 2D array of synthetic SIA ice thickness [m].
              - 'transverse_lines' : List of Shapely LineString geometries for cross profiles.
              - 'centerline' : Shapely LineString geometry for glacier flow centerline.
              - 'waypoints' : List of planned survey waypoint coordinates.
              - 'saved_files' : List of exported track file paths (GPX, GeoJSON, CSV).
        """
        cfg = getattr(self, "config", None) or {}
        outputs = cfg.get("outputs", {})

        prefix = output_prefix if output_prefix is not None else outputs.get("output_prefix", "")
        raw_prefix = (prefix or "").strip()
        explicit_out_dir = output_dir or self.output_dir or outputs.get("output_dir")
        if explicit_out_dir:
            out_d = str(resolve_output_dir(explicit_out_dir, config_path=self.config_path))
        elif raw_prefix and Path(raw_prefix).expanduser().is_absolute():
            out_d = str(Path(raw_prefix).expanduser().parent)
        else:
            out_d = str(resolve_output_dir(None, config_path=self.config_path))
        raw_plots = plots_dir if plots_dir is not None else self._raw_plots_dir
        plots_d = resolve_plots_dir(raw_plots, output_prefix=raw_prefix or None, output_dir=out_d, config_path=self.config_path)
        fmt = output_format or outputs.get("output_format", "tif")
        if isinstance(fmt, list):
            fmt = fmt[0]

        if interactive is None:
            interactive = not getattr(self, "is_batch_mode", False)

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
            lambda_c=lambda_c,
            tau_0=tau_0,
            max_length_km=max_length_km,
            output_prefix=prefix,
            output_dir=out_d,
            plots_dir=plots_d,
            output_format=fmt,
            interactive=interactive,
        )

    def run_pipeline(
        self,
        survey_data_path: str | Path | os.PathLike | None = None,
        interactive: bool | None = None,
    ) -> BedrockMap:
        """Executes the complete 5-step PySole glaciological workflow.

        Automates the full scientific processing sequence:
          1. Loading and validation of surface DEM, boundary outline, and survey soundings.
          2. First-pass traveltime interpolation and 3D Eikonal normal ray migration.
          3. Basal Shear Stress (BSS) variance optimization to find optimal spectral slope smoothing k_c.
          4. Second-pass Dual Kriging predicting continuous ice thickness and estimation uncertainty.
          5. Topography finalization (depth smoothing, RF void filling, margin blending, raster export).

        If no survey data path is provided, automatically runs the Unprobed Glacier Survey Planner.

        Parameters
        ----------
        survey_data_path : str, Path, or os.PathLike, optional
            Path to survey observation file. Overrides configured survey_data_path attribute.
        interactive : bool, optional
            Whether to prompt interactively on the terminal. If None, automatically determined
            by batch mode and terminal capabilities.

        Returns
        -------
        bedrock_map : BedrockMap
            Final predicted bedrock elevation grid [m] wrapped with spatial georeferencing metadata.
        """
        if survey_data_path is not None:
            self.survey_data_path = str(survey_data_path)

        cfg = getattr(self, "config", {})
        inputs = cfg.get("inputs", {})
        migration = cfg.get("migration_parameters", {})
        opt = cfg.get("optimization_parameters", {})
        fin_cfg = cfg.get("finalization_parameters", {})

        survey_data_path = self.survey_data_path or inputs.get("survey_data_path")
        if not survey_data_path:
            logger.info("   [Survey Planner] No survey_data_path provided. Automatically switching execution mode to Unprobed Glacier Survey Planner.")
            is_interactive = not getattr(self, "is_batch_mode", False) if interactive is None else bool(interactive)
            res = self.plan_survey(interactive=is_interactive)
            return BedrockMap(grid=res["sia_modelled_depth"], bounds=self.bounds, crs=self.meta.get("crs"), transform=self.meta.get("transform"), name="sia_modelled_depth")

        cfg_name = getattr(self, "config_path", "pysole.json")
        logger.info("================================================================================")
        logger.info(f"       STARTING NEW PYSOLE BEDROCK TOPOGRAPHY CALCULATION ({cfg_name})")
        logger.info("================================================================================")
        logger.info(f"1. Initializing PySole Solver from '{cfg_name}'...")
        logger.info(f"   DEM Resolution: dx = {self.dx} m, dy = {self.dy} m")
        logger.info(f"   Survey Data Type: {self.survey_data_type}")
        logger.info(f"   Pre-Migration Kriging: {self._drift_plan('pre_migration').describe()} (variogram = {self.pre_variogram_model})")
        logger.info(f"   Post-Migration Kriging: {self._drift_plan('post_migration').describe()} (variogram = {self.post_variogram_model})")
        logger.info(f"   Perform Migration: {self.perform_migration}")
        logger.info(f"   Interactive Migration: {migration.get('interactive_migration', False)}")
        logger.info(f"   Plots Output Directory: {self.plots_dir}")

        is_batch = (not self.show_progress) or (not sys.stdin.isatty())
        opt_interactive = False if is_batch else opt.get("interactive_optimization", False)
        mig_interactive = False if is_batch else migration.get("interactive_migration", False)
        fin_interactive = False if is_batch else fin_cfg.get("interactive_finalization", False)
        if self.survey_points is None and (survey_data_path is not None or self.survey_data_path is not None):
            self.load_survey_points(survey_data_path)

        if self.perform_migration and self._normalize_survey_data_type() != "depth":
            logger.info("2. Performing Pre-Migration BSS Slope Optimization & Traveltime Variogram Modeling...")
            self.optimize_bss(stage="pre_migration", interactive=opt_interactive, plotit=True)
            logger.info("3. Performing Pass 1 Kriging & 3D Eikonal Ray Migration...")
        else:
            logger.info("2. Skipping 3D Eikonal Ray Migration...")

        self.migrate_eikonal(
            travel_times=survey_data_path,
            velocity=migration.get("velocity"),
            interactive=mig_interactive,
            interactive_optimization=opt_interactive,
            plotit=True,
        )

        logger.info("4. Performing Post-Migration BSS Slope Optimization & Depth Interpolation (Field D(x,y))...")
        opt_kc = self.optimize_bss(
            stage="post_migration",
            interactive=opt_interactive,
            plotit=True,
        )
        opt_wl = compute_cutoff_wavelength(opt_kc)
        logger.info(f"   [BSS Optimization] Optimal Post-Migration Corner Frequency k_c = {opt_kc:.4f} (cutoff wavelength λ_c = {opt_wl:.2f} m)")
        self.calculate_bedrock(interactive=opt_interactive, interactive_optimization=opt_interactive)

        logger.info("5. Finalizing Bedrock Topography...")
        bedrock_map = self.finalize_bedrock(
            interactive=fin_interactive,
            random_forest_gap_filling=fin_cfg.get("random_forest_gap_filling", False),
            apply_margin_blend=fin_cfg.get("apply_margin_blend", False),
            min_gap_dist=fin_cfg.get("min_gap_dist", 50.0),
            smooth_bedrock=fin_cfg.get("smooth_bedrock", False),
            smoothing_method=fin_cfg.get("smoothing_method", "gaussian"),
            smoothing_sigma=fin_cfg.get("smoothing_sigma", 1.5),
            smoothing_kernel_size=fin_cfg.get("smoothing_kernel_size", 3),
            smoothing_kc_cutoff=fin_cfg.get("smoothing_kc_cutoff", None),
        )

        # Migration-stage outputs were exported by migrate_eikonal() (only if a migration actually ran) and
        # finalization-stage outputs (bedrock raster, thickness, BSS, uncertainties) by finalize_bedrock().

        return bedrock_map

