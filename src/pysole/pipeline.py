"""
Pipeline Orchestration and File Export Module for PySole.
Handles configuration ingestion, workspace path resolution, output export management,
and high-level execution wrappers.
"""

from pathlib import Path
import os
import sys
from typing import Any
import numpy as np

from .config import load_config, resolve_output_dir, configured_output_dir
from .logging import logger, setup_logging, print_welcome_banner
from .raster import BedrockMap


class PipelineExporter:
    """
    Handles disk export operations for PySole output grids and survey points.

    Coordinates saving raster datasets (GeoTIFF, ASCII Grid, CSV, NumPy) and point
    tables (scattered migrated radar soundings) to disk according to the configuration
    settings.

    Parameters
    ----------
    solver : Any
        Instance of `pysole.solver.Solver` containing computational results and export methods.
    """

    def __init__(self, solver: Any):
        self.solver = solver
        self.outputs_cfg = solver.outputs_config_obj

    def get_resolved_output_prefix(self) -> str:
        """
        Resolve the base filesystem path prefix for all exported output files.

        Returns
        -------
        str
            Absolute base path prefix combining output directory and file prefix.
        """
        return self.solver._get_resolved_output_prefix()

    def export_raster(self, grid: np.ndarray | None, suffix: str, name: str) -> str | list[str] | None:
        """
        Export a 2D numerical raster grid to disk in configured formats.

        Parameters
        ----------
        grid : np.ndarray or None
            2D grid array of shape (M_rows, N_cols) to export. If None, no file is written.
        suffix : str
            File naming suffix (e.g., '_kriging_bedrock', '_ice_thickness').
        name : str
            Descriptive dataset name for logging and metadata.

        Returns
        -------
        saved_paths : str, list of str, or None
            Path(s) of exported files, or None if grid is None.
        """
        return self.solver._export_optional_raster(grid, suffix=suffix, name=name)

    def export_points_csv(self, points: np.ndarray | None, suffix: str) -> str | None:
        """
        Export scattered point coordinates and values to a CSV file.

        Parameters
        ----------
        points : np.ndarray or None
            2D array of point data (e.g., shape (N, 4) for [x, y, z_surf, depth]). If None,
            no export is performed.
        suffix : str
            File naming suffix (e.g., '_migrated_points.csv').

        Returns
        -------
        saved_path : str or None
            Path to the saved CSV file, or None if points is None.
        """
        return self.solver._export_optional_points_csv(points, suffix=suffix)

    def export_all(self, stage: str | None = None) -> list[str]:
        """
        Export all active output rasters and point datasets for a given pipeline stage.

        Parameters
        ----------
        stage : str, optional
            Pipeline stage identifier: 'migration', 'variogram', 'kriging', 'finalization',
            or None to export all currently available results.

        Returns
        -------
        saved_files : list of str
            List of absolute filesystem paths to all created files.
        """
        return self.solver.export_outputs(stage=stage)


def run_from_config(
    config_path: str | Path = "pysole.json",
    log_level: str | None = None,
    is_batch: bool = False,
    drift_analyzer: bool | None = None,
    survey_profile_column: str | None = None,
    show_banner: bool = True,
) -> BedrockMap:
    """
    Executes the complete PySole glaciological processing pipeline from a configuration file.

    Parameters
    ----------
    config_path : str or Path, default "pysole.json"
        File path to the JSON configuration template.
    log_level : str, optional
        Logging level override ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    is_batch : bool, default False
        If True, disables all interactive terminal prompts and variogram optimization dialogs.
    drift_analyzer : bool, optional
        If True, enables the Drift Analyzer of both Kriging stages (equivalent to ``drift_analyzer: true`` in
        the configuration); it only takes effect with method 'universal'. None keeps the configured values.
    survey_profile_column : str, optional
        Survey profile column name for spatial cross-validation.
    show_banner : bool, default True
        If True and not in batch mode, prints the centered PySole welcome banner to stdout.

    Returns
    -------
    BedrockMap
        The final predicted bedrock elevation grid and spatial metadata object.

    Examples
    --------
    >>> import pysole
    >>> bedrock = pysole.run_from_config("examples/wuk/pysole_wuk.json")
    >>> print(bedrock.shape, bedrock.bounds)
    >>> bedrock.save("wuk_final_bedrock.tif")
    """
    if show_banner and not is_batch:
        print_welcome_banner()

    from .solver import Solver

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file.resolve()}")

    cfg = load_config(config_file, log_level=log_level, drift_analyzer=drift_analyzer)
    inputs_cfg = cfg.get("inputs", {})
    outputs_cfg = cfg.get("outputs", {})

    log_lvl = log_level or os.environ.get("PYSOLE_LOG_LEVEL") or inputs_cfg.get("log_level", "INFO")

    output_dir = resolve_output_dir(
        output_dir=configured_output_dir(cfg),
        survey_data_path=inputs_cfg.get("survey_data_path"),
        config_path=config_file,
    )
    log_file = output_dir / "pysole.log"

    setup_logging(log_file=log_file, log_level=log_lvl, log_init_msg=False)

    try:
        survey_data_path = inputs_cfg.get("survey_data_path")
        is_unprobed = (survey_data_path is None) or (isinstance(survey_data_path, str) and not survey_data_path.strip())
        if is_unprobed:
            logger.info("=" * 80)
            logger.info(f"        STARTING PYSOLE UNPROBED GLACIER SURVEY PLANNER ({config_file.resolve()})")
            logger.info("=" * 80)
            logger.info(f"Initialized PySole log file at: {log_file.resolve()}")
            logger.info("   [INFO] No survey_data_path provided. Automatically switching execution mode to Unprobed Glacier Survey Planner.")

            solver = Solver.from_config(cfg, config_file=config_file, log_level=log_level, drift_analyzer=drift_analyzer)
            if is_batch:
                solver.show_progress = False
                solver.is_batch_mode = True

            interactive = not is_batch and sys.stdin.isatty()
            plan_res = solver.plan_survey(interactive=interactive)

            sia_grid = plan_res["sia_modelled_depth"]
            final_raster = BedrockMap(
                grid=sia_grid,
                bounds=solver.bounds,
                crs=solver.meta.get("crs"),
                transform=solver.meta.get("transform"),
                name="SIA Modelled Ice Thickness",
            )
            return final_raster

        logger.info("=" * 80)
        logger.info(f"        STARTING NEW PYSOLE BEDROCK TOPOGRAPHY CALCULATION ({config_file.resolve()})")
        logger.info("=" * 80)
        logger.info(f"Initialized PySole log file at: {log_file.resolve()}")

        solver = Solver.from_config(cfg, config_file=config_file, log_level=log_level, drift_analyzer=drift_analyzer)
        if is_batch:
            solver.show_progress = False
            solver.is_batch_mode = True

        if survey_profile_column is not None:
            solver.survey_profile_column = survey_profile_column

        mig_cfg = cfg.get("migration_parameters", {})
        opt_cfg = cfg.get("optimization_parameters", {})
        fin_cfg = cfg.get("finalization_parameters", {})

        interactive_mig = not is_batch and sys.stdin.isatty() and bool(mig_cfg.get("interactive_migration", True))
        interactive_opt = not is_batch and sys.stdin.isatty() and bool(opt_cfg.get("interactive_optimization", True))
        interactive_fin = not is_batch and sys.stdin.isatty() and bool(fin_cfg.get("interactive_finalization", True))

        if solver.survey_points is None and solver.survey_data_path is not None:
            solver.load_survey_points()

        if solver.perform_migration and solver._normalize_survey_data_type() != "depth":
            solver.optimize_bss(stage="pre_migration", interactive=interactive_opt)

        solver.migrate_eikonal(
            velocity=mig_cfg.get("velocity", 0.16),
            interactive=interactive_mig,
            interactive_optimization=interactive_opt,
        )
        solver.optimize_bss(stage="post_migration", interactive=interactive_opt)
        solver.calculate_bedrock(interactive=interactive_opt, interactive_optimization=interactive_opt)
        final_raster = solver.finalize_bedrock(
            interactive=interactive_fin,
            random_forest_gap_filling=fin_cfg.get("random_forest_gap_filling"),
            apply_margin_blend=fin_cfg.get("apply_margin_blend"),
            min_gap_dist=fin_cfg.get("min_gap_dist"),
            smooth_bedrock=fin_cfg.get("smooth_bedrock", False),
            smoothing_method=fin_cfg.get("smoothing_method", "gaussian"),
            smoothing_sigma=fin_cfg.get("smoothing_sigma", 1.5),
            smoothing_kernel_size=fin_cfg.get("smoothing_kernel_size", 3),
            smoothing_kc_cutoff=fin_cfg.get("smoothing_kc_cutoff"),
        )

        return final_raster
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        raise
