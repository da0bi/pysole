"""
Pipeline Orchestration and File Export Module for PySole.
Handles configuration ingestion, workspace path resolution, output export management,
and high-level execution wrappers.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
import numpy as np

from .config import load_config, resolve_path, resolve_input_path, resolve_output_dir, OutputsConfig
from .logging import logger, setup_logging
from .raster import BedrockMap, save_points_csv


class PipelineExporter:
    """
    Handles file export operations to disk for PySole output grids and survey points.
    """

    def __init__(self, solver: Any):
        self.solver = solver
        self.outputs_cfg = OutputsConfig.from_dict(solver.config.get("outputs", {}))

    def get_resolved_output_prefix(self) -> str:
        raw_prefix = self.outputs_cfg.output_prefix or "final"
        prefix_path = Path(raw_prefix)
        if prefix_path.suffix.lower() in [".tif", ".tiff", ".asc", ".csv", ".npy", ".geotiff"]:
            prefix_path = prefix_path.with_suffix("")
        return resolve_path(str(prefix_path), config_path=self.solver.config_path, output_dir=self.solver.output_dir)

    def export_raster(self, grid: np.ndarray | None, suffix: str, name: str) -> str | list[str] | None:
        if grid is None:
            return None
        base_prefix = self.get_resolved_output_prefix()
        filepath = f"{base_prefix}_{suffix}"
        fmt = self.outputs_cfg.output_format
        raster = BedrockMap(
            grid=grid,
            bounds=self.solver.bounds,
            crs=self.solver.meta.get("crs"),
            transform=self.solver.meta.get("transform"),
            name=name,
        )
        saved = raster.save(filepath, formats=fmt)
        if isinstance(saved, list):
            for sf in saved:
                logger.info(f"   Saved optional {name} map to: {sf}")
        else:
            logger.info(f"   Saved optional {name} map to: {saved}")
        return saved

    def export_points_csv(self, points: np.ndarray | None, suffix: str) -> str | None:
        if points is None:
            return None
        base_prefix = self.get_resolved_output_prefix()
        filepath = f"{base_prefix}_{suffix}.csv"
        saved = save_points_csv(points, filepath)
        logger.info(f"   Saved optional migrated survey points to: {saved}")
        return saved

    def export_all(self, stage: str | None = None) -> list[str]:
        saved: list[str] = []
        cfg_out = self.outputs_cfg

        if stage is None or stage == "migration":
            if cfg_out.save_traveltime_grid and self.solver.traveltime_grid is not None:
                res = self.export_raster(self.solver.traveltime_grid, "pre_migration_traveltime_grid", "Traveltime")
                if res:
                    saved.extend(res if isinstance(res, list) else [res])
            if cfg_out.save_migrated_points and self.solver.migrated_points is not None:
                res_csv = self.export_points_csv(self.solver.migrated_points, "migrated_points")
                if res_csv:
                    saved.append(res_csv)

        if stage is None or stage == "finalization":
            if cfg_out.save_ice_thickness_map and self.solver.thickness_grid is not None:
                res = self.export_raster(self.solver.thickness_grid, "final_ice_thickness_map", "Ice Thickness")
                if res:
                    saved.extend(res if isinstance(res, list) else [res])
            if cfg_out.save_basal_shear_stress_map and self.solver.basal_shear_stress_grid is not None:
                res = self.export_raster(self.solver.basal_shear_stress_grid, "final_basal_shear_stress_map", "Basal Shear Stress")
                if res:
                    saved.extend(res if isinstance(res, list) else [res])
            if cfg_out.save_bedrock_elevation_map and self.solver.final_grid is not None:
                res = self.export_raster(self.solver.final_grid, "final_bedrock_elevation_map", "Bedrock Elevation")
                if res:
                    saved.extend(res if isinstance(res, list) else [res])

        return saved


def run_from_config(
    config_path: str | Path = "pysole.json",
    log_level: str | None = None,
    is_batch: bool = False,
    drift_analyzer: bool | None = None,
    survey_profile_column: str | None = None,
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
        If True, enables the interactive Drift Analyzer helper tool.
    survey_profile_column : str, optional
        Survey profile column name for spatial cross-validation.

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
    from .solver import Solver

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file.resolve()}")

    cfg = load_config(config_file)
    inputs_cfg = cfg.get("inputs", {})
    outputs_cfg = cfg.get("outputs", {})

    log_lvl = log_level if log_level is not None else inputs_cfg.get("log_level", "INFO")

    output_dir = resolve_output_dir(inputs_cfg.get("output_dir"), config_file)
    log_file = output_dir / "pysole.log"

    setup_logging(log_file=log_file, log_level=log_lvl)

    try:
        survey_data_path = inputs_cfg.get("survey_data_path")
        if survey_data_path is None:
            logger.info("   [INFO] No survey_data_path provided. Automatically switching execution mode to Unprobed Glacier Survey Planner.")

            solver = Solver.from_config(config_file)
            if is_batch:
                solver.show_progress = False

            plan_res = solver.plan_survey()

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

        solver = Solver.from_config(config_file)
        if is_batch:
            solver.show_progress = False

        if drift_analyzer is not None:
            solver.drift_analyzer = drift_analyzer
        if survey_profile_column is not None:
            solver.survey_profile_column = survey_profile_column

        mig_cfg = cfg.get("migration_parameters", {})
        opt_cfg = cfg.get("optimization_parameters", {})
        fin_cfg = cfg.get("finalization_parameters", {})

        interactive_mig = not is_batch and sys.stdin.isatty() and bool(mig_cfg.get("interactive_migration", True))
        interactive_opt = not is_batch and sys.stdin.isatty() and bool(opt_cfg.get("interactive_optimization", True))

        solver.migrate_eikonal(velocity=mig_cfg.get("velocity", 0.16), interactive=interactive_mig)
        solver.optimize_bss(interactive=interactive_opt)
        solver.calculate_bedrock(interactive=interactive_opt)
        final_raster = solver.finalize_bedrock(
            interactive=interactive_opt,
            random_forest_gap_filling=fin_cfg.get("random_forest_gap_filling"),
            apply_margin_blend=fin_cfg.get("apply_margin_blend"),
            min_gap_dist=fin_cfg.get("min_gap_dist"),
            smooth_bedrock=fin_cfg.get("smooth_bedrock", False),
            smoothing_method=fin_cfg.get("smoothing_method", "gaussian"),
            smoothing_sigma=fin_cfg.get("smoothing_sigma", 1.5),
            smoothing_kernel_size=fin_cfg.get("smoothing_kernel_size", 3),
            smoothing_kc_cutoff=fin_cfg.get("smoothing_kc_cutoff"),
        )

        exporter = PipelineExporter(solver)
        exporter.export_all()

        output_bedrock_file = outputs_cfg.get("output_bedrock_map")
        if output_bedrock_file:
            full_out_path = resolve_path(output_bedrock_file, output_dir=solver.output_dir, config_path=config_file)
            fmt = outputs_cfg.get("output_format", "geotiff")
            saved_paths = final_raster.save(full_out_path, formats=fmt)
            if isinstance(saved_paths, list):
                for sp in saved_paths:
                    logger.info(f" 5. Saved predicted bedrock map to: {sp}")
            else:
                logger.info(f" 5. Saved predicted bedrock map to: {saved_paths}")

        return final_raster
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        raise
