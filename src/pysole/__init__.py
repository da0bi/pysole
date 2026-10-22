"""
PySole: Physically-Informed Bedrock Interpolation & 3D Migration for Sparse Datasets.

PySole is a high-performance Python package for reconstructing high-resolution glacier
and ice-cap bedrock topography from sparse ground-penetrating radar (GPR) or seismic
soundings. It integrates 3D Eikonal normal ray migration, physical Basal Shear Stress (BSS)
spectral slope optimization, Dual Kriging geostatistical interpolation, machine learning
void filling, and geomorphological boundary margin blending.
"""

from .solver import Solver
from .raster import BedrockMap, load_dem, load_outline, ensure_spatial_coords, GridGeometry, load_survey_points, calculate_ice_volume, IceVolumeResult
from .migration import migrate_eikonal_points, EikonalMigrator, MigrationResult
from .smoothing import compute_gradients, fft_gaussian_smooth
from .variogram import optimize_bss_variance, calculate_variogram, fit_variogram_model, BSSOptimizer, OptimizationResult
from .interpolation import blend_margin_topography, kriging_interpolation, random_forest_hole_filling, KrigingEngine, BedrockFinalizer, KrigingResult
from .drift_analyzer import DriftAnalyzer
from .survey_planner import SurveyPlanner
from .config import load_config, main_cli, OutputsConfig, ConfigError
from .pipeline import run_from_config, PipelineExporter
from .logging import (
    logger,
    setup_logging,
    reset_logging_state,
    get_welcome_banner,
    print_welcome_banner,
    log_section_header,
    end_terminal_block,
    get_progress_bar,
)
from . import plotting

__version__ = "0.5.3"

__all__ = [
    "Solver",
    "BedrockMap",
    "GridGeometry",
    "load_dem",
    "load_outline",
    "load_survey_points",
    "calculate_ice_volume",
    "IceVolumeResult",
    "ensure_spatial_coords",
    "migrate_eikonal_points",
    "EikonalMigrator",
    "MigrationResult",
    "compute_gradients",
    "fft_gaussian_smooth",
    "optimize_bss_variance",
    "BSSOptimizer",
    "OptimizationResult",
    "calculate_variogram",
    "fit_variogram_model",
    "blend_margin_topography",
    "kriging_interpolation",
    "KrigingEngine",
    "KrigingResult",
    "random_forest_hole_filling",
    "BedrockFinalizer",
    "DriftAnalyzer",
    "SurveyPlanner",
    "run_from_config",
    "load_config",
    "OutputsConfig",
    "ConfigError",
    "PipelineExporter",
    "main_cli",
    "logger",
    "setup_logging",
    "reset_logging_state",
    "get_welcome_banner",
    "print_welcome_banner",
    "log_section_header",
    "end_terminal_block",
    "get_progress_bar",
    "plotting",
]
