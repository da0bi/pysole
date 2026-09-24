"""
PySole: Physically-Informed Bedrock Interpolation & 3D Migration for Sparse Datasets.
"""

from .solver import Solver
from .raster import BedrockMap, load_dem, load_outline, ensure_spatial_coords, GridGeometry, load_survey_points
from .migration import migrate_eikonal_points, EikonalMigrator, MigrationResult
from .smoothing import compute_gradients, fft_gaussian_smooth
from .variogram import optimize_bss_variance, calculate_variogram, fit_variogram_model, BSSOptimizer, OptimizationResult
from .interpolation import blend_margin_topography, kriging_interpolation, random_forest_hole_filling, KrigingEngine, BedrockFinalizer, KrigingResult
from .config import run_from_config, load_config, main_cli
from .logging import logger, setup_logging
from . import plotting

__version__ = "0.2.0"

__all__ = [
    "Solver",
    "BedrockMap",
    "GridGeometry",
    "load_dem",
    "load_outline",
    "load_survey_points",
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
    "run_from_config",
    "load_config",
    "main_cli",
    "logger",
    "setup_logging",
    "plotting",
]
