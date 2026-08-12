"""
PySole: Physically-Informed Bedrock Interpolation & 3D Migration for Sparse Datasets.
"""

from .solver import Solver
from .raster import BedrockMap, load_dem, load_outline, ensure_spatial_coords
from .migration import migrate_eikonal_points
from .smoothing import compute_gradients, fft_gaussian_smooth
from .variogram import optimize_bss_variance, calculate_variogram, fit_variogram_model
from .interpolation import blend_margin_topography, kriging_interpolation, random_forest_hole_filling
from .config import run_from_config, load_config, create_default_config

__version__ = "0.1.0"

__all__ = [
    "Solver",
    "BedrockMap",
    "load_dem",
    "load_outline",
    "ensure_spatial_coords",
    "migrate_eikonal_points",
    "compute_gradients",
    "fft_gaussian_smooth",
    "optimize_bss_variance",
    "calculate_variogram",
    "fit_variogram_model",
    "blend_margin_topography",
    "kriging_interpolation",
    "random_forest_hole_filling",
    "run_from_config",
    "load_config",
    "create_default_config",
]
