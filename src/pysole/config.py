"""
Configuration manager for pysole.json configuration files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import json
import os


@dataclass
class OutputsConfig:
    """Structured configuration parameters for PySole output exports."""
    output_format: str | list[str] = "tif"
    output_prefix: str = "final"
    plots_dir: str = "figures"
    save_traveltime_grid: bool = False
    save_migrated_points: bool = False
    save_thickness_grid: bool = False
    save_thickness_uncertainty: bool = False
    save_basal_shear_stress: bool = False
    save_basal_shear_stress_uncertainty: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputsConfig":
        """Constructs an OutputsConfig instance from a parameters dictionary."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def active_exports(self) -> dict[str, bool]:
        """Returns a dictionary mapping optional output flag names to their boolean states."""
        return {
            "save_traveltime_grid": self.save_traveltime_grid,
            "save_migrated_points": self.save_migrated_points,
            "save_thickness_grid": self.save_thickness_grid,
            "save_thickness_uncertainty": self.save_thickness_uncertainty,
            "save_basal_shear_stress": self.save_basal_shear_stress,
            "save_basal_shear_stress_uncertainty": self.save_basal_shear_stress_uncertainty,
        }

    def to_dict(self) -> dict[str, Any]:
        """Exports dataclass fields to a dictionary representation."""
        return {
            "output_format": self.output_format,
            "output_prefix": self.output_prefix,
            "plots_dir": self.plots_dir,
            **self.active_exports(),
        }


DEFAULT_CONFIG: dict[str, Any] = {
    "inputs": {
        "dem_path": None,
        "outline_path": None,
        "survey_data_path": None,
        "base_dir": None,
        "survey_data_type": "one_way_travel_time",
        "ice_density": 900.0,
        "g": 9.81,
        "n_cores": -1,
        "log_level": "INFO",
        "show_progress": True,
    },
    "spatial_parameters": {
        "dx": None,
        "dy": None,
        "bounds": None,
    },
    "migration_parameters": {
        "perform_migration": True,
        "velocity": 0.16,
        "interactive_migration": False,
    },
    "optimization_parameters": {
        "kc_max": 10.0,
        "kc_min": 0.01,
        "d_kc": 0.1,
        "nrbins": None,
        "slope_floor_deg": 5.0,
        "interactive_optimization": False,
    },
    "kriging_parameters": {
        "built_in_kriging": True,
        "pre_migration": {
            "method": "universal",
            "drift_terms": ["sia_thickness"],
            "variogram_model": "spherical",
            "include_zero_boundary_condition": True,
        },
        "post_migration": {
            "method": "universal",
            "drift_terms": ["sia_thickness"],
            "variogram_model": "spherical",
            "include_zero_boundary_condition": True,
        },
    },
    "finalization_parameters": {
        "random_forest_gap_filling": False,
        "apply_margin_blend": False,
        "min_gap_dist": 50.0,
        "smooth_bedrock": False,
        "smoothing_method": "gaussian",
        "smoothing_sigma": 1.5,
        "smoothing_kernel_size": 3,
        "smoothing_kc_cutoff": None,
    },
    "outputs": {
        "output_format": "tif",
        "output_prefix": "final",
        "plots_dir": "figures",
        "save_traveltime_grid": False,
        "save_migrated_points": False,
        "save_thickness_grid": False,
        "save_thickness_uncertainty": False,
        "save_basal_shear_stress": False,
        "save_basal_shear_stress_uncertainty": False,
    },
}


def resolve_path(
    path: str | Path | os.PathLike | None,
    base_dir: str | Path | os.PathLike | None = None,
    survey_data_path: str | Path | os.PathLike | None = None,
) -> str | None:
    """
    Resolves a target file or directory path relative to base_dir or survey_data_path parent directory.

    Parameters
    ----------
    path : str, Path, or None
        Target path string or Path object.
    base_dir : str, Path, or None, optional
        Base directory override.
    survey_data_path : str, Path, or None, optional
        Survey data file path used to infer parent directory if base_dir is None.

    Returns
    -------
    resolved_path : str or None
    """
    if not path:
        return None
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)

    if base_dir:
        eff_base = Path(base_dir).expanduser()
    elif survey_data_path:
        eff_base = Path(survey_data_path).expanduser().parent
    else:
        eff_base = None

    if eff_base:
        return str(eff_base / path_obj)
    return str(path_obj)


def load_config(
    config_path: str | Path | os.PathLike | dict[str, Any] = "pysole.json",
    log_level: str | None = None,
) -> dict[str, Any]:
    """
    Loads pysole.json configuration file or dictionary, falling back to default configuration values.
    Initializes package logging based on the configured log_level.

    Parameters
    ----------
    config_path : str, Path, or dict
        Path to JSON configuration file or configuration dictionary.
    log_level : str, optional
        Explicit log level override (e.g. 'DEBUG', 'INFO').

    Returns
    -------
    config : dict
    """
    from .logging import setup_logging

    config = copy.deepcopy(DEFAULT_CONFIG)
    user_config = None
    if isinstance(config_path, dict):
        user_config = config_path
    elif config_path:
        cfg_p = Path(config_path).expanduser()
        if cfg_p.exists():
            with open(cfg_p, "r") as f:
                user_config = json.load(f)

    if user_config:
        for key, section in user_config.items():
            if key in config and isinstance(section, dict):
                config[key].update(section)
            else:
                config[key] = section

    inputs = config.get("inputs", {})
    base_dir = inputs.get("base_dir")
    survey_data_path = inputs.get("survey_data_path")
    log_file = resolve_path("pysole.log", base_dir=base_dir, survey_data_path=survey_data_path)

    eff_log_level = log_level or os.environ.get("PYSOLE_LOG_LEVEL") or inputs.get("log_level", "INFO")
    setup_logging(log_file=log_file, log_level=eff_log_level)
    return config


def create_template_config(config_path: str | Path | os.PathLike = "pysole.json") -> str:
    """
    Creates a default template pysole.json configuration file.

    Parameters
    ----------
    config_path : str or Path
        Target filepath for pysole.json.

    Returns
    -------
    filepath : str
    """
    target = Path(config_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    return str(target)


def run_from_config(
    config_path: str | Path | os.PathLike = "pysole.json",
    log_level: str | None = None,
) -> Any:
    """
    Executes the full PySole workflow using parameters defined in pysole.json.

    Parameters
    ----------
    config_path : str or Path
        Path to pysole.json configuration file.
    log_level : str, optional
        Explicit log level override (e.g. 'DEBUG').

    Returns
    -------
    bedrock_map : BedrockMap
        Final predicted bedrock elevation grid.
    """
    from .solver import Solver

    solver = Solver.from_config(config_path, log_level=log_level)
    return solver.run_pipeline()


def main_cli() -> None:
    """
    CLI terminal entry point for executing PySole workflow via pysole command line.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="PySole: Physically-Informed Bedrock Interpolation & 3D Migration for Sparse Geophysical Datasets."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="pysole.json",
        help="Path to pysole.json configuration file (default: pysole.json).",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Creates a template pysole.json configuration file in current directory.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        "--debug",
        dest="debug",
        action="store_true",
        help="Enables DEBUG level logging verbosity.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="PySole 0.2.0",
        help="Show PySole package version and exit.",
    )
    args = parser.parse_args()

    if args.init:
        from .logging import logger
        target = create_template_config(args.config if args.config != "pysole.json" else "pysole.json")
        logger.info(f"Created template configuration file at: {target}")
        sys.exit(0)

    log_level = "DEBUG" if args.debug else None
    run_from_config(args.config, log_level=log_level)
