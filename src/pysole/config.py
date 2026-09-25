"""
Configuration manager for pysole.json configuration files.
"""

from typing import Dict, Any, Optional, Union
import copy
import json
import os

DEFAULT_CONFIG: Dict[str, Any] = {
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
        "output_name": "final_bedrock",
        "plots_dir": "figures",
    },
}


def load_config(config_path: Union[str, os.PathLike] = "pysole.json") -> Dict[str, Any]:
    """
    Loads pysole.json configuration file, falling back to default configuration values.
    Initializes package logging based on the configured log_level.

    Parameters
    ----------
    config_path : str or Path
        Path to JSON configuration file.

    Returns
    -------
    config : dict
    """
    from .logging import setup_logging, logger

    config = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = json.load(f)

        for key, section in user_config.items():
            if key in config and isinstance(section, dict):
                config[key].update(section)
            else:
                config[key] = section

    inputs = config.get("inputs", {})
    base_dir = inputs.get("base_dir")
    survey_data_path = inputs.get("survey_data_path")
    if base_dir:
        eff_base = os.path.expanduser(base_dir)
    elif survey_data_path and isinstance(survey_data_path, (str, os.PathLike)):
        eff_base = os.path.dirname(os.path.expanduser(str(survey_data_path)))
    else:
        eff_base = ""

    log_file = "pysole.log"
    if eff_base and not os.path.isabs(log_file):
        log_file = os.path.join(eff_base, log_file)

    log_level = os.environ.get("PYSOLE_LOG_LEVEL") or inputs.get("log_level", "INFO")
    setup_logging(log_file=log_file, log_level=log_level)
    return config


def create_template_config(config_path: Union[str, os.PathLike] = "pysole.json") -> str:
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
    filepath = str(config_path)
    with open(filepath, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    return filepath


def run_from_config(config_path: Union[str, os.PathLike] = "pysole.json") -> Any:
    """
    Executes the full PySole workflow using parameters defined in pysole.json.

    Parameters
    ----------
    config_path : str or Path
        Path to pysole.json configuration file.

    Returns
    -------
    bedrock_map : BedrockMap
        Final predicted bedrock elevation grid.
    """
    from .solver import Solver

    solver = Solver.from_config(config_path)
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

    if args.debug:
        os.environ["PYSOLE_LOG_LEVEL"] = "DEBUG"

    run_from_config(args.config)

