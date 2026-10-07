"""
Configuration manager for pysole.json configuration files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import json
import os
import numpy as np
from .logging import logger


@dataclass
class OutputsConfig:
    """Structured configuration parameters for PySole output exports."""
    output_dir: str | None = None
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
            "output_dir": self.output_dir,
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
        "survey_data_type": "one_way_traveltime",
        "survey_profile_column": None,
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
        "origin": None,
        "crs": None,
    },
    "migration_parameters": {
        "perform_migration": True,
        "velocity": 0.16,
        "interactive_migration": False,
    },
    "optimization_parameters": {
        "fft_filter_metric": "wavenumber",
        "kc_max": None,
        "kc_min": 0.01,
        "d_kc": 0.01,
        "lambda_min": None,
        "lambda_max": None,
        "d_lambda": None,
        "nrbins": None,
        "slope_floor_deg": 5.0,
        "interactive_optimization": False,
    },
    "kriging_parameters": {
        "engine": "native",
        "pre_migration": {
            "interpolation_target": "P",
            "method": "ordinary",
            "drift_analyzer": False,
            "drift_terms": [],
            "variogram_model": "spherical",
            "include_zero_boundary_condition": True,
        },
        "post_migration": {
            "interpolation_target": "P",
            "method": "ordinary",
            "drift_analyzer": False,
            "drift_terms": [],
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
        "output_dir": None,
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


def _get_config_parent(config_path: str | Path | os.PathLike | None) -> Path:
    if config_path and not isinstance(config_path, dict):
        p = Path(config_path).expanduser()
        if p.exists():
            return p.parent
    return Path.cwd()


def resolve_output_dir(
    output_dir: str | Path | os.PathLike | None = None,
    survey_data_path: str | Path | os.PathLike | None = None,
    config_path: str | Path | os.PathLike | None = None,
) -> Path:
    """
    If output_dir is explicitly specified, returns output_dir as a Path (resolved relative to config_path if relative).
    If output_dir is None, creates and returns a 'pysole' folder inside the survey_data_path parent directory,
    and a 'figures' subfolder inside the pysole directory.
    """
    cfg_parent = _get_config_parent(config_path)

    if output_dir:
        target = Path(output_dir).expanduser()
        if not target.is_absolute():
            target = cfg_parent / target
        target.mkdir(parents=True, exist_ok=True)
        return target

    if survey_data_path:
        surv_p = Path(survey_data_path).expanduser()
        if not surv_p.is_absolute():
            surv_p = cfg_parent / surv_p
        parent_dir = surv_p.parent
    else:
        parent_dir = cfg_parent

    pysole_dir = parent_dir / "pysole"
    pysole_dir.mkdir(parents=True, exist_ok=True)
    (pysole_dir / "figures").mkdir(parents=True, exist_ok=True)
    return pysole_dir


def resolve_input_path(
    path: Any,
    config_path: str | Path | os.PathLike | None = None,
) -> Any:
    """Resolves an input file path relative to config_path parent directory or working directory."""
    if path is None or isinstance(path, np.ndarray):
        return path
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)

    cfg_parent = _get_config_parent(config_path)
    candidate = cfg_parent / path_obj
    if candidate.exists():
        return str(candidate)
    return str(path_obj)


def resolve_path(
    path: Any,
    output_dir: str | Path | os.PathLike | None = None,
    survey_data_path: str | Path | os.PathLike | None = None,
    config_path: str | Path | os.PathLike | None = None,
) -> Any:
    """Resolves an output file or directory path relative to the effective output_dir."""
    if path is None or isinstance(path, np.ndarray):
        return path
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)

    eff_dir = resolve_output_dir(output_dir=output_dir, survey_data_path=survey_data_path, config_path=config_path)
    return str(eff_dir / path_obj)


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

    # Automatically align Kriging defaults based on interpolation_target if method was not explicitly user-defined
    kp = config.get("kriging_parameters", {})
    user_kp = (user_config or {}).get("kriging_parameters", {})

    for section_name, target_key, direct_target_char in [("pre_migration", "pre_migration", "T"), ("post_migration", "post_migration", "D")]:
        sec = kp.get(section_name, {})
        user_sec = user_kp.get(section_name, {})
        target = str(sec.get("interpolation_target", "P")).upper()

        if "method" not in user_sec:
            if target == direct_target_char:
                sec["method"] = "universal"
                if "drift_terms" not in user_sec:
                    sec["drift_terms"] = ["sia"]
            else:
                sec["method"] = "ordinary"
                if "drift_terms" not in user_sec:
                    sec["drift_terms"] = []

    inputs = config.get("inputs", {})
    outputs = config.get("outputs", {})
    output_dir = outputs.get("output_dir")
    survey_data_path = inputs.get("survey_data_path")
    cfg_file = config_path if not isinstance(config_path, dict) else None
    log_file = resolve_path("pysole.log", output_dir=output_dir, survey_data_path=survey_data_path, config_path=cfg_file)

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
    target = Path(config_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    logger.info(f"Created default template configuration file at: {target}")
    return str(target)


def run_from_config(
    config_path: str | Path | os.PathLike = "pysole.json",
    log_level: str | None = None,
    is_batch: bool = False,
) -> Any:
    """
    Executes the full PySole workflow using parameters defined in pysole.json.

    Parameters
    ----------
    config_path : str or Path
        Path to pysole.json configuration file.
    log_level : str, optional
        Explicit log level override (e.g. 'DEBUG').
    is_batch : bool, default False
        If True, disables interactive terminal prompts.

    Returns
    -------
    bedrock_map : BedrockMap
        Final predicted bedrock elevation grid.
    """
    from .pipeline import run_from_config as _run_pipeline
    return _run_pipeline(config_path=config_path, log_level=log_level, is_batch=is_batch)


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
    from . import __version__

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"PySole {__version__}",
        help="Show PySole package version and exit.",
    )
    parser.add_argument(
        "--batch",
        "--non-interactive",
        dest="batch",
        action="store_true",
        help="Runs PySole in batch mode, automatically disabling all interactive prompts.",
    )
    parser.add_argument(
        "--drift-analyzer",
        dest="drift_analyzer",
        action="store_true",
        help="Enables the interactive Drift Analyzer helper tool.",
    )
    parser.add_argument(
        "--profile-col",
        dest="survey_profile_column",
        type=str,
        default=None,
        metavar="SURVEY_PROFILE_COLUMN",
        help="Specifies the survey profile column name in the survey dataset CSV.",
    )

    # Subparsers for plan-survey CLI
    subparsers = parser.add_subparsers(dest="subcommand", help="Optional subcommands.")
    plan_parser = subparsers.add_parser("plan-survey", help="Run the unprobed glacier survey planner.")
    plan_parser.add_argument("--dem", required=True, help="Path to surface DEM raster.")
    plan_parser.add_argument("--outline", default=None, help="Path to glacier boundary outline.")
    plan_parser.add_argument("--kc", type=float, default=0.5, help="DEM smoothing parameter k_c.")
    plan_parser.add_argument("--tau", type=float, default=100.0, help="Target basal shear stress tau_0 in kPa.")
    plan_parser.add_argument("--max-km", type=float, default=5.0, help="Maximum total survey length budget in km.")
    plan_parser.add_argument("--prefix", default="final", help="Output prefix.")
    plan_parser.add_argument("--out-dir", default=None, help="Output directory.")
    plan_parser.add_argument("--plots-dir", default=None, help="Plots output directory.")
    plan_parser.add_argument("--format", default="tif", help="Output raster format (tif, asc).")

    args = parser.parse_args()

    if args.init:
        from .logging import logger
        target = create_template_config(args.config if args.config != "pysole.json" else "pysole.json")
        logger.info(f"Created template configuration file at: {target}")
        sys.exit(0)

    if args.subcommand == "plan-survey":
        from .survey_planner import SurveyPlanner
        planner = SurveyPlanner(dem=args.dem, outline=args.outline)
        planner.plan_survey(
            kc=args.kc,
            tau_0=args.tau * 1000.0,
            max_length_km=args.max_km,
            output_prefix=args.prefix,
            output_dir=args.out_dir,
            plots_dir=args.plots_dir,
            output_format=args.format,
        )
        sys.exit(0)

    log_level = "DEBUG" if args.debug else None
    from .pipeline import run_from_config
    run_from_config(args.config, log_level=log_level, is_batch=args.batch)
