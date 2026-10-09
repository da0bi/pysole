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
from .logging import logger, setup_logging


class ConfigError(ValueError):
    """Raised for an invalid PySole configuration (unreadable JSON, wrongly typed parameter value)."""


_TRUE_STRINGS = frozenset({"true", "yes", "on", "1"})
_FALSE_STRINGS = frozenset({"false", "no", "off", "0"})


def coerce_bool(value: Any, name: str, default: bool) -> bool:
    """
    Interprets a configuration value as a boolean.

    ``bool`` and ``numpy.bool_`` pass through, ``None`` (JSON ``null``) selects ``default``, the integers 0/1 and the
    case-insensitive strings ``true/false/yes/no/on/off/1/0`` are converted. Anything else raises ``ConfigError``
    (a plain ``bool("false")`` would silently be ``True``).
    """
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE_STRINGS:
            return True
        if s in _FALSE_STRINGS:
            return False
    raise ConfigError(f"Configuration parameter '{name}' must be a boolean (true/false), got {value!r}.")


def _coerce_bool_leaves(config: dict[str, Any], defaults: dict[str, Any], prefix: str = "") -> None:
    """Validates / converts, in place, every config value whose default is a boolean."""
    for key, default in defaults.items():
        if key not in config:
            continue
        name = f"{prefix}{key}"
        if isinstance(default, dict):
            if isinstance(config[key], dict):
                _coerce_bool_leaves(config[key], default, name + ".")
        elif isinstance(default, bool):
            config[key] = coerce_bool(config[key], name, default)


@dataclass
class OutputsConfig:
    """Structured configuration parameters for PySole output exports."""
    output_dir: str | None = None
    output_format: str | list[str] = "tif"
    output_prefix: str = "final"
    plots_dir: str = "figures"
    save_traveltime_grid: bool = False
    save_traveltime_uncertainty: bool = False
    save_migrated_points: bool = False
    save_thickness_grid: bool = False
    save_thickness_uncertainty: bool = False
    save_basal_shear_stress: bool = False
    save_basal_shear_stress_uncertainty: bool = False
    save_bedrock_elevation_map: bool = True
    compute_uncertainty: bool = True

    @property
    def uncertainty_rasters_requested(self) -> bool:
        """True if any exported uncertainty raster needs the Kriging estimation variance."""
        return bool(
            self.save_thickness_uncertainty
            or self.save_basal_shear_stress_uncertainty
            or self.save_traveltime_uncertainty
        )

    @property
    def effective_compute_uncertainty(self) -> bool:
        """
        Whether the Kriging variance is evaluated. ``compute_uncertainty=False`` is overridden
        (auto-enabled) when an uncertainty raster export is requested, since those need the variance.
        """
        return bool(self.compute_uncertainty or self.uncertainty_rasters_requested)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OutputsConfig":
        """Constructs an OutputsConfig instance from a parameters dictionary."""
        if not data:
            return cls()
        valid_keys = cls.__dataclass_fields__
        filtered = {}
        for k, v in data.items():
            if k not in valid_keys:
                continue
            default = valid_keys[k].default
            filtered[k] = coerce_bool(v, f"outputs.{k}", default) if isinstance(default, bool) else v
        return cls(**filtered)

    def active_exports(self) -> dict[str, bool]:
        """Returns a dictionary mapping optional output flag names to their boolean states."""
        return {
            "save_traveltime_grid": self.save_traveltime_grid,
            "save_traveltime_uncertainty": self.save_traveltime_uncertainty,
            "save_migrated_points": self.save_migrated_points,
            "save_thickness_grid": self.save_thickness_grid,
            "save_thickness_uncertainty": self.save_thickness_uncertainty,
            "save_basal_shear_stress": self.save_basal_shear_stress,
            "save_basal_shear_stress_uncertainty": self.save_basal_shear_stress_uncertainty,
            "save_bedrock_elevation_map": self.save_bedrock_elevation_map,
        }

    def to_dict(self) -> dict[str, Any]:
        """Exports dataclass fields to a dictionary representation."""
        return {
            "output_dir": self.output_dir,
            "output_format": self.output_format,
            "output_prefix": self.output_prefix,
            "plots_dir": self.plots_dir,
            "compute_uncertainty": self.compute_uncertainty,
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
        "kc_min": None,
        "lambda_min": None,
        "lambda_max": None,
        "n_steps": None,
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
        "save_traveltime_uncertainty": False,
        "save_migrated_points": False,
        "save_thickness_grid": False,
        "save_thickness_uncertainty": False,
        "save_basal_shear_stress": False,
        "save_basal_shear_stress_uncertainty": False,
        "save_bedrock_elevation_map": True,
        "compute_uncertainty": True,
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


def resolve_plots_dir(
    plots_dir: Any = None,
    output_prefix: str | Path | os.PathLike | None = None,
    output_dir: str | Path | os.PathLike | None = None,
    survey_data_path: str | Path | os.PathLike | None = None,
    config_path: str | Path | os.PathLike | None = None,
) -> str:
    """
    Resolves the directory for diagnostic figures so that they are saved next to the data files.

    Rules
    -----
    1. An absolute ``plots_dir`` is used as is.
    2. A relative or unset ``plots_dir`` (``None`` is equivalent to ``"figures"``) is resolved against the
       directory holding the data files: the parent folder of ``output_prefix`` if it is an absolute path,
       otherwise the effective ``output_dir``.
    """
    target = Path(plots_dir if plots_dir else "figures").expanduser()
    if target.is_absolute():
        return str(target)

    if output_prefix:
        prefix_path = Path(output_prefix).expanduser()
        if prefix_path.is_absolute():
            return str(prefix_path.parent / target)

    return str(resolve_path(target, output_dir=output_dir, survey_data_path=survey_data_path, config_path=config_path))


def configured_output_dir(config: dict[str, Any]) -> str | None:
    """
    Returns the explicitly configured base output folder, or None for the default ``<survey dir>/pysole``.

    ``outputs.output_dir`` wins; otherwise an absolute ``outputs.output_prefix``
    makes its parent folder the base, so data files, figures and the log file all live together.
    """
    outputs = config.get("outputs", {}) or {}
    explicit = outputs.get("output_dir")
    if explicit:
        return str(explicit)
    prefix_raw = outputs.get("output_prefix")
    if prefix_raw and Path(prefix_raw).expanduser().is_absolute():
        return str(Path(prefix_raw).expanduser().parent)
    return None


def _deep_merge_dict(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively merges source dictionary into target dictionary in-place."""
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            _deep_merge_dict(target[k], v)
        else:
            target[k] = copy.deepcopy(v)
    return target


def load_config(
    config_path: str | Path | os.PathLike | dict[str, Any] = "pysole.json",
    log_level: str | int | None = None,
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
    config = copy.deepcopy(DEFAULT_CONFIG)
    user_config = None
    if isinstance(config_path, dict):
        user_config = config_path
    elif config_path:
        cfg_p = Path(config_path).expanduser()
        if cfg_p.exists():
            try:
                with open(cfg_p, "r") as f:
                    user_config = json.load(f)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"Invalid JSON in configuration file '{cfg_p}': {exc}") from exc
            if not isinstance(user_config, dict):
                raise ConfigError(f"Configuration file '{cfg_p}' must contain a JSON object at the top level.")

    if user_config:
        _deep_merge_dict(config, user_config)
    _coerce_bool_leaves(config, DEFAULT_CONFIG)

    # Automatically align Kriging defaults based on interpolation_target if method was not explicitly user-defined
    kp = config.get("kriging_parameters", {})
    user_kp = (user_config or {}).get("kriging_parameters", {})

    for section_name, direct_target_char in [("pre_migration", "T"), ("post_migration", "D")]:
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
    output_dir = configured_output_dir(config)
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
        description="PySole: Physically-Informed Bedrock Interpolation & 3D Migration for Sparse Geophysical Datasets.",
        epilog="Subcommand: 'pysole plan-survey --dem DEM [options]' runs the unprobed glacier survey planner "
        "(see 'pysole plan-survey --help').",
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

    # ``plan-survey`` is dispatched manually: an optional positional ``config`` combined with an argparse
    # sub-parser makes argparse treat ``pysole <config.json>`` as an invalid sub-command.
    argv = sys.argv[1:]
    plan_parser = argparse.ArgumentParser(prog="pysole plan-survey", description="Run the unprobed glacier survey planner.")
    plan_parser.add_argument("--dem", required=True, help="Path to surface DEM raster.")
    plan_parser.add_argument("--outline", default=None, help="Path to glacier boundary outline.")
    plan_parser.add_argument("--kc", type=float, default=0.0314, help="FFT corner wavenumber cutoff k_c [rad/m] (default: 0.0314 rad/m, cutoff wavelength lambda_c ≈ 200m).")
    plan_parser.add_argument("--tau", type=float, default=100.0, help="Target basal shear stress tau_0 in kPa.")
    plan_parser.add_argument("--max-km", type=float, default=5.0, help="Maximum total survey length budget in km.")
    plan_parser.add_argument("--prefix", default="final", help="Output prefix.")
    plan_parser.add_argument("--out-dir", default=None, help="Output directory.")
    plan_parser.add_argument("--plots-dir", default=None, help="Plots output directory.")
    plan_parser.add_argument("--format", default="tif", help="Output raster format (tif, asc).")

    plan_args = None
    if "plan-survey" in argv:
        idx = argv.index("plan-survey")
        plan_args = plan_parser.parse_args(argv[idx + 1 :])
        argv = argv[:idx]

    args = parser.parse_args(argv)
    args.subcommand = "plan-survey" if plan_args is not None else None

    if args.init:
        target = create_template_config(args.config if args.config != "pysole.json" else "pysole.json")
        print(f"Created template configuration file at: {target}")
        sys.exit(0)

    try:
        if args.subcommand == "plan-survey":
            from .survey_planner import SurveyPlanner
            planner = SurveyPlanner(dem=plan_args.dem, outline=plan_args.outline)
            planner.plan_survey(
                kc=plan_args.kc,
                tau_0=plan_args.tau * 1000.0,
                max_length_km=plan_args.max_km,
                output_prefix=plan_args.prefix,
                output_dir=plan_args.out_dir,
                plots_dir=plan_args.plots_dir,
                output_format=plan_args.format,
            )
            sys.exit(0)

        log_level = "DEBUG" if args.debug else None
        from .pipeline import run_from_config
        run_from_config(
            args.config,
            log_level=log_level,
            is_batch=args.batch,
            drift_analyzer=True if args.drift_analyzer else None,  # None -> keep the config-file value
            survey_profile_column=args.survey_profile_column,
        )
    except (ConfigError, FileNotFoundError) as exc:
        if args.debug:
            raise
        print(f"pysole: error: {exc}", file=sys.stderr)
        sys.exit(1)

