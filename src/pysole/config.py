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
from difflib import get_close_matches
from .drift import default_model, normalize_engine, resolve_drift_plan
from .errors import ConfigError
from .logging import logger, setup_logging
from .memory import MAX_FRACTION_CAP


_TRUE_STRINGS = frozenset({"true", "yes", "on", "1"})
_FALSE_STRINGS = frozenset({"false", "no", "off", "0"})


def coerce_bool(value: Any, name: str, default: bool) -> bool:
    """
    Safely coerces a configuration setting into a strict Python boolean.

    Interprets explicit boolean types (``bool``, ``numpy.bool_``), integer flags
    (0 or 1), and case-insensitive string representations (``"true"``, ``"false"``,
    ``"yes"``, ``"no"``, ``"on"``, ``"off"``). If ``value`` is ``None`` (equivalent to
    a JSON ``null`` entry in ``pysole.json``), the specified default boolean value
    is returned.

    Parameters
    ----------
    value : Any
        Raw input value parsed from configuration or command-line arguments.
    name : str
        Human-readable parameter path (e.g., ``'outputs.save_thickness_grid'``)
        used for clear diagnostic messages.
    default : bool
        Fallback boolean value returned when ``value`` is ``None``.

    Returns
    -------
    coerced_bool : bool
        Validated boolean value.

    Raises
    ------
    ConfigError
        If ``value`` cannot be unambiguous coerced into a boolean.
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


def validate_memory_fraction(value: Any, name: str = "kriging_parameters.max_memory_fraction") -> float:
    """
    Validates that a memory budget fraction falls within safe limits.

    Ensures the user-configured RAM budget fraction satisfies ``0.0 < value <= 0.9``,
    leaving at least 10% of physical memory for operating system services.

    Parameters
    ----------
    value : Any
        Numeric fraction of available system memory.
    name : str, default "kriging_parameters.max_memory_fraction"
        Configuration key name used in diagnostic error messages.

    Returns
    -------
    fraction : float
        Validated memory budget fraction as a float.

    Raises
    ------
    ConfigError
        If ``value`` is non-numeric or outside the valid range ``(0.0, 0.9]``.
    """
    try:
        frac = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"Configuration parameter '{name}' must be a number, got {value!r}.") from None
    if not (0.0 < frac <= MAX_FRACTION_CAP):
        raise ConfigError(f"Configuration parameter '{name}' must satisfy 0 < value <= {MAX_FRACTION_CAP}, got {value!r}.")
    return frac


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


def sanitize_config(
    user_config: dict[str, Any] | None,
    defaults: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """
    Removes every key that is not part of the known configuration schema.

    A configuration must not carry unknown (typically outdated or misspelled) keys, but they must also never
    abort a run. Unknown keys are therefore dropped from the returned configuration and reported to the caller,
    which logs a single warning (see :func:`warn_unknown_config_keys`).

    Parameters
    ----------
    user_config : dict or None
        User configuration (may be partial).
    defaults : dict, optional
        Schema to check against, ``DEFAULT_CONFIG`` by default.

    Returns
    -------
    (cleaned, unknown) : tuple
        ``cleaned`` is ``user_config`` itself if nothing had to be removed, otherwise a pruned copy;
        ``unknown`` lists the removed keys as dotted paths, each with a "did you mean" hint where one is close.

    Raises
    ------
    ConfigError
        If a section that must be an object holds any other value (the run cannot continue meaningfully).
    """
    schema = DEFAULT_CONFIG if defaults is None else defaults
    if not user_config:
        return user_config, []

    unknown: list[str] = []

    def _walk(user: dict[str, Any], sch: dict[str, Any], prefix: str) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in user.items():
            path = f"{prefix}{key}"
            if key not in sch:
                hint = get_close_matches(str(key), list(sch), n=1, cutoff=0.6)
                unknown.append(f"{path} (did you mean '{prefix}{hint[0]}'?)" if hint else path)
                continue
            if isinstance(sch[key], dict):
                if not isinstance(value, dict):
                    raise ConfigError(f"Configuration section '{path}' must be a JSON object, got {value!r}.")
                cleaned[key] = _walk(value, sch[key], path + ".")
            else:
                cleaned[key] = value
        return cleaned

    cleaned = _walk(user_config, schema, "")
    if not unknown:
        return user_config, []
    return cleaned, unknown


def warn_unknown_config_keys(unknown: list[str]) -> None:
    """Logs one warning listing configuration keys that were ignored (see :func:`sanitize_config`)."""
    if unknown:
        logger.warning(
            "Ignoring unknown configuration key(s) (outdated or misspelled?): " + "; ".join(unknown)
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OutputsConfig":
        """Constructs an OutputsConfig instance from a parameters dictionary."""
        if not data:
            return cls()
        valid_keys = cls.__dataclass_fields__
        filtered = {}
        unknown = [str(k) for k in data if k not in valid_keys]
        warn_unknown_config_keys([f"outputs.{k}" for k in unknown])
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
        "max_memory_fraction": 0.5,
        "pre_migration": {
            "interpolation_target": "P",
            "method": "ordinary",
            "drift_analyzer": False,
            "drift_terms": None,
            "variogram_model": "spherical",
            "include_zero_boundary_condition": True,
        },
        "post_migration": {
            "interpolation_target": "P",
            "method": "ordinary",
            "drift_analyzer": False,
            "drift_terms": None,
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
    Resolves the primary workspace output directory on disk.

    Resolution logic:
    1. If ``output_dir`` is explicitly provided and relative, it is resolved relative
       to the parent directory of ``config_path`` (or the current working directory).
    2. If ``output_dir`` is None, a dedicated ``pysole/`` subdirectory is created
       beside ``survey_data_path`` (or inside the config folder), containing a
       ``figures/`` subfolder.

    Parameters
    ----------
    output_dir : str, Path, or os.PathLike, optional
        Explicitly configured output directory.
    survey_data_path : str, Path, or os.PathLike, optional
        Path to input survey dataset used for automatic directory co-location.
    config_path : str, Path, or os.PathLike, optional
        Path to JSON configuration file used for relative path anchoring.

    Returns
    -------
    resolved_path : Path
        Guaranteed existing Path object pointing to the effective output directory.
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
    """
    Resolves an input file path relative to the configuration file directory.

    If ``path`` is an in-memory NumPy array or None, it is passed through unchanged.
    Absolute paths remain untouched. Relative paths are first checked relative to
    the parent folder of ``config_path``; if found, that resolved absolute path is
    returned.

    Parameters
    ----------
    path : str, Path, os.PathLike, np.ndarray, or None
        Input file path or raster array.
    config_path : str, Path, or os.PathLike, optional
        Path to the configuration file.

    Returns
    -------
    resolved_path : str, np.ndarray, or None
        Resolved absolute string filepath or original object.
    """
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
    """
    Resolves an output file or directory path within the effective output directory.

    Parameters
    ----------
    path : str, Path, os.PathLike, or None
        Target output file or subdirectory name.
    output_dir : str, Path, or os.PathLike, optional
        Base output directory override.
    survey_data_path : str, Path, or os.PathLike, optional
        Survey file path used for fallback co-location.
    config_path : str, Path, or os.PathLike, optional
        Configuration file path used for anchoring relative paths.

    Returns
    -------
    resolved_path : str or Any
        Absolute string path anchored in the effective output directory.
    """
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
    Resolves the destination folder for diagnostic plots and figures.

    Ensures diagnostic plots (PNG figures) are saved adjacent to exported rasters:
    1. If ``plots_dir`` is an absolute path, it is used directly.
    2. If ``output_prefix`` is an absolute path, ``plots_dir`` is placed in its parent directory.
    3. Otherwise, ``plots_dir`` is placed within the effective output directory.

    Parameters
    ----------
    plots_dir : str, Path, or None, default "figures"
        Subdirectory or absolute directory name for figures.
    output_prefix : str, Path, or os.PathLike, optional
        Configured output file prefix.
    output_dir : str, Path, or os.PathLike, optional
        Base output directory.
    survey_data_path : str, Path, or os.PathLike, optional
        Survey file path for co-location.
    config_path : str, Path, or os.PathLike, optional
        Configuration file path.

    Returns
    -------
    plots_path : str
        Resolved absolute string directory path for saving figures.
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
    Extracts the user-configured base output directory string from a configuration dict.

    Priority order:
    1. Explicit ``outputs.output_dir`` setting.
    2. Parent directory of an absolute ``outputs.output_prefix``.
    3. None (indicating default automatic co-location beside survey data).

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.

    Returns
    -------
    base_dir : str or None
        Resolved base directory path or None.
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
    drift_analyzer: bool | None = None,
) -> dict[str, Any]:
    """
    Loads pysole.json configuration file or dictionary, falling back to default configuration values.
    Initializes package logging based on the configured log_level.

    The Kriging drift configuration of both stages is resolved and validated here, before any computation:
    an omitted ``method`` receives its default (see :func:`pysole.drift.default_model`) and contradictory or
    invalid ``method`` / ``drift_analyzer`` / ``drift_terms`` combinations raise :class:`ConfigError`.

    Parameters
    ----------
    config_path : str, Path, or dict
        Path to JSON configuration file or configuration dictionary.
    log_level : str, optional
        Explicit log level override (e.g. 'DEBUG', 'INFO').
    drift_analyzer : bool, optional
        If True, enables the drift analyzer of both stages (the command-line ``--drift-analyzer`` flag); it
        takes effect before the defaults are resolved. None keeps the configured values.

    Returns
    -------
    config : dict

    Raises
    ------
    ConfigError
        For unreadable JSON, wrongly typed values and invalid Kriging / drift settings.
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

    unknown_keys: list[str] = []
    if user_config:
        user_config, unknown_keys = sanitize_config(user_config)
    if user_config:
        _deep_merge_dict(config, user_config)
    _coerce_bool_leaves(config, DEFAULT_CONFIG)
    config["kriging_parameters"]["max_memory_fraction"] = validate_memory_fraction(
        config["kriging_parameters"]["max_memory_fraction"]
    )

    # Resolve and validate the Kriging drift configuration of both stages (fail fast, before any data is read)
    kp = config.get("kriging_parameters", {})
    user_kp = (user_config or {}).get("kriging_parameters", {})
    normalize_engine(kp.get("engine", "native"))

    for section_name, direct_target_char in [("pre_migration", "T"), ("post_migration", "D")]:
        sec = kp.get(section_name, {})
        user_sec = user_kp.get(section_name, {})
        if drift_analyzer:
            sec["drift_analyzer"] = True

        if "method" in user_sec:
            method = user_sec["method"]
            terms = sec.get("drift_terms")
        else:
            method, terms = default_model(
                None,
                sec.get("drift_terms"),
                sec.get("interpolation_target", "P"),
                direct_target_char,
                bool(sec.get("drift_analyzer", False)),
            )
        sec["method"] = method
        sec["drift_terms"] = terms
        try:
            resolve_drift_plan(method, sec.get("drift_analyzer", False), terms, kp.get("engine", "native"))
        except ConfigError as exc:
            raise ConfigError(f"kriging_parameters.{section_name}: {exc}") from exc

    inputs = config.get("inputs", {})
    output_dir = configured_output_dir(config)
    survey_data_path = inputs.get("survey_data_path")
    cfg_file = config_path if not isinstance(config_path, dict) else None
    log_file = resolve_path("pysole.log", output_dir=output_dir, survey_data_path=survey_data_path, config_path=cfg_file)

    eff_log_level = log_level or os.environ.get("PYSOLE_LOG_LEVEL") or inputs.get("log_level", "INFO")
    setup_logging(log_file=log_file, log_level=eff_log_level)
    warn_unknown_config_keys(unknown_keys)  # after logging setup, so the warning reaches the log file
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
        help="Enables the drift analyzer of both Kriging stages, i.e. a drift_analyzer set to true in the "
        "configuration file. It needs method 'universal' (an omitted method is 'universal' only for a direct "
        "target T / D); with method 'ordinary' it is ignored with a warning.",
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
    plan_parser.add_argument("--kc", type=float, default=None, help="FFT corner wavenumber cutoff k_c [rad/m] (default: 0.0314 rad/m, cutoff wavelength lambda_c ≈ 200m).")
    plan_parser.add_argument("--tau", type=float, default=None, help="Target basal shear stress tau_0 in kPa (default: 100.0 kPa).")
    plan_parser.add_argument("--max-km", type=float, default=None, help="Maximum total survey length budget in km (default: 5.0 km).")
    plan_parser.add_argument("--batch", action="store_true", help="Non-interactive batch mode (accept defaults for omitted parameters without prompting).")
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
            is_interactive = not plan_args.batch
            tau_0_pa = plan_args.tau * 1000.0 if plan_args.tau is not None else None
            planner.plan_survey(
                kc=plan_args.kc,
                tau_0=tau_0_pa,
                max_length_km=plan_args.max_km,
                output_prefix=plan_args.prefix,
                output_dir=plan_args.out_dir,
                plots_dir=plan_args.plots_dir,
                output_format=plan_args.format,
                interactive=is_interactive,
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

