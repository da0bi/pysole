"""
Configuration manager for pysole.json configuration files.
"""

from typing import Dict, Any, Optional, Union
import json
import os

DEFAULT_CONFIG: Dict[str, Any] = {
    "inputs": {
        "dem_path": "surface_dem.asc",
        "outline_path": "creeping_body.shp",
        "survey_data_path": "sparse_survey.csv",
        "survey_data_type": "one_way_travel_time",
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
        "num_steps": 10,
        "interactive_optimization": False,
    },
    "kriging_parameters": {
        "pre_migration": {
            "method": "universal",
            "drift_terms": ["quadratic"],
            "variogram_model": "spherical",
            "include_zero_boundary_condition": False,
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
        "output_path": "final_bedrock.tif",
        "output_format": "tif",
        "plots_dir": None,
    },
}


def load_config(config_path: Union[str, os.PathLike] = "pysole.json") -> Dict[str, Any]:
    """
    Loads pysole.json configuration file, falling back to default configuration values.

    Parameters
    ----------
    config_path : str or Path
        Path to JSON configuration file.

    Returns
    -------
    config : dict
    """
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = json.load(f)

        for key, section in user_config.items():
            if key in config and isinstance(section, dict):
                config[key].update(section)
            else:
                config[key] = section

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


create_default_config = create_template_config


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

    cfg = load_config(config_path)

    inputs = cfg.get("inputs", {})
    spatial = cfg.get("spatial_parameters", {})
    migration = cfg.get("migration_parameters", {})
    opt = cfg.get("optimization_parameters", {})
    kriging_cfg = cfg.get("kriging_parameters", {})
    fin_cfg = cfg.get("finalization_parameters", {})
    outputs = cfg.get("outputs", {})

    dem_path = inputs.get("dem_path")
    outline_path = inputs.get("outline_path")
    survey_data_path = inputs.get("survey_data_path") or inputs.get("travel_times_path")

    if not dem_path:
        raise ValueError("Configuration 'inputs.dem_path' must be specified.")
    if not survey_data_path:
        raise ValueError("Configuration 'inputs.survey_data_path' must be specified.")

    # Check survey_data_type in inputs section with fallback to migration_parameters
    survey_data_type = inputs.get("survey_data_type") or migration.get("survey_data_type", "one_way_travel_time")

    pre_krig_cfg = kriging_cfg.get("pre_migration", {}) if isinstance(kriging_cfg.get("pre_migration"), dict) else {}
    post_krig_cfg = kriging_cfg.get("post_migration", {}) if isinstance(kriging_cfg.get("post_migration"), dict) else {}

    pre_method = pre_krig_cfg.get("method") or kriging_cfg.get("method", "universal")
    pre_drifts = pre_krig_cfg.get("drift_terms") or kriging_cfg.get("drift_terms", ["quadratic"])
    pre_var_model = pre_krig_cfg.get("variogram_model") or kriging_cfg.get("variogram_model", "spherical")
    pre_zero_boundary = pre_krig_cfg.get("include_zero_boundary_condition", False)

    post_method = post_krig_cfg.get("method") or kriging_cfg.get("method", "universal")
    post_drifts = post_krig_cfg.get("drift_terms") or kriging_cfg.get("drift_terms", ["sia_thickness"])
    post_var_model = post_krig_cfg.get("variogram_model") or kriging_cfg.get("variogram_model", "spherical")
    post_zero_boundary = post_krig_cfg.get("include_zero_boundary_condition", False)

    print(f"1. Initializing PySole Solver from '{config_path}'...")
    solver = Solver(
        dem=dem_path,
        outline=outline_path,
        dx=spatial.get("dx"),
        dy=spatial.get("dy"),
        bounds=spatial.get("bounds"),
        pre_kriging_method=pre_method,
        pre_drift_terms=pre_drifts,
        pre_variogram_model=pre_var_model,
        pre_zero_boundary=pre_zero_boundary,
        post_kriging_method=post_method,
        post_drift_terms=post_drifts,
        post_variogram_model=post_var_model,
        post_zero_boundary=post_zero_boundary,
        perform_migration=migration.get("perform_migration", True),
        survey_data_type=survey_data_type,
        plots_dir=outputs.get("plots_dir", None),
    )
    print(f"   DEM Resolution: dx = {solver.dx} m, dy = {solver.dy} m")
    print(f"   Survey Data Type: {solver.survey_data_type}")
    print(f"   Pre-Migration Kriging: {solver.pre_kriging_method} (drift = {solver.pre_drift_terms}, variogram = {solver.pre_variogram_model})")
    print(f"   Post-Migration Kriging: {solver.post_kriging_method} (drift = {solver.post_drift_terms}, variogram = {solver.post_variogram_model})")
    print(f"   Perform Migration: {solver.perform_migration}")
    print(f"   Interactive Migration: {migration.get('interactive_migration', False)}")
    print(f"   Plots Output Directory: {solver.plots_dir}")

    if solver.perform_migration and solver.survey_data_type.lower().strip() not in ["thickness", "ice_thickness", "depth"]:
        print("2. Performing 3D Eikonal Ray Migration...")
    else:
        print("2. Skipping 3D Eikonal Ray Migration...")

    solver.migrate_eikonal(
        travel_times=survey_data_path,
        velocity=migration.get("velocity"),
        interactive=migration.get("interactive_migration", False),
        plotit=True,
    )

    print("3. Performing BSS Surface Slope Optimization...")
    opt_kc = solver.optimize_bss(
        kc_max=opt.get("kc_max", 10.0) if opt.get("kc_max") is not None else 10.0,
        kc_min=opt.get("kc_min", 0.01),
        d_kc=opt.get("d_kc", 0.1) if opt.get("d_kc") is not None else 0.1,
        num_steps=opt.get("num_steps", 10),
        interactive=opt.get("interactive_optimization", False),
        plotit=True,
    )
    print(f"   Optimal Corner Frequency k_c = {opt_kc:.4f}")

    print("4. Finalizing Bedrock Topography...")
    bedrock_map = solver.finalize_topography(
        interactive=opt.get("interactive_optimization", False),
        plotit=True,
        random_forest_gap_filling=fin_cfg.get("random_forest_gap_filling", False),
        apply_margin_blend=fin_cfg.get("apply_margin_blend", False),
        min_gap_dist=fin_cfg.get("min_gap_dist", 50.0),
        smooth_bedrock=fin_cfg.get("smooth_bedrock", False),
        smoothing_method=fin_cfg.get("smoothing_method", "gaussian"),
        smoothing_sigma=fin_cfg.get("smoothing_sigma", 1.5),
        smoothing_kernel_size=fin_cfg.get("smoothing_kernel_size", 3),
        smoothing_kc_cutoff=fin_cfg.get("smoothing_kc_cutoff", None),
    )

    output_path = outputs.get("output_path")
    output_format = outputs.get("output_format")
    if output_path:
        saved_res = bedrock_map.save(output_path, formats=output_format)
        if isinstance(saved_res, list):
            for sf in saved_res:
                print(f"5. Saved predicted bedrock map to: {sf}")
        else:
            print(f"5. Saved predicted bedrock map to: {saved_res}")

    return bedrock_map
