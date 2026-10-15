"""
Unprobed Glacier Survey Planner for forward campaign design.
Generates synthetic SIA ice thickness targets D_SIA(x,y) from surface DEM topography
and computes optimal longitudinal and transverse survey profiles subject to total length budget L_max.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

from .config import resolve_plots_dir
from .logging import logger
from .plotting import _save_figure
from .raster import BedrockMap, load_dem, load_outline
from .smoothing import compute_gradients, fft_gaussian_smooth


DEFAULT_SURVEY_TAU_KPA: float = 100.0  # 100 kPa = 100,000 Pa
DEFAULT_SURVEY_KC: float = 0.0314      # rad/m (lambda_c ≈ 200 m)
DEFAULT_SURVEY_MAX_KM: float = 5.0     # km


def _prompt_survey_parameter(
    prompt_label: str,
    default_value: float,
    unit: str,
) -> float:
    """Prompts user for a positive numeric survey parameter; empty input uses default."""
    prompt_msg = f"[SurveyPlanner] {prompt_label} [{unit}] [default: {default_value}]: "
    while True:
        try:
            raw = input(prompt_msg).strip()
        except EOFError:
            return default_value
        if not raw:
            return default_value
        try:
            val = float(raw)
            if np.isfinite(val) and val > 0:
                return val
            print(f"[SurveyPlanner] Value must be a positive number (> 0). Got {val}.", flush=True)
        except ValueError:
            print(f"[SurveyPlanner] '{raw}' is not a valid number. Please enter a positive number or press Enter for default.", flush=True)


@dataclass
class PlannedProfileTrack:
    """
    Structure representing a planned geophysical GPR or seismic survey track.

    Attributes
    ----------
    track_id : str
        Unique identifier for the profile track (e.g. 'L1_longitudinal', 'T1_transverse').
    track_type : str
        Track geometry classification: 'longitudinal' (along flowline) or 'transverse' (cross-valley).
    length_m : float
        Total line-transect length in meters [m].
    coordinates : np.ndarray
        Array of projected 2D coordinates with shape ``(N, 2)`` in projected metric units [m].
    max_depth_m : float
        Maximum modeled SIA ice thickness expected along this transect [m].
    """
    track_id: str
    track_type: str  # 'longitudinal' or 'transverse'
    length_m: float
    coordinates: np.ndarray  # Shape (N, 2) in projected [X, Y]
    max_depth_m: float


def _transform_coords_to_wgs84(coords: np.ndarray, crs: Any) -> np.ndarray:
    """
    Converts projected metric coordinates [X, Y] to geographic WGS84 coordinates [lon, lat].

    Parameters
    ----------
    coords : np.ndarray
        2D coordinate array of shape ``(N, 2)`` in projected meters.
    crs : Any
        Coordinate Reference System identifier (e.g. EPSG code or PROJ string).

    Returns
    -------
    wgs_coords : np.ndarray
        2D array of coordinates in decimal degrees ``[longitude, latitude]``.
    """
    if crs is None:
        logger.warning("No CRS metadata provided for survey tracks. Exporting projected metric coordinates.")
        return coords
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([lon, lat])
    except Exception as e:
        logger.warning(f"Failed to transform coordinates to WGS84 (EPSG:4326): {e}. Exporting raw coordinates.")
        return coords


class SurveyPlanner:
    """
    Forward campaign survey planning engine for unprobed glaciers.

    Designs optimized reconnaissance GPR or seismic survey layouts prior to fieldwork
    for glaciers lacking ice thickness soundings. The planner:
    1. Computes synthetic ice thickness targets using the Shallow Ice Approximation (SIA):
       :math:`D_{\\text{SIA}}(x, y) = \\frac{\\tau_0}{\\rho_{\\text{ice}} g \\sin\\alpha}`
    2. Identifies the central flowline along the maximum thickness ridge.
    3. Generates transverse cross-valley transects distributed along the glacier body,
       strictly constrained by a maximum travel/flight distance budget :math:`L_{\\max}`.
    4. Exports flight tracks to standard navigation formats (GPX for GPS units and
       drone autopilots; GeoJSON for GIS software).

    Parameters
    ----------
    dem : np.ndarray, str, or Path
        Surface digital elevation model [m] as a 2D array or raster file path.
    outline : np.ndarray, str, or Path, optional
        Glacier boundary outline as a boolean mask, Shapefile, or GeoJSON.
    dx : float, optional
        Spatial grid spacing along X [m]. If None, inferred from raster metadata.
    dy : float, optional
        Spatial grid spacing along Y [m]. If None, inferred from raster metadata.
    bounds : tuple of float, optional
        Bounding box coordinates ``(minx, miny, maxx, maxy)`` [m].
    crs : Any, optional
        Coordinate Reference System (e.g. 'EPSG:32633').
    ice_density : float, default 900.0
        Glacier ice density :math:`\\rho` [kg/m³].
    g : float, default 9.81
        Gravitational acceleration :math:`g` [m/s²].
    """

    def __init__(
        self,
        dem: np.ndarray | str | Path,
        outline: np.ndarray | str | Path | None = None,
        dx: float | None = None,
        dy: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        crs: Any = None,
        ice_density: float = 900.0,
        g: float = 9.81,
    ):
        if isinstance(dem, (str, Path)):
            self.dem, self.meta = load_dem(dem, dx=dx, dy=dy, bounds=bounds, crs=crs)
        else:
            self.dem = np.asarray(dem)
            ny, nx = self.dem.shape
            eff_dx = dx if dx is not None else 10.0
            eff_dy = dy if dy is not None else 10.0
            computed_bounds = bounds or (0.0, 0.0, float(nx * eff_dx), float(ny * eff_dy))
            self.meta = {"dx": eff_dx, "dy": eff_dy, "bounds": computed_bounds, "crs": crs}

        self.dx = float(self.meta.get("dx", 10.0 if dx is None else dx))
        self.dy = float(self.meta.get("dy", 10.0 if dy is None else dy))
        self.bounds = self.meta.get("bounds", bounds or (0.0, 0.0, float(self.dem.shape[1] * self.dx), float(self.dem.shape[0] * self.dy)))
        self.crs = self.meta.get("crs", crs)

        self.outline = outline
        self.outline_mask = load_outline(outline, self.dem, self.meta)
        self.ice_density = ice_density
        self.g = g
        self._transformer: Any = None
        self._crs_warned: bool = False
        self._transform_warned: bool = False

    def _get_wgs84_transformer(self) -> Any:
        """Returns a cached pyproj.Transformer to EPSG:4326, or None if CRS is missing/invalid."""
        if self._transformer is None:
            if self.crs is None:
                if not self._crs_warned:
                    logger.warning("No CRS metadata provided for survey tracks. Exporting projected metric coordinates.")
                    self._crs_warned = True
            else:
                try:
                    from pyproj import Transformer
                    self._transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
                except Exception as e:
                    if not self._crs_warned:
                        logger.warning(f"Failed to initialize WGS84 transformer for CRS {self.crs!r}: {e}. Exporting raw coordinates.")
                        self._crs_warned = True
        return self._transformer

    def _transform_coords_to_wgs84(self, coords: np.ndarray) -> np.ndarray:
        """Transforms projected metric coordinates [X, Y] to WGS84 [lon, lat] using the cached transformer."""
        transformer = self._get_wgs84_transformer()
        if transformer is None:
            return coords
        try:
            lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
            return np.column_stack([lon, lat])
        except Exception as e:
            if not self._transform_warned:
                logger.warning(f"Failed to transform coordinates to WGS84 (EPSG:4326): {e}. Exporting raw coordinates.")
                self._transform_warned = True
            return coords

    def compute_synthetic_sia_depth(
        self,
        kc: float = 0.0314,
        tau_0: float = 100e3,
        slope_floor_deg: float = 5.0,
    ) -> np.ndarray:
        """
        Computes synthetic Shallow Ice Approximation (SIA) ice thickness grid D_SIA(x,y).

        Applies Gaussian spectral smoothing at wavenumber :math:`k_c` to the surface DEM
        to isolate the regional driving slope, then applies the classical SIA equation:
        .. math::
            D_{\\text{SIA}} = \\frac{\\tau_0}{\\rho_{\\text{ice}} g \\sin\\alpha}

        Flat areas are stabilized using a slope floor :math:`\\alpha_{\\min}` to prevent
        singularities (:math:`1/\\sin\\alpha \\to \\infty`). Cells outside the glacier
        outline or outside the valid DEM area are set to zero thickness.

        Parameters
        ----------
        kc : float, default 0.0314
            Corner frequency cutoff wavenumber [rad/m] for surface DEM smoothing
            (:math:`\\approx 0.0314\\text{ rad/m}` corresponds to :math:`\\lambda_c \\approx 200\\text{ m}`).
        tau_0 : float, default 100e3
            Characteristic basal shear stress target in Pascals [Pa] (100 kPa = 100,000 Pa).
        slope_floor_deg : float, default 5.0
            Minimum surface slope angle [°] used in the SIA denominator to avoid numerical explosion.

        Returns
        -------
        d_sia : np.ndarray
            2D synthetic ice thickness raster grid [m] with shape identical to ``dem``.
        """
        # Smooth surface DEM
        smoothed_dem, _, _ = fft_gaussian_smooth(self.dem, self.dx, self.dy, kc)
        grads = compute_gradients(smoothed_dem, self.dx, self.dy)
        slope_rad = grads["slope_rad"]
        slope_floor_rad = np.radians(slope_floor_deg)
        slope_clamped = np.maximum(slope_rad, slope_floor_rad)

        # SIA ice thickness equation: D = tau_0 / (rho * g * sin(alpha))
        d_sia = tau_0 / (self.ice_density * self.g * np.sin(slope_clamped))

        # Apply outline mask and NaN DEM terrain masking
        d_sia[~self.outline_mask] = 0.0
        d_sia[np.isnan(self.dem)] = 0.0

        return np.maximum(d_sia, 0.0)

    def generate_survey_tracks(
        self,
        d_sia: np.ndarray,
        max_length_km: float = 5.0,
    ) -> list[PlannedProfileTrack]:
        """
        Generates longitudinal flowline and transverse cross-profile tracks within a distance budget.

        Parameters
        ----------
        d_sia : np.ndarray
            2D synthetic SIA ice thickness grid [m].
        max_length_km : float, default 5.0
            Total survey track length budget in kilometers [km].

        Returns
        -------
        tracks : list of PlannedProfileTrack
            List of generated track structures with coordinates and predicted depths.

        Raises
        ------
        ValueError
            If ``max_length_km <= 0``.
        """
        if max_length_km <= 0:
            raise ValueError(f"Survey track length budget (max_length_km={max_length_km}) must be greater than zero.")

        max_length_m = max_length_km * 1000.0
        ny, nx = d_sia.shape
        minx, miny, maxx, _ = self.bounds

        # Cell center spatial coordinates in bottom-up grid (row 0 = miny)
        x_coords = minx + (np.arange(nx) + 0.5) * self.dx
        y_coords = miny + (np.arange(ny) + 0.5) * self.dy

        # 1. Longitudinal central flowline along maximum thickness ridge
        center_col = np.argmax(np.sum(d_sia, axis=0))
        valid_rows = np.where(d_sia[:, center_col] > 0)[0]
        if len(valid_rows) >= 2:
            y_pts = y_coords[valid_rows]
            x_pts = np.full_like(y_pts, x_coords[center_col])
        else:
            y_pts = y_coords
            x_pts = np.full_like(y_pts, (minx + maxx) / 2.0)

        long_coords = np.column_stack([x_pts, y_pts])
        seg_lengths = np.hypot(np.diff(long_coords[:, 0]), np.diff(long_coords[:, 1]))
        cum_length = np.insert(np.cumsum(seg_lengths), 0, 0.0)
        long_len = float(cum_length[-1])

        target_long_len = min(long_len, max_length_m * 0.4)
        if long_len > target_long_len and long_len > 0:
            valid_k = np.where(cum_length <= target_long_len)[0]
            if len(valid_k) >= 2:
                long_coords = long_coords[:valid_k[-1] + 1]
                long_len = float(cum_length[valid_k[-1]])

        tracks: list[PlannedProfileTrack] = []
        tracks.append(PlannedProfileTrack(
            track_id="L1_longitudinal",
            track_type="longitudinal",
            length_m=long_len,
            coordinates=long_coords,
            max_depth_m=float(np.max(d_sia)),
        ))

        # 2. Transverse cross-profiles subject to remaining length budget
        remaining_budget = max_length_m - long_len
        if remaining_budget > 0:
            n_cross = max(2, int(remaining_budget / (0.2 * max_length_m)))
            row_indices = np.linspace(ny * 0.15, ny * 0.85, n_cross, dtype=int)

            for i, row in enumerate(row_indices, 1):
                if remaining_budget <= 0:
                    break
                mask_row = d_sia[row, :] > 0
                if not np.any(mask_row):
                    continue
                cols = np.where(mask_row)[0]
                x_cross = x_coords[cols]
                y_cross = np.full_like(x_cross, y_coords[row])
                cross_coords = np.column_stack([x_cross, y_cross])
                c_len = float(np.sum(np.hypot(np.diff(cross_coords[:, 0]), np.diff(cross_coords[:, 1]))))

                if c_len > remaining_budget:
                    continue  # Skip profile if it exceeds remaining budget

                remaining_budget -= c_len
                tracks.append(PlannedProfileTrack(
                    track_id=f"T{i}_transverse",
                    track_type="transverse",
                    length_m=c_len,
                    coordinates=cross_coords,
                    max_depth_m=float(np.max(d_sia[row, cols])),
                ))

        return tracks

    def export_geojson(self, tracks: list[PlannedProfileTrack], filepath: str | Path) -> str:
        """
        Exports planned survey tracks to GeoJSON format in WGS84 geographic coordinates (EPSG:4326).

        Parameters
        ----------
        tracks : list of PlannedProfileTrack
            Survey tracks to export.
        filepath : str or Path
            Destination GeoJSON file path.

        Returns
        -------
        saved_path : str
            Absolute file path of the saved GeoJSON file.
        """
        features = []
        for trk in tracks:
            wgs_coords = self._transform_coords_to_wgs84(trk.coordinates)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": wgs_coords.tolist(),
                },
                "properties": {
                    "track_id": trk.track_id,
                    "track_type": trk.track_type,
                    "length_m": trk.length_m,
                    "max_depth_m": trk.max_depth_m,
                },
            })
        geojson_data = {"type": "FeatureCollection", "features": features}
        out_p = Path(filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)
        return str(out_p)

    def export_gpx(self, tracks: list[PlannedProfileTrack], filepath: str | Path) -> str:
        """
        Exports planned survey tracks to GPS Exchange Format (GPX) in WGS84 (EPSG:4326).

        Generates GPX XML compatible with handheld GPS receivers, mapping apps, and
        drone flight mission planners.

        Parameters
        ----------
        tracks : list of PlannedProfileTrack
            Survey tracks to export.
        filepath : str or Path
            Destination GPX file path.

        Returns
        -------
        saved_path : str
            Absolute file path of the saved GPX file.
        """
        out_p = Path(filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="PySole SurveyPlanner" xmlns="http://www.topografix.com/GPX/1/1">',
        ]
        for trk in tracks:
            wgs_coords = self._transform_coords_to_wgs84(trk.coordinates)
            lines.append(f'  <trk><name>{trk.track_id}</name><trkseg>')
            for pt in wgs_coords:
                lines.append(f'    <trkpt lat="{pt[1]:.6f}" lon="{pt[0]:.6f}"></trkpt>')
            lines.append('  </trkseg></trk>')
        lines.append('</gpx>')

        with open(out_p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return str(out_p)

    def plan_survey(
        self,
        kc: float | None = None,
        tau_0: float | None = None,
        max_length_km: float | None = None,
        output_prefix: str = "final",
        output_dir: str | Path | None = None,
        plots_dir: str | Path | None = None,
        output_format: str = "tif",
        interactive: bool = True,
    ) -> dict[str, Any]:
        """
        Executes the full forward survey planning workflow for unprobed glaciers.

        Computes synthetic SIA target thickness, designs longitudinal and transverse
        transect lines subject to total length constraint `max_length_km`, and exports
        GeoTIFF/ASCII rasters, GeoJSON lines, GPX tracks, and diagnostic plan map figures.

        Parameters
        ----------
        kc : float, optional
            Corner frequency cutoff wavenumber [rad/m] for DEM surface smoothing. If None,
            prompts interactively (default: 0.0314 rad/m, lambda_c ≈ 200 m).
        tau_0 : float, optional
            Characteristic basal shear stress target in Pascals [Pa]. If None, prompts
            interactively in kPa (default: 100 kPa = 100,000 Pa).
        max_length_km : float, optional
            Maximum total line length budget in kilometers [km]. If None, prompts
            interactively (default: 5.0 km).
        output_prefix : str, default "final"
            File prefix for generated raster and vector outputs.
        output_dir : str or Path, optional
            Base directory for output data files.
        plots_dir : str or Path, optional
            Directory for saving the diagnostic plan map figure.
        output_format : str, default "tif"
            Raster export format ('tif' or 'asc').
        interactive : bool, default True
            If True and running in an interactive terminal, prompts for any omitted parameters.

        Returns
        -------
        results : dict of str to Any
            Dictionary containing:
            - ``'sia_modelled_depth'``: 2D synthetic SIA ice thickness grid [m].
            - ``'tracks'``: List of PlannedProfileTrack instances.
            - ``'saved_raster'``: Path of exported depth GeoTIFF/ASCII raster.
            - ``'saved_geojson'``: Path of exported GeoJSON track lines.
            - ``'saved_gpx'``: Path of exported GPX track lines.
            - ``'saved_plot'``: Path of saved diagnostic survey plan map figure.
        """
        is_interactive = bool(interactive and sys.stdin.isatty())

        if tau_0 is None:
            if is_interactive:
                tau_kpa = _prompt_survey_parameter("Target basal shear stress tau_0", DEFAULT_SURVEY_TAU_KPA, "kPa")
                tau_0 = tau_kpa * 1000.0
            else:
                tau_0 = DEFAULT_SURVEY_TAU_KPA * 1000.0

        if kc is None:
            if is_interactive:
                kc = _prompt_survey_parameter("Corner frequency cutoff wavenumber k_c", DEFAULT_SURVEY_KC, "rad/m")
            else:
                kc = DEFAULT_SURVEY_KC

        if max_length_km is None:
            if is_interactive:
                max_length_km = _prompt_survey_parameter("Maximum survey track length budget", DEFAULT_SURVEY_MAX_KM, "km")
            else:
                max_length_km = DEFAULT_SURVEY_MAX_KM
        prefix_path = Path(output_prefix).expanduser()
        if output_dir:
            out_dir = Path(output_dir)
        elif prefix_path.is_absolute():
            out_dir = prefix_path.parent
        else:
            out_dir = Path.cwd()
        fig_dir = Path(resolve_plots_dir(plots_dir, output_prefix=output_prefix, output_dir=out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        logger.info("================================================================================")
        logger.info("       STARTING PYSOLE UNPROBED GLACIER SURVEY PLANNER")
        logger.info("================================================================================")
        logger.info(f"1. Computing synthetic SIA ice thickness target (tau_0 = {tau_0/1e3:.1f} kPa, k_c = {kc})...")

        d_sia = self.compute_synthetic_sia_depth(kc=kc, tau_0=tau_0)

        # Export synthetic SIA raster map: <output_prefix>_sia_modelled_depth.<ext>
        raster_filename = f"{output_prefix}_sia_modelled_depth.{output_format}"
        raster_path = out_dir / raster_filename
        bedrock_map = BedrockMap(
            grid=d_sia,
            bounds=self.bounds,
            crs=self.crs,
            name="sia_modelled_depth",
        )
        saved_raster = bedrock_map.save(str(raster_path), formats=output_format)
        logger.info(f"2. Saved synthetic SIA modeled depth raster to: {saved_raster}")

        # Generate survey tracks
        logger.info(f"3. Recommending survey profile tracks for max budget L_max = {max_length_km:.1f} km...")
        tracks = self.generate_survey_tracks(d_sia, max_length_km=max_length_km)

        # Export vector files: <output_prefix>_survey_plan.gpx / geojson
        geojson_path = out_dir / f"{output_prefix}_survey_plan.geojson"
        gpx_path = out_dir / f"{output_prefix}_survey_plan.gpx"
        saved_geojson = self.export_geojson(tracks, geojson_path)
        saved_gpx = self.export_gpx(tracks, gpx_path)
        logger.info(f"4. Saved planned survey tracks to: {saved_geojson} & {saved_gpx}")

        # Render & save plot: <plots_dir>/<output_prefix_name>_survey_plan_map.png
        file_prefix = Path(output_prefix).name
        plot_path = fig_dir / f"{file_prefix}_survey_plan_map.png"
        _, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            d_sia,
            extent=[self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3]],
            origin="lower",
            cmap="viridis",
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Modeled SIA Depth D(x,y) [m]")

        for trk in tracks:
            color = "crimson" if trk.track_type == "longitudinal" else "cyan"
            ax.plot(trk.coordinates[:, 0], trk.coordinates[:, 1], color=color, linewidth=2, label=trk.track_id)

        ax.set_title(f"Unprobed Glacier Survey Plan (L_max = {max_length_km} km)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.legend(loc="upper right")
        _save_figure(plt, fig_dir, f"{file_prefix}_survey_plan_map.png")
        plt.close()

        return {
            "sia_modelled_depth": d_sia,
            "tracks": tracks,
            "saved_raster": saved_raster,
            "saved_geojson": saved_geojson,
            "saved_gpx": saved_gpx,
            "saved_plot": plot_path,
        }
