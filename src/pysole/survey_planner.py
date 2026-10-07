"""
Unprobed Glacier Survey Planner for forward campaign design.
Generates synthetic SIA ice thickness targets D_SIA(x,y) from surface DEM topography
and computes optimal longitudinal and transverse survey profiles subject to total length budget L_max.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging
import numpy as np
import matplotlib.pyplot as plt

from .logging import logger
from .plotting import _save_figure
from .raster import BedrockMap, GridGeometry, load_dem, load_outline
from .smoothing import compute_gradients, fft_gaussian_smooth


@dataclass
class PlannedProfileTrack:
    """Structure representing a planned GPR/seismic survey track."""
    track_id: str
    track_type: str  # 'longitudinal' or 'transverse'
    length_m: float
    coordinates: np.ndarray  # Shape (N, 2) in projected [X, Y]
    max_depth_m: float


def _transform_coords_to_wgs84(coords: np.ndarray, crs: Any) -> np.ndarray:
    """Converts projected [X, Y] coordinates to WGS84 [lon, lat]."""
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
    """

    def __init__(
        self,
        dem: np.ndarray | str | Path,
        outline: np.ndarray | str | Path | None = None,
        dx: float = 10.0,
        dy: float = 10.0,
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
            computed_bounds = bounds or (0.0, 0.0, float(nx * dx), float(ny * dy))
            self.meta = {"dx": dx, "dy": dy, "bounds": computed_bounds, "crs": crs}

        self.dx = float(self.meta.get("dx", dx))
        self.dy = float(self.meta.get("dy", dy))
        self.bounds = self.meta.get("bounds", bounds or (0.0, 0.0, float(self.dem.shape[1] * self.dx), float(self.dem.shape[0] * self.dy)))
        self.crs = self.meta.get("crs", crs)

        self.outline = outline
        self.outline_mask = load_outline(outline, self.dem, self.meta)
        self.ice_density = ice_density
        self.g = g

    def compute_synthetic_sia_depth(
        self,
        kc: float = 0.0314,
        tau_0: float = 100e3,
        slope_floor_deg: float = 5.0,
    ) -> np.ndarray:
        """
        Computes synthetic SIA ice thickness grid D_SIA(x,y) [m].
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
        Generates central longitudinal flowline and transverse cross-profiles subject to total length budget max_length_km.
        """
        if max_length_km <= 0:
            raise ValueError(f"Survey track length budget (max_length_km={max_length_km}) must be greater than zero.")

        max_length_m = max_length_km * 1000.0
        ny, nx = d_sia.shape
        minx, miny, maxx, maxy = self.bounds

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
        """Exports planned tracks to GeoJSON format in WGS84 (EPSG:4326)."""
        features = []
        for trk in tracks:
            wgs_coords = _transform_coords_to_wgs84(trk.coordinates, self.crs)
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
        """Exports planned tracks to GPX format in WGS84 (EPSG:4326)."""
        out_p = Path(filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="PySole SurveyPlanner" xmlns="http://www.topografix.com/GPX/1/1">',
        ]
        for trk in tracks:
            wgs_coords = _transform_coords_to_wgs84(trk.coordinates, self.crs)
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
        kc: float = 0.5,
        tau_0: float = 100e3,
        max_length_km: float = 5.0,
        output_prefix: str = "final",
        output_dir: str | Path | None = None,
        plots_dir: str | Path | None = None,
        output_format: str = "tif",
    ) -> dict[str, Any]:
        """
        Executes full forward survey planning pipeline.
        """
        out_dir = Path(output_dir) if output_dir else Path.cwd()
        fig_dir = Path(plots_dir) if plots_dir else out_dir / "figures"
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
        fig, ax = plt.subplots(figsize=(10, 8))
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
