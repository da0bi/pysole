"""
Raster I/O, GeoTIFF Handling, DEM Resampling, Coordinate Systems, and Boundary Polygon Masking.
Ported from MATLAB scripts INREAD.m and RAND.m by Daniel Binder (2011).
Handles Shapefiles/GeoJSON with internal holes (nunataks) and DEM grid resampling.
"""

import os
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from shapely.geometry import Polygon, MultiPolygon
from .logging import logger


@dataclass
class GridGeometry:
    """
    Standardized spatial metadata container for DEM grids.
    Stores shape, resolution (dx, dy), 1D coordinates (x_coords, y_coords),
    bounding box (minx, miny, maxx, maxy), and Matplotlib extent [minx, maxx, miny, maxy].
    """
    shape: Tuple[int, int]
    dx: float
    dy: float
    x_coords: np.ndarray
    y_coords: np.ndarray
    bounds: Tuple[float, float, float, float]

    @property
    def extent(self) -> List[float]:
        """Returns Matplotlib plot extent [minx, maxx, miny, maxy]."""
        return [self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3]]

    @cached_property
    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lazy-evaluated cached 2D spatial meshgrid (xx, yy).
        Generates 2D spatial coordinate matrices ONCE in memory and caches the result.
        """
        return np.meshgrid(self.x_coords, self.y_coords)

    @classmethod
    def create(
        cls,
        shape: Tuple[int, int],
        dx: float = 1.0,
        dy: float = 1.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
    ) -> "GridGeometry":
        x_c, y_c, b = ensure_spatial_coords(shape, dx=dx, dy=dy, bounds=bounds, x_coords=x_coords, y_coords=y_coords)
        return cls(shape=shape, dx=float(dx), dy=float(dy), x_coords=x_c, y_coords=y_c, bounds=b)

    def create_interpolator(
        self, grid: np.ndarray, fill_value: Any = np.nan, method: str = "linear"
    ) -> RegularGridInterpolator:
        """
        Factory creating a RegularGridInterpolator bound to this spatial geometry.
        Automatically aligns (y_coords, x_coords) and disables bounds_error.
        """
        return RegularGridInterpolator(
            (self.y_coords, self.x_coords),
            grid,
            bounds_error=False,
            fill_value=fill_value,
            method=method,
        )


class BedrockMap:
    """
    Represents the final predicted bedrock elevation raster result.
    Supports saving to GeoTIFF (.tif) or CSV (.csv).
    """

    def __init__(
        self,
        grid: np.ndarray,
        bounds: Tuple[float, float, float, float],
        crs: Any = None,
        transform: Any = None,
        name: str = "final_bedrock",
    ):
        self.grid = grid
        self.bounds = bounds
        self.crs = crs
        self.transform = transform
        self.name = name
        self.shape = grid.shape

    def save(self, filepath: str, formats: Union[str, List[str], None] = None) -> Union[str, List[str]]:
        """
        Saves the predicted bedrock grid to GeoTIFF (.tif), ESRI ASCII Grid (.asc),
        CSV (.csv), or NumPy (.npy) format(s). Supports exporting multiple or all formats.

        Parameters
        ----------
        filepath : str
            Base or target file path (e.g. 'final_bedrock.tif' or 'examples/wuk/wuk_final_bedrock').
        formats : str or list of str, optional
            Output format(s): 'tif', 'asc', 'csv', 'npy', or 'all' (exports all four formats).
            If None, inferred directly from filepath extension.

        Returns
        -------
        saved_files : str or list of str
            Path of saved file, or list of saved file paths if multiple formats exported.
        """
        base_path, ext = os.path.splitext(filepath)
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        fmt_list = []
        if formats is not None:
            if isinstance(formats, str):
                if formats.lower().strip() == "all":
                    fmt_list = ["tif", "asc", "csv", "npy"]
                else:
                    fmt_list = [formats.lower().strip(".")]
            elif isinstance(formats, list):
                for f in formats:
                    if str(f).lower().strip() == "all":
                        fmt_list = ["tif", "asc", "csv", "npy"]
                        break
                    fmt_list.append(str(f).lower().strip("."))
        else:
            if ext:
                fmt_list = [ext.lower().strip(".")]
            else:
                fmt_list = ["tif"]

        saved_files = []
        for fmt in fmt_list:
            if len(fmt_list) == 1 and ext and not formats:
                target_path = filepath
            else:
                target_path = f"{base_path}.{fmt}"

            if fmt in ["tif", "tiff", "geotiff"]:
                import rasterio
                from rasterio.transform import from_bounds

                height, width = self.shape
                transform = self.transform
                if transform is None:
                    transform = from_bounds(*self.bounds, width, height)
                if self.crs is None:
                    logger.warning(f"Exporting GeoTIFF '{target_path}' without Coordinate Reference System (CRS) metadata.")

                grid_export = self.grid[::-1, :]
                with rasterio.open(
                    target_path,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=1,
                    dtype=self.grid.dtype,
                    crs=self.crs,
                    transform=transform,
                    nodata=np.nan,
                ) as dst:
                    dst.write(grid_export, 1)

            elif fmt in ["asc", "txt"]:
                height, width = self.shape
                minx, miny, maxx, maxy = self.bounds
                cellsize = (maxx - minx) / float(width)

                grid_asc = np.nan_to_num(self.grid[::-1, :], nan=-9999.0)
                header = (
                    f"ncols         {width}\n"
                    f"nrows         {height}\n"
                    f"xllcorner     {minx:.6f}\n"
                    f"yllcorner     {miny:.6f}\n"
                    f"cellsize      {cellsize:.6f}\n"
                    f"NODATA_value  -9999"
                )
                np.savetxt(target_path, grid_asc, header=header, comments="", fmt="%.4f")

            elif fmt == "csv":
                np.savetxt(target_path, self.grid, delimiter=",")

            elif fmt == "npy":
                np.save(target_path, self.grid)

            saved_files.append(target_path)

        return saved_files[0] if len(saved_files) == 1 else saved_files


def ensure_spatial_coords(
    shape: Tuple[int, int],
    dx: float = 1.0,
    dy: float = 1.0,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    """
    Ensures 1D spatial coordinate vectors (x_coords, y_coords) and bounding box (minx, miny, maxx, maxy)
    are consistently defined and aligned for a given grid shape (M_rows, N_cols).

    Returns
    -------
    x_coords : 1D np.ndarray (size N)
    y_coords : 1D np.ndarray (size M)
    bounds : Tuple[float, float, float, float] (minx, miny, maxx, maxy)
    """
    M, N = shape
    if bounds is not None:
        minx, miny, maxx, maxy = bounds
    else:
        minx, miny = 0.0, 0.0
        maxx, maxy = float(N) * dx, float(M) * dy
        bounds = (minx, miny, maxx, maxy)

    if x_coords is None:
        x_coords = minx + (np.arange(N) + 0.5) * dx

    if y_coords is None:
        y_coords = miny + (np.arange(M) + 0.5) * dy

    return x_coords, y_coords, bounds


def check_projected_metric_crs(
    crs: Any = None,
    coords: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """
    Verifies that input spatial coordinates use a projected metric coordinate system (e.g. UTM meters).
    Performs dual-level verification:
    1. CRS metadata check via PyProj (is_projected vs is_geographic).
    2. Empirical numerical bounds check (detecting lat/lon degrees in range [-180, 180] x [-90, 90]).
    """
    if crs is not None:
        try:
            from pyproj import CRS

            c = CRS.from_user_input(crs)
            if c.is_geographic or not c.is_projected:
                err_msg = (
                    f"\n[Unprojected Geographic CRS Error] Input CRS '{c.name}' ({c.to_epsg() or 'Custom'}) "
                    f"uses geographic degrees instead of projected metric units!\n"
                    f"PySole requires a projected metric Coordinate Reference System (e.g. UTM) for Euclidean variogram calculations."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
        except ValueError:
            raise
        except Exception:
            pass

    # Level 2: Empirical numerical bounds check for unprojected Lat/Lon degrees
    if coords is not None and len(coords) > 0:
        minx, miny = float(coords[:, 0].min()), float(coords[:, 1].min())
        maxx, maxy = float(coords[:, 0].max()), float(coords[:, 1].max())
        # Lat/Lon degrees typically span non-zero coordinates outside origin (0, 0) within [-180, 180] x [-90, 90]
        is_origin_grid = (minx == 0.0 and miny == 0.0)
        if not is_origin_grid and (-180.0 <= minx and maxx <= 180.0) and (-90.0 <= miny and maxy <= 90.0):
            err_msg = (
                f"\n[Geographic Coordinates Detected] Survey profile coordinates appear to be unprojected geographic degrees (X in [-180, 180], Y in [-90, 90]).\n"
                f"PySole requires a projected metric coordinate system in meters (e.g. UTM) for variogram Euclidean distance calculations."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)


def check_crs_alignment(dem_crs: Any, vector_crs: Any) -> None:
    """
    Checks if DEM CRS and vector/profile CRS match strictly and use projected metric coordinates.
    If they differ or use unprojected geographic degrees, logs error and raises ValueError.
    """
    if dem_crs is not None:
        check_projected_metric_crs(crs=dem_crs)
    if vector_crs is not None:
        check_projected_metric_crs(crs=vector_crs)

    if dem_crs is None or vector_crs is None:
        return

    try:
        from pyproj import CRS

        c1 = CRS.from_user_input(dem_crs)
        c2 = CRS.from_user_input(vector_crs)
        if not c1.equals(c2):
            err_msg = (
                f"\n[CRS Mismatch Error] Input coordinate reference systems do not match!\n"
                f"  - Surface DEM CRS: {c1.name} ({c1.to_epsg() or 'Custom'})\n"
                f"  - Vector/Profile CRS: {c2.name} ({c2.to_epsg() or 'Custom'})\n"
                f"All input datasets (DEM, outline, survey profiles) must use the exact same Coordinate Reference System."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
    except ValueError:
        raise
    except Exception:
        pass


def resample_dem(
    grid: np.ndarray,
    native_dx: float,
    native_dy: float,
    target_dx: float,
    target_dy: float,
    bounds: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Resamples a 2D DEM grid to target dx and dy pixel resolutions using bilinear interpolation.
    """
    M_native, N_native = grid.shape
    minx, miny, maxx, maxy = bounds

    # Calculate target grid dimension
    N_target = max(int(round((maxx - minx) / target_dx)), 1)
    M_target = max(int(round((maxy - miny) / target_dy)), 1)

    if N_target == N_native and M_target == M_native and abs(native_dx - target_dx) < 1e-6 and abs(native_dy - target_dy) < 1e-6:
        return grid.copy(), bounds

    # Original coordinate vectors
    orig_x = minx + (np.arange(N_native) + 0.5) * native_dx
    orig_y = miny + (np.arange(M_native) + 0.5) * native_dy

    # Target coordinate vectors
    new_x = minx + (np.arange(N_target) + 0.5) * target_dx
    new_y = miny + (np.arange(M_target) + 0.5) * target_dy
    new_xx, new_yy = np.meshgrid(new_x, new_y)

    interp = RegularGridInterpolator(
        (orig_y, orig_x),
        grid,
        bounds_error=False,
        fill_value=np.nan,
    )

    pts = np.column_stack((new_yy.ravel(), new_xx.ravel()))
    resampled = interp(pts).reshape((M_target, N_target))

    # Fill boundary NaNs if any with nearest neighbor
    nan_mask = np.isnan(resampled)
    if np.any(nan_mask):
        orig_xx, orig_yy = np.meshgrid(orig_x, orig_y)
        sample_pts = np.column_stack((orig_xx.ravel(), orig_yy.ravel()))
        sample_vals = grid.ravel()
        valid = ~np.isnan(sample_vals)
        if np.any(valid):
            near_vals = griddata(sample_pts[valid], sample_vals[valid], (new_xx[nan_mask], new_yy[nan_mask]), method="nearest")
            resampled[nan_mask] = near_vals

    new_bounds = (minx, miny, minx + N_target * target_dx, miny + M_target * target_dy)
    return resampled, new_bounds


def load_dem(
    dem_input: Union[str, np.ndarray],
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a DEM from a file path (GeoTIFF, ASCII Grid, CSV, NPY) or numpy array,
    extracting spatial metadata directly from the DEM file, and performing DEM resampling
    if target dx and dy parameters are defined.
    """
    grid = None
    transform = None
    crs = None
    calc_bounds = bounds
    native_dx = dx if dx is not None else 1.0
    native_dy = dy if dy is not None else 1.0

    if isinstance(dem_input, np.ndarray):
        grid = dem_input.astype(np.float64)
        height, width = grid.shape
        if calc_bounds is None:
            logger.warning(
                f"DEM input passed as raw in-memory np.ndarray without spatial bounds. "
                f"Defaulting origin to (0.0, 0.0) with pixel spacing dx={native_dx:.2f} m, dy={native_dy:.2f} m."
            )
            calc_bounds = (0.0, 0.0, float(width) * native_dx, float(height) * native_dy)
    elif isinstance(dem_input, str):
        ext = os.path.splitext(dem_input)[1].lower()

        # 1. GeoTIFF / Raster formats
        if ext in [".tif", ".tiff", ".geotiff"]:
            import rasterio

            with rasterio.open(dem_input) as src:
                grid = np.array(src.read(1), dtype=np.float64, copy=True)
                if src.nodata is not None:
                    grid[grid == src.nodata] = np.nan
                transform = src.transform
                crs = src.crs
                b = src.bounds
                calc_bounds = (b.left, b.bottom, b.right, b.top)
                native_dx = abs(transform.a) if transform.a != 0 else (b.right - b.left) / src.width
                native_dy = abs(transform.e) if transform.e != 0 else (b.top - b.bottom) / src.height

        # 2. ESRI ASCII Grid (.asc, .txt)
        elif ext in [".asc", ".txt"]:
            header = {}
            header_lines = 0
            with open(dem_input, "r") as f:
                for _ in range(6):
                    line = f.readline().strip()
                    parts = line.split()
                    if len(parts) == 2 and not parts[0].replace(".", "", 1).isdigit():
                        header[parts[0].lower()] = float(parts[1])
                        header_lines += 1
                    else:
                        break

            grid = np.loadtxt(dem_input, skiprows=header_lines)
            grid = grid.astype(np.float64)
            if "nodata_value" in header:
                grid[grid == header["nodata_value"]] = np.nan

            ncols = int(header.get("ncols", grid.shape[1]))
            nrows = int(header.get("nrows", grid.shape[0]))
            cellsize = header.get("cellsize", 1.0)
            xll = header.get("xllcorner", header.get("xllcenter", 0.0))
            yll = header.get("yllcorner", header.get("yllcenter", 0.0))

            native_dx = cellsize
            native_dy = cellsize
            calc_bounds = (xll, yll, xll + ncols * cellsize, yll + nrows * cellsize)

        # 3. CSV or NPY formats
        elif ext == ".csv":
            try:
                grid = np.loadtxt(dem_input, delimiter=",").astype(np.float64)
            except ValueError:
                import pandas as pd
                df_tmp = pd.read_csv(dem_input)
                grid = df_tmp.select_dtypes(include=[np.number]).to_numpy().astype(np.float64)
            height, width = grid.shape
            if calc_bounds is None:
                calc_bounds = (0.0, 0.0, float(width) * native_dx, float(height) * native_dy)
        elif ext == ".npy":
            grid = np.load(dem_input).astype(np.float64)
            height, width = grid.shape
            if calc_bounds is None:
                calc_bounds = (0.0, 0.0, float(width) * native_dx, float(height) * native_dy)
        else:
            raise ValueError(f"Unsupported DEM file extension: {ext}")

    if grid is None:
        raise ValueError(f"Could not load DEM dataset from {dem_input}")

    # Standard GIS rasters (GeoTIFF, ASCII Grid, CSV, NPY) store Row 0 at Y_max (top-down).
    # PySole's spatial coordinate vector y_coords[0] represents Y_min (bottom-up).
    # Flip grid vertically on file load so Row 0 aligns with y_coords[0] (Y_min).
    if isinstance(dem_input, str):
        grid = grid[::-1, :]

    calc_dx = dx if dx is not None else native_dx
    calc_dy = dy if dy is not None else native_dy

    # Perform DEM resampling if dx and dy target resolutions are specified
    if (dx is not None and abs(dx - native_dx) > 1e-4) or (dy is not None and abs(dy - native_dy) > 1e-4):
        logger.info(f"Resampling DEM grid from native ({native_dx:.2f}m x {native_dy:.2f}m) to target ({calc_dx:.2f}m x {calc_dy:.2f}m)...")
        grid_out, calc_bounds = resample_dem(
            grid,
            native_dx=native_dx,
            native_dy=native_dy,
            target_dx=calc_dx,
            target_dy=calc_dy,
            bounds=calc_bounds,
        )
    else:
        grid_out = grid

    meta = {
        "transform": transform,
        "crs": crs,
        "bounds": calc_bounds,
        "dx": calc_dx,
        "dy": calc_dy,
        "native_dx": native_dx,
        "native_dy": native_dy,
    }
    return grid_out, meta


def validate_and_extract_polygons(gdf: Any) -> List[Any]:
    """
    Validates vector geometries for internal hole compliance.
    If interior holes fail topological criteria (e.g. self-intersecting or invalid),
    prints an error message and falls back to using the outer boundary shell only.
    """
    valid_geoms = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        if not geom.is_valid:
            logger.warning(f"Boundary geometry at feature #{idx} is invalid. Extracting outer boundary shell only for processing.")
            try:
                if isinstance(geom, Polygon):
                    geom = Polygon(geom.exterior)
                elif isinstance(geom, MultiPolygon):
                    geom = MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
            except Exception:
                continue

        has_invalid_hole = False
        if isinstance(geom, Polygon) and len(geom.interiors) > 0:
            for h_idx, hole in enumerate(geom.interiors):
                hole_poly = Polygon(hole)
                if not hole_poly.is_valid or hole_poly.area <= 0:
                    has_invalid_hole = True
                    logger.warning(f"Interior hole #{h_idx} in feature #{idx} fails shapefile criteria. Falling back to outer boundary shell only.")
                    break

        if has_invalid_hole:
            if isinstance(geom, Polygon):
                geom = Polygon(geom.exterior)
            elif isinstance(geom, MultiPolygon):
                geom = MultiPolygon([Polygon(p.exterior) for p in geom.geoms])

        valid_geoms.append(geom)

    return valid_geoms


def load_outline(
    outline_input: Union[str, np.ndarray, None],
    dem_grid: np.ndarray,
    meta: Dict[str, Any],
) -> np.ndarray:
    """
    Load body outline (Shapefile/GeoJSON/polygon coordinates or raster mask),
    returning a boolean mask (True = inside glacier/body, False = outside or inside rock outcrop holes).

    Validates polygon interior holes; if holes fail criteria, prints an error message and falls back
    to the outer boundary shell for processing. Performs strict CRS alignment check against DEM.
    """
    if outline_input is None:
        return ~np.isnan(dem_grid)

    if isinstance(outline_input, np.ndarray):
        if outline_input.dtype == bool:
            return outline_input
        if outline_input.shape == dem_grid.shape:
            return outline_input > 0

    height, width = dem_grid.shape
    bounds = meta.get("bounds", (0.0, 0.0, float(width), float(height)))

    if isinstance(outline_input, str):
        ext = os.path.splitext(outline_input)[1].lower()

        # 1. Shapefile / GeoJSON / GeoPackage vector polygons with interior holes (nunataks)
        if ext in [".shp", ".geojson", ".gpkg"]:
            try:
                import geopandas as gpd
                from rasterio import features
                from rasterio.transform import from_bounds

                gdf = gpd.read_file(outline_input)

                # Check CRS alignment against DEM
                dem_crs = meta.get("crs")
                if dem_crs is not None and gdf.crs is not None:
                    check_crs_alignment(dem_crs, gdf.crs)

                transform = meta.get("transform")
                if transform is None:
                    transform = from_bounds(*bounds, width, height)

                geoms = validate_and_extract_polygons(gdf)
                shapes = [(geom, 1) for geom in geoms]
                mask = features.rasterize(
                    shapes=shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
                return mask.astype(bool)[::-1, :]
            except ValueError:
                raise
            except Exception:
                pass

        # 2. Text / CSV polygon coordinates (supports NaN-separated exterior and interior hole rings & text headers)
        try:
            try:
                raw_coords = np.loadtxt(outline_input, delimiter="," if ext == ".csv" else None)
            except ValueError:
                import pandas as pd
                df_tmp = pd.read_csv(outline_input)
                raw_coords = df_tmp.select_dtypes(include=[np.number]).to_numpy()

            if raw_coords.ndim == 2 and raw_coords.shape[1] >= 2:
                # Split coordinate blocks by NaN rows
                nan_mask = np.isnan(raw_coords[:, 0]) | np.isnan(raw_coords[:, 1])
                if np.any(nan_mask):
                    split_indices = np.where(nan_mask)[0]
                    rings = []
                    prev_idx = 0
                    for s_idx in split_indices:
                        ring = raw_coords[prev_idx:s_idx, :2]
                        if len(ring) >= 3:
                            rings.append(ring)
                        prev_idx = s_idx + 1
                    if prev_idx < len(raw_coords):
                        ring = raw_coords[prev_idx:, :2]
                        if len(ring) >= 3:
                            rings.append(ring)
                else:
                    rings = [raw_coords[:, :2]]

                if len(rings) > 0:
                    from matplotlib.path import Path

                    minx, miny, maxx, maxy = bounds
                    dx = meta.get("dx", (maxx - minx) / width)
                    dy = meta.get("dy", (maxy - miny) / height)

                    x_c, y_c, _ = ensure_spatial_coords(
                        (height, width),
                        dx=dx,
                        dy=dy,
                        bounds=bounds,
                    )
                    xx, yy = np.meshgrid(x_c, y_c)
                    pts = np.column_stack((xx.ravel(), yy.ravel()))

                    outer_mask = Path(rings[0]).contains_points(pts).reshape((height, width))
                    hole_mask = np.zeros((height, width), dtype=bool)
                    for hole_ring in rings[1:]:
                        if len(hole_ring) >= 3:
                            hole_mask |= Path(hole_ring).contains_points(pts).reshape((height, width))

                    return outer_mask & ~hole_mask
        except Exception:
            pass

    return ~np.isnan(dem_grid)


def load_survey_points(
    survey_input: Union[str, np.ndarray, os.PathLike],
    bounds: Optional[Tuple[float, float, float, float]] = None,
    dem_grid: Optional[np.ndarray] = None,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Unified ingestion and validation for scattered survey point datasets.
    Loads CSV, whitespace-delimited files, or NumPy arrays, converts coordinates, and validates bounds.
    """
    if isinstance(survey_input, (str, os.PathLike)):
        filepath = str(survey_input)
        try:
            pts = np.loadtxt(filepath, delimiter="," if filepath.endswith(".csv") else None)
        except ValueError:
            import pandas as pd
            df_tmp = pd.read_csv(filepath)
            pts = df_tmp.select_dtypes(include=[np.number]).to_numpy()
    else:
        pts = np.array(survey_input, dtype=np.float64)

    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    if len(pts) > 0:
        check_projected_metric_crs(coords=pts[:, :2])

    if bounds is not None and len(pts) > 0:
        minx, miny, maxx, maxy = bounds
        px_min, py_min = pts[:, 0].min(), pts[:, 1].min()
        px_max, py_max = pts[:, 0].max(), pts[:, 1].max()

        buf_x = max((maxx - minx) * 0.1, 1.0)
        buf_y = max((maxy - miny) * 0.1, 1.0)

        if (px_max < minx - buf_x) or (px_min > maxx + buf_x) or (py_max < miny - buf_y) or (py_min > maxy + buf_y):
            err_msg = (
                f"\n[Spatial Coordinate System Error] Survey profile coordinates do not match DEM spatial bounds!\n"
                f"  - Survey Points Extent: X=[{px_min:.2f}, {px_max:.2f}], Y=[{py_min:.2f}, {py_max:.2f}]\n"
                f"  - Surface DEM Bounds:  X=[{minx:.2f}, {maxx:.2f}], Y=[{miny:.2f}, {maxy:.2f}]\n"
                f"All input datasets (DEM, outline, survey profiles) must use the exact same Coordinate Reference System."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    # 3-column point set [X, Y, value] -> sample surface DEM elevation Z for column 3
    if pts.shape[1] == 3 and dem_grid is not None and x_coords is not None and y_coords is not None:
        interp_z = RegularGridInterpolator((y_coords, x_coords), dem_grid, bounds_error=False, fill_value=np.nan)
        pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))  # (Y, X)
        z_surf = interp_z(pts_xy)
        pts = np.column_stack((pts[:, 0], pts[:, 1], z_surf, pts[:, 2]))

    # Deduplicate / consolidate duplicate survey points with identical (X, Y) coordinates
    if len(pts) > 1:
        coords_rounded = np.round(pts[:, :2], decimals=1)
        unique_coords, inverse_indices = np.unique(coords_rounded, axis=0, return_inverse=True)
        if len(unique_coords) < len(pts):
            n_dups = len(pts) - len(unique_coords)
            logger.warning(
                f"Detected {n_dups} duplicate survey point coordinate(s) (X, Y). "
                f"Consolidating duplicate points by averaging values to ensure numerical stability in Kriging."
            )
            n_cols = pts.shape[1]
            consolidated = np.zeros((len(unique_coords), n_cols), dtype=np.float64)
            for idx in range(len(unique_coords)):
                mask = (inverse_indices == idx)
                consolidated[idx, :2] = pts[mask, :2].mean(axis=0)
                for col in range(2, n_cols):
                    consolidated[idx, col] = np.nanmean(pts[mask, col])
            pts = consolidated

    return pts

