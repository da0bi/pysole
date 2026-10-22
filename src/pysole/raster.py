"""
Spatial Raster I/O, Coordinate System Alignment, DEM Resampling, and Polygon Masking.

Provides tools for ingesting surface Digital Elevation Models (DEMs), scattered
geophysical survey points (radar/seismic), and complex boundary outlines (glacier margins,
rock outcrops, and nunataks). Ensures strict projected metric coordinate system (e.g., UTM)
consistency required for Euclidean spatial variography and glaciological ice flow physics.
Supports multi-format raster exports (GeoTIFF, ESRI ASCII Grid, CSV, NumPy).
"""

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from shapely.geometry import Polygon, MultiPolygon
from .logging import logger


@dataclass
class GridGeometry:
    """
    Standardized spatial metadata container for DEM raster grids.

    Stores the 2D matrix shape, spatial grid resolution (dx, dy) in meters,
    1D coordinate vectors (x_coords, y_coords) in projected meters, bounding box
    (minx, miny, maxx, maxy), and Matplotlib extent [minx, maxx, miny, maxy].
    Provides factory methods and utilities for coordinate transformations and
    regular grid interpolation.

    Parameters
    ----------
    shape : tuple of int
        Grid dimensions as (M_rows, N_cols).
    dx : float
        Cell resolution along the easting/X axis in meters (dx > 0).
    dy : float
        Cell resolution along the northing/Y axis in meters (dy > 0).
    x_coords : np.ndarray
        1D array of length N_cols representing cell center X coordinates in meters.
    y_coords : np.ndarray
        1D array of length M_rows representing cell center Y coordinates in meters.
    bounds : tuple of float
        Spatial bounding coordinates (minx, miny, maxx, maxy) in meters.
    """
    shape: tuple[int, int]
    dx: float
    dy: float
    x_coords: np.ndarray
    y_coords: np.ndarray
    bounds: tuple[float, float, float, float]

    @property
    def extent(self) -> list[float]:
        """
        Matplotlib plotting extent [minx, maxx, miny, maxy] in meters.

        Returns
        -------
        list of float
            Four-element bounding extent suitable for the `extent` keyword in `plt.imshow`.
        """
        return [self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3]]

    @cached_property
    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Lazy-evaluated cached 2D spatial coordinate meshgrid (xx, yy).

        Evaluates 2D spatial coordinate matrices once in memory and caches the result
        to avoid redundant coordinate allocations across repeated calculations.

        Returns
        -------
        xx : np.ndarray
            2D array of shape (M_rows, N_cols) with X coordinates in meters.
        yy : np.ndarray
            2D array of shape (M_rows, N_cols) with Y coordinates in meters.
        """
        return np.meshgrid(self.x_coords, self.y_coords)

    @classmethod
    def create(
        cls,
        shape: tuple[int, int],
        dx: float = 1.0,
        dy: float = 1.0,
        bounds: tuple[float, float, float, float] | None = None,
        x_coords: np.ndarray | None = None,
        y_coords: np.ndarray | None = None,
    ) -> "GridGeometry":
        """
        Construct a GridGeometry instance with validated and aligned spatial coordinates.

        Parameters
        ----------
        shape : tuple of int
            Grid shape as (M_rows, N_cols).
        dx : float, default=1.0
            Cell resolution along X in meters.
        dy : float, default=1.0
            Cell resolution along Y in meters.
        bounds : tuple of float, optional
            Bounding box (minx, miny, maxx, maxy) in meters.
        x_coords : np.ndarray, optional
            1D array of cell center X coordinates in meters.
        y_coords : np.ndarray, optional
            1D array of cell center Y coordinates in meters.

        Returns
        -------
        GridGeometry
            Fully initialized spatial metadata instance.
        """
        x_c, y_c, b = ensure_spatial_coords(shape, dx=dx, dy=dy, bounds=bounds, x_coords=x_coords, y_coords=y_coords)
        return cls(shape=shape, dx=float(dx), dy=float(dy), x_coords=x_c, y_coords=y_c, bounds=b)

    def create_interpolator(
        self, grid: np.ndarray, fill_value: Any = np.nan, method: str = "linear"
    ) -> RegularGridInterpolator:
        """
        Create a RegularGridInterpolator bound to this spatial geometry.

        Automatically aligns coordinate ordering (y_coords, x_coords) to match
        the row-column layout of the input grid and disables out-of-bounds errors.

        Parameters
        ----------
        grid : np.ndarray
            2D array of shape (M_rows, N_cols) containing values to interpolate.
        fill_value : float or Any, default=np.nan
            Value assigned to query points falling outside the grid domain bounds.
        method : {"linear", "nearest", "slinear", "cubic", "quintic"}, default="linear"
            Interpolation method supported by `scipy.interpolate.RegularGridInterpolator`.

        Returns
        -------
        scipy.interpolate.RegularGridInterpolator
            Callable interpolator accepting coordinates as [[y, x], ...] points.
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
    Container representing a predicted bedrock elevation raster or glaciological field.

    Encapsulates the 2D numerical grid, spatial bounding box, Coordinate Reference
    System (CRS), affine georeferencing transform, and multi-format export capabilities.

    Parameters
    ----------
    grid : np.ndarray
        2D array of shape (M_rows, N_cols) containing predicted bedrock elevations
        or derived quantities (e.g., ice thickness, basal shear stress) in meters or kPa.
    bounds : tuple of float
        Spatial bounding coordinates (minx, miny, maxx, maxy) in projected meters.
    crs : Any, optional
        Coordinate Reference System identifier (e.g., EPSG integer, WKT, or rasterio CRS object).
    transform : Any, optional
        Affine georeferencing transformation mapping pixel coordinates to spatial metric units.
    name : str, default="bedrock"
        Descriptive identifier for the dataset (e.g., 'bedrock', 'ice_thickness').
    """

    def __init__(
        self,
        grid: np.ndarray,
        bounds: tuple[float, float, float, float],
        crs: Any = None,
        transform: Any = None,
        name: str = "bedrock",
    ):
        self.grid = grid
        self.bounds = bounds
        self.crs = crs
        self.transform = transform
        self.name = name
        self.shape = grid.shape

    def save(self, filepath: str | Path, formats: str | list[str] | None = None) -> str | list[str]:
        """
        Saves the predicted bedrock grid to GeoTIFF (.tif), ESRI ASCII Grid (.asc),
        CSV (.csv), or NumPy (.npy) format(s). Supports exporting multiple or all formats.

        Parameters
        ----------
        filepath : str or Path
            Base or target file path (e.g. 'final_bedrock.tif' or 'examples/wuk/wuk_final_bedrock').
        formats : str or list of str, optional
            Output format(s): 'tif', 'asc', 'csv', 'npy', or 'all' (exports all four formats).
            If None, inferred directly from filepath extension.

        Returns
        -------
        saved_files : str or list of str
            Path of saved file, or list of saved file paths if multiple formats exported.
        """
        path_obj = Path(filepath).expanduser()
        if path_obj.is_dir() or str(filepath).endswith("/") or str(filepath).endswith("\\"):
            path_obj = path_obj / "final_bedrock.tif"

        path_obj.parent.mkdir(parents=True, exist_ok=True)
        ext = path_obj.suffix.lower().lstrip(".")
        stem = str(path_obj.with_suffix(""))

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
                fmt_list = [ext]
            else:
                fmt_list = ["tif"]

        saved_files = []
        for fmt in fmt_list:
            if len(fmt_list) == 1 and ext and not formats:
                target_path = str(path_obj)
            else:
                target_path = f"{stem}.{fmt}"

            abs_target_path = str(Path(target_path).expanduser().resolve())
            Path(abs_target_path).parent.mkdir(parents=True, exist_ok=True)

            if fmt in ["tif", "tiff", "geotiff"]:
                import rasterio
                from rasterio.transform import from_bounds

                height, width = self.shape
                transform = self.transform
                if transform is None:
                    transform = from_bounds(*self.bounds, width, height)
                if self.crs is None:
                    logger.warning(f"Exporting GeoTIFF '{abs_target_path}' without Coordinate Reference System (CRS) metadata.")

                grid_export = self.grid[::-1, :]
                with rasterio.open(
                    abs_target_path,
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
                cellsize_x = (maxx - minx) / float(width)
                cellsize_y = (maxy - miny) / float(height)

                if abs(cellsize_x - cellsize_y) > 1e-4:
                    tif_target_path = str(Path(abs_target_path).with_suffix(".tif"))
                    logger.warning(
                        f"ESRI ASCII grid (.asc) format requires square pixels (dx == dy), but raster is anisotropic "
                        f"(dx = {cellsize_x:.2f}m, dy = {cellsize_y:.2f}m). Automatically converting export format "
                        f"to GeoTIFF ('{tif_target_path}') to preserve 2D affine spatial transform and prevent GIS distortion."
                    )
                    import rasterio
                    from rasterio.transform import from_bounds

                    transform = self.transform
                    if transform is None:
                        transform = from_bounds(*self.bounds, width, height)
                    if self.crs is None:
                        logger.warning(f"Exporting GeoTIFF '{tif_target_path}' without Coordinate Reference System (CRS) metadata.")

                    grid_export = self.grid[::-1, :]
                    with rasterio.open(
                        tif_target_path,
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
                    saved_files.append(tif_target_path)
                    continue

                grid_asc = np.nan_to_num(self.grid[::-1, :], nan=-9999.0)
                header = (
                    f"ncols         {width}\n"
                    f"nrows         {height}\n"
                    f"xllcorner     {minx:.6f}\n"
                    f"yllcorner     {miny:.6f}\n"
                    f"cellsize      {cellsize_x:.6f}\n"
                    f"NODATA_value  -9999"
                )
                np.savetxt(abs_target_path, grid_asc, header=header, comments="", fmt="%.4f")

            elif fmt == "csv":
                np.savetxt(abs_target_path, self.grid[::-1, :], delimiter=",")

            elif fmt == "npy":
                np.save(abs_target_path, self.grid[::-1, :])

            saved_files.append(abs_target_path)

        return saved_files[0] if len(saved_files) == 1 else saved_files


def ensure_spatial_coords(
    shape: tuple[int, int],
    dx: float = 1.0,
    dy: float = 1.0,
    bounds: tuple[float, float, float, float] | None = None,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """
    Ensure 1D spatial coordinate vectors and bounding box are aligned.

    Guarantees consistent 1D coordinate vectors (cell center eastings and northings)
    and spatial bounding extent (minx, miny, maxx, maxy) for a 2D DEM grid of shape
    (M_rows, N_cols).

    Parameters
    ----------
    shape : tuple of int
        Grid dimensions as (M_rows, N_cols).
    dx : float, default=1.0
        Cell spacing along the X axis (easting) in meters.
    dy : float, default=1.0
        Cell spacing along the Y axis (northing) in meters.
    bounds : tuple of float, optional
        Four-element tuple (minx, miny, maxx, maxy) in meters. If None, derived
        from grid shape and cell spacing starting from local origin (0.0, 0.0).
    x_coords : np.ndarray, optional
        1D array of cell center X coordinates in meters. If None, generated as
        `minx + (arange(N) + 0.5) * dx`.
    y_coords : np.ndarray, optional
        1D array of cell center Y coordinates in meters. If None, generated as
        `miny + (arange(M) + 0.5) * dy`.

    Returns
    -------
    x_coords : np.ndarray
        1D array of length N_cols with cell center X coordinates in meters.
    y_coords : np.ndarray
        1D array of length M_rows with cell center Y coordinates in meters.
    bounds : tuple of float
        Spatial bounding coordinates (minx, miny, maxx, maxy) in meters.
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
    coords: np.ndarray | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    """
    Verify that spatial coordinates use a projected metric system (e.g., UTM in meters).

    PySole relies on Euclidean distance calculations (in meters) for experimental
    variogram estimation, spatial covariance modeling, and glaciological ice flow /
    surface slope physics. Operating directly in unprojected geographic coordinates
    (latitude/longitude degrees) leads to distorted distances, anisotropic distortion
    varying with latitude, and erroneous shear stress values.

    Performs a two-tier verification:
    1. CRS metadata verification via PyProj (checks that `is_projected` is True).
    2. Empirical numerical bounds check (detects coordinates falling inside
       [-180, 180] x [-90, 90] degrees away from local origin (0, 0)).

    Parameters
    ----------
    crs : Any, optional
        Coordinate Reference System identifier (EPSG integer, WKT string, or pyproj CRS).
    coords : np.ndarray, optional
        Scattered survey point coordinates of shape (N, 2) in meters.
    bounds : tuple of float, optional
        Bounding box (minx, miny, maxx, maxy) in meters.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an unprojected geographic coordinate system or degree coordinates are detected.
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
                f"\n[Geographic Coordinates Detected] Survey profile coordinates appear to be unprojected geographic degrees (X in [{minx:.4f}, {maxx:.4f}], Y in [{miny:.4f}, {maxy:.4f}]).\n"
                f"PySole requires a projected metric coordinate system in meters (e.g. UTM) for variogram Euclidean distance calculations."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

    if bounds is not None:
        b_minx, b_miny, b_maxx, b_maxy = float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
        is_origin_grid = (b_minx == 0.0 and b_miny == 0.0)
        if not is_origin_grid and (-180.0 <= b_minx and b_maxx <= 180.0) and (-90.0 <= b_miny and b_maxy <= 90.0):
            err_msg = (
                f"\n[Geographic Coordinates Detected] DEM bounding box coordinates appear to be unprojected geographic degrees (X in [{b_minx:.4f}, {b_maxx:.4f}], Y in [{b_miny:.4f}, {b_maxy:.4f}]).\n"
                f"PySole requires a projected metric coordinate system in meters (e.g. UTM) for variogram Euclidean distance and surface slope calculations."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)


def check_crs_alignment(dem_crs: Any, vector_crs: Any) -> None:
    """
    Verify that DEM and vector outline / survey profile coordinate systems match.

    Both inputs must share the exact same projected metric coordinate system
    to ensure spatial overlay precision. If systems diverge, an error is raised.

    Parameters
    ----------
    dem_crs : Any
        Coordinate Reference System of the surface DEM.
    vector_crs : Any
        Coordinate Reference System of the vector boundary outline or profile dataset.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the two CRS definitions do not match or if either is unprojected geographic degrees.
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
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Resample a 2D DEM grid to target cell resolutions using bilinear interpolation.

    Computes target grid dimensions from the spatial bounding box and resamples the
    elevation surface using `scipy.interpolate.RegularGridInterpolator`. Cells outside
    the original sample extent that produce NaNs are seamlessly padded with nearest-neighbor
    extrapolation, while interior NoData / NaN regions (e.g., nunataks) are preserved.

    Parameters
    ----------
    grid : np.ndarray
        Original 2D elevation grid of shape (M_native, N_native) in meters.
    native_dx : float
        Original grid cell spacing along the easting/X axis in meters.
    native_dy : float
        Original grid cell spacing along the northing/Y axis in meters.
    target_dx : float
        Target grid cell spacing along the easting/X axis in meters.
    target_dy : float
        Target grid cell spacing along the northing/Y axis in meters.
    bounds : tuple of float
        Spatial bounding coordinates (minx, miny, maxx, maxy) in meters.

    Returns
    -------
    resampled : np.ndarray
        Resampled 2D elevation grid of shape (M_target, N_target) in meters.
    new_bounds : tuple of float
        Updated spatial bounding coordinates (minx, miny, maxx, maxy) in meters.
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

    # Only fill out-of-bounds cells (outside original coordinate range) with nearest-neighbor, preserving interior NaNs
    nan_mask = np.isnan(resampled)
    if np.any(nan_mask):
        out_of_bounds = (
            (new_xx < orig_x.min()) | (new_xx > orig_x.max()) |
            (new_yy < orig_y.min()) | (new_yy > orig_y.max())
        )
        fill_mask = nan_mask & out_of_bounds
        if np.any(fill_mask):
            orig_xx, orig_yy = np.meshgrid(orig_x, orig_y)
            sample_pts = np.column_stack((orig_xx.ravel(), orig_yy.ravel()))
            sample_vals = grid.ravel()
            valid = ~np.isnan(sample_vals)
            if np.any(valid):
                near_vals = griddata(
                    sample_pts[valid], sample_vals[valid],
                    (new_xx[fill_mask], new_yy[fill_mask]), method="nearest"
                )
                resampled[fill_mask] = near_vals

    new_bounds = (minx, miny, minx + N_target * target_dx, miny + M_target * target_dy)
    return resampled, new_bounds


def load_dem(
    dem_input: str | Path | os.PathLike | np.ndarray,
    dx: float | None = None,
    dy: float | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    origin: tuple[float, float] | None = None,
    crs: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load and parse a Digital Elevation Model (DEM) with spatial metadata.

    Supports reading DEMs from GeoTIFF (`.tif`, `.tiff`), ESRI ASCII Grid (`.asc`, `.txt`),
    CSV (`.csv` as either 2D matrix or 3-column XYZ table), NumPy binary (`.npy`),
    or directly from a 2D `np.ndarray`. Extracts spatial metadata (bounding box,
    native grid resolution, CRS, and affine georeferencing transform) and resamples
    the grid to target resolution if `dx` or `dy` are explicitly requested.

    Parameters
    ----------
    dem_input : str, Path, os.PathLike, or np.ndarray
        Input DEM source: path to raster/text file or a 2D float array of elevations in meters.
    dx : float, optional
        Target grid resolution along the easting/X axis in meters. If specified and different
        from native resolution, the DEM is automatically resampled via bilinear interpolation.
    dy : float, optional
        Target grid resolution along the northing/Y axis in meters. If specified and different
        from native resolution, the DEM is automatically resampled via bilinear interpolation.
    bounds : tuple of float, optional
        Explicit spatial bounding box (minx, miny, maxx, maxy) in meters. If None, extracted
        directly from raster header or derived from grid dimensions and resolution.
    origin : tuple of float, optional
        Lower-left coordinate origin (xll, yll) in meters for headerless inputs. Default is (0.0, 0.0).
    crs : Any, optional
        Coordinate Reference System identifier. Overrides or supplements file metadata. Must
        represent a projected metric system (e.g., UTM).

    Returns
    -------
    grid_out : np.ndarray
        2D array of elevations in meters with shape (M_rows, N_cols). Row 0 corresponds
        to minimum Y (northing) to align with bottom-up Cartesian spatial coordinates.
    meta : dict of str to Any
        Dictionary containing extracted spatial metadata:
        - ``'transform'``: Affine georeferencing transform or None.
        - ``'crs'``: Coordinate Reference System object or EPSG code.
        - ``'bounds'``: Four-element tuple (minx, miny, maxx, maxy) in meters.
        - ``'dx'``: Effective cell resolution along X in meters.
        - ``'dy'``: Effective cell resolution along Y in meters.
        - ``'native_dx'``: Original cell resolution along X in meters.
        - ``'native_dy'``: Original cell resolution along Y in meters.

    Notes
    -----
    Standard GIS raster formats (GeoTIFF, ASCII Grid) index Row 0 at the top (maximum Y).
    PySole vertically flips loaded grids upon ingestion so that Row 0 matches the bottom
    coordinate (minimum Y), ensuring consistency with bottom-up coordinate arrays.
    """
    grid = None
    transform = None
    dem_crs = crs
    calc_bounds = bounds
    native_dx = dx if dx is not None else 1.0
    native_dy = dy if dy is not None else 1.0

    is_headerless = False

    if isinstance(dem_input, np.ndarray):
        grid = dem_input.astype(np.float64)
        is_headerless = True
    elif dem_input is not None:
        ext = Path(dem_input).suffix.lower()

        # 1. GeoTIFF / Raster formats
        if ext in [".tif", ".tiff", ".geotiff"]:
            import rasterio

            with rasterio.open(dem_input) as src:
                # Read first band preserving array shape across all NumPy versions
                grid = np.array(src.read([1])[0], dtype=np.float64)
                if src.nodata is not None:
                    grid[grid == src.nodata] = np.nan
                transform = src.transform
                if dem_crs is None:
                    dem_crs = src.crs
                b = src.bounds
                if calc_bounds is None:
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
            if "xllcorner" in header:
                xll = float(header["xllcorner"])
            elif "xllcenter" in header:
                xll = float(header["xllcenter"]) - 0.5 * cellsize
            else:
                xll = 0.0

            if "yllcorner" in header:
                yll = float(header["yllcorner"])
            elif "yllcenter" in header:
                yll = float(header["yllcenter"]) - 0.5 * cellsize
            else:
                yll = 0.0

            native_dx = cellsize
            native_dy = cellsize
            if calc_bounds is None:
                calc_bounds = (xll, yll, xll + ncols * cellsize, yll + nrows * cellsize)

        # 3. CSV or NPY formats
        elif ext == ".csv":
            import pandas as pd
            df_csv = pd.read_csv(dem_input)
            numeric_cols = df_csv.select_dtypes(include=[np.number])

            # Detect 3-column XYZ grid table format (X, Y, Z) vs 2D Matrix
            if numeric_cols.shape[1] == 3:
                col_names = [c.lower() for c in numeric_cols.columns]
                x_col, y_col, z_col = numeric_cols.columns[0], numeric_cols.columns[1], numeric_cols.columns[2]
                for idx, c in enumerate(col_names):
                    if c in ["x", "easting", "e"]:
                        x_col = numeric_cols.columns[idx]
                    elif c in ["y", "northing", "n"]:
                        y_col = numeric_cols.columns[idx]
                    elif c in ["z", "ele", "elevation", "height"]:
                        z_col = numeric_cols.columns[idx]

                x_vals = np.sort(df_csv[x_col].unique())
                y_vals = np.sort(df_csv[y_col].unique())
                if len(x_vals) * len(y_vals) == len(df_csv):
                    piv = df_csv.pivot(index=y_col, columns=x_col, values=z_col)
                    piv = piv.sort_index(ascending=False)  # top-down Y (will be flipped at end)
                    grid = piv.to_numpy().astype(np.float64)

                    inferred_dx = float(np.min(np.diff(x_vals))) if len(x_vals) > 1 else 1.0
                    inferred_dy = float(np.min(np.diff(y_vals))) if len(y_vals) > 1 else 1.0
                    if dx is None:
                        native_dx = inferred_dx
                    if dy is None:
                        native_dy = inferred_dy
                    minx = float(x_vals.min()) - 0.5 * native_dx
                    maxx = float(x_vals.max()) + 0.5 * native_dx
                    miny = float(y_vals.min()) - 0.5 * native_dy
                    maxy = float(y_vals.max()) + 0.5 * native_dy
                    if calc_bounds is None:
                        calc_bounds = (minx, miny, maxx, maxy)

            if grid is None:
                try:
                    grid = np.loadtxt(dem_input, delimiter=",").astype(np.float64)
                except ValueError:
                    grid = numeric_cols.to_numpy().astype(np.float64)
                is_headerless = True

        elif ext == ".npy":
            grid = np.load(dem_input).astype(np.float64)
            is_headerless = True
        else:
            raise ValueError(f"Unsupported DEM file extension: {ext}")

    if grid is None:
        raise ValueError(f"Could not load DEM dataset from {dem_input}")

    if is_headerless:
        height, width = grid.shape
        if dx is None or dy is None:
            lbl = "np.ndarray" if isinstance(dem_input, np.ndarray) else f"Headerless DEM '{dem_input}'"
            logger.warning(
                f"   [DEM Grid] {lbl} loaded without explicit pixel spacing (dx, dy). "
                f"Defaulting cell spacing to dx={native_dx:.2f} m, dy={native_dy:.2f} m."
            )
        if calc_bounds is None:
            if origin is not None:
                xll, yll = float(origin[0]), float(origin[1])
                calc_bounds = (xll, yll, xll + float(width) * native_dx, yll + float(height) * native_dy)
            else:
                lbl = "np.ndarray" if isinstance(dem_input, np.ndarray) else f"Headerless DEM '{dem_input}'"
                logger.warning(
                    f"   [DEM Grid] {lbl} loaded without spatial bounds or origin. "
                    f"Defaulting coordinate origin to (0.0, 0.0) in local meters."
                )
                calc_bounds = (0.0, 0.0, float(width) * native_dx, float(height) * native_dy)

    # Standard GIS rasters (GeoTIFF, ASCII Grid, CSV, NPY) store Row 0 at Y_max (top-down).
    # PySole's spatial coordinate vector y_coords[0] represents Y_min (bottom-up).
    # Flip grid vertically on file load so Row 0 aligns with y_coords[0] (Y_min).
    if not isinstance(dem_input, np.ndarray) and not is_headerless:
        grid = grid[::-1, :]

    calc_dx = dx if dx is not None else native_dx
    calc_dy = dy if dy is not None else native_dy

    # Validate projected metric coordinate system
    check_projected_metric_crs(crs=dem_crs, bounds=calc_bounds)

    # Perform DEM resampling if dx and dy target resolutions are specified
    if (dx is not None and abs(dx - native_dx) > 1e-4) or (dy is not None and abs(dy - native_dy) > 1e-4):
        logger.info(f"   [DEM Grid] Resampling DEM grid from native ({native_dx:.2f}m x {native_dy:.2f}m) to target ({calc_dx:.2f}m x {calc_dy:.2f}m)...")
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
        "crs": dem_crs,
        "bounds": calc_bounds,
        "dx": calc_dx,
        "dy": calc_dy,
        "native_dx": native_dx,
        "native_dy": native_dy,
    }
    return grid_out, meta


def load_aligned_raster(
    raster_input: str | Path | os.PathLike | np.ndarray | BedrockMap | None,
    geometry: GridGeometry,
    ref_meta: dict[str, Any] | None = None,
    name: str = "auxiliary raster",
) -> np.ndarray | None:
    """
    Loads an auxiliary 2D spatial raster (e.g. surface mass balance or surface velocity) and co-registers it
    onto the reference DEM ``GridGeometry``.

    Supports all 5 PySole raster formats (GeoTIFF, ESRI ASCII Grid, CSV, NumPy ``.npy``, and in-memory
    ``np.ndarray`` or :class:`BedrockMap`). If the source raster has a different spatial resolution or bounding
    extent than ``geometry``, it is bilinearly resampled onto ``geometry``'s cell centers ``(x_coords, y_coords)``.

    Parameters
    ----------
    raster_input : str, Path, os.PathLike, np.ndarray, BedrockMap, or None
        Source path or array of the auxiliary spatial field. If ``None``, returns ``None``.
    geometry : GridGeometry
        Reference spatial grid geometry of the surface DEM.
    ref_meta : dict of str to Any, optional
        Metadata dictionary of the reference DEM (used for CRS compatibility checks).
    name : str, default "auxiliary raster"
        Human-readable label for log and error messages (e.g. ``'mass_balance'``, ``'velocity'``).

    Returns
    -------
    aligned_grid : np.ndarray or None
        2D float64 array of shape ``geometry.shape`` aligned with ``geometry``, or ``None`` if ``raster_input``
        is ``None``.

    Raises
    ------
    ValueError
        If an in-memory ``np.ndarray`` does not match ``geometry.shape``, if the file does not exist, or if CRS
        projections are incompatible.
    """
    if raster_input is None:
        return None

    target_shape = tuple(geometry.shape)
    if isinstance(raster_input, BedrockMap):
        aux_grid = np.asarray(raster_input.grid, dtype=np.float64).copy()
        aux_bounds = tuple(float(b) for b in raster_input.bounds)
        aux_crs = raster_input.crs
        aux_dx = (aux_bounds[2] - aux_bounds[0]) / max(aux_grid.shape[1], 1)
        aux_dy = (aux_bounds[3] - aux_bounds[1]) / max(aux_grid.shape[0], 1)
    elif isinstance(raster_input, np.ndarray):
        if raster_input.shape != target_shape:
            raise ValueError(
                f"In-memory '{name}' array has shape {raster_input.shape}, expected DEM shape {target_shape}."
            )
        return np.asarray(raster_input, dtype=np.float64).copy()
    else:
        p = Path(raster_input).expanduser()
        if not p.exists():
            raise ValueError(f"Auxiliary '{name}' raster file does not exist: {p}")
        ext = p.suffix.lower()
        is_headerless_file = ext == ".npy"
        aux_grid, aux_meta = load_dem(
            p,
            dx=geometry.dx if is_headerless_file else None,
            dy=geometry.dy if is_headerless_file else None,
            bounds=geometry.bounds if is_headerless_file else None,
        )
        aux_bounds = aux_meta["bounds"]
        aux_crs = aux_meta.get("crs")
        aux_dx = float(aux_meta.get("dx", geometry.dx))
        aux_dy = float(aux_meta.get("dy", geometry.dy))

    ref_crs = ref_meta.get("crs") if isinstance(ref_meta, dict) else None
    if ref_crs is not None and aux_crs is not None:
        check_crs_alignment(ref_crs, aux_crs)

    if aux_grid.shape == target_shape and np.allclose(aux_bounds, geometry.bounds, atol=1e-3):
        logger.info(
            f"   [Auxiliary Raster] Loaded '{name}' grid ({target_shape[0]}x{target_shape[1]}, "
            f"dx={geometry.dx:.2f} m, dy={geometry.dy:.2f} m)."
        )
        return aux_grid

    aux_geom = GridGeometry.create(
        aux_grid.shape,
        dx=aux_dx,
        dy=aux_dy,
        bounds=aux_bounds,
    )
    interp = aux_geom.create_interpolator(aux_grid, fill_value=np.nan, method="linear")
    xx, yy = geometry.meshgrid
    pts = np.column_stack((yy.ravel(), xx.ravel()))
    aligned = interp(pts).reshape(target_shape)
    logger.info(
        f"   [Auxiliary Raster] Co-registered '{name}' grid from {aux_grid.shape[0]}x{aux_grid.shape[1]} "
        f"(dx={aux_dx:.2f} m, dy={aux_dy:.2f} m) onto DEM geometry {target_shape[0]}x{target_shape[1]} "
        f"(dx={geometry.dx:.2f} m, dy={geometry.dy:.2f} m)."
    )
    return np.asarray(aligned, dtype=np.float64)


def validate_and_extract_polygons(gdf: Any) -> list[Any]:
    """
    Validate vector geometries for interior hole compliance and topological validity.

    Inspects polygons or multi-polygons in a GeoDataFrame, ensuring that exterior
    rings and interior holes (representing rock outcrops or nunataks) are topologically
    valid and non-empty. If interior holes fail topological checks, logs a warning
    and gracefully falls back to extracting the outer boundary shell.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Loaded vector dataset containing polygon or multi-polygon features.

    Returns
    -------
    list of shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        List of validated Shapely geometry objects ready for rasterization.
    """
    valid_geoms = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        if not geom.is_valid:
            logger.warning(f"   [Boundary Outline] Boundary geometry at feature #{idx} is invalid. Extracting outer boundary shell only for processing.")
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
                    logger.warning(f"   [Boundary Outline] Interior hole #{h_idx} in feature #{idx} fails shapefile criteria. Falling back to outer boundary shell only.")
                    break

        if has_invalid_hole:
            if isinstance(geom, Polygon):
                geom = Polygon(geom.exterior)
            elif isinstance(geom, MultiPolygon):
                geom = MultiPolygon([Polygon(p.exterior) for p in geom.geoms])

        valid_geoms.append(geom)

    return valid_geoms


def load_outline(
    outline_input: str | Path | os.PathLike | np.ndarray | None,
    dem_grid: np.ndarray,
    meta: dict[str, Any],
) -> np.ndarray:
    """
    Load glacier or study area boundary outline and convert to a 2D boolean mask.

    Ingests vector outlines (Shapefile `.shp`, GeoJSON `.geojson`, GeoPackage `.gpkg`),
    coordinate ring text files (ASCII/CSV with NaN row separators for holes), or
    pre-existing boolean arrays. Validates topological validity, verifies strict
    CRS alignment with the surface DEM, and rasterizes polygons onto the DEM grid.
    If no outline is provided, delineates active body boundaries from DEM NaN values.

    Parameters
    ----------
    outline_input : str, Path, os.PathLike, np.ndarray, or None
        Path to vector outline file, coordinate table, or boolean array. If None,
        the non-NaN cells of `dem_grid` are treated as the active domain.
    dem_grid : np.ndarray
        2D surface DEM array of shape (M_rows, N_cols) in meters.
    meta : dict of str to Any
        Spatial metadata dictionary from `load_dem` containing bounds, CRS, and cell dimensions.

    Returns
    -------
    mask : np.ndarray
        2D boolean array of shape (M_rows, N_cols) where True represents the active glaciated
        domain and False indicates ice-free terrain or interior rock nunataks.

    Raises
    ------
    ValueError
        If the outline file does not exist, coordinate reference systems do not match,
        or vector geometry parsing fails.
    """
    if outline_input is None:
        nan_mask = np.isnan(dem_grid)
        nan_count = int(np.count_nonzero(nan_mask))
        if nan_count > 0:
            valid_count = dem_grid.size - nan_count
            logger.info(
                f"   [Boundary Outline] No boundary outline provided. Delineated active body boundary and rock outcrops "
                f"from DEM NaN values ({nan_count:,} NaN pixels, {valid_count:,} valid domain pixels)."
            )
        else:
            logger.warning(
                "   [Boundary Outline] No boundary outline provided and no NaN values found in DEM. "
                "Entire DEM grid rectangle will be treated as the active domain without boundary constraints."
            )
        return ~nan_mask

    if isinstance(outline_input, np.ndarray):
        if outline_input.dtype == bool:
            return outline_input
        if outline_input.shape == dem_grid.shape:
            return outline_input > 0

    height, width = dem_grid.shape
    bounds = meta.get("bounds", (0.0, 0.0, float(width), float(height)))

    if isinstance(outline_input, (str, Path, os.PathLike)):
        p = Path(outline_input).expanduser()
        if not p.exists():
            raise ValueError(f"Boundary outline file does not exist: {p.resolve()}")

    ext = Path(outline_input).suffix.lower() if isinstance(outline_input, (str, Path, os.PathLike)) else ""

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

            # Always construct transform from bounds and target dimensions to guarantee alignment after resampling
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
            # Rasterize output is top-down; flip vertically to align with bottom-up DEM grid
            return mask[::-1].astype(bool)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Vector outline parsing failed for '{outline_input}': {e}")

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

            if rings:
                from matplotlib.path import Path as MplPath

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

                outer_mask = MplPath(rings[0]).contains_points(pts).reshape((height, width))
                hole_mask = np.zeros((height, width), dtype=bool)
                for hole_ring in rings[1:]:
                    if len(hole_ring) >= 3:
                        hole_mask |= MplPath(hole_ring).contains_points(pts).reshape((height, width))

                return outer_mask & ~hole_mask
    except Exception as e:
        raise ValueError(f"Outline parsing failed for '{outline_input}': {e}")

    return ~np.isnan(dem_grid)


def load_survey_points(
    survey_input: str | Path | os.PathLike | np.ndarray,
    bounds: tuple[float, float, float, float] | None = None,
    dem_grid: np.ndarray | None = None,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    profile_column: str | None = None,
) -> np.ndarray:
    """
    Ingest, validate, and standardize scattered radar or seismic survey point data.

    Accepts file paths (CSV, whitespace-delimited text) or NumPy arrays containing
    either 3 columns ``[X, Y, value]`` or 4 columns ``[X, Y, Z_surface, value]``,
    with an optional profile ID column. Verifies projected metric coordinates, checks
    bounds alignment against the DEM domain, samples DEM surface elevation if column 3
    is absent, and consolidates duplicate coordinates by spatial averaging to ensure
    numerical stability during variography and Kriging.

    Parameters
    ----------
    survey_input : str, Path, os.PathLike, or np.ndarray
        Survey points data source.
    bounds : tuple of float, optional
        Spatial bounding coordinates (minx, miny, maxx, maxy) in meters from the DEM.
    dem_grid : np.ndarray, optional
        2D surface DEM grid of shape (M_rows, N_cols) in meters, used to sample surface
        elevations if missing in the survey points dataset.
    x_coords : np.ndarray, optional
        1D array of DEM cell center X coordinates in meters.
    y_coords : np.ndarray, optional
        1D array of DEM cell center Y coordinates in meters.
    profile_column : str, optional
        Name of the survey profile ID column in CSV input. When present, discrete profile
        identifiers are extracted as an extra column for profile-based cross-validation.

    Returns
    -------
    points : np.ndarray
        Standardized 2D array of shape (N, 4) or (N, 5) with columns:
        - Col 0: X coordinate in projected meters (Easting).
        - Col 1: Y coordinate in projected meters (Northing).
        - Col 2: Surface elevation Z in meters.
        - Col 3: Measured value (e.g., ice thickness in meters or travel time in ns).
        - Col 4 (optional): Discrete survey profile identifier if profile_column was provided.

    Raises
    ------
    ValueError
        If unprojected geographic coordinates (degrees) are detected or if points fall
        completely outside the surface DEM spatial bounds.
    """
    prof_arr: np.ndarray | None = None
    pts: np.ndarray | None = None
    if isinstance(survey_input, (str, Path, os.PathLike)):
        filepath = str(survey_input)
        is_csv = filepath.lower().endswith(".csv")
        if profile_column and is_csv:
            try:
                import pandas as pd

                df_csv = pd.read_csv(filepath)
                if profile_column in df_csv.columns:
                    s_prof = df_csv[profile_column]
                    if pd.api.types.is_numeric_dtype(s_prof):
                        prof_arr = pd.to_numeric(s_prof, errors="coerce").to_numpy(dtype=np.float64)
                    else:
                        codes = pd.factorize(s_prof)[0].astype(np.float64)
                        codes[codes < 0] = np.nan  # factorize flags missing labels with -1
                        prof_arr = codes
                    # Drop the profile column by NAME before building the numeric [X, Y, (Z), value] matrix
                    pts = df_csv.drop(columns=[profile_column]).select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
                else:
                    logger.warning(
                        f"   [Survey Points] Survey profile column '{profile_column}' not found in CSV header {list(df_csv.columns)}. "
                        "Profile-based cross-validation is disabled for this dataset."
                    )
            except Exception as e:
                logger.warning(f"   [Survey Points] Could not parse survey profile column '{profile_column}': {e}")
                prof_arr = None
                pts = None
        if pts is None:
            try:
                pts = np.loadtxt(filepath, delimiter="," if is_csv else None)
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
    if pts.shape[1] == 3:
        if dem_grid is not None and x_coords is not None and y_coords is not None:
            interp_z = RegularGridInterpolator((y_coords, x_coords), dem_grid, bounds_error=False, fill_value=np.nan)
            pts_xy = np.column_stack((pts[:, 1], pts[:, 0]))  # (Y, X)
            z_surf = interp_z(pts_xy)
            pts = np.column_stack((pts[:, 0], pts[:, 1], z_surf, pts[:, 2]))
        else:
            z_surf = np.full(len(pts), np.nan)
            pts = np.column_stack((pts[:, 0], pts[:, 1], z_surf, pts[:, 2]))

    if prof_arr is not None and len(prof_arr) == len(pts) and pts.shape[1] == 4:
        pts = np.column_stack((pts, prof_arr))

    # Deduplicate / consolidate duplicate survey points with identical (X, Y) coordinates
    if len(pts) > 1:
        coords_rounded = np.round(pts[:, :2], decimals=1)
        unique_coords, inverse_indices = np.unique(coords_rounded, axis=0, return_inverse=True)
        if len(unique_coords) < len(pts):
            n_dups = len(pts) - len(unique_coords)
            logger.warning(
                f"   [Survey Points] Detected {n_dups} duplicate survey point coordinate(s) (X, Y). "
                f"Consolidating duplicate points by averaging values to ensure numerical stability in Kriging."
            )
            n_cols = pts.shape[1]
            consolidated = np.zeros((len(unique_coords), n_cols), dtype=np.float64)
            for idx in range(len(unique_coords)):
                mask = (inverse_indices == idx)
                consolidated[idx, :2] = pts[mask, :2].mean(axis=0)
                for col in range(2, n_cols):
                    if col == 4:  # discrete profile ID
                        consolidated[idx, col] = pts[mask, col][0]
                    else:
                        consolidated[idx, col] = np.nanmean(pts[mask, col])
            pts = consolidated

    return pts


def save_points_csv(
    points: np.ndarray,
    filepath: str | Path,
    headers: list[str] | None = None,
) -> str:
    """
    Exports a 2D point array (e.g. migrated survey points) to a CSV text file.

    Parameters
    ----------
    points : np.ndarray
        2D array of point coordinates and values (e.g. shape (N, 4) for [x, y, z_surf, depth]).
    filepath : str or Path
        Target file path (e.g. 'output_migrated_points.csv').
    headers : list of str, optional
        Header column names. If None, defaults to ['x', 'y', 'z_surface', 'depth_migrated'].

    Returns
    -------
    saved_path : str
        Path of the saved CSV file.
    """
    path_obj = Path(filepath).expanduser()
    if path_obj.suffix.lower() != ".csv":
        path_obj = path_obj.with_suffix(".csv")

    abs_target_path = str(path_obj.resolve())
    abs_parent = Path(abs_target_path).parent
    abs_parent.mkdir(parents=True, exist_ok=True)

    if headers is None:
        if points.ndim > 1 and points.shape[1] == 4:
            headers = ["x", "y", "z_surface", "depth_migrated"]
        elif points.ndim > 1 and points.shape[1] == 3:
            headers = ["x", "y", "value"]
        else:
            headers = [f"col_{i+1}" for i in range(points.shape[1] if points.ndim > 1 else 1)]

    header_str = ",".join(headers)
    np.savetxt(abs_target_path, points, delimiter=",", header=header_str, comments="", fmt="%.6f")
    return abs_target_path


@dataclass
class IceVolumeResult:
    """Quantitative glacier domain volume, area, and uncertainty summary.

    Parameters
    ----------
    volume_m3 : float
        Total ice volume in cubic meters [m³].
    volume_km3 : float
        Total ice volume in cubic kilometers [km³].
    uncertainty_m3 : float or None
        Propagated volume uncertainty in cubic meters [± m³] if Kriging uncertainty was evaluated.
    uncertainty_km3 : float or None
        Propagated volume uncertainty in cubic kilometers [± km³] if Kriging uncertainty was evaluated.
    area_m2 : float
        Glacier active domain surface area in square meters [m²].
    area_km2 : float
        Glacier active domain surface area in square kilometers [km²].
    mean_thickness : float
        Domain-average ice thickness in meters [m].
    mean_uncertainty : float or None
        Domain-average Kriging thickness uncertainty in meters [± m].
    cell_count : int
        Number of active glaciated DEM cells evaluated.
    """

    volume_m3: float
    volume_km3: float
    uncertainty_m3: float | None
    uncertainty_km3: float | None
    area_m2: float
    area_km2: float
    mean_thickness: float
    mean_uncertainty: float | None
    cell_count: int

    def __iter__(self):
        """Allows 2-tuple unpacking: volume_m3, uncertainty_m3 = result."""
        yield self.volume_m3
        yield self.uncertainty_m3


def calculate_ice_volume(
    thickness: np.ndarray,
    dx: float,
    dy: float,
    outline_mask: np.ndarray | None = None,
    kriged_std: np.ndarray | None = None,
) -> IceVolumeResult:
    """
    Computes total ice volume, active glacier domain area, and propagated Kriging uncertainty.

    Following established glaciological error propagation methodology (Binder et al., 2009; Fischer, 2009),
    spatial autocorrelation across the glacier domain is captured via:
    .. math::
        V = A_{\\text{total}} \\cdot \\bar{D} = \\sum_{i \\in \\text{glacier}} D_i \\cdot \\Delta A
    .. math::
        \\sigma_V = A_{\\text{total}} \\cdot \\bar{\\sigma}_D = \\sum_{i \\in \\text{glacier}} \\sigma_i \\cdot \\Delta A

    Parameters
    ----------
    thickness : np.ndarray
        2D array of ice thickness in meters.
    dx : float
        Pixel grid resolution along X in meters.
    dy : float
        Pixel grid resolution along Y in meters.
    outline_mask : np.ndarray or None, optional
        2D boolean active domain mask. If None, derived from positive finite thickness values.
    kriged_std : np.ndarray or None, optional
        2D array of Kriging thickness standard uncertainty in meters.

    Returns
    -------
    result : IceVolumeResult
        Structured dataclass containing area, volume, and propagated uncertainty in both m³ and km³.
    """
    cell_area = abs(float(dx) * float(dy))
    finite_thick = np.isfinite(thickness) & (thickness > 0)
    valid_mask = (outline_mask & finite_thick) if (outline_mask is not None and np.any(outline_mask)) else finite_thick

    valid_thick = thickness[valid_mask]
    n_cells = int(valid_thick.size)
    if n_cells == 0:
        return IceVolumeResult(
            volume_m3=0.0,
            volume_km3=0.0,
            uncertainty_m3=None,
            uncertainty_km3=None,
            area_m2=0.0,
            area_km2=0.0,
            mean_thickness=0.0,
            mean_uncertainty=None,
            cell_count=0,
        )

    area_m2 = float(n_cells * cell_area)
    area_km2 = float(area_m2 / 1e6)
    mean_thick = float(np.mean(valid_thick))
    vol_m3 = float(np.sum(valid_thick) * cell_area)
    vol_km3 = float(vol_m3 / 1e9)

    unc_m3: float | None = None
    unc_km3: float | None = None
    mean_unc: float | None = None
    if kriged_std is not None:
        valid_unc = kriged_std[valid_mask & np.isfinite(kriged_std)]
        if valid_unc.size > 0:
            mean_unc = float(np.mean(valid_unc))
            unc_m3 = float(area_m2 * mean_unc)
            unc_km3 = float(unc_m3 / 1e9)

    return IceVolumeResult(
        volume_m3=vol_m3,
        volume_km3=vol_km3,
        uncertainty_m3=unc_m3,
        uncertainty_km3=unc_km3,
        area_m2=area_m2,
        area_km2=area_km2,
        mean_thickness=mean_thick,
        mean_uncertainty=mean_unc,
        cell_count=n_cells,
    )

