"""
Surface Gradient, Slope, and Frequency Domain FFT Smoothing for PySole.
Ported from MATLAB scripts GradRad.m and FFTSmooth.m by Daniel Binder (2011).
"""

from dataclasses import dataclass
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq


def compute_surface_curvature(
    dem: np.ndarray, dx: float = 1.0, dy: float = 1.0
) -> np.ndarray:
    """
    Computes 2D surface Laplacian curvature using 2nd-order central finite differences.

    The surface Laplacian is defined as:
    .. math::
        \\kappa = \\nabla^2 z = \\frac{\\partial^2 z}{\\partial x^2} + \\frac{\\partial^2 z}{\\partial y^2}

    In glaciology and geomorphology, positive curvature values represent concave
    depressions, cirques, and valley troughs, while negative values represent
    convex peaks, nunataks, and ridges.

    Parameters
    ----------
    dem : np.ndarray
        2D raster grid of surface elevations [m] with shape ``(height, width)``.
    dx : float, default 1.0
        Grid cell spacing along the X-axis (easting) [m].
    dy : float, default 1.0
        Grid cell spacing along the Y-axis (northing) [m].

    Returns
    -------
    curvature : np.ndarray
        2D grid of surface Laplacian curvature values [m⁻¹] with identical shape to ``dem``.
    """
    slope_y, slope_x = np.gradient(dem, dy, dx)
    d2z_dy2, _ = np.gradient(slope_y, dy, dx)
    _, d2z_dx2 = np.gradient(slope_x, dy, dx)
    return d2z_dx2 + d2z_dy2


def compute_slope_rad(dem: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """
    Computes the maximum local terrain slope angle in radians.

    Calculated from the Euclidean norm of spatial elevation gradients:
    .. math::
        \\alpha = \\arctan\\left(\\sqrt{(\\partial z / \\partial x)^2 + (\\partial z / \\partial y)^2}\\right)

    Parameters
    ----------
    dem : np.ndarray
        2D raster grid of surface elevations [m].
    dx : float, default 1.0
        Spatial grid resolution along X [m].
    dy : float, default 1.0
        Spatial grid resolution along Y [m].

    Returns
    -------
    slope_rad : np.ndarray
        2D grid of terrain slope angles in radians [rad], ranging from 0.0 (flat) to < pi/2 (vertical).
    """
    slope_y, slope_x = np.gradient(dem, dy, dx)
    return np.arctan(np.hypot(slope_x, slope_y))


def compute_gradients(
    dem: np.ndarray, dx: float = 1.0, dy: float = 1.0
) -> dict[str, np.ndarray]:
    """
    Computes terrain slopes, surface normals, and Laplacian curvature for glaciological modeling.

    Evaluates directional partial derivatives using central finite differences and derives
    trigonometric projection factors required for 3D Eikonal ray normal migration and
    Shallow Ice Approximation (SIA) basal driving stress calculations.

    Parameters
    ----------
    dem : np.ndarray
        2D surface digital elevation model [m] with shape ``(M, N)``.
    dx : float, default 1.0
        Spatial cell resolution along the X dimension [m].
    dy : float, default 1.0
        Spatial cell resolution along the Y dimension [m].

    Returns
    -------
    gradients : dict of str to np.ndarray
        Dictionary containing spatial derivative fields:
        - ``'slope_rad'``: Maximum terrain slope in radians [rad].
        - ``'slope_grad'``: Maximum terrain slope in degrees [°].
        - ``'slope_x'``: Directional partial derivative dz/dx [m/m] (dimensionless).
        - ``'slope_y'``: Directional partial derivative dz/dy [m/m] (dimensionless).
        - ``'cos_alpha_x_grid'``: Direction cosine nx of surface normal vector.
        - ``'sin_alpha_x_grid'``: Slope projection factor along X.
        - ``'cos_alpha_y_grid'``: Direction cosine ny of surface normal vector.
        - ``'sin_alpha_y_grid'``: Slope projection factor along Y.
        - ``'sinus_alpha_grid'``: True sine of total surface slope sin(alpha) (dimensionless).
        - ``'curvature'``: 2D surface Laplacian curvature [m⁻¹].
    """
    slope_y, slope_x = np.gradient(dem, dy, dx)

    slope_sq = slope_x**2 + slope_y**2
    slope = np.sqrt(slope_sq)
    slope_rad = np.arctan(slope)
    slope_grad = np.degrees(slope_rad)

    d2z_dy2, _ = np.gradient(slope_y, dy, dx)
    _, d2z_dx2 = np.gradient(slope_x, dy, dx)
    curvature = d2z_dx2 + d2z_dy2

    inv_hypot_x = 1.0 / np.sqrt(1.0 + slope_x**2)
    inv_hypot_y = 1.0 / np.sqrt(1.0 + slope_y**2)
    inv_hypot_slope = 1.0 / np.sqrt(1.0 + slope_sq)

    cos_alpha_x_grid = inv_hypot_x
    sin_alpha_x_grid = slope_x * inv_hypot_x
    cos_alpha_y_grid = inv_hypot_y
    sin_alpha_y_grid = slope_y * inv_hypot_y
    sinus_alpha_grid = slope * inv_hypot_slope

    return {
        "slope_rad": slope_rad,
        "slope_grad": slope_grad,
        "slope_x": slope_x,
        "slope_y": slope_y,
        "cos_alpha_x_grid": cos_alpha_x_grid,
        "cos_alpha_y_grid": cos_alpha_y_grid,
        "sin_alpha_x_grid": sin_alpha_x_grid,
        "sin_alpha_y_grid": sin_alpha_y_grid,
        "sinus_alpha_grid": sinus_alpha_grid,
        "curvature": curvature,
    }


@dataclass
class FFTSpectrum:
    """
    Container storing pre-computed 2D Fast Fourier Transform (FFT) complex spectra.

    Attributes
    ----------
    data_fft : np.ndarray
        Shifted 2D complex Fourier transform of the padded elevation grid.
    mask_fft : np.ndarray or None
        Shifted 2D complex Fourier transform of the binary valid-data mask (for NaN normalization).
    pad_m : int
        Number of padded grid rows added to top and bottom to eliminate periodic boundary artifacts.
    pad_n : int
        Number of padded grid columns added to left and right.
    orig_shape : tuple of (int, int)
        Original unpadded raster grid dimensions ``(height, width)``.
    has_nans : bool
        True if the original input grid contains NaN (missing data or ice-free) cells.
    nan_mask : np.ndarray or None
        Boolean mask of NaN locations in the original unpadded grid.
    """
    data_fft: np.ndarray
    mask_fft: np.ndarray | None
    pad_m: int
    pad_n: int
    orig_shape: tuple[int, int]
    has_nans: bool
    nan_mask: np.ndarray | None


def precompute_fft_grid(
    grid: np.ndarray, dx: float = 1.0, dy: float = 1.0, kc: float | None = None
) -> tuple[FFTSpectrum, np.ndarray, float]:
    """
    Pre-computes 2D Forward FFT spectra with reflection boundary padding.

    In glacier surface smoothing, local high-frequency topographic roughness (crevasses,
    seracs, and wind-blown sastrugi) is filtered out in the frequency domain. Reflective
    boundary padding is applied around the DEM perimeter prior to the forward FFT to prevent
    spurious edge oscillations (Gibbs phenomenon).

    Parameters
    ----------
    grid : np.ndarray
        2D input elevation raster grid [m].
    dx : float, default 1.0
        Spatial grid spacing along X [m].
    dy : float, default 1.0
        Spatial grid spacing along Y [m].
    kc : float, optional
        Target corner frequency cutoff wavenumber [rad/m] used to size boundary padding.
        If None, a conservative default padding width is used.

    Returns
    -------
    spectrum : FFTSpectrum
        Container storing pre-computed Fourier spectra and padding metadata.
    k_grid : np.ndarray
        2D grid of continuous radial spatial wavenumbers :math:`k = \\sqrt{k_x^2 + k_y^2}` [rad/m].
    k_max : float
        Maximum spatial wavenumber present in the discrete Fourier spectrum [rad/m].
    """
    M, N = grid.shape
    nan_mask = np.isnan(grid)
    has_nans = bool(np.any(nan_mask))

    grid_clean = np.where(nan_mask, 0.0, grid)

    kc_eff = float(kc) if (kc is not None and kc > 0) else 0.01
    sigma_px_y = 1.0 / (kc_eff * abs(dy))
    sigma_px_x = 1.0 / (kc_eff * abs(dx))
    pad_m = min(M, max(32, int(np.ceil(4.0 * sigma_px_y))))
    pad_n = min(N, max(32, int(np.ceil(4.0 * sigma_px_x))))

    if pad_m > 0 or pad_n > 0:
        grid_padded = np.pad(grid_clean, ((pad_m, pad_m), (pad_n, pad_n)), mode="reflect")
        mask_padded = np.pad((~nan_mask).astype(np.float64), ((pad_m, pad_m), (pad_n, pad_n)), mode="reflect") if has_nans else None
    else:
        grid_padded = grid_clean
        mask_padded = (~nan_mask).astype(np.float64) if has_nans else None

    M_pad, N_pad = grid_padded.shape

    kx = fftshift(fftfreq(N_pad, d=abs(dx))) * (2.0 * np.pi)
    ky = fftshift(fftfreq(M_pad, d=abs(dy))) * (2.0 * np.pi)

    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    k_max = float(np.ceil(np.max(k_grid)))

    data_fft = fftshift(fft2(grid_padded))
    mask_fft = fftshift(fft2(mask_padded)) if mask_padded is not None else None

    spectrum = FFTSpectrum(
        data_fft=data_fft,
        mask_fft=mask_fft,
        pad_m=pad_m,
        pad_n=pad_n,
        orig_shape=(M, N),
        has_nans=has_nans,
        nan_mask=nan_mask if has_nans else None,
    )
    return spectrum, k_grid, k_max


def fft_gaussian_smooth_precomputed(
    spectrum: FFTSpectrum | np.ndarray, k_grid: np.ndarray, kc: float = 0.05
) -> np.ndarray:
    """
    Applies Gaussian low-pass filtering using pre-computed Fourier spectra.

    Multiplies the 2D Fourier spectrum by the Gaussian transfer function:
    .. math::
        G(k) = \\exp\\left(-\\frac{k^2}{2 k_c^2}\\right)

    where :math:`k_c` is the corner frequency cutoff wavenumber [rad/m], corresponding
    to cutoff wavelength :math:`\\lambda_c = 2\\pi / k_c` [m]. The inverse FFT is then
    evaluated and boundary padding is unpadded.

    Parameters
    ----------
    spectrum : FFTSpectrum or np.ndarray
        Pre-computed FFT spectrum container (from :func:`precompute_fft_grid`).
    k_grid : np.ndarray
        2D spatial wavenumber grid [rad/m].
    kc : float, default 0.05
        Corner frequency cutoff wavenumber :math:`k_c` [rad/m]. Smaller :math:`k_c`
        values produce heavier smoothing (longer cutoff wavelengths).

    Returns
    -------
    smoothed_grid : np.ndarray
        Filtered spatial elevation grid [m] matching the original unpadded dimensions.
    """
    if isinstance(spectrum, FFTSpectrum):
        data_fft = spectrum.data_fft
        mask_fft = spectrum.mask_fft
        pad_m, pad_n = spectrum.pad_m, spectrum.pad_n
        M, N = spectrum.orig_shape
        has_nans = spectrum.has_nans
        nan_mask = spectrum.nan_mask
    else:
        data_fft = spectrum
        mask_fft = None
        pad_m, pad_n = 0, 0
        M, N = spectrum.shape
        has_nans = False
        nan_mask = None

    if kc < 0:
        raise ValueError(f"Invalid corner frequency kc={kc}. Corner frequency must be non-negative (kc >= 0).")
    elif kc == 0:
        filt = np.zeros_like(k_grid, dtype=np.float64)
        filt[k_grid == 0] = 1.0
    else:
        filt = np.exp(-(k_grid**2) / (2.0 * (kc**2)))

    data_filtered_padded = np.real(ifft2(ifftshift(data_fft * filt)))

    if pad_m > 0 or pad_n > 0:
        data_filtered = data_filtered_padded[pad_m : pad_m + M, pad_n : pad_n + N]
    else:
        data_filtered = data_filtered_padded

    if has_nans and mask_fft is not None:
        mask_filtered_padded = np.real(ifft2(ifftshift(mask_fft * filt)))
        if pad_m > 0 or pad_n > 0:
            mask_filtered = mask_filtered_padded[pad_m : pad_m + M, pad_n : pad_n + N]
        else:
            mask_filtered = mask_filtered_padded

        valid_mask = mask_filtered > 1e-3
        grid_filtered = np.where(valid_mask, data_filtered / np.maximum(mask_filtered, 1e-3), np.nan)
        if nan_mask is not None and np.any(nan_mask):
            grid_filtered[nan_mask] = np.nan
    else:
        grid_filtered = data_filtered

    return grid_filtered


def fft_gaussian_smooth(
    grid: np.ndarray, dx: float = 1.0, dy: float = 1.0, kc: float = 0.05
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Performs 2D spatial smoothing in the frequency domain using a Gaussian low-pass filter.

    This is the complete end-to-end wrapper that pre-computes the boundary-padded FFT,
    applies the Gaussian spectral transfer function, and computes the inverse FFT.

    Parameters
    ----------
    grid : np.ndarray
        2D input elevation raster grid [m].
    dx : float, default 1.0
        Spatial grid spacing along X [m].
    dy : float, default 1.0
        Spatial grid spacing along Y [m].
    kc : float, default 0.05
        Corner frequency cutoff wavenumber :math:`k_c` [rad/m]. Corresponding cutoff
        wavelength is :math:`\\lambda_c = 2\\pi / k_c` [m].

    Returns
    -------
    smoothed_grid : np.ndarray
        Filtered spatial elevation grid [m].
    k_grid : np.ndarray
        2D spatial wavenumber grid [rad/m].
    k_max : float
        Maximum spatial wavenumber present in the discrete Fourier spectrum [rad/m].
    """
    A_shift, k_grid, k_max = precompute_fft_grid(grid, dx=dx, dy=dy, kc=kc)
    grid_filtered = fft_gaussian_smooth_precomputed(A_shift, k_grid, kc=kc)
    return grid_filtered, k_grid, k_max

