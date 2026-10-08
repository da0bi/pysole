"""
Surface Gradient, Slope, and Frequency Domain FFT Smoothing for PySole.
Ported from MATLAB scripts GradRad.m and FFTSmooth.m by Daniel Binder (2011).
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq


def compute_surface_curvature(
    dem: np.ndarray, dx: float = 1.0, dy: float = 1.0
) -> np.ndarray:
    """
    Computes 2D surface Laplacian curvature kappa = d2Z/dx2 + d2Z/dy2
    using 2nd-order central finite differences.
    """
    slope_y, slope_x = np.gradient(dem, dy, dx)
    d2z_dy2, _ = np.gradient(slope_y, dy, dx)
    _, d2z_dx2 = np.gradient(slope_x, dy, dx)
    return d2z_dx2 + d2z_dy2


def compute_slope_rad(dem: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Fast computation of surface slope in radians without auxiliary aspect/curvature grids."""
    slope_y, slope_x = np.gradient(dem, dy, dx)
    return np.arctan(np.hypot(slope_x, slope_y))


def compute_gradients(
    dem: np.ndarray, dx: float = 1.0, dy: float = 1.0
) -> dict[str, np.ndarray]:
    """
    Computes spatial slope gradients, surface curvature, and surface normal trigonometric grids.
    Ported from GradRad.m.
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
    """Encapsulates pre-computed 2D FFT spectra with boundary padding and mask metadata."""
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
    Pre-computes 2D Forward FFT spectra with reflect boundary padding and spatial wavenumber grid.
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
    Fast Gaussian low-pass filtering using pre-computed FFT spectra.
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

    if kc <= 0:
        filt = np.ones_like(k_grid, dtype=np.float64)
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
    """
    A_shift, k_grid, k_max = precompute_fft_grid(grid, dx=dx, dy=dy, kc=kc)
    grid_filtered = fft_gaussian_smooth_precomputed(A_shift, k_grid, kc=kc)
    return grid_filtered, k_grid, k_max

