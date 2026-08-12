"""
Surface Gradient, Slope, and Frequency Domain FFT Smoothing for PySole.
Ported from MATLAB scripts GradRad.m and FFTSmooth.m by Daniel Binder (2011).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def compute_gradients(
    dem: np.ndarray, dx: float = 1.0, dy: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Computes spatial slope gradients and surface normal trigonometric grids.
    Ported from GradRad.m.

    Parameters
    ----------
    dem : 2D np.ndarray
        Surface elevation grid.
    dx : float
        Grid spacing along X (columns).
    dy : float
        Grid spacing along Y (rows).

    Returns
    -------
    dict containing:
        - 'slope_rad': slope angle in radians
        - 'slope_grad': slope angle in degrees
        - 'slope_x': gradient in X direction
        - 'slope_y': gradient in Y direction
        - 'cos_alpha_x_grid': cos(atan(Slope_x))
        - 'cos_alpha_y_grid': cos(atan(Slope_y))
        - 'sin_alpha_x_grid': sin(atan(Slope_x))
        - 'sin_alpha_y_grid': sin(atan(Slope_y))
        - 'sinus_alpha_grid': sin(Slope_rad)
    """
    # np.gradient returns gradients along axis 0 (rows/y) then axis 1 (cols/x)
    slope_y, slope_x = np.gradient(dem, dy, dx)

    slope = np.sqrt(slope_x**2 + slope_y**2)
    slope_rad = np.arctan(slope)
    slope_x_rad = np.arctan(slope_x)
    slope_y_rad = np.arctan(slope_y)

    cos_alpha_x_grid = np.cos(slope_x_rad)
    cos_alpha_y_grid = np.cos(slope_y_rad)
    sin_alpha_x_grid = np.sin(slope_x_rad)
    sin_alpha_y_grid = np.sin(slope_y_rad)

    sinus_alpha_grid = np.sin(slope_rad)
    slope_grad = np.degrees(slope_rad)

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
    }


def fft_gaussian_smooth(
    grid: np.ndarray, dx: float = 1.0, dy: float = 1.0, kc: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Performs 2D spatial smoothing in the frequency domain using a Gaussian low-pass filter.
    Ported from FFTSmooth.m.

    Parameters
    ----------
    grid : 2D np.ndarray
        Input surface grid to smooth.
    dx : float
        Grid spacing along X (columns).
    dy : float
        Grid spacing along Y (rows).
    kc : float
        Corner frequency for the Gaussian low-pass filter.

    Returns
    -------
    grid_filtered : 2D np.ndarray
        Smoothed surface grid.
    k_grid : 2D np.ndarray
        Wavenumber magnitude grid.
    k_max : float
        Maximum wavenumber.
    """
    grid_clean = np.nan_to_num(grid, nan=np.nanmean(grid))
    M, N = grid_clean.shape

    kx1 = np.mod(0.5 + np.arange(N) / N, 1.0) - 0.5
    kx = np.sort(kx1 * (2.0 * np.pi * abs(dx)))

    ky1 = np.mod(0.5 + np.arange(M) / M, 1.0) - 0.5
    ky = np.sort(ky1 * (2.0 * np.pi * abs(dy)))

    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    k_max = float(np.ceil(np.max(k_grid)))

    A = np.fft.fftshift(np.fft.fft2(grid_clean))

    if kc <= 0:
        filt = np.ones((M, N), dtype=np.float64)
    else:
        filt = np.exp(-(k_grid**2) / (2.0 * kc**2))

    A_filtered = A * filt
    grid_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(A_filtered)))

    return grid_filtered, k_grid, k_max
