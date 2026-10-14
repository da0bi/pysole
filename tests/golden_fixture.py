"""Deterministic fixture + case list for the golden regression of the native kriging engine.

The stored arrays in ``tests/data/golden_kriging_engine.npz`` were produced by the engine *before* the
shared ``KrigingSystem`` refactor (Phase 0 of the multi-drift follow-up); the engine must keep reproducing
them (Ordinary + all 23 explicit drift models, kriging estimate and variance).
"""
from __future__ import annotations

import itertools

import numpy as np

from pysole.raster import GridGeometry

PRIMITIVES = ["sia", "z_dem", "curvature_dem", "linear_xy", "quadratic_xy"]
VARIOGRAM_PARAMS = (150.0, 25.0, 0.5)  # range, sill, nugget


def drift_models() -> list[list[str]]:
    """All 23 explicit drift models (non-empty subsets; ``linear_xy`` and ``quadratic_xy`` never together)."""
    out = []
    for r in range(1, len(PRIMITIVES) + 1):
        for comb in itertools.combinations(PRIMITIVES, r):
            if "linear_xy" in comb and "quadratic_xy" in comb:
                continue
            out.append(list(comb))
    return out


def cases() -> list[tuple[str, str, list[str] | None]]:
    """(case name, method, drift_terms)."""
    out = [("ordinary", "ordinary", None)]
    out += [("universal:" + "+".join(t), "universal", t) for t in drift_models()]
    return out


def build_fixture() -> dict:
    """Small asymmetric (north/south, east/west) surface with 60 survey points on a 30 x 30 grid."""
    rng = np.random.default_rng(20261013)
    geom = GridGeometry.create((30, 30), dx=20.0, dy=20.0, bounds=(0, 0, 600, 600))
    X, Y = geom.meshgrid
    dem = 2000 - 0.3 * X - 0.1 * Y + 20 * np.sin(X / 60) * np.cos(Y / 80)
    gy, gx = np.gradient(dem, 20.0)
    slope = np.arctan(np.hypot(gx, gy)) + np.radians(6.0)
    n = 60
    xs, ys = rng.uniform(10, 590, n), rng.uniform(10, 590, n)
    zs = 0.02 * (2000 - dem[(ys // 20).astype(int), (xs // 20).astype(int)]) + rng.normal(0, 0.5, n) + 30
    return {"pts": np.column_stack((xs, ys, zs)), "geometry": geom, "dem": dem, "slope_rad": slope}
