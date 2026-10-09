"""
Memory budgeting for the native dual-Kriging engine.

The native engine needs one dense ``(N + d) x (N + d)`` float64 matrix per sample-point count ``N`` plus a
per-thread working set that scales with ``N`` times the evaluation chunk size. For large survey datasets
(``N`` in the tens of thousands) this can exceed the available RAM, so ``plan_native_kriging`` derives a
safe execution plan *before* any large allocation:

1. run as requested if the estimate fits into ``fraction`` of the available memory,
2. otherwise reduce the number of worker threads (the per-thread term is the only part that scales with
   the thread count),
3. otherwise skip the Kriging variance (which keeps the LU factorisation alive during grid evaluation),
4. otherwise run single-threaded without variance and warn (nothing more can be saved).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .logging import logger

_FLOAT64 = 8

# Calibrated against the measured peak RSS of ``built_in_kriging_interpolation`` (N = 1 500 - 4 500 points,
# 1 - 8 threads, 90 000 grid cells; measured: ~3.0 N_aug^2 matrices at the assembly peak, ~5.6 / ~3.4 per-thread
# (N x chunk) arrays with / without variance). The values below are the measurements plus a margin of about 25%.
_SETUP_MATRICES = 4.0  # N_aug^2 matrices alive at the peak of the matrix assembly phase (measured ~3.0)
_PER_THREAD_ARRAYS_VARIANCE = 7.0  # (N x chunk) arrays alive per worker thread, with variance (measured ~5.6)
_PER_THREAD_ARRAYS_PLAIN = 4.5  # same, without variance (measured ~3.4)
_GRID_ARRAYS = 6.0  # grid-sized float64 arrays alive during evaluation (coordinates, outputs, chunk copies)

MAX_FRACTION_CAP = 0.9
"""Upper bound accepted for ``max_memory_fraction``."""


def kriging_chunk_size(n_pts: int) -> int:
    """Number of grid cells evaluated per work chunk (shared by the engine and the estimator)."""
    return max(500, min(10000, 5_000_000 // max(int(n_pts), 1)))


def _cgroup_available_bytes() -> int | None:
    """Remaining memory under a cgroup (container / systemd slice) limit, or None if unlimited."""

    def _read_int(path: str) -> int | None:
        try:
            with open(path) as fh:
                txt = fh.read().strip()
            return None if txt in ("", "max") else int(txt)
        except (OSError, ValueError):
            return None

    def _stat_value(path: str, key: str) -> int:
        try:
            with open(path) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) == 2 and parts[0] == key:
                        return int(parts[1])
        except (OSError, ValueError):
            pass
        return 0

    # cgroup v2, then v1
    limit = _read_int("/sys/fs/cgroup/memory.max")
    if limit is not None:
        used = _read_int("/sys/fs/cgroup/memory.current") or 0
        used -= _stat_value("/sys/fs/cgroup/memory.stat", "inactive_file")  # reclaimable page cache
    else:
        limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if limit is None or limit >= 2**60:  # v1 reports a huge number when unlimited
            return None
        used = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes") or 0
        used -= _stat_value("/sys/fs/cgroup/memory/memory.stat", "total_inactive_file")
    return max(limit - max(used, 0), 0)


def available_memory_bytes() -> int | None:
    """
    Memory currently available to this process in bytes, or None if it cannot be determined.

    Uses ``psutil.virtual_memory().available`` and, if lower, the remaining headroom of a cgroup memory limit.
    """
    try:
        import psutil

        avail: int | None = int(psutil.virtual_memory().available)
    except Exception:  # pragma: no cover - psutil is a hard dependency, this is a last resort
        avail = None

    try:
        cg = _cgroup_available_bytes()
    except Exception:  # pragma: no cover - unreadable cgroup files must never break a run
        cg = None

    if avail is None:
        return cg
    if cg is None:
        return avail
    return min(avail, cg)


def _gb(n_bytes: float) -> float:
    return n_bytes / 1024**3


def estimate_native_kriging_bytes(
    n_pts: int,
    n_drift: int,
    n_threads: int,
    with_variance: bool,
    n_grid_cells: int = 0,
) -> int:
    """
    Estimated peak memory [bytes] of the native Kriging engine.

    The peak is the larger of the matrix-assembly phase (several ``N_aug^2`` matrices, single thread) and the
    grid-evaluation phase (the LU factorisation if the variance is requested, otherwise nothing, plus
    ``n_threads`` per-thread working sets), each on top of the grid-sized arrays.
    """
    n_aug = int(n_pts) + int(n_drift)
    chunk = kriging_chunk_size(n_pts)
    n_chunks = max(1, -(-int(n_grid_cells) // chunk)) if n_grid_cells > 0 else int(n_threads)
    eff_threads = max(1, min(int(n_threads), n_chunks))

    grid_bytes = _GRID_ARRAYS * _FLOAT64 * max(int(n_grid_cells), 0)
    setup_peak = _SETUP_MATRICES * _FLOAT64 * n_aug**2
    arrays = _PER_THREAD_ARRAYS_VARIANCE if with_variance else _PER_THREAD_ARRAYS_PLAIN
    per_thread = arrays * _FLOAT64 * n_pts * chunk
    eval_fixed = _FLOAT64 * n_aug**2 if with_variance else 0.0
    eval_peak = eval_fixed + eff_threads * per_thread
    return int(grid_bytes + max(setup_peak, eval_peak))


@dataclass
class MemoryPlan:
    """Execution plan returned by :func:`plan_native_kriging`."""

    return_variance: bool
    n_threads: int
    est_bytes: int
    budget_bytes: int | None
    message: str = ""


def plan_native_kriging(
    n_pts: int,
    n_drift: int,
    n_threads: int,
    want_variance: bool,
    fraction: float,
    n_grid_cells: int = 0,
) -> MemoryPlan:
    """
    Plans thread count and variance evaluation so that the native engine stays within a RAM budget.

    Parameters
    ----------
    n_pts, n_drift : int
        Number of valid sample points and drift columns (including the constant).
    n_threads : int
        Requested number of worker threads.
    want_variance : bool
        Whether the Kriging variance is requested.
    fraction : float
        Budget as a fraction of the currently available memory, ``0 < fraction <= 0.9``.
    n_grid_cells : int
        Number of output grid cells (``M * N``).
    """
    n_threads = max(1, int(n_threads))
    avail = available_memory_bytes()
    if avail is None:
        logger.info("   [Memory Guard] Available memory could not be determined; guard disabled.")
        return MemoryPlan(want_variance, n_threads, 0, None)

    budget = int(float(fraction) * avail)

    def est(threads: int, var: bool) -> int:
        return estimate_native_kriging_bytes(n_pts, n_drift, threads, var, n_grid_cells)

    # 1. fits as requested
    requested = est(n_threads, want_variance)
    if requested <= budget:
        return MemoryPlan(want_variance, n_threads, requested, budget)

    # 2. reduce threads first (variance kept)
    for threads in range(n_threads - 1, 0, -1):
        if est(threads, want_variance) <= budget:
            msg = (
                f"   [Memory Guard] Reducing worker threads {n_threads} -> {threads} to stay within "
                f"{_gb(budget):.1f} GB ({fraction:.0%} of {_gb(avail):.1f} GB available); "
                f"estimated peak {_gb(est(threads, want_variance)):.1f} GB (was {_gb(requested):.1f} GB)."
            )
            logger.info(msg)
            return MemoryPlan(want_variance, threads, est(threads, want_variance), budget, msg)

    # 3. even one thread does not fit with variance: skip the variance
    if want_variance:
        need_var = est(1, True)
        for threads in range(n_threads, 0, -1):
            if est(threads, False) <= budget:
                msg = (
                    f"   [Memory Guard] Kriging variance skipped: needs about {_gb(need_var):.1f} GB even single-"
                    f"threaded but only {_gb(budget):.1f} GB are allowed ({fraction:.0%} of {_gb(avail):.1f} GB "
                    f"available). Uncertainty will be unavailable. Raise 'max_memory_fraction' (max "
                    f"{MAX_FRACTION_CAP}) or free memory to enable it."
                )
                logger.warning(msg)
                return MemoryPlan(False, threads, est(threads, False), budget, msg)

    # 4. nothing fits: best effort
    msg = (
        f"   [Memory Guard] Estimated {_gb(est(1, False)):.1f} GB needed even single-threaded without variance, "
        f"but only {_gb(budget):.1f} GB are allowed ({fraction:.0%} of {_gb(avail):.1f} GB available). "
        "Proceeding single-threaded without variance; the run may exhaust memory."
    )
    logger.warning(msg)
    return MemoryPlan(False, 1, est(1, False), budget, msg)


def resolve_threads(n_cores: int | None) -> int:
    """Translates the ``n_cores`` setting (-1/None = all CPUs) into a worker thread count."""
    return (os.cpu_count() or 1) if (n_cores == -1 or n_cores is None) else max(1, int(n_cores))
