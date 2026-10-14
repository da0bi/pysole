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

The module also owns the BLAS thread policy (:func:`blas_thread_limit`): dense factorisations run with at most
the number of physical cores (more BLAS threads are slower, e.g. 15.6 s versus 0.5 s for a 2501 x 2501 inverse on a
16-thread machine), and the already parallel grid chunks run with one BLAS thread each.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psutil
from threadpoolctl import threadpool_limits

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
    """
    Computes the optimal grid cell chunk size for parallel Dual Kriging evaluation.

    To balance computational throughput and peak memory usage, the output raster
    grid is partitioned into contiguous spatial chunks. Each worker thread processes
    one chunk of cells at a time, calculating cross-distances against all `n_pts`
    sample observations.

    Parameters
    ----------
    n_pts : int
        Number of valid survey observations (e.g., radar sounding points).

    Returns
    -------
    chunk_size : int
        Number of grid cells evaluated per parallel work chunk (clamped between 500 and 10,000 cells).
    """
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
    Estimates the peak RAM footprint [bytes] of the native vector Kriging engine.

    The peak memory consumption is determined by the maximum of:
    1. The covariance matrix assembly phase: allocating the dense augmented
       Kriging system matrix of size ``(n_pts + n_drift) x (n_pts + n_drift)``
       along with scratch matrices.
    2. The parallel grid evaluation phase: storing the LU factorisation
       (required for Kriging variance) plus per-thread working memory
       proportional to ``n_pts * chunk_size``.

    Parameters
    ----------
    n_pts : int
        Number of valid survey observations.
    n_drift : int
        Number of drift columns in the augmented Kriging system (including the constant 1.0).
    n_threads : int
        Number of parallel worker threads planned.
    with_variance : bool
        If True, includes the memory overhead of the LU factorisation and variance weight solves.
    n_grid_cells : int, default 0
        Total number of grid cells in the output raster (``height * width``).

    Returns
    -------
    peak_bytes : int
        Estimated peak memory requirement in bytes.
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
    """
    Execution plan returned by :func:`plan_native_kriging`.

    Attributes
    ----------
    return_variance : bool
        Whether Kriging estimation variance should be calculated or skipped.
    n_threads : int
        Effective number of parallel worker threads allocated.
    est_bytes : int
        Estimated peak memory footprint [bytes].
    budget_bytes : int or None
        Allocated memory budget ceiling [bytes], or None if system memory is undetermined.
    message : str
        Diagnostic log message explaining any thread throttling or variance deactivation.
    """

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
    Derives an adaptive thread and memory allocation plan for native Dual Kriging.

    Prevents out-of-memory (OOM) operating system kills during large-scale raster
    interpolation. If the estimated peak memory exceeds the available RAM budget:
    1. Worker thread count is reduced iteratively (lowering per-thread scratch memory).
    2. If even a single thread with estimation variance exceeds the budget, the Kriging
       variance calculation is skipped (filling the variance grid with NaN) while
       preserving the high-performance Dual Kriging elevation prediction.
    3. If even single-threaded elevation prediction exceeds the budget, a warning is logged
       and execution proceeds single-threaded as best effort.

    Parameters
    ----------
    n_pts : int
        Number of valid survey observations.
    n_drift : int
        Number of drift columns in the augmented Kriging system.
    n_threads : int
        Requested worker thread count.
    want_variance : bool
        Whether Kriging estimation variance is desired by the user.
    fraction : float
        Allowed memory fraction of currently free RAM (``0.0 < fraction <= 0.9``).
    n_grid_cells : int, default 0
        Total number of output raster cells (``height * width``).

    Returns
    -------
    plan : MemoryPlan
        Validated execution plan containing the final thread count, variance evaluation flag,
        and peak memory estimate.
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
    """
    Translates user-configured CPU core settings into an effective worker thread count.

    Parameters
    ----------
    n_cores : int or None
        Requested CPU core count. If -1 or None, defaults to all logical CPU cores
        detected on the system via ``os.cpu_count()``. Positive integers specify an
        exact core limit.

    Returns
    -------
    worker_threads : int
        Effective positive worker thread count (minimum 1).
    """
    return (os.cpu_count() or 1) if (n_cores == -1 or n_cores is None) else max(1, int(n_cores))


def physical_core_count() -> int:
    """Number of physical CPU cores (falls back to the logical count, minimum 1)."""
    return max(1, psutil.cpu_count(logical=False) or os.cpu_count() or 1)


@contextmanager
def blas_thread_limit(limit: int | None = None) -> Iterator[int]:
    """
    Context manager that caps the number of BLAS/LAPACK threads and restores the previous setting on exit
    (also when the body raises).

    Parameters
    ----------
    limit : int, optional
        Maximum number of BLAS threads. ``None`` uses :func:`physical_core_count`. The cap is process-wide while
        the context is active, so use ``blas_thread_limit(1)`` around a block that already runs in parallel
        threads (such as the grid chunks) instead of inside the individual threads.

    Yields
    ------
    effective_limit : int
        The cap that was applied.
    """
    effective = max(1, int(limit)) if limit is not None else physical_core_count()
    with threadpool_limits(limits=effective, user_api="blas"):
        yield effective


# Drift Analyzer cross-validation: the Ordinary base system needs the point matrix, its LU factors and the explicit
# inverse (3 matrices of (N + 1)^2 doubles at the peak, 2 without the inverse), plus the stacked block matrices.
_CV_MATRICES_FULL = 3.5  # measured/derived 3.0 plus margin
_CV_MATRICES_FALLBACK = 2.5  # point matrix + LU factors (2.0) plus margin
_CV_WORKING_BYTES = 256 * 1024**2  # stacked (g, m, m) block matrices and temporaries
CV_MIN_EVALUATION_POINTS = 100
"""Smallest number of evaluation points of the large-N fallback."""


@dataclass
class CvMemoryPlan:
    """
    Plan returned by :func:`plan_cv_memory`.

    Attributes
    ----------
    full_inverse : bool
        True if the explicit inverse of the base system fits into the budget (fastest path).
    n_eval : int or None
        Number of evaluation points of the large-N fallback (a deterministic farthest-point subset);
        ``None`` evaluates every observation.
    message : str
        Log message describing a fallback (empty if everything fits).
    """

    full_inverse: bool
    n_eval: int | None
    message: str = ""


def plan_cv_memory(n_pts: int, mean_block_size: float = 1.0, fraction: float = 0.5) -> CvMemoryPlan:
    """
    Decides how the Drift Analyzer evaluates its cross-validation for ``n_pts`` observations.

    The exact block leave-out evaluation needs about ``3 N^2`` doubles (point matrix, LU factors, inverse; N = 5000
    -> 0.6 GB, N = 10 000 -> 2.4 GB). If that does not fit into ``fraction`` of the available memory, the LU factors
    are kept instead of the inverse and only the inverse columns of a deterministic farthest-point subset of
    evaluation points (and the members of their blocks) are computed, never fewer than
    :data:`CV_MIN_EVALUATION_POINTS`.

    Parameters
    ----------
    n_pts : int
        Number of observations.
    mean_block_size : float, default 1.0
        Mean number of observations removed per evaluation point (buffer neighbourhood or profile size).
    fraction : float, default 0.5
        Fraction of the available memory that may be used.

    Returns
    -------
    plan : CvMemoryPlan
    """
    avail = available_memory_bytes()
    if avail is None:
        return CvMemoryPlan(True, None)
    n_aug = int(n_pts) + 1
    budget = float(fraction) * avail
    full_bytes = _CV_MATRICES_FULL * _FLOAT64 * n_aug**2 + _CV_WORKING_BYTES
    if full_bytes <= budget:
        return CvMemoryPlan(True, None)

    fixed = _CV_MATRICES_FALLBACK * _FLOAT64 * n_aug**2 + _CV_WORKING_BYTES
    per_column = _FLOAT64 * n_aug
    n_columns = max(0.0, (budget - fixed) / per_column)
    n_eval = int(n_columns / max(float(mean_block_size), 1.0))
    if n_eval >= n_pts:
        return CvMemoryPlan(False, None, "   [Memory Guard] The cross-validation keeps LU factors instead of the explicit inverse.")
    n_eval = max(CV_MIN_EVALUATION_POINTS, n_eval)
    n_eval = min(n_eval, int(n_pts))
    msg = (
        f"   [Memory Guard] Exact cross-validation of all N = {n_pts} points needs about "
        f"{_gb(full_bytes):.1f} GB but only {_gb(budget):.1f} GB are allowed ({fraction:.0%} of {_gb(avail):.1f} GB "
        f"available). Evaluating a deterministic farthest-point subset of {n_eval} points instead."
    )
    if fixed > budget:
        msg += " Even the LU factors exceed the budget; the run may exhaust memory."
    return CvMemoryPlan(False, n_eval, msg)
