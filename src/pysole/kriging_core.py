"""
Shared dual-kriging core.

:class:`KrigingSystem` owns the algebra of the augmented kriging system

    A = [[Γ + r·I,  F],
         [Fᵀ,       0]]            with F = [1, drift columns]

for one set of observations: variogram evaluation, the single regularisation rule, the LU factorisation, the dual
weights, the explicit inverse (needed for the estimation variance) and prediction at arbitrary points. The native grid
engine (:func:`pysole.interpolation.built_in_kriging_interpolation`) is a thin chunked loop around it, and the Drift
Analyzer builds on the same assembly and regularisation, so the two can never diverge.

:class:`LeaveBlockOut` evaluates the leave-block-out cross-validation residuals of Ordinary Kriging and of every
drift-augmented model of the same observation set exactly, from one factorisation of the Ordinary system.

Thread safety: after :meth:`KrigingSystem.factorize` (and :meth:`KrigingSystem.inverse` when the variance is wanted)
:meth:`KrigingSystem.predict_points` only reads shared state and may be called from several threads.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist

from .memory import blas_thread_limit
from .variogram import evaluate_variogram_model

#: Relative Tikhonov regularisation added to the diagonal of the point block (fraction of the sill).
REGULARISATION_FRACTION = 1e-6


def resolve_variogram_params(
    variogram_params: tuple[float, float, float] | dict[str, float] | None,
    values: np.ndarray,
    max_distance: float,
) -> tuple[float, float, float]:
    """
    Returns ``(range, sill, nugget)`` for the native engine.

    Missing information is replaced by data-driven defaults: range = 0.6 x the largest inter-point distance
    (at least 1 m), sill = variance of the values (1 if the values are constant or a single point is given),
    nugget = 0. A dict may give any subset of ``"range"``, ``"sill"``, ``"nugget"``.
    """
    n_pts = len(values)
    default_range = max(max_distance * 0.6, 1.0)
    default_sill = float(np.var(values)) if n_pts > 1 else 1.0
    if variogram_params is None:
        return default_range, (default_sill if default_sill != 0 else 1.0), 0.0
    if isinstance(variogram_params, dict):
        return (
            float(variogram_params.get("range", default_range)),
            float(variogram_params.get("sill", default_sill)),
            float(variogram_params.get("nugget", 0.0)),
        )
    range_a, sill, nugget = variogram_params
    return range_a, sill, nugget


def polynomial_drift_columns(x_norm: np.ndarray, y_norm: np.ndarray, quadratic: bool) -> list[np.ndarray]:
    """
    Polynomial drift columns of normalised coordinates: ``[x, y]`` (``linear_xy``) or ``[x, y, x², y², xy]``
    (``quadratic_xy``, which contains the linear columns).
    """
    columns = [x_norm, y_norm]
    if quadratic:
        columns += [x_norm**2, y_norm**2, x_norm * y_norm]
    return columns


def max_pairwise_distance(xy: np.ndarray) -> float:
    """
    Largest distance between any two points of ``xy`` (shape (N, 2)); 100 m for fewer than two points.

    The maximum is attained at vertices of the convex hull, so only the hull vertices are compared (exact, and
    without an N x N matrix).
    """
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2:
        return 100.0
    try:
        vertices = xy[ConvexHull(xy).vertices]
    except Exception:  # fewer than three points or all points collinear
        vertices = xy
    return float(np.max(pdist(vertices))) if len(vertices) > 1 else 100.0


class KrigingSystem:
    """
    Augmented (dual) kriging system of one observation set with a fixed variogram and drift model.

    Parameters
    ----------
    x, y, z : np.ndarray, shape (N,)
        Observation coordinates [m] and values.
    drift_columns : np.ndarray, shape (N, p), optional
        Drift covariates at the observations in the column order that the prediction rows must follow. The
        constant (unbiasedness) column is always added; ``None`` or ``p = 0`` gives Ordinary Kriging.
    variogram_model : str
        Model name understood by :func:`pysole.variogram.evaluate_variogram_model`.
    variogram_params : tuple (range, sill, nugget) or dict, optional
        See :func:`resolve_variogram_params`.

    Attributes
    ----------
    range_a, sill, nugget : float
        The variogram parameters in use.
    reg_value : float
        Tikhonov term added to the diagonal of the point block.
    n_points, n_drift : int
        Number of observations and of drift rows including the constant (``1 + p``).
    dual_weights : np.ndarray
        Solution of ``A w = [z, 0]`` after :meth:`factorize` (first N entries: point weights).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        drift_columns: np.ndarray | None = None,
        variogram_model: str = "spherical",
        variogram_params: tuple[float, float, float] | dict[str, float] | None = None,
    ) -> None:
        self.xy = np.column_stack((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))
        self.z = np.asarray(z, dtype=np.float64)
        n_pts = len(self.z)
        self.n_points = n_pts
        self.variogram_model = variogram_model

        if drift_columns is None:
            drift_columns = np.zeros((n_pts, 0))
        drift_columns = np.asarray(drift_columns, dtype=np.float64)
        if drift_columns.ndim != 2 or drift_columns.shape[0] != n_pts:
            raise ValueError(f"drift_columns must have shape (N, p) with N = {n_pts}, got {drift_columns.shape}.")
        self.n_drift = 1 + drift_columns.shape[1]

        sample_dists = cdist(self.xy, self.xy)
        max_d = float(np.max(sample_dists)) if n_pts > 1 else 100.0
        self.range_a, self.sill, self.nugget = resolve_variogram_params(variogram_params, self.z, max_d)

        zero_lag = sample_dists == 0
        gamma = evaluate_variogram_model(sample_dists, self.variogram_model, self.range_a, self.sill, self.nugget)
        del sample_dists  # free one N x N matrix before the augmented matrix is allocated
        gamma[zero_lag] = 0.0  # in place: no further N x N temporary
        del zero_lag

        k_matrix = np.zeros((n_pts + self.n_drift, n_pts + self.n_drift))
        k_matrix[:n_pts, :n_pts] = gamma
        del gamma
        k_matrix[:n_pts, n_pts] = 1.0
        k_matrix[n_pts, :n_pts] = 1.0
        for j in range(drift_columns.shape[1]):
            k_matrix[:n_pts, n_pts + 1 + j] = drift_columns[:, j]
            k_matrix[n_pts + 1 + j, :n_pts] = drift_columns[:, j]

        # The variogram matrix has a zero diagonal, so the regularisation is a fixed fraction of the sill.
        self.reg_value = REGULARISATION_FRACTION * float(self.sill)
        k_matrix[np.arange(n_pts), np.arange(n_pts)] += self.reg_value

        self._matrix: np.ndarray | None = k_matrix
        self._lu: tuple[np.ndarray, np.ndarray] | None = None
        self._inverse: np.ndarray | None = None
        self.dual_weights: np.ndarray | None = None

    # ------------------------------------------------------------------ variogram / factorisation
    def variogram(self, h: np.ndarray) -> np.ndarray:
        """Semivariance at lag ``h`` (exactly zero at ``h == 0``)."""
        gamma = evaluate_variogram_model(h, self.variogram_model, self.range_a, self.sill, self.nugget)
        return np.where(h == 0, 0.0, gamma)

    def factorize(self) -> np.ndarray:
        """
        LU-factorises the system and computes the dual weights. Falls back to a least-squares solution if the
        matrix is singular (the explicit inverse is then the pseudo-inverse). Returns the dual weights.
        """
        if self._matrix is None:
            raise RuntimeError("The factorisation was released; build a new KrigingSystem.")
        rhs = np.zeros(self.n_points + self.n_drift, dtype=np.float64)
        rhs[: self.n_points] = self.z
        try:
            self._lu = lu_factor(self._matrix)
            self.dual_weights = lu_solve(self._lu, rhs)
        except Exception:
            self._lu = None
            self.dual_weights = np.linalg.lstsq(self._matrix, rhs, rcond=None)[0]
        return self.dual_weights

    def inverse(self) -> np.ndarray:
        """Explicit inverse of the augmented matrix (``lu_solve(I)``), cached; needed for the variance."""
        if self._inverse is None:
            if self._matrix is None:
                raise RuntimeError("The factorisation was released; build a new KrigingSystem.")
            if self.dual_weights is None:
                self.factorize()
            if self._lu is not None:
                self._inverse = lu_solve(self._lu, np.eye(self.n_points + self.n_drift, dtype=np.float64))
            else:
                self._inverse = np.linalg.pinv(self._matrix)
        return self._inverse

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solves ``A x = rhs`` with the factorisation (least squares if the matrix was singular)."""
        if self._matrix is None:
            raise RuntimeError("The factorisation was released; build a new KrigingSystem.")
        if self.dual_weights is None:
            self.factorize()
        if self._lu is not None:
            return lu_solve(self._lu, rhs)
        return np.linalg.lstsq(self._matrix, rhs, rcond=None)[0]

    def release(self) -> None:
        """Drops the assembled matrix and the LU factors (the dual weights and a computed inverse are kept)."""
        self._matrix = None
        self._lu = None

    # ------------------------------------------------------------------ prediction
    def predict_points(
        self,
        x: np.ndarray,
        y: np.ndarray,
        drift_rows: np.ndarray | None = None,
        return_variance: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Predicts at the points ``(x, y)``.

        Parameters
        ----------
        x, y : np.ndarray, shape (m,)
            Prediction coordinates [m].
        drift_rows : np.ndarray, shape (p, m), optional
            Drift covariates at the prediction points in the same order as the ``drift_columns`` of the
            constructor (the constant row is added here).
        return_variance : bool
            Also return the estimation variance (needs :meth:`inverse`, which is computed on demand but should
            be called once before the threads start).

        Returns
        -------
        estimate : np.ndarray, shape (m,)
        variance : np.ndarray, shape (m,) or None
            Non-negative kriging variance.
        """
        if self.dual_weights is None:
            self.factorize()
        n_pts = self.n_points
        m = len(x)
        gamma = self.variogram(cdist(self.xy, np.column_stack((x, y))))

        rhs_drift = np.zeros((self.n_drift, m), dtype=np.float64)
        rhs_drift[0, :] = 1.0
        if self.n_drift > 1:
            if drift_rows is None or np.shape(drift_rows) != (self.n_drift - 1, m):
                raise ValueError(f"drift_rows must have shape ({self.n_drift - 1}, {m}).")
            rhs_drift[1:, :] = drift_rows

        w_sample = self.dual_weights[:n_pts]
        w_drift = self.dual_weights[n_pts:]
        estimate = np.dot(w_sample, gamma) + np.dot(w_drift, rhs_drift)
        if not return_variance:
            return estimate, None

        rhs = np.zeros((n_pts + self.n_drift, m), dtype=np.float64)
        rhs[:n_pts, :] = gamma
        rhs[n_pts:, :] = rhs_drift
        solved = np.dot(self.inverse(), rhs)
        mu_drift = np.sum(solved[n_pts:, :] * rhs_drift, axis=0)
        variance = np.maximum(np.sum(solved[:n_pts, :] * gamma, axis=0) + mu_drift, 0.0)
        return estimate, variance


class _Candidate:
    """Low-rank update data of one drift candidate relative to the Ordinary base system."""

    __slots__ = ("columns", "n_columns", "update", "s_matrix", "s_inv", "dual", "ids", "s_inv_t")

    def __init__(self, columns, update, s_matrix, s_inv, dual, ids, s_inv_t):
        self.columns = columns
        self.n_columns = 0 if columns is None else columns.shape[1]
        self.update = update  # U = P_NN G, shape (N, p), or None for Ordinary
        self.s_matrix = s_matrix  # S = -G^T P_NN G, shape (p, p), or None
        self.s_inv = s_inv  # S^-1, shape (p, p), or None
        self.dual = dual  # dual weights of the candidate system (point part), shape (N,)
        self.ids = ids  # positions of the candidate's columns among the unique columns of all candidates
        self.s_inv_t = s_inv_t  # S^-1 G^T v, shape (p,), or None


class LeaveBlockOut:
    """
    Exact leave-block-out cross-validation residuals of Ordinary Kriging and of drift-augmented Kriging from ONE
    factorisation of the Ordinary system.

    Let ``P = A0^-1`` be the inverse of the Ordinary system (``KrigingSystem`` without drift columns) and
    ``v = P [z, 0]``. A candidate with the extra drift columns ``G`` has the inverse (Schur complement, a rank-``p``
    update)

        Q = P + U S^-1 U^T,   U = P_NN G,   S = -G^T P_NN G

    and the dual weights ``c = v + U S^-1 G^T v``. Removing a block ``B`` of observations (their covariance rows and
    drift rows; the unbiasedness rows stay) and predicting every ``b`` in ``B`` from the remaining observations gives
    the residuals ``e_B = z_B - zhat_B = (Q_BB)^-1 c_B`` (the leave-one-out identity of Dubrule generalised to
    blocks). So no per-fold or per-candidate factorisation is needed and every observation can be evaluated; the
    result equals a fresh fit per block up to rounding error (see ``tests/test_leave_block_out.py``).

    Parameters
    ----------
    system : KrigingSystem
        Ordinary base system (no drift columns). It is factorised here; with ``full_inverse`` it is released
        afterwards because only the inverse is needed.
    blocks : sequence of np.ndarray
        Index arrays of the removed observations (one block per evaluation).
    targets : sequence of np.ndarray
        For every block the indices (members of the block) whose residual is wanted.
    full_inverse : bool, default True
        Keep the full inverse of the base system in memory (about N^2 doubles; fastest). If False, the LU factors
        are kept instead and only the inverse columns of the observations that occur in the blocks are computed
        (large-N fallback).
    """

    #: Maximum number of float64 entries of the stacked (g, m, m) matrices that are processed at once.
    STACK_ENTRIES = 4_000_000

    def __init__(
        self,
        system: KrigingSystem,
        blocks: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        full_inverse: bool = True,
    ) -> None:
        if system.n_drift != 1:
            raise ValueError("LeaveBlockOut needs the Ordinary base system (no drift columns).")
        if len(blocks) != len(targets):
            raise ValueError("blocks and targets must have the same length.")
        self.system = system
        self.n = n = system.n_points
        self.blocks = [np.sort(np.asarray(b, dtype=np.intp)) for b in blocks]
        self.targets = [np.asarray(t, dtype=np.intp) for t in targets]
        self.target_positions = []
        for block, target in zip(self.blocks, self.targets):
            pos = np.searchsorted(block, target)
            if np.any(pos >= len(block)) or np.any(block[np.minimum(pos, len(block) - 1)] != target):
                raise ValueError("Every target index must be a member of its block.")
            self.target_positions.append(pos)

        self.groups: dict[int, list[int]] = {}
        for j, block in enumerate(self.blocks):
            self.groups.setdefault(len(block), []).append(j)

        # Same BLAS policy as the production engine: dense factorisations use at most the physical core count.
        with blas_thread_limit():
            system.factorize()
            self.v = system.dual_weights[:n]
            if full_inverse:
                self._inverse = system.inverse()[:n, :n]
                self._columns = None
                self._position = None
                system.release()
            else:
                union = np.unique(np.concatenate(self.blocks)) if self.blocks else np.zeros(0, dtype=np.intp)
                rhs = np.zeros((n + 1, len(union)))
                rhs[union, np.arange(len(union))] = 1.0
                self._inverse = None
                self._columns = system.solve(rhs)[:n]
                self._position = np.full(n, -1, dtype=np.intp)
                self._position[union] = np.arange(len(union))

    @property
    def n_blocks(self) -> int:
        """Number of blocks (the total of the increments reported to the ``progress`` callback of :meth:`residuals`)."""
        return len(self.blocks)

    # ------------------------------------------------------------------ helpers
    def _apply_inverse(self, columns: np.ndarray) -> np.ndarray:
        """P_NN @ columns for a drift column matrix of shape (N, p)."""
        if self._inverse is not None:
            return self._inverse @ columns
        rhs = np.zeros((self.n + 1, columns.shape[1]))
        rhs[: self.n] = columns
        return self.system.solve(rhs)[: self.n]

    def _gather(self, stacked: np.ndarray) -> np.ndarray:
        """Stack of the diagonal blocks P_BB for the index stack of shape (g, m)."""
        if self._inverse is not None:
            return self._inverse[stacked[:, :, None], stacked[:, None, :]]
        return self._columns[stacked[:, :, None], self._position[stacked][:, None, :]]

    def _unique_columns(self, candidates: Sequence[np.ndarray | None]) -> tuple[np.ndarray, list[list[int]]]:
        """Unique drift columns over all candidates (byte-identical columns are shared) and every candidate's ids."""
        positions: dict[bytes, int] = {}
        unique: list[np.ndarray] = []
        ids: list[list[int]] = []
        for columns in candidates:
            current: list[int] = []
            if columns is not None and np.shape(columns)[1] > 0:
                columns = np.asarray(columns, dtype=np.float64)
                if columns.shape[0] != self.n:
                    raise ValueError(f"Candidate columns need {self.n} rows, got {columns.shape[0]}.")
                for j in range(columns.shape[1]):
                    column = np.ascontiguousarray(columns[:, j])
                    key = column.tobytes()
                    if key not in positions:
                        positions[key] = len(unique)
                        unique.append(column)
                    current.append(positions[key])
            ids.append(current)
        return (np.column_stack(unique) if unique else np.zeros((self.n, 0))), ids

    def _prepare(self, columns: np.ndarray | None, update: np.ndarray | None, ids: list[int]) -> _Candidate | None:
        """
        Low-rank update data of one candidate (``update`` = ``P_NN @ columns``); ``None`` if its drift system is
        numerically singular.
        """
        if columns is None or np.shape(columns)[1] == 0:
            return _Candidate(None, None, None, None, self.v, [], None)
        columns = np.asarray(columns, dtype=np.float64)
        s_matrix = -(columns.T @ update)
        try:
            if not np.isfinite(s_matrix).all() or np.linalg.cond(s_matrix) > 1e14:
                return None
            s_inv = np.linalg.inv(s_matrix)
        except np.linalg.LinAlgError:
            return None
        s_inv_t = s_inv @ (columns.T @ self.v)
        dual = self.v + update @ s_inv_t
        if not np.isfinite(dual).all():
            return None
        return _Candidate(columns, update, s_matrix, s_inv, dual, ids, s_inv_t)

    def _direct_residuals(self, candidate: _Candidate, block: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Fallback: residuals of one block from a fresh fit on the remaining observations (NaN on failure)."""
        sys0 = self.system
        train = np.ones(self.n, dtype=bool)
        train[block] = False
        columns = candidate.columns
        try:
            fresh = KrigingSystem(
                sys0.xy[train, 0],
                sys0.xy[train, 1],
                sys0.z[train],
                None if columns is None else columns[train],
                sys0.variogram_model,
                (sys0.range_a, sys0.sill, sys0.nugget),
            )
            fresh.factorize()
            estimate, _ = fresh.predict_points(
                sys0.xy[target, 0], sys0.xy[target, 1], None if columns is None else columns[target].T
            )
            residual = sys0.z[target] - estimate
            return residual if np.isfinite(residual).all() else np.full(len(target), np.nan)
        except Exception:
            return np.full(len(target), np.nan)

    # ------------------------------------------------------------------ evaluation
    def residuals(
        self, candidates: Sequence[np.ndarray | None], progress: Callable[[int], None] | None = None
    ) -> np.ndarray:
        """
        Leave-block-out residuals ``z - zhat`` of every candidate.

        Parameters
        ----------
        candidates : sequence of np.ndarray or None
            Drift columns of every candidate at the observations, shape (N, p); ``None`` (or ``p = 0``) is Ordinary
            Kriging.
        progress : callable, optional
            Called as ``progress(k)`` with the number ``k`` of blocks just processed (once per processed stack and
            once per group that is skipped because its blocks leave too few observations); the increments sum to
            :attr:`n_blocks`. It only reports; it never influences the result.

        Returns
        -------
        residuals : np.ndarray, shape (n_candidates, N)
            Residual of every observation that is a target of some block; NaN for observations that are no target,
            blocks that leave fewer than ``p + 3`` observations, numerical failures, and the rows of candidates whose
            drift system is singular.
        """
        n = self.n
        out = np.full((len(candidates), n), np.nan)
        unique, ids = self._unique_columns(candidates)
        update_unique = self._apply_inverse(unique) if unique.shape[1] else unique
        prepared = [
            self._prepare(columns, update_unique[:, cid] if cid else None, cid)
            for columns, cid in zip(candidates, ids)
        ]
        # The stacked solves are many small matrices: a multi-threaded BLAS is orders of magnitude slower on them
        # (thread start-up per call), so this loop runs single-threaded.
        with blas_thread_limit(1):
            for m, members in self.groups.items():
                if n - m < 3:
                    if progress is not None:
                        progress(len(members))
                    continue
                per_stack = max(1, self.STACK_ENTRIES // (m * (m + 1 + unique.shape[1])))
                for start in range(0, len(members), per_stack):
                    chunk = members[start : start + per_stack]
                    stacked = np.stack([self.blocks[j] for j in chunk])
                    p_blocks = self._gather(stacked)
                    rows = np.concatenate([np.full(len(self.targets[j]), r) for r, j in enumerate(chunk)])
                    cols = np.concatenate([self.target_positions[j] for j in chunk])
                    points = np.concatenate([self.targets[j] for j in chunk])
                    shared = self._shared_solve(p_blocks, stacked, update_unique)
                    for k, cand in enumerate(prepared):
                        if cand is None or n - m < cand.n_columns + 3:
                            continue
                        e = self._woodbury(cand, shared, stacked, update_unique)
                        bad = np.flatnonzero(~np.isfinite(e).all(axis=1))
                        if len(bad):  # ill-conditioned block: dense solve of Q_BB, then a fresh fit
                            e[bad] = self._block_solve(cand, p_blocks[bad], stacked[bad], [chunk[r] for r in bad])
                        out[k, points] = e[rows, cols]
                    if progress is not None:
                        progress(len(chunk))
        return out

    def _shared_solve(self, p_blocks: np.ndarray, stacked: np.ndarray, update_unique: np.ndarray) -> np.ndarray:
        """``P_BB^-1 [v_B, U_B]`` for every block of the stack with all unique drift columns, shape (g, m, 1 + q)."""
        rhs = np.concatenate([self.v[stacked][..., None], update_unique[stacked]], axis=2)
        try:
            return np.linalg.solve(p_blocks, rhs)
        except np.linalg.LinAlgError:
            return np.full(rhs.shape, np.nan)

    @staticmethod
    def _woodbury(cand: _Candidate, shared: np.ndarray, stacked: np.ndarray, update_unique: np.ndarray) -> np.ndarray:
        """
        ``e_B = Q_BB^-1 c_B`` from the shared solve: with ``Q_BB = P_BB + U_B S^-1 U_B^T`` the Woodbury identity gives
        ``e = y - Y (S + U_B^T Y)^-1 U_B^T y`` for ``y = P_BB^-1 c_B`` and ``Y = P_BB^-1 U_B``. Rows that are not
        finite (singular ``P_BB`` or ``S + U_B^T Y``) are returned as NaN.
        """
        y_v = shared[..., 0]
        if not cand.ids:
            return y_v.copy()
        y_u = shared[..., 1:][..., cand.ids]
        u_b = update_unique[stacked][..., cand.ids]
        y_c = y_v + y_u @ cand.s_inv_t
        w = u_b.transpose(0, 2, 1) @ y_u
        try:
            z = np.linalg.solve(cand.s_matrix + w, (u_b.transpose(0, 2, 1) @ y_c[..., None]))[..., 0]
        except np.linalg.LinAlgError:
            return np.full(y_c.shape, np.nan)
        return y_c - (y_u @ z[..., None])[..., 0]

    def _block_solve(
        self, cand: _Candidate, p_blocks: np.ndarray, stacked: np.ndarray, chunk: list[int]
    ) -> np.ndarray:
        """``e_B = Q_BB^-1 c_B`` for a stack of equally sized blocks of one candidate, shape (g, m)."""
        c_blocks = cand.dual[stacked]
        if cand.update is None:
            q_blocks = p_blocks
        else:
            u_blocks = cand.update[stacked]
            q_blocks = p_blocks + (u_blocks @ cand.s_inv) @ u_blocks.transpose(0, 2, 1)
        try:
            e = np.linalg.solve(q_blocks, c_blocks[..., None])[..., 0]
            bad = np.flatnonzero(~np.isfinite(e).all(axis=1))
        except np.linalg.LinAlgError:
            e = np.full(c_blocks.shape, np.nan)
            bad = np.arange(len(chunk))
        for r in bad:
            try:
                e[r] = np.linalg.solve(q_blocks[r], c_blocks[r])
                if np.isfinite(e[r]).all():
                    continue
            except np.linalg.LinAlgError:
                pass
            j = chunk[r]
            e[r] = np.nan
            e[r, self.target_positions[j]] = self._direct_residuals(cand, self.blocks[j], self.targets[j])
        return e
