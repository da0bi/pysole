"""Exact leave-block-out residuals (``LeaveBlockOut``) versus explicit per-block refits."""
import unittest
from unittest import mock

import numpy as np

from pysole.kriging_core import KrigingSystem, LeaveBlockOut, max_pairwise_distance, polynomial_drift_columns

PARAMS = (350.0, 10.0, 0.4)


def _fixture(n=70, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1000, n)
    y = rng.uniform(0, 800, n)
    z = 40 + 0.03 * x - 0.02 * y + 4 * np.sin(x / 150) + rng.normal(0, 1.5, n)
    xn, yn = (x - 500) / 1000, (y - 400) / 800
    lin = np.column_stack(polynomial_drift_columns(xn, yn, False))
    quad = np.column_stack(polynomial_drift_columns(xn, yn, True))
    extra = np.column_stack((np.cos(x / 200), np.sin(y / 170)))
    candidates = [None, lin, quad, extra, np.hstack((extra, lin))]
    return x, y, z, candidates


def _reference(x, y, z, columns, block, targets):
    """Fresh fit without the block, prediction at the targets (completely independent of LeaveBlockOut)."""
    train = np.ones(len(z), bool)
    train[block] = False
    system = KrigingSystem(
        x[train], y[train], z[train], None if columns is None else columns[train], "spherical", PARAMS
    )
    est, _ = system.predict_points(x[targets], y[targets], None if columns is None else columns[targets].T)
    return z[targets] - est


def _buffer_blocks(x, y, radius, indices):
    blocks, targets = [], []
    for i in indices:
        blocks.append(np.flatnonzero(np.hypot(x - x[i], y - y[i]) <= radius))
        targets.append(np.array([i]))
    return blocks, targets


class TestLeaveBlockOut(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y, cls.z, cls.cands = _fixture()
        cls.n = len(cls.z)

    def _engine(self, blocks, targets, full=True):
        return LeaveBlockOut(KrigingSystem(self.x, self.y, self.z, None, "spherical", PARAMS), blocks, targets, full)

    def _check(self, blocks, targets, full=True, tol=1e-9):
        res = self._engine(blocks, targets, full).residuals(self.cands)
        for k, columns in enumerate(self.cands):
            for block, target in zip(blocks, targets):
                ref = _reference(self.x, self.y, self.z, columns, block, target)
                np.testing.assert_allclose(res[k, target], ref, rtol=tol, atol=tol, err_msg=f"candidate {k}")
        return res

    def test_leave_one_out_closed_form(self):
        idx = np.arange(0, self.n, 3)
        self._check([np.array([i]) for i in idx], [np.array([i]) for i in idx])

    def test_buffer_blocks(self):
        blocks, targets = _buffer_blocks(self.x, self.y, 120.0, range(0, self.n, 4))
        self.assertGreater(max(len(b) for b in blocks), 2)
        self._check(blocks, targets)

    def test_profile_blocks_all_members(self):
        labels = np.arange(self.n) % 7
        blocks = [np.flatnonzero(labels == k) for k in range(7)]
        self._check(blocks, blocks)

    def test_large_n_fallback_matches_full_inverse(self):
        blocks, targets = _buffer_blocks(self.x, self.y, 120.0, range(0, self.n, 5))
        full = self._engine(blocks, targets, True).residuals(self.cands)
        partial = self._engine(blocks, targets, False).residuals(self.cands)
        np.testing.assert_allclose(partial, full, rtol=1e-8, atol=1e-8, equal_nan=True)

    def test_blocks_leaving_too_few_points_are_nan(self):
        big = np.arange(self.n - 4)  # leaves 4 points: fine for Ordinary (needs 3), too few for 5 drift columns
        res = self._engine([big], [np.array([0])]).residuals(self.cands)
        self.assertTrue(np.isfinite(res[0, 0]))
        self.assertTrue(np.isnan(res[2, 0]))
        tiny = np.arange(self.n - 2)
        self.assertTrue(np.all(np.isnan(self._engine([tiny], [np.array([0])]).residuals(self.cands)[:, 0])))

    def test_singular_candidate_row_is_nan_and_others_unaffected(self):
        dup = np.column_stack((self.cands[1], self.cands[1][:, 0]))  # exactly collinear column
        blocks, targets = _buffer_blocks(self.x, self.y, 80.0, range(0, self.n, 6))
        res = self._engine(blocks, targets).residuals([None, dup, self.cands[1]])
        self.assertTrue(np.all(np.isnan(res[1])))
        for t in targets:
            self.assertTrue(np.isfinite(res[0, t]).all() and np.isfinite(res[2, t]).all())

    def test_direct_fallback_equals_identity(self):
        blocks, targets = _buffer_blocks(self.x, self.y, 120.0, range(0, self.n, 9))
        engine = self._engine(blocks, targets)
        expected = engine.residuals(self.cands)
        real_solve = np.linalg.solve

        def failing_solve(a, b):
            raise np.linalg.LinAlgError("forced")

        with mock.patch("pysole.kriging_core.np.linalg.solve", failing_solve):
            fallback = self._engine(blocks, targets).residuals(self.cands)
        self.assertIs(np.linalg.solve, real_solve)
        np.testing.assert_allclose(fallback, expected, rtol=1e-8, atol=1e-8)

    def test_targets_must_be_block_members_and_base_must_be_ordinary(self):
        with self.assertRaises(ValueError):
            self._engine([np.array([1, 2])], [np.array([3])])
        with self.assertRaises(ValueError):
            LeaveBlockOut(
                KrigingSystem(self.x, self.y, self.z, self.cands[1], "spherical", PARAMS), [np.array([1])], [np.array([1])]
            )

    def test_large_blocks_are_processed_in_stacks(self):
        blocks, targets = _buffer_blocks(self.x, self.y, 200.0, range(0, self.n, 2))
        engine = self._engine(blocks, targets)
        expected = engine.residuals(self.cands)
        with mock.patch.object(LeaveBlockOut, "STACK_ENTRIES", 1):
            chunked = self._engine(blocks, targets).residuals(self.cands)
        np.testing.assert_allclose(chunked, expected, rtol=1e-10, atol=1e-10, equal_nan=True)


class TestMaxPairwiseDistance(unittest.TestCase):
    def test_equals_brute_force(self):
        rng = np.random.default_rng(5)
        xy = rng.uniform(0, 500, (200, 2))
        brute = float(np.max(np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])))
        self.assertAlmostEqual(max_pairwise_distance(xy), brute, places=9)

    def test_degenerate_inputs(self):
        self.assertEqual(max_pairwise_distance(np.array([[1.0, 2.0]])), 100.0)
        self.assertAlmostEqual(max_pairwise_distance(np.array([[0.0, 0.0], [3.0, 4.0]])), 5.0)
        collinear = np.column_stack((np.arange(10.0), np.arange(10.0)))
        self.assertAlmostEqual(max_pairwise_distance(collinear), 9 * np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
