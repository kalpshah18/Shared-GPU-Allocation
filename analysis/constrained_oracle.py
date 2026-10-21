"""Offline maximum welfare subject to capacity and a finite-episode wait bound.

For every user and every length B+1 window, require at least one service.
With zero initial wait this is equivalent to q_i(t) <= B for t=0,...,T,
including the terminal state. The MILP sees the entire true-value trace.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from sim.metrics import wait_matrix


def terminal_max_wait(X):
    Q = wait_matrix(X)
    terminal = np.where(X[:, -1] == 1, 0, Q[:, -1] + 1)
    return int(max(Q.max(), terminal.max()))


@dataclass
class OracleResult:
    allocation: np.ndarray
    welfare: float
    upper_bound: float
    optimal: bool
    mip_gap: float


def constrained_welfare(valuations, k, max_wait, time_limit=30.0):
    v = np.asarray(valuations, dtype=float)
    if v.ndim != 2 or not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError("Values must be a finite non-negative users-by-rounds matrix.")
    n, T = v.shape
    if n < 2 or T < 1 or not isinstance(k, (int, np.integer)) or not 1 <= k < n:
        raise ValueError("Require n >= 2, T >= 1 and integer 1 <= k < n.")
    if not isinstance(max_wait, (int, np.integer)) or max_wait < 0:
        raise ValueError("max_wait must be a non-negative integer.")
    if time_limit <= 0:
        raise ValueError("time_limit must be positive.")
    L = max_wait + 1
    if T >= L and k * L < n:
        raise ValueError("Wait bound is infeasible: k*(B+1) < n.")
    rows, cols, data, lower, upper = [], [], [], [], []
    for t in range(T):
        row = len(lower)
        rows.extend([row] * n)
        cols.extend((np.arange(n) * T + t).tolist())
        data.extend([1.0] * n)
        lower.append(float(k))
        upper.append(float(k))
    for user in range(n):
        for start in range(T - L + 1):
            row = len(lower)
            rows.extend([row] * L)
            cols.extend((user * T + np.arange(start, start + L)).tolist())
            data.extend([1.0] * L)
            lower.append(1.0)
            upper.append(np.inf)
    A = coo_matrix((data, (rows, cols)), shape=(len(lower), n * T)).tocsc()
    result = milp(-v.ravel(), integrality=np.ones(n * T), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(A, lower, upper),
                  options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if result.x is None:
        raise RuntimeError(f"Oracle did not find a feasible schedule: {result.message}")
    X = np.rint(result.x).astype(np.int64).reshape(n, T)
    if np.max(np.abs(result.x - X.ravel())) > 1e-5 or np.any(X.sum(axis=0) != k) or terminal_max_wait(X) > max_wait:
        raise RuntimeError("MILP solution failed independent capacity/wait validation.")
    welfare = float(np.sum(v * X))
    bound = max(welfare, float(-result.mip_dual_bound))
    return OracleResult(X, welfare, bound, result.status == 0, float(result.mip_gap))
