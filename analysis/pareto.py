"""
analysis/pareto.py
==================
Pareto-dominance filter — Owner: Kalp Shah

Used in experiment E2 to identify non-dominated (λ, mechanism) settings
on the (WR, J_A, SR_Δ, M) objective space.

Dominance definition (from proposal §3)
-----------------------------------------
Setting A Pareto-dominates setting B if:
  - A has no lower welfare (WR) or fairness (J_A, J_B)
  - A has no higher starvation (SR_Δ) or manipulation gain (M)
  - A is strictly better on at least one measure.

The filter below is general: the caller specifies which objectives are
"higher is better" and which are "lower is better".
"""

from __future__ import annotations

import numpy as np


def pareto_front(
    points: np.ndarray,
    maximize: list[bool],
) -> np.ndarray:
    """
    Return the indices of non-dominated points.

    Parameters
    ----------
    points   : ndarray, shape (m, d)
               m points in d-dimensional objective space.
    maximize : list of booleans, length d.
               maximize[j] = True  → dimension j is "higher is better"
               maximize[j] = False → dimension j is "lower is better"

    Returns
    -------
    front_indices : ndarray of int, shape (f,)
        Indices of the non-dominated points (the Pareto front).
    """
    m, d = points.shape
    if len(maximize) != d:
        raise ValueError("len(maximize) must equal points.shape[1].")

    # Normalise: convert all objectives to "higher is better"
    normed = points.copy().astype(np.float64)
    for j, do_max in enumerate(maximize):
        if not do_max:
            normed[:, j] = -normed[:, j]

    is_dominated = np.zeros(m, dtype=bool)
    for i in range(m):
        if is_dominated[i]:
            continue
        for j in range(m):
            if i == j or is_dominated[j]:
                continue
            # Does j dominate i?
            if np.all(normed[j] >= normed[i]) and np.any(normed[j] > normed[i]):
                is_dominated[i] = True
                break

    return np.where(~is_dominated)[0]


def filter_e2_results(rows: list[dict]) -> list[dict]:
    """
    Convenience wrapper for experiment E2.

    Each row must contain keys: 'WR', 'J_A', 'SR_delta', 'M_mean'.
    Returns only the non-dominated rows.

    Objective directions:
        WR       → maximize
        J_A      → maximize
        SR_delta → minimize
        M_mean   → minimize
    """
    keys     = ["WR", "J_A", "SR_delta", "M_mean"]
    maximize = [True, True, False, False]

    points = np.array([
        [r[k]["mean"] if isinstance(r[k], dict) and "mean" in r[k] else float(r[k]) for k in keys]
        for r in rows
    ], dtype=np.float64)
    front  = pareto_front(points, maximize)
    return [rows[i] for i in front]
