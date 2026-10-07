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


def _mean(v) -> float:
    return v["mean"] if isinstance(v, dict) and "mean" in v else float(v)


def filter_results(rows: list[dict], keys: list[str], maximize: list[bool]) -> list[dict]:
    """Return the non-dominated rows over the named objectives.

    Each row maps key -> {"mean": ...} (a summary entry) or a plain number.
    """
    points = np.array([[_mean(r[k]) for k in keys] for r in rows], dtype=np.float64)
    return [rows[i] for i in pareto_front(points, maximize)]


E2_OBJECTIVES = {
    "WR"      : True,    # maximize
    "J_A"     : True,    # maximize
    "SR_delta": False,   # minimize
    "M_uni"   : False,   # minimize (individual incentive to inflate)
}


def filter_e2_results(rows: list[dict], manipulation_key: str = "M_uni") -> list[dict]:
    """
    Convenience wrapper for experiment E2: non-dominated rows over
    (WR ↑, J_A ↑, SR_Δ ↓, manipulation gain ↓).

    `manipulation_key` is "M_uni" (unilateral gain, default) or "M_mean"
    (coalition gain).  J_B equals J_A in the homogeneous E2 population, so it
    is not a separate objective here.
    """
    objectives = dict(E2_OBJECTIVES)
    del objectives["M_uni"]
    objectives[manipulation_key] = False
    return filter_results(rows, list(objectives), list(objectives.values()))
