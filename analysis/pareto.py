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


def _value(r: dict, key: str) -> float:
    v = r[key]
    if isinstance(v, dict):
        v = v.get("mean")
    return float("nan") if v is None else float(v)


def pareto_flags(rows: list, gain_key: str = "M_uni") -> list:
    """
    Boolean list marking the non-dominated rows on the proposal's objectives
    (WR max, J_A max, SR_delta min, manipulation gain min).  Each row maps
    metric names to floats or to summary dicts with a 'mean'.
    """
    keys     = ["WR", "J_A", "SR_delta", gain_key]
    maximize = [True, True, False, False]
    points = np.array([[_value(r, k) for k in keys] for r in rows], dtype=np.float64)
    if np.isnan(points).any():
        raise ValueError("pareto_flags: NaN objective value; summarise before filtering.")
    front = set(pareto_front(points, maximize).tolist())
    return [i in front for i in range(len(rows))]


def filter_e2_results(rows: list, gain_key: str = "M_uni") -> list:
    """Return only the non-dominated rows (see ``pareto_flags``)."""
    flags = pareto_flags(rows, gain_key)
    return [r for r, f in zip(rows, flags) if f]
