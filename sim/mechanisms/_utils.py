"""
sim/mechanisms/_utils.py
=========================
Shared utilities for mechanism implementations.

Tie-breaking
------------
Winners are the k largest scores.  Exact ties are broken uniformly at random
with a *secondary sort key*, never by perturbing the scores: an additive
jitter would distort orderings whenever scores are tiny (for example M4 with a
large lambda and large cumulative allocations).
"""

from __future__ import annotations

import numpy as np


def seeded_top_k(scores: np.ndarray, k: int, tie_seed: int) -> np.ndarray:
    """
    Indices of the k largest scores, ties broken uniformly using `tie_seed`.

    The primary key is the score (exact comparison); the secondary key is an
    i.i.d. uniform draw from ``default_rng(tie_seed)``.  The result is ordered
    from highest to lowest.
    """
    tiebreak = np.random.default_rng(tie_seed).random(len(scores))
    order = np.lexsort((tiebreak, scores))        # ascending by score, then tiebreak
    return order[::-1][:k]


def batch_top_k_mask(scores: np.ndarray, tiebreak: np.ndarray, k: int) -> np.ndarray:
    """
    Vectorised ``seeded_top_k``: for every row of `scores` (shape (B, n)) return
    a 0/1 int64 matrix with exactly k ones marking the winners.  `tiebreak` has
    the same shape and supplies the secondary key.
    """
    order = np.lexsort((tiebreak, scores), axis=-1)     # ascending along last axis
    winners = order[..., -k:]
    x = np.zeros(scores.shape, dtype=np.int64)
    np.put_along_axis(x, winners, 1, axis=-1)
    return x
