"""
sim/mechanisms/_utils.py
=========================
Shared utilities for mechanism implementations.
"""

from __future__ import annotations

import numpy as np


def seeded_top_k(scores: np.ndarray, k: int, tie_seed: int) -> np.ndarray:
    """
    Return the indices of the k largest scores, with ties broken uniformly
    using a seeded RNG.

    Algorithm
    ---------
    1. Add a tiny uniform jitter (from `tie_seed`) to every score.
    2. Take argpartition for efficiency, then sort the top-k subset.

    The jitter is on the order of 1e-12 relative to scores in [0, 1], so
    it never changes a strict ordering — it only resolves exact ties.

    Parameters
    ----------
    scores : ndarray, shape (n,)
    k : int
    tie_seed : int  — per-round seed from the pre-generated tie_seeds array

    Returns
    -------
    winners : ndarray, shape (k,)  — indices of the k selected users
    """
    rng    = np.random.default_rng(tie_seed)
    jitter = rng.uniform(0.0, 1e-12, size=len(scores))
    jittered = scores + jitter

    # argpartition is O(n); the top-k are in arbitrary order, so we sort them
    top_k_unsorted = np.argpartition(jittered, -k)[-k:]
    # Sort by descending score so the "winner" ordering is deterministic
    top_k = top_k_unsorted[np.argsort(jittered[top_k_unsorted])[::-1]]
    return top_k
