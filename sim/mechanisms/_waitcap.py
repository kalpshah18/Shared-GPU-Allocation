"""
sim/mechanisms/_waitcap.py
===========================
The hard waiting guarantee shared by the proposed mechanisms (M6, M7).

Rule: a user whose consecutive wait has reached the cap W is *forced* into the
allocation, oldest first (ties broken at random).  At most k users are forced
in one round; any surplus keeps waiting and is first in line next round, so the
guarantee degrades gracefully instead of being violated silently.  This is the
deficit-style override used by bounded-lag schedulers (Deficit Round Robin,
stride scheduling, Gavel's priority), added on top of a value-aware rule.

All functions are vectorised over B independent scenarios (rows).
"""

from __future__ import annotations

import numpy as np

# Priority given to forced users so they sort ahead of every ordinary score.
FORCED_PRIORITY = 1e18


def forced_mask(wait: np.ndarray, cap: int, k: int, tiebreak: np.ndarray) -> np.ndarray:
    """
    0/1 int64 matrix (B, n): the (at most k) users with ``wait >= cap`` that are
    served first this round, oldest wait first.
    """
    eligible = wait >= cap
    key = np.where(eligible, wait, -1)
    order = np.lexsort((tiebreak, key), axis=-1)          # ascending: oldest waits last
    top = order[..., -k:]
    mask = np.zeros(wait.shape, dtype=np.int64)
    np.put_along_axis(mask, top, 1, axis=-1)
    return mask * eligible


def with_forced_priority(scores: np.ndarray, forced: np.ndarray) -> np.ndarray:
    """Scores with forced users lifted above every ordinary score."""
    return np.where(forced == 1, FORCED_PRIORITY, scores)
