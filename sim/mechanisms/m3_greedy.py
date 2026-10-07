"""
sim/mechanisms/m3_greedy.py
============================
M3 — Greedy Reported Value

Allocates to the k users with the highest reported values.
Ties are broken uniformly using the per-round tie_seed.

When all reports equal true values, M3 attains the truthful welfare oracle W*.
Increasing a report can only improve a user's one-round allocation probability,
making M3 vulnerable to bounded over-reporting.
"""

from __future__ import annotations

import numpy as np

from sim.config import Config
from sim.environment import History
from sim.mechanisms.base import Mechanism
from sim.mechanisms._utils import batch_top_k_mask, seeded_top_k


class GreedyMechanism(Mechanism):
    """M3: allocate to k highest reports, no payments."""

    def allocate(
        self,
        reports: np.ndarray,
        history: History,
        tie_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        winners = seeded_top_k(reports, self.cfg.k, tie_seed)

        x = np.zeros(self.cfg.n, dtype=np.int64)
        x[winners] = 1

        p = np.zeros(self.cfg.n, dtype=np.float64)
        return x, p

    def allocate_batch(self, reports, cumulative, tiebreak):
        x = batch_top_k_mask(reports, tiebreak, self.cfg.k)
        return x, np.zeros(reports.shape, dtype=np.float64)
