"""
sim/mechanisms/m1_random.py
============================
M1 — Random Allocation

Chooses k users uniformly at random without replacement each round.
Ignores both reports and history.  Report-invariant by construction.
"""

from __future__ import annotations

import numpy as np

from sim.config import Config
from sim.environment import History
from sim.mechanisms.base import Mechanism


class RandomMechanism(Mechanism):
    """M1: uniform random allocation, report-invariant."""

    def allocate(
        self,
        reports: np.ndarray,
        history: History,
        tie_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(tie_seed)
        winners = rng.choice(self.cfg.n, size=self.cfg.k, replace=False)

        x = np.zeros(self.cfg.n, dtype=np.int64)
        x[winners] = 1

        p = np.zeros(self.cfg.n, dtype=np.float64)
        return x, p
