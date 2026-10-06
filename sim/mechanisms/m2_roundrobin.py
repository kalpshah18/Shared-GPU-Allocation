"""
sim/mechanisms/m2_roundrobin.py
================================
M2 — Round-Robin Allocation

Maintains a seeded queue and serves the next k users in order each round.
Report-invariant by construction.  Guarantees each user is served at least
once every ⌈n/k⌉ rounds.

Implementation notes
--------------------
* The initial queue permutation is fixed by the tie_seed passed at round 0.
  For subsequent rounds the queue simply advances (no re-randomisation).
* Round-robin is stateful: the queue position carries over across rounds.
  The queue is stored on the instance, so a fresh RoundRobinMechanism instance
  must be used per simulation run.
"""

from __future__ import annotations

import math

import numpy as np

from sim.config import Config
from sim.environment import History
from sim.mechanisms.base import Mechanism


class RoundRobinMechanism(Mechanism):
    """M2: round-robin allocation, report-invariant."""

    def __init__(self, cfg: Config, init_seed: int) -> None:
        """
        Parameters
        ----------
        init_seed : int
            Seed used to generate the initial queue permutation.
            Pass the tie_seed of round 0 here.
        """
        super().__init__(cfg)
        rng = np.random.default_rng(init_seed)
        self._queue: np.ndarray = rng.permutation(cfg.n)
        self._pos: int = 0

    def allocate(
        self,
        reports: np.ndarray,
        history: History,
        tie_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Serve the next k users in the circular queue
        n, k = self.cfg.n, self.cfg.k
        indices = [(self._pos + i) % n for i in range(k)]
        winners = self._queue[indices]
        self._pos = (self._pos + k) % n

        x = np.zeros(n, dtype=np.int64)
        x[winners] = 1

        p = np.zeros(n, dtype=np.float64)
        return x, p
