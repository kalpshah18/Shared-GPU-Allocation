"""
sim/mechanisms/m4_score.py
===========================
M4 — History-Penalised Score

Score for user i in round t:

    s_{i,t} = v̂_{i,t} / (1 + a_i(t))^λ

where a_i(t) is user i's cumulative allocation count and λ ≥ 0 is the
history-penalty exponent (Config.lambda_).

* At λ=0, scores equal reports and M4 is identical to M3.
* As λ→∞, the rule approaches least-served-first with reports used only
  to break ties.
* λ is swept in experiment E2 to trace the fairness–efficiency frontier.
"""

from __future__ import annotations

import numpy as np

from sim.config import Config
from sim.environment import History
from sim.mechanisms.base import Mechanism
from sim.mechanisms._utils import seeded_top_k


class ScoreMechanism(Mechanism):
    """M4: history-penalised score allocation, no payments."""

    def allocate(
        self,
        reports: np.ndarray,
        history: History,
        tie_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        lam = self.cfg.lambda_
        # denominator: (1 + a_i(t))^λ — safe against division by zero since
        # denominator ≥ 1 for all λ ≥ 0.
        denominator = (1.0 + history.cumulative.astype(np.float64)) ** lam
        scores = reports / denominator

        winners = seeded_top_k(scores, self.cfg.k, tie_seed)

        x = np.zeros(self.cfg.n, dtype=np.int64)
        x[winners] = 1

        p = np.zeros(self.cfg.n, dtype=np.float64)
        return x, p
