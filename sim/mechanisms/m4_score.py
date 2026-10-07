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
from sim.mechanisms._utils import batch_top_k_mask, seeded_top_k


class ScoreMechanism(Mechanism):
    """M4: history-penalised score allocation, no payments."""

    def scores(self, reports: np.ndarray, cumulative: np.ndarray) -> np.ndarray:
        """s = v̂ / (1 + a)^λ; the denominator is >= 1, so no division by zero."""
        return reports / (1.0 + cumulative.astype(np.float64)) ** self.cfg.lambda_

    def allocate(
        self,
        reports: np.ndarray,
        history: History,
        tie_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = self.scores(reports, history.cumulative)

        winners = seeded_top_k(scores, self.cfg.k, tie_seed)

        x = np.zeros(self.cfg.n, dtype=np.int64)
        x[winners] = 1

        p = np.zeros(self.cfg.n, dtype=np.float64)
        return x, p

    def allocate_batch(self, reports, cumulative, tiebreak):
        x = batch_top_k_mask(self.scores(reports, cumulative), tiebreak, self.cfg.k)
        return x, np.zeros(reports.shape, dtype=np.float64)
