"""
sim/mechanisms/m7_rank.py
==========================
M7 — Rank-Cap: rank-normalised reports, a history penalty and a hard waiting cap.

Idea (linking decisions, Jackson & Sonnenschein 2007; Casella 2005): a report
is only meaningful *relative to the reporter's own distribution*.  Instead of
the raw report the mechanism scores the user's *quantile*

    q_i = (#past reports below + 0.5 * #ties + 0.5) / (m + 1),     m = past reports of i,

so q_i is approximately Uniform(0,1) for every truthful user, whatever their
value distribution.  Consequences:
  * any increasing distortion of the reports (e.g. multiplying by c) leaves the
    quantiles unchanged, and clipping at v_max creates ties that *lower* the
    quantile of the user's best values, so exaggeration cannot raise a score;
  * users with different value scales are treated alike, which equalises
    normalised benefit (J_B) but gives up welfare when users really differ.

Score:  s_i = ((1 - beta) * q_i + beta * report_i) / (1 + a_i)^lambda,
with a_i the cumulative allocation, lambda = cfg.lambda_, beta = cfg.rank_blend
(beta = 0 is pure rank; beta = 1 is the M4 Score rule).  Users whose wait has
reached cfg.wait_limit are served first (see _waitcap).  No payments.

Truthfulness is not claimed: lying by *selective* distortion (not a monotone
map) can still pay.  The incentive claim is empirical (experiments e9-e11).
"""

from __future__ import annotations

import bisect

import numpy as np

from sim.config import Config
from sim.environment import History
from sim.mechanisms._utils import batch_top_k_mask
from sim.mechanisms._waitcap import forced_mask, with_forced_priority
from sim.mechanisms.base import Mechanism


class RankCapMechanism(Mechanism):
    """M7: rank-normalised score + wait cap.  Stateful: use one instance per run."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self._past = [[] for _ in range(cfg.n)]        # sorted past reports of each user

    # ── quantile of a report within a user's own past reports ────────────────
    @staticmethod
    def _quantile(sorted_past, value):
        lo = bisect.bisect_left(sorted_past, value)
        hi = bisect.bisect_right(sorted_past, value)
        return (lo + 0.5 * (hi - lo) + 0.5) / (len(sorted_past) + 1)

    def _score(self, q, reports, cumulative):
        beta = self.cfg.rank_blend
        base = (1.0 - beta) * q + beta * reports / self.cfg.v_max
        return base / (1.0 + cumulative.astype(np.float64)) ** self.cfg.lambda_

    # ── single scenario ───────────────────────────────────────────────────────
    def allocate(self, reports: np.ndarray, history: History, tie_seed: int):
        n, k = self.cfg.n, self.cfg.k
        q = np.array([self._quantile(self._past[i], float(reports[i])) for i in range(n)])
        tiebreak = np.random.default_rng(tie_seed).random(n)[None, :]
        scores = self._score(q, reports, history.cumulative)[None, :]
        forced = forced_mask(history.consecutive_wait[None, :], self.cfg.wait_limit, k, tiebreak)
        x = batch_top_k_mask(with_forced_priority(scores, forced), tiebreak, k)
        for i in range(n):                              # the report becomes part of the user's past
            bisect.insort(self._past[i], float(reports[i]))
        return x[0], np.zeros(n, dtype=np.float64)

    # ── rollout interface: quantile functions are frozen at the current state ─
    def rollout_init(self, history: History, B: int) -> dict:
        return {"past": [np.asarray(p, dtype=np.float64) for p in self._past],
                "cum": np.tile(history.cumulative[None, :], (B, 1)).astype(np.int64),
                "wait": np.tile(history.consecutive_wait[None, :], (B, 1)).astype(np.int64)}

    def rollout_step(self, reports, state, tiebreak):
        k = self.cfg.k
        q = np.empty(reports.shape, dtype=np.float64)
        for i, past in enumerate(state["past"]):
            lo = np.searchsorted(past, reports[:, i], side="left")
            hi = np.searchsorted(past, reports[:, i], side="right")
            q[:, i] = (lo + 0.5 * (hi - lo) + 0.5) / (len(past) + 1)
        scores = self._score(q, reports, state["cum"])
        forced = forced_mask(state["wait"], self.cfg.wait_limit, k, tiebreak)
        x = batch_top_k_mask(with_forced_priority(scores, forced), tiebreak, k)
        state["cum"] = state["cum"] + x
        state["wait"] = np.where(x == 1, 0, state["wait"] + 1)
        return x, np.zeros(reports.shape, dtype=np.float64)
