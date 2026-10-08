"""
sim/mechanisms/m6_karma.py
===========================
M6 — Karma-Cap: a closed-economy karma auction with a hard waiting bound.

Idea (proposal extension, motivated by Gorokh-Banerjee-Iyer, Elokda et al. and
Karma, OSDI'23): turn the one-shot Vickrey rule into a *repeated* mechanism
whose payments are an internal, non-tradable currency ("karma") instead of
money, and make karma scarce.  Exaggerating a report then has a price that is
paid in future bidding power, which is exactly the deterrent a history penalty
lacks.  A deficit-style wait cap supplies the hard service guarantee that none
of the incentive results provide.

Per-round rule (n users, k GPUs, cap W):
  1. Bid.        b_i = min(report_i, balance_i)          (a bid never exceeds karma)
  2. Cap.        users with wait_i >= W are served first, oldest first
                 (at most k per round)
  3. Auction.    the remaining r slots go to the r highest bids among the other
                 users (random tie-break); every auction winner pays the
                 (r+1)-st highest bid among those users  (Vickrey price)
                 and every forced winner pays min(price of a slot, balance)
                 where that price is the (k+1)-st highest bid overall
  4. Dividend.   all payments are redistributed equally to all n users
                 (karma is conserved except for the ceiling below)
  5. Ceiling.    balances are clipped to [0, karma_cap_mult * karma_init]

Karma is *not money*: the payment vector returned to the simulator is zero, so
welfare and utility are measured in value only.  Truthful reporting is not a
dominant strategy; the incentive claim is empirical (see experiments e9-e11).
"""

from __future__ import annotations

import numpy as np

from sim.config import Config
from sim.environment import History
from sim.mechanisms._utils import batch_top_k_mask
from sim.mechanisms._waitcap import forced_mask, with_forced_priority
from sim.mechanisms.base import Mechanism


class KarmaCapMechanism(Mechanism):
    """M6: karma auction + hard waiting cap.  Stateful: use one instance per run."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.balance = np.full(cfg.n, cfg.karma_init, dtype=np.float64)
        self.balance_ceiling = cfg.karma_cap_mult * cfg.karma_init

    # ── core step, vectorised over B scenarios ────────────────────────────────
    def _step(self, balance, wait, reports, tiebreak):
        """Return (x, new_balance) for balances/waits/reports/tiebreak of shape (B, n)."""
        cfg = self.cfg
        n, k = cfg.n, cfg.k
        bids = np.minimum(reports, balance)

        forced = forced_mask(wait, cfg.wait_limit, k, tiebreak)
        n_forced = forced.sum(axis=-1)
        r = k - n_forced                                           # auction slots per scenario
        x = batch_top_k_mask(with_forced_priority(bids, forced), tiebreak, k)
        auction_win = x * (1 - forced)

        # Vickrey price among non-forced users: (r+1)-st highest bid (index r, descending)
        bids_e = np.where(forced == 1, -1.0, bids)
        desc = -np.sort(-bids_e, axis=-1)
        idx = np.minimum(r, n - 1)[:, None]
        price_auction = np.take_along_axis(desc, idx, axis=-1)[:, 0]
        price_auction = np.maximum(price_auction, 0.0)
        # price of a slot overall: (k+1)-st highest bid among everyone
        desc_all = -np.sort(-bids, axis=-1)
        price_slot = desc_all[:, k]

        pay = (auction_win * np.minimum(price_auction[:, None], balance)
               + forced * np.minimum(price_slot[:, None], balance))
        dividend = pay.sum(axis=-1, keepdims=True) / n
        new_balance = np.clip(balance - pay + dividend, 0.0, self.balance_ceiling)
        return x, new_balance

    # ── single scenario ───────────────────────────────────────────────────────
    def allocate(self, reports: np.ndarray, history: History, tie_seed: int):
        tiebreak = np.random.default_rng(tie_seed).random(self.cfg.n)[None, :]
        x, new_balance = self._step(self.balance[None, :], history.consecutive_wait[None, :],
                                    reports[None, :], tiebreak)
        self.balance = new_balance[0]
        return x[0], np.zeros(self.cfg.n, dtype=np.float64)

    # ── rollout interface ─────────────────────────────────────────────────────
    def rollout_init(self, history: History, B: int) -> dict:
        return {"bal": np.tile(self.balance[None, :], (B, 1)),
                "wait": np.tile(history.consecutive_wait[None, :], (B, 1)).astype(np.int64)}

    def rollout_step(self, reports, state, tiebreak):
        x, state["bal"] = self._step(state["bal"], state["wait"], reports, tiebreak)
        state["wait"] = np.where(x == 1, 0, state["wait"] + 1)
        return x, np.zeros(reports.shape, dtype=np.float64)
