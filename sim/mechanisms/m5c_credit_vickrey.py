"""
sim/mechanisms/m5c_credit_vickrey.py
====================================
M5c — Credit-Vickrey: the k-unit Vickrey auction paid in a non-monetary
credit budget instead of money.

Motivation (reviewer feedback): shared clusters rarely charge real money, so
M5's quasi-linear payments are not a deployable incentive.  The simplest
non-monetary replacement is a per-user *credit quota*: every user receives a
fixed allowance each round, bids are capped by the current balance, and the
Vickrey price is paid in credits.  Credits have no value outside the
mechanism, so a user's utility is the value of the GPUs they receive only.

Per-round rule (n users, k GPUs):
  1. Income.   every balance grows by  income = credit_rate * (k / n) * v_max
               and is clipped at the ceiling  credit_horizon * income
               (unused quota expires, as in "use it or lose it" cluster quotas)
  2. Bid.      b_i = min(report_i, balance_i)
  3. Auction.  the k highest bids win (seeded tie-break); each winner pays the
               (k+1)-st highest bid in credits.  Credits leave the economy.
  4. Balances start full (at the ceiling).

``credit_rate`` is the allowance relative to a fair share of GPU-rounds at the
maximum possible price: at rate 1 a user can afford its k/n share even if every
slot cost v_max; at lower rates the budget binds.

Incentives.  Because payments are not utility, truthful bidding is *not* a
dominant strategy: a user who skips low-value rounds saves credits for later
high-value rounds.  Per-round DSIC of M5 therefore does not carry over; the
incentive question is empirical (experiments/e20_credit_budget.py).  The
payment vector returned to the simulator is zero, so welfare and utility are
measured in value only (as for M6 Karma-Cap).
"""

from __future__ import annotations

import numpy as np

from sim.config import Config
from sim.environment import History
from sim.mechanisms._utils import batch_top_k_mask
from sim.mechanisms.base import Mechanism


class CreditVickreyMechanism(Mechanism):
    """M5c: k-unit Vickrey auction paid from a renewable credit quota.  Stateful."""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.income = cfg.credit_rate * (cfg.k / cfg.n) * cfg.v_max
        self.ceiling = cfg.credit_horizon * self.income
        self.balance = np.full(cfg.n, self.ceiling, dtype=np.float64)
        self.spent = np.zeros(cfg.n, dtype=np.float64)   # cumulative credits paid (diagnostic)

    # ── core step, vectorised over B scenarios ────────────────────────────────
    def _step(self, balance, reports, tiebreak):
        """Return (x, payment_in_credits, new_balance), all shape (B, n)."""
        k = self.cfg.k
        balance = np.minimum(balance + self.income, self.ceiling)
        bids = np.minimum(reports, balance)
        x = batch_top_k_mask(bids, tiebreak, k)
        price = -np.sort(-bids, axis=-1)[:, k]                    # (k+1)-st highest bid
        pay = x * np.minimum(price[:, None], balance)              # never exceeds the balance
        return x, pay, balance - pay

    # ── single scenario ───────────────────────────────────────────────────────
    def allocate(self, reports: np.ndarray, history: History, tie_seed: int):
        tiebreak = np.random.default_rng(tie_seed).random(self.cfg.n)[None, :]
        x, pay, new_balance = self._step(self.balance[None, :], np.asarray(reports)[None, :], tiebreak)
        self.balance = new_balance[0]
        self.spent += pay[0]
        return x[0], np.zeros(self.cfg.n, dtype=np.float64)

    # ── rollout interface ─────────────────────────────────────────────────────
    def rollout_init(self, history: History, B: int) -> dict:
        return {"bal": np.tile(self.balance[None, :], (B, 1)),
                "cum": np.tile(history.cumulative[None, :], (B, 1)).astype(np.int64),
                "wait": np.tile(history.consecutive_wait[None, :], (B, 1)).astype(np.int64)}

    def rollout_step(self, reports, state, tiebreak):
        x, _, state["bal"] = self._step(state["bal"], reports, tiebreak)
        state["wait"] = np.where(x == 1, 0, state["wait"] + 1)
        state["cum"] = state["cum"] + x
        return x, np.zeros(reports.shape, dtype=np.float64)
