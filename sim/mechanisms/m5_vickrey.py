"""
sim/mechanisms/m5_vickrey.py
=============================
M5 — k-unit Vickrey Auction (VCG for k identical units)

Allocation rule : give GPUs to the k highest bidders.
Payment rule    : each winner pays the (k+1)-st highest bid.

    p_{i,t} = x_{i,t} · v̂_{(k+1),t}

Under unit demand, quasi-linear utility, and no inter-temporal budget, this
is the VCG outcome for k identical units and is DSIC (dominant-strategy
incentive compatible) in each round independently [Vickrey 1961].

Important caveats (from proposal §4.1)
---------------------------------------
* Payments are utility-equivalent transfers, not reusable virtual credits.
* M5 is used only as a mechanism-design baseline; it is not a deployment
  policy.
* DSIC is a one-round property; the simulation cannot establish dynamic
  strategy-proofness.
"""

from __future__ import annotations

import numpy as np

from sim.config import Config
from sim.environment import History
from sim.mechanisms.base import Mechanism
from sim.mechanisms._utils import seeded_top_k


class VickreyMechanism(Mechanism):
    """M5: k-unit Vickrey auction, per-round DSIC."""

    def allocate(
        self,
        reports: np.ndarray,
        history: History,
        tie_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        k = self.cfg.k
        n = self.cfg.n

        # Allocate to top-k bidders (seeded tie-break)
        winners = seeded_top_k(reports, k, tie_seed)

        x = np.zeros(n, dtype=np.int64)
        x[winners] = 1

        # (k+1)-st highest bid: sort descending, index k (0-based)
        sorted_bids = np.sort(reports)[::-1]
        threshold = sorted_bids[k] if k < n else 0.0  # edge-case: k == n-1

        p = np.zeros(n, dtype=np.float64)
        p[winners] = threshold  # all winners pay the same threshold price

        return x, p
