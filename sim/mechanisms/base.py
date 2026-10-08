"""
sim/mechanisms/base.py
======================
Abstract base class for all allocation mechanisms (M1–M5).

Every concrete mechanism must implement `allocate()`.  The interface
deliberately keeps payment computation inside the mechanism so that M5
(Vickrey) can return non-zero payments while M1–M4 always return zeros.

Tie-breaking contract
---------------------
Mechanisms that need randomness for tie-breaking must accept a `tie_seed`
argument and create their own `np.random.default_rng(tie_seed)` internally.
This keeps each mechanism's randomness deterministic and isolated from the
environment's generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sim.config import Config
from sim.environment import History


class Mechanism(ABC):
    """Abstract base class for a repeated allocation mechanism."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def allocate(
        self,
        reports: np.ndarray,
        history: History,
        tie_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Decide allocations and payments for one round.

        Parameters
        ----------
        reports : ndarray, shape (n,)
            Reported values v̂_{i,t} ∈ [0, v_max] for all n users.
        history : History
            Mutable history object (read-only inside this call — the
            runner updates history *after* calling allocate).
        tie_seed : int
            Seed for this round's tie-breaking RNG.

        Returns
        -------
        x : ndarray, shape (n,), dtype int64
            Allocation vector; x_i ∈ {0, 1}, Σ x_i = k.
        p : ndarray, shape (n,), dtype float64
            Payment vector; p_i = 0 whenever x_i = 0.
        """

    def allocate_batch(
        self,
        reports: np.ndarray,
        cumulative: np.ndarray,
        tiebreak: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorised ``allocate`` over B independent scenarios (used by the
        rollout attack).  Only mechanisms whose decision depends on history
        solely through the cumulative allocation counts can implement this.

        Parameters
        ----------
        reports, cumulative, tiebreak : ndarray, shape (B, n)
            Reports, cumulative allocations a_i(t) and the secondary
            tie-breaking key for every scenario.

        Returns
        -------
        (x, p) : both shape (B, n); x in {0,1} with k ones per row.
        """
        raise NotImplementedError(f"{self.name} has no batch allocation rule.")

    # ── Rollout interface (used by the finite-horizon rollout attack) ─────────
    # A rollout simulates B hypothetical futures of the *current* state.  The
    # default implementation covers mechanisms whose decision depends on the
    # past only through cumulative allocations (M3, M4, M5); stateful
    # mechanisms override both methods.

    def rollout_init(self, history: History, B: int) -> dict:
        """Mutable simulation state for B rollouts, copied from the current state."""
        return {"cum": np.tile(history.cumulative[None, :], (B, 1)).astype(np.int64)}

    def rollout_step(
        self,
        reports: np.ndarray,
        state: dict,
        tiebreak: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Allocate one simulated round for every rollout, advancing `state`."""
        x, p = self.allocate_batch(reports, state["cum"], tiebreak)
        state["cum"] = state["cum"] + x
        return x, p

    @property
    def name(self) -> str:
        return self.__class__.__name__
