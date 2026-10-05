"""
sim/environment.py
==================
Core simulator engine — Owner: Aayush Kuloor

Responsibilities
----------------
* Two independent seeded RNG streams per master seed:
    - val_rng  : samples the full valuation tensor v[n, T]
    - tie_rng  : resolves allocation ties uniformly
* Valuation processes: i.i.d. Uniform, Beta, and AR(1) persistence
* Per-round mutable history: cumulative allocations a_i(t), consecutive
  wait q_i(t), and the public history h_t = (x_1, ..., x_{t-1})
* Result serialisation to JSON

The paired-randomness contract
-------------------------------
Before any mechanism runs, pre_generate() materialises the complete
valuation tensor and tie-breaking seeds for a given master seed.  Every
mechanism then reads from the same pre-generated arrays, guaranteeing
identical inputs across M1–M5 for the same master seed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from sim.config import Config


# ── Valuation tensor generation ───────────────────────────────────────────────

def _sample_valuations(
    rng: np.random.Generator,
    n: int,
    T: int,
    v_max: float,
    dist: str,
    alpha: float,
) -> np.ndarray:
    """
    Return a (n, T) float64 tensor of true valuations in [0, v_max].

    Parameters
    ----------
    dist : "uniform" | "beta_low" | "beta_high" | "mixed"
        "mixed" assigns the first n//2 users to Beta(2,5) and the rest
        to Beta(5,2), as in experiment E4.
    alpha : AR(1) persistence coefficient; 0 means i.i.d.
    """
    if dist == "uniform":
        base = rng.uniform(0.0, v_max, size=(n, T))
    elif dist == "beta_low":
        base = rng.beta(2, 5, size=(n, T)) * v_max
    elif dist == "beta_high":
        base = rng.beta(5, 2, size=(n, T)) * v_max
    elif dist == "mixed":
        half = n // 2
        low  = rng.beta(2, 5, size=(half,     T)) * v_max
        high = rng.beta(5, 2, size=(n - half, T)) * v_max
        base = np.vstack([low, high])
    else:
        raise ValueError(f"Unknown valuation_dist='{dist}'.")

    if alpha == 0.0:
        return base  # i.i.d. — no further processing needed

    # AR(1):  v_{i,t} = alpha * v_{i,t-1} + (1-alpha) * epsilon_{i,t}
    # We use `base` as the epsilon draws; the first column is initialised
    # from a fresh uniform draw so the process starts in the stationary range.
    v = np.empty((n, T), dtype=np.float64)
    v[:, 0] = rng.uniform(0.0, v_max, size=n)
    for t in range(1, T):
        v[:, t] = alpha * v[:, t - 1] + (1.0 - alpha) * base[:, t]
    # Clip to [0, v_max] to keep values in the stated bounded domain
    return np.clip(v, 0.0, v_max)


# ── Pre-generated seed package ────────────────────────────────────────────────

class SeedPackage:
    """
    All randomness required for one (master_seed, Config) combination.

    Attributes
    ----------
    valuations : ndarray, shape (n, T)
        True private valuations.
    tie_seeds : ndarray, shape (T,), dtype=uint64
        One 64-bit integer per round for seeded tie-breaking inside
        mechanisms.  Each mechanism must use its own
        np.random.default_rng(tie_seed) so that tie-breaking is
        deterministic and independent of mechanism internals.
    strategic_set : ndarray, shape (floor(rho*n),), dtype=int
        Sorted indices of the ⌊ρ·n⌋ strategic users.
    master_seed : int
    """

    def __init__(
        self,
        valuations: np.ndarray,
        tie_seeds: np.ndarray,
        strategic_set: np.ndarray,
        master_seed: int,
    ) -> None:
        self.valuations    = valuations
        self.tie_seeds     = tie_seeds
        self.strategic_set = strategic_set
        self.master_seed   = master_seed

    @classmethod
    def generate(cls, master_seed: int, cfg: Config) -> "SeedPackage":
        """
        Materialise the full seed package from a single master seed.
        Uses two independent child generators (SeedSequence.spawn) so
        val_rng and tie_rng are statistically independent.
        """
        ss       = np.random.SeedSequence(master_seed)
        val_ss, tie_ss, strat_ss = ss.spawn(3)

        val_rng   = np.random.default_rng(val_ss)
        tie_rng   = np.random.default_rng(tie_ss)
        strat_rng = np.random.default_rng(strat_ss)

        valuations    = _sample_valuations(
            val_rng, cfg.n, cfg.T, cfg.v_max, cfg.valuation_dist, cfg.alpha
        )
        tie_seeds     = tie_rng.integers(0, 2**63, size=cfg.T, dtype=np.uint64)
        n_strategic   = math.floor(cfg.rho * cfg.n)
        strategic_set = np.sort(
            strat_rng.choice(cfg.n, size=n_strategic, replace=False)
        )

        return cls(valuations, tie_seeds, strategic_set, master_seed)


# ── Per-round mutable history ─────────────────────────────────────────────────

class History:
    """
    Mutable state updated after each allocation round.

    Attributes
    ----------
    cumulative : ndarray, shape (n,)
        a_i(t) = Σ_{τ<t} x_{i,τ}  — cumulative allocations up to (not
        including) the current round.
    consecutive_wait : ndarray, shape (n,)
        q_i(t) = rounds since user i last received a GPU (0 if allocated
        last round, increments each round without service).
    allocations : list of ndarray
        x_t for each completed round, in order.
    payments : list of ndarray
        p_t for each completed round, in order.
    round : int
        Index of the *next* round to be played (0-indexed).
    """

    def __init__(self, n: int) -> None:
        self.n                = n
        self.cumulative       = np.zeros(n, dtype=np.int64)
        self.consecutive_wait = np.zeros(n, dtype=np.int64)
        self.allocations: list[np.ndarray] = []
        self.payments:    list[np.ndarray] = []
        self.round: int = 0

    def update(self, x: np.ndarray, p: np.ndarray) -> None:
        """
        Record the allocation x and payment p for the current round, then
        advance internal counters.

        Parameters
        ----------
        x : ndarray, shape (n,), dtype int  —  x_i ∈ {0, 1}
        p : ndarray, shape (n,), dtype float
        """
        self.allocations.append(x.copy())
        self.payments.append(p.copy())

        self.cumulative += x

        # q_i: reset to 0 if allocated, else increment
        self.consecutive_wait = np.where(x == 1, 0, self.consecutive_wait + 1)

        self.round += 1

    def reset(self) -> None:
        """Reset to the start-of-simulation state (round 0)."""
        self.cumulative[:]       = 0
        self.consecutive_wait[:] = 0
        self.allocations.clear()
        self.payments.clear()
        self.round = 0


# ── Result store ──────────────────────────────────────────────────────────────

def save_seed_result(
    result: dict,
    experiment: str,
    master_seed: int,
    results_dir: str = "results",
) -> Path:
    """
    Serialise a per-seed result dict to
        results/<experiment>/<master_seed>.json

    numpy arrays are converted to lists for JSON compatibility.
    """
    out_dir = Path(results_dir) / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{master_seed}.json"

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")

    with open(out_path, "w") as fh:
        json.dump(result, fh, default=_convert, indent=2)

    return out_path
