"""
sim/config.py
=============
Configuration dataclass for all experimental hyperparameters.
Loaded from a YAML file; individual fields can be overridden in code or via
the experiment CLIs.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALUATION_DISTS = ("uniform", "beta_low", "beta_high", "mixed")
AR1_MODES = ("proposal", "copula")


@dataclass
class Config:
    # ── Population & capacity ────────────────────────────────────────────────
    n: int = 50           # number of users
    k: int = 10           # GPUs available each round  (1 <= k < n)
    T: int = 1000         # number of allocation rounds
    v_max: float = 1.0    # common upper bound on values and reports

    # ── Strategic population ─────────────────────────────────────────────────
    rho: float = 0.25     # fraction of strategic users  ρ ∈ [0, 1]
    n_focal: int = 5      # focal users per seed for the unilateral gain M_i

    # ── Mechanism-specific ───────────────────────────────────────────────────
    lambda_: float = 1.0  # history-penalty exponent for M4 (0 ≡ M3 Greedy)
    c: float = 2.0        # capped-exaggeration multiplier (1.25, 1.5, or 2)

    # ── Proposed mechanisms (M6 Karma-Cap, M7 Rank-Cap) ───────────────────────
    wait_cap: int | None = None   # hard waiting bound W (None -> Delta = 2*ceil(n/k))
    karma_init: float = 2.0       # initial / mean karma balance per user (M6)
    karma_cap_mult: float = 3.0   # balance ceiling = karma_cap_mult * karma_init (M6)
    rank_blend: float = 0.0       # M7 score = (1-b)*rank + b*report, b in [0, 1]

    # ── Valuation process ────────────────────────────────────────────────────
    # "uniform"   → Uniform(0, v_max)        (base case)
    # "beta_low"  → Beta(2, 5)               (low-value group, E4)
    # "beta_high" → Beta(5, 2)               (high-value group, E4)
    # "mixed"     → first n//2 low, rest high (heterogeneous, E4)
    valuation_dist: str = "uniform"
    alpha: float = 0.0    # AR(1) persistence  (0 = i.i.d., sweep in E5)
    # "proposal": v_t = α v_{t-1} + (1-α) ε_t  (proposal §4.3; shrinks variance)
    # "copula"  : Gaussian-AR(1) copula; keeps the marginal law of D_i exactly
    ar1_mode: str = "proposal"

    # ── Reproducibility ──────────────────────────────────────────────────────
    n_seeds: int = 30
    n_bootstrap: int = 10_000
    delta_multiplier: int = 2   # Δ = delta_multiplier * ceil(n / k)

    # ── Paths ────────────────────────────────────────────────────────────────
    results_dir: str = "results"
    figures_dir: str = "figures"
    seeds_file: str = "seeds/master_seeds.json"

    # ── Derived (not set by user) ─────────────────────────────────────────────
    delta: int = field(init=False)
    wait_limit: int = field(init=False)   # effective waiting cap W

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"n={self.n} must be at least 2.")
        if self.k < 1 or self.k >= self.n:
            raise ValueError(f"k={self.k} must satisfy 1 <= k < n={self.n}.")
        if self.T < 1:
            raise ValueError(f"T={self.T} must be at least 1.")
        if not self.v_max > 0:
            raise ValueError(f"v_max={self.v_max} must be positive.")
        if not (0.0 <= self.rho <= 1.0):
            raise ValueError(f"rho={self.rho} must be in [0, 1].")
        if self.lambda_ < 0:
            raise ValueError(f"lambda_={self.lambda_} must be >= 0.")
        if self.c < 1:
            raise ValueError(f"c={self.c} must be >= 1.")
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError(f"alpha={self.alpha} must be in [0, 1).")
        if self.valuation_dist not in VALUATION_DISTS:
            raise ValueError(
                f"valuation_dist='{self.valuation_dist}' not in {VALUATION_DISTS}."
            )
        if self.ar1_mode not in AR1_MODES:
            raise ValueError(f"ar1_mode='{self.ar1_mode}' not in {AR1_MODES}.")
        if self.n_focal < 1:
            raise ValueError(f"n_focal={self.n_focal} must be at least 1.")
        if self.wait_cap is not None and self.wait_cap < 1:
            raise ValueError(f"wait_cap={self.wait_cap} must be >= 1 (or None).")
        if self.karma_init <= 0:
            raise ValueError(f"karma_init={self.karma_init} must be positive.")
        if self.karma_cap_mult < 1:
            raise ValueError(f"karma_cap_mult={self.karma_cap_mult} must be >= 1.")
        if not (0.0 <= self.rank_blend <= 1.0):
            raise ValueError(f"rank_blend={self.rank_blend} must be in [0, 1].")
        self.delta = self.delta_multiplier * math.ceil(self.n / self.k)
        self.wait_limit = self.delta if self.wait_cap is None else int(self.wait_cap)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def replace(self, **changes) -> "Config":
        """Return a copy with `changes` applied (re-validated, delta re-derived)."""
        kwargs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init}
        kwargs.update(changes)
        return Config(**kwargs)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot of every field (including derived delta)."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    # ── Factories ─────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "Config":
        """Load from a YAML file, then apply keyword overrides.

        Unknown keys are ignored; derived fields (``delta``) cannot be set.
        """
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        data.update(overrides)
        known = {f.name for f in dataclasses.fields(cls) if f.init}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def default(cls, **overrides) -> "Config":
        """Create a default config, optionally overriding specific fields."""
        return cls(**overrides)
