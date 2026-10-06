"""
sim/config.py
=============
Configuration dataclass for all experimental hyperparameters.
Loaded from a YAML file; individual fields can be overridden via CLI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    # ── Population & capacity ────────────────────────────────────────────────
    n: int = 50           # number of users
    k: int = 10           # GPUs available each round  (k < n always)
    T: int = 1000         # number of allocation rounds
    v_max: float = 1.0    # common upper bound on values and reports

    # ── Strategic population ─────────────────────────────────────────────────
    rho: float = 0.25     # fraction of strategic users  ρ ∈ [0, 1]

    # ── Mechanism-specific ───────────────────────────────────────────────────
    lambda_: float = 1.0  # history-penalty exponent for M4 (0 ≡ M3 Greedy)
    c: float = 2.0        # capped-exaggeration multiplier (1.25, 1.5, or 2)

    # ── Valuation process ────────────────────────────────────────────────────
    # "uniform"   → Uniform(0, v_max)        (base case)
    # "beta_low"  → Beta(2, 5)               (low-value group, E4)
    # "beta_high" → Beta(5, 2)               (high-value group, E4)
    # "mixed"     → first n//2 low, rest high (heterogeneous, E4)
    valuation_dist: str = "uniform"
    alpha: float = 0.0    # AR(1) persistence  (0 = i.i.d., sweep in E5)

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

    def __post_init__(self) -> None:
        if self.k >= self.n:
            raise ValueError(f"k={self.k} must be strictly less than n={self.n}.")
        if not (0.0 <= self.rho <= 1.0):
            raise ValueError(f"rho={self.rho} must be in [0, 1].")
        if self.lambda_ < 0:
            raise ValueError(f"lambda_={self.lambda_} must be >= 0.")
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError(f"alpha={self.alpha} must be in [0, 1).")
        self.delta = self.delta_multiplier * math.ceil(self.n / self.k)

    # ── Factories ─────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "Config":
        """Load from a YAML file, then apply keyword overrides."""
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        data.update(overrides)
        # Only pass fields that Config actually knows about
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def default(cls, **overrides) -> "Config":
        """Create a default config, optionally overriding specific fields."""
        return cls(**overrides)
