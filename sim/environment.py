"""
sim/environment.py
==================
Core simulator engine — Owner: Aayush Kuloor

Responsibilities
----------------
* Independent seeded RNG streams per master seed (valuations, tie-breaking,
  strategic-set selection), derived with ``SeedSequence.spawn``.
* Valuation processes: i.i.d. Uniform / Beta, and two AR(1) variants.
* Per-round mutable history: cumulative allocations a_i(t), consecutive
  wait q_i(t), and the public history h_t = (x_1, ..., x_{t-1}).
* JSON-safe result serialisation.

The paired-randomness contract
------------------------------
Before any mechanism runs, ``SeedPackage.generate`` materialises the complete
valuation tensor, tie-breaking seeds and strategic set for a master seed.
Every mechanism and every counterfactual run then reads from the same
pre-generated arrays, so all comparisons are paired on one omega.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import ndtr

from sim.config import Config

# Beta parameters of the two E4 groups.
BETA_LOW  = (2.0, 5.0)
BETA_HIGH = (5.0, 2.0)


# ── Marginal distributions ────────────────────────────────────────────────────

def _user_dists(n: int, dist: str) -> list:
    """Per-user marginal: None = Uniform(0, v_max), (a, b) = Beta(a, b) * v_max."""
    if dist == "uniform":
        return [None] * n
    if dist == "beta_low":
        return [BETA_LOW] * n
    if dist == "beta_high":
        return [BETA_HIGH] * n
    if dist == "mixed":
        half = n // 2
        return [BETA_LOW] * half + [BETA_HIGH] * (n - half)
    raise ValueError(f"Unknown valuation_dist='{dist}'.")


def expected_values(cfg: Config) -> np.ndarray:
    """mu_i = E_{v ~ D_i}[v] for every user (shape (n,)); used by J_B."""
    mu = np.empty(cfg.n)
    for i, d in enumerate(_user_dists(cfg.n, cfg.valuation_dist)):
        mu[i] = 0.5 if d is None else d[0] / (d[0] + d[1])
    return mu * cfg.v_max


def sample_iid_values(rng: np.random.Generator, cfg: Config, size: tuple) -> np.ndarray:
    """
    Draw i.i.d. values from each user's marginal D_i.  Returns shape ``size + (n,)``.
    Used by the rollout attack to simulate future rounds.
    """
    out = np.empty(tuple(size) + (cfg.n,), dtype=np.float64)
    for i, d in enumerate(_user_dists(cfg.n, cfg.valuation_dist)):
        if d is None:
            out[..., i] = rng.uniform(0.0, cfg.v_max, size=size)
        else:
            out[..., i] = rng.beta(d[0], d[1], size=size) * cfg.v_max
    return out


def _sample_base(rng: np.random.Generator, cfg: Config) -> np.ndarray:
    """i.i.d. (n, T) draw from each user's marginal."""
    return np.moveaxis(sample_iid_values(rng, cfg, (cfg.T,)), -1, 0)


# ── Valuation tensor generation ───────────────────────────────────────────────

def _sample_valuations(rng: np.random.Generator, cfg: Config) -> np.ndarray:
    """
    Return a (n, T) float64 tensor of true valuations in [0, v_max].

    alpha == 0                    -> i.i.d. draws from D_i.
    alpha  > 0, ar1_mode=proposal -> v_t = a v_{t-1} + (1-a) eps_t, eps_t ~ D_i
        (proposal section 4.3).  v_0 is initialised at the stationary mean and
        variance, so there is no transient.  NOTE: this recursion also shrinks
        the marginal std by sqrt((1-a)/(1+a)); high alpha therefore narrows the
        value spread as well as adding persistence.
    alpha  > 0, ar1_mode=copula   -> latent Gaussian AR(1) z_t = a z_{t-1} +
        sqrt(1-a^2) xi_t pushed through Phi and D_i^{-1}: the marginal law of
        every v_{i,t} is exactly D_i, so only the persistence changes.
    """
    n, T, alpha = cfg.n, cfg.T, cfg.alpha

    if alpha == 0.0:
        return _sample_base(rng, cfg)

    if cfg.ar1_mode == "proposal":
        eps = _sample_base(rng, cfg)
        mu  = expected_values(cfg)
        v = np.empty((n, T), dtype=np.float64)
        shrink = math.sqrt((1.0 - alpha) / (1.0 + alpha))   # stationary std ratio
        v[:, 0] = mu + shrink * (eps[:, 0] - mu)
        for t in range(1, T):
            v[:, t] = alpha * v[:, t - 1] + (1.0 - alpha) * eps[:, t]
        return np.clip(v, 0.0, cfg.v_max)

    # copula
    xi = rng.standard_normal(size=(n, T))
    z = np.empty((n, T), dtype=np.float64)
    z[:, 0] = xi[:, 0]
    s = math.sqrt(1.0 - alpha * alpha)
    for t in range(1, T):
        z[:, t] = alpha * z[:, t - 1] + s * xi[:, t]
    u = np.clip(ndtr(z), 1e-12, 1.0 - 1e-12)
    v = np.empty((n, T), dtype=np.float64)
    for i, d in enumerate(_user_dists(n, cfg.valuation_dist)):
        v[i] = u[i] * cfg.v_max if d is None else stats.beta.ppf(u[i], d[0], d[1]) * cfg.v_max
    return np.clip(v, 0.0, cfg.v_max)


# ── Pre-generated seed package ────────────────────────────────────────────────

class SeedPackage:
    """
    All randomness required for one (master_seed, Config) combination.

    Attributes
    ----------
    valuations : ndarray, shape (n, T)    true private valuations.
    tie_seeds  : ndarray, shape (T,), uint64
        One 64-bit integer per round for seeded tie-breaking inside
        mechanisms.  Each mechanism builds its own generator from it.
    strategic_set : ndarray of int
        Sorted indices of the floor(rho*n) strategic users.
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
        Materialise the full seed package from a single master seed using three
        independent child generators (``SeedSequence.spawn``).  The strategic
        set is a prefix of one random permutation, so for a fixed seed raising
        rho only adds users (the rho=0.1 set is contained in the rho=0.25 set).
        """
        ss = np.random.SeedSequence(master_seed)
        val_ss, tie_ss, strat_ss = ss.spawn(3)

        val_rng   = np.random.default_rng(val_ss)
        tie_rng   = np.random.default_rng(tie_ss)
        strat_rng = np.random.default_rng(strat_ss)

        valuations  = _sample_valuations(val_rng, cfg)
        tie_seeds   = tie_rng.integers(0, 2**63, size=cfg.T, dtype=np.uint64)
        n_strategic = math.floor(cfg.rho * cfg.n)
        order       = strat_rng.permutation(cfg.n)
        strategic_set = np.sort(order[:n_strategic])

        return cls(valuations, tie_seeds, strategic_set, master_seed)


# ── Per-round mutable history ─────────────────────────────────────────────────

class History:
    """
    Mutable state updated after each allocation round.

    Attributes
    ----------
    cumulative : ndarray (n,)        a_i(t) = sum_{tau<t} x_{i,tau}
    consecutive_wait : ndarray (n,)  q_i(t): rounds since i last received a GPU
    allocations, payments : lists of per-round (n,) arrays
    round : int                      index of the next round to be played
    """

    def __init__(self, n: int) -> None:
        self.n                = n
        self.cumulative       = np.zeros(n, dtype=np.int64)
        self.consecutive_wait = np.zeros(n, dtype=np.int64)
        self.allocations: list = []
        self.payments:    list = []
        self.round: int = 0

    def update(self, x: np.ndarray, p: np.ndarray) -> None:
        """Record x and p for the current round and advance the counters."""
        self.allocations.append(x.copy())
        self.payments.append(p.copy())
        self.cumulative += x
        self.consecutive_wait = np.where(x == 1, 0, self.consecutive_wait + 1)
        self.round += 1

    def reset(self) -> None:
        """Reset to the start-of-simulation state (round 0)."""
        self.cumulative[:]       = 0
        self.consecutive_wait[:] = 0
        self.allocations.clear()
        self.payments.clear()
        self.round = 0

    def snapshot(self) -> "History":
        """Cheap copy of the counters only (no per-round records)."""
        h = History(self.n)
        h.cumulative[:]       = self.cumulative
        h.consecutive_wait[:] = self.consecutive_wait
        h.round               = self.round
        return h


# ── Result store ──────────────────────────────────────────────────────────────

def to_jsonable(obj):
    """Recursively convert numpy types to Python types; NaN/inf become None."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def save_json(obj, path) -> Path:
    """Write `obj` as strict JSON (no NaN literals), creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(to_jsonable(obj), fh, indent=2, allow_nan=False)
    return path
