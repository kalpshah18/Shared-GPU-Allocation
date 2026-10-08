"""
sim/runner.py
=============
End-to-end simulation runners — Owner: Aayush Kuloor

Orchestrates one simulation of a (mechanism, seed package, config) triple.

Separation of concerns
-----------------------
* The runner owns the round loop and joins environment, mechanism and policies.
* It does NOT compute metrics or bootstrap intervals (sim/metrics.py,
  analysis/bootstrap.py).
* Every counterfactual run takes the same SeedPackage (one omega) and a *fresh
  deep copy* of the mechanism, so stateful mechanisms (M2) start identically.

Policies are elementwise callables ``policy(values, history, cfg)`` (see
sim/policies/strategic.py).
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from sim.config import Config
from sim.environment import History, SeedPackage
from sim.mechanisms.base import Mechanism
from sim.metrics import utilities
from sim.policies.strategic import call_policy, rollout_report


def _checked(reports, cfg: Config) -> np.ndarray:
    """Validate that reports are finite and inside [0, v_max]."""
    r = np.asarray(reports, dtype=np.float64)
    if not np.all(np.isfinite(r)) or r.min() < 0.0 or r.max() > cfg.v_max + 1e-12:
        raise ValueError(f"reports outside [0, v_max={cfg.v_max}]: min={r.min()}, max={r.max()}")
    return r


def run_mixed(
    mechanism: Mechanism,
    pkg: SeedPackage,
    cfg: Config,
    strategic_policy: Callable | None,
    strategic_set: np.ndarray | None = None,
) -> History:
    """
    Simulate T rounds.  Users in `strategic_set` (default ``pkg.strategic_set``)
    report through `strategic_policy`; everyone else reports truthfully.
    ``strategic_policy=None`` means all users are truthful.
    """
    if strategic_set is None:
        strategic_set = pkg.strategic_set
    strategic_set = np.asarray(strategic_set, dtype=np.int64)
    active = strategic_policy is not None and len(strategic_set) > 0

    history = History(cfg.n)
    for t in range(cfg.T):
        v_t = pkg.valuations[:, t]
        reports = v_t.copy()
        if active:
            reports[strategic_set] = call_policy(strategic_policy, v_t[strategic_set], history, cfg,
                                                 users=strategic_set)
        x, p = mechanism.allocate(_checked(reports, cfg), history, int(pkg.tie_seeds[t]))
        history.update(x, p)
    return history


def run_single(
    mechanism: Mechanism,
    pkg: SeedPackage,
    cfg: Config,
    policy_fn: Callable | None = None,
) -> History:
    """
    Simulate T rounds with ALL n users following `policy_fn` (None = truthful).
    Use ``run_mixed`` / ``run_paired`` for mixed populations.
    """
    return run_mixed(mechanism, pkg, cfg, policy_fn, np.arange(cfg.n))


def run_paired(
    mechanism: Mechanism,
    pkg: SeedPackage,
    cfg: Config,
    strategic_policy: Callable,
) -> tuple[History, History]:
    """
    Run one seed twice on the same omega: (1) fully truthful and (2) with the
    strategic set following `strategic_policy`.  Returns (truthful, strategic).
    """
    h_truth = run_mixed(copy.deepcopy(mechanism), pkg, cfg, None)
    h_strat = run_mixed(copy.deepcopy(mechanism), pkg, cfg, strategic_policy)
    return h_truth, h_strat


# ── Unilateral manipulation gains ─────────────────────────────────────────────

def focal_users(pkg: SeedPackage, cfg: Config) -> np.ndarray:
    """
    The users whose individual incentive M_i is measured for this seed: the
    first ``cfg.n_focal`` strategic users, or (when rho = 0) ``n_focal``
    evenly spaced users, each a lone deviator in an otherwise truthful
    population.
    """
    S = pkg.strategic_set
    if len(S) > 0:
        return S[: cfg.n_focal].astype(np.int64)
    return np.unique(np.linspace(0, cfg.n - 1, min(cfg.n_focal, cfg.n)).astype(np.int64))


def unilateral_gains(
    mechanism: Mechanism,
    pkg: SeedPackage,
    cfg: Config,
    strategic_policy: Callable,
    h_strat: History | None = None,
    focals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Individual manipulation gains M_f(sigma_f, sigma_{-f}; omega) for each
    focal user f, holding every other user's behaviour fixed:

        f in S :  U_f(S deviates)      - U_f(S \\ {f} deviates)   (f drops out)
        f not in S: U_f(S u {f} deviates) - U_f(S deviates)       (f joins in)

    where S = pkg.strategic_set.  Both terms use the same omega.  `h_strat`
    is the shared run in which exactly S deviates (computed if omitted).

    Returns (focals, gains), both of shape (len(focals),).
    """
    S = pkg.strategic_set
    if focals is None:
        focals = focal_users(pkg, cfg)
    if h_strat is None:
        h_strat = run_mixed(copy.deepcopy(mechanism), pkg, cfg, strategic_policy, S)
    U_shared = utilities(h_strat, pkg.valuations)

    in_S = set(int(s) for s in S)
    gains = np.empty(len(focals))
    for j, f in enumerate(focals):
        f = int(f)
        if f in in_S:
            others = np.array([s for s in S if s != f], dtype=np.int64)
            h_base = run_mixed(copy.deepcopy(mechanism), pkg, cfg, strategic_policy, others)
            gains[j] = U_shared[f] - utilities(h_base, pkg.valuations)[f]
        else:
            joined = np.union1d(S, [f]).astype(np.int64)
            h_dev = run_mixed(copy.deepcopy(mechanism), pkg, cfg, strategic_policy, joined)
            gains[j] = utilities(h_dev, pkg.valuations)[f] - U_shared[f]
    return np.asarray(focals, dtype=np.int64), gains


# ── Rollout attack runner ─────────────────────────────────────────────────────

def run_rollout(
    mechanism: Mechanism,
    pkg: SeedPackage,
    cfg: Config,
    focal: int,
    opponent_policy: Callable | None,
    rng: np.random.Generator,
    H: int = 5,
    n_rollouts: int = 100,
    grid: tuple | None = None,
) -> tuple[History, np.ndarray]:
    """
    Simulate T rounds in which user `focal` plays the finite-horizon rollout
    attack and every other user follows `opponent_policy` (None = truthful).

    Returns (history, focal_reports) where focal_reports has shape (T,).
    """
    kw = {} if grid is None else {"grid": grid}
    history = History(cfg.n)
    focal_reports = np.empty(cfg.T)
    for t in range(cfg.T):
        v_t = pkg.valuations[:, t]
        reports = v_t.copy()
        if opponent_policy is not None:
            reports = np.asarray(call_policy(opponent_policy, v_t, history, cfg),
                                 dtype=np.float64).copy()
        reports[focal] = rollout_report(
            float(v_t[focal]), history, mechanism, cfg, focal,
            opponent_policy if opponent_policy is not None else (lambda v, h, c: v),
            rng, H=H, n_rollouts=n_rollouts, **kw,
        )
        focal_reports[t] = reports[focal]
        x, p = mechanism.allocate(_checked(reports, cfg), history, int(pkg.tie_seeds[t]))
        history.update(x, p)
    return history, focal_reports
