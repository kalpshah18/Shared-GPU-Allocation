"""
sim/policies/strategic.py
==========================
Strategic reporting policies — Owner: Raj Modi

Four bounded policies from proposal §4.2:

1. Truthful          : v̂ = v
2. Capped exaggeration: v̂ = min(c·v, v_max)   for c ∈ {1.25, 1.5, 2}
3. Maximum claim     : v̂ = v_max               (worst-case priority inflation)
4. Rollout attack    : finite-horizon Monte Carlo search (diagnostic only,
                       n=10 small-instance setting, H=5 rounds ahead,
                       100 rollouts, report grid G={0, 0.1, ..., 1})

All policies expose a uniform call signature:

    report(true_value, history, cfg, **kwargs) -> float

so they can be stored in a dict and called uniformly by the runner.

Notes
-----
* The rollout attack is a unilateral heuristic, not an equilibrium
  computation or a proof of manipulability.  It is separated from the
  scalable policies (used in E2–E3).  The proposal (§4.2, §4.3) planned it
  for selected E3 settings on M3–M5; it is implemented and unit-tested but
  not run by any experiment script, so no rollout results are reported.
* Policies operate on scalar values; vectorisation across users is done in
  the runner.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from sim.config import Config
from sim.environment import History


# ── Policy protocol ──────────────────────────────────────────────────────────

class Policy(Protocol):
    """Callable protocol: (true_value, history, cfg, **kw) -> reported_value."""

    def __call__(
        self,
        true_value: float,
        history: History,
        cfg: Config,
        **kwargs,
    ) -> float: ...


# ── 1. Truthful ───────────────────────────────────────────────────────────────

def truthful(true_value: float, history: History, cfg: Config, **_) -> float:
    """Report true value exactly."""
    return true_value


# ── 2. Capped exaggeration ───────────────────────────────────────────────────

def capped_exaggeration(
    true_value: float,
    history: History,
    cfg: Config,
    c: float = 2.0,
    **_,
) -> float:
    """
    v̂ = min(c · v, v_max).

    c ∈ {1.25, 1.5, 2} (pass via kwargs when building the policy).
    """
    return min(c * true_value, cfg.v_max)


# ── 3. Maximum claim ─────────────────────────────────────────────────────────

def maximum_claim(true_value: float, history: History, cfg: Config, **_) -> float:
    """Always report v_max — worst-case priority inflation baseline."""
    return cfg.v_max


# ── 4. Rollout attack (diagnostic) ───────────────────────────────────────────

def rollout_attack(
    true_value: float,
    history: History,
    cfg: Config,
    focal_user: int,
    opponent_policies: list,
    mechanism_factory: Callable,
    rng: np.random.Generator,
    H: int = 5,
    n_rollouts: int = 100,
    grid: tuple[float, ...] = tuple(round(x * 0.1, 1) for x in range(11)),
    **_,
) -> float:
    """
    Finite-horizon rollout attack (diagnostic, small n ≤ 10 only).

    For each candidate report r ∈ grid, run `n_rollouts` Monte Carlo
    simulations of the next H rounds, holding opponent policies fixed.
    Select the report with the highest estimated cumulative utility.

    Parameters
    ----------
    focal_user      : index of the attacking user
    opponent_policies : list of policy callables, length n; entry for
                        focal_user is ignored (replaced by constant r).
    mechanism_factory : callable() -> Mechanism  (fresh instance each rollout)
    rng             : np.random.Generator for rollout randomness
    H               : horizon (rounds ahead), default 5
    n_rollouts      : Monte Carlo rollouts per candidate report, default 100
    grid            : discrete report candidates, default {0, 0.1, ..., 1}
    """
    best_report   = true_value
    best_utility  = -np.inf

    for r in grid:
        total_utility = 0.0
        for _ in range(n_rollouts):
            mech     = mechanism_factory()
            hist_sim = _clone_history(history, cfg.n)
            utility  = 0.0

            for h in range(H):
                # Sample valuations for this rollout step
                vals = rng.uniform(0.0, cfg.v_max, size=cfg.n)
                vals[focal_user] = true_value  # keep focal user's true value

                # Build reports: opponents use their policies
                reports = np.array([
                    opponent_policies[i](vals[i], hist_sim, cfg)
                    if i != focal_user else r
                    for i in range(cfg.n)
                ])

                tie_seed = int(rng.integers(0, 2**63))
                x, p = mech.allocate(reports, hist_sim, tie_seed)
                utility += vals[focal_user] * x[focal_user] - p[focal_user]
                hist_sim.update(x, p)

            total_utility += utility

        mean_utility = total_utility / n_rollouts
        if mean_utility > best_utility:
            best_utility = mean_utility
            best_report  = r

    return best_report


def _clone_history(history: History, n: int) -> History:
    """Shallow-copy the relevant fields of History for a rollout simulation."""
    from sim.environment import History as H
    h2 = H(n)
    h2.cumulative[:]       = history.cumulative
    h2.consecutive_wait[:] = history.consecutive_wait
    h2.round               = history.round
    # Don't copy raw allocation lists — they're not needed for mechanisms
    return h2


# ── Policy registry ──────────────────────────────────────────────────────────

POLICY_REGISTRY: dict[str, Policy] = {
    "truthful"   : truthful,       # type: ignore[dict-item]
    "cap_1.25"   : lambda v, h, c, **kw: capped_exaggeration(v, h, c, c=1.25),
    "cap_1.5"    : lambda v, h, c, **kw: capped_exaggeration(v, h, c, c=1.5),
    "cap_2"      : lambda v, h, c, **kw: capped_exaggeration(v, h, c, c=2.0),
    "max_claim"  : maximum_claim,  # type: ignore[dict-item]
    # "rollout" is not in the registry because it requires extra keyword args;
    # instantiate it directly in the experiment script.
}
