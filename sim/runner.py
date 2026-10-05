"""
sim/runner.py
=============
End-to-end experiment runner — Owner: Aayush Kuloor

Orchestrates a single simulation run for one (mechanism, seed_package, cfg)
triple, returning the raw per-seed result dict ready for serialisation.

Separation of concerns
-----------------------
* The runner handles the round loop and ties together environment, mechanism,
  and policies.
* It does NOT compute metrics or bootstrap intervals — those live in
  sim/metrics.py and analysis/bootstrap.py.
* Paired runs (truthful vs strategic) are handled by run_paired().
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from sim.config import Config
from sim.environment import History, SeedPackage
from sim.mechanisms.base import Mechanism


def run_single(
    mechanism: Mechanism,
    pkg: SeedPackage,
    cfg: Config,
    policy_fn: Callable[[float, History, Config], float] | None = None,
) -> History:
    """
    Simulate T rounds for a single mechanism under a given reporting policy.

    Parameters
    ----------
    mechanism : Mechanism
        A freshly instantiated mechanism (not shared across seeds).
    pkg : SeedPackage
        Pre-generated valuation tensor, tie seeds, and strategic set.
    cfg : Config
    policy_fn : callable or None
        If None, all users report truthfully.
        If provided, ALL n users use this policy (use run_paired() for
        mixed-population experiments).

    Returns
    -------
    history : History  — completed history after T rounds
    """
    history = History(cfg.n)

    for t in range(cfg.T):
        v_t = pkg.valuations[:, t]

        # Build reports
        if policy_fn is None:
            reports = v_t.copy()
        else:
            reports = np.array([
                policy_fn(v_t[i], history, cfg) for i in range(cfg.n)
            ])

        x, p = mechanism.allocate(reports, history, int(pkg.tie_seeds[t]))
        history.update(x, p)

    return history


def run_mixed(
    mechanism: Mechanism,
    pkg: SeedPackage,
    cfg: Config,
    strategic_policy: Callable[[float, History, Config], float],
) -> History:
    """
    Simulate T rounds with a mixed population:
    * Strategic users (indices in pkg.strategic_set) use `strategic_policy`.
    * All other users report truthfully.

    Parameters
    ----------
    strategic_policy : callable
        The reporting policy for strategic users.

    Returns
    -------
    history : History
    """
    is_strategic = np.zeros(cfg.n, dtype=bool)
    is_strategic[pkg.strategic_set] = True

    history = History(cfg.n)

    for t in range(cfg.T):
        v_t = pkg.valuations[:, t]
        reports = np.empty(cfg.n, dtype=np.float64)

        for i in range(cfg.n):
            if is_strategic[i]:
                reports[i] = strategic_policy(v_t[i], history, cfg)
            else:
                reports[i] = v_t[i]

        x, p = mechanism.allocate(reports, history, int(pkg.tie_seeds[t]))
        history.update(x, p)

    return history


def run_paired(
    mechanism: Mechanism,
    pkg: SeedPackage,
    cfg: Config,
    strategic_policy: Callable[[float, History, Config], float],
) -> tuple[History, History]:
    """
    Run the same seed twice:
      1. fully truthful (baseline)
      2. mixed population with strategic_policy

    Both runs use identical ω (same pkg), so differences are due solely to
    the strategic policy.  Histories are independent objects.

    Returns
    -------
    (history_truthful, history_strategic)
    """
    # Need fresh mechanism instances for each run to reset stateful mechanisms
    # (e.g. M2's queue).  The caller is responsible for passing fresh instances
    # when mechanism is stateful.  Document this clearly.
    history_truthful  = run_single(mechanism, pkg, cfg, policy_fn=None)
    return history_truthful, run_mixed(mechanism, pkg, cfg, strategic_policy)
