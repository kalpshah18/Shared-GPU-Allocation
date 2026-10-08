"""
sim/policies/strategic.py
==========================
Strategic reporting policies — Owner: Raj Modi

Four bounded policies from proposal section 4.2, plus one extension:

1. Truthful            : v_hat = v
2. Capped exaggeration : v_hat = min(c v, v_max)   for c in {1.25, 1.5, 2}
3. Maximum claim       : v_hat = v_max             (worst-case priority inflation)
4. Finite-horizon rollout attack : Monte Carlo search over the report grid
   G = {0, 0.1, ..., 1} using H = 5 rounds and 100 rollouts per candidate
   (a diagnostic for small n, see experiments/e3b_rollout.py).
5. Timed exaggeration (extension, tests hypothesis H2): inflate by the capped
   rule only in rounds where the user's own cumulative allocation is at or
   below the population mean, i.e. when the history penalty is smallest, and
   report truthfully otherwise.  It needs to know *which* user it controls, so
   it declares ``needs_users`` and receives ``users=<indices>`` (see
   ``call_policy``).

Call contract
-------------
Scalable policies share one signature

    policy(true_value, history, cfg, **kwargs) -> report

and are *elementwise*: `true_value` may be a float or an ndarray of values,
and the report has the same shape.  Reports always lie in [0, v_max].

Notes
-----
* The rollout attack is a unilateral heuristic, not an equilibrium
  computation and not a proof of manipulability.
* Future rounds in a rollout are simulated from the i.i.d. marginals D_i,
  with all other users following `opponent_policy`; the focal user reports
  truthfully after the first simulated round (the standard rollout base
  policy).  Candidates share common random numbers, so differences between
  candidate reports are not Monte Carlo noise.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from sim.config import Config
from sim.environment import History, sample_iid_values
from sim.mechanisms.base import Mechanism

ROLLOUT_GRID = tuple(round(0.1 * i, 1) for i in range(11))


# ── 1. Truthful ───────────────────────────────────────────────────────────────

def truthful(true_value, history: History | None, cfg: Config, **_):
    """Report the true value exactly."""
    return true_value


# ── 2. Capped exaggeration ───────────────────────────────────────────────────

def capped_exaggeration(true_value, history: History | None, cfg: Config, c: float = 2.0, **_):
    """v_hat = min(c * v, v_max), elementwise."""
    return np.minimum(c * np.asarray(true_value, dtype=np.float64), cfg.v_max)[()]


def make_capped(c: float) -> Callable:
    """Capped-exaggeration policy with multiplier `c` bound in."""
    def policy(true_value, history, cfg, **_):
        return capped_exaggeration(true_value, history, cfg, c=c)
    policy.__name__ = f"cap_{c:g}"
    return policy


# ── 3. Maximum claim ─────────────────────────────────────────────────────────

def maximum_claim(true_value, history: History | None, cfg: Config, **_):
    """Always report v_max (worst-case priority inflation)."""
    return np.full(np.shape(true_value), cfg.v_max, dtype=np.float64)[()]


# ── 4. Rollout attack (diagnostic) ───────────────────────────────────────────

def rollout_report(
    true_value: float,
    history: History,
    mechanism: Mechanism,
    cfg: Config,
    focal: int,
    opponent_policy: Callable,
    rng: np.random.Generator,
    H: int = 5,
    n_rollouts: int = 100,
    grid: tuple = ROLLOUT_GRID,
) -> float:
    """
    Choose the focal user's report for the current round.

    For every candidate report r in `grid` (scaled by v_max), simulate the
    next H rounds `n_rollouts` times: the current round uses r and the focal
    user's known `true_value`; the other users' values are drawn from their
    marginals and reported through `opponent_policy`; later rounds use fresh
    values with a truthful focal user.  The candidate with the highest mean
    cumulative focal utility wins; ties go to the candidate closest to the
    true value.

    `mechanism` must support the rollout interface (``rollout_init`` /
    ``rollout_step``): M3, M4, M5 via ``allocate_batch``; M6, M7 natively.
    """
    n = cfg.n
    cand = np.asarray(grid, dtype=np.float64) * cfg.v_max
    G, R = len(cand), n_rollouts

    vals = sample_iid_values(rng, cfg, (H, R))              # (H, R, n)
    vals[0, :, focal] = true_value
    tiebreak = rng.random((H, R, n))

    state = mechanism.rollout_init(history, G * R)
    util = np.zeros((G, R))
    for h in range(H):
        opp = np.asarray(call_policy(opponent_policy, vals[h], history, cfg),
                         dtype=np.float64)                                     # (R, n)
        reports = np.broadcast_to(opp, (G, R, n)).copy()
        reports[:, :, focal] = cand[:, None] if h == 0 else vals[h, :, focal][None, :]
        x, p = mechanism.rollout_step(reports.reshape(G * R, n), state,
                                      np.broadcast_to(tiebreak[h], (G, R, n)).reshape(G * R, n))
        x = x.reshape(G, R, n)
        p = p.reshape(G, R, n)
        util += vals[h, :, focal][None, :] * x[:, :, focal] - p[:, :, focal]

    mean_util = util.mean(axis=1)
    best = np.flatnonzero(mean_util >= mean_util.max() - 1e-12)
    return float(cand[best[np.argmin(np.abs(cand[best] - true_value))]])


# ── 5. Timed exaggeration (extension) ────────────────────────────────────────

def make_timed(c: float = 2.0, quantile: float | None = None) -> Callable:
    """
    Capped exaggeration applied only while the user's own cumulative allocation
    a_i(t) is at or below a population threshold (low history penalty), and
    truthful otherwise.  The threshold is the population mean of a(t) by
    default, or its `quantile` (e.g. 0.25 = inflate only when among the
    least-served quarter).  Works elementwise: `true_value` has one entry per
    controlled user (indices passed as ``users``) or one column per user when
    ``users`` is None (shape (..., n)).
    """
    def policy(true_value, history, cfg, users=None, **_):
        v = np.asarray(true_value, dtype=np.float64)
        cum = history.cumulative if users is None else history.cumulative[np.asarray(users)]
        threshold = (history.cumulative.mean() if quantile is None
                     else np.quantile(history.cumulative, quantile))
        return np.where(cum <= threshold, np.minimum(c * v, cfg.v_max), v)[()]
    policy.needs_users = True
    policy.__name__ = f"timed_cap_{c:g}" + ("" if quantile is None else f"_q{round(100 * quantile)}")
    return policy


def call_policy(policy: Callable, values, history, cfg, users=None):
    """Invoke `policy`, passing ``users`` only to policies that declare ``needs_users``."""
    if getattr(policy, "needs_users", False):
        return policy(values, history, cfg, users=users)
    return policy(values, history, cfg)


# ── Policy registry ──────────────────────────────────────────────────────────

POLICY_REGISTRY: dict = {
    "truthful" : truthful,
    "cap_1.25" : make_capped(1.25),
    "cap_1.5"  : make_capped(1.5),
    "cap_2"    : make_capped(2.0),
    "max_claim": maximum_claim,
    "timed_cap_1.25": make_timed(1.25),
    "timed_cap_2"   : make_timed(2.0),
    # "rollout" needs a mechanism, RNG and opponent profile; see rollout_report
    # and sim.runner.run_rollout.
}


def get_policy(name: str) -> Callable:
    """Look up a policy by name; ``cap_<c>`` builds a capped policy on the fly."""
    if name in POLICY_REGISTRY:
        return POLICY_REGISTRY[name]
    if name.startswith("cap_"):
        return make_capped(float(name[4:]))
    if name.startswith("timed_cap_"):
        c, _, q = name[10:].partition("_q")
        return make_timed(float(c), None if not q else int(q) / 100)
    raise KeyError(f"Unknown policy '{name}'. Known: {sorted(POLICY_REGISTRY)}")
