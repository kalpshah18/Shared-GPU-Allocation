"""
sim/metrics.py
==============
All evaluation metrics — Owner: Raj Modi

Metrics are computed from the raw simulation output stored in a History object
and the true valuation tensor.

Notation (proposal section 2)
-----------------------------
n, k, T   : users, GPUs per round, rounds
A_i       : cumulative allocations of user i over T rounds
v[i, t]   : true valuation;  x[i, t] : allocation;  p[i, t] : payment
W*        : oracle welfare, the sum of the top-k true values each round
q_i(t)    : consecutive rounds with no service immediately BEFORE round t
            (q_i(0) = 0), as in the proposal
Delta     : starvation threshold = 2 * ceil(n / k)
"""

from __future__ import annotations

import numpy as np

from sim.config import Config
from sim.environment import History, expected_values


# ── Helpers: matrices built from a history ────────────────────────────────────

def _matrices(history: History) -> tuple[np.ndarray, np.ndarray]:
    """Allocation (n x T) and payment (n x T) matrices."""
    X = np.stack(history.allocations, axis=1)
    P = np.stack(history.payments,    axis=1)
    return X, P


def wait_matrix(X: np.ndarray) -> np.ndarray:
    """
    Pre-round waiting times q_i(t) for an allocation matrix X of shape (n, T):
    the number of consecutive rounds immediately before round t in which user
    i was not served (so q_i(0) = 0, and q_i(t) = 0 if i was served in round
    t-1).  Vectorised; shape (n, T).
    """
    n, T = X.shape
    idx = np.arange(T)
    served_at = np.where(X == 1, idx[None, :], -1)
    last_served = np.maximum.accumulate(served_at, axis=1)      # last service <= t
    last_before = np.concatenate(                                # last service <= t-1
        [np.full((n, 1), -1, dtype=last_served.dtype), last_served[:, :-1]], axis=1
    )
    # rounds since the last service before t; with no service yet (last = -1)
    # this equals t, the number of rounds played so far.
    return (idx[None, :] - 1 - last_before).astype(np.int64)


def utilities(history: History, valuations: np.ndarray) -> np.ndarray:
    """Per-user quasi-linear utility U_i = sum_t (v x - p), shape (n,)."""
    X, P = _matrices(history)
    return np.sum(valuations * X - P, axis=1)


def gross_benefits(history: History, valuations: np.ndarray) -> np.ndarray:
    """Per-user gross benefit G_i = sum_t v x (payments excluded), shape (n,)."""
    X, _ = _matrices(history)
    return np.sum(valuations * X, axis=1)


# ── Welfare ───────────────────────────────────────────────────────────────────

def oracle_welfare(valuations: np.ndarray, cfg: Config) -> float:
    """W* = sum_t (sum of the k largest true values in round t)."""
    return float(np.sum(np.sort(valuations, axis=0)[-cfg.k:, :]))


def welfare_ratio(history: History, valuations: np.ndarray, cfg: Config) -> float:
    """WR = W / W*; payments are transfers, so W uses gross benefits."""
    W      = float(np.sum(gross_benefits(history, valuations)))
    W_star = oracle_welfare(valuations, cfg)
    if W_star == 0.0:
        return 1.0                     # degenerate: every value is 0
    return W / W_star


def nash_social_welfare(history: History, valuations: np.ndarray, eps: float = 1e-8) -> float:
    """NSW = sum_i log(G_i + eps)."""
    return float(np.sum(np.log(gross_benefits(history, valuations) + eps)))


# ── Fairness ──────────────────────────────────────────────────────────────────

def _jain(a: np.ndarray) -> float:
    """Jain's index (sum a)^2 / (n sum a^2); 1.0 for the all-zero vector."""
    a = np.asarray(a, dtype=np.float64)
    ss = float(np.sum(a ** 2))
    if ss == 0.0:
        return 1.0
    return float(np.sum(a) ** 2 / (len(a) * ss))


def jain_allocation(history: History) -> float:
    """J_A over the cumulative allocation counts A_i."""
    return _jain(history.cumulative)


def jain_benefit(history: History, valuations: np.ndarray, mu: np.ndarray) -> float:
    """
    J_B over the normalised gross benefits B_i = G_i / (T mu_i).
    `mu` holds the per-user expected values E_{D_i}[v].
    """
    T = history.round
    G = gross_benefits(history, valuations)
    denom = T * np.asarray(mu, dtype=np.float64)
    B = np.where(denom > 0, G / np.where(denom > 0, denom, 1.0), 0.0)
    return _jain(B)


# ── Waiting & starvation ──────────────────────────────────────────────────────

def max_wait(history: History, valuations: np.ndarray | None = None) -> int:
    """Q_max = max_{i,t} q_i(t).  (`valuations` is accepted for backward compatibility.)"""
    X, _ = _matrices(history)
    return int(wait_matrix(X).max())


def starvation_rate(history: History, cfg: Config) -> float:
    """SR_Delta = (1 / nT) sum_{i,t} 1{q_i(t) > Delta}."""
    X, _ = _matrices(history)
    Q = wait_matrix(X)
    return float(np.mean(Q > cfg.delta))


def percentile_wait(history: History, pct: float = 95.0) -> float:
    """`pct`-th percentile of all per-user, per-round waiting times q_i(t)."""
    X, _ = _matrices(history)
    return float(np.percentile(wait_matrix(X), pct))


# ── Strategic manipulation ────────────────────────────────────────────────────

def manipulation_gain(
    history_strategic: History,
    history_truthful: History,
    valuations: np.ndarray,
    strategic_set: np.ndarray,
) -> dict:
    """
    Coalition manipulation gain for every strategic user i:

        M_i^coal = U_i(sigma_S, truthful_{-S}) - U_i(truthful)

    `history_strategic` is the mixed run (all of S deviate together) and
    `history_truthful` the all-truthful run, both on the same omega.  Strategic
    users compete with each other, so this is a *group* deviation: a negative
    value does NOT mean an individual is better off truthful (use
    ``unilateral_gain`` for that).

    Returns M_mean, M_max and frac_pos (fraction with M_i > 0) over S; all NaN
    when S is empty (rho = 0).
    """
    gains = (utilities(history_strategic, valuations)
             - utilities(history_truthful, valuations))[strategic_set]
    if len(gains) == 0:
        nan = float("nan")
        return {"M_mean": nan, "M_max": nan, "frac_pos": nan}
    return {
        "M_mean"  : float(np.mean(gains)),
        "M_max"   : float(np.max(gains)),
        "frac_pos": float(np.mean(gains > 0)),
    }


def unilateral_gain(
    history_deviate: History,
    history_truthful_focal: History,
    valuations: np.ndarray,
    focal: int,
) -> float:
    """
    M_f = U_f(sigma_f, sigma_{-f}) - U_f(truthful, sigma_{-f}).

    The two histories share omega and every other user's behaviour; only the
    focal user's report policy differs (see ``sim.runner.unilateral_gains``).
    """
    U_d = utilities(history_deviate,        valuations)[focal]
    U_t = utilities(history_truthful_focal, valuations)[focal]
    return float(U_d - U_t)


def unilateral_summary(gains) -> dict:
    """
    Summarise per-focal-user unilateral gains: mean (M_uni), max (M_uni_max) and
    the fraction of focal users with a strictly positive gain (frac_pos_uni).
    """
    g = np.asarray(gains, dtype=np.float64)
    return {
        "M_uni"       : float(np.mean(g)),
        "M_uni_max"   : float(np.max(g)),
        "frac_pos_uni": float(np.mean(g > 0)),
    }


def price_of_strategy(
    history_truthful: History,
    history_strategic: History,
    valuations: np.ndarray,
    cfg: Config | None = None,
) -> float:
    """PoS = (W_truthful - W_strategic) / W_truthful on a shared omega."""
    W_t = float(np.sum(gross_benefits(history_truthful,  valuations)))
    W_s = float(np.sum(gross_benefits(history_strategic, valuations)))
    if W_t == 0.0:
        return 0.0
    return (W_t - W_s) / W_t


def price_of_fairness(history: History, valuations: np.ndarray, cfg: Config) -> float:
    """PoF(F) = (W* - W_F) / W*: welfare loss vs. the truthful oracle."""
    W_star = oracle_welfare(valuations, cfg)
    if W_star == 0.0:
        return 0.0
    W_F = float(np.sum(gross_benefits(history, valuations)))
    return (W_star - W_F) / W_star


# ── Convenience: every metric computable from one run ─────────────────────────

def compute_all(
    history: History,
    valuations: np.ndarray,
    cfg: Config,
    mu: np.ndarray | None = None,
) -> dict:
    """
    All single-run metrics.  `mu` (per-user expected values) defaults to the
    configured distribution's means, so J_B is always reported.
    """
    if mu is None:
        mu = expected_values(cfg)
    X, _ = _matrices(history)
    Q = wait_matrix(X)
    return {
        "WR"        : welfare_ratio(history, valuations, cfg),
        "J_A"       : jain_allocation(history),
        "J_B"       : jain_benefit(history, valuations, mu),
        "NSW"       : nash_social_welfare(history, valuations),
        "Q_max"     : int(Q.max()),
        "SR_delta"  : float(np.mean(Q > cfg.delta)),
        "pct95_wait": float(np.percentile(Q, 95.0)),
        "PoF"       : price_of_fairness(history, valuations, cfg),
    }
