"""
sim/metrics.py
==============
All evaluation metrics — Owner: Raj Modi

Metrics are computed from the raw simulation output stored in a History object
and the true valuation tensor.

Notation (from proposal §2)
----------------------------
n         : number of users
k         : GPUs per round
T         : number of rounds
A_i       : cumulative allocations of user i over T rounds (= history.cumulative after run)
v[i, t]   : true valuation of user i in round t
x[i, t]   : allocation of user i in round t (0 or 1)
p[i, t]   : payment of user i in round t
W*        : optimal (oracle) welfare — sum of top-k true values each round
Δ         : starvation threshold = 2 * ceil(n / k)

All metric functions accept History and the valuation tensor v (n×T ndarray)
and return a scalar or small dict.
"""

from __future__ import annotations

import math

import numpy as np

from sim.config import Config
from sim.environment import History


# ── Helper: build allocation/payment matrices from history ────────────────────

def _matrices(history: History) -> tuple[np.ndarray, np.ndarray]:
    """Return allocation (n×T) and payment (n×T) matrices."""
    X = np.stack(history.allocations, axis=1)  # shape (n, T)
    P = np.stack(history.payments,    axis=1)
    return X, P


# ── Welfare ───────────────────────────────────────────────────────────────────

def welfare_ratio(history: History, valuations: np.ndarray, cfg: Config) -> float:
    """
    WR = W / W*

    W  = Σ_{i,t} v_{i,t} · x_{i,t}       (achieved gross welfare)
    W* = Σ_t  Σ_{i ∈ S*_t} v_{i,t}        (oracle: top-k true values each round)
    """
    X, _ = _matrices(history)
    W = float(np.sum(valuations * X))

    # Oracle: for each round, sum the k largest true values
    W_star = float(np.sum(np.sort(valuations, axis=0)[-cfg.k:, :]))

    if W_star == 0.0:
        return 1.0  # degenerate case: all values are 0
    return W / W_star


def oracle_welfare(valuations: np.ndarray, cfg: Config) -> float:
    """W* — used by PoF."""
    return float(np.sum(np.sort(valuations, axis=0)[-cfg.k:, :]))


def nash_social_welfare(history: History, valuations: np.ndarray, eps: float = 1e-8) -> float:
    """
    NSW = Σ_i log(G_i + ε),  ε = 1e-8

    G_i = Σ_t v_{i,t} · x_{i,t}  (gross benefit for user i)
    """
    X, _ = _matrices(history)
    G = np.sum(valuations * X, axis=1)  # shape (n,)
    return float(np.sum(np.log(G + eps)))


# ── Fairness ──────────────────────────────────────────────────────────────────

def jain_allocation(history: History) -> float:
    """
    J_A = (Σ_i A_i)² / (n · Σ_i A_i²)

    where A_i = cumulative allocations of user i over T rounds.
    J_A ∈ (0, 1], equals 1 iff all A_i are equal.
    """
    A = history.cumulative.astype(np.float64)
    if np.all(A == 0):
        return 1.0  # no allocations made; vacuously fair
    return float(np.sum(A) ** 2 / (len(A) * np.sum(A ** 2)))


def jain_benefit(
    history: History,
    valuations: np.ndarray,
    mu: np.ndarray,
) -> float:
    """
    J_B = (Σ_i B_i)² / (n · Σ_i B_i²)

    B_i = G_i / (T · μ_i)  where μ_i = E[v_{i,·}] (per-user expected value)

    Used when user distributions differ (experiment E4).
    mu : ndarray, shape (n,) — per-user expected values, pre-computed from D_i.
    """
    X, _ = _matrices(history)
    T = history.round
    G = np.sum(valuations * X, axis=1)
    # Avoid division by zero for users with μ_i = 0
    denom = T * mu
    B = np.where(denom > 0, G / denom, 0.0)
    if np.all(B == 0):
        return 1.0
    return float(np.sum(B) ** 2 / (len(B) * np.sum(B ** 2)))


# ── Waiting & starvation ──────────────────────────────────────────────────────

def max_wait(history: History, valuations: np.ndarray) -> int:
    """
    Q_max = max_{i, t} q_i(t)

    Reconstructed from the allocation matrix (post-run).
    """
    X, _ = _matrices(history)
    n, T = X.shape
    q_max = 0
    current_wait = np.zeros(n, dtype=np.int64)
    for t in range(T):
        current_wait = np.where(X[:, t] == 1, 0, current_wait + 1)
        q_max = max(q_max, int(current_wait.max()))
    return q_max


def starvation_rate(history: History, cfg: Config) -> float:
    """
    SR_Δ = (1 / nT) Σ_{i,t} 1{q_i(t) > Δ}

    Δ = 2 · ⌈n/k⌉  (twice the round-robin service cycle).
    """
    X, _ = _matrices(history)
    n, T = X.shape
    delta = cfg.delta
    count = 0
    current_wait = np.zeros(n, dtype=np.int64)
    for t in range(T):
        current_wait = np.where(X[:, t] == 1, 0, current_wait + 1)
        count += int(np.sum(current_wait > delta))
    return count / (n * T)


def percentile_wait(history: History, pct: float = 95.0) -> float:
    """Return the `pct`-th percentile of all per-user per-round waiting times."""
    X, _ = _matrices(history)
    n, T = X.shape
    waits = []
    current_wait = np.zeros(n, dtype=np.int64)
    for t in range(T):
        current_wait = np.where(X[:, t] == 1, 0, current_wait + 1)
        waits.extend(current_wait.tolist())
    return float(np.percentile(waits, pct))


# ── Strategic manipulation ────────────────────────────────────────────────────

def manipulation_gain(
    history_strategic: History,
    history_truthful: History,
    valuations: np.ndarray,
    strategic_set: np.ndarray,
) -> dict:
    """
    Coalition manipulation gain for each strategic user i:

        M_i^coal(ω) = U_i(σ_S, truthful_{-S}; ω) − U_i(truthful; ω)

    where U_i = Σ_t (v_{i,t} · x_{i,t} − p_{i,t}) and S is the strategic set.

    `history_strategic` is the mixed run (all of S deviate together) and
    `history_truthful` is the all-truthful run, both on the same ω.  This is a
    *group* deviation: strategic users compete with each other, so a negative
    value does NOT imply that an individual user is better off truthful.  Use
    `unilateral_gain` for the individual incentive M_i(σ_i, σ_{-i}; ω).

    Returns
    -------
    dict with keys:
        M_mean   : mean manipulation gain over strategic users
        M_max    : maximum manipulation gain over strategic users
        frac_pos : fraction of strategic users with M_i > 0
    """
    Xs, Ps = _matrices(history_strategic)
    Xt, Pt = _matrices(history_truthful)

    U_strat   = np.sum(valuations * Xs - Ps, axis=1)
    U_truth   = np.sum(valuations * Xt - Pt, axis=1)
    gains     = (U_strat - U_truth)[strategic_set]

    if len(gains) == 0:
        # No strategic users (ρ = 0): the coalition gain is undefined.
        return {"M_mean": float("nan"), "M_max": float("nan"), "frac_pos": float("nan")}
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
    M_f(σ_f; ω) = U_f(σ_f, σ_{-f}; ω) − U_f(truthful, σ_{-f}; ω)

    Histories come from `sim.runner.run_unilateral`: identical ω and opponent
    behaviour; only the focal user's report policy differs.
    """
    Xd, Pd = _matrices(history_deviate)
    Xt, Pt = _matrices(history_truthful_focal)
    U_d = float(np.sum(valuations[focal] * Xd[focal] - Pd[focal]))
    U_t = float(np.sum(valuations[focal] * Xt[focal] - Pt[focal]))
    return U_d - U_t


def price_of_strategy(
    history_truthful: History,
    history_strategic: History,
    valuations: np.ndarray,
    cfg: Config,
) -> float:
    """
    PoS = (W_truthful − W_strategic) / W_truthful

    Both histories share the same ω.
    """
    Xt, _ = _matrices(history_truthful)
    Xs, _ = _matrices(history_strategic)
    W_t = float(np.sum(valuations * Xt))
    W_s = float(np.sum(valuations * Xs))
    if W_t == 0.0:
        return 0.0
    return (W_t - W_s) / W_t


def price_of_fairness(
    history: History,
    valuations: np.ndarray,
    cfg: Config,
) -> float:
    """
    PoF(F) = (W* − W_F) / W*

    Welfare loss of mechanism F under truthful reports relative to oracle.
    """
    W_star = oracle_welfare(valuations, cfg)
    X, _ = _matrices(history)
    W_F = float(np.sum(valuations * X))
    if W_star == 0.0:
        return 0.0
    return (W_star - W_F) / W_star


# ── Convenience: compute all scalable metrics in one call ─────────────────────

def compute_all(
    history: History,
    valuations: np.ndarray,
    cfg: Config,
    mu: np.ndarray | None = None,
) -> dict:
    """
    Return a dict of all metrics computable from a single simulation run
    (no paired comparison needed).

    `mu` is required for J_B; if None, J_B is omitted.
    """
    result = {
        "WR"     : welfare_ratio(history, valuations, cfg),
        "J_A"    : jain_allocation(history),
        "NSW"    : nash_social_welfare(history, valuations),
        "Q_max"  : max_wait(history, valuations),
        "SR_delta": starvation_rate(history, cfg),
        "pct95_wait": percentile_wait(history, 95.0),
        "PoF"    : price_of_fairness(history, valuations, cfg),
    }
    if mu is not None:
        result["J_B"] = jain_benefit(history, valuations, mu)
    return result
