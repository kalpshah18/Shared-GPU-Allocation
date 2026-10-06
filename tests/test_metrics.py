"""
tests/test_metrics.py
======================
Unit tests for sim/metrics.py — Owner: Raj Modi

Validates all metrics on hand-computed 2-user, 2-round examples.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from sim.config import Config
from sim.environment import History
from sim import metrics as M


def _make_history(n: int, allocations: list, payments: list | None = None) -> History:
    """Build a History from explicit allocation and payment lists."""
    h = History(n)
    p_zero = np.zeros(n, dtype=np.float64)
    for i, x in enumerate(allocations):
        p = np.array(payments[i], dtype=np.float64) if payments else p_zero.copy()
        h.update(np.array(x, dtype=np.int64), p)
    return h


# ── 2-user, 2-round example ───────────────────────────────────────────────────
#
#  n=2, k=1, T=2
#  allocations: round 0 → user 0 gets GPU, round 1 → user 1 gets GPU
#  valuations:  [[0.8, 0.3], [0.4, 0.9]]
#               user 0: v=(0.8, 0.3), user 1: v=(0.4, 0.9)

CFG = Config(n=2, k=1, T=2)
ALLOC = [[1, 0], [0, 1]]          # shape: T × n (list of length n)
VAL   = np.array([[0.8, 0.3],
                  [0.4, 0.9]])    # shape: n × T


def _h():
    return _make_history(CFG.n, ALLOC)


# ── Welfare Ratio ─────────────────────────────────────────────────────────────

def test_welfare_ratio():
    h = _h()
    # W  = 0.8*1 + 0.3*0 + 0.4*0 + 0.9*1 = 1.7
    # W* = max(0.8,0.4) + max(0.3,0.9) = 0.8 + 0.9 = 1.7
    # WR = 1.0
    wr = M.welfare_ratio(h, VAL, CFG)
    assert abs(wr - 1.0) < 1e-10


def test_welfare_ratio_suboptimal():
    # Swap allocations: user 1 gets round 0, user 0 gets round 1
    h = _make_history(2, [[0, 1], [1, 0]])
    # W  = 0.4 + 0.3 = 0.7;  W* = 0.8 + 0.9 = 1.7
    wr = M.welfare_ratio(h, VAL, CFG)
    assert abs(wr - 0.7 / 1.7) < 1e-10


# ── Jain Allocation Fairness ──────────────────────────────────────────────────

def test_jain_allocation_equal():
    h = _h()
    # A_0 = 1, A_1 = 1  → J_A = (1+1)^2 / (2*(1+1)) = 4/4 = 1.0
    assert abs(M.jain_allocation(h) - 1.0) < 1e-10


def test_jain_allocation_unequal():
    # user 0 gets both rounds
    h = _make_history(2, [[1, 0], [1, 0]])
    # A_0=2, A_1=0 → J_A = 4 / (2*4) = 0.5
    assert abs(M.jain_allocation(h) - 0.5) < 1e-10


# ── Nash Social Welfare ───────────────────────────────────────────────────────

def test_nsw():
    h = _h()
    # G_0 = 0.8, G_1 = 0.9
    # NSW = log(0.8 + 1e-8) + log(0.9 + 1e-8)
    expected = np.log(0.8 + 1e-8) + np.log(0.9 + 1e-8)
    assert abs(M.nash_social_welfare(h, VAL) - expected) < 1e-8


# ── Starvation Rate ───────────────────────────────────────────────────────────

def test_starvation_rate_zero():
    # With k=1, n=2, T=2 and alternating: no user ever exceeds Δ=2*ceil(2/1)=4
    h   = _h()
    cfg = Config(n=2, k=1, T=2)
    assert M.starvation_rate(h, cfg) == 0.0


# ── Manipulation Gain ─────────────────────────────────────────────────────────

def test_manipulation_gain_truthful():
    # If strategic and truthful histories are identical, gain = 0
    h1 = _h()
    h2 = _h()
    strat_set = np.array([0])
    result = M.manipulation_gain(h1, h2, VAL, strat_set)
    assert abs(result["M_mean"]) < 1e-10
    assert abs(result["M_max"])  < 1e-10
    assert result["frac_pos"]    == 0.0


def test_manipulation_gain_empty_strategic_set_is_nan():
    result = M.manipulation_gain(_h(), _h(), VAL, np.array([], dtype=np.int64))
    assert np.isnan(result["M_mean"])


def test_unilateral_gain_hand_computed():
    # Deviating run: user 0 wins both rounds and pays 0.5 in round 1.
    # Truthful run: user 0 wins round 0 only (ALLOC).
    h_dev = _make_history(2, [[1, 0], [1, 0]], [[0.0, 0.0], [0.5, 0.0]])
    h_tru = _h()
    # U_dev = 0.8 + 0.3 - 0.5 = 0.6 ; U_tru = 0.8  ->  gain = -0.2
    assert abs(M.unilateral_gain(h_dev, h_tru, VAL, focal=0) - (-0.2)) < 1e-12


# ── Price of Strategy ─────────────────────────────────────────────────────────

def test_price_of_strategy_zero():
    h1 = _h()
    h2 = _h()
    pos = M.price_of_strategy(h1, h2, VAL, CFG)
    assert abs(pos) < 1e-10


# ── Price of Fairness ─────────────────────────────────────────────────────────

def test_price_of_fairness_zero():
    # Oracle mechanism → PoF = 0
    h   = _h()
    pof = M.price_of_fairness(h, VAL, CFG)
    assert abs(pof) < 1e-10


def test_price_of_fairness_positive():
    # Suboptimal allocation → PoF > 0
    h   = _make_history(2, [[0, 1], [1, 0]])
    pof = M.price_of_fairness(h, VAL, CFG)
    # W* = 1.7, W = 0.7  → PoF = 1.0/1.7
    assert abs(pof - 1.0 / 1.7) < 1e-10
