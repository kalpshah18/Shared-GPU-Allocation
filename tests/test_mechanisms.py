"""
tests/test_mechanisms.py
=========================
Unit tests for M1–M5 — Owner: Harsh Dhru

Covers all E0 checks as pytest cases, plus additional edge cases.
"""

import sys
sys.path.insert(0, ".")

import math
import numpy as np
import pytest

from sim.config import Config
from sim.environment import SeedPackage, History
from sim.mechanisms import (
    RandomMechanism, RoundRobinMechanism,
    GreedyMechanism, ScoreMechanism, VickreyMechanism,
)

CFG = Config(n=6, k=2, T=1)

def _pkg(seed=0):
    return SeedPackage.generate(seed, CFG)


# ── Capacity invariant ────────────────────────────────────────────────────────

@pytest.mark.parametrize("MechCls", [
    RandomMechanism, GreedyMechanism, ScoreMechanism, VickreyMechanism,
])
def test_capacity(MechCls):
    cfg  = Config(n=8, k=3, T=1)
    mech = MechCls(cfg)
    h    = History(cfg.n)
    rng  = np.random.default_rng(0)
    for t in range(10):
        r = rng.uniform(0, 1, cfg.n)
        x, _ = mech.allocate(r, h, t)
        assert int(x.sum()) == cfg.k, f"{MechCls.__name__} capacity failed"


def test_capacity_roundrobin():
    cfg  = Config(n=8, k=3, T=10)
    mech = RoundRobinMechanism(cfg, init_seed=0)
    h    = History(cfg.n)
    rng  = np.random.default_rng(0)
    for t in range(cfg.T):
        r = rng.uniform(0, 1, cfg.n)
        x, _ = mech.allocate(r, h, t)
        assert int(x.sum()) == cfg.k


# ── Payment invariant ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("MechCls", [
    RandomMechanism, RoundRobinMechanism, GreedyMechanism,
    ScoreMechanism, VickreyMechanism,
])
def test_payment_invariant(MechCls):
    cfg  = Config(n=6, k=2, T=1)
    mech = (MechCls(cfg) if MechCls != RoundRobinMechanism
            else RoundRobinMechanism(cfg, init_seed=0))
    h    = History(cfg.n)
    r    = np.array([0.8, 0.1, 0.5, 0.3, 0.9, 0.2])
    x, p = mech.allocate(r, h, 0)
    assert np.all(p[x == 0] == 0.0), "non-zero payment for unallocated user"


# ── Report-invariance (M1, M2) ────────────────────────────────────────────────

@pytest.mark.parametrize("MechCls", [RandomMechanism])
def test_report_invariance_m1(MechCls):
    cfg  = Config(n=6, k=2, T=1)
    mech = MechCls(cfg)
    h    = History(cfg.n)
    tie  = 42
    x1, _ = mech.allocate(np.zeros(cfg.n),          h, tie)
    x2, _ = mech.allocate(np.ones(cfg.n),           h, tie)
    x3, _ = mech.allocate(np.array([0.3,0.9,0.1,0.7,0.2,0.8]), h, tie)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(x1, x3)


def test_report_invariance_m2():
    cfg = Config(n=6, k=2, T=1)
    tie = 0
    h   = History(cfg.n)
    x1, _ = RoundRobinMechanism(cfg, init_seed=0).allocate(np.zeros(cfg.n), h, tie)
    x2, _ = RoundRobinMechanism(cfg, init_seed=0).allocate(np.ones(cfg.n),  h, tie)
    np.testing.assert_array_equal(x1, x2)


# ── M3 ≡ M4 at λ=0 ───────────────────────────────────────────────────────────

def test_m3_equals_m4_lambda_zero():
    cfg = Config(n=8, k=3, T=1, lambda_=0.0)
    h3  = History(cfg.n)
    h4  = History(cfg.n)
    rng = np.random.default_rng(0)
    for t in range(20):
        r = rng.uniform(0, 1, cfg.n)
        x3, _ = GreedyMechanism(cfg).allocate(r, h3, t)
        x4, _ = ScoreMechanism(cfg).allocate(r, h4, t)
        np.testing.assert_array_equal(x3, x4, err_msg=f"M3!=M4 at t={t}")


# ── Round-robin wait bound ────────────────────────────────────────────────────

@pytest.mark.parametrize("n,k", [(6, 2), (7, 3), (50, 10)])
def test_roundrobin_wait_bound(n, k):
    # Consecutive unserved rounds never exceed ⌈n/k⌉ − 1, checked every round.
    cfg  = Config(n=n, k=k, T=100)
    mech = RoundRobinMechanism(cfg, init_seed=7)
    h    = History(cfg.n)
    rng  = np.random.default_rng(7)
    bound = math.ceil(cfg.n / cfg.k) - 1
    for t in range(cfg.T):
        r = rng.uniform(0, 1, cfg.n)
        x, p = mech.allocate(r, h, t)
        h.update(x, p)
        assert h.consecutive_wait.max() <= bound


# ── M3 welfare oracle ─────────────────────────────────────────────────────────

def test_m3_welfare_oracle():
    cfg = Config(n=5, k=2, T=1)
    rng = np.random.default_rng(0)
    for trial in range(20):
        v    = rng.uniform(0, 1, cfg.n)
        h    = History(cfg.n)
        x, _ = GreedyMechanism(cfg).allocate(v, h, trial)
        W    = float((v * x).sum())
        Wstar = float(np.sort(v)[-cfg.k:].sum())
        assert abs(W - Wstar) < 1e-10, f"Oracle failed trial {trial}: W={W}, W*={Wstar}"


# ── M5 Vickrey payment structure ─────────────────────────────────────────────

def test_vickrey_payment_threshold():
    """Winners pay exactly the (k+1)-st highest bid."""
    cfg = Config(n=5, k=2, T=1)
    r   = np.array([0.9, 0.3, 0.7, 0.5, 0.1])
    h   = History(cfg.n)
    x, p = VickreyMechanism(cfg).allocate(r, h, 0)
    threshold = sorted(r, reverse=True)[cfg.k]   # 3rd highest = 0.5
    winners = np.where(x == 1)[0]
    for w in winners:
        assert abs(p[w] - threshold) < 1e-10, f"user {w} paid {p[w]}, expected {threshold}"


def test_vickrey_truthfulness_grid():
    """Truthful report should be utility-maximising on a coarse grid."""
    cfg  = Config(n=4, k=2, T=1)
    grid = [round(x * 0.1, 1) for x in range(11)]
    rng  = np.random.default_rng(77)

    for _ in range(10):
        v     = rng.uniform(0, 1, cfg.n)
        focal = rng.integers(0, cfg.n)

        def utility(r_focal: float) -> float:
            reports = v.copy()
            reports[focal] = r_focal
            h = History(cfg.n)
            x, p = VickreyMechanism(cfg).allocate(reports, h, 0)
            return float(v[focal] * x[focal] - p[focal])

        u_truth = utility(v[focal])
        u_best  = max(utility(r) for r in grid)
        assert u_truth >= u_best - 1e-9, f"Truthfulness violated: {u_truth:.4f} < {u_best:.4f}"
