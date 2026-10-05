"""
tests/test_policies.py
=======================
Unit tests for sim/policies/strategic.py — Owner: Raj Modi
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from sim.config import Config
from sim.environment import History
from sim.policies import (
    truthful, capped_exaggeration, maximum_claim, POLICY_REGISTRY,
)

CFG = Config(n=6, k=2, T=10)


def _h():
    return History(CFG.n)


# ── Truthful ──────────────────────────────────────────────────────────────────

def test_truthful_returns_value():
    for v in [0.0, 0.5, 1.0]:
        assert truthful(v, _h(), CFG) == v


# ── Capped exaggeration ───────────────────────────────────────────────────────

def test_capped_exaggeration_below_cap():
    # c=2, v=0.3 → 2*0.3=0.6 < v_max=1.0 → report = 0.6
    assert abs(capped_exaggeration(0.3, _h(), CFG, c=2.0) - 0.6) < 1e-10


def test_capped_exaggeration_at_cap():
    # c=2, v=0.8 → 2*0.8=1.6 > v_max=1.0 → report = 1.0
    assert abs(capped_exaggeration(0.8, _h(), CFG, c=2.0) - 1.0) < 1e-10


def test_capped_exaggeration_all_c_values():
    for c in [1.25, 1.5, 2.0]:
        v   = 0.4
        r   = capped_exaggeration(v, _h(), CFG, c=c)
        assert r <= CFG.v_max + 1e-10
        assert r >= v  # exaggeration never reduces report below true value


# ── Maximum claim ─────────────────────────────────────────────────────────────

def test_maximum_claim():
    for v in [0.0, 0.3, 0.99]:
        assert abs(maximum_claim(v, _h(), CFG) - CFG.v_max) < 1e-10


# ── Policy registry ───────────────────────────────────────────────────────────

def test_registry_keys():
    expected = {"truthful", "cap_1.25", "cap_1.5", "cap_2", "max_claim"}
    assert expected.issubset(set(POLICY_REGISTRY.keys()))


def test_registry_truthful_callable():
    fn = POLICY_REGISTRY["truthful"]
    assert fn(0.7, _h(), CFG) == 0.7


# ── Report bounds ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("policy_name", ["truthful", "cap_1.25", "cap_1.5", "cap_2", "max_claim"])
def test_report_in_bounds(policy_name):
    fn  = POLICY_REGISTRY[policy_name]
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = float(rng.uniform(0, CFG.v_max))
        r = fn(v, _h(), CFG)
        assert 0.0 <= r <= CFG.v_max + 1e-10, f"{policy_name}: report {r} out of bounds"
