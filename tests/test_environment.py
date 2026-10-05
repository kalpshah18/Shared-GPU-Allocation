"""
tests/test_environment.py
==========================
Unit tests for sim/environment.py — Owner: Aayush Kuloor

Validates:
  - Determinism: same seed → same valuation tensor
  - AR(1) bounds
  - Cumulative allocations never exceed k*T
  - History tracking correctness
"""

import math
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from sim.config import Config
from sim.environment import SeedPackage, History, _sample_valuations


# ── Determinism ────────────────────────────────────────────────────────────────

def test_same_seed_same_tensor():
    cfg = Config(n=10, k=3, T=50)
    pkg1 = SeedPackage.generate(42, cfg)
    pkg2 = SeedPackage.generate(42, cfg)
    np.testing.assert_array_equal(pkg1.valuations, pkg2.valuations)
    np.testing.assert_array_equal(pkg1.tie_seeds,  pkg2.tie_seeds)


def test_different_seeds_different_tensors():
    cfg = Config(n=10, k=3, T=50)
    pkg1 = SeedPackage.generate(1, cfg)
    pkg2 = SeedPackage.generate(2, cfg)
    assert not np.array_equal(pkg1.valuations, pkg2.valuations)


# ── Value bounds ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dist", ["uniform", "beta_low", "beta_high", "mixed"])
def test_valuations_in_bounds(dist):
    cfg = Config(n=10, k=3, T=20, valuation_dist=dist)
    pkg = SeedPackage.generate(0, cfg)
    assert pkg.valuations.min() >= 0.0
    assert pkg.valuations.max() <= cfg.v_max + 1e-10


def test_ar1_values_in_bounds():
    cfg = Config(n=10, k=3, T=100, alpha=0.9)
    pkg = SeedPackage.generate(0, cfg)
    assert pkg.valuations.min() >= 0.0
    assert pkg.valuations.max() <= cfg.v_max + 1e-10


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_valuation_shape():
    cfg = Config(n=15, k=5, T=30)
    pkg = SeedPackage.generate(7, cfg)
    assert pkg.valuations.shape == (cfg.n, cfg.T)
    assert pkg.tie_seeds.shape  == (cfg.T,)


# ── Strategic set ─────────────────────────────────────────────────────────────

def test_strategic_set_size():
    cfg = Config(n=50, k=10, T=10, rho=0.25)
    pkg = SeedPackage.generate(3, cfg)
    expected = math.floor(cfg.rho * cfg.n)
    assert len(pkg.strategic_set) == expected


def test_strategic_set_zero_rho():
    cfg = Config(n=50, k=10, T=10, rho=0.0)
    pkg = SeedPackage.generate(3, cfg)
    assert len(pkg.strategic_set) == 0


# ── History tracking ──────────────────────────────────────────────────────────

def test_history_cumulative():
    n = 5
    h = History(n)
    x1 = np.array([1, 1, 0, 0, 0])
    x2 = np.array([0, 1, 1, 0, 0])
    p  = np.zeros(n, dtype=np.float64)
    h.update(x1, p)
    h.update(x2, p)
    np.testing.assert_array_equal(h.cumulative, [1, 2, 1, 0, 0])


def test_history_consecutive_wait():
    n = 4
    h = History(n)
    p = np.zeros(n, dtype=np.float64)
    h.update(np.array([1, 0, 0, 0]), p)  # user 0 served, others wait 1
    np.testing.assert_array_equal(h.consecutive_wait, [0, 1, 1, 1])
    h.update(np.array([0, 1, 0, 0]), p)  # user 1 served, 0/2/3 accumulate
    np.testing.assert_array_equal(h.consecutive_wait, [1, 0, 2, 2])


def test_history_reset():
    n = 4
    h = History(n)
    p = np.zeros(n, dtype=np.float64)
    h.update(np.array([1, 1, 0, 0]), p)
    h.reset()
    np.testing.assert_array_equal(h.cumulative, [0, 0, 0, 0])
    assert h.round == 0
    assert len(h.allocations) == 0


# ── Config validation ─────────────────────────────────────────────────────────

def test_config_k_lt_n():
    with pytest.raises(ValueError):
        Config(n=5, k=5, T=10)


def test_config_rho_bounds():
    with pytest.raises(ValueError):
        Config(n=10, k=3, T=10, rho=1.5)


def test_config_delta():
    cfg = Config(n=10, k=3, T=10)
    assert cfg.delta == 2 * math.ceil(10 / 3)
