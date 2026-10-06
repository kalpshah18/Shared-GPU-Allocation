"""
tests/test_runner.py
====================
Unit tests for sim/runner.py and NaN handling in analysis/bootstrap.py.
"""

import sys
sys.path.insert(0, ".")

import numpy as np

from analysis.bootstrap import summarise_seeds
from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import GreedyMechanism, RoundRobinMechanism
from sim.policies import maximum_claim, truthful
from sim.runner import focal_user, run_paired, run_single, run_unilateral


def test_run_paired_resets_stateful_mechanism():
    # n/k does not divide T*k evenly, so a reused queue would start shifted.
    cfg = Config(n=7, k=2, T=5, rho=0.3)
    pkg = SeedPackage.generate(123, cfg)
    mech = RoundRobinMechanism(cfg, init_seed=int(pkg.tie_seeds[0]))
    h_truth, h_strat = run_paired(mech, pkg, cfg, maximum_claim)
    # M2 is report-invariant, so both runs must allocate identically.
    for a, b in zip(h_truth.allocations, h_strat.allocations):
        assert np.array_equal(a, b)


def test_run_unilateral_truthful_policy_is_zero_gain():
    cfg = Config(n=10, k=3, T=20, rho=0.3)
    pkg = SeedPackage.generate(7, cfg)
    h_dev, h_tru, f = run_unilateral(GreedyMechanism(cfg), pkg, cfg, truthful)
    assert f == focal_user(pkg)
    for a, b in zip(h_dev.allocations, h_tru.allocations):
        assert np.array_equal(a, b)


def test_run_unilateral_rho_zero_matches_single_deviator():
    cfg = Config(n=10, k=3, T=20, rho=0.0)
    pkg = SeedPackage.generate(7, cfg)
    h_dev, h_tru, f = run_unilateral(GreedyMechanism(cfg), pkg, cfg, maximum_claim)
    assert f == 0
    # Baseline run is fully truthful.
    h_ref = run_single(GreedyMechanism(cfg), pkg, cfg)
    for a, b in zip(h_tru.allocations, h_ref.allocations):
        assert np.array_equal(a, b)
    # A max-claimer under Greedy wins every round.
    assert h_dev.cumulative[0] == cfg.T


def test_summarise_seeds_drops_nan_and_returns_none_when_empty():
    rows = [{"a": 1.0, "b": float("nan")}, {"a": float("nan"), "b": float("nan")},
            {"a": 3.0, "b": float("nan")}]
    s = summarise_seeds(rows, ["a", "b"], n_resamples=100)
    assert s["a"]["mean"] == 2.0
    assert s["b"] == {"mean": None, "ci_lower": None, "ci_upper": None, "std": None}
