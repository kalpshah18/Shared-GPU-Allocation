"""
tests/test_credit_vickrey.py

M5c Credit-Vickrey: hand-computed rounds, budget accounting, reduction to
Greedy with an unlimited budget, rollout consistency, and the renamed
strategic-welfare-loss metric.
"""

import copy

import numpy as np
import pytest

from sim import metrics as M
from sim.config import Config
from sim.environment import History, SeedPackage
from sim.mechanisms import CreditVickreyMechanism, make_mechanism
from sim.runner import run_mixed


def test_config_validation_and_derived_quota():
    cfg = Config(n=10, k=2, credit_rate=0.5, credit_horizon=10)
    mech = CreditVickreyMechanism(cfg)
    assert mech.income == pytest.approx(0.5 * 0.2 * 1.0)
    assert mech.ceiling == pytest.approx(10 * mech.income)
    np.testing.assert_allclose(mech.balance, mech.ceiling)
    for bad in (dict(credit_rate=0.0), dict(credit_horizon=0.5)):
        with pytest.raises(ValueError):
            Config(**bad)


def test_one_round_hand_computed():
    cfg = Config(n=4, k=1, credit_rate=1.0, credit_horizon=2)       # income 0.25, ceiling 0.5
    mech = CreditVickreyMechanism(cfg)
    x, p = mech.allocate(np.array([0.9, 0.3, 0.6, 0.1]), History(4), tie_seed=0)
    # bids are capped by the balance (0.5): [0.5, 0.3, 0.5, 0.1] -> tie between users 0 and 2
    assert x.sum() == 1 and x[[0, 2]].sum() == 1
    winner = int(np.flatnonzero(x)[0])
    np.testing.assert_allclose(p, 0.0)                               # credits are not utility
    assert mech.spent[winner] == pytest.approx(0.5)                  # (k+1)-st bid = 0.5
    assert mech.balance[winner] == pytest.approx(0.0)
    assert np.all(mech.balance[np.arange(4) != winner] == pytest.approx(0.5))


def test_budget_never_negative_and_bounded_by_income():
    cfg = Config(n=10, k=2, T=200, credit_rate=0.25)
    pkg = SeedPackage.generate(7, cfg)
    mech = make_mechanism("CreditVickreyMechanism", cfg, pkg)
    run_mixed(mech, pkg, cfg, None)
    assert np.all(mech.balance >= -1e-12) and np.all(mech.balance <= mech.ceiling + 1e-12)
    # total spending cannot exceed the starting balance plus all income
    assert mech.spent.sum() <= cfg.n * (mech.ceiling + cfg.T * mech.income) + 1e-9


def test_unlimited_budget_allocates_exactly_like_greedy():
    cfg = Config(n=12, k=3, T=150, credit_rate=1e6)
    pkg = SeedPackage.generate(3, cfg)
    h_c = run_mixed(make_mechanism("CreditVickreyMechanism", cfg, pkg), pkg, cfg, None)
    h_g = run_mixed(make_mechanism("GreedyMechanism", cfg, pkg), pkg, cfg, None)
    np.testing.assert_array_equal(h_c.cumulative, h_g.cumulative)
    assert M.welfare_ratio(h_c, pkg.valuations, cfg) == pytest.approx(M.welfare_ratio(h_g, pkg.valuations, cfg))


def test_tight_budget_spreads_service():
    cfg = Config(n=20, k=4, T=300, credit_rate=0.25)
    pkg = SeedPackage.generate(11, cfg)
    h_c = run_mixed(make_mechanism("CreditVickreyMechanism", cfg, pkg), pkg, cfg, None)
    h_g = run_mixed(make_mechanism("GreedyMechanism", cfg, pkg), pkg, cfg, None)
    assert M.jain_allocation(h_c) >= M.jain_allocation(h_g)


def test_rollout_step_matches_allocate():
    cfg = Config(n=6, k=2, credit_rate=0.5, credit_horizon=3)
    rng = np.random.default_rng(0)
    mech = CreditVickreyMechanism(cfg)
    twin = copy.deepcopy(mech)
    state = twin.rollout_init(History(6), 1)
    for t in range(20):
        reports = rng.random(6)
        tb = np.random.default_rng(t).random(6)
        x, _ = mech.allocate(reports, History(6), tie_seed=t)
        xr, _ = twin.rollout_step(reports[None, :], state, tb[None, :])
        np.testing.assert_array_equal(x, xr[0])
        np.testing.assert_allclose(mech.balance, state["bal"][0])


def test_strategic_welfare_loss_alias():
    assert M.price_of_strategy is M.strategic_welfare_loss
