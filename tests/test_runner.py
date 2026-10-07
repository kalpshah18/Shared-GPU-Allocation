"""
tests/test_runner.py — Owner: Aayush Kuloor

Round-loop runners: mixed populations, the paired design, unilateral gains and
the rollout runner.
"""

import copy

import numpy as np
import pytest

from sim import metrics as M
from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import (
    GreedyMechanism, RoundRobinMechanism, ScoreMechanism, VickreyMechanism, make_mechanism,
)
from sim.policies import make_capped, maximum_claim, truthful
from sim.runner import (
    focal_users, run_mixed, run_paired, run_rollout, run_single, unilateral_gains,
)


def U(h, pkg):
    return M.utilities(h, pkg.valuations)


# ── run_single / run_mixed ────────────────────────────────────────────────────

def test_run_single_shapes_and_round_count():
    cfg = Config(n=8, k=3, T=25)
    pkg = SeedPackage.generate(1, cfg)
    h = run_single(GreedyMechanism(cfg), pkg, cfg)
    assert h.round == 25 and len(h.allocations) == len(h.payments) == 25
    assert h.cumulative.sum() == 3 * 25


def test_none_policy_is_truthful_everywhere():
    cfg = Config(n=8, k=3, T=25, rho=0.5)
    pkg = SeedPackage.generate(1, cfg)
    ref = run_single(GreedyMechanism(cfg), pkg, cfg, policy_fn=None)
    for h in (run_single(GreedyMechanism(cfg), pkg, cfg, policy_fn=truthful),
              run_mixed(GreedyMechanism(cfg), pkg, cfg, None),
              run_mixed(GreedyMechanism(cfg), pkg, cfg, truthful)):
        np.testing.assert_array_equal(np.stack(h.allocations), np.stack(ref.allocations))


def test_run_mixed_only_strategic_users_deviate():
    cfg = Config(n=6, k=2, T=5, rho=0.5)
    pkg = SeedPackage.generate(2, cfg)
    seen = []

    class Spy(GreedyMechanism):
        def allocate(self, reports, history, tie_seed):
            seen.append(reports.copy())
            return super().allocate(reports, history, tie_seed)

    run_mixed(Spy(cfg), pkg, cfg, maximum_claim)
    S = set(pkg.strategic_set.tolist())
    for t, rep in enumerate(seen):
        for i in range(cfg.n):
            if i in S:
                assert rep[i] == cfg.v_max
            else:
                assert rep[i] == pkg.valuations[i, t]


def test_run_mixed_explicit_set_overrides_pkg_set():
    cfg = Config(n=6, k=2, T=3, rho=0.0)
    pkg = SeedPackage.generate(2, cfg)
    h = run_mixed(GreedyMechanism(cfg), pkg, cfg, maximum_claim, np.array([4]))
    assert h.cumulative[4] == cfg.T                          # always reports v_max -> always wins (k=2)


def test_run_mixed_empty_strategic_set_equals_truthful():
    cfg = Config(n=6, k=2, T=20, rho=0.0)
    pkg = SeedPackage.generate(2, cfg)
    a = run_mixed(GreedyMechanism(cfg), pkg, cfg, maximum_claim)             # rho = 0 -> empty set
    b = run_mixed(GreedyMechanism(cfg), pkg, cfg, None)
    np.testing.assert_array_equal(np.stack(a.allocations), np.stack(b.allocations))


@pytest.mark.parametrize("bad", [lambda v, h, c: v + 5.0, lambda v, h, c: v * 0 - 0.1,
                                 lambda v, h, c: v * np.nan])
def test_out_of_range_reports_are_rejected(bad):
    cfg = Config(n=6, k=2, T=3, rho=1.0)
    pkg = SeedPackage.generate(2, cfg)
    with pytest.raises(ValueError):
        run_mixed(GreedyMechanism(cfg), pkg, cfg, bad)


def test_runs_do_not_mutate_the_seed_package():
    cfg = Config(n=8, k=2, T=15, rho=0.5)
    pkg = SeedPackage.generate(3, cfg)
    snap = (pkg.valuations.copy(), pkg.tie_seeds.copy(), pkg.strategic_set.copy())
    run_paired(GreedyMechanism(cfg), pkg, cfg, maximum_claim)
    for a, b in zip(snap, (pkg.valuations, pkg.tie_seeds, pkg.strategic_set)):
        np.testing.assert_array_equal(a, b)


# ── run_paired ────────────────────────────────────────────────────────────────

def test_paired_truthful_run_is_the_all_truthful_baseline():
    cfg = Config(n=10, k=3, T=30, rho=0.4)
    pkg = SeedPackage.generate(4, cfg)
    h_truth, h_strat = run_paired(GreedyMechanism(cfg), pkg, cfg, maximum_claim)
    ref = run_single(GreedyMechanism(cfg), pkg, cfg)
    np.testing.assert_array_equal(np.stack(h_truth.allocations), np.stack(ref.allocations))
    assert not np.array_equal(np.stack(h_truth.allocations), np.stack(h_strat.allocations))


def test_paired_resets_stateful_mechanism():
    cfg = Config(n=7, k=2, T=5, rho=0.3)                     # n/k not integral: reused queue would drift
    pkg = SeedPackage.generate(123, cfg)
    mech = RoundRobinMechanism(cfg, init_seed=int(pkg.tie_seeds[0]))
    h_truth, h_strat = run_paired(mech, pkg, cfg, maximum_claim)
    for a, b in zip(h_truth.allocations, h_strat.allocations):
        np.testing.assert_array_equal(a, b)                  # M2 ignores reports
    assert mech._pos == 0                                    # the passed instance itself is untouched


def test_report_invariant_mechanisms_have_zero_manipulation_gain():
    cfg = Config(n=12, k=3, T=40, rho=0.5)
    pkg = SeedPackage.generate(5, cfg)
    for name in ("RandomMechanism", "RoundRobinMechanism"):
        mech = make_mechanism(name, cfg, pkg)
        h_t, h_s = run_paired(mech, pkg, cfg, maximum_claim)
        g = M.manipulation_gain(h_s, h_t, pkg.valuations, pkg.strategic_set)
        assert g["M_mean"] == 0.0 and g["M_max"] == 0.0 and g["frac_pos"] == 0.0
        _, gains = unilateral_gains(mech, pkg, cfg, maximum_claim, h_s)
        np.testing.assert_array_equal(gains, 0.0)


# ── focal users & unilateral gains ────────────────────────────────────────────

def test_focal_users_come_from_the_strategic_set():
    cfg = Config(n=40, k=8, T=3, rho=0.5, n_focal=5)
    pkg = SeedPackage.generate(1, cfg)
    f = focal_users(pkg, cfg)
    assert len(f) == 5 and set(f) <= set(pkg.strategic_set.tolist())


def test_focal_users_capped_by_strategic_set_size():
    cfg = Config(n=40, k=8, T=3, rho=0.05, n_focal=5)         # only 2 strategic users
    assert len(focal_users(SeedPackage.generate(1, cfg), cfg)) == 2


def test_focal_users_at_rho_zero_are_spread_and_unique():
    cfg = Config(n=40, k=8, T=3, rho=0.0, n_focal=5)
    f = focal_users(SeedPackage.generate(1, cfg), cfg)
    assert len(f) == len(set(f.tolist())) == 5 and f.min() == 0 and f.max() == 39


def test_unilateral_gain_is_zero_for_the_truthful_policy():
    cfg = Config(n=10, k=3, T=20, rho=0.3)
    pkg = SeedPackage.generate(7, cfg)
    _, g = unilateral_gains(GreedyMechanism(cfg), pkg, cfg, truthful)
    np.testing.assert_allclose(g, 0.0)


def test_unilateral_gain_matches_manual_counterfactual_for_a_member_of_S():
    cfg = Config(n=12, k=3, T=40, rho=0.5, n_focal=3)
    pkg = SeedPackage.generate(8, cfg)
    pol = make_capped(2.0)
    focals, gains = unilateral_gains(GreedyMechanism(cfg), pkg, cfg, pol)
    S = pkg.strategic_set
    for f, g in zip(focals, gains):
        with_f = run_mixed(GreedyMechanism(cfg), pkg, cfg, pol, S)
        without_f = run_mixed(GreedyMechanism(cfg), pkg, cfg, pol, S[S != f])
        assert g == pytest.approx(U(with_f, pkg)[f] - U(without_f, pkg)[f])


def test_unilateral_gain_for_a_lone_deviator_at_rho_zero():
    cfg = Config(n=10, k=3, T=60, rho=0.0, n_focal=2)
    pkg = SeedPackage.generate(7, cfg)
    focals, gains = unilateral_gains(GreedyMechanism(cfg), pkg, cfg, maximum_claim)
    base = run_single(GreedyMechanism(cfg), pkg, cfg)
    for f, g in zip(focals, gains):
        solo = run_mixed(GreedyMechanism(cfg), pkg, cfg, maximum_claim, np.array([f]))
        assert solo.cumulative[f] == cfg.T                   # a max-claimer always wins under Greedy
        assert g == pytest.approx(U(solo, pkg)[f] - U(base, pkg)[f])
        assert g > 0


def test_greedy_rewards_and_vickrey_punishes_inflation():
    cfg = Config(n=20, k=4, T=100, rho=0.25)
    pkg = SeedPackage.generate(9, cfg)
    pol = make_capped(2.0)
    _, g_greedy = unilateral_gains(GreedyMechanism(cfg), pkg, cfg, pol)
    _, g_vick = unilateral_gains(VickreyMechanism(cfg), pkg, cfg, pol)
    assert np.all(g_greedy > 0) and np.all(g_vick < 0)


def test_unilateral_reuses_a_supplied_shared_run():
    cfg = Config(n=12, k=3, T=30, rho=0.5, n_focal=2)
    pkg = SeedPackage.generate(8, cfg)
    pol = make_capped(2.0)
    _, h_strat = run_paired(ScoreMechanism(cfg), pkg, cfg, pol)
    f1, g1 = unilateral_gains(ScoreMechanism(cfg), pkg, cfg, pol, h_strat)
    f2, g2 = unilateral_gains(ScoreMechanism(cfg), pkg, cfg, pol)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_allclose(g1, g2)


# ── rollout runner ────────────────────────────────────────────────────────────

def test_run_rollout_shapes_bounds_and_determinism():
    cfg = Config(n=10, k=2, T=15, rho=0.0)
    pkg = SeedPackage.generate(3, cfg)
    outs = [run_rollout(VickreyMechanism(cfg), pkg, cfg, 0, None, np.random.default_rng(1),
                        n_rollouts=20) for _ in range(2)]
    (h1, r1), (h2, r2) = outs
    assert h1.round == 15 and r1.shape == (15,)
    assert np.all((r1 >= 0) & (r1 <= cfg.v_max))
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(np.stack(h1.allocations), np.stack(h2.allocations))


def test_run_rollout_greedy_attacker_wins_far_more_than_its_share():
    cfg = Config(n=10, k=2, T=40, rho=0.0)
    pkg = SeedPackage.generate(3, cfg)
    h, reports = run_rollout(GreedyMechanism(cfg), pkg, cfg, 0, None, np.random.default_rng(0),
                             n_rollouts=50)
    base = run_single(GreedyMechanism(cfg), pkg, cfg)
    assert h.cumulative[0] > base.cumulative[0]
    assert reports.mean() > pkg.valuations[0].mean()


def test_run_rollout_leaves_other_users_on_the_opponent_policy():
    cfg = Config(n=6, k=2, T=5, rho=0.0)
    pkg = SeedPackage.generate(3, cfg)
    seen = []

    class Spy(GreedyMechanism):
        def allocate(self, reports, history, tie_seed):
            seen.append(reports.copy())
            return super().allocate(reports, history, tie_seed)

    run_rollout(Spy(cfg), pkg, cfg, 0, make_capped(2.0), np.random.default_rng(0), n_rollouts=10)
    for t, rep in enumerate(seen):
        np.testing.assert_allclose(rep[1:], np.minimum(2 * pkg.valuations[1:, t], 1.0))
