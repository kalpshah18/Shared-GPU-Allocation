"""
tests/test_metrics_extra.py — Owner: Raj Modi

Further metric tests beyond the hand-computed 2-user example in test_metrics.py:
waiting-time semantics, properties of Jain's index, J_B, NSW, strategic metrics
and ``compute_all``.
"""

import math

import numpy as np
import pytest

from sim import metrics as M
from sim.config import Config
from sim.environment import History, SeedPackage, expected_values
from sim.mechanisms import GreedyMechanism, RoundRobinMechanism, make_mechanism
from sim.runner import run_single


def make_history(n, allocations, payments=None):
    h = History(n)
    for i, x in enumerate(allocations):
        p = np.zeros(n) if payments is None else np.array(payments[i], dtype=float)
        h.update(np.array(x, dtype=np.int64), p)
    return h


def random_history(n, k, T, seed):
    rng = np.random.default_rng(seed)
    h = History(n)
    for _ in range(T):
        x = np.zeros(n, dtype=np.int64)
        x[rng.choice(n, k, replace=False)] = 1
        h.update(x, np.zeros(n))
    return h


# ── Waiting times (pre-round semantics) ───────────────────────────────────────

def test_wait_matrix_hand_example():
    # user 0 served in rounds 0 and 3; user 1 in round 1; user 2 never
    X = np.array([[1, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
    expected = np.array([[0, 0, 1, 2, 0],
                         [0, 1, 0, 1, 2],
                         [0, 1, 2, 3, 4]])
    np.testing.assert_array_equal(M.wait_matrix(X), expected)


@pytest.mark.parametrize("seed", range(5))
def test_wait_matrix_matches_history_counters(seed):
    h = random_history(8, 2, 60, seed)
    Q = M.wait_matrix(np.stack(h.allocations, axis=1))
    # rebuild the pre-round states from the history's own post-round counter
    replay = History(8)
    for t, x in enumerate(h.allocations):
        np.testing.assert_array_equal(Q[:, t], replay.consecutive_wait)
        replay.update(x, np.zeros(8))
    np.testing.assert_array_equal(replay.consecutive_wait, h.consecutive_wait)


def test_max_wait_and_percentile_hand_values():
    h = make_history(3, [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    Q = M.wait_matrix(np.stack(h.allocations, axis=1))
    assert M.max_wait(h) == int(Q.max()) == 3          # user 2 waited rounds 0-2
    assert M.percentile_wait(h, 100.0) == 3.0
    assert M.percentile_wait(h, 0.0) == 0.0


def test_starvation_rate_counts_user_rounds_over_delta():
    cfg = Config(n=3, k=1, T=6, delta_multiplier=1)          # Delta = ceil(3/1) = 3
    assert cfg.delta == 3
    # user 2 is never served: waits are 0,1,2,3,4,5 -> only waits 4 and 5 exceed Delta
    h = make_history(3, [[1, 0, 0], [0, 1, 0]] * 3)
    assert M.starvation_rate(h, cfg) == pytest.approx(2 / (3 * 6))


def test_round_robin_has_zero_starvation_and_known_qmax():
    cfg = Config(n=50, k=10, T=200)
    pkg = SeedPackage.generate(1, cfg)
    h = run_single(RoundRobinMechanism(cfg, int(pkg.tie_seeds[0])), pkg, cfg)
    assert M.starvation_rate(h, cfg) == 0.0
    assert M.max_wait(h) == math.ceil(cfg.n / cfg.k) - 1 == 4


# ── Jain's index ──────────────────────────────────────────────────────────────

def test_jain_bounds_and_extremes():
    n = 7
    assert M._jain(np.ones(n)) == pytest.approx(1.0)
    assert M._jain(np.eye(n)[0]) == pytest.approx(1.0 / n)
    assert M._jain(np.zeros(n)) == 1.0                       # vacuous
    rng = np.random.default_rng(0)
    for _ in range(50):
        j = M._jain(rng.random(n) * rng.integers(1, 100))
        assert 1.0 / n - 1e-12 <= j <= 1.0 + 1e-12


def test_jain_scale_invariant():
    a = np.array([1.0, 2.0, 5.0])
    assert M._jain(a) == pytest.approx(M._jain(7.3 * a))


def test_jain_allocation_hand_value():
    h = make_history(4, [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]])   # A = (3,1,1,1)
    assert M.jain_allocation(h) == pytest.approx(36 / (4 * 12))


def test_jain_benefit_divides_by_expected_value():
    VAL = np.array([[0.2, 0.2], [0.8, 0.8]])
    h = make_history(2, [[1, 0], [1, 0]])                                # user 0 wins both rounds
    # G = (0.4, 0): B = G / (T mu) = (0.4/(2*0.2), 0) = (1, 0) -> J_B = 1/2
    assert M.jain_benefit(h, VAL, np.array([0.2, 0.8])) == pytest.approx(0.5)
    # mu = 0.2 for user 1 would give B = (1, 0) as well; equalised benefit gives J_B = 1
    h2 = make_history(2, [[1, 0], [0, 1]])
    G = np.array([0.2, 0.8])
    assert M.jain_benefit(h2, VAL, G / 2) == pytest.approx(1.0)


def test_jain_benefit_zero_mu_is_safe():
    VAL = np.ones((2, 2))
    h = make_history(2, [[1, 0], [1, 0]])
    assert np.isfinite(M.jain_benefit(h, VAL, np.array([0.0, 1.0])))


def test_equal_service_vs_equal_benefit_diverge_under_heterogeneity():
    """With unequal means, equal allocation counts leave J_B < J_A (the E4 point)."""
    cfg = Config(n=6, k=3, T=60, valuation_dist="mixed")
    pkg = SeedPackage.generate(0, cfg)
    h = run_single(RoundRobinMechanism(cfg, 0), pkg, cfg)
    assert M.jain_allocation(h) > 0.999
    assert M.jain_benefit(h, pkg.valuations, expected_values(cfg)) < M.jain_allocation(h)


# ── Welfare ───────────────────────────────────────────────────────────────────

def test_payments_do_not_enter_welfare():
    cfg = Config(n=2, k=1, T=2)
    VAL = np.array([[0.8, 0.3], [0.4, 0.9]])
    free = make_history(2, [[1, 0], [0, 1]])
    paid = make_history(2, [[1, 0], [0, 1]], [[0.5, 0], [0, 0.5]])
    assert M.welfare_ratio(free, VAL, cfg) == M.welfare_ratio(paid, VAL, cfg) == pytest.approx(1.0)
    np.testing.assert_allclose(M.utilities(paid, VAL), [0.3, 0.4])
    np.testing.assert_allclose(M.gross_benefits(paid, VAL), [0.8, 0.9])


def test_oracle_welfare_and_wr_bounds_on_random_runs():
    cfg = Config(n=12, k=3, T=80)
    pkg = SeedPackage.generate(3, cfg)
    manual = sum(np.sort(pkg.valuations[:, t])[-3:].sum() for t in range(80))
    assert M.oracle_welfare(pkg.valuations, cfg) == pytest.approx(manual)
    for name in ("RandomMechanism", "GreedyMechanism", "ScoreMechanism", "VickreyMechanism"):
        h = run_single(make_mechanism(name, cfg, pkg), pkg, cfg)
        wr = M.welfare_ratio(h, pkg.valuations, cfg)
        assert 0.0 < wr <= 1.0 + 1e-12
        assert M.price_of_fairness(h, pkg.valuations, cfg) == pytest.approx(1.0 - wr)


def test_greedy_truthful_attains_oracle_and_zero_pof():
    cfg = Config(n=20, k=4, T=100)
    pkg = SeedPackage.generate(2, cfg)
    h = run_single(GreedyMechanism(cfg), pkg, cfg)
    assert M.welfare_ratio(h, pkg.valuations, cfg) == pytest.approx(1.0)
    assert M.price_of_fairness(h, pkg.valuations, cfg) == pytest.approx(0.0, abs=1e-12)


def test_degenerate_zero_valuations():
    cfg = Config(n=2, k=1, T=2)
    VAL = np.zeros((2, 2))
    h = make_history(2, [[1, 0], [0, 1]])
    assert M.welfare_ratio(h, VAL, cfg) == 1.0
    assert M.price_of_fairness(h, VAL, cfg) == 0.0
    assert M.price_of_strategy(h, h, VAL, cfg) == 0.0


def test_nsw_penalises_a_zero_benefit_user():
    VAL = np.array([[0.5, 0.5], [0.5, 0.5]])
    fair = make_history(2, [[1, 0], [0, 1]])
    unfair = make_history(2, [[1, 0], [1, 0]])
    assert M.nash_social_welfare(fair, VAL) > M.nash_social_welfare(unfair, VAL)
    assert np.isfinite(M.nash_social_welfare(unfair, VAL))


# ── Strategic metrics ─────────────────────────────────────────────────────────

VAL2 = np.array([[0.8, 0.3], [0.4, 0.9]])


def test_coalition_gain_hand_computed():
    truthful = make_history(2, [[1, 0], [0, 1]])
    strategic = make_history(2, [[1, 0], [1, 0]], [[0.0, 0], [0.2, 0]])
    # user 0: truthful 0.8, strategic 0.8 + 0.3 - 0.2 = 0.9 -> +0.1;  user 1: 0.9 -> 0 -> -0.9
    out = M.manipulation_gain(strategic, truthful, VAL2, np.array([0, 1]))
    assert out["M_mean"] == pytest.approx((0.1 - 0.9) / 2)
    assert out["M_max"] == pytest.approx(0.1)
    assert out["frac_pos"] == pytest.approx(0.5)
    only0 = M.manipulation_gain(strategic, truthful, VAL2, np.array([0]))
    assert only0["M_mean"] == pytest.approx(0.1) and only0["frac_pos"] == 1.0


def test_price_of_strategy_hand_value_and_sign():
    truthful = make_history(2, [[1, 0], [0, 1]])             # W = 1.7
    strategic = make_history(2, [[0, 1], [1, 0]])            # W = 0.7
    assert M.price_of_strategy(truthful, strategic, VAL2, None) == pytest.approx(1.0 / 1.7)
    assert M.price_of_strategy(strategic, truthful, VAL2, None) < 0     # strategic run improved welfare


def test_unilateral_summary():
    out = M.unilateral_summary([2.0, -1.0, 0.0, 5.0])
    assert out == {"M_uni": pytest.approx(1.5), "M_uni_max": 5.0, "frac_pos_uni": 0.5}
    assert M.unilateral_summary([-3.0])["frac_pos_uni"] == 0.0


# ── compute_all ───────────────────────────────────────────────────────────────

def test_compute_all_keys_and_consistency():
    cfg = Config(n=20, k=4, T=60)
    pkg = SeedPackage.generate(4, cfg)
    h = run_single(GreedyMechanism(cfg), pkg, cfg)
    out = M.compute_all(h, pkg.valuations, cfg)
    assert set(out) == {"WR", "J_A", "J_B", "NSW", "Q_max", "SR_delta", "pct95_wait", "PoF"}
    assert out["WR"] == pytest.approx(M.welfare_ratio(h, pkg.valuations, cfg))
    assert out["J_A"] == pytest.approx(M.jain_allocation(h))
    assert out["Q_max"] == M.max_wait(h) and isinstance(out["Q_max"], int)
    assert out["SR_delta"] == pytest.approx(M.starvation_rate(h, cfg))
    assert out["pct95_wait"] == pytest.approx(M.percentile_wait(h, 95))
    assert out["J_B"] == pytest.approx(M.jain_benefit(h, pkg.valuations, expected_values(cfg)))
    # a different mu vector changes only J_B
    override = M.compute_all(h, pkg.valuations, cfg, mu=np.linspace(0.1, 1.0, 20))
    assert override["J_B"] != pytest.approx(out["J_B"])
    assert override["WR"] == out["WR"] and override["J_A"] == out["J_A"]
