"""
tests/test_mechanisms.py — Owner: Harsh Dhru

Invariants and properties of M1-M5, the tie-breaking utilities and the factory.
"""

import copy
import math
from itertools import product

import numpy as np
import pytest

from sim.config import Config
from sim.environment import History, SeedPackage
from sim.mechanisms import (
    ALL_MECHANISM_NAMES, MECHANISM_LABELS, MECHANISM_NAMES, GreedyMechanism, RandomMechanism,
    RoundRobinMechanism, ScoreMechanism, VickreyMechanism, make_mechanism,
)
from sim.mechanisms._utils import batch_top_k_mask, seeded_top_k
from sim.runner import run_single

ALL = [RandomMechanism, RoundRobinMechanism, GreedyMechanism, ScoreMechanism, VickreyMechanism]


def build(cls, cfg, seed=0):
    return cls(cfg, init_seed=seed) if cls is RoundRobinMechanism else cls(cfg)


def hist(n, cum=None):
    h = History(n)
    if cum is not None:
        h.cumulative[:] = cum
    return h


# ── Factory ───────────────────────────────────────────────────────────────────

def test_factory_names_labels_and_types():
    cfg = Config(n=6, k=2, T=3)
    pkg = SeedPackage.generate(1, cfg)
    assert MECHANISM_NAMES == [c.__name__ for c in ALL]
    assert set(MECHANISM_LABELS) == set(ALL_MECHANISM_NAMES)
    for name, cls in zip(MECHANISM_NAMES, ALL):
        m = make_mechanism(name, cfg, pkg)
        assert type(m) is cls and m.name == name


def test_factory_errors():
    cfg = Config(n=6, k=2, T=3)
    with pytest.raises(ValueError):
        make_mechanism("Nope", cfg)
    with pytest.raises(ValueError):
        make_mechanism("RoundRobinMechanism", cfg)          # needs a SeedPackage


def test_factory_roundrobin_uses_round0_tie_seed():
    cfg = Config(n=9, k=3, T=4)
    pkg = SeedPackage.generate(2, cfg)
    a = make_mechanism("RoundRobinMechanism", cfg, pkg)
    b = RoundRobinMechanism(cfg, init_seed=int(pkg.tie_seeds[0]))
    np.testing.assert_array_equal(a.allocate(np.zeros(9), hist(9), 0)[0],
                                  b.allocate(np.zeros(9), hist(9), 0)[0])


# ── Capacity / payment invariants ─────────────────────────────────────────────

@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("n,k", [(6, 1), (6, 2), (8, 3), (10, 9), (50, 10)])
def test_capacity_binary_and_payment_invariants(cls, n, k):
    cfg = Config(n=n, k=k, T=1)
    mech, h, rng = build(cls, cfg), hist(n), np.random.default_rng(0)
    for t in range(20):
        x, p = mech.allocate(rng.random(n), h, t)
        h.update(x, p)
        assert x.dtype == np.int64 and set(np.unique(x)) <= {0, 1}
        assert int(x.sum()) == k
        assert p.shape == (n,) and np.all(p >= 0) and np.all(p[x == 0] == 0.0)


@pytest.mark.parametrize("cls", [RandomMechanism, RoundRobinMechanism, GreedyMechanism, ScoreMechanism])
def test_nonmonetary_mechanisms_charge_nothing(cls):
    cfg = Config(n=8, k=3, T=1)
    _, p = build(cls, cfg).allocate(np.random.default_rng(1).random(8), hist(8), 0)
    assert np.all(p == 0.0)


@pytest.mark.parametrize("cls", ALL)
def test_allocation_does_not_mutate_history_or_reports(cls):
    cfg = Config(n=6, k=2, T=1)
    h = hist(6, [1, 2, 3, 4, 5, 6])
    r = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2])
    r0, c0 = r.copy(), h.cumulative.copy()
    build(cls, cfg).allocate(r, h, 3)
    np.testing.assert_array_equal(r, r0)
    np.testing.assert_array_equal(h.cumulative, c0)
    assert h.round == 0 and not h.allocations


@pytest.mark.parametrize("cls", ALL)
def test_same_inputs_same_outcome(cls):
    cfg = Config(n=8, k=3, T=1)
    r = np.random.default_rng(3).random(8)
    x1, p1 = build(cls, cfg).allocate(r, hist(8, [1, 0, 2, 0, 1, 0, 3, 0]), 11)
    x2, p2 = build(cls, cfg).allocate(r, hist(8, [1, 0, 2, 0, 1, 0, 3, 0]), 11)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(p1, p2)


# ── M1 Random ─────────────────────────────────────────────────────────────────

def test_m1_report_invariant_and_history_blind():
    cfg = Config(n=6, k=2, T=1)
    outs = [RandomMechanism(cfg).allocate(r, hist(6, c), 42)[0]
            for r in (np.zeros(6), np.ones(6), np.linspace(0, 1, 6))
            for c in (None, [9, 0, 0, 0, 0, 0])]
    for o in outs[1:]:
        np.testing.assert_array_equal(outs[0], o)


def test_m1_is_uniform():
    cfg = Config(n=10, k=3, T=1)
    counts = np.zeros(10)
    for s in range(4000):
        counts += RandomMechanism(cfg).allocate(np.zeros(10), hist(10), s)[0]
    freq = counts / 4000
    assert np.all(np.abs(freq - 0.3) < 0.04)


# ── M2 Round-robin ────────────────────────────────────────────────────────────

def test_m2_report_invariant():
    cfg = Config(n=6, k=2, T=1)
    x1, _ = RoundRobinMechanism(cfg, 0).allocate(np.zeros(6), hist(6), 0)
    x2, _ = RoundRobinMechanism(cfg, 0).allocate(np.ones(6), hist(6), 0)
    np.testing.assert_array_equal(x1, x2)


@pytest.mark.parametrize("n,k", [(6, 2), (7, 3), (50, 10), (10, 9), (5, 1)])
def test_m2_wait_bound_and_equal_service(n, k):
    cfg = Config(n=n, k=k, T=1)
    mech, h = RoundRobinMechanism(cfg, init_seed=7), hist(n)
    bound = math.ceil(n / k) - 1
    for t in range(4 * n):
        x, p = mech.allocate(np.zeros(n), h, t)
        h.update(x, p)
        assert h.consecutive_wait.max() <= bound
        assert h.cumulative.max() - h.cumulative.min() <= 1


def test_m2_serves_everyone_once_per_cycle_when_k_divides_n():
    cfg = Config(n=12, k=3, T=1)
    mech, h = RoundRobinMechanism(cfg, init_seed=1), hist(12)
    total = np.zeros(12, dtype=int)
    for t in range(4):                                      # n/k = 4 rounds = one cycle
        x, p = mech.allocate(np.zeros(12), h, t)
        h.update(x, p)
        total += x
    np.testing.assert_array_equal(total, np.ones(12))


def test_m2_initial_queue_depends_on_seed():
    cfg = Config(n=20, k=5, T=1)
    a = RoundRobinMechanism(cfg, 1).allocate(np.zeros(20), hist(20), 0)[0]
    b = RoundRobinMechanism(cfg, 2).allocate(np.zeros(20), hist(20), 0)[0]
    assert not np.array_equal(a, b)


# ── M3 Greedy ─────────────────────────────────────────────────────────────────

def test_m3_picks_top_k_reports():
    cfg = Config(n=6, k=2, T=1)
    x, _ = GreedyMechanism(cfg).allocate(np.array([.1, .9, .5, .3, .7, .2]), hist(6), 0)
    np.testing.assert_array_equal(np.flatnonzero(x), [1, 4])


def test_m3_ignores_history():
    cfg = Config(n=6, k=2, T=1)
    r = np.array([.1, .9, .5, .3, .7, .2])
    x, _ = GreedyMechanism(cfg).allocate(r, hist(6, [100, 0, 0, 0, 0, 0]), 0)
    np.testing.assert_array_equal(np.flatnonzero(x), [1, 4])


def test_m3_welfare_oracle_on_random_instances():
    cfg = Config(n=9, k=4, T=1)
    rng = np.random.default_rng(0)
    for t in range(100):
        v = rng.random(9)
        x, _ = GreedyMechanism(cfg).allocate(v, hist(9), t)
        assert abs(v @ x - np.sort(v)[-4:].sum()) < 1e-12


def test_ties_are_broken_uniformly_and_reproducibly():
    cfg = Config(n=5, k=1, T=1)
    r = np.full(5, 0.5)
    winners = [int(np.flatnonzero(GreedyMechanism(cfg).allocate(r, hist(5), s)[0])[0]) for s in range(2000)]
    freq = np.bincount(winners, minlength=5) / 2000
    assert np.all(np.abs(freq - 0.2) < 0.04)
    again = [int(np.flatnonzero(GreedyMechanism(cfg).allocate(r, hist(5), s)[0])[0]) for s in range(50)]
    assert again == winners[:50]


# ── M4 Score ──────────────────────────────────────────────────────────────────

def test_m4_equals_m3_at_lambda_zero_under_any_history():
    cfg = Config(n=8, k=3, T=1, lambda_=0.0)
    rng = np.random.default_rng(0)
    for t in range(50):
        r, cum = rng.random(8), rng.integers(0, 50, 8)
        x3, _ = GreedyMechanism(cfg).allocate(r, hist(8, cum), t)
        x4, _ = ScoreMechanism(cfg).allocate(r, hist(8, cum), t)
        np.testing.assert_array_equal(x3, x4)


def test_m4_score_formula():
    cfg = Config(n=3, k=1, T=1, lambda_=2.0)
    s = ScoreMechanism(cfg).scores(np.array([1.0, 0.5, 0.2]), np.array([0, 1, 4]))
    np.testing.assert_allclose(s, [1.0, 0.5 / 4, 0.2 / 25])


def test_m4_penalises_past_service():
    cfg = Config(n=2, k=1, T=1, lambda_=1.0)
    # equal reports: the user served less wins
    for ts in range(20):
        x, _ = ScoreMechanism(cfg).allocate(np.array([0.5, 0.5]), hist(2, [5, 2]), ts)
        assert x[1] == 1
    # a sufficiently higher report overcomes the penalty: 0.9/(1+3) vs 0.2/(1+0)
    x, _ = ScoreMechanism(cfg).allocate(np.array([0.9, 0.2]), hist(2, [3, 0]), 0)
    assert x[0] == 1


def test_m4_approaches_least_served_first_for_large_lambda():
    cfg = Config(n=4, k=1, T=1, lambda_=50.0)
    x, _ = ScoreMechanism(cfg).allocate(np.array([1.0, 0.01, 1.0, 1.0]), hist(4, [3, 2, 3, 3]), 0)
    assert x[1] == 1


def test_m4_tiebreak_is_scale_free_for_tiny_scores():
    """Regression: an additive 1e-12 jitter used to scramble scores ~1e-12 (large lambda)."""
    cfg = Config(n=3, k=1, T=1, lambda_=5.0)
    # scores: 1/(1+200)^5 ≈ 3.1e-12 vs 0.5/(1+200)^5 ≈ 1.6e-12 — a clear 2x gap
    for ts in range(300):
        x, _ = ScoreMechanism(cfg).allocate(np.array([1.0, 0.5, 0.25]), hist(3, [200, 200, 200]), ts)
        assert x[0] == 1


def test_m4_monotone_in_own_report():
    cfg = Config(n=5, k=2, T=1, lambda_=1.0)
    rng = np.random.default_rng(0)
    for t in range(100):
        r = rng.random(5)
        cum = rng.integers(0, 10, 5)
        i = int(rng.integers(0, 5))
        won_low = ScoreMechanism(cfg).allocate(r, hist(5, cum), t)[0][i]
        r2 = r.copy()
        r2[i] = min(1.0, r[i] + 0.3)
        won_high = ScoreMechanism(cfg).allocate(r2, hist(5, cum), t)[0][i]
        assert won_high >= won_low                          # raising a report never loses a slot


# ── M5 Vickrey ────────────────────────────────────────────────────────────────

def test_m5_winners_pay_the_k_plus_1_highest_bid():
    cfg = Config(n=5, k=2, T=1)
    r = np.array([0.9, 0.3, 0.7, 0.5, 0.1])
    x, p = VickreyMechanism(cfg).allocate(r, hist(5), 0)
    np.testing.assert_array_equal(np.flatnonzero(x), [0, 2])
    np.testing.assert_allclose(p[x == 1], 0.5)
    assert np.all(p[x == 0] == 0)


def test_m5_k_equals_n_minus_1_threshold_is_lowest_bid():
    cfg = Config(n=4, k=3, T=1)
    x, p = VickreyMechanism(cfg).allocate(np.array([0.9, 0.2, 0.6, 0.4]), hist(4), 0)
    np.testing.assert_allclose(p[x == 1], 0.2)


@pytest.mark.parametrize("n,k,levels", [(3, 1, 11), (4, 2, 5), (5, 2, 4)])
def test_m5_dominant_strategy_truthfulness_exhaustive(n, k, levels):
    grid = np.linspace(0, 1, levels)
    cfg = Config(n=n, k=k, T=1)

    def util(v, r, others, ts):
        reports = np.concatenate([[r], others])
        x, p = VickreyMechanism(cfg).allocate(reports, hist(n), ts)
        return v * x[0] - p[0]

    for others in product(grid, repeat=n - 1):
        others = np.array(others)
        for ts in (0, 1, 2):
            for v in grid:
                u_truth = util(v, v, others, ts)
                assert u_truth >= -1e-12                    # individual rationality
                for r in grid:
                    assert util(v, r, others, ts) <= u_truth + 1e-12


@pytest.mark.parametrize("cls,kwargs", [(GreedyMechanism, {}), (ScoreMechanism, {"lambda_": 1.0})])
def test_m3_m4_are_not_dsic_so_the_check_has_teeth(cls, kwargs):
    cfg = Config(n=3, k=1, T=1, **kwargs)
    # focal value 0.4 vs a rival at 0.6: truthful loses (utility 0), inflating wins free utility
    rival = np.array([0.6, 0.1])
    x_t, _ = cls(cfg).allocate(np.concatenate([[0.4], rival]), hist(3), 0)
    x_d, _ = cls(cfg).allocate(np.concatenate([[1.0], rival]), hist(3), 0)
    assert x_t[0] == 0 and x_d[0] == 1


# ── Tie-break utilities ───────────────────────────────────────────────────────

def test_seeded_top_k_basics():
    out = seeded_top_k(np.array([3.0, 1.0, 5.0, 4.0]), 2, 0)
    assert list(out) == [2, 3]                              # descending by score
    assert len(set(seeded_top_k(np.arange(10.0), 4, 1))) == 4


def test_seeded_top_k_is_scale_free():
    s = np.array([3.0, 1.0, 5.0, 4.0, 2.0])
    for scale in (1e-300, 1e-12, 1.0, 1e12):
        assert set(seeded_top_k(s * scale, 2, 0)) == {2, 3}


def test_batch_top_k_mask_matches_single():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 4, size=(20, 7)).astype(float)    # many exact ties
    tb = rng.random(scores.shape)
    mask = batch_top_k_mask(scores, tb, 3)
    assert mask.shape == scores.shape and np.all(mask.sum(axis=1) == 3)
    for row in range(20):
        order = np.lexsort((tb[row], scores[row]))
        ref = np.zeros(7, dtype=np.int64)
        ref[order[-3:]] = 1
        np.testing.assert_array_equal(mask[row], ref)


# ── Batch allocation ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [RandomMechanism, RoundRobinMechanism])
def test_batch_unsupported_for_report_invariant_mechanisms(cls):
    cfg = Config(n=4, k=2, T=1)
    with pytest.raises(NotImplementedError):
        build(cls, cfg).allocate_batch(np.zeros((1, 4)), np.zeros((1, 4), dtype=int), np.zeros((1, 4)))


@pytest.mark.parametrize("cls", [GreedyMechanism, ScoreMechanism, VickreyMechanism])
@pytest.mark.parametrize("lam", [0.0, 1.0, 3.0])
def test_batch_matches_single_scenario_rule(cls, lam):
    n, k, B = 7, 3, 40
    cfg = Config(n=n, k=k, T=1, lambda_=lam)
    rng = np.random.default_rng(1)
    reports = rng.random((B, n))
    cum = rng.integers(0, 30, (B, n))
    tb = rng.random((B, n))
    xb, pb = cls(cfg).allocate_batch(reports, cum, tb)
    assert xb.shape == pb.shape == (B, n) and np.all(xb.sum(axis=1) == k)
    for b in range(B):
        scores = reports[b] / (1.0 + cum[b]) ** lam if cls is ScoreMechanism else reports[b]
        order = np.lexsort((tb[b], scores))
        ref = np.zeros(n, dtype=np.int64)
        ref[order[-k:]] = 1
        np.testing.assert_array_equal(xb[b], ref)
        if cls is VickreyMechanism:
            np.testing.assert_allclose(pb[b][ref == 1], np.sort(reports[b])[::-1][k])
        assert np.all(pb[b][ref == 0] == 0)


# ── End-to-end properties over a full run ─────────────────────────────────────

@pytest.mark.parametrize("name", MECHANISM_NAMES)
def test_runs_are_deterministic_and_deepcopy_safe(name):
    cfg = Config(n=10, k=3, T=40, rho=0.0)
    pkg = SeedPackage.generate(5, cfg)
    mech = make_mechanism(name, cfg, pkg)
    h1 = run_single(copy.deepcopy(mech), pkg, cfg)
    h2 = run_single(copy.deepcopy(mech), pkg, cfg)
    np.testing.assert_array_equal(np.stack(h1.allocations), np.stack(h2.allocations))
    np.testing.assert_array_equal(np.stack(h1.payments), np.stack(h2.payments))
    assert h1.cumulative.sum() == cfg.k * cfg.T


def test_all_mechanisms_see_identical_valuations_for_a_seed():
    cfg = Config(n=10, k=3, T=20)
    pkg = SeedPackage.generate(8, cfg)
    snapshot = pkg.valuations.copy()
    for name in MECHANISM_NAMES:
        run_single(make_mechanism(name, cfg, pkg), pkg, cfg)
    np.testing.assert_array_equal(pkg.valuations, snapshot)  # runs never mutate omega
