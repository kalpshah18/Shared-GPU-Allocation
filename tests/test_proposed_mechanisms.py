"""
tests/test_proposed_mechanisms.py — Owner: Harsh Dhru

M6 Karma-Cap, M7 Rank-Cap and the shared waiting cap: hand-computed rounds,
invariants, the hard waiting bound, karma conservation, rank invariance, and
consistency of the rollout interface with the real allocation rule.
"""

import copy
import math

import numpy as np
import pytest

from sim.config import Config
from sim.environment import History, SeedPackage
from sim.mechanisms import (
    ALL_MECHANISM_NAMES, MECHANISM_LABELS, MECHANISM_NAMES, PROPOSED_MECHANISM_NAMES,
    GreedyMechanism, KarmaCapMechanism, RankCapMechanism, ScoreMechanism, VickreyMechanism,
    make_mechanism,
)
from sim.mechanisms._waitcap import FORCED_PRIORITY, forced_mask, with_forced_priority
from sim.policies import make_capped, make_timed, maximum_claim, rollout_report
from sim.runner import run_mixed, run_rollout, run_single

PROPOSED = [KarmaCapMechanism, RankCapMechanism]


def hist(n, cum=None, wait=None):
    h = History(n)
    if cum is not None:
        h.cumulative[:] = cum
    if wait is not None:
        h.consecutive_wait[:] = wait
    return h


def big(**kw):
    """A config where the waiting cap and the karma ceiling never bind."""
    return Config(wait_cap=10**6, karma_cap_mult=1000.0, **kw)


# ── Config / factory ──────────────────────────────────────────────────────────

def test_new_config_fields_defaults_and_derived_wait_limit():
    c = Config(n=50, k=10)
    assert c.wait_cap is None and c.wait_limit == c.delta == 10
    assert (c.karma_init, c.karma_cap_mult, c.rank_blend) == (2.0, 3.0, 0.0)
    assert Config(n=50, k=10, wait_cap=7).wait_limit == 7
    assert Config(n=50, k=10).replace(wait_cap=12).wait_limit == 12


@pytest.mark.parametrize("kw", [dict(wait_cap=0), dict(wait_cap=-3), dict(karma_init=0.0),
                                dict(karma_init=-1.0), dict(karma_cap_mult=0.5),
                                dict(rank_blend=-0.1), dict(rank_blend=1.1)])
def test_invalid_proposed_parameters_rejected(kw):
    with pytest.raises(ValueError):
        Config(**kw)


def test_factory_labels_and_names_include_proposed():
    assert PROPOSED_MECHANISM_NAMES == ["KarmaCapMechanism", "RankCapMechanism"]
    assert ALL_MECHANISM_NAMES == MECHANISM_NAMES + PROPOSED_MECHANISM_NAMES
    assert set(MECHANISM_LABELS) == set(ALL_MECHANISM_NAMES)
    cfg = Config(n=8, k=2, T=3)
    pkg = SeedPackage.generate(1, cfg)
    assert type(make_mechanism("KarmaCapMechanism", cfg, pkg)) is KarmaCapMechanism
    assert type(make_mechanism("RankCapMechanism", cfg, pkg)) is RankCapMechanism


# ── Shared waiting cap ───────────────────────────────────────────────────────

def test_forced_mask_oldest_first_and_at_most_k():
    wait = np.array([[5, 9, 2, 9, 7, 0]])
    tb = np.array([[0.1, 0.5, 0.3, 0.2, 0.4, 0.6]])
    m = forced_mask(wait, cap=5, k=2, tiebreak=tb)
    assert m.sum() == 2 and m[0, 1] == 1 and m[0, 3] == 1        # the two waits of 9


def test_forced_mask_only_eligible_users():
    wait = np.array([[4, 4, 4, 4]])
    assert forced_mask(wait, cap=5, k=2, tiebreak=np.random.default_rng(0).random((1, 4))).sum() == 0
    wait = np.array([[5, 1, 1, 1]])
    m = forced_mask(wait, cap=5, k=3, tiebreak=np.random.default_rng(0).random((1, 4)))
    assert list(m[0]) == [1, 0, 0, 0]


def test_forced_mask_breaks_ties_with_the_supplied_key():
    wait = np.array([[8, 8, 8]])
    for _ in range(10):
        tb = np.random.default_rng(_).random((1, 3))
        m = forced_mask(wait, cap=5, k=1, tiebreak=tb)
        assert m[0].argmax() == tb[0].argmax()


def test_with_forced_priority_dominates_every_score():
    s = np.array([[0.2, 1e6, 0.9]])
    out = with_forced_priority(s, np.array([[1, 0, 0]]))
    assert out[0, 0] == FORCED_PRIORITY and out[0, 1] == 1e6 and out[0, 2] == 0.9


# ── Invariants common to M6 and M7 ───────────────────────────────────────────

@pytest.mark.parametrize("cls", PROPOSED)
@pytest.mark.parametrize("n,k", [(6, 1), (8, 3), (10, 2), (50, 10)])
def test_capacity_binary_and_no_payments(cls, n, k):
    cfg = Config(n=n, k=k, T=1)
    mech, h, rng = cls(cfg), hist(n), np.random.default_rng(0)
    for t in range(40):
        x, p = mech.allocate(rng.random(n), h, t)
        h.update(x, p)
        assert x.dtype == np.int64 and set(np.unique(x)) <= {0, 1} and int(x.sum()) == k
        assert np.all(p == 0.0)                              # karma is not money


@pytest.mark.parametrize("cls", PROPOSED)
def test_deterministic_and_deepcopy_independent(cls):
    cfg = Config(n=10, k=3, T=50, rho=0.0)
    pkg = SeedPackage.generate(5, cfg)
    a = run_single(copy.deepcopy(cls(cfg)), pkg, cfg)
    b = run_single(copy.deepcopy(cls(cfg)), pkg, cfg)
    np.testing.assert_array_equal(np.stack(a.allocations), np.stack(b.allocations))
    m1, m2 = cls(cfg), cls(cfg)
    m1.allocate(np.random.default_rng(0).random(10), hist(10), 3)
    run_single(m2, pkg, cfg)                                  # separate instances share no state
    fresh = cls(cfg)
    x_fresh, _ = fresh.allocate(np.full(10, 0.5), hist(10), 7)
    x_again, _ = cls(cfg).allocate(np.full(10, 0.5), hist(10), 7)
    np.testing.assert_array_equal(x_fresh, x_again)


@pytest.mark.parametrize("cls", PROPOSED)
@pytest.mark.parametrize("n,k,W", [(10, 2, 4), (10, 2, 10), (20, 4, 6), (12, 3, 5), (7, 3, 4)])
@pytest.mark.parametrize("pattern", ["random", "equal", "rich_poor", "inverse"])
def test_hard_waiting_bound_holds_for_any_reports(cls, n, k, W, pattern):
    """Q_max <= W + ceil(n/k) - 1 whatever the reports (W if no start-up overflow)."""
    cfg = Config(n=n, k=k, T=1, wait_cap=W)
    mech, h = cls(cfg), hist(n)
    rng = np.random.default_rng(1)
    q_max = 0
    for t in range(12 * n):
        if pattern == "random":
            r = rng.random(n)
        elif pattern == "equal":
            r = np.full(n, 0.5)
        elif pattern == "rich_poor":                          # the same few users always win on reports
            r = np.where(np.arange(n) < k, 1.0, 0.01)
        else:
            r = 1.0 - np.arange(n) / n
        x, p = mech.allocate(r, h, t)
        h.update(x, p)
        q_max = max(q_max, int(h.consecutive_wait.max()))
    assert q_max <= W + math.ceil(n / k) - 1


@pytest.mark.parametrize("cls", PROPOSED)
def test_cap_guarantees_service_even_to_a_user_that_always_reports_zero(cls):
    cfg = Config(n=10, k=2, T=1, wait_cap=6)
    mech, h = cls(cfg), hist(10)
    served = 0
    for t in range(120):
        r = np.ones(10)
        r[0] = 0.0                                            # user 0 never bids
        x, p = mech.allocate(r, h, t)
        h.update(x, p)
        served += x[0]
    assert served >= 120 // (6 + 5) - 1                       # at least once per W + overflow rounds


@pytest.mark.parametrize("cls", PROPOSED)
def test_forced_users_are_served_ahead_of_higher_bidders(cls):
    cfg = Config(n=5, k=1, T=1, wait_cap=3)
    mech = cls(cfg)
    h = hist(5, wait=[0, 3, 0, 0, 0])
    x, _ = mech.allocate(np.array([1.0, 0.0, 0.9, 0.8, 0.7]), h, 0)
    assert x[1] == 1


@pytest.mark.parametrize("cls", PROPOSED)
def test_cap_overflow_serves_oldest_first(cls):
    cfg = Config(n=6, k=2, T=1, wait_cap=3)
    mech = cls(cfg)
    h = hist(6, wait=[7, 9, 8, 3, 3, 3])                      # four forced, only two slots
    x, _ = mech.allocate(np.ones(6), h, 0)
    assert list(np.flatnonzero(x)) == [1, 2]


# ── M6 Karma-Cap: hand-computed rounds and economics ─────────────────────────

def test_karma_hand_computed_round_with_dividend():
    cfg = big(n=3, k=1, T=1, karma_init=5.0)
    m = KarmaCapMechanism(cfg)
    x, p = m.allocate(np.array([1.0, 0.6, 0.2]), hist(3), 0)
    assert list(x) == [1, 0, 0] and np.all(p == 0)
    # winner pays the (k+1)-st highest bid 0.6; the pot is shared equally among all three
    np.testing.assert_allclose(m.balance, [5 - 0.6 + 0.2, 5.2, 5.2])


def test_karma_forced_winner_pays_the_slot_price():
    cfg = Config(n=3, k=1, T=1, wait_cap=2, karma_init=5.0, karma_cap_mult=1000.0)
    m = KarmaCapMechanism(cfg)
    x, _ = m.allocate(np.array([1.0, 0.0, 0.5]), hist(3, wait=[0, 2, 0]), 0)
    assert list(x) == [0, 1, 0]
    # price of a slot = (k+1)-st highest bid overall = 0.5
    np.testing.assert_allclose(m.balance, [5 + 0.5 / 3, 5 - 0.5 + 0.5 / 3, 5 + 0.5 / 3])


def test_karma_bid_is_capped_by_balance():
    cfg = Config(n=3, k=1, T=1, wait_cap=100)
    m = KarmaCapMechanism(cfg)
    m.balance[0] = 0.0                                        # user 0 is broke
    x, _ = m.allocate(np.array([1.0, 0.5, 0.4]), hist(3), 0)
    assert list(x) == [0, 1, 0]
    m.balance[:] = [0.3, 5.0, 5.0]                            # partly broke: bid is min(report, 0.3)
    x, _ = m.allocate(np.array([1.0, 0.31, 0.2]), hist(3), 0)
    assert list(x) == [0, 1, 0]


def test_karma_is_conserved_when_the_ceiling_does_not_bind_and_stays_non_negative():
    cfg = big(n=20, k=4, T=1, karma_init=2.0)
    m, h, rng = KarmaCapMechanism(cfg), hist(20), np.random.default_rng(2)
    total = m.balance.sum()
    for t in range(300):
        x, p = m.allocate(rng.random(20), h, t)
        h.update(x, p)
        assert m.balance.min() >= -1e-12
        assert m.balance.sum() == pytest.approx(total)


def test_karma_ceiling_clips_balances():
    cfg = Config(n=5, k=1, T=1, karma_init=1.0, karma_cap_mult=1.2, wait_cap=100)
    m, h, rng = KarmaCapMechanism(cfg), hist(5), np.random.default_rng(3)
    for t in range(200):
        x, p = m.allocate(rng.random(5), h, t)
        h.update(x, p)
        assert m.balance.max() <= 1.2 + 1e-12


def test_karma_repeated_inflation_depletes_the_liar():
    """A user who always reports v_max wins a lot at first, then runs out of karma."""
    cfg = Config(n=10, k=2, T=1, wait_cap=100, karma_init=2.0)
    m, h, rng = KarmaCapMechanism(cfg), hist(10), np.random.default_rng(4)
    wins, balance = [], []
    for t in range(400):
        r = rng.random(10)
        r[0] = 1.0
        x, p = m.allocate(r, h, t)
        h.update(x, p)
        wins.append(x[0])
        balance.append(m.balance[0])
    assert np.mean(wins[:50]) > np.mean(wins[-200:])           # early burst, then budget-limited
    assert np.mean(balance[-200:]) < 2.0


def test_karma_truthful_rich_user_with_high_value_wins():
    cfg = Config(n=4, k=1, T=1, wait_cap=100)
    m = KarmaCapMechanism(cfg)
    x, _ = m.allocate(np.array([0.2, 0.95, 0.4, 0.1]), hist(4), 0)
    assert x[1] == 1


# ── M7 Rank-Cap ───────────────────────────────────────────────────────────────

def test_rank_quantile_hand_values():
    q = RankCapMechanism._quantile
    assert q([], 0.7) == pytest.approx(0.5)
    past = [0.1, 0.5, 0.9]
    assert q(past, 0.5) == pytest.approx((1 + 0.5 + 0.5) / 4)
    assert q(past, 0.95) == pytest.approx((3 + 0.5) / 4)
    assert q(past, 0.0) == pytest.approx(0.5 / 4)
    assert q([0.5, 0.5, 0.5, 0.5], 0.5) == pytest.approx((0 + 2.0 + 0.5) / 5)    # all ties -> mid-rank


def test_rank_quantiles_of_truthful_reports_are_roughly_uniform():
    rng = np.random.default_rng(0)
    past = sorted(rng.random(5000).tolist())     # large past: its own sampling error is negligible
    qs = np.array([RankCapMechanism._quantile(past, v) for v in rng.random(2000)])
    assert abs(qs.mean() - 0.5) < 0.02 and abs(qs.std() - math.sqrt(1 / 12)) < 0.02


def test_rank_is_invariant_to_an_increasing_distortion_of_one_users_reports():
    cfg = Config(n=10, k=2, T=80, rho=0.0, wait_cap=10**6)
    pkg = SeedPackage.generate(3, cfg)
    base = run_single(RankCapMechanism(cfg), pkg, cfg)
    lied = run_mixed(RankCapMechanism(cfg), pkg, cfg, lambda v, h, c: v ** 3, np.array([0]))
    np.testing.assert_array_equal(np.stack(base.allocations), np.stack(lied.allocations))


def test_rank_invariant_to_scaling_everyones_reports():
    cfg = Config(n=10, k=2, T=60, rho=0.0, wait_cap=10**6)
    pkg = SeedPackage.generate(4, cfg)
    base = run_single(RankCapMechanism(cfg), pkg, cfg)
    scaled = run_single(RankCapMechanism(cfg), pkg, cfg, lambda v, h, c: 0.5 * v)
    np.testing.assert_array_equal(np.stack(base.allocations), np.stack(scaled.allocations))


def test_rank_clipping_creates_ties_that_lower_the_quantile_of_good_values():
    rng = np.random.default_rng(1)
    vals = rng.random(600)
    truthful_past = sorted(vals.tolist())
    capped_past = sorted(np.minimum(2.0 * vals, 1.0).tolist())
    q_true = RankCapMechanism._quantile(truthful_past, 0.99)
    q_capped = RankCapMechanism._quantile(capped_past, 1.0)
    assert q_true > 0.97 and q_capped < 0.8


def test_rank_max_claim_gets_the_median_quantile_not_the_top():
    past = [1.0] * 100
    assert RankCapMechanism._quantile(past, 1.0) == pytest.approx((0 + 50 + 0.5) / 101)


def test_rank_with_blend_one_and_no_cap_is_exactly_the_score_rule():
    cfg_r = Config(n=10, k=3, T=1, lambda_=1.0, rank_blend=1.0, wait_cap=10**6)
    cfg_s = Config(n=10, k=3, T=1, lambda_=1.0)
    rank, score = RankCapMechanism(cfg_r), ScoreMechanism(cfg_s)
    hr, hs, rng = hist(10), hist(10), np.random.default_rng(5)
    for t in range(60):
        r = rng.random(10)
        xr, _ = rank.allocate(r, hr, t)
        xs, _ = score.allocate(r, hs, t)
        np.testing.assert_array_equal(xr, xs)
        hr.update(xr, np.zeros(10))
        hs.update(xs, np.zeros(10))


def test_rank_penalises_past_service_like_score():
    cfg = Config(n=2, k=1, T=1, lambda_=1.0, wait_cap=10**6)
    m = RankCapMechanism(cfg)
    for ts in range(20):
        x, _ = RankCapMechanism(cfg).allocate(np.array([0.5, 0.5]), hist(2, cum=[6, 1]), ts)
        assert x[1] == 1


def test_rank_treats_users_with_different_scales_alike():
    """A low-value user's best days rank as high as a high-value user's best days."""
    cfg = Config(n=2, k=1, T=1, lambda_=0.0, wait_cap=10**6)
    m = RankCapMechanism(cfg)
    rng = np.random.default_rng(0)
    h = hist(2)
    wins = np.zeros(2)
    for t in range(3000):
        r = np.array([0.1 * rng.random(), 0.9 + 0.1 * rng.random()])   # user 0 always far below user 1
        x, p = m.allocate(r, h, t)
        h.update(x, p)
        wins += x
    assert 0.4 < wins[0] / 3000 < 0.6                          # a raw-report rule would give user 0 almost nothing


# ── Rollout interface ────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [GreedyMechanism, ScoreMechanism, VickreyMechanism])
def test_default_rollout_step_matches_allocate_batch_and_advances_cum(cls):
    cfg = Config(n=6, k=2, T=1, lambda_=1.0)
    m = cls(cfg)
    h = hist(6, cum=[3, 0, 1, 0, 2, 0])
    st = m.rollout_init(h, 4)
    assert st["cum"].shape == (4, 6) and np.all(st["cum"][0] == [3, 0, 1, 0, 2, 0])
    rng = np.random.default_rng(0)
    r, tb = rng.random((4, 6)), rng.random((4, 6))
    x, p = m.rollout_step(r, st, tb)
    xb, pb = m.allocate_batch(r, np.tile([3, 0, 1, 0, 2, 0], (4, 1)), tb)
    np.testing.assert_array_equal(x, xb)
    np.testing.assert_array_equal(st["cum"], np.tile([3, 0, 1, 0, 2, 0], (4, 1)) + x)


def test_karma_rollout_reproduces_the_real_allocation_and_balances():
    cfg = Config(n=8, k=2, T=1, wait_cap=4, karma_init=2.0)
    m, h, rng = KarmaCapMechanism(cfg), hist(8), np.random.default_rng(6)
    for t in range(60):
        r = rng.random(8)
        tie = int(rng.integers(1, 10**9))
        st = m.rollout_init(h, 1)
        xr, pr = m.rollout_step(r[None, :], st, np.random.default_rng(tie).random(8)[None, :])
        x, _ = m.allocate(r, h, tie)
        np.testing.assert_array_equal(xr[0], x)
        np.testing.assert_allclose(st["bal"][0], m.balance)
        assert np.all(pr == 0)
        h.update(x, np.zeros(8))
        np.testing.assert_array_equal(st["wait"][0], h.consecutive_wait)


def test_rank_rollout_reproduces_the_real_allocation():
    cfg = Config(n=8, k=2, T=1, wait_cap=4)
    m, h, rng = RankCapMechanism(cfg), hist(8), np.random.default_rng(7)
    for t in range(60):
        r = rng.random(8)
        tie = int(rng.integers(1, 10**9))
        st = m.rollout_init(h, 1)
        xr, _ = m.rollout_step(r[None, :], st, np.random.default_rng(tie).random(8)[None, :])
        x, _ = m.allocate(r, h, tie)
        np.testing.assert_array_equal(xr[0], x)
        h.update(x, np.zeros(8))


def test_rollout_init_is_a_copy_not_a_view():
    cfg = Config(n=5, k=1, T=1)
    m = KarmaCapMechanism(cfg)
    st = m.rollout_init(hist(5), 3)
    st["bal"][:] = 99.0
    assert np.all(m.balance == cfg.karma_init)


@pytest.mark.parametrize("cls", PROPOSED)
def test_rollout_report_works_and_leaves_the_mechanism_untouched(cls):
    cfg = Config(n=10, k=2, T=1)
    m = cls(cfg)
    h, rng = hist(10), np.random.default_rng(0)
    for t in range(25):                                       # build some state
        x, p = m.allocate(rng.random(10), h, t)
        h.update(x, p)
    bal = getattr(m, "balance", np.zeros(1)).copy()
    past = [list(p) for p in getattr(m, "_past", [])]
    r = rollout_report(0.5, h, m, cfg, 0, lambda v, hh, c: v, np.random.default_rng(1), H=4, n_rollouts=20)
    assert 0.0 <= r <= 1.0
    np.testing.assert_array_equal(getattr(m, "balance", np.zeros(1)), bal)
    assert [list(p) for p in getattr(m, "_past", [])] == past


@pytest.mark.parametrize("cls", PROPOSED)
def test_run_rollout_with_proposed_mechanisms(cls):
    cfg = Config(n=10, k=2, T=20, rho=0.0)
    pkg = SeedPackage.generate(2, cfg)
    h, reports = run_rollout(cls(cfg), pkg, cfg, 0, None, np.random.default_rng(0), H=3, n_rollouts=15)
    assert h.round == 20 and reports.shape == (20,) and np.all((reports >= 0) & (reports <= 1))


# ── The design goal, in miniature ────────────────────────────────────────────

@pytest.mark.parametrize("cls", PROPOSED)
def test_proposed_mechanisms_punish_blanket_inflation_that_greedy_rewards(cls):
    from sim.runner import unilateral_gains
    cfg = Config(n=20, k=4, T=200, rho=0.25, n_focal=3)
    pkg = SeedPackage.generate(1, cfg)
    _, g_greedy = unilateral_gains(GreedyMechanism(cfg), pkg, cfg, maximum_claim)
    _, g_new = unilateral_gains(cls(cfg), pkg, cfg, maximum_claim)
    assert np.all(g_greedy > 0) and np.all(g_new < 0)


@pytest.mark.parametrize("cls", PROPOSED)
def test_proposed_mechanisms_keep_most_of_greedys_welfare_with_zero_starvation(cls):
    from sim import metrics as M
    cfg = Config(n=50, k=10, T=300, rho=0.0)
    pkg = SeedPackage.generate(2, cfg)
    h = run_single(cls(cfg), pkg, cfg)
    assert M.welfare_ratio(h, pkg.valuations, cfg) > 0.9
    assert M.starvation_rate(h, cfg) == 0.0
    assert M.max_wait(h) <= cfg.wait_limit + math.ceil(cfg.n / cfg.k) - 1
    assert M.jain_allocation(h) > 0.99
