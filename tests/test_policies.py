"""
tests/test_policies.py — Owner: Raj Modi

Scalable reporting policies (elementwise, bounded) and the rollout attack.
"""

import numpy as np
import pytest

from sim.config import Config
from sim.environment import History
from sim.mechanisms import (
    GreedyMechanism, RandomMechanism, ScoreMechanism, VickreyMechanism,
)
from sim.policies import (
    POLICY_REGISTRY, ROLLOUT_GRID, capped_exaggeration, get_policy, make_capped,
    maximum_claim, rollout_report, truthful,
)

CFG = Config(n=6, k=2, T=10)
H0 = History(CFG.n)


# ── Truthful / capped / max claim ─────────────────────────────────────────────

def test_truthful_identity_scalar_and_array():
    assert truthful(0.37, H0, CFG) == 0.37
    v = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(truthful(v, H0, CFG), v)


@pytest.mark.parametrize("v,c,expected", [(0.3, 2.0, 0.6), (0.8, 2.0, 1.0), (0.4, 1.25, 0.5),
                                          (0.9, 1.5, 1.0), (0.0, 2.0, 0.0), (1.0, 1.25, 1.0)])
def test_capped_exaggeration_values(v, c, expected):
    assert capped_exaggeration(v, H0, CFG, c=c) == pytest.approx(expected)


def test_capped_exaggeration_respects_v_max_scaling():
    cfg = Config(n=4, k=1, T=1, v_max=3.0)
    assert capped_exaggeration(2.5, History(4), cfg, c=2.0) == pytest.approx(3.0)
    assert capped_exaggeration(1.0, History(4), cfg, c=2.0) == pytest.approx(2.0)


def test_capped_is_elementwise_and_matches_scalar_calls():
    v = np.linspace(0, 1, 11)
    vec = capped_exaggeration(v, H0, CFG, c=1.5)
    assert vec.shape == v.shape
    for a, b in zip(vec, v):
        assert a == pytest.approx(capped_exaggeration(float(b), H0, CFG, c=1.5))


def test_maximum_claim_scalar_and_array():
    assert maximum_claim(0.2, H0, CFG) == CFG.v_max
    out = maximum_claim(np.array([0.0, 0.5]), H0, CFG)
    np.testing.assert_array_equal(out, [CFG.v_max, CFG.v_max])
    assert maximum_claim(np.zeros((2, 3)), H0, CFG).shape == (2, 3)


def test_make_capped_binds_c_and_names_itself():
    pol = make_capped(1.25)
    assert pol(0.4, H0, CFG) == pytest.approx(0.5)
    assert pol.__name__ == "cap_1.25"


# ── Registry / lookup ─────────────────────────────────────────────────────────

def test_registry_keys_and_lookup():
    assert {"truthful", "cap_1.25", "cap_1.5", "cap_2", "max_claim"} == set(POLICY_REGISTRY)
    assert get_policy("cap_2")(0.3, H0, CFG) == pytest.approx(0.6)
    assert get_policy("cap_1.5")(0.4, H0, CFG) == pytest.approx(0.6)
    assert get_policy("cap_3")(0.2, H0, CFG) == pytest.approx(0.6)       # built on the fly
    with pytest.raises(KeyError):
        get_policy("nonsense")


@pytest.mark.parametrize("name", sorted(POLICY_REGISTRY))
def test_every_policy_reports_within_bounds(name):
    pol = POLICY_REGISTRY[name]
    v = np.random.default_rng(0).random(500) * CFG.v_max
    r = pol(v, H0, CFG)
    assert np.all(r >= 0.0) and np.all(r <= CFG.v_max + 1e-12)
    for x in v[:25]:
        assert 0.0 <= pol(float(x), H0, CFG) <= CFG.v_max + 1e-12


@pytest.mark.parametrize("name", ["cap_1.25", "cap_1.5", "cap_2", "max_claim"])
def test_inflating_policies_never_underreport(name):
    v = np.random.default_rng(1).random(200)
    assert np.all(POLICY_REGISTRY[name](v, H0, CFG) >= v - 1e-12)


# ── Rollout attack ────────────────────────────────────────────────────────────

RCFG = Config(n=10, k=2, T=1)


def _roll(mech, true_value, seed=0, opp=None, hist=None, **kw):
    opp = opp or (lambda v, h, c: v)
    return rollout_report(true_value, hist or History(RCFG.n), mech, RCFG, 0, opp,
                          np.random.default_rng(seed), **kw)


def test_rollout_grid_is_the_proposals():
    assert ROLLOUT_GRID == tuple(round(0.1 * i, 1) for i in range(11))


def test_rollout_under_greedy_inflates():
    r = _roll(GreedyMechanism(RCFG), 0.3, n_rollouts=200)
    assert r >= 0.8


def test_rollout_under_vickrey_is_truthful_on_the_grid():
    """Vickrey is stateless and DSIC: with common random numbers 0.5 is exactly optimal."""
    for seed in range(5):
        assert _roll(VickreyMechanism(RCFG), 0.5, seed=seed, n_rollouts=100) == pytest.approx(0.5)
    assert _roll(VickreyMechanism(RCFG), 0.0) == 0.0


def test_rollout_vickrey_picks_a_grid_neighbour_of_off_grid_values():
    r = _roll(VickreyMechanism(RCFG), 0.55, n_rollouts=300)
    assert r in (0.5, 0.6)


def test_rollout_returns_grid_value_and_is_deterministic_given_rng():
    mech = ScoreMechanism(RCFG)
    a = _roll(mech, 0.42, seed=7)
    assert a in ROLLOUT_GRID
    assert a == _roll(mech, 0.42, seed=7)


def test_rollout_does_not_mutate_history_or_mechanism():
    h = History(RCFG.n)
    h.cumulative[:] = np.arange(RCFG.n)
    wait = h.consecutive_wait.copy()
    mech = ScoreMechanism(RCFG)
    _roll(mech, 0.5, hist=h)
    np.testing.assert_array_equal(h.cumulative, np.arange(RCFG.n))
    np.testing.assert_array_equal(h.consecutive_wait, wait)
    assert h.round == 0 and not h.allocations


def test_rollout_accounts_for_the_history_penalty():
    """A user already served a lot has low scores, so Score rewards inflation more than for a fresh user."""
    cfg = Config(n=10, k=2, T=1, lambda_=1.0)
    mech = ScoreMechanism(cfg)
    h = History(10)
    h.cumulative[:] = [200] + [0] * 9                       # focal user is heavily penalised
    r = rollout_report(0.5, h, mech, cfg, 0, lambda v, hh, c: v, np.random.default_rng(0), n_rollouts=200)
    assert r in ROLLOUT_GRID                                # well-defined even when nothing wins


def test_rollout_requires_batch_capable_mechanism():
    with pytest.raises(NotImplementedError):
        _roll(RandomMechanism(RCFG), 0.5)


def test_rollout_scales_grid_with_v_max():
    cfg = Config(n=10, k=2, T=1, v_max=4.0)
    r = rollout_report(2.0, History(10), VickreyMechanism(cfg), cfg, 0, lambda v, h, c: v,
                       np.random.default_rng(0))
    assert r == pytest.approx(2.0)


def test_rollout_respects_opponent_policy():
    """Against max-claiming opponents a Greedy focal user cannot win by inflating below 1."""
    opp = lambda v, h, c: np.where(np.arange(c.n) == 0, v, c.v_max)   # everyone else reports v_max
    r = _roll(GreedyMechanism(RCFG), 0.3, opp=opp, n_rollouts=50)
    assert r in ROLLOUT_GRID
