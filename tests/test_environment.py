"""
tests/test_environment.py — Owner: Aayush Kuloor

Determinism, valuation processes (i.i.d., Beta, both AR(1) variants), strategic
sets, History bookkeeping and strict-JSON persistence.
"""

import json
import math

import numpy as np
import pytest

from sim.config import Config
from sim.environment import (
    History, SeedPackage, expected_values, sample_iid_values, save_json, to_jsonable,
)


# ── Determinism & paired randomness ──────────────────────────────────────────

def test_same_seed_same_package():
    cfg = Config(n=10, k=3, T=50, rho=0.3)
    a, b = SeedPackage.generate(42, cfg), SeedPackage.generate(42, cfg)
    np.testing.assert_array_equal(a.valuations, b.valuations)
    np.testing.assert_array_equal(a.tie_seeds, b.tie_seeds)
    np.testing.assert_array_equal(a.strategic_set, b.strategic_set)


def test_different_seeds_differ():
    cfg = Config(n=10, k=3, T=50)
    assert not np.array_equal(SeedPackage.generate(1, cfg).valuations,
                              SeedPackage.generate(2, cfg).valuations)


def test_streams_are_independent_of_rho():
    """Changing rho must not change valuations or tie seeds (paired design)."""
    a = SeedPackage.generate(5, Config(n=20, k=4, T=30, rho=0.0))
    b = SeedPackage.generate(5, Config(n=20, k=4, T=30, rho=0.5))
    np.testing.assert_array_equal(a.valuations, b.valuations)
    np.testing.assert_array_equal(a.tie_seeds, b.tie_seeds)


def test_strategic_sets_are_nested_in_rho():
    sets = [set(SeedPackage.generate(9, Config(n=40, k=8, T=5, rho=r)).strategic_set)
            for r in (0.0, 0.1, 0.25, 0.5, 1.0)]
    for small, big in zip(sets, sets[1:]):
        assert small <= big
    assert sets[-1] == set(range(40))


@pytest.mark.parametrize("rho,expected", [(0.0, 0), (0.1, 5), (0.25, 12), (0.5, 25), (1.0, 50)])
def test_strategic_set_size_and_validity(rho, expected):
    pkg = SeedPackage.generate(3, Config(n=50, k=10, T=5, rho=rho))
    s = pkg.strategic_set
    assert len(s) == expected == math.floor(rho * 50)
    assert len(set(s.tolist())) == len(s)                       # distinct
    assert np.all(np.diff(s) > 0) if len(s) > 1 else True       # sorted
    assert s.min(initial=0) >= 0 and s.max(initial=0) < 50


def test_shapes_and_dtypes():
    cfg = Config(n=15, k=5, T=30)
    pkg = SeedPackage.generate(7, cfg)
    assert pkg.valuations.shape == (15, 30) and pkg.valuations.dtype == np.float64
    assert pkg.tie_seeds.shape == (30,) and pkg.tie_seeds.dtype == np.uint64


# ── Valuation distributions ───────────────────────────────────────────────────

@pytest.mark.parametrize("dist", ["uniform", "beta_low", "beta_high", "mixed"])
@pytest.mark.parametrize("alpha,mode", [(0.0, "proposal"), (0.5, "proposal"), (0.9, "proposal"),
                                        (0.5, "copula"), (0.9, "copula")])
def test_values_in_bounds(dist, alpha, mode):
    cfg = Config(n=10, k=3, T=200, valuation_dist=dist, alpha=alpha, ar1_mode=mode, v_max=2.0)
    v = SeedPackage.generate(0, cfg).valuations
    assert v.min() >= 0.0 and v.max() <= 2.0 and np.all(np.isfinite(v))


@pytest.mark.parametrize("dist,mean", [("uniform", 0.5), ("beta_low", 2 / 7), ("beta_high", 5 / 7)])
def test_iid_marginal_means(dist, mean):
    cfg = Config(n=40, k=5, T=1500, valuation_dist=dist)
    v = SeedPackage.generate(1, cfg).valuations
    assert abs(v.mean() - mean) < 0.01
    np.testing.assert_allclose(expected_values(cfg), mean)


def test_mixed_groups_and_expected_values():
    cfg = Config(n=10, k=2, T=2000, valuation_dist="mixed")
    v = SeedPackage.generate(2, cfg).valuations
    mu = expected_values(cfg)
    np.testing.assert_allclose(mu[:5], 2 / 7)
    np.testing.assert_allclose(mu[5:], 5 / 7)
    assert abs(v[:5].mean() - 2 / 7) < 0.01 and abs(v[5:].mean() - 5 / 7) < 0.01


def test_expected_values_scale_with_v_max_and_odd_n():
    cfg = Config(n=7, k=2, valuation_dist="mixed", v_max=3.0)
    mu = expected_values(cfg)
    assert mu.shape == (7,)
    np.testing.assert_allclose(mu[:3], 3 * 2 / 7)
    np.testing.assert_allclose(mu[3:], 3 * 5 / 7)


def test_sample_iid_values_shape_and_range():
    cfg = Config(n=6, k=2, valuation_dist="mixed")
    out = sample_iid_values(np.random.default_rng(0), cfg, (4, 7))
    assert out.shape == (4, 7, 6) and out.min() >= 0 and out.max() <= 1


# ── AR(1) variants ────────────────────────────────────────────────────────────

def _lag1_corr(v):
    a, b = v[:, :-1].ravel(), v[:, 1:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


@pytest.mark.parametrize("alpha", [0.5, 0.9])
@pytest.mark.parametrize("mode", ["proposal", "copula"])
def test_ar1_autocorrelation_equals_alpha(alpha, mode):
    cfg = Config(n=40, k=5, T=800, alpha=alpha, ar1_mode=mode)
    assert abs(_lag1_corr(SeedPackage.generate(3, cfg).valuations) - alpha) < 0.05


def test_iid_has_no_autocorrelation():
    cfg = Config(n=40, k=5, T=800, alpha=0.0)
    assert abs(_lag1_corr(SeedPackage.generate(3, cfg).valuations)) < 0.05


@pytest.mark.parametrize("alpha", [0.5, 0.9])
def test_proposal_ar1_shrinks_marginal_variance_as_documented(alpha):
    cfg = Config(n=60, k=5, T=600, alpha=alpha, ar1_mode="proposal")
    v = SeedPackage.generate(4, cfg).valuations
    expected_std = math.sqrt(1 / 12) * math.sqrt((1 - alpha) / (1 + alpha))
    assert abs(v.std() - expected_std) / expected_std < 0.1
    assert abs(v.mean() - 0.5) < 0.01


@pytest.mark.parametrize("alpha", [0.5, 0.9])
def test_copula_ar1_preserves_the_uniform_marginal(alpha):
    cfg = Config(n=60, k=5, T=600, alpha=alpha, ar1_mode="copula")
    v = SeedPackage.generate(4, cfg).valuations
    assert abs(v.mean() - 0.5) < 0.02
    assert abs(v.std() - math.sqrt(1 / 12)) < 0.02
    # uniform deciles each hold ~10% of the mass
    hist, _ = np.histogram(v, bins=10, range=(0, 1))
    assert hist.min() / v.size > 0.08


def test_copula_ar1_preserves_beta_marginals():
    cfg = Config(n=40, k=5, T=800, alpha=0.9, ar1_mode="copula", valuation_dist="beta_high")
    v = SeedPackage.generate(4, cfg).valuations
    assert abs(v.mean() - 5 / 7) < 0.02


# ── History ───────────────────────────────────────────────────────────────────

def test_history_cumulative_and_wait():
    h, p = History(4), np.zeros(4)
    h.update(np.array([1, 0, 0, 0]), p)
    np.testing.assert_array_equal(h.consecutive_wait, [0, 1, 1, 1])
    h.update(np.array([0, 1, 0, 0]), p)
    np.testing.assert_array_equal(h.consecutive_wait, [1, 0, 2, 2])
    np.testing.assert_array_equal(h.cumulative, [1, 1, 0, 0])
    assert h.round == 2 and len(h.allocations) == 2 and len(h.payments) == 2


def test_history_stores_copies():
    h, x, p = History(3), np.array([1, 0, 1]), np.zeros(3)
    h.update(x, p)
    x[:] = 0
    np.testing.assert_array_equal(h.allocations[0], [1, 0, 1])


def test_history_reset_and_snapshot():
    h, p = History(3), np.zeros(3)
    h.update(np.array([1, 1, 0]), p)
    snap = h.snapshot()
    h.reset()
    np.testing.assert_array_equal(h.cumulative, [0, 0, 0])
    assert h.round == 0 and not h.allocations
    np.testing.assert_array_equal(snap.cumulative, [1, 1, 0])
    assert snap.round == 1 and not snap.allocations


def test_cumulative_never_exceeds_rounds():
    cfg = Config(n=8, k=3, T=40)
    h = History(cfg.n)
    rng = np.random.default_rng(0)
    for _ in range(cfg.T):
        x = np.zeros(cfg.n, dtype=np.int64)
        x[rng.choice(cfg.n, cfg.k, replace=False)] = 1
        h.update(x, np.zeros(cfg.n))
    assert h.cumulative.max() <= cfg.T and h.cumulative.sum() == cfg.k * cfg.T


# ── JSON persistence ──────────────────────────────────────────────────────────

def test_to_jsonable_handles_numpy_and_nonfinite():
    obj = {"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1.0, np.nan, np.inf]),
           "d": (np.bool_(True), None), "e": float("nan"), 7: "k"}
    out = to_jsonable(obj)
    assert out == {"a": 1.5, "b": 3, "c": [1.0, None, None], "d": [True, None], "e": None, "7": "k"}
    json.dumps(out, allow_nan=False)


def test_save_json_is_strict_and_creates_directories(tmp_path):
    path = save_json({"x": float("nan"), "y": np.arange(3)}, tmp_path / "a" / "b" / "r.json")
    text = path.read_text()
    assert "NaN" not in text
    assert json.loads(text) == {"x": None, "y": [0, 1, 2]}
