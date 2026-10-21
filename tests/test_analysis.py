"""
tests/test_analysis.py — Owner: Kalp Shah

Bootstrap confidence intervals, paired differences and the Pareto filter.
"""

import numpy as np
import pytest

from analysis.bootstrap import (
    bootstrap_ci, paired_bootstrap_ci, paired_differences, summarise_seeds,
)
from analysis.pareto import filter_e2_results, pareto_flags, pareto_front


# ── bootstrap_ci ──────────────────────────────────────────────────────────────

def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    x = np.random.default_rng(0).normal(5.0, 2.0, 30)
    a, b = bootstrap_ci(x, n_resamples=2000), bootstrap_ci(x, n_resamples=2000)
    assert a == b
    assert a["mean"] == pytest.approx(x.mean())
    assert a["ci_lower"] < a["mean"] < a["ci_upper"]
    assert a["std"] == pytest.approx(x.std(ddof=1))


def test_bootstrap_ci_constant_data_has_zero_width():
    out = bootstrap_ci(np.full(30, 3.0), n_resamples=500)
    assert out["ci_lower"] == out["ci_upper"] == out["mean"] == 3.0 and out["std"] == 0.0


def test_bootstrap_ci_width_shrinks_with_more_data():
    rng = np.random.default_rng(1)
    def width(n):
        ci = bootstrap_ci(rng.normal(0, 1, n), 2000)
        return ci["ci_upper"] - ci["ci_lower"]
    assert width(400) < width(20)


def test_bootstrap_ci_coverage_is_roughly_nominal():
    rng = np.random.default_rng(2)
    hits = 0
    trials = 150
    for _ in range(trials):
        ci = bootstrap_ci(rng.normal(0.0, 1.0, 30), n_resamples=400, seed=int(rng.integers(1e6)))
        hits += ci["ci_lower"] <= 0.0 <= ci["ci_upper"]
    assert 0.88 <= hits / trials <= 0.99


def test_bootstrap_ci_seed_changes_resamples_not_the_mean():
    x = np.random.default_rng(3).random(30)
    a, b = bootstrap_ci(x, 500, seed=1), bootstrap_ci(x, 500, seed=2)
    assert a["mean"] == b["mean"] and a["ci_lower"] != b["ci_lower"]


def test_bootstrap_ci_single_value():
    out = bootstrap_ci(np.array([4.0]), 100)
    assert out["mean"] == 4.0 and out["std"] == 0.0


def test_ci_level_changes_width():
    x = np.random.default_rng(4).random(40)
    wide, narrow = bootstrap_ci(x, 2000, ci=99.0), bootstrap_ci(x, 2000, ci=50.0)
    assert wide["ci_upper"] - wide["ci_lower"] > narrow["ci_upper"] - narrow["ci_lower"]


# ── paired_bootstrap_ci ───────────────────────────────────────────────────────

def test_paired_difference_of_a_constant_shift_is_exact_with_zero_width():
    b = np.random.default_rng(0).random(30)
    out = paired_bootstrap_ci(b + 0.25, b, n_resamples=500)
    assert out["mean_diff"] == pytest.approx(0.25)
    assert out["ci_lower"] == pytest.approx(0.25) and out["ci_upper"] == pytest.approx(0.25)
    assert out["std_diff"] == pytest.approx(0.0, abs=1e-12)


def test_pairing_beats_unpaired_when_seeds_are_correlated():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 5.0, 30)                              # large between-seed variation
    a, b = base + 0.3 + rng.normal(0, 0.05, 30), base
    paired = paired_bootstrap_ci(a, b, 3000)
    unpaired_a, unpaired_b = bootstrap_ci(a, 3000), bootstrap_ci(b, 3000)
    assert (paired["ci_upper"] - paired["ci_lower"]) < 0.5 * (unpaired_a["ci_upper"] - unpaired_a["ci_lower"])
    assert paired["ci_lower"] > 0                               # the small effect is detected


def test_paired_ci_accepts_lists():
    out = paired_bootstrap_ci([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], 100)
    assert out["mean_diff"] == pytest.approx(1.0)


# ── summarise_seeds / paired_differences ──────────────────────────────────────

def test_summarise_seeds_drops_nan_and_returns_none_when_empty():
    rows = [{"a": 1.0, "b": float("nan")}, {"a": float("nan"), "b": float("nan")},
            {"a": 3.0, "b": float("nan")}]
    s = summarise_seeds(rows, ["a", "b"], n_resamples=100)
    assert s["a"]["mean"] == 2.0
    assert s["b"] == {"mean": None, "ci_lower": None, "ci_upper": None, "std": None}


def test_summaries_are_strict_json():
    import json
    rows = [{"a": float("nan")}] * 3
    json.dumps(summarise_seeds(rows, ["a"], 50), allow_nan=False)


def _rows(vals, key="WR"):
    return [{key: v, "seed": i} for i, v in enumerate(vals)]


def test_paired_differences_basic_and_reference_excluded():
    base = np.random.default_rng(0).random(20)
    out = paired_differences({"ref": _rows(base), "x": _rows(base + 1.0), "y": _rows(base - 0.5)},
                             "ref", ["WR"], n_resamples=200)
    assert set(out) == {"x", "y"}
    assert out["x"]["WR"]["mean_diff"] == pytest.approx(1.0)
    assert out["y"]["WR"]["mean_diff"] == pytest.approx(-0.5)


def test_paired_differences_drops_nan_pairs_and_handles_all_nan():
    ref = _rows([1.0, 2.0, float("nan"), 4.0])
    x = _rows([2.0, 3.0, 5.0, float("nan")])
    out = paired_differences({"ref": ref, "x": x}, "ref", ["WR"], n_resamples=100)
    assert out["x"]["WR"]["mean_diff"] == pytest.approx(1.0)   # only the first two pairs survive
    none = paired_differences({"ref": _rows([float("nan")] * 3), "x": _rows([1.0] * 3)}, "ref", ["WR"], 50)
    assert none["x"]["WR"]["mean_diff"] is None


def test_paired_differences_rejects_misaligned_groups():
    with pytest.raises(ValueError):
        paired_differences({"ref": _rows([1.0, 2.0]), "x": _rows([1.0])}, "ref", ["WR"], 50)


# ── Pareto ────────────────────────────────────────────────────────────────────

def test_pareto_front_simple_cases():
    pts = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 0.5], [0.5, 3.0]])
    assert set(pareto_front(pts, [True, True]).tolist()) == {1, 2, 3}
    assert set(pareto_front(pts, [False, False]).tolist()) == {0, 2, 3}
    # maximise x, minimise y: (3, .5) beats (1,1), (2,2); (0.5,3) is worst on both -> only point 2 survives
    assert pareto_front(pts, [True, False]).tolist() == [2]


def test_pareto_direction_semantics():
    pts = np.array([[1.0, 5.0], [2.0, 5.0]])
    assert pareto_front(pts, [True, True]).tolist() == [1]
    assert pareto_front(pts, [False, True]).tolist() == [0]


def test_pareto_keeps_duplicates_and_ties():
    pts = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    assert pareto_front(pts, [True, True]).tolist() == [0, 1]


def test_pareto_validates_dimension():
    with pytest.raises(ValueError):
        pareto_front(np.zeros((3, 2)), [True])


def test_pareto_front_matches_bruteforce_on_random_points():
    rng = np.random.default_rng(0)
    pts = rng.random((60, 4))
    maximize = [True, True, False, False]
    sign = np.array([1, 1, -1, -1])
    front = set(pareto_front(pts, maximize).tolist())
    for i in range(60):
        dominated = any(np.all(pts[j] * sign >= pts[i] * sign) and np.any(pts[j] * sign > pts[i] * sign)
                        for j in range(60) if j != i)
        assert (i in front) == (not dominated)


def _row(wr, ja, sr, gain, **extra):
    return {"WR": {"mean": wr}, "J_A": {"mean": ja}, "SR_delta": {"mean": sr},
            "M_uni": {"mean": gain}, "M_mean": {"mean": gain}, **extra}


def test_pareto_flags_with_summary_dicts():
    rows = [_row(1.0, 0.5, 0.3, 100.0),      # greedy-like: efficient, unfair, manipulable
            _row(0.96, 0.99, 0.07, 2.0),     # score lambda=1
            _row(0.95, 0.99, 0.07, 5.0),     # dominated by the previous row
            _row(0.56, 1.0, 0.0, 0.0)]       # round-robin-like
    assert pareto_flags(rows) == [True, True, False, True]
    assert filter_e2_results(rows) == [rows[0], rows[1], rows[3]]


def test_pareto_flags_gain_key_switch_and_plain_floats():
    rows = [{"WR": 1.0, "J_A": 1.0, "SR_delta": 0.0, "M_mean": 9.0, "M_uni": 0.0},
            {"WR": 1.0, "J_A": 1.0, "SR_delta": 0.0, "M_mean": 0.0, "M_uni": 9.0}]
    assert pareto_flags(rows, gain_key="M_uni") == [True, False]
    assert pareto_flags(rows, gain_key="M_mean") == [False, True]


def test_pareto_flags_reject_missing_objectives():
    with pytest.raises(ValueError):
        pareto_flags([_row(1.0, 1.0, 0.0, None)])
