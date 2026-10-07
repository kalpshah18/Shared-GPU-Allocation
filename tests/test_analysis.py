"""
tests/test_analysis.py
======================
Unit tests for analysis/pareto.py and analysis/bootstrap.py.
"""

import sys
sys.path.insert(0, ".")

import numpy as np

from analysis.bootstrap import paired_bootstrap_ci
from analysis.pareto import filter_e2_results, filter_results, pareto_front


def test_pareto_front_basic():
    pts = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 0.5], [0.5, 3.0]])
    front = pareto_front(pts, [True, True])
    assert set(front.tolist()) == {1, 2, 3}      # (1,1) is dominated by (2,2)


def test_pareto_minimise_direction():
    pts = np.array([[1.0, 5.0], [2.0, 1.0]])      # maximise x, minimise y
    assert pareto_front(pts, [True, False]).tolist() == [1]


def test_pareto_equal_points_not_dominated():
    pts = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert len(pareto_front(pts, [True, True])) == 2


def _row(name, wr, ja, sr, m):
    return {"mechanism": name, "lambda_": None,
            "WR": {"mean": wr}, "J_A": {"mean": ja},
            "SR_delta": {"mean": sr}, "M_uni": {"mean": m}, "M_mean": {"mean": m}}


def test_filter_e2_results_uses_all_four_objectives():
    rows = [_row("A", 0.9, 0.9, 0.1, 10.0),
            _row("B", 0.9, 0.9, 0.1, -5.0),       # same as A but lower manipulation gain
            _row("C", 0.5, 1.0, 0.0, 0.0)]        # best fairness/starvation, worst welfare
    names = {r["mechanism"] for r in filter_e2_results(rows)}
    assert names == {"B", "C"}


def test_filter_results_custom_keys():
    rows = [_row("A", 0.9, 0.5, 0.1, 0.0), _row("B", 0.8, 0.9, 0.1, 0.0)]
    assert len(filter_results(rows, ["WR", "J_A"], [True, True])) == 2


def test_paired_bootstrap_detects_consistent_difference():
    a = np.array([1.1, 1.2, 1.15, 1.05, 1.12])
    b = a - 0.1
    ci = paired_bootstrap_ci(a, b, n_resamples=500)
    assert abs(ci["mean_diff"] - 0.1) < 1e-12
    assert ci["ci_lower"] > 0
