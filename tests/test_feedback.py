"""Regression checks for reviewer feedback and controlled timing inference."""
import json

import numpy as np
import pytest

from analysis.bootstrap import paired_differences
from experiments.e19_value_threshold import controlled_timing_row, run_e19
from sim.config import Config
from sim.environment import SeedPackage


def test_timing_sacrifice_has_hand_computed_history_benefit():
    # Boost wins value .6 now, loses value 1 later. Sandbag preserves priority
    # and wins the later round. Opponent 1 always boosts; opponent 2 is truthful.
    cfg = Config(n=3, k=1, T=2, n_focal=1, lambda_=1)
    pkg = SeedPackage(np.array([[.6, 1.], [.2, .9], [.1, .1]]),
                      np.array([0, 1]), np.array([0, 1]), 7)
    row = controlled_timing_row("ScoreMechanism", cfg, pkg, .75)
    assert row["timing_premium"] == pytest.approx(.4)
    assert row["sandbag_gain"] == pytest.approx(.4)
    assert row["boost_gain"] == 0
    record = row["focal_results"][0]
    assert record["boost_utility"] == .6 and record["sandbag_utility"] == 1
    assert pkg.strategic_set.tolist() == [0, 1]


@pytest.mark.parametrize("mechanism", ["GreedyMechanism", "VickreyMechanism"])
def test_stateless_controls_cannot_gain_by_suppressing_low_reports(mechanism):
    cfg = Config(n=8, k=2, T=40)
    row = controlled_timing_row(mechanism, cfg, SeedPackage.generate(19, cfg), .9)
    assert row["timing_premium"] <= 1e-12
    if mechanism == "VickreyMechanism":
        assert row["sandbag_gain"] <= 1e-12 and row["boost_gain"] <= 1e-12


@pytest.mark.parametrize("mechanism", ["GreedyMechanism", "ScoreMechanism", "VickreyMechanism",
                                      "CreditVickreyMechanism"])
def test_batched_control_matches_independent_episode_replays(mechanism):
    from sim.adaptation import run_profile
    from sim.metrics import utilities
    cfg = Config(n=8, k=2, T=35, rho=.5, credit_horizon=2)
    pkg = SeedPackage.generate(23, cfg)
    original_values = pkg.valuations.copy()
    row = controlled_timing_row(mechanism, cfg, pkg, .75)
    for record in row["focal_results"]:
        profile = np.zeros(cfg.n, dtype=int)
        profile[pkg.strategic_set] = 1
        for policy, key in enumerate(("truth_utility", "boost_utility", "sandbag_utility")):
            profile[record["focal"]] = policy
            history = run_profile(mechanism, pkg, cfg, profile,
                                  names=("truthful", "boost_0.75", "sandbag_0.75"))
            assert record[key] == pytest.approx(utilities(history, pkg.valuations)[record["focal"]])
    np.testing.assert_array_equal(pkg.valuations, original_values)


def test_e19_saves_controlled_pairs_trends_and_actual_config(tmp_path):
    run_e19([11, 17], tmp_path, n=4, k=1, T=10, n_bootstrap=25, verbose=False,
            settings=[("ScoreMechanism", {"lambda_": lam}, f"M4 lambda={lam:g}")
                      for lam in (.5, 2)], taus=[.75])
    directory = tmp_path / "e19"
    controlled = json.loads((directory / "controlled.json").read_text())
    assert len(controlled) == 2
    paired = json.loads((directory / "paired.json").read_text())
    trend = next(r for r in paired if r["comparison"] == "lambda_effect")
    assert trend["metrics"]["timing_premium"]["mean_diff"] == pytest.approx(
        controlled[1]["timing_premium"]["mean"] - controlled[0]["timing_premium"]["mean"])
    for path in directory.rglob("*.json"):
        payload = json.loads(path.read_text(), parse_constant=lambda v: pytest.fail(v))
        if path.parent.name == "raw":
            assert payload["config"]["n_seeds"] == 2
            assert payload["config"]["n_bootstrap"] == 25
    from analysis.verify_feedback import verify_feedback
    audit = verify_feedback(tmp_path, experiments=("e19",))
    assert audit["passed"] and audit["controlled_cells"] == 2
    from analysis.report import build_report
    report = build_report(tmp_path)
    assert "Timing premium [95% CI]" in report and "Premium change [95% CI]" in report
    assert "opponents differ" in report


def test_e20_config_and_audit_describe_actual_run(tmp_path):
    from experiments.e20_credit_budget import run_e20
    from analysis.verify_feedback import verify_feedback
    run_e20([19], tmp_path, n=4, k=1, T=8, n_bootstrap=25, verbose=False,
            settings=[("CreditVickreyMechanism", {"credit_rate": .25}, "M5c rate=0.25")])
    assert verify_feedback(tmp_path, experiments=("e20",))["profile_cells"] == 5


def test_paired_comparison_rejects_reordered_seed_ids():
    with pytest.raises(ValueError, match="aligned"):
        paired_differences({"base": [{"seed": 1, "x": 0}, {"seed": 2, "x": 1}],
                            "other": [{"seed": 2, "x": 0}, {"seed": 1, "x": 1}]},
                           "base", ["x"], n_resamples=10)
