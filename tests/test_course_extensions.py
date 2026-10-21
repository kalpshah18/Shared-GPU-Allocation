"""Independent trajectory, exhaustive oracle and game-payoff checks."""
import copy
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis.constrained_oracle import constrained_welfare, terminal_max_wait
from experiments.extensions import TRAINING_SEEDS, extension_seeds
from sim.adaptation import ProfileEvaluator, best_response_dynamics, run_profile
from sim.config import Config
from sim.environment import History, SeedPackage
from sim.mechanisms import KarmaCapMechanism, RankCapMechanism, ScoreCapMechanism, ScoreMechanism, VickreyMechanism
from sim.metrics import utilities
from sim.policies import make_timed, rollout_report
from sim.runner import run_mixed

ROOT = Path(__file__).resolve().parent.parent


def test_holdout_is_locked_reproducible_and_disjoint():
    data = json.loads((ROOT / "seeds/extension_seeds.json").read_text())
    old = json.loads((ROOT / "seeds/master_seeds.json").read_text())["seeds"]
    assert len(extension_seeds()) == len(set(extension_seeds())) == 30
    assert not set(extension_seeds()) & set(old + TRAINING_SEEDS + list(range(101, 106)))
    expected = np.random.default_rng(data["meta_seed"]).integers(0, 2**31, size=30).tolist()
    assert expected == data["seeds"] and data["training_seeds"] == TRAINING_SEEDS


@pytest.mark.parametrize("name", ["enable_wait_cap", "rank_rollout_update"])
def test_ablation_switches_are_boolean(name):
    with pytest.raises(ValueError):
        Config(**{name: "false"})


def test_disabled_score_cap_reproduces_score_over_an_episode():
    cfg = Config(n=7, k=2, T=60, enable_wait_cap=False)
    pkg = SeedPackage.generate(14, cfg)
    a = run_mixed(ScoreMechanism(cfg), pkg, cfg, None)
    b = run_mixed(ScoreCapMechanism(cfg), pkg, cfg, None)
    np.testing.assert_array_equal(a.allocations, b.allocations)


@pytest.mark.parametrize("cls", [ScoreCapMechanism, KarmaCapMechanism, RankCapMechanism])
def test_multistep_rollouts_match_independent_real_mechanisms(cls):
    cfg = Config(n=5, k=2, T=100, wait_cap=3)
    initial, h = cls(cfg), History(5)
    rng = np.random.default_rng(7)
    for t in range(6):
        x, p = initial.allocate(rng.random(5), h, t)
        h.update(x, p)
    mechanisms = [copy.deepcopy(initial) for _ in range(3)]
    histories = [h.snapshot() for _ in range(3)]
    state = initial.rollout_init(h, 3)
    for step in range(10):
        reports = rng.choice([0.0, 0.3, 0.7, 1.0], size=(3, 5))
        seeds = rng.integers(0, 100000, size=3)
        tb = np.stack([np.random.default_rng(int(s)).random(5) for s in seeds])
        xb, pb = initial.rollout_step(reports, state, tb)
        for b in range(3):
            x, p = mechanisms[b].allocate(reports[b], histories[b], int(seeds[b]))
            np.testing.assert_array_equal(xb[b], x)
            np.testing.assert_allclose(pb[b], p)
            histories[b].update(x, p)
            np.testing.assert_array_equal(state["wait"][b], histories[b].consecutive_wait)
            np.testing.assert_array_equal(state["cum"][b], histories[b].cumulative)
            if cls == KarmaCapMechanism:
                np.testing.assert_allclose(state["bal"][b], mechanisms[b].balance)


def test_rank_frozen_control_keeps_hypothetical_history_empty():
    cfg = Config(n=3, k=1, T=4, rank_rollout_update=False)
    m = RankCapMechanism(cfg)
    state = m.rollout_init(History(3), 1)
    m.rollout_step(np.array([[0.1, 0.2, 0.3]]), state, np.array([[0.1, 0.2, 0.3]]))
    assert not state["future"]


@pytest.mark.parametrize("cls", [ScoreCapMechanism, KarmaCapMechanism, RankCapMechanism])
def test_service_bound_holds_with_adversarial_zero_reports_and_large_scale(cls):
    cfg = Config(n=9, k=2, T=100, v_max=1e20, wait_cap=3)
    m, h = cls(cfg), History(cfg.n)
    for t in range(cfg.T):
        r = np.full(cfg.n, cfg.v_max)
        r[0] = 0
        x, p = m.allocate(r, h, t)
        h.update(x, p)
        assert h.consecutive_wait.max() <= cfg.wait_limit + math.ceil(cfg.n / cfg.k) - 1


def test_rank_invariance_with_active_service_override():
    cfg = Config(n=7, k=2, T=80, wait_cap=3)
    pkg = SeedPackage.generate(31, cfg)
    a = run_mixed(RankCapMechanism(cfg), pkg, cfg, None)
    b = run_mixed(RankCapMechanism(cfg), pkg, cfg, lambda v, h, c: v ** 3, np.array([0, 2, 5]))
    np.testing.assert_array_equal(a.allocations, b.allocations)


def test_clipping_can_raise_a_quantile_so_is_outside_invariance_theorem():
    assert RankCapMechanism._quantile([1.0, 1.0], 1.0) > RankCapMechanism._quantile([0.6, 0.9], 0.55)


def test_rollout_truncates_and_uses_current_simulated_opponent_history():
    cfg = Config(n=4, k=1, T=3)
    h = History(4)
    h.round = 1
    seen = []

    def opponent(v, history, config):
        seen.append((history.round, history.cumulative.copy()))
        return v

    rollout_report(0.37, h, ScoreMechanism(cfg), cfg, 0, opponent, np.random.default_rng(8), H=80, n_rollouts=5)
    assert [r for r, _ in seen] == [1, 2]
    assert np.all(seen[0][1].sum(axis=1) == 0)
    assert np.all(seen[1][1].sum(axis=1) == 1)
    assert h.round == 1 and np.all(h.cumulative == 0)
    h.round = 3
    with pytest.raises(ValueError, match="No allocation"):
        rollout_report(0.37, h, ScoreMechanism(cfg), cfg, 0, opponent, np.random.default_rng(8))


def test_timed_opponents_use_each_scenarios_threshold():
    cfg = Config(n=3, k=1)
    h = SimpleNamespace(cumulative=np.array([[0, 3, 3], [3, 0, 0]]))
    out = make_timed(2)(np.full((2, 3), 0.2), h, cfg)
    np.testing.assert_allclose(out, [[0.4, 0.2, 0.2], [0.2, 0.4, 0.4]])


def test_vickrey_rollout_is_exact_truth_for_off_grid_values():
    cfg = Config(n=5, k=2, T=6)
    for v in [0.037, 0.291, 0.839]:
        assert rollout_report(v, History(5), VickreyMechanism(cfg), cfg, 0, lambda v, h, c: v,
                              np.random.default_rng(2), H=4, n_rollouts=30) == v


def test_profile_payoffs_match_a_manual_mixed_counterfactual():
    cfg = Config(n=3, k=1, T=20)
    pkg = SeedPackage.generate(3, cfg)
    a = run_profile("ScoreMechanism", pkg, cfg, [0, 1, 0], names=("truthful", "cap_2"))
    b = run_mixed(ScoreMechanism(cfg), pkg, cfg, lambda v, h, c: np.minimum(2 * v, 1), np.array([1]))
    np.testing.assert_array_equal(a.allocations, b.allocations)


def test_best_response_training_is_deterministic_and_vickrey_converges_to_truth():
    cfg = Config(n=3, k=1, T=20)
    names = ("truthful", "cap_2")
    a = best_response_dynamics(ProfileEvaluator("VickreyMechanism", cfg, [201, 202], names), [1, 1, 1], 4)
    b = best_response_dynamics(ProfileEvaluator("VickreyMechanism", cfg, [201, 202], names), [1, 1, 1], 4)
    assert a == b and a["status"] == "converged" and a["profile"] == [0, 0, 0]
    assert a["training_exploitability"] == 0


def test_exploitability_averages_before_selecting_the_best_policy():
    from experiments.e13_adaptive import exploitability_band
    gains = np.array([[[0, 10, -10]], [[0, -10, 10]]])
    result = exploitability_band(gains, 100)
    assert result["mean"] == 0 and result["ci_upper"] > 0


@pytest.mark.parametrize("B", [2, 3, 4])
def test_milp_oracle_matches_exhaustive_schedule_enumeration(B):
    values = np.array([[0.1, 0.9, 0.3, 0.6], [0.7, 0.4, 0.5, 0.2], [0.4, 0.6, 0.8, 0.9]])
    best = -np.inf
    for schedule in itertools.product(range(3), repeat=4):
        X = np.zeros((3, 4), dtype=int)
        X[list(schedule), np.arange(4)] = 1
        if terminal_max_wait(X) <= B:
            best = max(best, float(np.sum(X * values)))
    result = constrained_welfare(values, 1, B)
    assert result.optimal and result.welfare == pytest.approx(best)
    assert result.upper_bound == pytest.approx(best)


def test_oracle_rejects_impossible_capacity_and_checks_terminal_state():
    with pytest.raises(ValueError, match="infeasible"):
        constrained_welfare(np.ones((3, 6)), 1, 1)
    X = np.array([[1, 1, 1], [0, 0, 0]])
    assert terminal_max_wait(X) == 3  # pre-round maximum would be only 2
    result = constrained_welfare(np.array([[0.9], [0.1], [0.2]]), 1, 1)
    assert result.welfare == pytest.approx(0.9)


@pytest.mark.parametrize("experiment", [12, 13, 14, 15])
def test_new_experiments_write_strict_raw_configs_and_paired_results(experiment, tmp_path):
    from experiments.e12_ablations import run_e12
    from experiments.e13_adaptive import run_e13
    from experiments.e14_constrained import run_e14
    from experiments.e15_attack_budget import run_e15
    seeds = extension_seeds(2)
    kw = dict(results_dir=str(tmp_path), n_bootstrap=50, verbose=False)
    if experiment == 12:
        rows = run_e12(seeds, n=4, T=12, caps=[4], **kw)
    elif experiment == 13:
        rows = run_e13(seeds, n=3, T=10, training_seeds=[201], max_sweeps=1,
                       specs=[("Score-Cap", "ScoreCapMechanism", {}), ("Vickrey", "VickreyMechanism", {})], **kw)
    elif experiment == 14:
        rows = run_e14(seeds, n=3, T=8, caps=[3], conditions={"base": {}}, **kw)
        assert all(r["oracle_optimal"]["mean"] == 1 for r in rows)
    else:
        rows = run_e15(seeds, n=4, T=8, budgets=[(2, 4), (4, 8)],
                       specs=[("Rank-Cap", "RankCapMechanism", {}), ("Score", "ScoreMechanism", {})], **kw)
        assert {r["mode"] for r in rows} == {"updating", "frozen", "exact"}
    assert rows and all(r["n_seeds"] == 2 for r in rows)
    directory = tmp_path / f"e{experiment}"
    assert (directory / "paired.json").exists()
    cfg = json.loads((directory / "config.json").read_text())
    assert cfg["seed_role"] == "heldout" and cfg["seeds"] == seeds and len(cfg["source_sha256"]) == 64
    assert cfg["n_bootstrap"] == 50
    for path in (directory / "raw").glob("*.json"):
        saved = json.loads(path.read_text())["config"]
        assert saved["seeds_file"] == "seeds/extension_seeds.json"
        assert saved["n_seeds"] == len(seeds) and saved["n_bootstrap"] == 50
    for path in directory.rglob("*.json"):
        json.loads(path.read_text(), parse_constant=lambda v: pytest.fail(f"Non-finite JSON: {v}"))
    from analysis.course_plots import run_course_plots, PLOTS
    from analysis.report import build_report
    figures = tmp_path / "figures"
    run_course_plots(str(tmp_path), str(figures))
    stem = next(stem for exp, stem, _ in PLOTS if exp == f"e{experiment}")
    assert all((figures / f"{stem}.{suffix}").stat().st_size > 2000 for suffix in ("png", "pdf"))
    assert f"## E{experiment}" in build_report(str(tmp_path))


def test_adaptive_study_refuses_training_seed_leakage():
    from experiments.e13_adaptive import run_e13
    with pytest.raises(ValueError, match="disjoint"):
        run_e13([201], training_seeds=[201])
