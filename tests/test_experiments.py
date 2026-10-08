"""
tests/test_experiments.py — Owner: Kalp Shah

End-to-end tests of every experiment on tiny instances: output files and
schema, strict JSON, determinism, and the qualitative properties each
experiment must reproduce even at small scale.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from experiments import (
    e0_validation, e1_scarcity, e2_frontier, e3_strategic, e3b_rollout, e3c_rollout_horizon,
    e4_heterogeneous, e5_persistence, e6_scalability, e7_sensitivity, e8_timing,
)
from experiments.common import (
    STRATEGIC_KEYS, TRUTHFUL_KEYS, load_seeds, scaled_k, strategic_row, truthful_row,
)
from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import MECHANISM_NAMES
from sim.policies import make_capped

SEEDS = load_seeds(n=3)
KW = dict(n=20, T=150, n_bootstrap=200, verbose=False)


def by(rows, **conds):
    out = [r for r in rows if all(r.get(k) == v for k, v in conds.items())]
    assert len(out) == 1, (conds, len(out))
    return out[0]


def mean(row, key):
    return row[key]["mean"]


def read(path):
    return json.loads(Path(path).read_text())          # json.loads would accept NaN; checked below


def assert_strict_json(path):
    json.loads(Path(path).read_text(), parse_constant=lambda c: pytest.fail(f"{c} in {path}"))


# ── common helpers ────────────────────────────────────────────────────────────

def test_load_seeds_returns_the_30_locked_seeds():
    seeds = load_seeds()
    assert len(seeds) == 30 and len(set(seeds)) == 30
    assert load_seeds(n=5) == seeds[:5]


@pytest.mark.parametrize("n,ratio,k", [(50, 0.1, 5), (50, 0.2, 10), (50, 0.8, 40), (10, 0.95, 9), (10, 0.0, 1)])
def test_scaled_k_clamps(n, ratio, k):
    assert scaled_k(n, ratio) == k


def test_strategic_row_truthful_policy_is_exact_zero_gain():
    cfg = Config(n=20, k=4, T=60, rho=0.25)
    row = strategic_row("GreedyMechanism", cfg, SeedPackage.generate(1, cfg), None)
    assert set(row) == set(STRATEGIC_KEYS)
    for key in ("M_mean", "M_max", "frac_pos", "M_uni", "M_uni_max", "frac_pos_uni", "PoS"):
        assert row[key] == 0.0


def test_strategic_row_at_rho_zero_has_nan_coalition_gain_but_defined_unilateral():
    cfg = Config(n=20, k=4, T=60, rho=0.0)
    row = strategic_row("GreedyMechanism", cfg, SeedPackage.generate(1, cfg), make_capped(2.0))
    assert math.isnan(row["M_mean"]) and math.isnan(row["frac_pos"])
    assert row["M_uni"] > 0


def test_truthful_row_has_all_keys():
    cfg = Config(n=20, k=4, T=60)
    assert set(truthful_row("ScoreMechanism", cfg, SeedPackage.generate(1, cfg))) == set(TRUTHFUL_KEYS)


# ── E0 ────────────────────────────────────────────────────────────────────────

def test_e0_all_checks_pass():
    results = e0_validation.run_all_checks(verbose=False)
    assert len(results) == len(e0_validation.CHECKS) == 10
    for name, n_inst, fails in results:
        assert n_inst > 0, name
        assert not fails, (name, fails[:3])


def test_e0_negative_control_detects_a_broken_vickrey(monkeypatch):
    """If M5 charged the winner's own bid, DSIC must fail: E0 can fail."""
    from sim.mechanisms import VickreyMechanism

    original = VickreyMechanism.allocate

    def first_price(self, reports, history, tie_seed):
        x, p = original(self, reports, history, tie_seed)
        return x, x * reports                                   # pay your own bid

    monkeypatch.setattr(VickreyMechanism, "allocate", first_price)
    _, fails = e0_validation.check_vickrey_dsic()
    assert fails


def test_e0_detects_a_capacity_bug(monkeypatch):
    from sim.mechanisms import GreedyMechanism

    original = GreedyMechanism.allocate

    def drops_one(self, reports, history, tie_seed):
        x, p = original(self, reports, history, tie_seed)
        x = x.copy()
        x[np.flatnonzero(x)[0]] = 0
        return x, p

    monkeypatch.setattr(GreedyMechanism, "allocate", drops_one)
    _, fails = e0_validation.check_capacity_and_payment()
    assert fails


# ── E1 ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e1(tmp_path_factory):
    d = tmp_path_factory.mktemp("e1")
    return e1_scarcity.run_e1(SEEDS, str(d), **KW), d


def test_e1_schema_and_files(e1):
    rows, d = e1
    assert len(rows) == len(e1_scarcity.KN_RATIOS) * 5
    for r in rows:
        assert set(TRUTHFUL_KEYS) <= set(r) and r["n_seeds"] == 3
        for key in TRUTHFUL_KEYS:
            assert set(r[key]) == {"mean", "ci_lower", "ci_upper", "std"}
    for f in ("summary.json", "paired.json", "config.json"):
        assert_strict_json(d / "e1" / f)
    raw = read(d / "e1" / "raw" / "GreedyMechanism_kn_0.1.json")
    assert len(raw["rows"]) == 3 and raw["config"]["n"] == 20
    assert [r["seed"] for r in raw["rows"]] == SEEDS
    cfg = read(d / "e1" / "config.json")
    assert cfg["seeds"] == SEEDS and cfg["n_seeds"] == 3


def test_e1_qualitative_properties(e1):
    rows, _ = e1
    for kn in e1_scarcity.KN_RATIOS:
        k = scaled_k(20, kn)
        for m in ("GreedyMechanism", "VickreyMechanism"):
            assert mean(by(rows, mechanism=m, kn_ratio=kn), "WR") == pytest.approx(1.0)
        rr = by(rows, mechanism="RoundRobinMechanism", kn_ratio=kn)
        assert mean(rr, "Q_max") <= math.ceil(20 / k) - 1 and mean(rr, "SR_delta") == 0.0
        assert mean(rr, "J_A") > 0.97
        assert mean(by(rows, mechanism="ScoreMechanism", kn_ratio=kn), "WR") <= 1.0 + 1e-12
    # welfare of the report-blind rules rises with k/n
    wr = [mean(by(rows, mechanism="RandomMechanism", kn_ratio=kn), "WR") for kn in e1_scarcity.KN_RATIOS]
    assert wr == sorted(wr)


def test_e1_paired_file_references_score(e1):
    _, d = e1
    paired = read(d / "e1" / "paired.json")
    assert {p["mechanism"] for p in paired} == set(MECHANISM_NAMES) - {"ScoreMechanism"}
    assert all(p["reference"] == "ScoreMechanism" for p in paired)
    g = next(p for p in paired if p["mechanism"] == "GreedyMechanism" and p["kn_ratio"] == 0.2)
    assert g["metrics"]["WR"]["mean_diff"] >= 0 and g["metrics"]["J_A"]["mean_diff"] < 0


def test_e1_is_deterministic(tmp_path):
    a = e1_scarcity.run_e1(SEEDS[:2], None, **KW)
    b = e1_scarcity.run_e1(SEEDS[:2], None, **KW)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ── E2 ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e2(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2")
    return e2_frontier.run_e2(SEEDS, str(d), **KW), d


def test_e2_rows_pareto_and_files(e2):
    rows, d = e2
    assert len(rows) == 4 + len(e2_frontier.LAMBDA_VALUES)
    assert all(isinstance(r["pareto"], bool) for r in rows) and any(r["pareto"] for r in rows)
    for f in ("summary.json", "paired.json", "config.json"):
        assert_strict_json(d / "e2" / f)
    assert len(list((d / "e2" / "raw").glob("*.json"))) == len(rows)
    assert read(d / "e2" / "config.json")["policy"] == "cap_2"


def test_e2_lambda_zero_equals_greedy_allocation(e2):
    rows, _ = e2
    g = by(rows, mechanism="GreedyMechanism")
    m4 = by(rows, mechanism="ScoreMechanism", lambda_=0.0)
    for key in ("WR", "J_A", "SR_delta", "M_uni", "M_mean"):
        assert mean(g, key) == pytest.approx(mean(m4, key))


def test_e2_history_penalty_trades_fairness_and_manipulability(e2):
    rows, _ = e2
    lam = lambda l: by(rows, mechanism="ScoreMechanism", lambda_=l)
    ja = [mean(lam(l), "J_A") for l in (0.0, 0.5, 1.0, 2.0)]
    assert ja == sorted(ja) and ja[-1] > 0.95
    gain = [mean(lam(l), "M_uni") for l in (0.0, 1.0, 2.0, 5.0)]
    assert gain == sorted(gain, reverse=True)
    assert mean(by(rows, mechanism="VickreyMechanism"), "M_uni") < 0
    for m in ("RandomMechanism", "RoundRobinMechanism"):
        assert mean(by(rows, mechanism=m), "M_uni") == 0.0


def test_e2_paired_reference_is_lambda_zero(e2):
    _, d = e2
    paired = read(d / "e2" / "paired.json")
    assert all(p["reference"] == "ScoreMechanism_lam_0" for p in paired)
    p1 = next(p for p in paired if p["cell"] == "ScoreMechanism_lam_1")
    assert p1["metrics"]["J_A"]["ci_lower"] > 0                # fairness gain is significant
    assert p1["metrics"]["M_uni"]["mean_diff"] < 0


# ── E3 ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e3(tmp_path_factory):
    d = tmp_path_factory.mktemp("e3")
    return e3_strategic.run_e3(SEEDS, str(d), **KW), d


def test_e3_full_factorial_and_order_is_deterministic(e3):
    rows, d = e3
    assert len(rows) == 5 * 3 * 5
    assert [r["mechanism"] for r in rows[::15]] == MECHANISM_NAMES      # canonical M1..M5 order
    assert len({(r["mechanism"], r["policy"], r["rho"]) for r in rows}) == 75
    for f in ("summary.json", "paired.json", "config.json"):
        assert_strict_json(d / "e3" / f)
    assert len(list((d / "e3" / "raw").glob("*.json"))) == 75


def test_e3_report_invariant_mechanisms_are_unaffected(e3):
    rows, _ = e3
    for m in ("RandomMechanism", "RoundRobinMechanism"):
        for r in (r for r in rows if r["mechanism"] == m):
            assert mean(r, "M_uni") == 0.0 and mean(r, "PoS") == 0.0
            assert mean(r, "M_uni_max") == 0.0 and mean(r, "frac_pos_uni") == 0.0


def test_e3_truthful_policy_has_no_gain_or_welfare_loss(e3):
    rows, _ = e3
    for r in (r for r in rows if r["policy"] == "truthful"):
        assert mean(r, "M_uni") == 0.0 and mean(r, "PoS") == 0.0


def test_e3_greedy_rewards_and_vickrey_punishes_inflation(e3):
    rows, _ = e3
    for rho in (0.1, 0.25, 0.5, 1.0):
        for pol in ("cap_2", "max_claim"):
            assert mean(by(rows, mechanism="GreedyMechanism", policy=pol, rho=rho), "M_uni") > 0
            assert mean(by(rows, mechanism="VickreyMechanism", policy=pol, rho=rho), "M_uni") < 0


def test_e3_rho_zero_conventions(e3):
    rows, _ = e3
    for r in rows:
        if r["rho"] == 0.0 and r["policy"] != "truthful":
            assert r["M_mean"]["mean"] is None and r["frac_pos"]["mean"] is None   # undefined, not 0
            assert r["M_uni"]["mean"] is not None                                    # lone deviator


def test_e3_max_claim_collapses_greedy_welfare(e3):
    rows, _ = e3
    full = by(rows, mechanism="GreedyMechanism", policy="max_claim", rho=1.0)
    assert mean(full, "WR") < 0.8 and mean(full, "PoS") > 0.2
    assert mean(by(rows, mechanism="GreedyMechanism", policy="truthful", rho=1.0), "WR") == pytest.approx(1.0)


def test_e3_paired_file_compares_to_truthful(e3):
    _, d = e3
    paired = read(d / "e3" / "paired.json")
    assert len(paired) == 5 * 2 * 5 and all(p["reference"] == "truthful" for p in paired)
    p = next(p for p in paired if p["mechanism"] == "GreedyMechanism"
             and p["policy"] == "max_claim" and p["rho"] == 1.0)
    assert p["metrics"]["WR"]["mean_diff"] < 0


# ── E3b ───────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e3b(tmp_path_factory):
    d = tmp_path_factory.mktemp("e3b")
    return e3b_rollout.run_e3b(SEEDS[:2], str(d), n=10, T=60, n_bootstrap=100,
                               verbose=False, n_rollouts=30), d


def test_e3b_structure(e3b):
    rows, d = e3b
    assert len(rows) == 4 * 2
    for r in rows:
        for key in e3b_rollout.METRIC_KEYS:
            assert r[key]["mean"] is not None
        assert 0.0 <= mean(r, "frac_max_report") <= 1.0 and 0.0 <= mean(r, "frac_inflated") <= 1.0
    for f in ("summary.json", "config.json"):
        assert_strict_json(d / "e3b" / f)
    cfg = read(d / "e3b" / "config.json")
    assert cfg["H"] == 5 and cfg["n"] == 10


def test_e3b_rollout_exploits_greedy_but_not_vickrey(e3b):
    rows, _ = e3b
    g = by(rows, label="M3", opponents="truthful")
    v = by(rows, label="M5", opponents="truthful")
    assert mean(g, "M_rollout") > 0 and mean(g, "frac_max_report") > 0.8
    assert mean(v, "M_rollout") > -0.5 and mean(v, "M_rollout") > mean(v, "M_cap2")   # ~truthful beats cap_2


def test_e3b_is_deterministic():
    a = e3b_rollout.run_e3b(SEEDS[:1], None, n=10, T=25, n_bootstrap=50, verbose=False, n_rollouts=20)
    b = e3b_rollout.run_e3b(SEEDS[:1], None, n=10, T=25, n_bootstrap=50, verbose=False, n_rollouts=20)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ── E4 ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e4(tmp_path_factory):
    d = tmp_path_factory.mktemp("e4")
    return e4_heterogeneous.run_e4(SEEDS, str(d), **KW), d


def test_e4_service_and_benefit_fairness_diverge(e4):
    rows, d = e4
    assert len(rows) == 10
    for f in ("summary.json", "paired.json", "config.json"):
        assert_strict_json(d / "e4" / f)
    # homogeneous: J_A ~ J_B for everybody
    for m in MECHANISM_NAMES:
        r = by(rows, mechanism=m, dist="uniform")
        assert abs(mean(r, "J_A") - mean(r, "J_B")) < 0.06
    # heterogeneous: welfare-maximisers hand GPUs to the high-value group
    for m in ("GreedyMechanism", "VickreyMechanism"):
        r = by(rows, mechanism=m, dist="mixed")
        assert mean(r, "J_A") < 0.7 and mean(r, "J_B") < 0.7 and mean(r, "WR") == pytest.approx(1.0)
    rr = by(rows, mechanism="RoundRobinMechanism", dist="mixed")
    assert mean(rr, "J_A") > 0.99 and mean(rr, "J_B") < mean(rr, "J_A")      # equal service, unequal benefit
    sc = by(rows, mechanism="ScoreMechanism", dist="mixed")
    assert mean(sc, "J_B") > mean(sc, "J_A")                                 # Score equalises *benefit*


# ── E5 ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e5(tmp_path_factory):
    d = tmp_path_factory.mktemp("e5")
    return e5_persistence.run_e5(SEEDS, str(d), n=20, T=200, n_bootstrap=200, verbose=False), d


def test_e5_grid_modes_and_files(e5):
    rows, d = e5
    # alpha=0 once (proposal) + {0.5, 0.9} for each of the two modes
    assert len(rows) == 5 * (1 + 2 + 2)
    assert all(r["ar1_mode"] == "proposal" for r in rows if r["alpha"] == 0.0)
    for f in ("summary.json", "paired.json", "config.json"):
        assert_strict_json(d / "e5" / f)
    paired = read(d / "e5" / "paired.json")
    assert len(paired) == 5 * 4 and all(p["reference"] == "alpha=0" for p in paired)


def test_e5_persistence_hurts_equal_service_of_greedy(e5):
    rows, _ = e5
    g = lambda a, mode="proposal": by(rows, mechanism="GreedyMechanism", alpha=a, ar1_mode=mode)
    assert mean(g(0.9, "copula"), "J_A") < mean(g(0.0), "J_A")                # copula isolates persistence
    assert mean(g(0.9, "copula"), "SR_delta") > mean(g(0.0), "SR_delta")
    assert mean(g(0.9), "WR") == pytest.approx(1.0)


def test_e5_proposal_variance_shrinkage_inflates_report_blind_welfare(e5):
    rows, _ = e5
    r = lambda a, mode: mean(by(rows, mechanism="RandomMechanism", alpha=a, ar1_mode=mode), "WR")
    assert r(0.9, "proposal") > r(0.0, "proposal") + 0.1       # artefact of the narrower value spread
    assert abs(r(0.9, "copula") - r(0.0, "proposal")) < 0.05    # copula keeps the spread: no such effect


# ── E6 ────────────────────────────────────────────────────────────────────────

def test_e6_measures_time_and_heap(tmp_path):
    rows = e6_scalability.run_e6(load_seeds(n=5), str(tmp_path), T=40, n_bootstrap=100,
                                 verbose=False, n_values=[10, 40])
    assert len(rows) == 2 * 5
    for r in rows:
        assert mean(r, "time_per_round_us") > 0 and mean(r, "peak_memory_kib") > 0
    assert_strict_json(tmp_path / "e6" / "summary.json")
    big = np.mean([mean(r, "peak_memory_kib") for r in rows if r["n"] == 40])
    small = np.mean([mean(r, "peak_memory_kib") for r in rows if r["n"] == 10])
    assert big > small                                        # measured memory grows with n
    raw = read(tmp_path / "e6" / "raw" / "GreedyMechanism_n_10.json")["rows"]
    assert raw[4]["peak_memory_kib"] is None and raw[0]["peak_memory_kib"] > 0   # only the first 3 seeds are profiled


# ── E7 ────────────────────────────────────────────────────────────────────────

def test_e7_lambda_by_c_grid(tmp_path):
    rows = e7_sensitivity.run_e7(SEEDS, str(tmp_path), **KW)
    assert len(rows) == 6 * 3
    assert_strict_json(tmp_path / "e7" / "summary.json")
    labels = ["M3", "M4 lambda=0.5", "M4 lambda=1", "M4 lambda=2", "M4 lambda=5", "M5"]
    assert [s[2] for s in e7_sensitivity.SETTINGS] == labels
    # more aggressive exaggeration costs welfare (PoS grows with c) under every mechanism
    for lab in labels:
        pos = [mean(by(rows, label=lab, c=c), "PoS") for c in (1.25, 1.5, 2.0)]
        assert pos == sorted(pos)
    # the individual gain falls monotonically with the penalty lambda, for every c
    for c in (1.25, 1.5, 2.0):
        gain = [mean(by(rows, label=f"M4 lambda={l}", c=c), "M_uni") for l in (0.5, 1, 2, 5)]
        assert gain == sorted(gain, reverse=True)
    assert all(mean(by(rows, label="M3", c=c), "M_uni") > 0 for c in (1.25, 1.5, 2.0))
    assert all(mean(by(rows, label="M5", c=c), "M_uni") < 0 for c in (1.25, 1.5, 2.0))


# ── E8: timing attack ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e8(tmp_path_factory):
    d = tmp_path_factory.mktemp("e8")
    return e8_timing.run_e8(SEEDS, str(d), **KW), d


def test_e8_structure_and_files(e8):
    rows, d = e8
    assert len(rows) == 6 * 2 * 4                                   # settings x c x variants
    assert {r["variant"] for r in rows} == {"always", "timed", "timed_q25", "timed_q75"}
    for f in ("summary.json", "paired.json", "config.json"):
        assert_strict_json(d / "e8" / f)
    paired = read(d / "e8" / "paired.json")
    assert len(paired) == 6 * 2 * 3 and all(p["reference"] == "always" for p in paired)
    assert len(list((d / "e8" / "raw").glob("*.json"))) == len(rows)


def test_e8_timed_policy_inflates_less_than_always_inflating_under_greedy(e8):
    """Greedy ignores history, so timing only withholds inflation: the gain can only shrink."""
    rows, _ = e8
    for c in (1.25, 2.0):
        always = mean(by(rows, label="M3", c=c, variant="always"), "M_uni")
        for v in ("timed", "timed_q25", "timed_q75"):
            assert 0 < mean(by(rows, label="M3", c=c, variant=v), "M_uni") < always


def test_e8_vickrey_punishes_every_variant(e8):
    rows, _ = e8
    for r in (r for r in rows if r["label"] == "M5"):
        assert mean(r, "M_uni") <= 1e-9


def test_e8_timing_premium_is_a_paired_difference(e8):
    rows, d = e8
    paired = read(d / "e8" / "paired.json")
    p = next(p for p in paired if p["label"] == "M3" and p["c"] == 2.0 and p["variant"] == "timed")
    expected = (mean(by(rows, label="M3", c=2.0, variant="timed"), "M_uni")
                - mean(by(rows, label="M3", c=2.0, variant="always"), "M_uni"))
    assert p["metrics"]["M_uni"]["mean_diff"] == pytest.approx(expected)
    assert p["metrics"]["M_uni"]["mean_diff"] < 0


# ── E3c: rollout horizon ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e3c(tmp_path_factory):
    d = tmp_path_factory.mktemp("e3c")
    rows = e3c_rollout_horizon.run_e3c(SEEDS[:2], str(d), n=10, T=40, n_bootstrap=100,
                                       verbose=False, n_rollouts=20, horizons=[2, 6])
    return rows, d


def test_e3c_structure(e3c):
    rows, d = e3c
    assert len(rows) == 2 * 2
    assert {(r["lambda_"], r["H"]) for r in rows} == {(1.0, 2), (1.0, 6), (2.0, 2), (2.0, 6)}
    for r in rows:
        for key in e3c_rollout_horizon.METRIC_KEYS:
            assert r[key]["mean"] is not None
        assert 0.0 <= mean(r, "frac_max_report") <= 1.0
    for f in ("summary.json", "config.json"):
        assert_strict_json(d / "e3c" / f)
    cfg = read(d / "e3c" / "config.json")
    assert cfg["horizons"] == [2, 6] and cfg["opponents"] == "truthful"


def test_e3c_capped_reference_is_independent_of_the_horizon(e3c):
    rows, _ = e3c
    for lam in (1.0, 2.0):
        a, b = by(rows, lambda_=lam, H=2), by(rows, lambda_=lam, H=6)
        assert mean(a, "M_cap2") == mean(b, "M_cap2") and mean(a, "M_cap1.25") == mean(b, "M_cap1.25")


def test_e3c_is_deterministic():
    kw = dict(n=10, T=25, n_bootstrap=50, verbose=False, n_rollouts=15, horizons=[3])
    a = e3c_rollout_horizon.run_e3c(SEEDS[:1], None, **kw)
    b = e3c_rollout_horizon.run_e3c(SEEDS[:1], None, **kw)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_scripts_run_end_to_end(tmp_path):
    import subprocess
    import sys
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "experiments/e1_scarcity.py", "--seeds", "2", "--n", "12", "--T", "40",
         "--bootstrap", "100", "--results-dir", str(tmp_path)],
        cwd=root, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "e1" / "summary.json").exists()
    assert "E1 complete" in out.stdout
