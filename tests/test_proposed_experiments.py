"""
tests/test_proposed_experiments.py — Owner: Kalp Shah

End-to-end tests of E9-E11 (the evaluation of the proposed mechanisms) on tiny
instances: schema, strict JSON, determinism and the qualitative design goals.
"""

import json
import math
from pathlib import Path

import pytest

from experiments import e9_proposed_truthful, e10_proposed_strategic, e11_proposed_rollout
from experiments.common import load_seeds
from experiments.proposed import BASELINES, DEV_SEEDS, PROPOSED, VARIANTS, slug

SEEDS = load_seeds(n=3)
KW = dict(n=20, T=150, n_bootstrap=100, verbose=False)


def by(rows, **conds):
    out = [r for r in rows if all(r.get(k) == v for k, v in conds.items())]
    assert len(out) == 1, (conds, len(out))
    return out[0]


def mean(row, key):
    return row[key]["mean"]


def assert_strict_json(path):
    json.loads(Path(path).read_text(), parse_constant=lambda c: pytest.fail(f"{c} in {path}"))


# ── specification ─────────────────────────────────────────────────────────────

def test_development_seeds_are_disjoint_from_the_locked_seeds():
    assert set(DEV_SEEDS).isdisjoint(load_seeds())            # tuning never touched the evaluation seeds


def test_specs_cover_the_baselines_and_both_proposals():
    assert [s[1] for s in PROPOSED] == ["KarmaCapMechanism", "RankCapMechanism"]
    assert {s[1] for s in BASELINES} >= {"GreedyMechanism", "ScoreMechanism", "VickreyMechanism"}
    assert VARIANTS[0][2] == {"rank_blend": 0.25}
    assert slug("M4 Score λ=1") == "M4_Score_lam1" and "/" not in slug("a/b")


# ── E9 ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e9(tmp_path_factory):
    d = tmp_path_factory.mktemp("e9")
    return e9_proposed_truthful.run_e9(SEEDS, str(d), **KW), d


def test_e9_structure_and_files(e9):
    rows, d = e9
    specs = len(e9_proposed_truthful.SPECS)
    assert len(rows) == 6 * specs + 2 * len(e9_proposed_truthful.WAIT_CAPS)
    for f in ("summary.json", "paired.json", "config.json"):
        assert_strict_json(d / "e9" / f)
    assert (d / "e9" / "raw" / f"base_{slug('M6 Karma-Cap')}.json").exists()
    paired = json.loads((d / "e9" / "paired.json").read_text())
    assert all(p["reference"] == "M4 Score λ=1" for p in paired)


def test_e9_proposed_mechanisms_bound_waiting_and_stay_fair(e9):
    rows, _ = e9
    bound = 10 + math.ceil(20 / 4) - 1                         # W = Delta = 10, n/k = 5
    for lab in ("M6 Karma-Cap", "M7 Rank-Cap"):
        r = by(rows, condition="base", label=lab)
        assert mean(r, "Q_max") <= bound and mean(r, "J_A") > 0.99
        assert mean(r, "SR_delta") <= 0.01
        assert mean(by(rows, condition="base", label="M3 Greedy"), "Q_max") > mean(r, "Q_max")
    greedy = by(rows, condition="base", label="M3 Greedy")
    rr = by(rows, condition="base", label="M2 Round-Robin")
    for lab in ("M6 Karma-Cap", "M7 Rank-Cap"):
        r = by(rows, condition="base", label=lab)
        assert mean(rr, "WR") < mean(r, "WR") <= mean(greedy, "WR")   # between the two extremes


def test_e9_rank_equalises_benefit_under_heterogeneity_better_than_greedy(e9):
    rows, _ = e9
    assert mean(by(rows, condition="mixed", label="M7 Rank-Cap"), "J_B") > \
        mean(by(rows, condition="mixed", label="M3 Greedy"), "J_B") + 0.3


def test_e9_wait_frontier_trades_welfare_for_waiting(e9):
    rows, _ = e9
    for lab in ("M6 Karma-Cap", "M7 Rank-Cap"):
        fr = {r["wait_cap"]: r for r in rows if r["condition"] == "wait_frontier" and r["label"] == lab}
        assert set(fr) == set(e9_proposed_truthful.WAIT_CAPS)
        assert mean(fr[6], "WR") < mean(fr[30], "WR")          # looser cap -> more welfare
        assert mean(fr[6], "Q_max") < mean(fr[30], "Q_max")    # ... and longer worst-case wait
        for W, r in fr.items():
            assert mean(r, "Q_max") <= W + math.ceil(20 / 4) - 1


# ── E10 ───────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e10(tmp_path_factory):
    d = tmp_path_factory.mktemp("e10")
    return e10_proposed_strategic.run_e10(SEEDS, str(d), **KW), d


def test_e10_structure_and_files(e10):
    rows, d = e10
    nl = len(e10_proposed_strategic.LIES_025)
    n_specs = len(e10_proposed_strategic.SPECS)
    n_sens = len(e10_proposed_strategic.SENSITIVITY)
    assert len(rows) == n_specs * nl + n_sens * nl + n_specs * len(e10_proposed_strategic.LIES_FULL)
    assert {r["rho"] for r in rows} == {0.25, 1.0}
    for f in ("summary.json", "paired.json", "config.json"):
        assert_strict_json(d / "e10" / f)
    assert len(list((d / "e10" / "raw").glob("*.json"))) == len(rows)


def test_e10_greedy_rewarded_vickrey_and_proposed_punished_for_blanket_inflation(e10):
    """At rho = 0.25 (the primary setting) every proposed mechanism punishes max claim."""
    rows, _ = e10
    assert mean(by(rows, label="M3 Greedy", policy="max_claim", rho=0.25), "M_uni") > 0
    for lab in ("M5 Vickrey", "M6 Karma-Cap", "M7 Rank-Cap"):
        r = by(rows, label=lab, policy="max_claim", rho=0.25)
        assert mean(r, "M_uni") < 0 and mean(r, "frac_pos_uni") == 0.0


def test_e10_full_population_stress_vickrey_and_rank_still_punish_lying(e10):
    """rho = 1: everyone else already lies.  (Karma-Cap is deliberately NOT asserted here:
    the full-scale run shows its deterrence can reverse in this regime; see the README.)"""
    rows, _ = e10
    assert mean(by(rows, label="M3 Greedy", policy="max_claim", rho=1.0), "M_uni") > 0
    for lab in ("M5 Vickrey", "M7 Rank-Cap"):
        for pol in ("cap_2", "max_claim"):
            assert mean(by(rows, label=lab, policy=pol, rho=1.0), "M_uni") < 0


def test_e10_proposed_beats_score_on_mild_exaggeration(e10):
    rows, _ = e10
    score = mean(by(rows, label="M4 Score λ=1", policy="cap_1.25", rho=0.25), "M_uni")
    assert score > 0
    for lab in ("M6 Karma-Cap", "M7 Rank-Cap"):
        assert mean(by(rows, label=lab, policy="cap_1.25", rho=0.25), "M_uni") < score


def test_e10_paired_file_compares_to_score(e10):
    _, d = e10
    paired = json.loads((d / "e10" / "paired.json").read_text())
    assert paired and all(p["reference"] == "M4 Score λ=1" for p in paired)
    assert not any(p["label"] == "M4 Score λ=1" for p in paired)
    p = next(p for p in paired if p["label"] == "M6 Karma-Cap" and p["policy"] == "cap_2" and p["rho"] == 0.25)
    assert p["metrics"]["M_uni"]["mean_diff"] < 0


def test_e10_karma_supply_sensitivity_is_present(e10):
    rows, _ = e10
    for lab in ("M6 Karma-Cap b0=1", "M6 Karma-Cap b0=5"):
        assert by(rows, label=lab, policy="cap_2", rho=0.25)["M_uni"]["mean"] is not None


# ── E11 ───────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e11(tmp_path_factory):
    d = tmp_path_factory.mktemp("e11")
    rows = e11_proposed_rollout.run_e11(SEEDS[:2], str(d), n=10, T=40, n_bootstrap=100, verbose=False,
                                        n_rollouts=20, horizons=[2, 6])
    return rows, d


def test_e11_structure(e11):
    rows, d = e11
    assert len(rows) == 2 * 2
    assert {(r["label"], r["H"]) for r in rows} == {(lab, h) for lab in ("M6 Karma-Cap", "M7 Rank-Cap")
                                                    for h in (2, 6)}
    for r in rows:
        for key in e11_proposed_rollout.METRIC_KEYS:
            assert r[key]["mean"] is not None
        assert 0.0 <= mean(r, "frac_max_report") <= 1.0
    for f in ("summary.json", "config.json"):
        assert_strict_json(d / "e11" / f)
    assert json.loads((d / "e11" / "config.json").read_text())["score_reference"] == "results/e3c"


def test_e11_is_deterministic():
    kw = dict(n=10, T=25, n_bootstrap=50, verbose=False, n_rollouts=15, horizons=[3])
    a = e11_proposed_rollout.run_e11(SEEDS[:1], None, **kw)
    b = e11_proposed_rollout.run_e11(SEEDS[:1], None, **kw)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
