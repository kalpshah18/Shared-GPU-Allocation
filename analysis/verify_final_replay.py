"""Reproduce representative saved paper cells using the current simulator.

Uses the first saved seed per selected cell, never claims a full-grid rerun.
The table/CI audit in build_final_report.py separately checks all paper seeds.
"""
import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from analysis.constrained_oracle import constrained_welfare
from experiments.common import strategic_row, truthful_row
from experiments.e19_value_threshold import controlled_timing_row
from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import make_mechanism
from sim.metrics import utilities
from sim.policies import get_policy
from sim.runner import run_mixed, run_rollout


def compare(actual, expected):
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            compare(a, b)
    elif expected is None:
        assert actual is None or np.isnan(actual)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


def load(relative):
    raw = json.loads((ROOT / relative).read_text())
    allowed = {f.name for f in dataclasses.fields(Config) if f.init}
    cfg = Config(**{k: v for k, v in raw["config"].items() if k in allowed})
    row = raw["rows"][0]
    return raw, cfg, row, SeedPackage.generate(row["seed"], cfg)


def main():
    checks = []
    cases = [
        ("e1", "ScoreMechanism_kn_0.2", "ScoreMechanism", None),
        ("e4", "ScoreMechanism_mixed", "ScoreMechanism", None),
        ("e4", "GreedyMechanism_mixed", "GreedyMechanism", None),
        ("e7", "M4_lambda=5_c_1.25", "ScoreMechanism", "cap_1.25"),
        ("e12", "base_cap_1.25_rho0.25_Score_W10", "ScoreCapMechanism", "cap_1.25"),
        ("e12", "base_cap_1.25_rho0.25_Karma_W10", "KarmaCapMechanism", "cap_1.25"),
        ("e12", "base_cap_1.25_rho0.25_Rank_W10", "RankCapMechanism", "cap_1.25"),
        ("e12", "base_cap_1.25_rho0.25_Karma_WNone", "KarmaCapMechanism", "cap_1.25"),
        ("e12", "base_cap_1.25_rho0.25_Rank_WNone", "RankCapMechanism", "cap_1.25"),
        ("e20", "M5c_rate=0.5_sandbag_0.75", "CreditVickreyMechanism", "sandbag_0.75"),
    ]
    for exp, name, mechanism, policy in cases:
        relative = f"results/{exp}/raw/{name}.json"
        _, cfg, saved, pkg = load(relative)
        row = (truthful_row(mechanism, cfg, pkg) if policy is None else
               strategic_row(mechanism, cfg, pkg, get_policy(policy)))
        compare(row, {k: saved[k] for k in row})
        checks.append({"raw_source": relative, "seed": saved["seed"], "verified_metrics": list(row)})
        print("Reproduced", exp, name, flush=True)
    relative = "results/e19/raw/controlled_M4_lambda=5_tau0.75.json"
    _, cfg, saved, pkg = load(relative)
    row = controlled_timing_row("ScoreMechanism", cfg, pkg, .75)
    compare(row, {k: saved[k] for k in row})
    checks.append({"raw_source": relative, "seed": saved["seed"], "verified_metrics": list(row)})
    print("Reproduced controlled timing", flush=True)

    relative = "results/e14/raw/mixed_Score-Cap_W6.json"
    _, cfg, saved, pkg = load(relative)
    result = constrained_welfare(pkg.valuations, cfg.k, 8)
    assert result.optimal
    compare(result.welfare, saved["oracle_welfare"])
    compare(result.upper_bound, saved["oracle_upper_bound"])
    h = run_mixed(make_mechanism("ScoreCapMechanism", cfg, pkg), pkg, cfg, None)
    welfare = float(np.sum(pkg.valuations * np.stack(h.allocations, axis=1)))
    compare(welfare, saved["online_welfare"])
    compare(welfare / result.upper_bound, saved["constrained_efficiency"])
    checks.append({"raw_source": relative, "seed": saved["seed"],
                   "verified_metrics": ["online_welfare", "oracle_welfare", "oracle_upper_bound", "constrained_efficiency"]})
    print("Reproduced constrained oracle", flush=True)

    relative = "results/e15/raw/Score lambda=2_exact_H40_R80.json"
    _, cfg, saved, pkg = load(relative)
    base = utilities(run_mixed(make_mechanism("ScoreMechanism", cfg, pkg), pkg, cfg, None), pkg.valuations)[0]
    history, _ = run_rollout(make_mechanism("ScoreMechanism", cfg, pkg), pkg, cfg, 0, None,
                             np.random.default_rng([pkg.master_seed, 159, 40, 80]), H=40, n_rollouts=80)
    gain = float(utilities(history, pkg.valuations)[0] - base)
    compare(gain, saved["M_rollout"])
    checks.append({"raw_source": relative, "seed": saved["seed"], "verified_metrics": ["M_rollout"]})
    print("Reproduced corrected planning attack", flush=True)
    output = ROOT / "Project/final_assets/replay_checks.json"
    output.write_text(json.dumps({"passed": True, "scope": "first saved seed in each listed cell; not a full-grid rerun",
                                  "cells_replayed": len(checks), "checks": checks}, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {len(checks)} representative code-to-data replay checks")


if __name__ == "__main__":
    main()
