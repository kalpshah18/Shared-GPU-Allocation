"""
experiments/e2_frontier.py
===========================
E2 — Fairness and Strategic-Vulnerability Frontier — Owner: Kalp Shah

Sweep λ ∈ {0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5} for M4 (Score mechanism).
Also runs M1–M3 and M5 at λ=N/A for comparison on the same scatter.

For each (mechanism, λ) setting:
  - Run 30 seeds under capped-exaggeration (c=2) for strategic users (ρ=0.25)
  - Report WR, J_A, and M (manipulation gain)
  - Identify non-dominated settings on the WR vs J_A frontier coloured by M

Output: results/e2/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")

from sim.config import Config
from sim.environment import SeedPackage, save_seed_result
from sim.mechanisms import (
    RandomMechanism, RoundRobinMechanism,
    GreedyMechanism, ScoreMechanism, VickreyMechanism,
)
from sim.policies import capped_exaggeration
from sim.runner import run_single, run_mixed, run_paired
from sim import metrics as M
from analysis.bootstrap import summarise_seeds
from analysis.pareto import filter_e2_results

LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
BASE_CFG_KWARGS = dict(n=50, k=10, T=1000, rho=0.25)
METRIC_KEYS = ["WR", "J_A", "SR_delta", "M_mean"]


def load_seeds(path="seeds/master_seeds.json"):
    with open(path) as fh:
        return json.load(fh)["seeds"]


def run_e2(seeds: list[int]) -> list[dict]:
    results = []
    cap2 = lambda v, h, c: capped_exaggeration(v, h, c, c=2.0)

    # Non-M4 mechanisms (λ is irrelevant; run once)
    static_mechs = {
        "RandomMechanism"    : lambda cfg: RandomMechanism(cfg),
        "RoundRobinMechanism": lambda cfg: RoundRobinMechanism(cfg, init_seed=0),
        "GreedyMechanism"    : lambda cfg: GreedyMechanism(cfg),
        "VickreyMechanism"   : lambda cfg: VickreyMechanism(cfg),
    }

    for mname, mfactory in static_mechs.items():
        cfg = Config(**BASE_CFG_KWARGS)
        per_seed = []
        for seed in seeds:
            pkg = SeedPackage.generate(seed, cfg)
            mech = mfactory(cfg)
            h_truth, h_strat = run_paired(mech, pkg, cfg, cap2)
            row = M.compute_all(h_strat, pkg.valuations, cfg)
            mgain = M.manipulation_gain(h_strat, h_truth, pkg.valuations, pkg.strategic_set)
            row.update(mgain)
            row["seed"] = seed
            per_seed.append(row)

        summary = summarise_seeds(per_seed, METRIC_KEYS)
        summary.update({"mechanism": mname, "lambda_": None})
        results.append(summary)
        print(f"  E2 {mname} WR={summary['WR']['mean']:.3f} J_A={summary['J_A']['mean']:.3f}")

    # M4 sweep over λ
    for lam in LAMBDA_VALUES:
        cfg = Config(**BASE_CFG_KWARGS, lambda_=lam)
        per_seed = []
        for seed in seeds:
            pkg  = SeedPackage.generate(seed, cfg)
            mech = ScoreMechanism(cfg)
            h_truth, h_strat = run_paired(mech, pkg, cfg, cap2)
            row = M.compute_all(h_strat, pkg.valuations, cfg)
            mgain = M.manipulation_gain(h_strat, h_truth, pkg.valuations, pkg.strategic_set)
            row.update(mgain)
            row["seed"] = seed
            per_seed.append(row)

        summary = summarise_seeds(per_seed, METRIC_KEYS)
        summary.update({"mechanism": "ScoreMechanism", "lambda_": lam})
        results.append(summary)
        print(f"  E2 M4 λ={lam} WR={summary['WR']['mean']:.3f} J_A={summary['J_A']['mean']:.3f}")

    # Identify Pareto front
    pareto = filter_e2_results([
        {k: r[k]["mean"] for k in METRIC_KEYS} | {"mechanism": r["mechanism"], "lambda_": r["lambda_"]}
        for r in results
    ])
    print(f"\n  E2: {len(pareto)} non-dominated settings on the frontier.")
    return results


if __name__ == "__main__":
    seeds = load_seeds()
    print(f"Running E2 with {len(seeds)} seeds, λ sweep {LAMBDA_VALUES}")
    results = run_e2(seeds)

    out = Path("results/e2")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("E2 complete. Summary saved to results/e2/summary.json")
