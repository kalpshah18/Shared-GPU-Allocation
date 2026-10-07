"""
experiments/e2_frontier.py
===========================
E2 — Fairness and Strategic-Vulnerability Frontier — Owner: Kalp Shah

Sweep λ ∈ {0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5} for M4 (Score mechanism).
Also runs M1–M3 and M5 at λ=N/A for comparison on the same scatter.

For each (mechanism, λ) setting:
  - Run 30 seeds under capped-exaggeration (c=2) for strategic users (ρ=0.25)
  - Report WR, J_A, SR_Δ, coalition gain M_mean (all strategic users deviate
    together vs. all truthful) and unilateral gain M_uni (one strategic user
    switches to truthful while the others keep inflating)
  - Identify non-dominated settings on the WR vs J_A frontier coloured by M

Output: results/e2/summary.json         (strategic sweep, ρ = 0.25, cap_2)
        results/e2/truthful_sweep.json  (same λ grid, truthful reports)
        results/e2/pareto.json          (non-dominated settings over WR, J_A, SR_Δ, M_uni)
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
from sim.runner import run_paired, run_single, run_unilateral
from sim import metrics as M
from analysis.bootstrap import summarise_seeds
from analysis.pareto import filter_e2_results

LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
BASE_CFG_KWARGS = dict(n=50, k=10, T=1000, rho=0.25)
METRIC_KEYS = ["WR", "J_A", "SR_delta", "Q_max", "pct95_wait", "M_mean", "M_uni"]
TRUTHFUL_KEYS = ["WR", "J_A", "SR_delta", "Q_max", "pct95_wait", "PoF", "NSW"]


def load_seeds(path="seeds/master_seeds.json"):
    with open(path) as fh:
        return json.load(fh)["seeds"]


def run_e2(seeds: list[int]) -> list[dict]:
    results = []
    cap2 = lambda v, h, c: capped_exaggeration(v, h, c, c=2.0)

    # Non-M4 mechanisms (λ is irrelevant; run once)
    static_mechs = {
        "RandomMechanism"    : lambda cfg, pkg: RandomMechanism(cfg),
        "RoundRobinMechanism": lambda cfg, pkg: RoundRobinMechanism(cfg, init_seed=int(pkg.tie_seeds[0])),
        "GreedyMechanism"    : lambda cfg, pkg: GreedyMechanism(cfg),
        "VickreyMechanism"   : lambda cfg, pkg: VickreyMechanism(cfg),
    }

    for mname, mfactory in static_mechs.items():
        cfg = Config(**BASE_CFG_KWARGS)
        per_seed = []
        for seed in seeds:
            pkg = SeedPackage.generate(seed, cfg)
            mech = mfactory(cfg, pkg)
            h_truth, h_strat = run_paired(mech, pkg, cfg, cap2)
            row = M.compute_all(h_strat, pkg.valuations, cfg)
            mgain = M.manipulation_gain(h_strat, h_truth, pkg.valuations, pkg.strategic_set)
            row.update(mgain)
            h_dev, h_uni, f = run_unilateral(mech, pkg, cfg, cap2)
            row["M_uni"] = M.unilateral_gain(h_dev, h_uni, pkg.valuations, f)
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
            h_dev, h_uni, f = run_unilateral(mech, pkg, cfg, cap2)
            row["M_uni"] = M.unilateral_gain(h_dev, h_uni, pkg.valuations, f)
            row["seed"] = seed
            per_seed.append(row)

        summary = summarise_seeds(per_seed, METRIC_KEYS)
        summary.update({"mechanism": "ScoreMechanism", "lambda_": lam})
        results.append(summary)
        print(f"  E2 M4 λ={lam} WR={summary['WR']['mean']:.3f} J_A={summary['J_A']['mean']:.3f} "
              f"M={summary['M_mean']['mean']:.2f} M_uni={summary['M_uni']['mean']:.2f}")

    return results


def run_e2_truthful(seeds: list[int]) -> list[dict]:
    """λ sweep under truthful reports (ρ = 0): the clean test of H1 (tail waiting
    and starvation vs. welfare loss) with no strategic behaviour."""
    results = []
    static_mechs = {
        "RandomMechanism"    : lambda cfg, pkg: RandomMechanism(cfg),
        "RoundRobinMechanism": lambda cfg, pkg: RoundRobinMechanism(cfg, init_seed=int(pkg.tie_seeds[0])),
        "GreedyMechanism"    : lambda cfg, pkg: GreedyMechanism(cfg),
        "VickreyMechanism"   : lambda cfg, pkg: VickreyMechanism(cfg),
    }
    settings = [(m, None, f, Config(n=50, k=10, T=1000, rho=0.0)) for m, f in static_mechs.items()]
    settings += [("ScoreMechanism", lam, lambda cfg, pkg: ScoreMechanism(cfg),
                  Config(n=50, k=10, T=1000, rho=0.0, lambda_=lam)) for lam in LAMBDA_VALUES]

    for mname, lam, mfactory, cfg in settings:
        per_seed = []
        for seed in seeds:
            pkg = SeedPackage.generate(seed, cfg)
            h = run_single(mfactory(cfg, pkg), pkg, cfg)
            per_seed.append(M.compute_all(h, pkg.valuations, cfg))
        summary = summarise_seeds(per_seed, TRUTHFUL_KEYS)
        summary.update({"mechanism": mname, "lambda_": lam})
        results.append(summary)
        print(f"  E2-truthful {mname} λ={lam} WR={summary['WR']['mean']:.3f} "
              f"Q_max={summary['Q_max']['mean']:.1f} SR={summary['SR_delta']['mean']:.4f}", flush=True)
    return results


def pareto_table(results: list[dict]) -> list[dict]:
    """Non-dominated settings over (WR, J_A, SR_Δ, M_uni) as flat rows."""
    front = filter_e2_results(results, "M_uni")
    rows = []
    for r in front:
        rows.append({
            "mechanism": r["mechanism"], "lambda_": r["lambda_"],
            **{k: r[k]["mean"] for k in ("WR", "J_A", "SR_delta", "M_uni")},
        })
    rows.sort(key=lambda r: -r["WR"])
    return rows


if __name__ == "__main__":
    seeds = load_seeds()
    print(f"Running E2 with {len(seeds)} seeds, λ sweep {LAMBDA_VALUES}")
    results = run_e2(seeds)
    truthful = run_e2_truthful(seeds)
    front = pareto_table(results)
    print(f"\n  E2: {len(front)} non-dominated settings on the (WR, J_A, SR_Δ, M_uni) frontier:")
    for r in front:
        lam = "-" if r["lambda_"] is None else f"{r['lambda_']:g}"
        print(f"    {r['mechanism']:<20s} λ={lam:<5s} WR={r['WR']:.3f} J_A={r['J_A']:.3f} "
              f"SR={r['SR_delta']:.3f} M_uni={r['M_uni']:.1f}")

    out = Path("results/e2")
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in [("summary", results), ("truthful_sweep", truthful), ("pareto", front)]:
        with open(out / f"{name}.json", "w") as fh:
            json.dump(obj, fh, indent=2, allow_nan=False)
    print("E2 complete. Saved summary.json, truthful_sweep.json, pareto.json in results/e2/")
