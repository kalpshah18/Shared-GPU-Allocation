"""
experiments/e3_strategic.py
============================
E3 — Strategic Population and Attack Type — Owner: Kalp Shah

Full factorial design over:
  ρ ∈ {0, 0.1, 0.25, 0.5, 1}    (fraction of strategic users)
  × 5 scalable policies (truthful, capped exaggeration c ∈ {1.25, 1.5, 2},
    maximum claim)

For all five mechanisms, with M4 at the default history penalty λ = 1.0.
(The diagnostic rollout attack in sim/policies/strategic.py is not run here.)

Metrics: PoS, WR, J_A, SR_Δ,
  coalition gain   M_mean / M_max / frac_pos  (strategic set deviates together
                   vs. all truthful; undefined at ρ = 0),
  unilateral gain  M_uni  (one focal user deviates vs. reports truthfully,
                   all other strategic users unchanged; at ρ = 0 the focal
                   user is a lone deviator in a truthful population).

Output: results/e3/summary.json
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

sys.path.insert(0, ".")

from sim.config import Config
from sim.environment import SeedPackage, save_seed_result
from sim.mechanisms import (
    RandomMechanism, RoundRobinMechanism,
    GreedyMechanism, ScoreMechanism, VickreyMechanism,
)
from sim.policies import capped_exaggeration, maximum_claim, truthful
from sim.runner import run_paired, run_unilateral
from sim import metrics as M
from analysis.bootstrap import summarise_seeds

RHO_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0]
POLICIES = {
    "truthful"  : lambda v, h, c: truthful(v, h, c),
    "cap_1.25"  : lambda v, h, c: capped_exaggeration(v, h, c, c=1.25),
    "cap_1.5"   : lambda v, h, c: capped_exaggeration(v, h, c, c=1.5),
    "cap_2"     : lambda v, h, c: capped_exaggeration(v, h, c, c=2.0),
    "max_claim" : lambda v, h, c: maximum_claim(v, h, c),
}
METRIC_KEYS = ["WR", "J_A", "SR_delta", "M_mean", "M_max", "frac_pos", "M_uni", "PoS"]


def load_seeds(path="seeds/master_seeds.json"):
    with open(path) as fh:
        return json.load(fh)["seeds"]


def make_mech(mname: str, cfg: Config, pkg: SeedPackage):
    return {
        "RandomMechanism"    : RandomMechanism(cfg),
        "RoundRobinMechanism": RoundRobinMechanism(cfg, init_seed=int(pkg.tie_seeds[0])),
        "GreedyMechanism"    : GreedyMechanism(cfg),
        "ScoreMechanism"     : ScoreMechanism(cfg),
        "VickreyMechanism"   : VickreyMechanism(cfg),
    }[mname]


MECH_NAMES = [
    "RandomMechanism", "RoundRobinMechanism",
    "GreedyMechanism", "ScoreMechanism", "VickreyMechanism",
]


def _run_condition(args: tuple) -> dict:
    """Run one (mechanism, policy, rho) cell over all seeds (picklable worker)."""
    mname, pname, rho, seeds = args
    pfn = POLICIES[pname]
    cfg = Config(n=50, k=10, T=1000, rho=rho)
    per_seed = []

    for seed in seeds:
        pkg  = SeedPackage.generate(seed, cfg)
        mech = make_mech(mname, cfg, pkg)

        h_truth, h_strat = run_paired(mech, pkg, cfg, pfn)

        row = M.compute_all(h_strat, pkg.valuations, cfg)
        mgain = M.manipulation_gain(h_strat, h_truth, pkg.valuations, pkg.strategic_set)
        row.update(mgain)
        h_dev, h_uni, f = run_unilateral(mech, pkg, cfg, pfn)
        row["M_uni"] = M.unilateral_gain(h_dev, h_uni, pkg.valuations, f)
        row["PoS"]  = M.price_of_strategy(h_truth, h_strat, pkg.valuations, cfg)
        row["seed"] = seed
        per_seed.append(row)

    summary = summarise_seeds(per_seed, METRIC_KEYS)
    summary.update({"mechanism": mname, "policy": pname, "rho": rho})
    return summary


def run_e3(seeds: list[int], workers: int = 4) -> list[dict]:
    jobs = [(m, p, r, seeds) for m, p, r in product(MECH_NAMES, POLICIES, RHO_VALUES)]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for summary in pool.map(_run_condition, jobs):
            results.append(summary)
            print(f"  E3 {summary['mechanism']} ρ={summary['rho']} {summary['policy']} "
                  f"WR={summary['WR']['mean']:.3f} M_uni={summary['M_uni']['mean']:.4f}",
                  flush=True)
    return results


if __name__ == "__main__":
    seeds = load_seeds()
    print(f"Running E3 factorial: {len(MECH_NAMES)} mechs × {len(POLICIES)} policies × {len(RHO_VALUES)} ρ values")
    results = run_e3(seeds)

    out = Path("results/e3")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2, allow_nan=False)
    print("E3 complete. Summary saved to results/e3/summary.json")
