"""
experiments/e3_strategic.py
============================
E3 — Strategic Population and Attack Type — Owner: Kalp Shah

Full factorial design over:
  ρ ∈ {0, 0.1, 0.25, 0.5, 1}    (fraction of strategic users)
  × 3 scalable policies (truthful, capped exaggeration c=2, maximum claim)

For all five mechanisms.  Rollout attack applied separately to M3–M5 at
selected ρ settings where it is most likely to separate behaviour.

Metrics: PoS, M_mean, M_max, frac_pos, WR, J_A.

Output: results/e3/summary.json
"""

from __future__ import annotations

import json
import sys
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
from sim.runner import run_single, run_paired
from sim import metrics as M
from analysis.bootstrap import summarise_seeds

RHO_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0]
POLICIES = {
    "truthful"  : lambda v, h, c: truthful(v, h, c),
    "cap_2"     : lambda v, h, c: capped_exaggeration(v, h, c, c=2.0),
    "max_claim" : lambda v, h, c: maximum_claim(v, h, c),
}
METRIC_KEYS = ["WR", "J_A", "SR_delta", "M_mean", "M_max", "frac_pos", "PoS"]


def load_seeds(path="seeds/master_seeds.json"):
    with open(path) as fh:
        return json.load(fh)["seeds"]


def make_mech(mname: str, cfg: Config):
    return {
        "RandomMechanism"    : RandomMechanism(cfg),
        "RoundRobinMechanism": RoundRobinMechanism(cfg, init_seed=0),
        "GreedyMechanism"    : GreedyMechanism(cfg),
        "ScoreMechanism"     : ScoreMechanism(cfg),
        "VickreyMechanism"   : VickreyMechanism(cfg),
    }[mname]


MECH_NAMES = list({
    "RandomMechanism", "RoundRobinMechanism",
    "GreedyMechanism", "ScoreMechanism", "VickreyMechanism",
})


def run_e3(seeds: list[int]) -> list[dict]:
    results = []

    for mname, (pname, pfn), rho in product(MECH_NAMES, POLICIES.items(), RHO_VALUES):
        cfg = Config(n=50, k=10, T=1000, rho=rho, lambda_=0.25)
        per_seed = []

        for seed in seeds:
            pkg    = SeedPackage.generate(seed, cfg)
            mech   = make_mech(mname, cfg)

            h_truth, h_strat = run_paired(mech, pkg, cfg, pfn)

            row = M.compute_all(h_strat, pkg.valuations, cfg)
            mgain = M.manipulation_gain(h_strat, h_truth, pkg.valuations, pkg.strategic_set)
            row.update(mgain)
            row["PoS"]  = M.price_of_strategy(h_truth, h_strat, pkg.valuations, cfg)
            row["seed"] = seed
            per_seed.append(row)

        summary = summarise_seeds(per_seed, METRIC_KEYS)
        summary.update({"mechanism": mname, "policy": pname, "rho": rho})
        results.append(summary)
        print(f"  E3 {mname} ρ={rho} {pname} "
              f"WR={summary['WR']['mean']:.3f} M={summary['M_mean']['mean']:.4f}")

    return results


if __name__ == "__main__":
    seeds = load_seeds()
    print(f"Running E3 factorial: {len(MECH_NAMES)} mechs × {len(POLICIES)} policies × {len(RHO_VALUES)} ρ values")
    results = run_e3(seeds)

    out = Path("results/e3")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("E3 complete. Summary saved to results/e3/summary.json")
