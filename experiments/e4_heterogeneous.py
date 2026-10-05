"""
experiments/e4_heterogeneous.py
================================
E4 — Heterogeneous Users — Owner: Kalp Shah

Compare the homogeneous base case (Uniform) with an equal-sized two-group
population:
  D_L = Beta(2,5)   (low-value group,  first n//2 users)
  D_H = Beta(5,2)   (high-value group, remaining users)

Both distributions remain in [0,1] but have different means.
Report J_A and J_B together to identify cases where equal service and
equal normalised benefit diverge.

Output: results/e4/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

sys.path.insert(0, ".")

from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import (
    RandomMechanism, RoundRobinMechanism,
    GreedyMechanism, ScoreMechanism, VickreyMechanism,
)
from sim.runner import run_single
from sim import metrics as M
from analysis.bootstrap import summarise_seeds

METRIC_KEYS = ["WR", "J_A", "J_B", "SR_delta", "PoF"]
MECH_NAMES  = [
    "RandomMechanism", "RoundRobinMechanism",
    "GreedyMechanism", "ScoreMechanism", "VickreyMechanism",
]

# Per-user expected values for mixed distribution (Beta(2,5) mean = 2/7,
# Beta(5,2) mean = 5/7)
def per_user_mu(n: int) -> np.ndarray:
    half = n // 2
    mu = np.empty(n)
    mu[:half]  = 2 / 7   # E[Beta(2,5)]
    mu[half:]  = 5 / 7   # E[Beta(5,2)]
    return mu


def make_mech(mname: str, cfg: Config):
    return {
        "RandomMechanism"    : RandomMechanism(cfg),
        "RoundRobinMechanism": RoundRobinMechanism(cfg, init_seed=0),
        "GreedyMechanism"    : GreedyMechanism(cfg),
        "ScoreMechanism"     : ScoreMechanism(cfg),
        "VickreyMechanism"   : VickreyMechanism(cfg),
    }[mname]


def load_seeds(path="seeds/master_seeds.json"):
    with open(path) as fh:
        return json.load(fh)["seeds"]


def run_e4(seeds: list[int]) -> list[dict]:
    results = []
    dists   = {
        "uniform": dict(valuation_dist="uniform"),
        "mixed"  : dict(valuation_dist="mixed"),
    }

    for dist_name, dist_kwargs in dists.items():
        mu = (np.full(50, 0.5) if dist_name == "uniform"
              else per_user_mu(50))

        for mname in MECH_NAMES:
            cfg = Config(n=50, k=10, T=1000, rho=0.0, **dist_kwargs)
            per_seed = []

            for seed in seeds:
                pkg     = SeedPackage.generate(seed, cfg)
                mech    = make_mech(mname, cfg)
                history = run_single(mech, pkg, cfg)

                row = M.compute_all(history, pkg.valuations, cfg, mu=mu)
                row["seed"] = seed
                per_seed.append(row)

            summary = summarise_seeds(per_seed, METRIC_KEYS)
            summary.update({"mechanism": mname, "dist": dist_name})
            results.append(summary)
            print(f"  E4 {dist_name} {mname} "
                  f"J_A={summary['J_A']['mean']:.3f} J_B={summary['J_B']['mean']:.3f}")

    return results


if __name__ == "__main__":
    seeds = load_seeds()
    print(f"Running E4 heterogeneous users with {len(seeds)} seeds")
    results = run_e4(seeds)

    out = Path("results/e4")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("E4 complete. Summary saved to results/e4/summary.json")
