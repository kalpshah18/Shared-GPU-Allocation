"""
experiments/e5_persistence.py
==============================
E5 — Temporal Persistence (Stretch Experiment) — Owner: Kalp Shah

Sweep AR(1) persistence coefficient α ∈ {0.0, 0.5, 0.9} under base parameters:
  n=50, k=10, T=1000, rho=0.0 (truthful reports).

Evaluates how temporal autocorrelation in valuations impacts efficiency (WR),
allocation fairness (J_A), starvation (SR_Δ), and tail waiting times.

Output: results/e5/summary.json
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
from sim.environment import SeedPackage, save_seed_result
from sim.mechanisms import (
    RandomMechanism, RoundRobinMechanism,
    GreedyMechanism, ScoreMechanism, VickreyMechanism,
)
from sim.runner import run_single
from sim import metrics as M
from analysis.bootstrap import summarise_seeds

ALPHA_VALUES = [0.0, 0.5, 0.9]
BASE_N = 50
BASE_K = 10
BASE_T = 1000
METRIC_KEYS = ["WR", "J_A", "Q_max", "SR_delta", "pct95_wait", "PoF"]
MECH_NAMES = [
    "RandomMechanism", "RoundRobinMechanism",
    "GreedyMechanism", "ScoreMechanism", "VickreyMechanism",
]


def load_seeds(path: str = "seeds/master_seeds.json") -> list[int]:
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


def run_e5(seeds: list[int]) -> list[dict]:
    all_results = []

    for alpha in ALPHA_VALUES:
        cfg = Config(n=BASE_N, k=BASE_K, T=BASE_T, rho=0.0, alpha=alpha)

        for mname in MECH_NAMES:
            per_seed = []
            for seed in seeds:
                pkg = SeedPackage.generate(seed, cfg)
                mech = make_mech(mname, cfg)
                history = run_single(mech, pkg, cfg, policy_fn=None)

                result = M.compute_all(history, pkg.valuations, cfg)
                result["seed"] = seed
                result["alpha"] = alpha
                result["mechanism"] = mname

                save_seed_result(result, f"e5/{mname}/alpha_{alpha}", seed)
                per_seed.append(result)

            summary = summarise_seeds(per_seed, METRIC_KEYS)
            summary["mechanism"] = mname
            summary["alpha"] = alpha
            all_results.append(summary)
            print(f"  E5 α={alpha:.1f} {mname} WR={summary['WR']['mean']:.3f} J_A={summary['J_A']['mean']:.3f}")

    return all_results


if __name__ == "__main__":
    seeds = load_seeds()
    print(f"Running E5 with {len(seeds)} seeds across α values {ALPHA_VALUES}")
    results = run_e5(seeds)

    out = Path("results/e5")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("E5 complete. Summary saved to results/e5/summary.json")
