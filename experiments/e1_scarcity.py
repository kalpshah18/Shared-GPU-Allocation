"""
experiments/e1_scarcity.py
===========================
E1 — Resource Scarcity — Owner: Kalp Shah

Vary k/n ∈ {0.1, 0.2, 0.4, 0.6, 0.8} and compare WR, J_A, Q_max,
95th-percentile wait, and SR_Δ under truthful reports.
M4 uses the default history penalty λ = cfg.lambda_ (1.0).

Output: results/e1/<seed>.json per seed, plus
        results/e1/summary.json (bootstrap CIs across seeds).
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

KN_RATIOS  = [0.1, 0.2, 0.4, 0.6, 0.8]
BASE_N     = 50
BASE_T     = 1000
METRIC_KEYS = ["WR", "J_A", "Q_max", "SR_delta", "pct95_wait", "PoF"]


def load_seeds(seeds_file: str = "seeds/master_seeds.json") -> list[int]:
    with open(seeds_file) as fh:
        return json.load(fh)["seeds"]


def mechanism_factory(name: str, cfg: Config, pkg: SeedPackage):
    return {
        "RandomMechanism"    : RandomMechanism(cfg),
        "RoundRobinMechanism": RoundRobinMechanism(cfg, init_seed=int(pkg.tie_seeds[0])),
        "GreedyMechanism"    : GreedyMechanism(cfg),
        "ScoreMechanism"     : ScoreMechanism(cfg),
        "VickreyMechanism"   : VickreyMechanism(cfg),
    }[name]


def run_e1(seeds: list[int]) -> list[dict]:
    all_results = []

    for kn in KN_RATIOS:
        k   = max(1, round(BASE_N * kn))
        cfg = Config(n=BASE_N, k=k, T=BASE_T, rho=0.0)  # truthful only

        mech_names = [
            "RandomMechanism", "RoundRobinMechanism",
            "GreedyMechanism", "ScoreMechanism", "VickreyMechanism",
        ]

        for mname in mech_names:
            per_seed = []
            for seed in seeds:
                pkg     = SeedPackage.generate(seed, cfg)
                mech    = mechanism_factory(mname, cfg, pkg)
                history = run_single(mech, pkg, cfg, policy_fn=None)

                result = M.compute_all(history, pkg.valuations, cfg)
                result["seed"]      = seed
                result["kn_ratio"]  = kn
                result["mechanism"] = mname

                save_seed_result(result, f"e1/{mname}/kn_{kn}", seed)
                per_seed.append(result)

            summary = summarise_seeds(per_seed, METRIC_KEYS,
                                       n_resamples=cfg.n_bootstrap)
            summary["mechanism"] = mname
            summary["kn_ratio"]  = kn
            all_results.append(summary)
            print(f"  E1 k/n={kn:.1f} {mname} WR={summary['WR']['mean']:.3f}")

    return all_results


if __name__ == "__main__":
    seeds = load_seeds()
    print(f"Running E1 with {len(seeds)} seeds across k/n ratios {KN_RATIOS}")
    results = run_e1(seeds)

    out = Path("results/e1")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2, allow_nan=False)
    print("E1 complete. Summary saved to results/e1/summary.json")
