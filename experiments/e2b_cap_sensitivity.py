"""
experiments/e2b_cap_sensitivity.py
===================================
E2b — Sensitivity of the λ sweep to the exaggeration factor — Owner: Kalp Shah

E2 sweeps λ under capped exaggeration with c = 2 only.  E3 shows that for M4
milder exaggeration (c = 1.25, 1.5) is *more* profitable than c = 2 (a doubled
report is capped at v_max and wins so often that the history penalty bites
quickly).  This experiment repeats the M4 λ sweep for c ∈ {1.25, 1.5, 2} so the
λ at which inflation stops paying is not an artefact of one attack strength.

Setting: n=50, k=10, T=1000, ρ=0.25, 30 seeds.
Metrics: WR, J_A, coalition gain M_mean, unilateral gain M_uni.

Output: results/e2/cap_sensitivity.json
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")

from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import ScoreMechanism
from sim.policies import capped_exaggeration
from sim.runner import run_paired, run_unilateral
from sim import metrics as M
from analysis.bootstrap import summarise_seeds

LAMBDA_VALUES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
C_VALUES = [1.25, 1.5, 2.0]
METRIC_KEYS = ["WR", "J_A", "M_mean", "M_uni"]


def load_seeds(path="seeds/master_seeds.json") -> list[int]:
    with open(path) as fh:
        return json.load(fh)["seeds"]


def _run_cell(args: tuple) -> dict:
    lam, c, seeds = args
    cfg = Config(n=50, k=10, T=1000, rho=0.25, lambda_=lam)
    policy = lambda v, h, cf: capped_exaggeration(v, h, cf, c=c)
    per_seed = []
    for seed in seeds:
        pkg = SeedPackage.generate(seed, cfg)
        mech = ScoreMechanism(cfg)
        h_truth, h_strat = run_paired(mech, pkg, cfg, policy)
        row = {"WR": M.welfare_ratio(h_strat, pkg.valuations, cfg),
               "J_A": M.jain_allocation(h_strat)}
        row.update(M.manipulation_gain(h_strat, h_truth, pkg.valuations, pkg.strategic_set))
        h_dev, h_uni, f = run_unilateral(mech, pkg, cfg, policy)
        row["M_uni"] = M.unilateral_gain(h_dev, h_uni, pkg.valuations, f)
        per_seed.append(row)
    summary = summarise_seeds(per_seed, METRIC_KEYS)
    summary.update({"mechanism": "ScoreMechanism", "lambda_": lam, "c": c})
    return summary


if __name__ == "__main__":
    seeds = load_seeds()
    jobs = [(lam, c, seeds) for lam, c in product(LAMBDA_VALUES, C_VALUES)]
    print(f"Running E2b: {len(jobs)} (λ, c) cells × {len(seeds)} seeds", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for s in pool.map(_run_cell, jobs):
            results.append(s)
            print(f"  E2b λ={s['lambda_']} c={s['c']} M_uni={s['M_uni']['mean']:.2f} "
                  f"WR={s['WR']['mean']:.3f}", flush=True)
    out = Path("results/e2")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cap_sensitivity.json", "w") as fh:
        json.dump(results, fh, indent=2, allow_nan=False)
    print("E2b complete. Summary saved to results/e2/cap_sensitivity.json")
