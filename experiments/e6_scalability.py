"""
experiments/e6_scalability.py
==============================
E6 — Computational Scalability (Stretch Experiment) — Owner: Kalp Shah

Sweep population size n ∈ {10, 25, 50, 100, 250, 500} with fixed scarcity ratio
k/n = 0.2 (k = round(0.2 * n)), T = 1000 rounds.

Measures:
  - Wall-clock runtime per round (microseconds)
  - Peak traced memory (tracemalloc) while generating the seed package and
    simulating T rounds (KiB), measured in a separate run from the timing
for mechanisms M1–M5.

Output: results/e6/summary.json
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
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
from analysis.bootstrap import summarise_seeds

N_VALUES = [10, 25, 50, 100, 250, 500]
BASE_T = 1000
MECH_NAMES = [
    "RandomMechanism", "RoundRobinMechanism",
    "GreedyMechanism", "ScoreMechanism", "VickreyMechanism",
]
SCALABILITY_METRIC_KEYS = ["time_per_round_us", "peak_memory_kib"]


def load_seeds(path: str = "seeds/master_seeds.json", max_seeds: int = 5) -> list[int]:
    with open(path) as fh:
        seeds = json.load(fh)["seeds"]
    return seeds[:max_seeds]  # Use 5 seeds for computational benchmarks


def make_mech(mname: str, cfg: Config, pkg: SeedPackage):
    if mname == "RandomMechanism":
        return RandomMechanism(cfg)
    if mname == "RoundRobinMechanism":
        return RoundRobinMechanism(cfg, init_seed=int(pkg.tie_seeds[0]))
    if mname == "GreedyMechanism":
        return GreedyMechanism(cfg)
    if mname == "ScoreMechanism":
        return ScoreMechanism(cfg)
    if mname == "VickreyMechanism":
        return VickreyMechanism(cfg)
    raise ValueError(f"Unknown mechanism {mname}")


def run_e6(seeds: list[int]) -> list[dict]:
    all_results = []

    for n in N_VALUES:
        k = max(1, round(n * 0.2))
        cfg = Config(n=n, k=k, T=BASE_T, rho=0.0)

        for mname in MECH_NAMES:
            per_seed = []
            for seed in seeds:
                pkg = SeedPackage.generate(seed, cfg)

                # Timing run (tracemalloc off: it slows allocation-heavy code).
                t0 = time.perf_counter()
                run_single(make_mech(mname, cfg, pkg), pkg, cfg, policy_fn=None)
                elapsed_s = time.perf_counter() - t0
                time_per_round_us = (elapsed_s / cfg.T) * 1e6

                # Memory run: peak traced Python/NumPy allocation while generating
                # the seed package and simulating all T rounds (state + history).
                tracemalloc.start()
                pkg_m = SeedPackage.generate(seed, cfg)
                run_single(make_mech(mname, cfg, pkg_m), pkg_m, cfg, policy_fn=None)
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_memory_kib = peak_bytes / 1024.0

                per_seed.append({
                    "time_per_round_us": time_per_round_us,
                    "peak_memory_kib": peak_memory_kib,
                    "seed": seed,
                })

            summary = summarise_seeds(per_seed, SCALABILITY_METRIC_KEYS)
            summary["mechanism"] = mname
            summary["n"] = n
            summary["k"] = k
            all_results.append(summary)
            print(f"  E6 n={n:3d} {mname:<20s} "
                  f"Time/round: {summary['time_per_round_us']['mean']:6.1f} µs | "
                  f"Peak mem: {summary['peak_memory_kib']['mean']:6.1f} KiB", flush=True)

    return all_results


if __name__ == "__main__":
    seeds = load_seeds()
    print(f"Running E6 benchmark with {len(seeds)} seeds across n ∈ {N_VALUES}")
    results = run_e6(seeds)

    out = Path("results/e6")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("E6 complete. Summary saved to results/e6/summary.json")
