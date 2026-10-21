"""
experiments/e6_scalability.py
==============================
E6 — Computational scalability (stretch) — Owner: Kalp Shah

Vary n in {10, 25, 50, 100, 250, 500} with k/n = 0.2 and T = 1000 (truthful
reports) and report, for M1-M5:

  * wall-clock time per simulated round (microseconds), measured around the
    round loop only (valuation generation excluded), best of 3 after a warm-up;
  * peak Python-heap memory (KiB) *measured* with ``tracemalloc`` over one
    full simulation, including the pre-generated seed package and the
    per-round allocation / payment records.  numpy buffers are tracked by
    tracemalloc; interpreter and library overhead outside the heap is not.

Timing and memory use separate passes because tracemalloc slows execution.
The rollout attack is excluded (it solves a separate, costlier problem).
Absolute numbers are machine-dependent.

Outputs: results/e6/{summary,config}.json and results/e6/raw/*.json
Usage:   python experiments/e6_scalability.py [--seeds N] [--results-dir DIR]
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bootstrap import summarise_seeds
from experiments.common import build_parser, load_seeds, save_outputs, save_raw
from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import MECHANISM_NAMES, make_mechanism
from sim.runner import run_mixed

N_VALUES = [10, 25, 50, 100, 250, 500]
MEMORY_SEEDS = 3                      # memory is near-deterministic; few seeds suffice
METRIC_KEYS = ["time_per_round_us", "peak_memory_kib"]


TIMING_REPEATS = 3


def measure_time_us(mname: str, cfg: Config, seed: int) -> float:
    """
    Microseconds per round of the simulation loop for one seed: one warm-up
    run, then the minimum over ``TIMING_REPEATS`` timed runs (the standard
    way to suppress scheduler and frequency noise on a shared machine).
    """
    pkg = SeedPackage.generate(seed, cfg)
    run_mixed(make_mechanism(mname, cfg, pkg), pkg, cfg, None)               # warm-up
    best = float("inf")
    for _ in range(TIMING_REPEATS):
        mech = make_mechanism(mname, cfg, pkg)
        t0 = time.perf_counter()
        run_mixed(mech, pkg, cfg, None)
        best = min(best, time.perf_counter() - t0)
    return best / cfg.T * 1e6


def measure_peak_kib(mname: str, cfg: Config, seed: int) -> float:
    """Peak traced heap (KiB) during seed-package generation plus one run."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        pkg = SeedPackage.generate(seed, cfg)
        mech = make_mechanism(mname, cfg, pkg)
        history = run_mixed(mech, pkg, cfg, None)
        _, peak = tracemalloc.get_traced_memory()
        del history
    finally:
        tracemalloc.stop()
    return peak / 1024.0


def run_e6(seeds, results_dir=None, T=1000, n_bootstrap=10_000, verbose=True, n_values=None) -> list:
    n_values = n_values or N_VALUES
    summaries = []
    for n in n_values:
        cfg = Config(n=n, k=max(1, round(0.2 * n)), T=T, rho=0.0)
        for mname in MECHANISM_NAMES:
            rows = []
            for j, seed in enumerate(seeds):
                row = {"seed": seed, "time_per_round_us": measure_time_us(mname, cfg, seed)}
                row["peak_memory_kib"] = (measure_peak_kib(mname, cfg, seed)
                                          if j < MEMORY_SEEDS else float("nan"))
                rows.append(row)
            summary = summarise_seeds(rows, METRIC_KEYS, n_resamples=n_bootstrap)
            summary.update({"mechanism": mname, "n": n, "k": cfg.k, "n_seeds": len(seeds)})
            summaries.append(summary)
            if results_dir:
                save_raw(results_dir, "e6", f"{mname}_n_{n}", rows, cfg.to_dict())
            if verbose:
                mem = summary["peak_memory_kib"]["mean"]
                print(f"  E6 n={n:<4d} {mname:<20s} "
                      f"{summary['time_per_round_us']['mean']:7.1f} us/round | "
                      f"peak heap {mem:9.1f} KiB", flush=True)
    if results_dir:
        save_outputs(results_dir, "e6", summaries,
                     {"n_values": n_values, "k_over_n": 0.2, "T": T,
                      "memory_seeds": MEMORY_SEEDS, "memory_tool": "tracemalloc"}, seeds)
    return summaries


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E6 benchmark with {len(seeds)} seeds across n = {N_VALUES}")
    run_e6(seeds, args.results_dir, T=args.T or 1000, n_bootstrap=args.bootstrap)
    print(f"E6 complete. Summary saved to {args.results_dir}/e6/summary.json")
