"""
experiments/e4_heterogeneous.py
================================
E4 — Heterogeneous users — Owner: Kalp Shah

Compare the homogeneous base case (Uniform(0,1) for everyone) with an
equal-sized two-group population,

    D_L = Beta(2, 5)  (low-value group,  first n//2 users)
    D_H = Beta(5, 2)  (high-value group, remaining users)

Both stay in [0, 1] but have different means (2/7 vs 5/7).  J_A (equal
service) and J_B (equal normalised benefit) are reported together to expose
cases where they diverge.  All reports are truthful.

Outputs: results/e4/{summary,paired,config}.json and results/e4/raw/*.json
Usage:   python experiments/e4_heterogeneous.py [--seeds N] [--results-dir DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bootstrap import paired_differences
from experiments.common import (
    TRUTHFUL_KEYS, build_parser, load_seeds, overrides_from_args, run_cell,
    save_outputs, save_raw, truthful_row,
)
from sim.config import Config
from sim.mechanisms import MECHANISM_NAMES

DISTS = {"uniform": "uniform", "mixed": "mixed"}
REFERENCE = "ScoreMechanism"


def run_e4(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True, k=None) -> list:
    k = k if k is not None else max(1, round(0.2 * n))
    summaries, paired = [], []
    for dname, dist in DISTS.items():
        cfg = Config(n=n, k=k, T=T, rho=0.0, valuation_dist=dist)
        per_mech = {}
        for mname in MECHANISM_NAMES:
            rows, summary = run_cell(lambda pkg, m=mname, c=cfg: truthful_row(m, c, pkg),
                                     seeds, cfg, TRUTHFUL_KEYS, n_bootstrap)
            summary.update({"mechanism": mname, "dist": dname, "n_seeds": len(seeds)})
            summaries.append(summary)
            per_mech[mname] = rows
            if results_dir:
                save_raw(results_dir, "e4", f"{mname}_{dname}", rows, cfg.to_dict())
            if verbose:
                print(f"  E4 {dname:<8s} {mname:<20s} J_A={summary['J_A']['mean']:.3f} "
                      f"J_B={summary['J_B']['mean']:.3f} WR={summary['WR']['mean']:.3f}", flush=True)
        diffs = paired_differences(per_mech, REFERENCE, TRUTHFUL_KEYS, n_resamples=n_bootstrap)
        paired += [{"dist": dname, "mechanism": m, "reference": REFERENCE, "metrics": d}
                   for m, d in diffs.items()]
    if results_dir:
        save_outputs(results_dir, "e4", summaries,
                     {"n": n, "k": k, "T": T, "dists": list(DISTS), "lambda_": 1.0,
                      "paired_reference": REFERENCE}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E4 heterogeneous users with {len(seeds)} seeds")
    run_e4(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E4 complete. Summary saved to {args.results_dir}/e4/summary.json")
