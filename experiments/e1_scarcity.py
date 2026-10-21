"""
experiments/e1_scarcity.py
===========================
E1 — Resource scarcity — Owner: Kalp Shah

Vary k/n in {0.1, 0.2, 0.4, 0.6, 0.8} and compare WR, J_A, Q_max, the 95th
percentile wait and SR_Delta (plus J_B, NSW, PoF) under truthful reports.
M4 uses the default history penalty lambda = 1.

Outputs: results/e1/{summary,paired,config}.json and results/e1/raw/*.json
Usage:   python experiments/e1_scarcity.py [--seeds N] [--results-dir DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bootstrap import paired_differences
from experiments.common import (
    TRUTHFUL_KEYS, build_parser, load_seeds, overrides_from_args, run_cell,
    save_outputs, save_raw, scaled_k, truthful_row,
)
from sim.config import Config
from sim.mechanisms import MECHANISM_NAMES

KN_RATIOS = [0.1, 0.2, 0.4, 0.6, 0.8]
REFERENCE = "ScoreMechanism"


def run_e1(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True) -> list:
    summaries, paired = [], []
    for kn in KN_RATIOS:
        cfg = Config(n=n, k=scaled_k(n, kn), T=T, rho=0.0)
        per_mech = {}
        for mname in MECHANISM_NAMES:
            rows, summary = run_cell(lambda pkg, m=mname, c=cfg: truthful_row(m, c, pkg),
                                     seeds, cfg, TRUTHFUL_KEYS, n_bootstrap)
            summary.update({"mechanism": mname, "kn_ratio": kn, "k": cfg.k, "n_seeds": len(seeds)})
            summaries.append(summary)
            per_mech[mname] = rows
            if results_dir:
                save_raw(results_dir, "e1", f"{mname}_kn_{kn}", rows, cfg.to_dict())
            if verbose:
                print(f"  E1 k/n={kn:.1f} {mname:<20s} WR={summary['WR']['mean']:.3f} "
                      f"J_A={summary['J_A']['mean']:.3f}", flush=True)
        diffs = paired_differences(per_mech, REFERENCE, TRUTHFUL_KEYS, n_resamples=n_bootstrap)
        paired += [{"kn_ratio": kn, "mechanism": m, "reference": REFERENCE, "metrics": d}
                   for m, d in diffs.items()]
    if results_dir:
        save_outputs(results_dir, "e1", summaries,
                     {"n": n, "T": T, "kn_ratios": KN_RATIOS, "lambda_": 1.0,
                      "paired_reference": REFERENCE}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E1 with {len(seeds)} seeds across k/n = {KN_RATIOS}")
    run_e1(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E1 complete. Summary saved to {args.results_dir}/e1/summary.json")
