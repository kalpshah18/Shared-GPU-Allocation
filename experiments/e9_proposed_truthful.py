"""
experiments/e9_proposed_truthful.py
====================================
E9 — Proposed mechanisms under truthful reports — Owner: Aayush Kuloor

Evaluates M6 Karma-Cap and M7 Rank-Cap (plus the M7 beta=0.25 variant) against
M1-M5 on the conditions of E1, E4 and E5, all with truthful reports:

  base        n=50, k=10, T=1000, i.i.d. Uniform
  scarce      k/n = 0.1   |   loose  k/n = 0.4
  mixed       Beta(2,5) / Beta(5,2) halves (E4)
  persist     AR(1) alpha = 0.9, proposal recursion and marginal-preserving copula (E5)

and a waiting-cap frontier: M6 and M7 at W in {6, 10, 15, 20, 30} on the base
condition, which traces welfare against the hard maximum wait (Q_max <= W).

Outputs: results/e9/{summary,paired,config}.json and results/e9/raw/*.json
Usage:   python experiments/e9_proposed_truthful.py [--seeds N] [--results-dir DIR]
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
from experiments.proposed import BASELINES, PROPOSED, VARIANTS, slug
from sim.config import Config

SPECS = BASELINES + PROPOSED + VARIANTS
REFERENCE = "M4 Score λ=1"
WAIT_CAPS = [6, 10, 15, 20, 30]
CONDITIONS = {                       # name -> (k/n, Config overrides)
    "base"            : (0.2, {}),
    "scarce"          : (0.1, {}),
    "loose"           : (0.4, {}),
    "mixed"           : (0.2, {"valuation_dist": "mixed"}),
    "persist_proposal": (0.2, {"alpha": 0.9, "ar1_mode": "proposal"}),
    "persist_copula"  : (0.2, {"alpha": 0.9, "ar1_mode": "copula"}),
}


def run_e9(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True) -> list:
    summaries, paired = [], []
    for cname, (kn, cover) in CONDITIONS.items():
        per_cell = {}
        for label, mname, mover in SPECS:
            cfg = Config(n=n, k=scaled_k(n, kn), T=T, rho=0.0, **{**cover, **mover})
            rows, summary = run_cell(lambda pkg, m=mname, c=cfg: truthful_row(m, c, pkg),
                                     seeds, cfg, TRUTHFUL_KEYS, n_bootstrap)
            summary.update({"condition": cname, "label": label, "mechanism": mname,
                            "wait_cap": cfg.wait_limit, "n_seeds": len(seeds)})
            summaries.append(summary)
            per_cell[label] = rows
            if results_dir:
                save_raw(results_dir, "e9", f"{cname}_{slug(label)}", rows, cfg.to_dict())
            if verbose:
                print(f"  E9 {cname:<16s} {label:<20s} WR={summary['WR']['mean']:.3f} "
                      f"J_A={summary['J_A']['mean']:.3f} J_B={summary['J_B']['mean']:.3f} "
                      f"Q_max={summary['Q_max']['mean']:.1f} SR={summary['SR_delta']['mean']:.3f}", flush=True)
        diffs = paired_differences(per_cell, REFERENCE, TRUTHFUL_KEYS, n_resamples=n_bootstrap)
        paired += [{"condition": cname, "label": lab, "reference": REFERENCE, "metrics": d}
                   for lab, d in diffs.items()]

    # waiting-cap frontier (base condition)
    for label, mname, mover in PROPOSED:
        for W in WAIT_CAPS:
            cfg = Config(n=n, k=scaled_k(n, 0.2), T=T, rho=0.0, wait_cap=W, **mover)
            rows, summary = run_cell(lambda pkg, m=mname, c=cfg: truthful_row(m, c, pkg),
                                     seeds, cfg, TRUTHFUL_KEYS, n_bootstrap)
            summary.update({"condition": "wait_frontier", "label": label, "mechanism": mname,
                            "wait_cap": W, "n_seeds": len(seeds)})
            summaries.append(summary)
            if results_dir:
                save_raw(results_dir, "e9", f"frontier_{slug(label)}_W{W}", rows, cfg.to_dict())
            if verbose:
                print(f"  E9 frontier {label:<14s} W={W:<3d} WR={summary['WR']['mean']:.3f} "
                      f"Q_max={summary['Q_max']['mean']:.1f}", flush=True)
    if results_dir:
        save_outputs(results_dir, "e9", summaries,
                     {"n": n, "T": T, "conditions": {k: {"k_over_n": v[0], **v[1]} for k, v in CONDITIONS.items()},
                      "specs": [s[0] for s in SPECS], "wait_caps": WAIT_CAPS,
                      "paired_reference": REFERENCE}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E9 (proposed mechanisms, truthful) with {len(seeds)} seeds")
    run_e9(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E9 complete. Summary saved to {args.results_dir}/e9/summary.json")
