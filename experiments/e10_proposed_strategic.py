"""
experiments/e10_proposed_strategic.py
======================================
E10 — Proposed mechanisms under strategic reporting — Owner: Raj Modi

Individual incentive to manipulate M6 Karma-Cap and M7 Rank-Cap, against M3,
M4 (lambda = 1, 2) and M5, on the grid that exposed the weaknesses of Score:

  lies at rho = 0.25 : capped exaggeration c in {1.25, 1.5, 2}, maximum claim,
                       timed exaggeration (c = 1.25 and c = 2, inflate only
                       while the user's own history is below the population mean)
  stress at rho = 1  : cap_2 and max claim with *everyone* lying

plus a karma-supply sensitivity for M6 (karma_init in {1, 5}).  Metrics are the
unilateral gain M_uni (mean over 5 focal users), its maximum and the fraction
of users who gain, the coalition gain, PoS and the welfare ratio under attack.

Outputs: results/e10/{summary,paired,config}.json and results/e10/raw/*.json
Usage:   python experiments/e10_proposed_strategic.py [--seeds N] [--results-dir DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bootstrap import paired_differences
from experiments.common import (
    STRATEGIC_KEYS, build_parser, load_seeds, overrides_from_args, run_cell,
    save_outputs, save_raw, strategic_row,
)
from experiments.proposed import BASELINES, PROPOSED, VARIANTS, slug
from sim.config import Config
from sim.policies import make_capped, make_timed, maximum_claim

SPECS = ([s for s in BASELINES if s[1] in ("GreedyMechanism", "ScoreMechanism", "VickreyMechanism")]
         + PROPOSED + VARIANTS)
SENSITIVITY = [("M6 Karma-Cap b0=1", "KarmaCapMechanism", {"karma_init": 1.0}),
               ("M6 Karma-Cap b0=5", "KarmaCapMechanism", {"karma_init": 5.0})]
LIES_025 = {                          # name -> policy
    "cap_1.25"    : make_capped(1.25),
    "cap_1.5"     : make_capped(1.5),
    "cap_2"       : make_capped(2.0),
    "max_claim"   : maximum_claim,
    "timed_cap_1.25": make_timed(1.25),
    "timed_cap_2" : make_timed(2.0),
}
LIES_FULL = {"cap_2": make_capped(2.0), "max_claim": maximum_claim}    # rho = 1
REFERENCE = "M4 Score λ=1"
PAIRED_KEYS = ["M_uni", "WR", "PoS", "J_A"]


def run_e10(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True, k=None) -> list:
    k = k if k is not None else max(1, round(0.2 * n))
    jobs = ([(spec, 0.25, pname, pol) for spec in SPECS for pname, pol in LIES_025.items()]
            + [(spec, 0.25, pname, pol) for spec in SENSITIVITY for pname, pol in LIES_025.items()]
            + [(spec, 1.0, pname, pol) for spec in SPECS for pname, pol in LIES_FULL.items()])
    summaries, cells = [], {}
    for (label, mname, mover), rho, pname, pol in jobs:
        cfg = Config(n=n, k=k, T=T, rho=rho, **mover)
        rows, summary = run_cell(lambda pkg, m=mname, c=cfg, p=pol: strategic_row(m, c, pkg, p),
                                 seeds, cfg, STRATEGIC_KEYS, n_bootstrap)
        summary.update({"label": label, "mechanism": mname, "policy": pname, "rho": rho,
                        "n_seeds": len(seeds)})
        summaries.append(summary)
        cells[(label, pname, rho)] = rows
        if results_dir:
            save_raw(results_dir, "e10", f"{slug(label)}_{pname}_rho_{rho:g}", rows, cfg.to_dict())
        if verbose:
            print(f"  E10 {label:<22s} rho={rho:<4g} {pname:<14s} M_uni={summary['M_uni']['mean']:+8.2f} "
                  f"frac>0={summary['frac_pos_uni']['mean']:.2f} WR={summary['WR']['mean']:.3f}", flush=True)

    paired = []
    for (label, pname, rho), rows in cells.items():
        ref = cells.get((REFERENCE, pname, rho))
        if ref is None or label == REFERENCE:
            continue
        d = paired_differences({"x": rows, "ref": ref}, "ref", PAIRED_KEYS, n_resamples=n_bootstrap)["x"]
        paired.append({"label": label, "policy": pname, "rho": rho, "reference": REFERENCE, "metrics": d})
    if results_dir:
        save_outputs(results_dir, "e10", summaries,
                     {"n": n, "k": k, "T": T, "lies_rho_0.25": list(LIES_025),
                      "lies_rho_1": list(LIES_FULL), "specs": [s[0] for s in SPECS + SENSITIVITY],
                      "paired_reference": REFERENCE}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E10 (proposed mechanisms, strategic) with {len(seeds)} seeds")
    run_e10(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E10 complete. Summary saved to {args.results_dir}/e10/summary.json")
