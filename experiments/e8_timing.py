"""
experiments/e8_timing.py
=========================
E8 — Does timing reports pay? (hypothesis H2) — Owner: Raj Modi

H2 (proposal section 4.4): "Increasing lambda will improve allocation equality
but may increase the value of strategically timing reports because current
service changes future scores."  The scalable policies of E2/E3 inflate in every
round, so they cannot test this.  Here a *timed* policy inflates (capped
exaggeration with factor c) only while the user's own cumulative allocation is
at or below a population threshold, i.e. when the history penalty is lowest, and
reports truthfully otherwise.  Three pre-specified thresholds are compared:

    timed      : a_i(t) <= mean of a(t)
    timed_q25  : a_i(t) <= 25th percentile of a(t)  (only the least served)
    timed_q75  : a_i(t) <= 75th percentile of a(t)  (all but the most served)

For M4 at lambda in {0.5, 1, 2, 5} and c in {1.25, 2} (rho = 0.25), the
*timing premium* is the paired difference
    M_uni(timed) - M_uni(always inflate)
on identical omega.  H2 predicts the premium grows with lambda.  M3 and M5 do
not use history and serve as controls.

Outputs: results/e8/{summary,paired,config}.json and results/e8/raw/*.json
Usage:   python experiments/e8_timing.py [--seeds N] [--results-dir DIR]
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
from sim.config import Config
from sim.policies import make_capped, make_timed

LAMBDAS = [0.5, 1.0, 2.0, 5.0]
C_VALUES = [1.25, 2.0]
VARIANTS = {                       # name -> factory(c)
    "always"   : lambda c: make_capped(c),
    "timed"    : lambda c: make_timed(c),
    "timed_q25": lambda c: make_timed(c, 0.25),
    "timed_q75": lambda c: make_timed(c, 0.75),
}
SETTINGS = ([("GreedyMechanism", None, "M3")]
            + [("ScoreMechanism", lam, f"M4 lambda={lam:g}") for lam in LAMBDAS]
            + [("VickreyMechanism", None, "M5")])
PAIRED_KEYS = ["M_uni", "frac_pos_uni", "WR", "PoS", "J_A"]


def run_e8(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True, k=None) -> list:
    k = k if k is not None else max(1, round(0.2 * n))
    summaries, paired = [], []
    for (mname, lam, label) in SETTINGS:
        for c in C_VALUES:
            cfg = Config(n=n, k=k, T=T, rho=0.25, lambda_=1.0 if lam is None else lam, c=c)
            per_variant = {}
            for vname, factory in VARIANTS.items():
                policy = factory(c)
                rows, summary = run_cell(
                    lambda pkg, m=mname, cf=cfg, p=policy: strategic_row(m, cf, pkg, p),
                    seeds, cfg, STRATEGIC_KEYS, n_bootstrap)
                summary.update({"mechanism": mname, "lambda_": lam, "label": label, "c": c,
                                "variant": vname, "n_seeds": len(seeds)})
                summaries.append(summary)
                per_variant[vname] = rows
                if results_dir:
                    save_raw(results_dir, "e8", f"{label.replace(' ', '_')}_c_{c:g}_{vname}",
                             rows, cfg.to_dict())
                if verbose:
                    print(f"  E8 {label:<12s} c={c:<4g} {vname:<9s} "
                          f"M_uni={summary['M_uni']['mean']:+8.2f} PoS={summary['PoS']['mean']:.3f}",
                          flush=True)
            diffs = paired_differences(per_variant, "always", PAIRED_KEYS, n_resamples=n_bootstrap)
            paired += [{"label": label, "mechanism": mname, "lambda_": lam, "c": c, "variant": v,
                        "reference": "always", "metrics": d} for v, d in diffs.items()]
    if results_dir:
        save_outputs(results_dir, "e8", summaries,
                     {"n": n, "k": k, "T": T, "rho": 0.25, "lambdas": LAMBDAS, "c_values": C_VALUES,
                      "variants": list(VARIANTS), "paired_reference": "always"}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E8 timing study with {len(seeds)} seeds")
    run_e8(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E8 complete. Summary saved to {args.results_dir}/e8/summary.json")
