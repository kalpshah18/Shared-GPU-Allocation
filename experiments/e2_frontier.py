"""
experiments/e2_frontier.py
===========================
E2 — Fairness and strategic-vulnerability frontier — Owner: Kalp Shah

Sweep lambda in {0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5} for M4 and add M1, M2, M3,
M5 as reference points.  Every setting runs with rho = 0.25 users using capped
exaggeration (c = 2) and reports WR, J_A, J_B, SR_Delta, the coalition gain
(M_mean, M_max, frac_pos), the unilateral gains (M_uni, M_uni_max,
frac_pos_uni) and PoS.  All non-dominated settings on (WR max, J_A max,
SR_Delta min, M_uni min) are flagged in the summary (``pareto``).

Outputs: results/e2/{summary,paired,config}.json and results/e2/raw/*.json
Usage:   python experiments/e2_frontier.py [--seeds N] [--results-dir DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bootstrap import paired_differences
from analysis.pareto import pareto_flags
from experiments.common import (
    STRATEGIC_KEYS, build_parser, load_seeds, overrides_from_args, run_cell,
    save_outputs, save_raw, strategic_row,
)
from sim.config import Config
from sim.policies import make_capped

LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
STATIC_MECHS = ["RandomMechanism", "RoundRobinMechanism", "GreedyMechanism", "VickreyMechanism"]
PAIRED_KEYS = ["WR", "J_A", "SR_delta", "M_mean", "M_uni", "PoS"]


def _cell_name(mname, lam):
    return mname if lam is None else f"{mname}_lam_{lam:g}"


def run_e2(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True,
           k=None, c=2.0) -> list:
    k = k if k is not None else max(1, round(0.2 * n))
    policy = make_capped(c)
    settings = [(m, None) for m in STATIC_MECHS] + [("ScoreMechanism", lam) for lam in LAMBDA_VALUES]

    summaries, per_cell = [], {}
    for mname, lam in settings:
        cfg = Config(n=n, k=k, T=T, rho=0.25, lambda_=1.0 if lam is None else lam, c=c)
        rows, summary = run_cell(lambda pkg, m=mname, cf=cfg: strategic_row(m, cf, pkg, policy),
                                 seeds, cfg, STRATEGIC_KEYS, n_bootstrap)
        summary.update({"mechanism": mname, "lambda_": lam, "n_seeds": len(seeds)})
        summaries.append(summary)
        per_cell[_cell_name(mname, lam)] = rows
        if results_dir:
            save_raw(results_dir, "e2", _cell_name(mname, lam), rows, cfg.to_dict())
        if verbose:
            tag = mname if lam is None else f"M4 lambda={lam:g}"
            print(f"  E2 {tag:<24s} WR={summary['WR']['mean']:.3f} J_A={summary['J_A']['mean']:.3f} "
                  f"M={summary['M_mean']['mean']:+.1f} M_uni={summary['M_uni']['mean']:+.1f}", flush=True)

    flags = pareto_flags(summaries, gain_key="M_uni")
    for s, f in zip(summaries, flags):
        s["pareto"] = bool(f)
    if verbose:
        print(f"\n  E2: {sum(flags)} non-dominated settings out of {len(flags)}.")

    reference = _cell_name("ScoreMechanism", 0.0)
    diffs = paired_differences(per_cell, reference, PAIRED_KEYS, n_resamples=n_bootstrap)
    paired = [{"cell": cell, "reference": reference, "metrics": d} for cell, d in diffs.items()]
    if results_dir:
        save_outputs(results_dir, "e2", summaries,
                     {"n": n, "k": k, "T": T, "rho": 0.25, "policy": f"cap_{c:g}",
                      "lambda_values": LAMBDA_VALUES, "paired_reference": reference,
                      "pareto_objectives": "WR max, J_A max, SR_delta min, M_uni min"},
                     seeds, paired)
    return summaries


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E2 with {len(seeds)} seeds, lambda sweep {LAMBDA_VALUES}")
    run_e2(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E2 complete. Summary saved to {args.results_dir}/e2/summary.json")
