"""
experiments/e3_strategic.py
============================
E3 — Strategic population and attack type — Owner: Kalp Shah

Factorial design over rho in {0, 0.1, 0.25, 0.5, 1} and the three scalable
policies (truthful, capped exaggeration c = 2, maximum claim) for all five
mechanisms (M4 at lambda = 1).  Metrics: PoS, WR, J_A, J_B, SR_Delta,
coalition gain (M_mean, M_max, frac_pos; undefined at rho = 0) and unilateral
gain (M_uni, M_uni_max, frac_pos_uni; at rho = 0 each focal user is a lone
deviator in an otherwise truthful population).

The diagnostic rollout attack is run separately for M3-M5 in
experiments/e3b_rollout.py (n = 10).

Outputs: results/e3/{summary,paired,config}.json and results/e3/raw/*.json
Usage:   python experiments/e3_strategic.py [--seeds N] [--results-dir DIR]
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bootstrap import paired_differences
from experiments.common import (
    STRATEGIC_KEYS, build_parser, load_seeds, overrides_from_args, run_cell,
    save_outputs, save_raw, strategic_row,
)
from sim.config import Config
from sim.mechanisms import MECHANISM_NAMES
from sim.policies import make_capped, maximum_claim

RHO_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0]
POLICIES = {                 # name -> policy (None = truthful, handled exactly)
    "truthful" : None,
    "cap_2"    : make_capped(2.0),
    "max_claim": maximum_claim,
}
PAIRED_KEYS = ["WR", "J_A", "SR_delta"]


def run_e3(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True, k=None) -> list:
    k = k if k is not None else max(1, round(0.2 * n))
    summaries, cells = [], {}
    for mname, (pname, policy), rho in product(MECHANISM_NAMES, POLICIES.items(), RHO_VALUES):
        cfg = Config(n=n, k=k, T=T, rho=rho)
        rows, summary = run_cell(lambda pkg, m=mname, c=cfg, p=policy: strategic_row(m, c, pkg, p),
                                 seeds, cfg, STRATEGIC_KEYS, n_bootstrap)
        summary.update({"mechanism": mname, "policy": pname, "rho": rho, "n_seeds": len(seeds)})
        summaries.append(summary)
        cells[(mname, pname, rho)] = rows
        if results_dir:
            save_raw(results_dir, "e3", f"{mname}_{pname}_rho_{rho:g}", rows, cfg.to_dict())
        if verbose:
            mu = summary["M_uni"]["mean"]
            print(f"  E3 {mname:<20s} rho={rho:<4g} {pname:<9s} WR={summary['WR']['mean']:.3f} "
                  f"M_uni={mu:+.3f}", flush=True)

    # Paired (same omega) differences: strategic policy minus the truthful cell.
    paired = []
    for (mname, pname, rho), rows in cells.items():
        if pname == "truthful":
            continue
        base = cells[(mname, "truthful", rho)]
        d = paired_differences({"x": rows, "ref": base}, "ref", PAIRED_KEYS, n_resamples=n_bootstrap)["x"]
        paired.append({"mechanism": mname, "policy": pname, "rho": rho,
                       "reference": "truthful", "metrics": d})
    if results_dir:
        save_outputs(results_dir, "e3", summaries,
                     {"n": n, "k": k, "T": T, "rho_values": RHO_VALUES,
                      "policies": list(POLICIES), "lambda_": 1.0, "c": 2.0}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E3 factorial: {len(MECHANISM_NAMES)} mechanisms x {len(POLICIES)} policies "
          f"x {len(RHO_VALUES)} rho values, {len(seeds)} seeds")
    run_e3(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E3 complete. Summary saved to {args.results_dir}/e3/summary.json")
