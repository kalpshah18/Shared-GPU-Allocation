"""
experiments/e20_credit_budget.py
================================
E20 — Vickrey with a credit budget instead of money (M5c Credit-Vickrey)

M5's per-round truthfulness relies on quasi-linear *money* payments, which a
shared academic cluster rarely charges.  M5c replaces money by a renewable
credit quota (see sim/mechanisms/m5c_credit_vickrey.py): credits are spent on
the Vickrey price but carry no utility, so the per-round DSIC argument no
longer applies.  This experiment asks how the budget changes efficiency,
fairness and the incentive to misreport.

Grid (n = 50, k = 10, T = 1000, rho = 0.25, 30 master seeds):
  * M5c with credit_rate in {0.25, 0.5, 1, 2}
    (allowance relative to a fair k/n share of GPU-rounds at price v_max)
  * references: M5 Vickrey (money) and M3 Greedy (no payments; M5c with an
    unlimited budget allocates exactly as M3)
  * reporting policies: truthful, capped exaggeration c in {1.25, 2},
    maximum claim, and the value-threshold rule sandbag_0.75 of E19
    (report 0 below 0.75 v_max, v_max above).

Outputs: results/e20/{summary,config}.json and results/e20/raw/*.json
Usage:   python experiments/e20_credit_budget.py [--seeds N] [--results-dir DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (
    STRATEGIC_KEYS, build_parser, load_seeds, overrides_from_args, run_cell,
    save_outputs, save_raw, strategic_row,
)
from sim.config import Config
from sim.policies import make_capped, make_value_threshold, maximum_claim

RATES = [0.25, 0.5, 1.0, 2.0]
SETTINGS = ([("CreditVickreyMechanism", {"credit_rate": r}, f"M5c rate={r:g}") for r in RATES]
            + [("VickreyMechanism", {}, "M5 (money)"), ("GreedyMechanism", {}, "M3")])
POLICIES = {
    "truthful": None,
    "cap_1.25": make_capped(1.25),
    "cap_2": make_capped(2.0),
    "max_claim": maximum_claim,
    "sandbag_0.75": make_value_threshold(0.75, low="zero"),
}


def run_e20(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True, k=None,
            settings=None) -> list:
    k = k if k is not None else max(1, round(0.2 * n))
    summaries = []
    for (mname, changes, label) in (SETTINGS if settings is None else settings):
        cfg = Config(n=n, k=k, T=T, rho=0.25, n_seeds=len(seeds),
                     n_bootstrap=n_bootstrap, **changes)
        for pname, policy in POLICIES.items():
            rows, summary = run_cell(
                lambda pkg, m=mname, cf=cfg, p=policy: strategic_row(m, cf, pkg, p),
                seeds, cfg, STRATEGIC_KEYS, n_bootstrap)
            summary.update({"mechanism": mname, "label": label, "credit_rate": changes.get("credit_rate"),
                            "policy": pname, "n_seeds": len(seeds)})
            summaries.append(summary)
            if results_dir:
                save_raw(results_dir, "e20", f"{label.replace(' ', '_')}_{pname}", rows, cfg.to_dict())
            if verbose:
                print(f"  E20 {label:<14s} {pname:<12s} WR={summary['WR']['mean']:.3f} "
                      f"J_A={summary['J_A']['mean']:.3f} Q_max={summary['Q_max']['mean']:.1f} "
                      f"M_uni={summary['M_uni']['mean']:+8.2f}", flush=True)
    if results_dir:
        save_outputs(results_dir, "e20", summaries,
                     {"n": n, "k": k, "T": T, "rho": 0.25, "credit_rates": RATES,
                      "credit_horizon": Config().credit_horizon, "policies": list(POLICIES),
                      "n_bootstrap": n_bootstrap,
                      "seed_role": "existing master seeds; exploratory feedback extension",
                      "settings": [{"mechanism": m, "changes": c, "label": l}
                                   for m, c, l in (SETTINGS if settings is None else settings)]}, seeds)
    return summaries


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E20 credit-budget study with {len(seeds)} seeds")
    run_e20(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E20 complete. Summary saved to {args.results_dir}/e20/summary.json")
