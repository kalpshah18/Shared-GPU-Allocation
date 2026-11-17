"""
experiments/e19_value_threshold.py
==================================
E19 — Value-threshold timing attack on the history-penalised Score rule (H2)

E8's timed policies decide *when* to lie from the user's own allocation
history.  This experiment tests the simpler timing rule suggested in review:
decide from the *current value*.

    sandbag_tau : report 0      if v <  tau * v_max   (skip the round, protect the score)
                  report v_max  if v >= tau * v_max   (claim the round)
    boost_tau   : report v      if v <  tau * v_max   (truthful)
                  report v_max  if v >= tau * v_max

``boost_tau`` isolates the inflation half of the rule, so the paired
difference

    sandbag premium = M_uni(sandbag_tau) - M_uni(boost_tau)

measures the value of *deliberately losing* low-value rounds.  Under M4 a
round won raises a_i and lowers every later score, so H2 predicts the premium
grows with lambda; under Greedy (M3) and per-round-truthful Vickrey (M5)
skipping a round can only lose value.  The credit-budget Vickrey (M5c, rate
0.5) is included because skipping rounds also saves credits there.
Always-max (``max_claim``) is the indiscriminate-inflation reference.

Setting as E8: n = 50, k = 10, T = 1000, rho = 0.25, 30 master seeds.

Outputs: results/e19/{summary,paired,config}.json and results/e19/raw/*.json
Usage:   python experiments/e19_value_threshold.py [--seeds N] [--results-dir DIR]
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
from sim.policies import make_value_threshold, maximum_claim

LAMBDAS = [0.5, 1.0, 2.0, 5.0]
TAUS = [0.5, 0.75, 0.9]
VARIANTS = {"max_claim": maximum_claim}
for _tau in TAUS:
    VARIANTS[f"sandbag_{_tau:g}"] = make_value_threshold(_tau, low="zero")
    VARIANTS[f"boost_{_tau:g}"] = make_value_threshold(_tau, low="truth")
SETTINGS = ([("GreedyMechanism", {}, "M3")]
            + [("ScoreMechanism", {"lambda_": lam}, f"M4 lambda={lam:g}") for lam in LAMBDAS]
            + [("VickreyMechanism", {}, "M5"),
               ("CreditVickreyMechanism", {"credit_rate": 0.5}, "M5c rate=0.5")])
PAIRED_KEYS = ["M_uni", "frac_pos_uni", "WR", "SWL", "J_A"]


def run_e19(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True, k=None,
            settings=None, taus=None) -> tuple[list, list]:
    k = k if k is not None else max(1, round(0.2 * n))
    taus = TAUS if taus is None else taus
    variants = {"max_claim": VARIANTS["max_claim"]}
    for tau in taus:
        variants[f"sandbag_{tau:g}"] = make_value_threshold(tau, low="zero")
        variants[f"boost_{tau:g}"] = make_value_threshold(tau, low="truth")
    summaries, paired = [], []
    for (mname, changes, label) in (SETTINGS if settings is None else settings):
        cfg = Config(n=n, k=k, T=T, rho=0.25, **changes)
        per_variant = {}
        for vname, policy in variants.items():
            rows, summary = run_cell(
                lambda pkg, m=mname, cf=cfg, p=policy: strategic_row(m, cf, pkg, p),
                seeds, cfg, STRATEGIC_KEYS, n_bootstrap)
            tau = None if vname == "max_claim" else float(vname.split("_")[1])
            summary.update({"mechanism": mname, "label": label, "lambda_": changes.get("lambda_"),
                            "variant": vname, "tau": tau, "n_seeds": len(seeds)})
            summaries.append(summary)
            per_variant[vname] = rows
            if results_dir:
                save_raw(results_dir, "e19", f"{label.replace(' ', '_')}_{vname}", rows, cfg.to_dict())
            if verbose:
                print(f"  E19 {label:<14s} {vname:<12s} M_uni={summary['M_uni']['mean']:+8.2f} "
                      f"SWL={summary['SWL']['mean']:.3f}", flush=True)
        for tau in taus:
            diffs = paired_differences({"sandbag": per_variant[f"sandbag_{tau:g}"],
                                        "boost": per_variant[f"boost_{tau:g}"],
                                        "max_claim": per_variant["max_claim"]},
                                       "boost", PAIRED_KEYS, n_resamples=n_bootstrap)
            paired.append({"label": label, "mechanism": mname, "lambda_": changes.get("lambda_"),
                           "tau": tau, "comparison": "sandbag - boost", "metrics": diffs["sandbag"]})
            paired.append({"label": label, "mechanism": mname, "lambda_": changes.get("lambda_"),
                           "tau": tau, "comparison": "max_claim - boost", "metrics": diffs["max_claim"]})
    if results_dir:
        save_outputs(results_dir, "e19", summaries,
                     {"n": n, "k": k, "T": T, "rho": 0.25, "taus": list(taus),
                      "variants": list(variants), "settings": [s[2] for s in (SETTINGS if settings is None else settings)]},
                     seeds, paired)
    return summaries, paired


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E19 value-threshold timing attack with {len(seeds)} seeds")
    run_e19(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E19 complete. Summary saved to {args.results_dir}/e19/summary.json")
