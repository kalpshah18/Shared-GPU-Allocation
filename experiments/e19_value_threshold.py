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

The profile-level M_uni cells have different opponent policies and their
difference does not isolate timing. A separate controlled comparison holds
all opponents' policies fixed (boost_tau for S minus the focal user, truthful
elsewhere), changing only the focal policy among truth, boost and sandbag.
Its timing premium is U_f(sandbag) - U_f(boost) on identical workloads.
Under M4 a
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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bootstrap import paired_differences
from experiments.common import (
    STRATEGIC_KEYS, build_parser, load_seeds, overrides_from_args, run_cell,
    save_outputs, save_raw, strategic_row,
)
from sim.config import Config
from sim.environment import History, save_json
from sim.mechanisms import make_mechanism
from sim.runner import focal_users
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
CONTROLLED_KEYS = ["sandbag_gain", "boost_gain", "timing_premium"]


def controlled_timing_row(mname, cfg, pkg, tau):
    """Switch only focal policies, batching independent exact episode replays.

    The rollout interface is just a vectorised mechanism step here: values
    and ties are the actual saved workload, with no sampled/planned futures.
    Each scenario owns separate cumulative counts, waits and credit balances.
    """
    focals = focal_users(pkg, cfg)
    users = np.repeat(focals, 3)
    policies = np.tile(np.arange(3), len(focals))  # truth, boost, sandbag
    scenarios = np.arange(len(users))
    mech = make_mechanism(mname, cfg, pkg)
    state = mech.rollout_init(History(cfg.n), len(users))
    utility = np.zeros(len(users))
    for t in range(cfg.T):
        values = pkg.valuations[:, t]
        reports = np.broadcast_to(values, (len(users), cfg.n)).copy()
        strategic_values = values[pkg.strategic_set]
        reports[:, pkg.strategic_set] = np.where(strategic_values >= tau * cfg.v_max,
                                                cfg.v_max, strategic_values)
        focal_values = values[users]
        reports[scenarios, users] = np.where(
            policies == 0, focal_values,
            np.where(focal_values >= tau * cfg.v_max, cfg.v_max,
                     np.where(policies == 1, focal_values, 0.0)))
        ties = np.random.default_rng(int(pkg.tie_seeds[t])).random(cfg.n)
        x, payments = mech.rollout_step(reports, state, np.broadcast_to(ties, reports.shape))
        utility += focal_values * x[scenarios, users] - payments[scenarios, users]
    records = []
    for focal, outcomes in zip(focals, utility.reshape(-1, 3)):
        truth, boost, sandbag = map(float, outcomes)
        records.append({"focal": int(focal), "truth_utility": truth,
                        "boost_utility": boost, "sandbag_utility": sandbag,
                        "sandbag_gain": sandbag - truth, "boost_gain": boost - truth,
                        "timing_premium": sandbag - boost})
    return {**{key: float(np.mean([r[key] for r in records])) for key in CONTROLLED_KEYS},
            "focal_results": records}


def run_e19(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True, k=None,
            settings=None, taus=None) -> tuple[list, list]:
    k = k if k is not None else max(1, round(0.2 * n))
    taus = TAUS if taus is None else taus
    variants = {"max_claim": VARIANTS["max_claim"]}
    for tau in taus:
        variants[f"sandbag_{tau:g}"] = make_value_threshold(tau, low="zero")
        variants[f"boost_{tau:g}"] = make_value_threshold(tau, low="truth")
    summaries, paired, controlled = [], [], []
    by_lambda = {}
    for (mname, changes, label) in (SETTINGS if settings is None else settings):
        cfg = Config(n=n, k=k, T=T, rho=0.25, n_seeds=len(seeds),
                     n_bootstrap=n_bootstrap, **changes)
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
                           "tau": tau, "comparison": "sandbag - boost",
                           "design": "different_opponent_profiles", "metrics": diffs["sandbag"]})
            paired.append({"label": label, "mechanism": mname, "lambda_": changes.get("lambda_"),
                           "tau": tau, "comparison": "max_claim - boost",
                           "design": "different_opponent_profiles", "metrics": diffs["max_claim"]})
            rows, summary = run_cell(
                lambda pkg, m=mname, cf=cfg, t=tau: controlled_timing_row(m, cf, pkg, t),
                seeds, cfg, CONTROLLED_KEYS, n_bootstrap)
            summary.update({"label": label, "mechanism": mname, "lambda_": changes.get("lambda_"),
                            "tau": tau, "n_seeds": len(seeds), "design": "fixed_opponent_policies"})
            controlled.append(summary)
            if mname == "ScoreMechanism":
                by_lambda[(changes["lambda_"], tau)] = rows
            if results_dir:
                save_raw(results_dir, "e19", f"controlled_{label.replace(' ', '_')}_tau{tau:g}",
                         rows, {**cfg.to_dict(), "tau": tau,
                                "opponents": "boost_tau on S minus focal; truthful elsewhere"})
            if verbose:
                print(f"  E19 controlled {label:<14s} tau={tau:g} "
                      f"premium={summary['timing_premium']['mean']:+.3f}", flush=True)
    # H2 concerns growth with lambda, so estimate paired changes in the premium,
    # rather than inferring a trend from separate point estimates or intervals.
    lambdas = sorted({lam for lam, _ in by_lambda})
    if lambdas:
        for tau in taus:
            groups = {f"{lam:g}": by_lambda[(lam, tau)] for lam in lambdas}
            diffs = paired_differences(groups, f"{lambdas[0]:g}", ["timing_premium"],
                                       n_resamples=n_bootstrap)
            for lam in lambdas[1:]:
                paired.append({"comparison": "lambda_effect", "design": "fixed_opponent_policies",
                               "lambda_": lam, "reference_lambda": lambdas[0], "tau": tau,
                               "metrics": diffs[f"{lam:g}"]})
    if results_dir:
        save_outputs(results_dir, "e19", summaries,
                     {"n": n, "k": k, "T": T, "rho": 0.25, "taus": list(taus),
                      "variants": list(variants), "n_bootstrap": n_bootstrap,
                      "seed_role": "existing master seeds; exploratory feedback extension",
                      "controlled_opponents": "boost_tau on S minus focal; truthful elsewhere",
                      "settings": [{"mechanism": m, "changes": c, "label": l}
                                   for m, c, l in (SETTINGS if settings is None else settings)]},
                     seeds, paired)
        save_json(controlled, Path(results_dir) / "e19" / "controlled.json")
    return summaries, paired


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E19 value-threshold timing attack with {len(seeds)} seeds")
    run_e19(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E19 complete. Summary saved to {args.results_dir}/e19/summary.json")
