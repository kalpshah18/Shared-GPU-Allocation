"""
experiments/e3b_rollout.py
===========================
E3b — Finite-horizon rollout attack on M3-M5 — Owner: Raj Modi

Proposal sections 4.2-4.3: for a diagnostic setting with n = 10, one focal
user evaluates every report on the grid G = {0, 0.1, ..., 1} with 100 Monte
Carlo rollouts over the next H = 5 rounds (opponent policies fixed) and
reports the candidate with the highest estimated cumulative utility.

The rollout policy approximates a unilateral attack.  It is neither an
equilibrium computation nor a proof of manipulability: it is a finite search
heuristic, reported separately from the scalable policies.

Selected settings
-----------------
Mechanisms : M3 Greedy, M4 Score (lambda = 1 and 2), M5 Vickrey
Opponents  : all other users truthful, or all other users cap_2

For each (mechanism, opponents) cell and seed the focal user (user 0) is
measured against two baselines on the same omega:
  M_rollout : U_0(rollout attack) - U_0(truthful)
  M_cap2    : U_0(cap_2)          - U_0(truthful)     (the scalable policy)
together with how aggressively the attack inflates.

Outputs: results/e3b/{summary,config}.json and results/e3b/raw/*.json
Usage:   python experiments/e3b_rollout.py [--seeds N] [--results-dir DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from analysis.bootstrap import summarise_seeds
from experiments.common import build_parser, load_seeds, save_outputs, save_raw
from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import make_mechanism
from sim.metrics import utilities
from sim.policies import make_capped
from sim.runner import run_mixed, run_rollout

FOCAL = 0
SETTINGS = [                       # (mechanism, lambda or None, label)
    ("GreedyMechanism",  None, "M3"),
    ("ScoreMechanism",   1.0,  "M4 lambda=1"),
    ("ScoreMechanism",   2.0,  "M4 lambda=2"),
    ("VickreyMechanism", None, "M5"),
]
OPPONENTS = {"truthful": None, "cap_2": make_capped(2.0)}
METRIC_KEYS = ["M_rollout", "M_cap2", "mean_inflation", "frac_inflated", "frac_max_report"]


def rollout_row(mname: str, cfg: Config, pkg: SeedPackage, opp_policy, H: int, n_rollouts: int) -> dict:
    """One seed: focal gains under the rollout attack and under cap_2."""
    cap2 = make_capped(2.0)
    others = np.array([i for i in range(cfg.n) if i != FOCAL], dtype=np.int64)
    everyone = np.arange(cfg.n)

    def baseline():          # focal truthful, others follow the opponent policy
        return run_mixed(make_mechanism(mname, cfg, pkg), pkg, cfg, opp_policy, others)

    U_base = utilities(baseline(), pkg.valuations)[FOCAL]

    # focal plays cap_2; the opponents follow the same profile as in the baseline
    if opp_policy is None:
        h_cap = run_mixed(make_mechanism(mname, cfg, pkg), pkg, cfg, cap2, np.array([FOCAL]))
    else:
        h_cap = run_mixed(make_mechanism(mname, cfg, pkg), pkg, cfg, cap2, everyone)
    U_cap = utilities(h_cap, pkg.valuations)[FOCAL]

    rng = np.random.default_rng([pkg.master_seed, 99])
    h_roll, reports = run_rollout(make_mechanism(mname, cfg, pkg), pkg, cfg, FOCAL,
                                  opp_policy, rng, H=H, n_rollouts=n_rollouts)
    U_roll = utilities(h_roll, pkg.valuations)[FOCAL]

    v = pkg.valuations[FOCAL]
    return {
        "M_rollout"      : float(U_roll - U_base),
        "M_cap2"         : float(U_cap - U_base),
        "mean_inflation" : float(np.mean(reports - v)),
        "frac_inflated"  : float(np.mean(reports > v + 1e-12)),
        "frac_max_report": float(np.mean(reports >= cfg.v_max - 1e-12)),
    }


def run_e3b(seeds, results_dir=None, n=10, T=500, n_bootstrap=10_000, verbose=True,
            k=None, H=5, n_rollouts=100) -> list:
    k = k if k is not None else max(1, round(0.2 * n))
    summaries = []
    for (mname, lam, label) in SETTINGS:
        for oname, opp in OPPONENTS.items():
            cfg = Config(n=n, k=k, T=T, rho=0.0, lambda_=1.0 if lam is None else lam)
            rows = []
            for seed in seeds:
                pkg = SeedPackage.generate(seed, cfg)
                row = rollout_row(mname, cfg, pkg, opp, H, n_rollouts)
                row["seed"] = seed
                rows.append(row)
            summary = summarise_seeds(rows, METRIC_KEYS, n_resamples=n_bootstrap)
            summary.update({"mechanism": mname, "lambda_": lam, "label": label,
                            "opponents": oname, "n_seeds": len(seeds)})
            summaries.append(summary)
            if results_dir:
                save_raw(results_dir, "e3b", f"{label.replace(' ', '_')}_opp_{oname}",
                         rows, cfg.to_dict())
            if verbose:
                print(f"  E3b {label:<12s} opp={oname:<8s} M_rollout={summary['M_rollout']['mean']:+.3f} "
                      f"M_cap2={summary['M_cap2']['mean']:+.3f} "
                      f"frac_max={summary['frac_max_report']['mean']:.2f}", flush=True)
    if results_dir:
        save_outputs(results_dir, "e3b", summaries,
                     {"n": n, "k": k, "T": T, "H": H, "n_rollouts": n_rollouts,
                      "grid": "0,0.1,...,1", "focal": FOCAL,
                      "settings": [s[2] for s in SETTINGS], "opponents": list(OPPONENTS)}, seeds)
    return summaries


if __name__ == "__main__":
    ap = build_parser(__doc__)
    ap.set_defaults(n=10, T=500)
    args = ap.parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E3b rollout attack: n={args.n}, T={args.T}, {len(seeds)} seeds")
    run_e3b(seeds, args.results_dir, n=args.n, T=args.T, n_bootstrap=args.bootstrap)
    print(f"E3b complete. Summary saved to {args.results_dir}/e3b/summary.json")
