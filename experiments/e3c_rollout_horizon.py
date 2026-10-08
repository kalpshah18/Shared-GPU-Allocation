"""
experiments/e3c_rollout_horizon.py
===================================
E3c — Rollout attack versus planning horizon — Owner: Raj Modi

E3b (proposal setting, H = 5) finds no profitable attack on M4: the search sees
the immediate win from reporting v_max but not the history penalty that persists
after the horizon, so it over-inflates and loses.  This study lengthens the
horizon to test whether that is a horizon artefact.  For M4 (lambda = 1 and 2)
with all other users truthful, the focal user plays the rollout attack with
H in {5, 10, 20, 40, 80}; the gain is compared with capped exaggeration (c = 2 and
1.25) on the same omega.  Everything else matches E3b (n = 10, 100 rollouts per
candidate, grid {0, 0.1, ..., 1}).

Reading the result: if the gain rises with H the short horizon was the problem
and the E3b null is not evidence of robustness; if it stays negative the
heuristic finds no profitable inflation even when it can see further.

Outputs: results/e3c/{summary,config}.json and results/e3c/raw/*.json
Usage:   python experiments/e3c_rollout_horizon.py [--seeds N] [--results-dir DIR]
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
LAMBDAS = [1.0, 2.0]
HORIZONS = [5, 10, 20, 40, 80]
METRIC_KEYS = ["M_rollout", "M_cap2", "M_cap1.25", "mean_inflation", "frac_max_report"]


def horizon_row(cfg: Config, pkg: SeedPackage, horizons: list, n_rollouts: int) -> dict:
    """One seed: focal gain of capped exaggeration and of the rollout attack for each H."""
    mname = "ScoreMechanism"
    focal = np.array([FOCAL])

    def focal_utility(policy):
        h = run_mixed(make_mechanism(mname, cfg, pkg), pkg, cfg, policy, focal)
        return utilities(h, pkg.valuations)[FOCAL]

    base = focal_utility(None)
    row = {"M_cap2": float(focal_utility(make_capped(2.0)) - base),
           "M_cap1.25": float(focal_utility(make_capped(1.25)) - base)}
    for H in horizons:
        rng = np.random.default_rng([pkg.master_seed, 99, H])
        h, reports = run_rollout(make_mechanism(mname, cfg, pkg), pkg, cfg, FOCAL, None, rng,
                                 H=H, n_rollouts=n_rollouts)
        v = pkg.valuations[FOCAL]
        row[f"H{H}"] = {"M_rollout": float(utilities(h, pkg.valuations)[FOCAL] - base),
                        "mean_inflation": float(np.mean(reports - v)),
                        "frac_max_report": float(np.mean(reports >= cfg.v_max - 1e-12))}
    return row


def run_e3c(seeds, results_dir=None, n=10, T=300, n_bootstrap=10_000, verbose=True, k=None,
            n_rollouts=100, horizons=None) -> list:
    horizons = horizons or HORIZONS
    k = k if k is not None else max(1, round(0.2 * n))
    summaries = []
    for lam in LAMBDAS:
        cfg = Config(n=n, k=k, T=T, rho=0.0, lambda_=lam)
        rows = []
        for seed in seeds:
            row = horizon_row(cfg, SeedPackage.generate(seed, cfg), horizons, n_rollouts)
            row["seed"] = seed
            rows.append(row)
        if results_dir:
            save_raw(results_dir, "e3c", f"M4_lambda_{lam:g}", rows, cfg.to_dict())
        for H in horizons:
            flat = [{"M_rollout": r[f"H{H}"]["M_rollout"], "M_cap2": r["M_cap2"],
                     "M_cap1.25": r["M_cap1.25"], "mean_inflation": r[f"H{H}"]["mean_inflation"],
                     "frac_max_report": r[f"H{H}"]["frac_max_report"]} for r in rows]
            summary = summarise_seeds(flat, METRIC_KEYS, n_resamples=n_bootstrap)
            summary.update({"lambda_": lam, "H": H, "label": f"M4 lambda={lam:g}", "n_seeds": len(seeds)})
            summaries.append(summary)
            if verbose:
                print(f"  E3c M4 lambda={lam:g} H={H:<3d} M_rollout={summary['M_rollout']['mean']:+7.2f} "
                      f"[cap2 {summary['M_cap2']['mean']:+.2f}, cap1.25 {summary['M_cap1.25']['mean']:+.2f}] "
                      f"frac_max={summary['frac_max_report']['mean']:.2f}", flush=True)
    if results_dir:
        save_outputs(results_dir, "e3c", summaries,
                     {"n": n, "k": k, "T": T, "n_rollouts": n_rollouts, "horizons": horizons,
                      "lambdas": LAMBDAS, "opponents": "truthful", "focal": FOCAL}, seeds)
    return summaries


if __name__ == "__main__":
    ap = build_parser(__doc__)
    ap.set_defaults(n=10, T=300)
    ap.add_argument("--horizons", type=int, nargs="+", default=None,
                    help=f"planning horizons to evaluate (default: {HORIZONS})")
    args = ap.parse_args()
    seeds = load_seeds(n=args.seeds)
    horizons = args.horizons or HORIZONS
    print(f"Running E3c rollout-horizon study: n={args.n}, T={args.T}, {len(seeds)} seeds, H={horizons}")
    run_e3c(seeds, args.results_dir, n=args.n, T=args.T, n_bootstrap=args.bootstrap, horizons=horizons)
    print(f"E3c complete. Summary saved to {args.results_dir}/e3c/summary.json")
