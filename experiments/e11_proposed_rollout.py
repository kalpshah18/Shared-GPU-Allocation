"""
experiments/e11_proposed_rollout.py
====================================
E11 — Far-sighted attacker against the proposed mechanisms — Owner: Raj Modi

E3c showed that a rollout attacker with a 40-80-round horizon earns a
significant gain against M4 Score even at lambda = 2, although every simple lie
loses.  This is the test the proposed mechanisms were *not* tuned on: M6
Karma-Cap and M7 Rank-Cap face the same attacker (n = 10, k = 2, T = 300, all
other users truthful, 100 rollouts per candidate report, grid {0, 0.1, ..., 1})
with horizons H in {5, 40, 80}, next to capped exaggeration c = 2 and c = 1.25.

The Score reference at the same settings is E3c (``results/e3c``), which uses
identical seeds, n, k, T and number of rollouts.

How the attacker sees each mechanism: for M6 the rollout simulates the karma
balances, waits and payments exactly; for M7 the users' report histories (their
quantile functions) are frozen at the current round, so within a rollout the
attacker cannot reshape its own quantile function (an approximation that favours
M7; see the limitations in the README).

Outputs: results/e11/{summary,config}.json and results/e11/raw/*.json
Usage:   python experiments/e11_proposed_rollout.py [--seeds N] [--results-dir DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from analysis.bootstrap import summarise_seeds
from experiments.common import build_parser, load_seeds, save_outputs, save_raw
from experiments.proposed import PROPOSED, slug
from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import make_mechanism
from sim.metrics import utilities
from sim.policies import make_capped
from sim.runner import run_mixed, run_rollout

FOCAL = 0
HORIZONS = [5, 40, 80]
METRIC_KEYS = ["M_rollout", "M_cap2", "M_cap1.25", "mean_inflation", "frac_max_report"]


def rollout_row(mname: str, cfg: Config, pkg: SeedPackage, horizons: list, n_rollouts: int) -> dict:
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


def run_e11(seeds, results_dir=None, n=10, T=300, n_bootstrap=10_000, verbose=True, k=None,
            n_rollouts=100, horizons=None) -> list:
    horizons = horizons or HORIZONS
    k = k if k is not None else max(1, round(0.2 * n))
    summaries = []
    for label, mname, mover in PROPOSED:
        cfg = Config(n=n, k=k, T=T, rho=0.0, **mover)
        rows = []
        for seed in seeds:
            row = rollout_row(mname, cfg, SeedPackage.generate(seed, cfg), horizons, n_rollouts)
            row["seed"] = seed
            rows.append(row)
        if results_dir:
            save_raw(results_dir, "e11", slug(label), rows, cfg.to_dict())
        for H in horizons:
            flat = [{"M_rollout": r[f"H{H}"]["M_rollout"], "M_cap2": r["M_cap2"],
                     "M_cap1.25": r["M_cap1.25"], "mean_inflation": r[f"H{H}"]["mean_inflation"],
                     "frac_max_report": r[f"H{H}"]["frac_max_report"]} for r in rows]
            summary = summarise_seeds(flat, METRIC_KEYS, n_resamples=n_bootstrap)
            summary.update({"label": label, "mechanism": mname, "H": H, "n_seeds": len(seeds)})
            summaries.append(summary)
            if verbose:
                print(f"  E11 {label:<14s} H={H:<3d} M_rollout={summary['M_rollout']['mean']:+7.2f} "
                      f"[cap2 {summary['M_cap2']['mean']:+.2f}, cap1.25 {summary['M_cap1.25']['mean']:+.2f}] "
                      f"frac_max={summary['frac_max_report']['mean']:.2f}", flush=True)
    if results_dir:
        save_outputs(results_dir, "e11", summaries,
                     {"n": n, "k": k, "T": T, "n_rollouts": n_rollouts, "horizons": horizons,
                      "opponents": "truthful", "focal": FOCAL, "score_reference": "results/e3c"}, seeds)
    return summaries


if __name__ == "__main__":
    ap = build_parser(__doc__)
    ap.set_defaults(n=10, T=300)
    ap.add_argument("--horizons", type=int, nargs="+", default=None)
    args = ap.parse_args()
    seeds = load_seeds(n=args.seeds)
    horizons = args.horizons or HORIZONS
    print(f"Running E11 rollout vs proposed mechanisms: n={args.n}, T={args.T}, {len(seeds)} seeds, H={horizons}")
    run_e11(seeds, args.results_dir, n=args.n, T=args.T, n_bootstrap=args.bootstrap, horizons=horizons)
    print(f"E11 complete. Summary saved to {args.results_dir}/e11/summary.json")
