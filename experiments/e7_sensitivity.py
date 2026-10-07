"""
experiments/e7_sensitivity.py
==============================
E7 — Sensitivity to the exaggeration factor c — Owner: Kalp Shah

The proposal lists capped exaggeration with c in {1.25, 1.5, 2}, but E2/E3
use c = 2 only.  This sensitivity check repeats the base strategic setting
(n = 50, k = 10, rho = 0.25) for every c and for the mechanisms whose
manipulation incentive is non-trivial: M3, M4 (lambda = 1 and 2) and M5.
(M1 and M2 are report-invariant, hence insensitive to c by construction.)

Outputs: results/e7/{summary,config}.json and results/e7/raw/*.json
Usage:   python experiments/e7_sensitivity.py [--seeds N] [--results-dir DIR]
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
from sim.policies import make_capped

C_VALUES = [1.25, 1.5, 2.0]
SETTINGS = [                      # (mechanism, lambda or None, label)
    ("GreedyMechanism",  None, "M3"),
    ("ScoreMechanism",   1.0,  "M4 lambda=1"),
    ("ScoreMechanism",   2.0,  "M4 lambda=2"),
    ("VickreyMechanism", None, "M5"),
]


def run_e7(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True, k=None) -> list:
    k = k if k is not None else max(1, round(0.2 * n))
    summaries = []
    for (mname, lam, label) in SETTINGS:
        for c in C_VALUES:
            cfg = Config(n=n, k=k, T=T, rho=0.25, lambda_=1.0 if lam is None else lam, c=c)
            policy = make_capped(c)
            rows, summary = run_cell(lambda pkg, m=mname, cf=cfg, p=policy: strategic_row(m, cf, pkg, p),
                                     seeds, cfg, STRATEGIC_KEYS, n_bootstrap)
            summary.update({"mechanism": mname, "lambda_": lam, "label": label, "c": c,
                            "n_seeds": len(seeds)})
            summaries.append(summary)
            if results_dir:
                save_raw(results_dir, "e7", f"{label.replace(' ', '_')}_c_{c:g}", rows, cfg.to_dict())
            if verbose:
                print(f"  E7 {label:<12s} c={c:<4g} WR={summary['WR']['mean']:.3f} "
                      f"M_uni={summary['M_uni']['mean']:+.2f} PoS={summary['PoS']['mean']:.3f}", flush=True)
    if results_dir:
        save_outputs(results_dir, "e7", summaries,
                     {"n": n, "k": k, "T": T, "rho": 0.25, "c_values": C_VALUES,
                      "settings": [s[2] for s in SETTINGS]}, seeds)
    return summaries


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E7 sensitivity with {len(seeds)} seeds over c = {C_VALUES}")
    run_e7(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E7 complete. Summary saved to {args.results_dir}/e7/summary.json")
