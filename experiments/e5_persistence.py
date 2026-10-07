"""
experiments/e5_persistence.py
==============================
E5 — Temporally persistent demand (stretch) — Owner: Kalp Shah

Replace i.i.d. values by the AR(1) process of proposal section 4.3,

    v_{i,t} = alpha v_{i,t-1} + (1 - alpha) eps_{i,t},   eps ~ Uniform(0, 1),

for alpha in {0, 0.5, 0.9} (truthful reports, n = 50, k = 10).

Sensitivity check: that recursion also shrinks the marginal standard
deviation by sqrt((1 - alpha)/(1 + alpha)), so "more persistent" also means
"less spread out".  A second series (``ar1_mode = "copula"``) keeps every
v_{i,t} exactly Uniform(0,1) and changes only the persistence, which
separates the two effects.  alpha = 0 is shared by both series.

Outputs: results/e5/{summary,paired,config}.json and results/e5/raw/*.json
Usage:   python experiments/e5_persistence.py [--seeds N] [--results-dir DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bootstrap import paired_differences
from experiments.common import (
    TRUTHFUL_KEYS, build_parser, load_seeds, overrides_from_args, run_cell,
    save_outputs, save_raw, truthful_row,
)
from sim.config import Config
from sim.mechanisms import MECHANISM_NAMES

ALPHA_VALUES = [0.0, 0.5, 0.9]
AR1_MODES = ["proposal", "copula"]


def run_e5(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True, k=None) -> list:
    k = k if k is not None else max(1, round(0.2 * n))
    summaries, paired = [], []
    iid_rows = {}                           # mechanism -> alpha=0 rows ('proposal' pass runs first)
    for mode in AR1_MODES:
        per_alpha = {}                      # (mechanism, alpha) -> rows
        for alpha in ALPHA_VALUES:
            if alpha == 0.0 and mode != "proposal":
                continue                    # i.i.d. is identical in both modes
            cfg = Config(n=n, k=k, T=T, rho=0.0, alpha=alpha, ar1_mode=mode)
            for mname in MECHANISM_NAMES:
                rows, summary = run_cell(lambda pkg, m=mname, c=cfg: truthful_row(m, c, pkg),
                                         seeds, cfg, TRUTHFUL_KEYS, n_bootstrap)
                summary.update({"mechanism": mname, "alpha": alpha, "ar1_mode": mode,
                                "n_seeds": len(seeds)})
                summaries.append(summary)
                per_alpha[(mname, alpha)] = rows
                if alpha == 0.0:
                    iid_rows[mname] = rows
                if results_dir:
                    save_raw(results_dir, "e5", f"{mname}_{mode}_alpha_{alpha:g}", rows, cfg.to_dict())
                if verbose:
                    print(f"  E5 {mode:<8s} alpha={alpha:.1f} {mname:<20s} "
                          f"WR={summary['WR']['mean']:.3f} J_A={summary['J_A']['mean']:.3f}", flush=True)
        # paired vs the i.i.d. cell of the same mechanism (same master seeds)
        for (mname, alpha), rows in per_alpha.items():
            if alpha == 0.0:
                continue
            d = paired_differences({"x": rows, "ref": iid_rows[mname]}, "ref",
                                   TRUTHFUL_KEYS, n_resamples=n_bootstrap)["x"]
            paired.append({"mechanism": mname, "alpha": alpha, "ar1_mode": mode,
                           "reference": "alpha=0", "metrics": d})
    if results_dir:
        save_outputs(results_dir, "e5", summaries,
                     {"n": n, "k": k, "T": T, "alpha_values": ALPHA_VALUES, "ar1_modes": AR1_MODES,
                      "lambda_": 1.0}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = build_parser(__doc__).parse_args()
    seeds = load_seeds(n=args.seeds)
    print(f"Running E5 with {len(seeds)} seeds across alpha {ALPHA_VALUES}, modes {AR1_MODES}")
    run_e5(seeds, args.results_dir, **overrides_from_args(args))
    print(f"E5 complete. Summary saved to {args.results_dir}/e5/summary.json")
