"""
experiments/common.py
=====================
Helpers shared by every experiment script: seed loading, the standard CLI,
raw / summary / config persistence, and the strategic-cell measurement used by
E2, E3 and E7.

Output layout (per experiment ``eX``)
-------------------------------------
    results/eX/summary.json   seed means + 95% bootstrap CIs, one row per cell
    results/eX/paired.json    paired-bootstrap differences between cells
    results/eX/config.json    base configuration, master seeds, grid, git-free
    results/eX/raw/<cell>.json  every per-seed metric row of that cell
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Allow ``python experiments/eX.py`` from the repository root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.bootstrap import summarise_seeds                      # noqa: E402
from sim import metrics as M                                         # noqa: E402
from sim.config import Config                                        # noqa: E402
from sim.environment import SeedPackage, save_json                   # noqa: E402
from sim.mechanisms import make_mechanism                            # noqa: E402
from sim.runner import run_mixed, run_paired, unilateral_gains       # noqa: E402

SEEDS_FILE = ROOT / "seeds" / "master_seeds.json"

# Metrics reported for every strategic cell (E2, E3, E7).
STRATEGIC_KEYS = [
    "WR", "J_A", "J_B", "NSW", "SR_delta", "pct95_wait", "Q_max", "PoF", "PoS",
    "M_mean", "M_max", "frac_pos", "M_uni", "M_uni_max", "frac_pos_uni",
]
# Metrics reported for every truthful cell (E1, E4, E5).
TRUTHFUL_KEYS = ["WR", "J_A", "J_B", "NSW", "SR_delta", "pct95_wait", "Q_max", "PoF"]


# ── Seeds & CLI ───────────────────────────────────────────────────────────────

def load_seeds(path: str | Path = SEEDS_FILE, n: int | None = None) -> list:
    """The locked master seeds (optionally only the first `n`)."""
    with open(path) as fh:
        seeds = json.load(fh)["seeds"]
    return seeds if n is None else seeds[:n]


def build_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--seeds", type=int, default=None,
                    help="use only the first N master seeds (default: all 30)")
    ap.add_argument("--results-dir", default="results", help="output root (default: results)")
    ap.add_argument("--n", type=int, default=None, help="override the population size n")
    ap.add_argument("--T", type=int, default=None, help="override the number of rounds T")
    ap.add_argument("--bootstrap", type=int, default=10_000, help="bootstrap resamples")
    return ap


def overrides_from_args(args: argparse.Namespace) -> dict:
    """Keyword overrides understood by every ``run_eX``."""
    kw = {"n_bootstrap": args.bootstrap}
    if args.n is not None:
        kw["n"] = args.n
    if args.T is not None:
        kw["T"] = args.T
    return kw


def scaled_k(n: int, ratio: float) -> int:
    """GPUs for a scarcity ratio k/n, clamped to 1 <= k < n."""
    return min(max(1, round(n * ratio)), n - 1)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_raw(results_dir, exp: str, cell: str, rows: list, config: dict) -> None:
    """Write every per-seed row of one cell to results/<exp>/raw/<cell>.json."""
    save_json({"cell": cell, "config": config, "rows": rows},
              Path(results_dir) / exp / "raw" / f"{cell}.json")


def save_outputs(results_dir, exp: str, summary: list, config: dict, seeds: list,
                 paired: list | dict | None = None) -> None:
    """Persist summary.json, config.json and (optionally) paired.json."""
    base = Path(results_dir) / exp
    save_json(summary, base / "summary.json")
    save_json({"experiment": exp, "n_seeds": len(seeds), "seeds": seeds, **config},
              base / "config.json")
    if paired is not None:
        save_json(paired, base / "paired.json")


# ── Measurement cells ─────────────────────────────────────────────────────────

def truthful_row(mname: str, cfg: Config, pkg: SeedPackage) -> dict:
    """All single-run metrics for one mechanism under truthful reports."""
    mech = make_mechanism(mname, cfg, pkg)
    history = run_mixed(mech, pkg, cfg, None)
    return M.compute_all(history, pkg.valuations, cfg)


def strategic_row(mname: str, cfg: Config, pkg: SeedPackage, policy) -> dict:
    """
    One seed of a strategic cell: the strategic run's metrics, the coalition
    gain (M_mean, M_max, frac_pos), the unilateral gains of ``cfg.n_focal``
    focal users (M_uni, M_uni_max, frac_pos_uni) and PoS.

    ``policy=None`` is the truthful policy: the strategic run equals the
    truthful run, all manipulation gains are exactly 0 and PoS = 0.
    """
    mech = make_mechanism(mname, cfg, pkg)
    if policy is None:
        h_truth = run_mixed(mech, pkg, cfg, None)
        row = M.compute_all(h_truth, pkg.valuations, cfg)
        n_s = len(pkg.strategic_set)
        row.update({"M_mean": 0.0 if n_s else float("nan"),
                    "M_max": 0.0 if n_s else float("nan"),
                    "frac_pos": 0.0 if n_s else float("nan"),
                    "M_uni": 0.0, "M_uni_max": 0.0, "frac_pos_uni": 0.0, "PoS": 0.0})
        return row

    h_truth, h_strat = run_paired(mech, pkg, cfg, policy)
    row = M.compute_all(h_strat, pkg.valuations, cfg)
    row.update(M.manipulation_gain(h_strat, h_truth, pkg.valuations, pkg.strategic_set))
    _, gains = unilateral_gains(mech, pkg, cfg, policy, h_strat)
    row.update(M.unilateral_summary(gains))
    row["PoS"] = M.price_of_strategy(h_truth, h_strat, pkg.valuations, cfg)
    return row


def run_cell(rows_fn, seeds: list, cfg: Config, keys: list, n_resamples: int):
    """Evaluate ``rows_fn(seed, pkg)`` on every seed; return (rows, summary)."""
    rows = []
    for seed in seeds:
        pkg = SeedPackage.generate(seed, cfg)
        row = rows_fn(pkg)
        row["seed"] = seed
        rows.append(row)
    return rows, summarise_seeds(rows, keys, n_resamples=n_resamples)
