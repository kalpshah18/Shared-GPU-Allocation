"""
experiments/e7_paired.py
========================
E7 — Paired mechanism comparisons — Owner: Kalp Shah

The proposal (§4.3) promises *paired* 95% bootstrap intervals.  E1–E5 report
per-mechanism means with unpaired-style CIs; this script reports the
per-seed *differences* A − B between mechanisms.  Every mechanism sees the same
valuation tensor and tie seeds within a seed, so the pairing removes the large
between-seed variance.

Conditions (n=50, k=10, T=1000, 30 seeds):
  base_truthful   Uniform(0,1) values, truthful reports
  mixed_truthful  Beta(2,5) / Beta(5,2) halves, truthful reports   (adds J_B)
  strategic_cap2  Uniform values, ρ=0.25 of users use capped exaggeration c=2
                  (adds the unilateral manipulation gain M_uni)

Mechanisms compared: Random, Round-Robin, Greedy, Vickrey, Score(λ=1), Score(λ=2).

Output: results/e7/paired.json, results/e7/paired.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

sys.path.insert(0, ".")

from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import (
    RandomMechanism, RoundRobinMechanism,
    GreedyMechanism, ScoreMechanism, VickreyMechanism,
)
from sim.policies import capped_exaggeration
from sim.runner import run_mixed, run_single, run_unilateral
from sim import metrics as M
from analysis.bootstrap import paired_bootstrap_ci

MECHS = {
    "Random"      : lambda cfg, pkg: RandomMechanism(cfg),
    "RoundRobin"  : lambda cfg, pkg: RoundRobinMechanism(cfg, init_seed=int(pkg.tie_seeds[0])),
    "Greedy"      : lambda cfg, pkg: GreedyMechanism(cfg),
    "Vickrey"     : lambda cfg, pkg: VickreyMechanism(cfg),
    "Score(l=1)"  : lambda cfg, pkg: ScoreMechanism(cfg),
    "Score(l=2)"  : lambda cfg, pkg: ScoreMechanism(cfg),
}
LAMBDA = {"Score(l=1)": 1.0, "Score(l=2)": 2.0}

PAIRS = [
    ("Score(l=1)", "Greedy"), ("Score(l=2)", "Greedy"), ("Score(l=2)", "Score(l=1)"),
    ("Score(l=1)", "RoundRobin"), ("Score(l=2)", "RoundRobin"),
    ("Score(l=1)", "Random"), ("Greedy", "RoundRobin"),
]
STRATEGIC_PAIRS = PAIRS + [("Score(l=1)", "Vickrey"), ("Score(l=2)", "Vickrey")]

CONDITIONS = {
    "base_truthful" : dict(dist="uniform", rho=0.0,  policy=None,
                           metrics=["WR", "J_A", "SR_delta", "Q_max", "NSW"], pairs=PAIRS),
    "mixed_truthful": dict(dist="mixed",   rho=0.0,  policy=None,
                           metrics=["WR", "J_A", "J_B", "SR_delta", "NSW"], pairs=PAIRS),
    "strategic_cap2": dict(dist="uniform", rho=0.25, policy="cap_2",
                           metrics=["WR", "J_A", "SR_delta", "M_uni"], pairs=STRATEGIC_PAIRS),
}


def per_user_mu(n: int) -> np.ndarray:
    mu = np.empty(n)
    mu[: n // 2] = 2 / 7
    mu[n // 2:] = 5 / 7
    return mu


def load_seeds(path="seeds/master_seeds.json") -> list[int]:
    with open(path) as fh:
        return json.load(fh)["seeds"]


def run_condition(name: str, spec: dict, seeds: list[int]) -> dict[str, dict[str, np.ndarray]]:
    """Return {mechanism: {metric: per-seed array}} for one condition."""
    mu = per_user_mu(50) if spec["dist"] == "mixed" else np.full(50, 0.5)
    cap2 = lambda v, h, c: capped_exaggeration(v, h, c, c=2.0)
    out = {m: {k: [] for k in spec["metrics"]} for m in MECHS}

    for seed in seeds:
        for mname, factory in MECHS.items():
            cfg = Config(n=50, k=10, T=1000, rho=spec["rho"], valuation_dist=spec["dist"],
                         lambda_=LAMBDA.get(mname, 1.0))
            pkg = SeedPackage.generate(seed, cfg)
            mech = factory(cfg, pkg)
            if spec["policy"] is None:
                h = run_single(mech, pkg, cfg)
            else:
                h = run_mixed(mech, pkg, cfg, cap2)
            row = M.compute_all(h, pkg.valuations, cfg, mu=mu)
            if "M_uni" in spec["metrics"]:
                h_dev, h_tru, f = run_unilateral(mech, pkg, cfg, cap2)
                row["M_uni"] = M.unilateral_gain(h_dev, h_tru, pkg.valuations, f)
            for k in spec["metrics"]:
                out[mname][k].append(row[k])
        print(f"  E7 {name} seed {seed} done", flush=True)

    return {m: {k: np.array(v) for k, v in d.items()} for m, d in out.items()}


def run_e7(seeds: list[int]) -> list[dict]:
    rows = []
    for cname, spec in CONDITIONS.items():
        data = run_condition(cname, spec, seeds)
        for a, b in spec["pairs"]:
            for metric in spec["metrics"]:
                ci = paired_bootstrap_ci(data[a][metric], data[b][metric])
                rows.append({"condition": cname, "A": a, "B": b, "metric": metric, **ci})
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = []
    for cname in CONDITIONS:
        sub = [r for r in rows if r["condition"] == cname]
        metrics = CONDITIONS[cname]["metrics"]
        lines += [f"### {cname}", "",
                  "| A − B | " + " | ".join(metrics) + " |",
                  "|---|" + "---|" * len(metrics)]
        for a, b in CONDITIONS[cname]["pairs"]:
            cells = []
            for m in metrics:
                r = next(r for r in sub if r["A"] == a and r["B"] == b and r["metric"] == m)
                sig = "*" if (r["ci_lower"] > 0 or r["ci_upper"] < 0) else ""
                cells.append(f"{r['mean_diff']:+.4g} [{r['ci_lower']:+.3g}, {r['ci_upper']:+.3g}]{sig}")
            lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("Mean paired difference [95% bootstrap CI]; * = CI excludes 0.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    seeds = load_seeds()
    print(f"Running E7 paired comparisons with {len(seeds)} seeds")
    rows = run_e7(seeds)

    out = Path("results/e7")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "paired.json", "w") as fh:
        json.dump(rows, fh, indent=2, allow_nan=False)
    with open(out / "paired.md", "w", encoding="utf-8") as fh:
        fh.write(to_markdown(rows))
    print("E7 complete. Saved results/e7/paired.json and paired.md")
