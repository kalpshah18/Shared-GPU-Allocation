"""
experiments/proposed.py
========================
Mechanism specifications shared by the experiments that evaluate the two
proposed mechanisms (e9, e10, e11).

Each spec is (label, mechanism class name, Config overrides).  The overrides
(karma_init = 2.0, wait_cap = Delta, rank_blend = 0) are the defaults chosen on
*development seeds* that are not among the 30 locked master seeds:

    DEV_SEEDS = [101, ..., 105]

Selection rule (fixed before the locked-seed evaluation): for M6 take the
largest karma supply whose individual gain from lying is <= 0 for every lie
on the development grid {cap 1.25, 1.5, 2, max claim, timed 1.25, timed 2};
the waiting cap is left at its natural value W = Delta = 2*ceil(n/k).
"""

from __future__ import annotations

DEV_SEEDS = [101, 102, 103, 104, 105]

BASELINES = [
    ("M1 Random",       "RandomMechanism",      {}),
    ("M2 Round-Robin",  "RoundRobinMechanism",  {}),
    ("M3 Greedy",       "GreedyMechanism",      {}),
    ("M4 Score λ=1",    "ScoreMechanism",       {"lambda_": 1.0}),
    ("M4 Score λ=2",    "ScoreMechanism",       {"lambda_": 2.0}),
    ("M5 Vickrey",      "VickreyMechanism",     {}),
]
PROPOSED = [
    ("M6 Karma-Cap",         "KarmaCapMechanism", {}),
    ("M7 Rank-Cap",          "RankCapMechanism",  {}),
]
VARIANTS = [
    ("M7 Rank-Cap β=0.25",   "RankCapMechanism",  {"rank_blend": 0.25}),
]


def slug(label: str) -> str:
    """File-system friendly cell name."""
    return (label.replace(" ", "_").replace("λ=", "lam").replace("β=", "beta")
            .replace("/", "-"))
