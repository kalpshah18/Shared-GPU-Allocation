"""sim/mechanisms/__init__.py — re-exports and the mechanism factory."""

from sim.mechanisms.base         import Mechanism
from sim.mechanisms.m1_random    import RandomMechanism
from sim.mechanisms.m2_roundrobin import RoundRobinMechanism
from sim.mechanisms.m3_greedy    import GreedyMechanism
from sim.mechanisms.m4_score     import ScoreMechanism
from sim.mechanisms.m5_vickrey   import VickreyMechanism
from sim.mechanisms.m6_karma     import KarmaCapMechanism
from sim.mechanisms.m7_rank      import RankCapMechanism

__all__ = [
    "Mechanism",
    "RandomMechanism",
    "RoundRobinMechanism",
    "GreedyMechanism",
    "ScoreMechanism",
    "VickreyMechanism",
    "KarmaCapMechanism",
    "RankCapMechanism",
    "MECHANISM_NAMES",
    "PROPOSED_MECHANISM_NAMES",
    "ALL_MECHANISM_NAMES",
    "MECHANISM_LABELS",
    "make_mechanism",
]

# Canonical M1..M5 order (deterministic; experiments iterate over this).
MECHANISM_NAMES = [
    "RandomMechanism",
    "RoundRobinMechanism",
    "GreedyMechanism",
    "ScoreMechanism",
    "VickreyMechanism",
]

# The two mechanisms proposed in this project (extensions beyond the five baselines).
PROPOSED_MECHANISM_NAMES = ["KarmaCapMechanism", "RankCapMechanism"]
ALL_MECHANISM_NAMES = MECHANISM_NAMES + PROPOSED_MECHANISM_NAMES

MECHANISM_LABELS = {
    "RandomMechanism"    : "M1 Random",
    "RoundRobinMechanism": "M2 Round-Robin",
    "GreedyMechanism"    : "M3 Greedy",
    "ScoreMechanism"     : "M4 Score",
    "VickreyMechanism"   : "M5 Vickrey",
    "KarmaCapMechanism"  : "M6 Karma-Cap",
    "RankCapMechanism"   : "M7 Rank-Cap",
}


def make_mechanism(name: str, cfg, pkg=None) -> Mechanism:
    """
    Build a fresh mechanism instance by class name.

    M2 is stateful: its initial queue is drawn from the round-0 tie seed of
    `pkg`, so a SeedPackage is required for it (and must not be reused across
    runs — build a new instance for every simulation).
    """
    if name == "RandomMechanism":
        return RandomMechanism(cfg)
    if name == "RoundRobinMechanism":
        if pkg is None:
            raise ValueError("RoundRobinMechanism needs a SeedPackage for its queue seed.")
        return RoundRobinMechanism(cfg, init_seed=int(pkg.tie_seeds[0]))
    if name == "GreedyMechanism":
        return GreedyMechanism(cfg)
    if name == "ScoreMechanism":
        return ScoreMechanism(cfg)
    if name == "VickreyMechanism":
        return VickreyMechanism(cfg)
    if name == "KarmaCapMechanism":
        return KarmaCapMechanism(cfg)
    if name == "RankCapMechanism":
        return RankCapMechanism(cfg)
    raise ValueError(f"Unknown mechanism '{name}'. Known: {ALL_MECHANISM_NAMES}")
