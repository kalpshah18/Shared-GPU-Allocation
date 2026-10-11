"""Fixed specifications, held-out seeds and provenance for the course extension."""
import hashlib
import json
from pathlib import Path

from experiments.common import ROOT, build_parser, load_seeds

PROTOCOL_VERSION = "course-extension-v1"
SEED_FILE = ROOT / "seeds" / "extension_seeds.json"
TRAINING_SEEDS = [201, 202, 203, 204, 205]
SPECS = [
    ("Greedy", "GreedyMechanism", {}),
    ("Score", "ScoreMechanism", {"lambda_": 1.0}),
    ("Score-Cap", "ScoreCapMechanism", {"lambda_": 1.0}),
    ("Vickrey", "VickreyMechanism", {}),
    ("Karma-Cap", "KarmaCapMechanism", {}),
    ("Rank-Cap", "RankCapMechanism", {}),
]
CONDITIONS = {"base": {}, "mixed": {"valuation_dist": "mixed"},
              "persistent": {"alpha": 0.9, "ar1_mode": "copula"}}


def extension_seeds(n=None):
    return load_seeds(SEED_FILE, n)


def parser(description, n, T):
    ap = build_parser(description)
    ap.set_defaults(n=n, T=T)
    return ap


def provenance():
    paths = sorted(p for directory in ("sim", "experiments", "analysis")
                   for p in (ROOT / directory).rglob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return {"protocol_version": PROTOCOL_VERSION, "seed_role": "heldout",
            "seed_file": "seeds/extension_seeds.json", "source_sha256": digest.hexdigest()}
