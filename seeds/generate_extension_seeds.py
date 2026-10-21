"""Lock a fresh holdout for E12-E15 before inspecting any new outcomes."""
import json
from pathlib import Path
import numpy as np

PATH = Path(__file__).with_name("extension_seeds.json")
META_SEED = 0x329_2026
TRAINING_SEEDS = [201, 202, 203, 204, 205]


def main():
    if PATH.exists():
        raise SystemExit("Extension seeds already exist; refusing to replace the holdout.")
    old = json.loads(PATH.with_name("master_seeds.json").read_text())["seeds"]
    excluded = set(old + TRAINING_SEEDS + [101, 102, 103, 104, 105])
    rng, seeds = np.random.default_rng(META_SEED), []
    while len(seeds) < 30:
        seed = int(rng.integers(0, 2**31))
        if seed not in excluded:
            seeds.append(seed)
            excluded.add(seed)
    PATH.write_text(json.dumps({"description": "Fresh held-out evaluation for fixed E12-E15 protocol; do not retune on these results.",
                                "meta_seed": META_SEED, "n_seeds": 30,
                                "training_seeds": TRAINING_SEEDS, "seeds": seeds}, indent=2) + "\n")


if __name__ == "__main__":
    main()
