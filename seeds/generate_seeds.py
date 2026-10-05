"""
seeds/generate_seeds.py
========================
Generate and lock the master seed list.

Run this script ONCE before any main experiments (E1–E6) to fix the 30
master seeds.  After generation, the JSON file should be committed to version
control and never regenerated — any change would invalidate reproducibility.

Usage
-----
    python seeds/generate_seeds.py

Output: seeds/master_seeds.json
"""

import json
import sys
from pathlib import Path

import numpy as np

N_SEEDS   = 30
SEED_FILE = Path(__file__).parent / "master_seeds.json"


def main() -> None:
    if SEED_FILE.exists():
        print(f"[WARN] {SEED_FILE} already exists.  Delete it manually if you "
              "truly need to regenerate.  Aborting.")
        sys.exit(1)

    META_SEED = 0xFA1_5EED
    meta_rng = np.random.default_rng(META_SEED)
    seeds = meta_rng.integers(0, 2**31, size=N_SEEDS).tolist()

    data = {
        "description": (
            "Master seeds for FAI GPU allocation simulation study. "
            "Generated once; do not regenerate after experiments begin."
        ),
        "n_seeds"    : N_SEEDS,
        "seeds"      : seeds,
    }

    SEED_FILE.parent.mkdir(exist_ok=True)
    with open(SEED_FILE, "w") as fh:
        json.dump(data, fh, indent=2)

    print(f"Generated {N_SEEDS} master seeds -> {SEED_FILE}")
    print("Commit this file to version control now.")


if __name__ == "__main__":
    main()
