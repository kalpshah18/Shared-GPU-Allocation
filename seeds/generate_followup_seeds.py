"""Lock independent evaluation seeds for the fixed extended-training protocol."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
META_SEED = 0x329182026


def generate():
    destination = ROOT / "followup_seeds.json"
    if destination.exists():
        raise FileExistsError("Follow-up seeds are already locked; refusing to replace them.")
    excluded = set(range(101, 106)) | set(range(201, 206))
    for name in ("master_seeds.json", "extension_seeds.json"):
        excluded.update(json.loads((ROOT / name).read_text())["seeds"])
    rng, seeds = np.random.default_rng(META_SEED), []
    while len(seeds) < 30:
        seed = int(rng.integers(0, 2**31))
        if seed not in excluded and seed not in seeds:
            seeds.append(seed)
    destination.write_text(json.dumps({"description": "Independent E18 evaluation; fixed four-vs-twenty training budget.",
                                       "meta_seed": META_SEED, "training_seeds": list(range(201, 206)),
                                       "n_seeds": 30, "seeds": seeds}, indent=2) + "\n")
    print(f"Locked {len(seeds)} follow-up seeds in {destination}")


if __name__ == "__main__":
    generate()
