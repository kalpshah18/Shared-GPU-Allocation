"""E13: asynchronous policy best responses trained on five independent seeds.

The heldout audit exhausts all unilateral switches in the finite policy library.
It estimates max_{i,a} E[U_i(a, profile_-i)-U_i(profile)] (max AFTER averaging).
Simultaneous bootstrap bands account for inspecting all user/policy switches.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from analysis.bootstrap import paired_differences, summarise_seeds
from experiments.common import TRUTHFUL_KEYS, overrides_from_args, save_outputs, save_raw
from experiments.extensions import SPECS, TRAINING_SEEDS, extension_seeds, parser, provenance
from sim.adaptation import POLICY_NAMES, ProfileEvaluator, best_response_dynamics, run_profile
from sim.config import Config
from sim.environment import SeedPackage
from sim.metrics import compute_all, utilities

KEYS = TRUTHFUL_KEYS + ["mean_utility", "utility_gain_vs_truthful", "truthful_fraction"]
INITIAL_POLICIES = ["truthful", "cap_2"]


def exploitability_band(gains, n_bootstrap=10_000):
    """Paired simultaneous 95% band for all expected unilateral gains."""
    values = np.asarray(gains, dtype=float)
    means = values.mean(axis=0)
    rng = np.random.default_rng(42)
    # Chunking avoids allocating (10000, seeds, users, policies) at once.
    radii = []
    for start in range(0, n_bootstrap, 200):
        idx = rng.integers(0, len(values), size=(min(200, n_bootstrap - start), len(values)))
        centered = values[idx].mean(axis=1) - means
        radii.extend(np.max(np.abs(centered), axis=(1, 2)).tolist())
    radius = float(np.percentile(radii, 95))
    maximum = max(0.0, float(means.max()))
    return {"mean": maximum, "ci_lower": max(0.0, maximum - radius),
            "ci_upper": maximum + radius, "simultaneous_radius": radius,
            "per_user": np.maximum(0, means.max(axis=1)).tolist(),
            "expected_gain_matrix": means.tolist()}


def audit_profile(mname, cfg, pkg, profile):
    history = run_profile(mname, pkg, cfg, profile)
    base = utilities(history, pkg.valuations)
    truthful = run_profile(mname, pkg, cfg, [0] * cfg.n)
    row = compute_all(history, pkg.valuations, cfg)
    row.update(mean_utility=float(base.mean()),
               utility_gain_vs_truthful=float((base - utilities(truthful, pkg.valuations)).mean()),
               truthful_fraction=float(np.mean(np.asarray(profile) == 0)))
    gains = np.zeros((cfg.n, len(POLICY_NAMES)))
    for user in range(cfg.n):
        for policy in range(len(POLICY_NAMES)):
            if policy == profile[user]:
                continue
            changed = list(profile)
            changed[user] = policy
            dev = run_profile(mname, pkg, cfg, changed)
            gains[user, policy] = utilities(dev, pkg.valuations)[user] - base[user]
    row["unilateral_gains"] = gains.tolist()
    row["seed"] = pkg.master_seed
    return row


def run_e13(seeds, results_dir=None, n=6, T=120, n_bootstrap=10_000, verbose=True,
            training_seeds=None, max_sweeps=4, specs=None):
    training_seeds = TRAINING_SEEDS if training_seeds is None else list(training_seeds)
    if set(seeds) & set(training_seeds):
        raise ValueError("Training and evaluation seeds must be disjoint.")
    summaries, cells = [], {}
    for label, mname, changes in (SPECS if specs is None else specs):
        cfg = Config(n=n, k=max(1, n // 3), T=T, rho=0.0, n_seeds=len(seeds),
                     n_bootstrap=n_bootstrap, seeds_file="seeds/extension_seeds.json", **changes)
        evaluator = ProfileEvaluator(mname, cfg, training_seeds)
        for initial in INITIAL_POLICIES:
            trained = best_response_dynamics(evaluator, [POLICY_NAMES.index(initial)] * n, max_sweeps)
            rows = [audit_profile(mname, cfg, SeedPackage.generate(seed, cfg), trained["profile"]) for seed in seeds]
            summary = summarise_seeds(rows, KEYS, n_bootstrap)
            summary.update(label=label, mechanism=mname, initial=initial, n_seeds=len(seeds),
                           profile=[POLICY_NAMES[p] for p in trained["profile"]],
                           training_status=trained["status"], training_sweeps=trained["sweeps"],
                           training_exploitability=trained["training_exploitability"],
                           restricted_exploitability=exploitability_band([r["unilateral_gains"] for r in rows], n_bootstrap))
            summaries.append(summary)
            cells[(label, initial)] = rows
            if results_dir:
                save_raw(results_dir, "e13", f"{label}_{initial}", rows,
                         {**cfg.to_dict(), "training_seeds": training_seeds, "policies": POLICY_NAMES,
                          "initial": initial, "training": trained})
            if verbose:
                print(f"E13 {label} from {initial}: {trained['status']} WR={summary['WR']['mean']:.3f} "
                      f"regret={summary['restricted_exploitability']['mean']:.3f} {summary['profile']}", flush=True)
    paired = []
    for initial in INITIAL_POLICIES:
        groups = {label: rows for (label, init), rows in cells.items() if init == initial}
        if "Score-Cap" in groups:
            diffs = paired_differences(groups, "Score-Cap", ["WR", "mean_utility", "Q_max"], n_bootstrap)
            paired.extend(dict(initial=initial, label=label, reference="Score-Cap", metrics=d) for label, d in diffs.items())
    if results_dir:
        save_outputs(results_dir, "e13", summaries,
                     {**provenance(), "n": n, "k": max(1, n // 3), "T": T, "n_bootstrap": n_bootstrap,
                      "training_seeds": training_seeds,
                      "policies": POLICY_NAMES, "max_sweeps": max_sweeps, "update_order_seed": 329,
                      "initial_policies": INITIAL_POLICIES,
                      "regret_definition": "max of expected paired unilateral gains, with simultaneous bootstrap band"}, seeds, paired)
    return summaries


if __name__ == "__main__":
    ap = parser(__doc__, 6, 120)
    ap.add_argument("--sweeps", type=int, default=4)
    args = ap.parse_args()
    run_e13(extension_seeds(args.seeds), args.results_dir, max_sweeps=args.sweeps, **overrides_from_args(args))
