"""Compare four and twenty training sweeps on a new independent evaluation set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bootstrap import paired_differences, summarise_seeds
from experiments.common import load_seeds, overrides_from_args, save_outputs, save_raw
from experiments.e13_adaptive import INITIAL_POLICIES, KEYS, audit_profile, exploitability_band
from experiments.extensions import ROOT, SPECS, TRAINING_SEEDS, parser, provenance
from sim.adaptation import POLICY_NAMES, ProfileEvaluator, best_response_dynamics
from sim.config import Config
from sim.environment import SeedPackage

SEED_FILE = ROOT/"seeds"/"followup_seeds.json"


def run_e18(seeds, results_dir=None, n=6, T=120, n_bootstrap=10000, verbose=True,
            budgets=(4, 20), specs=None, training_seeds=None):
    training_seeds = TRAINING_SEEDS if training_seeds is None else training_seeds
    if set(seeds) & set(training_seeds):
        raise ValueError("Training and evaluation seeds must be disjoint.")
    summaries, groups = [], {}
    for label, mechanism, changes in (SPECS if specs is None else specs):
        cfg = Config(n=n, k=max(1, n//3), T=T, rho=0, n_seeds=len(seeds), n_bootstrap=n_bootstrap,
                     seeds_file="seeds/followup_seeds.json", **changes)
        evaluator = ProfileEvaluator(mechanism, cfg, training_seeds)
        for initial in INITIAL_POLICIES:
            for budget in budgets:
                trained = best_response_dynamics(evaluator, [POLICY_NAMES.index(initial)]*n, budget,
                                                   stop_on_recurrence=False)
                rows = [audit_profile(mechanism, cfg, SeedPackage.generate(seed, cfg), trained["profile"]) for seed in seeds]
                summary = summarise_seeds(rows, KEYS, n_bootstrap)
                summary.update(label=label, initial=initial, budget=budget, n_seeds=len(seeds),
                               profile=[POLICY_NAMES[p] for p in trained["profile"]],
                               training_status=trained["status"], training_sweeps=trained["sweeps"],
                               recurrent_sweeps=trained["recurrent_sweeps"],
                               training_exploitability=trained["training_exploitability"],
                               restricted_exploitability=exploitability_band([r["unilateral_gains"] for r in rows], n_bootstrap))
                summaries.append(summary)
                groups[(label, initial, budget)] = rows
                if results_dir:
                    save_raw(results_dir, "e18", f"{label}_{initial}_budget{budget}", rows,
                             {**cfg.to_dict(), "training_seeds": training_seeds, "initial": initial,
                              "budget": budget, "policies": POLICY_NAMES, "training": trained})
                if verbose:
                    print(f"E18 {label} {initial} budget={budget}: {trained['status']} "
                          f"sweeps={trained['sweeps']} train_regret={trained['training_exploitability']:.3f} "
                          f"heldout_regret={summary['restricted_exploitability']['mean']:.3f}", flush=True)
    paired = []
    for label, initial, budget in groups:
        if budget == min(budgets):
            continue
        diff = paired_differences({"longer": groups[(label, initial, budget)],
                                   "shorter": groups[(label, initial, min(budgets))]}, "shorter", ["WR", "mean_utility"], n_bootstrap)
        paired.append(dict(label=label, initial=initial, budget=budget, reference_budget=min(budgets), metrics=diff["longer"]))
    if results_dir:
        save_outputs(results_dir, "e18", summaries,
                     {**provenance(), "protocol_version": "failure-followup-v1", "seed_file": "seeds/followup_seeds.json",
                      "n": n, "k": cfg.k, "T": T, "n_bootstrap": n_bootstrap, "training_seeds": training_seeds,
                      "budgets": list(budgets), "stop_on_recurrence": False, "update_order_seed": 329,
                      "policies": POLICY_NAMES, "initial_policies": INITIAL_POLICIES}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = parser(__doc__, 6, 120).parse_args()
    run_e18(load_seeds(SEED_FILE, args.seeds), args.results_dir, **overrides_from_args(args))
