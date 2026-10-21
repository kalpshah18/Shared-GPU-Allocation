"""E15: corrected planning attack, computation budgets and a frozen-rank control.

All candidates include exact truth; horizons stop at episode end. Updating rank
histories is the default. Frozen-history rows change only this approximation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from analysis.bootstrap import paired_differences, summarise_seeds
from experiments.common import overrides_from_args, save_outputs, save_raw, scaled_k
from experiments.extensions import SPECS, extension_seeds, parser, provenance
from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import make_mechanism
from sim.metrics import utilities
from sim.runner import run_mixed, run_rollout

BUDGETS = [(5, 20), (5, 80), (40, 20), (40, 80)]
KEYS = ["M_rollout", "mean_inflation", "frac_max_report", "frac_truthful_report"]
ATTACK_SPECS = SPECS + [("Score lambda=2", "ScoreMechanism", {"lambda_": 2.0})]


def run_e15(seeds, results_dir=None, n=10, T=120, n_bootstrap=10_000, verbose=True,
            budgets=None, specs=None):
    budgets = BUDGETS if budgets is None else budgets
    specs = ATTACK_SPECS if specs is None else specs
    summaries, grouped = [], {}
    for label, mname, changes in specs:
        modes = ["updating", "frozen"] if mname == "RankCapMechanism" else ["exact"]
        for mode in modes:
            selected = budgets if mode != "frozen" else [b for b in budgets if b[0] == max(h for h, _ in budgets)]
            cfg = Config(n=n, k=scaled_k(n, 0.2), T=T, rho=0.0, n_seeds=len(seeds),
                         n_bootstrap=n_bootstrap, seeds_file="seeds/extension_seeds.json",
                         **{**changes, "rank_rollout_update": mode != "frozen"})
            packages = [SeedPackage.generate(seed, cfg) for seed in seeds]
            bases = [utilities(run_mixed(make_mechanism(mname, cfg, pkg), pkg, cfg, None), pkg.valuations)[0]
                     for pkg in packages]
            for H, R in selected:
                rows = []
                for pkg, base in zip(packages, bases):
                    rng = np.random.default_rng([pkg.master_seed, 159, H, R])
                    history, reports = run_rollout(make_mechanism(mname, cfg, pkg), pkg, cfg, 0, None,
                                                    rng, H=H, n_rollouts=R)
                    v = pkg.valuations[0]
                    rows.append(dict(seed=pkg.master_seed,
                                     M_rollout=float(utilities(history, pkg.valuations)[0] - base),
                                     mean_inflation=float(np.mean(reports - v)),
                                     frac_max_report=float(np.mean(reports >= cfg.v_max - 1e-12)),
                                     frac_truthful_report=float(np.mean(np.abs(reports - v) < 1e-12))))
                summary = summarise_seeds(rows, KEYS, n_bootstrap)
                summary.update(label=label, mechanism=mname, mode=mode, H=H, n_rollouts=R,
                               n_seeds=len(seeds))
                summaries.append(summary)
                grouped[(label, mode, H, R)] = rows
                if results_dir:
                    save_raw(results_dir, "e15", f"{label}_{mode}_H{H}_R{R}", rows,
                             {**cfg.to_dict(), "H": H, "n_rollouts": R, "exact_truth_candidate": True,
                              "truncate_at_episode_end": True})
                if verbose:
                    print(f"E15 {label} {mode} H={H} R={R}: gain={summary['M_rollout']['mean']:+.3f}", flush=True)
    paired = []
    for (label, mode, H, R), rows in grouped.items():
        comparisons = []
        if mode == "updating" and (label, "frozen", H, R) in grouped:
            comparisons.append(("rank_history_effect", (label, "frozen", H, R)))
        baseline = min(budgets)
        if (H, R) != baseline and (label, mode, *baseline) in grouped:
            comparisons.append(("budget_effect", (label, mode, *baseline)))
        for comparison, reference in comparisons:
            d = paired_differences({"candidate": rows, "reference": grouped[reference]},
                                   "reference", ["M_rollout"], n_bootstrap)["candidate"]
            paired.append(dict(label=label, mode=mode, H=H, n_rollouts=R, comparison=comparison,
                               reference=list(reference), metrics=d))
    if results_dir:
        save_outputs(results_dir, "e15", summaries,
                     {**provenance(), "n": n, "k": scaled_k(n, 0.2), "T": T, "n_bootstrap": n_bootstrap,
                      "budgets": budgets,
                      "focal": 0, "opponents": "truthful", "future_values": "iid marginals",
                      "continuation_policy": "truthful after the first simulated round",
                      "exact_truth_candidate": True, "truncate_at_episode_end": True,
                      "rank_histories": "updating plus matched frozen controls"}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = parser(__doc__, 10, 120).parse_args()
    run_e15(extension_seeds(args.seeds), args.results_dir, **overrides_from_args(args))
