"""Explain the Rank-Cap planning failure using trajectories and ledger replay."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from analysis.bootstrap import paired_differences, summarise_seeds
from analysis.failure_diagnostics import rank_statistics, rank_trace, service_totals
from experiments.common import overrides_from_args, save_outputs, save_raw, scaled_k
from experiments.extensions import extension_seeds, parser, provenance
from sim.config import Config
from sim.environment import SeedPackage, save_json

KEYS = ["M_rollout", "forced_gain", "auction_gain", "clean_reference_gain", "ledger_effect",
        "underreport_fraction", "overreport_fraction", "mean_report", "mean_value",
        "mean_history_rank_shift", "history_boost_fraction", "mean_shift_high_values", "underreport_value"]


def run_e16(seeds, results_dir=None, n=10, T=120, n_bootstrap=10000, verbose=True, H=40, R=80):
    if not seeds:
        raise ValueError("At least one seed is required.")
    rows_by_mode = {"updating": [], "frozen": []}
    examples = {}
    cfg = Config(n=n, k=scaled_k(n, .2), T=T, rho=0, n_seeds=len(seeds), n_bootstrap=n_bootstrap,
                 seeds_file="seeds/extension_seeds.json")
    for seed in seeds:
        pkg = SeedPackage.generate(seed, cfg)
        _, baseline = rank_trace(pkg, cfg, reports=pkg.valuations[0])
        base = service_totals(baseline)
        for mode in rows_by_mode:
            attack_cfg = cfg.replace(rank_rollout_update=mode == "updating")
            _, trace = rank_trace(pkg, attack_cfg, H=H, R=R)
            sequence = np.array([r["report"] for r in trace])
            _, clean = rank_trace(pkg, cfg, reports=sequence, truth_reference=True)
            attack, replay = service_totals(trace), service_totals(clean)
            row = dict(seed=seed, M_rollout=attack["utility"]-base["utility"],
                       forced_gain=attack["forced_utility"]-base["forced_utility"],
                       auction_gain=attack["auction_utility"]-base["auction_utility"],
                       clean_reference_gain=replay["utility"]-base["utility"],
                       ledger_effect=attack["utility"]-replay["utility"], **rank_statistics(trace))
            assert abs(row["forced_gain"]+row["auction_gain"]-row["M_rollout"]) < 1e-10
            rows_by_mode[mode].append(row)
            if results_dir:
                save_json(dict(seed=seed, mode=mode, baseline=baseline, attack=trace, clean_reference=clean),
                          Path(results_dir)/"e16"/"traces"/f"{mode}_{seed}.json")
            if seed == seeds[0]:
                examples[mode] = dict(seed=seed, baseline=baseline, attack=trace, clean_reference=clean)
            if verbose:
                print(f"E16 seed={seed} {mode}: gain={row['M_rollout']:+.3f} "
                      f"ledger_effect={row['ledger_effect']:+.3f}", flush=True)
    summaries = []
    for mode, rows in rows_by_mode.items():
        summary = summarise_seeds(rows, KEYS, n_bootstrap)
        summary.update(mode=mode, n_seeds=len(seeds))
        summaries.append(summary)
        if results_dir:
            save_raw(results_dir, "e16", mode, rows, {**cfg.to_dict(), "H": H, "n_rollouts": R,
                                                     "rank_rollout_update": mode == "updating"})
    differences = paired_differences(rows_by_mode, "frozen", KEYS, n_bootstrap)
    if results_dir:
        save_json(examples, Path(results_dir)/"e16"/"example.json")
        save_outputs(results_dir, "e16", summaries,
                     {**provenance(), "protocol_version": "failure-followup-v1", "seed_role": "descriptive_reanalysis",
                      "n": n, "k": cfg.k, "T": T, "H": H, "n_rollouts": R, "n_bootstrap": n_bootstrap,
                      "example_seed": seeds[0], "example_selection": "first locked seed by list order",
                      "intervention": "fixed report sequence; truthful focal reference; privileged information"},
                     seeds, differences)
    return summaries


if __name__ == "__main__":
    args = parser(__doc__, 10, 120).parse_args()
    run_e16(extension_seeds(args.seeds), args.results_dir, **overrides_from_args(args))
