"""Explain Karma's population reversal with exact service and credit accounting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from analysis.bootstrap import paired_differences, summarise_seeds
from analysis.failure_diagnostics import karma_statistics, karma_trace
from experiments.common import overrides_from_args, save_outputs, save_raw, scaled_k
from experiments.extensions import extension_seeds, parser, provenance
from sim.config import Config
from sim.environment import SeedPackage, save_json
from sim.runner import focal_users

KEYS = ["M_uni", "forced_gain", "auction_gain", "wins_change", "forced_wins_change", "auction_wins_change",
        "credit_spent_change", "forced_credit_spent_change", "auction_credit_spent_change",
        "deviating_balance_limited", "truthful_balance_limited", "deviating_mean_balance", "truthful_mean_balance",
        "deviating_value_per_win", "truthful_value_per_win"]


def focal_accounting(dev, truth):
    row = {"M_uni": dev["utility"]-truth["utility"]}
    for key in ("forced_utility", "auction_utility", "wins", "forced_wins", "auction_wins",
                "credit_spent", "forced_credit_spent", "auction_credit_spent"):
        name = {"forced_utility": "forced_gain", "auction_utility": "auction_gain"}.get(key, key+"_change")
        row[name] = dev[key]-truth[key]
    for prefix, data in (("deviating", dev), ("truthful", truth)):
        row.update({f"{prefix}_balance_limited": data["balance_limited_fraction"],
                    f"{prefix}_mean_balance": data["mean_balance"],
                    f"{prefix}_value_per_win": data["value_per_win"]})
    return row


def run_e17(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10000, verbose=True):
    summaries, groups, examples = [], {}, {}
    for rho in (.25, 1.0):
        for enabled in (False, True):
            cfg = Config(n=n, k=scaled_k(n, .2), T=T, rho=rho, wait_cap=10, enable_wait_cap=enabled,
                         n_seeds=len(seeds), n_bootstrap=n_bootstrap, seeds_file="seeds/extension_seeds.json")
            label = "Karma-Cap" if enabled else "Karma"
            rows = []
            for seed in seeds:
                pkg = SeedPackage.generate(seed, cfg)
                focals = focal_users(pkg, cfg)
                S = set(int(i) for i in pkg.strategic_set)
                _, shared = karma_trace(pkg, cfg, sorted(S), observe=focals)
                individual = []
                for f in focals:
                    f = int(f)
                    changed = sorted(S-{f}) if f in S else sorted(S|{f})
                    _, other = karma_trace(pkg, cfg, changed, observe=[f])
                    dev, truth = (shared[f], other[f]) if f in S else (other[f], shared[f])
                    item = focal_accounting(karma_statistics(dev), karma_statistics(truth))
                    item["focal"] = f
                    individual.append(item)
                    if seed == seeds[0] and f == int(focals[0]):
                        examples[f"{label}_rho{rho:g}"] = dict(seed=seed, focal=f, dev=dev, truthful=truth)
                row = {key: float(np.mean([item[key] for item in individual])) for key in KEYS}
                row.update(seed=seed, individual=individual)
                assert abs(row["forced_gain"]+row["auction_gain"]-row["M_uni"]) < 1e-9
                rows.append(row)
            summary = summarise_seeds(rows, KEYS, n_bootstrap)
            summary.update(label=label, rho=rho, n_seeds=len(seeds))
            summaries.append(summary)
            groups[(rho, label)] = rows
            if results_dir:
                save_raw(results_dir, "e17", f"{label}_rho{rho:g}", rows, cfg.to_dict())
            if verbose:
                print(f"E17 {label} rho={rho:g}: gain={summary['M_uni']['mean']:+.3f} "
                      f"forced={summary['forced_gain']['mean']:+.3f} auction={summary['auction_gain']['mean']:+.3f}", flush=True)
    paired = []
    for rho in (.25, 1.0):
        diff = paired_differences({label: rows for (r, label), rows in groups.items() if r == rho}, "Karma", KEYS, n_bootstrap)
        paired.append(dict(rho=rho, comparison="cap_effect", metrics=diff["Karma-Cap"]))
    if results_dir:
        save_json(examples, Path(results_dir)/"e17"/"example.json")
        save_outputs(results_dir, "e17", summaries,
                     {**provenance(), "protocol_version": "failure-followup-v1", "seed_role": "descriptive_reanalysis",
                      "n": n, "k": scaled_k(n, .2), "T": T, "wait_cap": 10, "policy": "cap_2",
                      "n_bootstrap": n_bootstrap, "n_focal": 5, "example_seed": seeds[0],
                      "accounting": "internal charges verified each round; utility is true benefit only"}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = parser(__doc__, 50, 1000).parse_args()
    run_e17(extension_seeds(args.seeds), args.results_dir, **overrides_from_args(args))
