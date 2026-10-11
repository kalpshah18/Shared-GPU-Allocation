"""E12: isolate waiting caps, karma and rank using matched thresholds and workloads."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.bootstrap import paired_differences
from experiments.common import (STRATEGIC_KEYS, overrides_from_args, run_cell, save_outputs,
                                save_raw, scaled_k, strategic_row)
from experiments.extensions import CONDITIONS, extension_seeds, parser, provenance
from sim.config import Config
from sim.policies import get_policy

CAPS = [10, 15, 20]
FAMILIES = [("Score", "ScoreMechanism", "ScoreCapMechanism"),
            ("Karma", "KarmaCapMechanism", "KarmaCapMechanism"),
            ("Rank", "RankCapMechanism", "RankCapMechanism")]
ATTACKS = [("cap_1.25", 0.25), ("cap_2", 0.25), ("timed_cap_1.25", 0.25), ("cap_2", 1.0)]


def run_e12(seeds, results_dir=None, n=50, T=1000, n_bootstrap=10_000, verbose=True, caps=None):
    caps = CAPS if caps is None else caps
    specs = []
    for family, plain, capped in FAMILIES:
        specs.append((family, plain, False, None))
        specs.extend((family, capped, True, W) for W in caps)
    jobs = [(condition, changes, "truthful", 0.0) for condition, changes in CONDITIONS.items()]
    jobs += [("base", {}, policy, rho) for policy, rho in ATTACKS]
    summaries, grouped = [], {}
    for condition, changes, policy, rho in jobs:
        cells = {}
        for family, mname, enabled, W in specs:
            cfg = Config(n=n, k=scaled_k(n, 0.2), T=T, rho=rho, enable_wait_cap=enabled,
                         wait_cap=W, n_seeds=len(seeds), n_bootstrap=n_bootstrap,
                         seeds_file="seeds/extension_seeds.json", **changes)
            label = family if not enabled else f"{family}-Cap W={W}"
            pol = None if policy == "truthful" else get_policy(policy)
            rows, summary = run_cell(lambda pkg: strategic_row(mname, cfg, pkg, pol),
                                     seeds, cfg, STRATEGIC_KEYS, n_bootstrap)
            summary.update(condition=condition, family=family, label=label, mechanism=mname,
                           cap_enabled=enabled, wait_cap=W, policy=policy, rho=rho, n_seeds=len(seeds))
            summaries.append(summary)
            cells[label] = rows
            if results_dir:
                save_raw(results_dir, "e12", f"{condition}_{policy}_rho{rho:g}_{family}_W{W}", rows, cfg.to_dict())
            if verbose:
                print(f"E12 {condition} {policy} rho={rho:g} {label}: WR={summary['WR']['mean']:.3f} M={summary['M_uni']['mean']:+.2f}", flush=True)
        grouped[(condition, policy, rho)] = cells
    paired = []
    for (condition, policy, rho), cells in grouped.items():
        # Within-family changes isolate the cap; matched-W comparisons isolate the incentive component.
        for family, _, _ in FAMILIES:
            groups = {key: val for key, val in cells.items() if key == family or key.startswith(family + "-Cap")}
            diffs = paired_differences(groups, family, ["WR", "M_uni", "Q_max", "SR_delta"], n_bootstrap)
            paired.extend(dict(condition=condition, policy=policy, rho=rho, comparison="cap_effect",
                               label=label, reference=family, metrics=d) for label, d in diffs.items())
        for W in caps:
            ref = f"Score-Cap W={W}"
            groups = {key: val for key, val in cells.items() if key.endswith(f"-Cap W={W}")}
            diffs = paired_differences(groups, ref, ["WR", "M_uni", "Q_max", "SR_delta"], n_bootstrap)
            paired.extend(dict(condition=condition, policy=policy, rho=rho, comparison="component_effect",
                               label=label, reference=ref, metrics=d) for label, d in diffs.items())
    if results_dir:
        save_outputs(results_dir, "e12", summaries,
                     {**provenance(), "n": n, "k": scaled_k(n, 0.2), "T": T, "n_bootstrap": n_bootstrap,
                      "caps": caps, "conditions": CONDITIONS, "attacks": ATTACKS}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = parser(__doc__, 50, 1000).parse_args()
    run_e12(extension_seeds(args.seeds), args.results_dir, **overrides_from_args(args))
