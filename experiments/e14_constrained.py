"""E14: offline welfare benchmark under the SAME certified service requirement."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from analysis.bootstrap import paired_differences, summarise_seeds
from analysis.constrained_oracle import constrained_welfare, terminal_max_wait
from experiments.common import TRUTHFUL_KEYS, overrides_from_args, save_outputs, save_raw
from experiments.extensions import CONDITIONS, extension_seeds, parser, provenance
from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import make_mechanism
from sim.metrics import compute_all, oracle_welfare
from sim.runner import run_mixed

SPECS = [("Score-Cap", "ScoreCapMechanism"), ("Karma-Cap", "KarmaCapMechanism"), ("Rank-Cap", "RankCapMechanism")]
KEYS = TRUTHFUL_KEYS + ["constrained_efficiency", "constrained_efficiency_upper", "unavoidable_loss",
                       "algorithm_loss", "oracle_optimal", "oracle_mip_gap", "terminal_Q_max"]
CAPS = [3, 6, 9]


def run_e14(seeds, results_dir=None, n=6, T=40, n_bootstrap=10_000, verbose=True,
            caps=None, conditions=None, time_limit=30.0):
    caps = CAPS if caps is None else caps
    conditions = CONDITIONS if conditions is None else conditions
    k = max(1, n // 3)
    summaries, paired = [], []
    for condition, changes in conditions.items():
        for W in caps:
            B = W + math.ceil(n / k) - 1
            cfg = Config(n=n, k=k, T=T, rho=0.0, wait_cap=W, n_seeds=len(seeds),
                         n_bootstrap=n_bootstrap, seeds_file="seeds/extension_seeds.json", **changes)
            per_label = {label: [] for label, _ in SPECS}
            for seed in seeds:
                pkg = SeedPackage.generate(seed, cfg)
                result = constrained_welfare(pkg.valuations, k, B, time_limit)
                unconstrained = oracle_welfare(pkg.valuations, cfg)
                for label, mname in SPECS:
                    h = run_mixed(make_mechanism(mname, cfg, pkg), pkg, cfg, None)
                    X = np.stack(h.allocations, axis=1)
                    welfare = float(np.sum(pkg.valuations * X))
                    if terminal_max_wait(X) > B or welfare > result.upper_bound + 1e-5:
                        raise RuntimeError("Online schedule violated the oracle comparison's constraints.")
                    row = compute_all(h, pkg.valuations, cfg)
                    row.update(seed=seed, online_welfare=welfare, oracle_welfare=result.welfare,
                               oracle_upper_bound=result.upper_bound, terminal_Q_max=terminal_max_wait(X),
                               constrained_efficiency=welfare / result.upper_bound if result.upper_bound else 1.0,
                               constrained_efficiency_upper=welfare / result.welfare if result.welfare else 1.0,
                               unavoidable_loss=(unconstrained - result.upper_bound) / unconstrained if unconstrained else 0.0,
                               algorithm_loss=(result.upper_bound - welfare) / unconstrained if unconstrained else 0.0,
                               oracle_optimal=float(result.optimal), oracle_mip_gap=result.mip_gap)
                    per_label[label].append(row)
            for label, mname in SPECS:
                rows = per_label[label]
                summary = summarise_seeds(rows, KEYS, n_bootstrap)
                summary.update(label=label, mechanism=mname, condition=condition, wait_cap=W,
                               certified_bound=B, n_seeds=len(seeds))
                summaries.append(summary)
                if results_dir:
                    save_raw(results_dir, "e14", f"{condition}_{label}_W{W}", rows, cfg.to_dict())
                if verbose:
                    print(f"E14 {condition} W={W} B={B} {label}: efficiency={summary['constrained_efficiency']['mean']:.3f} "
                          f"optimal={summary['oracle_optimal']['mean']:.2f}", flush=True)
            diffs = paired_differences(per_label, "Score-Cap", ["WR", "constrained_efficiency", "algorithm_loss"], n_bootstrap)
            paired.extend(dict(condition=condition, wait_cap=W, label=label, reference="Score-Cap", metrics=d)
                          for label, d in diffs.items())
    if results_dir:
        save_outputs(results_dir, "e14", summaries,
                     {**provenance(), "n": n, "k": k, "T": T, "n_bootstrap": n_bootstrap,
                      "caps": caps, "conditions": conditions,
                      "oracle": "binary MILP, capacity equality, all B+1 service windows including terminal wait",
                      "bound": "B=W+ceil(n/k)-1", "time_limit_seconds": time_limit}, seeds, paired)
    return summaries


if __name__ == "__main__":
    args = parser(__doc__, 6, 40).parse_args()
    run_e14(extension_seeds(args.seeds), args.results_dir, **overrides_from_args(args))
