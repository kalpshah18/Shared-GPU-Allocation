"""
experiments/e0_validation.py
=============================
E0 — Validation and sanity checks — Owner: Harsh Dhru

Exhaustive correctness checks on small discrete instances (proposal section
4.3).  They must all pass before any main experiment runs; they exist to
catch implementation errors.  Every check enumerates *all* report profiles on
a finite grid rather than sampling them.

Checks
------
 1. Capacity invariant        sum_i x_i = k, for every mechanism, every report
                              profile, every history and tie seed
 2. Payment invariant         p_i = 0 whenever x_i = 0 (and p >= 0)
 3. Report-invariance         M1, M2: identical outcome for every pair of
                              report profiles
 4. M3 == M4 at lambda = 0    identical allocation on every instance
 5. Round-robin wait bound    Q_max <= ceil(n/k) - 1 for every 1 <= k < n <= 9,
                              and service counts differ by at most 1
 6. Welfare oracle            M3 (and M4 at lambda = 0) attain W* on all
                              truthful profiles
 7. M5 one-round DSIC         truthful bidding maximises utility against
                              every opponent profile; payments equal the
                              (k+1)-st highest bid; individual rationality
 8. Negative controls         the DSIC enumerator FINDS a profitable
                              inflation under M3 and M4, so a pass for M5 is
                              not vacuous
 9. Determinism               identical seeds give identical histories
10. Batch consistency         allocate_batch == allocate (M3, M4, M5)
11. Metric cross-check        vectorised waits/Q_max/SR/percentile equal a
                              naive per-round loop on random histories

Usage
-----
    python experiments/e0_validation.py

Prints one line per check and exits with code 1 on any failure.
"""

from __future__ import annotations

import math
import sys
from itertools import product
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sim import metrics as M
from sim.config import Config
from sim.environment import History, SeedPackage
from sim.mechanisms import (
    GreedyMechanism,
    RandomMechanism,
    RoundRobinMechanism,
    ScoreMechanism,
    VickreyMechanism,
)
from sim.runner import run_single

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
TIE_SEEDS = (0, 1, 2)


def _mech_factories(cfg: Config) -> dict:
    return {
        "M1": lambda: RandomMechanism(cfg),
        "M2": lambda: RoundRobinMechanism(cfg, init_seed=0),
        "M3": lambda: GreedyMechanism(cfg),
        "M4": lambda: ScoreMechanism(cfg),
        "M5": lambda: VickreyMechanism(cfg),
    }


def _history(n: int, cumulative=None) -> History:
    h = History(n)
    if cumulative is not None:
        h.cumulative[:] = cumulative
    return h


def _grid(levels: int) -> list:
    return [i / (levels - 1) for i in range(levels)]


def _profiles(n: int, levels: int):
    for combo in product(_grid(levels), repeat=n):
        yield np.array(combo, dtype=np.float64)


# ── Individual checks: each returns (n_instances, [failure messages]) ────────

def check_capacity_and_payment(n=4, k=2, levels=3):
    cfg = Config(n=n, k=k, T=1, lambda_=1.0)
    histories = [None, [0, 1, 2, 3], [5, 0, 0, 2]]
    count, fails = 0, []
    for mname, make in _mech_factories(cfg).items():
        for prof in _profiles(n, levels):
            for cum in histories:
                for ts in TIE_SEEDS:
                    x, p = make().allocate(prof, _history(n, cum), ts)
                    count += 1
                    if int(x.sum()) != k or not set(np.unique(x)) <= {0, 1}:
                        fails.append(f"{mname} capacity profile={prof} x={x}")
                    if np.any(p[x == 0] != 0.0) or np.any(p < 0):
                        fails.append(f"{mname} payment profile={prof} p={p}")
    return count, fails


def check_report_invariance(n=4, k=2, levels=3):
    cfg = Config(n=n, k=k, T=1)
    profs = list(_profiles(n, levels))
    count, fails = 0, []
    for mname in ("M1", "M2"):
        make = _mech_factories(cfg)[mname]
        for ts in TIE_SEEDS:
            ref, _ = make().allocate(profs[0], _history(n), ts)
            for prof in profs[1:]:
                x, _ = make().allocate(prof, _history(n), ts)
                count += 1
                if not np.array_equal(x, ref):
                    fails.append(f"{mname} outcome depends on reports (tie_seed={ts})")
                    break
    return count, fails


def check_m3_equals_m4_at_lambda_zero(n=4, k=2, levels=3):
    cfg = Config(n=n, k=k, T=1, lambda_=0.0)
    count, fails = 0, []
    for prof in _profiles(n, levels):
        for cum in ([0, 0, 0, 0], [3, 0, 7, 1]):
            for ts in TIE_SEEDS:
                x3, _ = GreedyMechanism(cfg).allocate(prof, _history(n, cum), ts)
                x4, _ = ScoreMechanism(cfg).allocate(prof, _history(n, cum), ts)
                count += 1
                if not np.array_equal(x3, x4):
                    fails.append(f"M3!=M4 at lambda=0: profile={prof} cum={cum}")
    return count, fails


def check_roundrobin_wait_bound(max_n=9, rounds_factor=3):
    count, fails = 0, []
    for n in range(2, max_n + 1):
        for k in range(1, n):
            cfg = Config(n=n, k=k, T=rounds_factor * n)
            for init_seed in (0, 1):
                mech = RoundRobinMechanism(cfg, init_seed=init_seed)
                h = _history(n)
                bound = math.ceil(n / k) - 1
                for t in range(cfg.T):
                    x, p = mech.allocate(np.zeros(n), h, t)
                    h.update(x, p)
                    count += 1
                    if h.consecutive_wait.max() > bound:
                        fails.append(f"RR n={n} k={k}: wait {h.consecutive_wait.max()} > {bound}")
                if h.cumulative.max() - h.cumulative.min() > 1:
                    fails.append(f"RR n={n} k={k}: unequal service {h.cumulative}")
    return count, fails


def check_welfare_oracle(n=4, k=2, levels=5):
    count, fails = 0, []
    for lam in (None, 0.0):
        cfg = Config(n=n, k=k, T=1, lambda_=lam if lam is not None else 1.0)
        make = (lambda: GreedyMechanism(cfg)) if lam is None else (lambda: ScoreMechanism(cfg))
        for prof in _profiles(n, levels):
            for ts in TIE_SEEDS:
                x, _ = make().allocate(prof, _history(n), ts)
                W, W_star = float(prof @ x), float(np.sort(prof)[-k:].sum())
                count += 1
                if abs(W - W_star) > 1e-12:
                    fails.append(f"oracle lambda={lam} profile={prof}: W={W} W*={W_star}")
    return count, fails


def _utility(make, n, focal, value, report, others, cum, ts):
    reports = np.empty(n)
    reports[focal] = report
    reports[np.arange(n) != focal] = others
    x, p = make().allocate(reports, _history(n, cum), ts)
    return value * x[focal] - p[focal], x, p, reports


def _dsic_scan(make, n, k, levels, cum=None):
    """
    Enumerate focal value x focal report x opponent profile x tie seed (focal =
    user 0) and return (n_instances, [(value, report, others, ts, u_dev, u_truth)])
    for every profile where deviating strictly beats truthful reporting.
    """
    grid = _grid(levels)
    violations, count = [], 0
    for others in product(grid, repeat=n - 1):
        others = np.array(others)
        for ts in TIE_SEEDS:
            for v in grid:
                u_truth = _utility(make, n, 0, v, v, others, cum, ts)[0]
                for r in grid:
                    u_dev = _utility(make, n, 0, v, r, others, cum, ts)[0]
                    count += 1
                    if u_dev > u_truth + 1e-12:
                        violations.append((v, r, tuple(others), ts, u_dev, u_truth))
    return count, violations


def check_vickrey_dsic():
    count, fails = 0, []
    for n, k, levels in [(3, 1, 11), (4, 2, 5), (4, 3, 4)]:
        cfg = Config(n=n, k=k, T=1)
        c, viol = _dsic_scan(lambda: VickreyMechanism(cfg), n, k, levels)
        count += c
        fails += [f"M5 n={n} k={k}: profitable deviation {v}" for v in viol[:3]]

    # payment = (k+1)-st highest bid; individual rationality of truthful bids
    n, k = 4, 2
    cfg = Config(n=n, k=k, T=1)
    for prof in _profiles(n, 5):
        x, p = VickreyMechanism(cfg).allocate(prof, _history(n), 0)
        thr = np.sort(prof)[::-1][k]
        count += 1
        if np.any(np.abs(p[x == 1] - thr) > 1e-12):
            fails.append(f"M5 payment != (k+1)-st bid for {prof}")
        if np.any(prof * x - p < -1e-12):
            fails.append(f"M5 truthful utility negative for {prof}")
    return count, fails


def check_negative_controls():
    """The DSIC enumerator must expose M3 and M4 (inflation helps there)."""
    n, k, levels = 3, 1, 6
    cfg = Config(n=n, k=k, T=1, lambda_=1.0)
    fails = []
    for name, make in [("M3", lambda: GreedyMechanism(cfg)),
                       ("M4", lambda: ScoreMechanism(cfg))]:
        _, viol = _dsic_scan(make, n, k, levels, cum=[0, 0, 0])
        if not viol:
            fails.append(f"{name}: enumerator found no profitable inflation (it should)")
    return 2, fails


def check_determinism():
    count, fails = 0, []
    cfg = Config(n=8, k=3, T=40, rho=0.25, lambda_=1.0)
    for seed in (11, 22):
        for mname in ("M1", "M2", "M3", "M4", "M5"):
            runs = []
            for _ in range(2):
                pkg = SeedPackage.generate(seed, cfg)
                make = {
                    "M1": lambda: RandomMechanism(cfg),
                    "M2": lambda: RoundRobinMechanism(cfg, init_seed=int(pkg.tie_seeds[0])),
                    "M3": lambda: GreedyMechanism(cfg),
                    "M4": lambda: ScoreMechanism(cfg),
                    "M5": lambda: VickreyMechanism(cfg),
                }[mname]
                h = run_single(make(), pkg, cfg)
                runs.append((np.stack(h.allocations), np.stack(h.payments)))
            count += 1
            if not (np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])):
                fails.append(f"{mname} seed={seed} is not deterministic")
    return count, fails


def check_batch_consistency():
    count, fails = 0, []
    n, k = 6, 2
    rng = np.random.default_rng(0)
    for lam in (0.0, 1.0, 2.0):
        cfg = Config(n=n, k=k, T=1, lambda_=lam)
        for MechCls in (GreedyMechanism, ScoreMechanism, VickreyMechanism):
            mech = MechCls(cfg)
            for _ in range(50):
                reports = rng.random((1, n))
                cum = rng.integers(0, 20, size=(1, n))
                tb = rng.random((1, n))
                xb, pb = mech.allocate_batch(reports, cum, tb)
                # single-scenario rule with the same tie-break key
                scores = (reports / (1.0 + cum) ** lam) if MechCls is ScoreMechanism else reports
                order = np.lexsort((tb[0], scores[0]))
                x_ref = np.zeros(n, dtype=np.int64)
                x_ref[order[-k:]] = 1
                count += 1
                if not np.array_equal(xb[0], x_ref):
                    fails.append(f"{MechCls.__name__} batch != single (lambda={lam})")
                if MechCls is VickreyMechanism:
                    thr = np.sort(reports[0])[::-1][k]
                    if np.any(np.abs(pb[0][x_ref == 1] - thr) > 1e-12) or np.any(pb[0][x_ref == 0] != 0):
                        fails.append("M5 batch payment wrong")
    return count, fails


def _naive_waits(X: np.ndarray) -> np.ndarray:
    n, T = X.shape
    Q = np.zeros((n, T), dtype=np.int64)
    q = np.zeros(n, dtype=np.int64)
    for t in range(T):
        Q[:, t] = q                                   # state before round t
        q = np.where(X[:, t] == 1, 0, q + 1)
    return Q


def check_metric_crosscheck():
    count, fails = 0, []
    rng = np.random.default_rng(5)
    for n, k, T in [(5, 2, 30), (9, 3, 50), (12, 1, 40)]:
        cfg = Config(n=n, k=k, T=T)
        for _ in range(20):
            h = _history(n)
            for _t in range(T):
                x = np.zeros(n, dtype=np.int64)
                x[rng.choice(n, size=k, replace=False)] = 1
                h.update(x, np.zeros(n))
            X = np.stack(h.allocations, axis=1)
            Q = _naive_waits(X)
            count += 1
            if not np.array_equal(M.wait_matrix(X), Q):
                fails.append(f"wait_matrix mismatch n={n} k={k}")
            if M.max_wait(h) != int(Q.max()):
                fails.append("Q_max mismatch")
            if abs(M.starvation_rate(h, cfg) - float(np.mean(Q > cfg.delta))) > 1e-12:
                fails.append("SR mismatch")
            if abs(M.percentile_wait(h, 95.0) - float(np.percentile(Q, 95.0))) > 1e-12:
                fails.append("p95 mismatch")
    return count, fails


CHECKS = [
    ("Capacity & payment invariants (all profiles x histories x tie seeds)", check_capacity_and_payment),
    ("Report-invariance of M1, M2 (all profile pairs)",                      check_report_invariance),
    ("M3 == M4 at lambda = 0",                                               check_m3_equals_m4_at_lambda_zero),
    ("Round-robin wait bound, all 1 <= k < n <= 9",                          check_roundrobin_wait_bound),
    ("Welfare oracle: M3 / M4(lambda=0) attain W*",                           check_welfare_oracle),
    ("M5 per-round DSIC, payments, individual rationality",                  check_vickrey_dsic),
    ("Negative controls: M3/M4 are NOT DSIC (enumerator can fail)",          check_negative_controls),
    ("Determinism of every mechanism",                                       check_determinism),
    ("Batch allocation == single allocation (M3, M4, M5)",                   check_batch_consistency),
    ("Vectorised wait metrics == naive loop",                                check_metric_crosscheck),
]


def run_all_checks(verbose: bool = True) -> list:
    """Run every check; return a list of (name, n_instances, failures)."""
    results = []
    for name, fn in CHECKS:
        n_inst, fails = fn()
        results.append((name, n_inst, fails))
        if verbose:
            tag = f"{GREEN}PASS{RESET}" if not fails else f"{RED}FAIL{RESET}"
            print(f"  [{tag}] {name}  ({n_inst:,} instances)")
            for f in fails[:5]:
                print(f"         - {f}")
    return results


def main() -> None:
    print("=" * 70)
    print("E0 — Validation & Sanity Checks (exhaustive on small instances)")
    print("=" * 70)
    results = run_all_checks()
    bad = [name for name, _, fails in results if fails]
    print("=" * 70)
    if bad:
        print(f"RESULT: {len(bad)} check(s) FAILED.")
        sys.exit(1)
    total = sum(n for _, n, _ in results)
    print(f"RESULT: all {len(results)} E0 checks PASSED ({total:,} instances). Proceed to E1-E7.")


if __name__ == "__main__":
    main()
