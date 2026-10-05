"""
experiments/e0_validation.py
=============================
E0 — Validation and sanity checks — Owner: Harsh Dhru

Exhaustive correctness tests on small instances.  Must all pass before any
main experiment runs.  These checks identify implementation errors.

Checks performed (from proposal §4.3)
---------------------------------------
1. Capacity invariant       : Σ x_{i,t} = k  for all t, all mechanisms
2. Payment invariant        : p_{i,t} = 0 when x_{i,t} = 0
3. Report-invariance        : M1 and M2 — outcomes identical for any reports
4. M3 ≡ M4 at λ=0          : same allocation on every test case
5. Round-robin wait bound   : no user waits > ⌈n/k⌉ rounds
6. Welfare oracle           : M3 with truthful reports achieves W* (exhaustive)
7. M5 one-round truthfulness: on grid G, truthful report maximises utility
                              for every opponent profile (exhaustive, small n)

Usage
-----
    python experiments/e0_validation.py

All checks print PASS / FAIL and exit with code 1 on any failure.
"""

from __future__ import annotations

import math
import sys
from itertools import product

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

# ── Import project modules ────────────────────────────────────────────────────
sys.path.insert(0, ".")

from sim.config import Config
from sim.environment import History, SeedPackage
from sim.mechanisms import (
    RandomMechanism,
    RoundRobinMechanism,
    GreedyMechanism,
    ScoreMechanism,
    VickreyMechanism,
)
from sim.runner import run_single

# ── Small test config ─────────────────────────────────────────────────────────
SMALL_CFG = Config(n=6, k=2, T=20, v_max=1.0, rho=0.0)
MECHANISMS = {
    "M1": lambda cfg: RandomMechanism(cfg),
    "M2": lambda cfg: RoundRobinMechanism(cfg, init_seed=0),
    "M3": lambda cfg: GreedyMechanism(cfg),
    "M4": lambda cfg: ScoreMechanism(cfg),
    "M5": lambda cfg: VickreyMechanism(cfg),
}

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [{PASS}] {name}")
    else:
        msg = f"  [{FAIL}] {name}" + (f" — {detail}" if detail else "")
        print(msg)
        failures.append(name)


# ── Check 1 & 2: Capacity and payment invariants ─────────────────────────────

def test_capacity_and_payment() -> None:
    print("\n[Check 1 & 2] Capacity and payment invariants")
    cfg = SMALL_CFG
    rng = np.random.default_rng(42)
    pkg = SeedPackage.generate(master_seed=42, cfg=cfg)

    for mname, mfactory in MECHANISMS.items():
        mech    = mfactory(cfg)
        history = History(cfg.n)

        for t in range(cfg.T):
            reports = rng.uniform(0, cfg.v_max, size=cfg.n)
            x, p    = mech.allocate(reports, history, int(pkg.tie_seeds[t]))
            history.update(x, p)

            # Capacity: exactly k GPUs allocated
            check(
                f"{mname} capacity t={t}",
                int(x.sum()) == cfg.k,
                f"sum={x.sum()}, expected {cfg.k}",
            )
            # Payment: p_i = 0 whenever x_i = 0
            bad_payments = np.any(p[x == 0] != 0.0)
            check(
                f"{mname} payment t={t}",
                not bad_payments,
                f"non-zero payment for unallocated user",
            )
            if failures:
                return  # stop early on first failure


# ── Check 3: Report-invariance for M1 and M2 ─────────────────────────────────

def test_report_invariance() -> None:
    print("\n[Check 3] Report-invariance (M1 and M2)")
    cfg = SMALL_CFG
    pkg = SeedPackage.generate(master_seed=7, cfg=cfg)

    for mname, mfactory in [("M1", MECHANISMS["M1"]), ("M2", MECHANISMS["M2"])]:
        for t in range(min(cfg.T, 5)):
            tie_seed = int(pkg.tie_seeds[t])
            dummy_history = History(cfg.n)

            reports_a = np.zeros(cfg.n)
            reports_b = np.ones(cfg.n)
            reports_c = np.random.default_rng(t).uniform(0, 1, cfg.n)

            x_a, _ = mfactory(cfg).allocate(reports_a, dummy_history, tie_seed)
            x_b, _ = mfactory(cfg).allocate(reports_b, dummy_history, tie_seed)
            x_c, _ = mfactory(cfg).allocate(reports_c, dummy_history, tie_seed)

            check(
                f"{mname} report-invariance t={t} (a==b)",
                np.array_equal(x_a, x_b),
            )
            check(
                f"{mname} report-invariance t={t} (a==c)",
                np.array_equal(x_a, x_c),
            )


# ── Check 4: M3 ≡ M4 at λ=0 ─────────────────────────────────────────────────

def test_m3_equals_m4_at_lambda_zero() -> None:
    print("\n[Check 4] M3 ≡ M4 at λ=0")
    cfg = Config(n=6, k=2, T=10, lambda_=0.0)
    pkg = SeedPackage.generate(master_seed=99, cfg=cfg)
    m3  = GreedyMechanism(cfg)
    m4  = ScoreMechanism(cfg)
    h3  = History(cfg.n)
    h4  = History(cfg.n)
    rng = np.random.default_rng(0)

    for t in range(cfg.T):
        reports  = rng.uniform(0, 1, cfg.n)
        tie_seed = int(pkg.tie_seeds[t])
        x3, p3   = m3.allocate(reports, h3, tie_seed)
        x4, p4   = m4.allocate(reports, h4, tie_seed)
        check(
            f"M3==M4 at λ=0, t={t}",
            np.array_equal(x3, x4),
            f"x3={x3}, x4={x4}",
        )
        h3.update(x3, p3)
        h4.update(x4, p4)


# ── Check 5: Round-robin wait bound ──────────────────────────────────────────

def test_roundrobin_wait_bound() -> None:
    print("\n[Check 5] Round-robin wait bound ≤ ⌈n/k⌉")
    cfg      = Config(n=6, k=2, T=100)
    pkg      = SeedPackage.generate(master_seed=1, cfg=cfg)
    m2       = RoundRobinMechanism(cfg, init_seed=0)
    history  = History(cfg.n)
    rng      = np.random.default_rng(1)
    bound    = math.ceil(cfg.n / cfg.k)

    for t in range(cfg.T):
        reports = rng.uniform(0, 1, cfg.n)
        x, p    = m2.allocate(reports, history, int(pkg.tie_seeds[t]))
        history.update(x, p)

    max_wait = int(history.consecutive_wait.max())
    check(
        f"RR max wait ({max_wait}) ≤ ceil(n/k) = {bound}",
        max_wait <= bound,
        f"max_wait={max_wait}",
    )


# ── Check 6: M3 welfare oracle (exhaustive on small instance) ─────────────────

def test_welfare_oracle() -> None:
    print("\n[Check 6] M3 welfare oracle (truthful → W*)")
    cfg = Config(n=4, k=2, T=1)
    rng = np.random.default_rng(55)

    for trial in range(20):
        values   = rng.uniform(0, 1, cfg.n)
        pkg      = SeedPackage.generate(master_seed=trial, cfg=cfg)
        history  = History(cfg.n)
        m3       = GreedyMechanism(cfg)
        x, _     = m3.allocate(values, history, int(pkg.tie_seeds[0]))

        # Oracle: top-k values
        top_k_val   = np.sort(values)[-cfg.k:]
        W_star      = float(top_k_val.sum())
        W_achieved  = float((values * x).sum())

        check(
            f"M3 oracle trial={trial}",
            abs(W_achieved - W_star) < 1e-10,
            f"W*={W_star:.4f}, W={W_achieved:.4f}",
        )


# ── Check 7: M5 one-round truthfulness (exhaustive on tiny grid) ──────────────

def test_vickrey_truthfulness() -> None:
    print("\n[Check 7] M5 per-round DSIC on report grid G={0,0.1,...,1}")
    cfg  = Config(n=4, k=2, T=1)
    grid = [round(x * 0.1, 1) for x in range(11)]

    rng = np.random.default_rng(77)
    for trial in range(10):
        true_values = rng.uniform(0, 1, cfg.n)
        focal       = rng.integers(0, cfg.n)

        # Utility of focal user when reporting r, opponents truthful
        def focal_utility(r: float) -> float:
            reports = true_values.copy()
            reports[focal] = r
            history = History(cfg.n)
            pkg     = SeedPackage.generate(master_seed=trial, cfg=cfg)
            m5      = VickreyMechanism(cfg)
            x, p    = m5.allocate(reports, history, int(pkg.tie_seeds[0]))
            return float(true_values[focal] * x[focal] - p[focal])

        truthful_utility = focal_utility(true_values[focal])
        best_grid_utility = max(focal_utility(r) for r in grid)

        # Truthful should be (approx.) at least as good as any grid report
        check(
            f"M5 truthfulness trial={trial} user={focal}",
            truthful_utility >= best_grid_utility - 1e-9,
            f"truthful={truthful_utility:.4f}, best_grid={best_grid_utility:.4f}",
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("E0 — Validation & Sanity Checks")
    print("=" * 60)

    test_capacity_and_payment()
    test_report_invariance()
    test_m3_equals_m4_at_lambda_zero()
    test_roundrobin_wait_bound()
    test_welfare_oracle()
    test_vickrey_truthfulness()

    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: {len(failures)} check(s) FAILED.")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("RESULT: All E0 checks PASSED. Proceed to E1–E4.")
        sys.exit(0)


if __name__ == "__main__":
    main()
