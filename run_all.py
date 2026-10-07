"""
run_all.py
==========
Cross-platform single entry point: validate, run every experiment from the
locked master seeds, and regenerate every figure.

Usage
-----
    python run_all.py                    # full pipeline (30 seeds, n=50, T=1000)
    python run_all.py --quick            # tiny smoke run (2 seeds, small n and T)
    python run_all.py --results-dir R --figures-dir F

Run it with the project's virtual environment (``.venv``) so that every step
uses the pinned dependencies; steps are launched with the same interpreter.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent

# (label, script, extra arguments in quick mode)
STEPS = [
    ("E1  — Resource scarcity",                      "experiments/e1_scarcity.py",      ["--n", "20", "--T", "80"]),
    ("E2  — Fairness / strategy frontier",           "experiments/e2_frontier.py",      ["--n", "20", "--T", "80"]),
    ("E3  — Strategic factorial",                    "experiments/e3_strategic.py",     ["--n", "20", "--T", "80"]),
    ("E3b — Rollout attack (n=10)",                  "experiments/e3b_rollout.py",      ["--n", "10", "--T", "30"]),
    ("E4  — Heterogeneous users",                    "experiments/e4_heterogeneous.py", ["--n", "20", "--T", "80"]),
    ("E5  — Temporal persistence (stretch)",         "experiments/e5_persistence.py",   ["--n", "20", "--T", "80"]),
    ("E6  — Scalability benchmark (stretch)",        "experiments/e6_scalability.py",   ["--T", "50"]),
    ("E7  — Sensitivity to the exaggeration factor", "experiments/e7_sensitivity.py",   ["--n", "20", "--T", "80"]),
]


def run_step(label: str, cmd: list) -> None:
    print("\n" + "=" * 70)
    print(f"[{label}] {' '.join(cmd)}")
    print("=" * 70, flush=True)
    t0 = time.perf_counter()
    subprocess.run([sys.executable] + cmd, check=True, cwd=ROOT)
    print(f"[{label}] done in {time.perf_counter() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (not for reporting)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    args = ap.parse_args()

    print("=" * 70)
    print(" FAI GPU Allocation — Full Experiment Pipeline" + ("  [QUICK]" if args.quick else ""))
    print("=" * 70)

    seed_file = ROOT / "seeds" / "master_seeds.json"
    if not seed_file.exists():
        run_step("Step 0 — Master seeds", ["seeds/generate_seeds.py"])
    else:
        print("\n[Step 0] Master seeds already exist — skipping generation.")

    run_step("E0  — Validation (must pass)", ["experiments/e0_validation.py"])

    common = ["--results-dir", args.results_dir]
    if args.quick:
        common += ["--seeds", "2", "--bootstrap", "200"]
    for label, script, quick_args in STEPS:
        run_step(label, [script] + common + (quick_args if args.quick else []))

    run_step("Figures", ["analysis/plots.py", "--results-dir", args.results_dir,
                         "--figures-dir", args.figures_dir])

    print("\n" + "=" * 70)
    print(f" Pipeline complete. Results in {args.results_dir}/, figures in {args.figures_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
