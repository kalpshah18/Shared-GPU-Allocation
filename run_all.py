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
    ("E3c — Rollout horizon study (n=10)",          "experiments/e3c_rollout_horizon.py", ["--n", "10", "--T", "25"]),
    ("E4  — Heterogeneous users",                    "experiments/e4_heterogeneous.py", ["--n", "20", "--T", "80"]),
    ("E5  — Temporal persistence (stretch)",         "experiments/e5_persistence.py",   ["--n", "20", "--T", "80"]),
    ("E6  — Scalability benchmark (stretch)",        "experiments/e6_scalability.py",   ["--T", "50"]),
    ("E7  — λ × c sensitivity grid",                 "experiments/e7_sensitivity.py",   ["--n", "20", "--T", "80"]),
    ("E8  — Timing attack (hypothesis H2)",          "experiments/e8_timing.py",        ["--n", "20", "--T", "80"]),
    ("E9  — Proposed mechanisms, truthful",          "experiments/e9_proposed_truthful.py",   ["--n", "20", "--T", "80"]),
    ("E10 — Proposed mechanisms, strategic",         "experiments/e10_proposed_strategic.py", ["--n", "20", "--T", "80"]),
    ("E11 — Proposed mechanisms vs rollout attacker", "experiments/e11_proposed_rollout.py",  ["--n", "10", "--T", "25"]),
]
EXTENSION_STEPS = [
    ("E12 — Component ablations", "experiments/e12_ablations.py", ["--n", "10", "--T", "30"]),
    ("E13 — Adaptive policy populations", "experiments/e13_adaptive.py", ["--n", "3", "--T", "12", "--sweeps", "1"]),
    ("E14 — Constrained welfare oracle", "experiments/e14_constrained.py", ["--n", "3", "--T", "10"]),
    ("E15 — Corrected attack budgets", "experiments/e15_attack_budget.py", ["--n", "4", "--T", "8"]),
]
FOLLOWUP_STEPS = [
    ("E16 — Rank history case study", "experiments/e16_rank_history.py", ["--n", "4", "--T", "8"]),
    ("E17 — Karma accounting", "experiments/e17_karma_accounting.py", ["--n", "4", "--T", "12"]),
    ("E18 — Longer adaptive training", "experiments/e18_extended_adaptation.py", ["--n", "3", "--T", "12"]),
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
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--extensions-only", action="store_true", help="run E12-E15 on their fresh holdout, plus figures and tables")
    mode.add_argument("--followups-only", action="store_true", help="run E16-E18, figures and failure case studies")
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
    extension_seed_file = ROOT / "seeds" / "extension_seeds.json"
    if not extension_seed_file.exists():
        run_step("Fresh extension holdout", ["seeds/generate_extension_seeds.py"])
    if not (ROOT / "seeds" / "followup_seeds.json").exists() and not args.extensions_only:
        run_step("Independent follow-up holdout", ["seeds/generate_followup_seeds.py"])
    steps = (FOLLOWUP_STEPS if args.followups_only else
             EXTENSION_STEPS if args.extensions_only else STEPS + EXTENSION_STEPS + FOLLOWUP_STEPS)
    for label, script, quick_args in steps:
        run_step(label, [script] + common + (quick_args if args.quick else []))
    if not args.extensions_only:
        run_step("Follow-up data audit", ["analysis/verify_followups.py", "--results-dir", args.results_dir])

    run_step("Figures", ["analysis/plots.py", "--results-dir", args.results_dir,
                         "--figures-dir", args.figures_dir])
    run_step("Tables", ["analysis/report.py", "--results-dir", args.results_dir])
    if not args.followups_only:
        manuscript_out = str(Path(args.results_dir) / "COURSE_REPORT.md") if args.quick or args.results_dir != "results" else "docs/COURSE_REPORT.md"
        followup_name = "FOLLOWUP_REPORT.md" if args.quick or args.results_dir != "results" else "FAILURE_CASE_STUDIES.md"
        run_step("Course report", ["analysis/course_manuscript.py", "--results-dir", args.results_dir,
                                  "--out", manuscript_out, "--followup-report", followup_name])
    if not args.extensions_only:
        followup_out = str(Path(args.results_dir) / "FOLLOWUP_REPORT.md") if args.quick or args.results_dir != "results" else "docs/FAILURE_CASE_STUDIES.md"
        run_step("Failure case studies", ["analysis/failure_report.py", "--results-dir", args.results_dir,
                                         "--figures-dir", args.figures_dir, "--out", followup_out])
        if args.followups_only and args.results_dir == "results" and not args.quick:
            run_step("Refresh course report", ["analysis/course_manuscript.py"])

    print("\n" + "=" * 70)
    print(f" Pipeline complete. Results in {args.results_dir}/, figures in {args.figures_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
