"""
run_all.py
==========
Cross-platform single entry point to execute the full experimental pipeline
and regenerate all figures from locked master seeds.

Usage
-----
    python run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_step(step_name: str, cmd: list[str]) -> None:
    print("\n" + "=" * 60)
    print(f"[{step_name}] Running: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run([sys.executable] + cmd, check=True)
    if result.returncode != 0:
        print(f"[ERROR] Step {step_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    print("=" * 60)
    print(" FAI GPU Allocation — Full Experiment Pipeline")
    print("=" * 60)

    # 0. Generate seeds if not present
    seed_file = Path("seeds/master_seeds.json")
    if not seed_file.exists():
        run_step("Step 0 — Master Seeds", ["seeds/generate_seeds.py"])
    else:
        print("\n[Step 0] Master seeds already exist — skipping generation.")

    # 1. E0 Validation
    run_step("E0 — Validation Checks", ["experiments/e0_validation.py"])

    # 2. E1 Scarcity
    run_step("E1 — Resource Scarcity Sweep", ["experiments/e1_scarcity.py"])

    # 3. E2 Frontier
    run_step("E2 — Fairness-Strategy Frontier", ["experiments/e2_frontier.py"])

    # 4. E3 Strategic
    run_step("E3 — Strategic Factorial Design", ["experiments/e3_strategic.py"])

    # 5. E4 Heterogeneous
    run_step("E4 — Heterogeneous Groups", ["experiments/e4_heterogeneous.py"])

    # 6. E5 Persistence
    run_step("E5 — Temporal Persistence (Stretch)", ["experiments/e5_persistence.py"])

    # 7. E6 Scalability
    run_step("E6 — Scalability Benchmark (Stretch)", ["experiments/e6_scalability.py"])

    # 8. Figures
    run_step("Figures — Regenerating All Figures", ["analysis/plots.py", "--all"])

    print("\n" + "=" * 60)
    print(" Pipeline complete! All figures saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
