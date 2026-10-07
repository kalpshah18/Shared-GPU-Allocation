#!/usr/bin/env bash
# run_all.sh
# ==========
# Single command to regenerate every main figure from locked master seeds.
# Run from the project root:
#   bash run_all.sh

set -euo pipefail

echo "================================================"
echo " FAI GPU Allocation — Full Experiment Pipeline"
echo "================================================"

# 0. Generate seeds (idempotent: fails safely if already generated)
if [ ! -f seeds/master_seeds.json ]; then
    echo "[Step 0] Generating master seeds..."
    python seeds/generate_seeds.py
else
    echo "[Step 0] Master seeds already exist — skipping generation."
fi

# 0. E0 Validation (must pass before any experiment runs)
echo ""
echo "[E0] Running validation and sanity checks..."
python experiments/e0_validation.py

# 1. E1 — Resource Scarcity
echo ""
echo "[E1] Resource scarcity sweep..."
python experiments/e1_scarcity.py

# 2. E2 — Fairness & Strategic-Vulnerability Frontier
echo ""
echo "[E2] Fairness-frontier sweep (lambda x mechanisms)..."
python experiments/e2_frontier.py

# 2b. E2b — Exaggeration-strength sensitivity
echo ""
echo "[E2b] Exaggeration-strength sensitivity..."
python experiments/e2b_cap_sensitivity.py

# 3. E3 — Strategic Population (factorial)
echo ""
echo "[E3] Strategic population factorial design..."
python experiments/e3_strategic.py

# 4. E4 — Heterogeneous Users
echo ""
echo "[E4] Heterogeneous user distributions..."
python experiments/e4_heterogeneous.py

# 5. E5 — Temporal Persistence (Stretch)
echo ""
echo "[E5] Temporal persistence sweep (alpha)..."
python experiments/e5_persistence.py

# 6. E6 — Computational Scalability (Stretch)
echo ""
echo "[E6] Computational scalability benchmark..."
python experiments/e6_scalability.py

# 7. E7 — Paired mechanism comparisons
echo ""
echo "[E7] Paired mechanism comparisons..."
python experiments/e7_paired.py

# 8. Generate all figures
echo ""
echo "[Figures] Generating all figures..."
python analysis/plots.py --all

echo ""
echo "================================================"
echo " All done. Figures saved to figures/"
echo "================================================"
