#!/usr/bin/env bash
# Single command to validate, run all experiments and regenerate every figure.
# Run from the project root with the virtual environment active:
#   bash run_all.sh [--quick] [--results-dir DIR] [--figures-dir DIR]
set -euo pipefail
python run_all.py "$@"
