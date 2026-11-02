#!/bin/bash
# =============================================================
# run_flow_units.sh — flow-units + d2min comparison pipeline
#
# Runs analysis/d2min/compute.py: g(r)-derived, species-resolved
# cutoffs (via cutoffs.CUTOFF_ZRZR/ZRCU/CUCU) + non-affine
# displacement (D^2_min) between the first and last frame of
# the trajectory.
#
# Then runs the d2min-vs-defects and d2min-vs-holes comparisons
# (analysis/d2min/comparison.py) and their animations
# (visualization/animate.py).
#
# Run from the project root:
#   bash run_flow_units.sh
# =============================================================

set -e  # stop on first error

echo "=== Activating environment ==="
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate glass_sem_env

echo "=== Checking trajectory file ==="
python -c "
import config, os
assert os.path.exists(config.TRAJECTORY_FILE), f'Trajectory file not found: {config.TRAJECTORY_FILE}'
print('Trajectory file OK:', config.TRAJECTORY_FILE)
print('CUTOFF_ZRZR (g(r)-derived):', cutoffs.CUTOFF_ZRZR, 'A')
print('CUTOFF_ZRCU (g(r)-derived):', cutoffs.CUTOFF_ZRCU, 'A')
print('CUTOFF_CUCU (g(r)-derived):', cutoffs.CUTOFF_CUCU, 'A')
"

echo ""
echo "=== Dynamics: pair correlation + D^2_min ==="
python analysis/d2min/compute.py

echo ""
echo "=== Flow-unit / D2min animations ==="
python scripts/build_d2min_animations.py

echo ""
echo "=== D2min vs defects: overlay animation + distance stats ==="
python scripts/run_d2min_defect_comparison.py

echo ""
echo "=== D2min vs holes: overlay animation + distance stats ==="
python scripts/run_d2min_hole_comparison.py

echo ""
echo "=== Done ==="