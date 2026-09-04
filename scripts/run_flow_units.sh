#!/bin/bash
# =============================================================
# run_flow_units.sh — flow-units pipeline only
#
# Runs dynamics/flow_units.py: g(r)-derived, species-resolved
# cutoffs (via config.CUTOFF_ZRZR/ZRCU/CUCU) + non-affine
# displacement (D^2_min) between the first and last frame of
# the trajectory.
#
# Run from the project root:
#   bash run_flow_units.sh
# =============================================================

set -e  # stop on first error

echo "=== Activating environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glass_sem_env

echo "=== Checking trajectory file ==="
python -c "
import config, os
assert os.path.exists(config.TRAJECTORY_FILE), f'Trajectory file not found: {config.TRAJECTORY_FILE}'
print('Trajectory file OK:', config.TRAJECTORY_FILE)
print('CUTOFF_ZRZR (g(r)-derived):', config.CUTOFF_ZRZR, 'A')
print('CUTOFF_ZRCU (g(r)-derived):', config.CUTOFF_ZRCU, 'A')
print('CUTOFF_CUCU (g(r)-derived):', config.CUTOFF_CUCU, 'A')
"

echo ""
echo "=== Dynamics: pair correlation + D^2_min ==="
python dynamics/flow_units.py

echo ""
echo "=== Done ==="