#!/bin/bash
# =============================================================
# run_flow_units.sh — flow-units + d2min comparison pipeline
#
# Runs dynamics/flow_units.py: g(r)-derived, species-resolved
# cutoffs (via config.CUTOFF_ZRZR/ZRCU/CUCU) + non-affine
# displacement (D^2_min) between the first and last frame of
# the trajectory.
#
# Then runs the d2min-vs-defects and d2min-vs-holes comparisons
# (analysis/d2min_comparison.py) and their animations
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
print('CUTOFF_ZRZR (g(r)-derived):', config.CUTOFF_ZRZR, 'A')
print('CUTOFF_ZRCU (g(r)-derived):', config.CUTOFF_ZRCU, 'A')
print('CUTOFF_CUCU (g(r)-derived):', config.CUTOFF_CUCU, 'A')
"

echo ""
echo "=== Dynamics: pair correlation + D^2_min ==="
python dynamics/flow_units.py

echo ""
echo "=== Flow-unit / D2min animations ==="
python scripts/build_d2min_animations.py

echo ""
echo "=== D2min vs defects: overlay animation + distance stats ==="
python -c "
from data_io.trajectory import load_trajectory
from config import TRAJECTORY_FILE

from visualization.animate import animate_d2min_defects_overlay_3d
from analysis.d2min_comparison import compute_d2min_defect_stats
from visualization.plot import plot_d2min_defect_comparison

trajectory = load_trajectory(TRAJECTORY_FILE)

stats = compute_d2min_defect_stats(trajectory)
plot_d2min_defect_comparison(stats)

animate_d2min_defects_overlay_3d(trajectory)
"

echo ""
echo "=== D2min vs holes: overlay animation + distance stats ==="
python -c "
from data_io.trajectory import load_trajectory
from config import TRAJECTORY_FILE

from visualization.animate import animate_d2min_holes_overlay_3d
from analysis.d2min_comparison import compute_d2min_hole_distance_stats
from visualization.plot import plot_d2min_hole_distance

trajectory = load_trajectory(TRAJECTORY_FILE)

animate_d2min_holes_overlay_3d(trajectory)

stats = compute_d2min_hole_distance_stats(trajectory)
plot_d2min_hole_distance(stats)
"

echo ""
echo "=== Done ==="