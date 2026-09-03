#!/bin/bash
# =============================================================
# run_pipeline.sh — full Glass_Under_SEM pipeline
#
# Stages, in order:
#   1. Dynamics    — g(r)-derived cutoff + D^2_min non-affine map
#                    (dynamics/flow_units.py)
#   2. Projection  — hole/defect projection maps + cross-correlation
#                    (scripts/build_projection_maps.py)
#   3. Diffraction — 4D-STEM simulation (heavy; see README's WSL2
#                    RAM note). Skip with --skip-diffraction.
#   4. Animations  — hole + defect trajectory animations (slow over
#                    all 101 frames). Skip with --skip-animations.
#
# Run from the project root:
#   bash run_pipeline.sh
#   bash run_pipeline.sh --skip-diffraction --skip-animations
# =============================================================

set -e  # stop on first error

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------
SKIP_DIFFRACTION=false
SKIP_ANIMATIONS=false

for arg in "$@"; do
    case "$arg" in
        --skip-diffraction) SKIP_DIFFRACTION=true ;;
        --skip-animations)  SKIP_ANIMATIONS=true ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash run_pipeline.sh [--skip-diffraction] [--skip-animations]"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
echo "=== Activating environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glass_sem_env

echo "=== Checking trajectory file ==="
python -c "
import config, os
assert os.path.exists(config.TRAJECTORY_FILE), f'Trajectory file not found: {config.TRAJECTORY_FILE}'
print('Trajectory file OK:', config.TRAJECTORY_FILE)
"

# ------------------------------------------------------------
# Stage 1 — dynamics: g(r) cutoff + D^2_min non-affine map
# ------------------------------------------------------------
echo ""
echo "=== [1/4] Dynamics: pair correlation + D^2_min ==="
python dynamics/flow_units.py

# ------------------------------------------------------------
# Stage 2 — hole + defect projection maps, cross-correlation
# ------------------------------------------------------------
echo ""
echo "=== [2/4] Hole + defect projection maps ==="
python scripts/build_projection_maps.py

# ------------------------------------------------------------
# Stage 3 — 4D-STEM diffraction simulation (heavy)
# ------------------------------------------------------------
if [ "$SKIP_DIFFRACTION" = true ]; then
    echo ""
    echo "=== [3/4] Skipping 4D-STEM diffraction simulation (--skip-diffraction) ==="
else
    echo ""
    echo "=== [3/4] 4D-STEM diffraction simulation ==="
    echo "(Memory-heavy — see README's WSL2 .wslconfig note if running locally.)"
    python diffraction/simulate.py
fi

# ------------------------------------------------------------
# Stage 4 — animations
# ------------------------------------------------------------
if [ "$SKIP_ANIMATIONS" = true ]; then
    echo ""
    echo "=== [4/4] Skipping animations (--skip-animations) ==="
else
    echo ""
    echo "=== [4/4] Animations ==="
    python scripts/build_animations.py
fi

echo ""
echo "=== Pipeline complete ==="