#!/bin/bash
# =============================================================
# scripts/run_pipeline.sh -- the whole analysis pipeline, one place.
# (Replaces the old run_pipeline.sh / run_graph_pipeline.sh / run_flow_units.sh.)
#
# Stages, in order:
#   1. dynamics     g(r) cutoffs + D^2_min map            (scripts/run_d2min.py)
#   2. defects      graph defect statistics               (scripts/run_defect_stats.py)
#   3. projection   hole/defect maps + cross-correlation  (scripts/build_projection_maps.py)
#   4. comparison   D^2_min vs defects / vs holes         (scripts/run_d2min_*_comparison.py)
#   5. animations   slow over all frames; --skip-animations
#   6. diffraction  4D-STEM simulation, heavy; --skip-diffraction
#                   (on the cluster use: sbatch slurm/run_simulate.sh)
#
# Run from anywhere:   bash scripts/run_pipeline.sh [--skip-diffraction] [--skip-animations]
# =============================================================
set -e
cd "$(dirname "$0")/.."          # always run from the project root

SKIP_DIFFRACTION=false
SKIP_ANIMATIONS=false
for arg in "$@"; do
    case "$arg" in
        --skip-diffraction) SKIP_DIFFRACTION=true ;;
        --skip-animations)  SKIP_ANIMATIONS=true ;;
        *) echo "Unknown option: $arg"
           echo "Usage: bash scripts/run_pipeline.sh [--skip-diffraction] [--skip-animations]"
           exit 1 ;;
    esac
done

if [ "${CONDA_DEFAULT_ENV:-}" != "glass_sem_env" ]; then
    echo "=== Activating glass_sem_env ==="
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate glass_sem_env
fi

echo "=== Checking trajectory file ==="
python -c "
import config, os
assert os.path.exists(config.TRAJECTORY_FILE), f'Trajectory file not found: {config.TRAJECTORY_FILE}'
print('Trajectory file OK:', config.TRAJECTORY_FILE)
"

echo; echo "=== [1/6] Dynamics: pair correlation + D^2_min ===";  python scripts/run_d2min.py
echo; echo "=== [2/6] Defect statistics ===";                      python scripts/run_defect_stats.py
echo; echo "=== [3/6] Hole + defect projection maps ===";          python scripts/build_projection_maps.py
echo; echo "=== [4/6] D^2_min vs defects / holes ==="
python scripts/run_d2min_defect_comparison.py
python scripts/run_d2min_hole_comparison.py

if [ "$SKIP_ANIMATIONS" = true ]; then
    echo; echo "=== [5/6] Skipping animations (--skip-animations) ==="
else
    echo; echo "=== [5/6] Animations ==="
    python scripts/build_holes_defects_animations.py
    python scripts/build_d2min_animations.py
fi

if [ "$SKIP_DIFFRACTION" = true ]; then
    echo; echo "=== [6/6] Skipping 4D-STEM simulation (--skip-diffraction) ==="
else
    echo; echo "=== [6/6] 4D-STEM diffraction simulation (memory-heavy; see README WSL2 note) ==="
    python scripts/simulate_4dstem.py
fi

echo; echo "=== Pipeline complete ==="
