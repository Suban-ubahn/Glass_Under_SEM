#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glass_sem_env

echo "=== Checking trajectory ==="
python -c "
import config, os
assert os.path.exists(config.TRAJECTORY_FILE)
print('Trajectory:', config.TRAJECTORY_FILE)
print('Using CUTOFF_DICT for graph construction')
print('Using DEFECT_K =', config.DEFECT_K)
"

echo ""
echo "=== Building graphs ==="
python graph/build.py

echo ""
echo "=== Defect analysis ==="
python graph/defects.py

echo ""
echo "=== Hole + defect projection maps ==="
python scripts/build_projection_maps.py

echo ""
echo "=== Hole + defect + D2min animations ==="
python scripts/build_holes_defects_animations.py

echo ""
echo "=== Graph pipeline complete ==="