#!/bin/bash
set -e  # stop on first error

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate glass_sem_env

cd "$(dirname "$0")"

echo "== Pre-flight checks =="
python -c "
import config
import os
assert os.path.exists(config.TRAJECTORY_FILE), f'Trajectory file not found: {config.TRAJECTORY_FILE}'
print('Trajectory file OK:', config.TRAJECTORY_FILE)
print('CONVERGENCE_SEMIANGLE_MRAD =', config.CONVERGENCE_SEMIANGLE_MRAD)
print('BEAM_ENERGY_KV =', config.BEAM_ENERGY_KV)
"

echo "== Running full 4D-STEM simulation =="
python diffraction/simulate.py

echo "== Done. Verifying output =="
python -c "
import os
assert os.path.exists('diffraction/4d_stem.zarr'), 'Expected output diffraction/4d_stem.zarr not found'
print('4d_stem.zarr created successfully')
"
