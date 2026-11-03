#!/bin/bash
#SBATCH --job-name=glass_diffraction
#SBATCH --output=slurm_simulate_%j.out
#SBATCH --error=slurm_simulate_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --requeue

set -e

MIN_GPU_MEM_MB=4000   # minimum GPU memory this job needs
source "$SLURM_SUBMIT_DIR/slurm/_gpu_guard.sh"   # submit from the project root

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate glass_sem_env

cd "$SLURM_SUBMIT_DIR"

python -c "
import config
import os
assert os.path.exists(config.TRAJECTORY_FILE), f'Trajectory file not found: {config.TRAJECTORY_FILE}'
print('Trajectory file OK:', config.TRAJECTORY_FILE)
"

echo "== Running full 4D-STEM simulation =="
python scripts/simulate_4dstem.py

echo "== Verifying output =="
python -c "
import os
assert os.path.exists('outputs/simulate_4dstem/4d_stem.zarr'), 'Expected output outputs/simulate_4dstem/4d_stem.zarr not found'
print('4d_stem.zarr created successfully')
"
