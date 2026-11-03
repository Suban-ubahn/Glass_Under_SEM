#!/bin/bash
#SBATCH --job-name=glass_pypty
#SBATCH --output=slurm_pypty_%j.out
#SBATCH --error=slurm_pypty_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --requeue

set -e

MIN_GPU_MEM_MB=4000   # minimum GPU memory this job needs
source "$SLURM_SUBMIT_DIR/slurm/_gpu_guard.sh"   # submit from the project root

source "$HOME/miniconda3/etc/profile.d/conda.sh"

cd "$SLURM_SUBMIT_DIR"

echo "== Step 1: zarr -> h5 setup (glass_sem_env) =="
conda activate glass_sem_env
python scripts/pypty_setup.py

echo "== Step 2: iterative reconstruction (pypty env) =="
conda activate pypty
python scripts/pypty_reconstruct.py

echo "== Pipeline complete =="
