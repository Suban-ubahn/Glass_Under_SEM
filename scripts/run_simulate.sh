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

MIN_GPU_MEM_MB=4000   # adjust to the real minimum this pipeline needs

echo "== GPU check on $SLURMD_NODENAME =="
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

if [[ -z "$gpu_mem" ]] || (( gpu_mem < MIN_GPU_MEM_MB )); then
    echo "Node $SLURMD_NODENAME: no usable GPU (found: '${gpu_mem:-none}' MB, need >= ${MIN_GPU_MEM_MB} MB)."
    echo "Excluding this node and requeuing job."
    current_excl=$(scontrol show job "$SLURM_JOB_ID" -o | grep -oP 'ExcNodeList=\K\S+')
    if [[ -n "$current_excl" && "$current_excl" != "(null)" ]]; then
        new_excl="${current_excl},${SLURMD_NODENAME}"
    else
        new_excl="${SLURMD_NODENAME}"
    fi
    scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="$new_excl"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

echo "Node $SLURMD_NODENAME: GPU OK (${gpu_mem} MB)"

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
python diffraction/simulate.py

echo "== Verifying output =="
python -c "
import os
assert os.path.exists('outputs/diffraction/4d_stem.zarr'), 'Expected output outputs/diffraction/4d_stem.zarr not found'
print('4d_stem.zarr created successfully')
"
