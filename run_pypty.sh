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

MIN_GPU_MEM_MB=4000

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

cd "$SLURM_SUBMIT_DIR"

echo "== Step 1: zarr -> h5 setup (glass_sem_env) =="
conda activate glass_sem_env
python diffraction/reconstruct_pypty_setup.py

echo "== Step 2: iterative reconstruction (pypty env) =="
conda activate pypty
python diffraction/reconstruct_pypty.py

echo "== Pipeline complete =="
