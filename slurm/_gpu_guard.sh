# =============================================================
# slurm/_gpu_guard.sh -- sourced by the SLURM batch scripts.
# If the node has no usable GPU, exclude it and requeue the job
# (the cluster does not track GPUs as SLURM GRES, so we can't pin one).
# Expects MIN_GPU_MEM_MB to be set by the caller. `exit 0` here ends the
# *calling* batch script on purpose (the requeued copy runs instead).
# =============================================================

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
