#!/bin/bash
# =============================================================
# setup_env.sh — create and configure the glass_sem_env conda
# environment on a new machine.
#
# Run from the project root:
#   bash setup_env.sh
# =============================================================

set -e  # exit on first error

ENV_NAME="glass_sem_env"
PYTHON_VERSION="3.11"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

echo "=== Activating environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Installing project as editable package ==="
pip install -e .

echo ""
echo "=== Done. Activate with: conda activate $ENV_NAME ==="