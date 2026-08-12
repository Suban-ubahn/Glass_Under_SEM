#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
cd "$(dirname "$0")"

echo "== Step 1: zarr -> h5 setup (glass_sem_env) =="
conda activate glass_sem_env
python diffraction/reconstruct_pypty_setup.py

echo "== Step 2: iterative reconstruction (pypty env) =="
conda activate pypty
python diffraction/reconstruct_pypty.py

echo "== Pipeline complete =="
