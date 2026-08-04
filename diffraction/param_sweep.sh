#!/bin/bash
# =============================================================
# diffraction/param_sweep.sh
#
# Sweeps over 4D-STEM simulation parameters to find optimal
# settings for ptychography.
#
# Fixed quality settings throughout (max safe for 12GB RAM):
#   GPTS=256, SLICE_THICKNESS=1, SAMPLING_FACTOR=3
#
# Run from project root:
#   bash diffraction/param_sweep.sh
# =============================================================

PYTHON=/home/suban/miniconda3/envs/glass_sem_env/bin/python
OUTDIR="diffraction/param_sweep"

# Save original config and restore on exit (even on Ctrl+C)
cp config.py config.py.bak
trap 'cp config.py.bak config.py && rm -f config.py.bak && echo "Config restored."' EXIT

# ---------------------------------------------------------------
# run_sweep PARAM_NAME VALUE
# ---------------------------------------------------------------
run_sweep() {
    local param=$1
    local value=$2
    local folder="$OUTDIR/$param"
    mkdir -p "$folder"

    # Start from backup, apply this param, lock quality settings
    cp config.py.bak config.py
    sed -i "s|^$param\s*=.*|$param = $value|" config.py
    sed -i "s|^GPTS\s*=.*|GPTS = 256|" config.py
    sed -i "s|^SLICE_THICKNESS\s*=.*|SLICE_THICKNESS = 1|" config.py
    sed -i "s|^SAMPLING_FACTOR\s*=.*|SAMPLING_FACTOR = 3|" config.py
    sed -i "s|^BLOCKED_VAL\s*=.*|BLOCKED_VAL = False|" config.py
    sed -i "s|^TO_ZARR\s*=.*|TO_ZARR = False|" config.py

    echo "  [$param = $value]"

    $PYTHON - <<EOF
import time
t0 = time.time()

from data_io.trajectory import load_trajectory
from config import TRAJECTORY_FILE
from diffraction.simulate import (
    compute_average_structure,
    correct_atom_types,
    simulate_4d_stem,
)
from visualization.plot import plot_diffraction

trajectory = load_trajectory(TRAJECTORY_FILE)
atoms      = compute_average_structure(trajectory)
atoms      = correct_atom_types(atoms)

dp = simulate_4d_stem(atoms, output_file=None)
plot_diffraction(
    dp,
    blocked   = False,
    save_path = "$folder/${param}_${value}.png",
)

print(f"Done in {(time.time()-t0)/60:.1f} min")
EOF
}

# ---------------------------------------------------------------
echo "=== 4D-STEM Parameter Sweep ==="
echo "Results → $OUTDIR/"
echo ""

echo "--- CONVERGENCE_SEMIANGLE_MRAD ---"
for val in 5 10 15 20 25 30; do
    run_sweep "CONVERGENCE_SEMIANGLE_MRAD" "$val"
done

echo ""
echo "--- DEFOCUS (Å) ---"
for val in 0 50 100 200 500 1000; do
    run_sweep "DEFOCUS" "$val"
done

echo ""
echo "--- BEAM_ENERGY_KV ---"
for val in 50 100 200 300; do
    run_sweep "BEAM_ENERGY_KV" "$val"
done

echo ""
echo "--- MAX_ANGLE_MRAD ---"
for val in 40 60 80 100; do
    run_sweep "MAX_ANGLE_MRAD" "$val"
done

echo ""
echo "--- PHONON_SIGMA (Å) ---"
for val in 0.01 0.05 0.08 0.1; do
    run_sweep "PHONON_SIGMA" "$val"
done

echo ""
echo "--- NUM_PHONON_CONFIGS ---"
for val in 1 4 8; do
    run_sweep "NUM_PHONON_CONFIGS" "$val"
done

echo ""
echo "=== Sweep complete ==="