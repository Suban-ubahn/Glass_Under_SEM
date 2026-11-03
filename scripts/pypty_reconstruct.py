# =============================================================
# scripts/pypty_reconstruct.py
#
# Iterative ptychographic reconstruction using pypty.
#
# Run with pypty env:
#   conda activate pypty
#   python scripts/pypty_reconstruct.py
#
# Requires: outputs/pypty_setup/{4d_stem.h5,scan_metadata.json}  (from scripts/pypty_setup.py)
# Output:   outputs/pypty_reconstruct/
# =============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import h5py
import pypty
from scipy import constants
import json

from config import (
    BEAM_ENERGY_KV,
    CONVERGENCE_SEMIANGLE_MRAD,
    DEFOCUS,
    SAMPLING_FACTOR,
)

H5_PATH       = "outputs/pypty_setup/4d_stem.h5"
META_PATH     = "outputs/pypty_setup/scan_metadata.json"
OUTPUT_FOLDER = "outputs/pypty_reconstruct"

# ------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------

def load_scan_metadata():
    with open(META_PATH) as f:
        m = json.load(f)
    print(f"  Scan grid:  {m['scan_x']} × {m['scan_y']} = {m['scan_x']*m['scan_y']} positions")
    print(f"  Scan step:  {m['scan_step_A']:.4f} Å")
    return m["scan_x"], m["scan_y"], m["scan_step_A"]

# ------------------------------------------------------------
# Build experimental parameters
# ------------------------------------------------------------

def build_experimental_params(scan_x, scan_y, scan_step_A):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    return {
        "output_folder"      : OUTPUT_FOLDER,
        "path_json"          : "",
        "data_path"          : H5_PATH,

        "rez_pixel_size_A"   : None,
        "rez_pixel_size_mrad": None,
        "conv_semiangle_mrad": CONVERGENCE_SEMIANGLE_MRAD,

        "scan_size"          : [scan_x, scan_y],
        "scan_step_A"        : scan_step_A,
        "fov_nm"             : None,

        "acc_voltage"        : BEAM_ENERGY_KV,
        "aberrations"        : [DEFOCUS, 0, 0],

        "PLRotation_deg"     : 0.0,
        "bright_threshold"   : 0.2,

        "total_thickness"    : 49.0,
        "num_slices"         : 1,

        "plot"               : False,
        "print_flag"         : 3,
        "data_pad"           : None,

        "chemical_formula"   : "Zr50Cu50",
        "sample_name"        : "Zr50Cu50_metallic_glass_MD_average",

        "compute_batch"      : 8,
        "backend"            : "gpu",
        # "default_dtype"      : "single",
        "load_one_by_one"    : True,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    print("Reading scan shape and scan step from jason metadata...")
    scan_x, scan_y, scan_step_A = load_scan_metadata()

    print("Building experimental parameters...")
    experimental_params = build_experimental_params(scan_x, scan_y, scan_step_A)

    print("Joining with default pypty preset...")
    pypty_params = pypty.initialize.append_exp_params(
        experimental_params,
        pypty_params=None,
    )

    pypty_params["compute_batch"] = 8

    # --- Resume from checkpoint, applied AFTER append_exp_params ---
    # append_exp_params() hardcodes obj/probe/epoch_prev to fresh-start
    # defaults regardless of what's in experimental_params, so these must
    # be set directly on pypty_params, not passed in earlier.
    co_path = os.path.join(OUTPUT_FOLDER, "co.npy")
    cp_path = os.path.join(OUTPUT_FOLDER, "cp.npy")
    cg_path = os.path.join(OUTPUT_FOLDER, "cg.npy")
    loss_path = os.path.join(OUTPUT_FOLDER, "loss.csv")

    if all(os.path.exists(p) for p in [co_path, cp_path, cg_path, loss_path]):
        import csv
        with open(loss_path) as f:
            rows = list(csv.DictReader(f))
        last_epoch = int(rows[-1]["epoch"]) if rows else 0

        pypty_params["obj"] = np.load(co_path)
        pypty_params["probe"] = np.load(cp_path)
        pypty_params["positions"] = np.load(cg_path)
        pypty_params["epoch_prev"] = last_epoch
        print(f"Existing checkpoint found — resuming from epoch {last_epoch}.")
    else:
        print("No checkpoint found — starting fresh.")

    print("Starting iterative ptychography...")
    pypty.iterative.run(pypty_params)

    print("Saving results as Nexus file...")
    nxs_path = os.path.join(OUTPUT_FOLDER, "reconstruction.nxs")
    pypty.utils.convert_to_nxs(OUTPUT_FOLDER, nxs_path)

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} minutes")
    print(f"Results in: {OUTPUT_FOLDER}")