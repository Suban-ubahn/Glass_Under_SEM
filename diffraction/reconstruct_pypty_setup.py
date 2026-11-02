# =============================================================
# diffraction/reconstruct_pypty_setup.py
#
# Prepares the 4D-STEM data for ptychographic reconstruction:
#   - Loads the abtem .zarr dataset
#   - Reshapes from (scan_x, scan_y, kx, ky) → (n_positions, kx, ky)
#   - Saves as .h5 with dataset name "data" (pypty format)
#
# Run with glass_sem_env:
#   conda activate glass_sem_env
#   python diffraction/reconstruct_pypty_setup.py
# =============================================================

import numpy as np
import abtem
import h5py
import json
from config import BEAM_ENERGY_KV, CONVERGENCE_SEMIANGLE_MRAD, SAMPLING_FACTOR

ZARR_PATH = "outputs/diffraction/4d_stem.zarr"
H5_PATH   = "outputs/diffraction/4d_stem.h5"


if __name__ == "__main__":
    print("Loading zarr dataset...")
    dp = abtem.from_zarr(ZARR_PATH)
    dp.compute()

    arr = np.array(dp.array)
    print(f"  Loaded shape: {arr.shape}  (scan_x, scan_y, kx, ky)")

    scan_x, scan_y, kx, ky = arr.shape
    arr_3d = arr.reshape(scan_x * scan_y, kx, ky)
    print(f"  Reshaped to:  {arr_3d.shape}  (n_positions, kx, ky)")

    print(f"Saving to {H5_PATH}...")
    with h5py.File(H5_PATH, "w") as f:
        f.create_dataset("data", data=arr_3d)

    print(f"Done. Scan grid: {scan_x} × {scan_y} = {scan_x*scan_y} positions")
    print(f"      Detector:  {kx} × {ky} pixels")

    probe = abtem.Probe(
        energy           = BEAM_ENERGY_KV * 1e3,
        semiangle_cutoff = CONVERGENCE_SEMIANGLE_MRAD,
    )
    scan_step_A = float(SAMPLING_FACTOR * probe.aperture.nyquist_sampling)
    print(f"Scan step: {scan_step_A:.4f} Å")

    metadata = {
        "scan_x"      : scan_x,
        "scan_y"      : scan_y,
        "scan_step_A" : scan_step_A,
        "kx"          : kx,
        "ky"          : ky,
    }
    with open("outputs/diffraction/scan_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved: outputs/diffraction/scan_metadata.json")