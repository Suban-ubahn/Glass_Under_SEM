# =============================================================
# scripts/run_pca_reliability.py
#
# How far can the PCA behind the soft-mode analysis be trusted with this many frames?
#   1. Eigenvalues: how much of each eigenvalue is still there in frames that were not
#      used to find the mode  (pca_reliability_test)
#   2. Softness map: does the per-atom softness map survive when it is computed from
#      only half of the frames  (softness_reproducibility_test)
# Both tests cut the trajectory into a first and a second half. The definitions, the
# null model and how to read the output are in the header of
# analysis/soft_modes/pca_reliability.py.
#
# Outputs go to outputs/run_pca_reliability/ (see utils.get_output_dir()).
# =============================================================

import os

import numpy as np

from analysis.soft_modes.pca_reliability import (
    pca_reliability_test, print_pca_reliability,
    softness_reproducibility_test, print_softness_reproducibility,
)
from config import TRAJECTORY_FILE, N_SOFT_MODES, SOFT_MODE_PR_MAX
from data_io.trajectory import load_trajectory
from utils import get_output_dir, unwrap_positions, compute_displacements
from visualization.plot import plot_pca_reliability, plot_softness_reproducibility

OUTPUT_DIR = get_output_dir()


if __name__ == "__main__":
    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames, {len(trajectory[0])} atoms.")

    # The displacement matrix the soft-mode analysis decomposes: displacements from the
    # mean structure, mass-weighted (soft_mode_analysis_3d() with mass_weight=True).
    positions = unwrap_positions(np.array([a.positions for a in trajectory]),
                                 np.array([a.cell.array.diagonal() for a in trajectory]))
    u, _ = compute_displacements(positions)
    masses = trajectory[0].get_masses()

    # --- 1. Eigenvalues: first half vs second half ---
    print("\n=== PCA reliability: eigenvalues ===")
    pca_res = pca_reliability_test(u, masses)
    print_pca_reliability(pca_res)
    plot_pca_reliability(pca_res, save_path=os.path.join(OUTPUT_DIR, "pca_reliability.png"))

    # --- 2. Softness map: first half vs second half ---
    print("\n=== PCA reliability: softness map ===")
    soft_res = softness_reproducibility_test(
        u, masses, n_soft_modes=N_SOFT_MODES, pr_max=SOFT_MODE_PR_MAX)
    print_softness_reproducibility(soft_res)
    plot_softness_reproducibility(
        soft_res, save_path=os.path.join(OUTPUT_DIR, "softness_reproducibility.png"))