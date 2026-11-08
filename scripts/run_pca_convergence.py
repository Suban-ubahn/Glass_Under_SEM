# =============================================================
# scripts/run_pca_convergence.py
#
# Statistical behaviour of the PCA behind the soft modes, beyond the split-half tests of
# run_pca_reliability.py:
#   1. Autocorrelation functions (ACF) of the leading mode amplitudes a_n(t) and of a_n(t)^2
#   2. Integrated autocorrelation time and effective sample size T_eff (windowed sum, with the
#      lag-one AR(1) estimate for comparison; unreliable estimates are flagged; shifted-column null
#      showing what the selection of the modes alone produces -- see the IN-SAMPLE CAVEAT)
#   3. Convergence of the PCA with the number of frames (first 20, 40, ..., 101 frames):
#      eigenvalues, eigenvectors matched by maximum overlap, and leading eigenspaces
# The definitions, the truncation rules, the flags and the LIMITATIONS (nested segments are not
# independent samples) are in the header of analysis/soft_modes/pca_convergence.py.
# All parameters are in config.py (PCA_ACF_*, PCA_CONV_*).
#
# Outputs go to outputs/run_pca_convergence/ (see utils.get_output_dir()).
# =============================================================

import os

import numpy as np

from analysis.soft_modes.pca_convergence import (
    pca_temporal_analysis, print_amplitude_acf, pca_convergence, print_pca_convergence, save_results,
)
from config import (
    TRAJECTORY_FILE,
    PCA_ACF_N_MODES, PCA_ACF_MAX_LAG, PCA_ACF_WINDOW_METHOD, PCA_ACF_WINDOW_C,
    PCA_ACF_MIN_T_OVER_TAU, PCA_ACF_MAX_REL_ERR, PCA_ACF_BAND_Z, PCA_ACF_AR1_LAGS, PCA_ACF_N_NULL,
    PCA_CONV_LENGTHS, PCA_CONV_N_MODES, PCA_CONV_SUBSPACE_DIM, PCA_CONV_MATCHING, PCA_CONV_N_NULL,
)
from data_io.trajectory import load_trajectory
from utils import get_output_dir, unwrap_positions, compute_displacements
from visualization.plot import plot_amplitude_acf, plot_pca_convergence

OUTPUT_DIR = get_output_dir()


if __name__ == "__main__":
    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames, {len(trajectory[0])} atoms.")

    # Same displacement matrix as the soft-mode analysis and run_pca_reliability.py:
    # unwrapped, from the time-averaged structure, mass-weighted inside the PCA.
    positions = unwrap_positions(np.array([a.positions for a in trajectory]),
                                 np.array([a.cell.array.diagonal() for a in trajectory]))
    u, _ = compute_displacements(positions)
    masses = trajectory[0].get_masses()

    # --- 1 + 2. ACF of the amplitudes, T_eff ---
    print("\n=== PCA mode amplitudes: autocorrelation and effective sample size ===")
    acf_res = pca_temporal_analysis(
        u, masses, n_modes=PCA_ACF_N_MODES, max_lag=PCA_ACF_MAX_LAG,
        window_method=PCA_ACF_WINDOW_METHOD, window_c=PCA_ACF_WINDOW_C,
        min_T_over_tau=PCA_ACF_MIN_T_OVER_TAU, max_rel_err=PCA_ACF_MAX_REL_ERR,
        band_z=PCA_ACF_BAND_Z, ar1_lags=PCA_ACF_AR1_LAGS, n_null=PCA_ACF_N_NULL)
    print_amplitude_acf(acf_res)
    plot_amplitude_acf(acf_res, save_path=os.path.join(OUTPUT_DIR, "pca_amplitude_acf.png"))
    save_results(acf_res, os.path.join(OUTPUT_DIR, "pca_amplitude_acf.npz"))

    # --- 3. Convergence with the number of frames ---
    print("\n=== PCA convergence with the number of frames ===")
    conv_res = pca_convergence(
        u, masses, lengths=PCA_CONV_LENGTHS, n_modes=PCA_CONV_N_MODES,
        subspace_dim=PCA_CONV_SUBSPACE_DIM, matching=PCA_CONV_MATCHING, n_null=PCA_CONV_N_NULL)
    print_pca_convergence(conv_res)
    plot_pca_convergence(conv_res, save_path=os.path.join(OUTPUT_DIR, "pca_convergence.png"))
    save_results(conv_res, os.path.join(OUTPUT_DIR, "pca_convergence.npz"))
