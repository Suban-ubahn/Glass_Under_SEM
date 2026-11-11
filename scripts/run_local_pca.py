# =============================================================
# scripts/run_local_pca.py
#
# Moving-window (local) PCA next to the full-system PCA: does decomposing a few atoms at a time
# identify collective fluctuations more reliably with 101 frames?
#   1. windows: LOCAL_PCA_WINDOW_MODE = "neighborhood" (centre atom + its n-1 nearest neighbours, minimum image on
#      the time-averaged positions; every LOCAL_PCA_STEP-th atom is a centre) or "index" (atoms 0..n-1, 1..n, 2..n+1, ...);
#      LOCAL_PCA_WINDOW_SIZE atoms per window
#   2. covariance + full eigen-decomposition of every window, eigenvalues, eigenvectors,
#      participation ratios (same mass weighting / displacement / periodic-boundary conventions
#      as the full-system soft-mode PCA)
#   3. comparison with the full-system PCA: eigenvalues, eigenvectors (normalised overlaps with the
#      global modes restricted to the window), leading eigenspaces (principal angles)
#   4. are the same local modes found in overlapping windows?
#   5. shifted-column null (chance level of every number above)
#   6. dips of the median participation curve (LOCAL_PCA_DIP_*): detected automatically, marked in the plots,
#      compared with the global modes / followed across windows even beyond LOCAL_PCA_N_COMPARE_MODES, and the modes
#      at the dips investigated (P(a_n), real-space eigenvectors, dominant atoms: analysis/soft_modes/local_pca_dips.py)
# Independent of the TLS detection pipeline. The mathematics, what the comparison can and cannot show,
# and the difference between numerical resolution and statistical accuracy are in the header of
# analysis/soft_modes/local_pca.py. All parameters are in config.py (LOCAL_PCA_*).
#
# Outputs go to outputs/run_local_pca/ (see utils.get_output_dir()).
# =============================================================

import os

import numpy as np

from analysis.soft_modes.local_pca import (
    local_pca_analysis, print_local_pca, save_local_pca, window_spatial_extent,
    order_locality,
)
from analysis.soft_modes.local_pca_dips import (
    investigate_dip_modes, write_dip_csv, dip_investigation_arrays, print_dip_investigation,
)
from analysis.soft_modes.pca_reliability import _flat, _effective_frames
from config import (
    TRAJECTORY_FILE,
    LOCAL_PCA_WINDOW_MODE, LOCAL_PCA_WINDOW_SIZE, LOCAL_PCA_STEP, LOCAL_PCA_N_GLOBAL_MODES, LOCAL_PCA_SUBSPACE_DIMS,
    LOCAL_PCA_N_COMPARE_MODES, LOCAL_PCA_CONSISTENCY_OFFSETS, LOCAL_PCA_TRACK_THRESHOLD,
    LOCAL_PCA_MIN_SHARED_WEIGHT, LOCAL_PCA_N_NULL, LOCAL_PCA_SAVE_EIGENVECTORS,
    LOCAL_PCA_DIP_PROMINENCE, LOCAL_PCA_DIP_REL_HEIGHT, LOCAL_PCA_DIP_SEARCH, LOCAL_PCA_DIP_MAX_MODES,
    LOCAL_PCA_DIP_QUANTILES, LOCAL_PCA_DIP_TOP_ATOMS,
)
from data_io.trajectory import load_trajectory
from utils import get_output_dir, unwrap_positions, compute_displacements
from visualization.plot import (
    plot_local_pca_spectra, plot_local_participation, plot_local_vs_global, plot_local_consistency,
    plot_mode_amplitude_grid, plot_dip_eigenvectors,
)

OUTPUT_DIR = get_output_dir()


if __name__ == "__main__":
    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames, {len(trajectory[0])} atoms.")

    # Same displacement matrix as the soft-mode analysis: unwrapped, from the time-averaged
    # structure; the mass weighting is applied inside the PCA.
    box = np.array([a.cell.array.diagonal() for a in trajectory])
    positions = unwrap_positions(np.array([a.positions for a in trajectory]), box)
    u, mean_pos = compute_displacements(positions)
    masses = trajectory[0].get_masses()

    # Effective number of independent frames, only used to print the white-noise range with T_eff
    # (pooled over all coordinates; see run_pca_convergence.py for the per-mode values).
    t_eff = _effective_frames(_flat(u, masses))

    print("\n=== Local (moving-window) PCA ===")
    if LOCAL_PCA_WINDOW_MODE == "neighborhood":
        off_diagonal = max(np.abs(a.cell.array - np.diag(np.diag(a.cell.array))).max() for a in trajectory)
        if off_diagonal > 1e-6:
            raise SystemExit("neighbourhood windows need an orthorhombic cell (the periodic neighbour search uses the edge lengths)")
    res = local_pca_analysis(
        u, masses, window_size=LOCAL_PCA_WINDOW_SIZE, step=LOCAL_PCA_STEP,
        window_mode=LOCAL_PCA_WINDOW_MODE, mean_positions=mean_pos, box=box[0],
        n_global_modes=LOCAL_PCA_N_GLOBAL_MODES, subspace_dims=LOCAL_PCA_SUBSPACE_DIMS,
        n_compare_modes=LOCAL_PCA_N_COMPARE_MODES, consistency_offsets=LOCAL_PCA_CONSISTENCY_OFFSETS,
        track_threshold=LOCAL_PCA_TRACK_THRESHOLD, min_shared_weight=LOCAL_PCA_MIN_SHARED_WEIGHT,
        t_eff=t_eff, n_null=LOCAL_PCA_N_NULL, dip_prominence=LOCAL_PCA_DIP_PROMINENCE,
        dip_rel_height=LOCAL_PCA_DIP_REL_HEIGHT, dip_search=LOCAL_PCA_DIP_SEARCH)
    print_local_pca(res)

    # Is a window a compact neighbourhood? Radius of gyration of the windows against random groups of the
    # same number of atoms (neighbourhood windows: compact by construction; index windows: only if the atom
    # numbering follows space).
    rng = np.random.default_rng(0)
    n_at = res["windows"].shape[1]
    N = len(mean_pos)
    random_windows = np.array([rng.choice(N, n_at, replace=False) for _ in range(500)])
    rg_index = window_spatial_extent(mean_pos, res["windows"], box[0])
    rg_random = window_spatial_extent(mean_pos, random_windows, box[0])
    res["spatial"] = {"radius_of_gyration_index_windows": rg_index, "radius_of_gyration_random_groups": rg_random}
    print(f"\n  radius of gyration of the {n_at}-atom windows: median {np.median(rg_index):.1f} Å "
          f"(random groups of {n_at} atoms: {np.median(rg_random):.1f} Å)"
          + ("  -> windows are NOT spatially compact" if np.median(rg_index) > 0.8 * np.median(rg_random)
             else "  -> windows are spatially compact"))

    # How far apart are neighbouring columns of the heat maps? (neighbourhood windows: spatial Hilbert order of the centres)
    wi = res["windows_info"]
    if wi.get("spatial_order") is not None:
        cp, bx = wi["center_positions"], wi["box"]
        print(f"\n  heat map column order: median distance between the centres of neighbouring columns "
              f"{order_locality(cp, bx, wi['spatial_order']):.1f} Å in the spatial (Hilbert) order, "
              f"{order_locality(cp, bx):.1f} Å in atom-index order")

    # What the modes at the dips are: P(a_n), real-space eigenvectors, dominant atoms and their environment.
    # Cutoffs: first minima of the partial g(r) (cutoffs.py), by element symbol.
    if res["dips"] is not None and len(res["dips"]["modes"]):
        from ase.data import chemical_symbols
        from cutoffs import CUTOFF_DICT
        symbols = np.array(trajectory[0].get_chemical_symbols())
        cutoff = {(chemical_symbols[a], chemical_symbols[b]): r for (a, b), r in CUTOFF_DICT.items()}
        inv = investigate_dip_modes(res, u, masses, symbols, mean_pos, box[0], cutoff,
                                    max_modes=LOCAL_PCA_DIP_MAX_MODES, quantiles=LOCAL_PCA_DIP_QUANTILES,
                                    top_atoms=LOCAL_PCA_DIP_TOP_ATOMS)
        print_dip_investigation(inv)
        res["dip_investigation"] = dip_investigation_arrays(inv)
        write_dip_csv(inv, os.path.join(OUTPUT_DIR, "local_pca_dip_atoms.csv"))
        plot_mode_amplitude_grid(inv["amplitude_diag"], ncols=len(inv["classes"]),
                                 save_path=os.path.join(OUTPUT_DIR, "local_pca_dip_amplitudes.png"))
        plot_dip_eigenvectors(inv, save_path=os.path.join(OUTPUT_DIR, "local_pca_dip_eigenvectors.png"))

    plot_local_pca_spectra(res, save_path=os.path.join(OUTPUT_DIR, "local_pca_spectra.png"))
    plot_local_participation(res, save_path=os.path.join(OUTPUT_DIR, "local_pca_participation.png"))
    plot_local_vs_global(res, save_path=os.path.join(OUTPUT_DIR, "local_pca_vs_global.png"))
    plot_local_consistency(res, save_path=os.path.join(OUTPUT_DIR, "local_pca_consistency.png"))
    save_local_pca(res, os.path.join(OUTPUT_DIR, "local_pca.npz"),
                   eigenvectors_dtype=np.float32 if LOCAL_PCA_SAVE_EIGENVECTORS else None)
