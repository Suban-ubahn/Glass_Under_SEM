# =============================================================
# scripts/test_modes_from_covariance.py
#
# Validates analysis.soft_modes.modes_from_covariance() (the entry
# point for a real CAVIAR/pypty Green's tensor, unused elsewhere in
# this pipeline) against modes_from_displacements() (the SVD-based
# path the rest of the pipeline actually runs) on the SAME real data
# -- they should agree, since they diagonalize the same mathematical
# object (see analysis/soft_modes.py's module docstring: the
# covariance matrix IS CAVIAR's Green's tensor) via two different
# numerical routes (economy SVD of the T x D displacement matrix vs.
# direct eigh of the explicit D x D covariance matrix).
#
# Run from the repo root: python scripts/test_modes_from_covariance.py
#
# Caught a real bug: an earlier version of modes_from_covariance's
# mass-weighting divided by sqrt(mass) where it should have
# multiplied (see the fix note in analysis/soft_modes.py's
# modes_from_covariance docstring) -- this script is what caught it,
# and is worth re-running after any future change to either function.
# =============================================================

import numpy as np

from data_io.trajectory import load_trajectory
from config import TRAJECTORY_FILE
from utils import unwrap_positions, compute_displacements
from analysis.soft_modes import (
    modes_from_displacements, modes_from_covariance,
    participation_ratios, assign_z_slices,
)


def compare_on_subset(positions_3d, masses, atom_idx, label, n_check=50):
    """
    Run both code paths on the same subset of atoms and compare.

    n_check : int -- only compare the TOP n_check modes (by SVD
        eigenvalue). Modes near the bottom of what the SVD can
        resolve have eigenvalues at or near machine precision (true
        zero, mathematically -- the covariance matrix has rank at
        most T-1), where "the" eigenvector direction is arbitrary
        (any vector works equally well for a ~0 eigenvalue) and SVD
        vs. eigh will legitimately disagree on which arbitrary
        direction they picked. That's expected numerical noise, not
        a bug -- and irrelevant here anyway, since the pipeline only
        ever uses the softest ~20 modes (N_SOFT_MODES in config.py),
        comfortably inside n_check's margin.
    """
    pos = positions_3d[:, atom_idx, :]
    m = masses[atom_idx]
    u, _ = compute_displacements(pos)   # default: mean-referenced
    T, N, D = u.shape

    import time
    t0 = time.time()
    result_svd = modes_from_displacements(u, masses=m, n_modes=None)
    t_svd = time.time() - t0

    # Build the RAW (NOT mass-weighted) covariance -- modes_from_covariance
    # does its own mass-weighting internally, same contract CAVIAR's own
    # Green's tensor (Eq. 2) follows (an unweighted second moment).
    flat_raw = u.reshape(T, N * D)
    C_raw = (flat_raw.T @ flat_raw) / T

    t0 = time.time()
    result_eigh = modes_from_covariance(C_raw, n_atoms=N, n_dim=D, masses=m, n_modes=None)
    t_eigh = time.time() - t0

    K = min(n_check, len(result_svd["eigenvalues"]))
    ev_svd = result_svd["eigenvalues"][:K]
    ev_eigh = result_eigh["eigenvalues"][:K]
    pr_svd = participation_ratios(result_svd["eigenvectors"])[:K]
    pr_eigh = participation_ratios(result_eigh["eigenvectors"])[:K]
    overlaps = np.array([
        abs(np.dot(result_svd["eigenvectors"][:, :, n].ravel(),
                   result_eigh["eigenvectors"][:, :, n].ravel()))
        for n in range(K)
    ])

    print(f"\n--- {label}: {N} atoms, D = {N*D}, T = {T} frames ---")
    print(f"  modes_from_displacements (SVD):  {t_svd:.3f} s, "
          f"{len(result_svd['eigenvalues'])} modes resolved")
    print(f"  modes_from_covariance (eigh):     {t_eigh:.3f} s, "
          f"{len(result_eigh['eigenvalues'])} modes resolved "
          f"({t_eigh/max(t_svd,1e-9):.0f}x slower, "
          f"{N*D} x {N*D} matrix explicitly formed and diagonalized)")
    print(f"  top-{K} eigenvalues: max relative difference = "
          f"{np.max(np.abs(ev_svd - ev_eigh) / (np.abs(ev_svd) + 1e-300)):.2e}")
    print(f"  top-{K} participation ratios: max absolute difference = "
          f"{np.max(np.abs(pr_svd - pr_eigh)):.2e}")
    print(f"  top-{K} eigenvector overlap |v_svd . v_eigh|: "
          f"min = {overlaps.min():.4f}, mean = {overlaps.mean():.6f}")
    low = np.where(overlaps < 0.99)[0]
    if len(low):
        print(f"  modes with overlap < 0.99: {list(low)} "
              f"(eigenvalues {np.round(ev_svd[low], 6)} -- "
              f"check these aren't near-zero/degenerate before worrying)")
    else:
        print("  all top modes agree to overlap > 0.99.")


if __name__ == "__main__":
    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames, {len(trajectory[0])} atoms.")

    positions_3d = np.array([a.positions for a in trajectory])
    box = np.array([a.cell.array.diagonal() for a in trajectory])
    positions_3d = unwrap_positions(positions_3d, box)
    masses_all = trajectory[0].get_masses()

    # modes_from_covariance's realistic use case: a small, local
    # problem (one z-slice here, ~660-700 atoms; a real CAVIAR Green's
    # tensor would typically be even smaller -- a single neighborhood).
    mean_pos = positions_3d.mean(axis=0)
    slice_idx, _ = assign_z_slices(mean_pos[:, 2], n_slices=10)
    slice0_idx = np.where(slice_idx == 0)[0]
    compare_on_subset(positions_3d, masses_all, slice0_idx, "one z-slice")

    # A much smaller, genuinely CAVIAR-Green's-tensor-sized problem:
    # one atom's local neighborhood (itself + its bonded shell).
    from ase.neighborlist import neighbor_list
    from config import CUTOFF_DICT
    i_idx, j_idx, _ = neighbor_list("ijS", trajectory[0], CUTOFF_DICT)
    center_atom = 0
    neighborhood = np.unique(np.concatenate([[center_atom], j_idx[i_idx == center_atom]]))
    compare_on_subset(positions_3d, masses_all, neighborhood,
                       f"local neighborhood around atom {center_atom}")

    # NOT run here: the full 6750-atom system (D=20250). Forming the
    # explicit 20250x20250 covariance matrix for modes_from_covariance
    # alone needs ~3.3 GB just to hold it, and eigh's LAPACK workspace
    # pushed this past available memory in testing (killed by the OS
    # OOM killer after >2 minutes) -- confirming modes_from_covariance
    # is for small, local problems only, exactly as its docstring says.
    # modes_from_displacements (the SVD path) has no such limit, since
    # it never forms the D x D matrix at all -- that's the whole reason
    # the rest of this pipeline uses it instead.
    print("\n(Full 6750-atom system intentionally skipped here -- "
          "see comment above: modes_from_covariance is not feasible "
          "at that size, by design.)")