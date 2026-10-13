# =============================================================
# dynamics/flow_units.py — non-affine displacement (D^2_min)
#
# Implements the Falk & Langer (1998) local non-affine squared
# displacement, used to identify STZ-like / "flow unit"-like
# rearrangement events between two configurations of the same
# system. No applied external deformation is required — this
# works directly on a purely thermal trajectory by comparing
# any two frames.
#
# Reference:
#   Falk, M. L. & Langer, J. S. (1998). Dynamics of viscoplastic
#   deformation in amorphous solids. Phys. Rev. E, 57, 7192-7205.
# =============================================================

import numpy as np
from ase.neighborlist import neighbor_list

from config import CUTOFF_DICT


def compute_d2min(atoms_ref, atoms_cur, cutoff=CUTOFF_DICT, min_neighbors=4):
    """
    Compute the non-affine squared displacement D^2_min for every
    atom, between a reference configuration (t0) and a comparison
    configuration (t) of the same system.

    Parameters
    ----------
    atoms_ref : ase.Atoms
        Reference configuration (t0).
    atoms_cur : ase.Atoms
        Comparison configuration (t).
    cutoff : float or dict
        Neighbor cutoff radius (Å) defining each atom's local
        neighborhood, evaluated in the REFERENCE configuration.
        Defaults to config.CUTOFF_DICT — a dict keyed by
        (Z_i, Z_j) real-atomic-number pairs, giving each of
        Zr-Zr/Zr-Cu/Cu-Cu its own g(r)-derived cutoff (the first
        MINIMUM after the first peak — the full coordination shell,
        not just the literal bond distance; see config.py's
        comment on CUTOFF_* vs PEAK_* for why D^2_min specifically
        wants the minimum-based one). A plain float also still
        works, for a single global cutoff.
    min_neighbors : int
        Atoms with fewer neighbors than this are skipped (too few
        vectors for a stable 3x3 fit) and assigned NaN.

    Returns
    -------
    d2min : (N,) np.ndarray
        Non-affine squared displacement per atom (Å^2). NaN where
        the local fit could not be performed.
    F_all : (N, 3, 3) np.ndarray
        Best-fit local affine deformation gradient per atom.
        NaN rows where skipped.

    Notes
    -----
    The neighbor TOPOLOGY (which atoms are neighbors of atom i) is
    built once from the reference configuration and reused — that
    part is standard practice, valid as long as the neighbor *set*
    doesn't change much between frames (true here for a thermal
    trajectory).

    However, the periodic-image bookkeeping is NOT reused: LAMMPS
    dump files store coordinates wrapped back into the primary cell
    at every frame (no image flags in this dump), so an atom that
    drifts across a boundary appears to "teleport" by one full box
    length even though nothing physically moved far. If we naively
    reused the reference frame's periodic shift S, that teleport
    would show up as a spurious, huge D^2_min for every atom
    involved. Instead, the current-frame relative vectors are
    recomputed with a fresh minimum-image reduction, which is
    robust to individual atoms being rewrapped between frames.
    """
    n_atoms = len(atoms_ref)

    # i, j : neighbor atom-index pairs (topology fixed at the REFERENCE frame)
    # S    : integer periodic-image shift per pair AT THE REFERENCE FRAME,
    #        so the reference bond vector is  r0 = pos[j] - pos[i] + S @ cell
    i_idx, j_idx, S = neighbor_list("ijS", atoms_ref, cutoff)

    pos_ref, pos_cur = atoms_ref.positions, atoms_cur.positions
    cell_ref, cell_cur = atoms_ref.cell.array, atoms_cur.cell.array
    box_cur = np.diag(cell_cur)  # assumes an orthogonal box (true here)

    r0 = (pos_ref[j_idx] - pos_ref[i_idx]) + S @ cell_ref  # distance between neighbors 

    # Current relative vectors: fresh minimum-image reduction, NOT a
    # reused S. This is robust even if atom i or j individually got
    # rewrapped into a different periodic image between frames.
    raw_r1 = pos_cur[j_idx] - pos_cur[i_idx]
    r1 = raw_r1 - box_cur * np.round(raw_r1 / box_cur)

    d2min = np.full(n_atoms, np.nan)
    F_all = np.full((n_atoms, 3, 3), np.nan)

    # Group neighbor pairs by central atom i.
    order = np.argsort(i_idx)
    i_sorted = i_idx[order]
    r0_sorted, r1_sorted = r0[order], r1[order]
    boundaries = np.searchsorted(i_sorted, np.arange(n_atoms + 1))

    for atom in range(n_atoms):
        start, end = boundaries[atom], boundaries[atom + 1]
        if end - start < min_neighbors:
            continue

        r0_i, r1_i = r0_sorted[start:end], r1_sorted[start:end]

        # Least-squares affine fit:  F = argmin_F sum_j |r1_j - F r0_j|^2
        # Closed form:  F = X @ Y^-1
        #   X = sum_j  r1_j (outer) r0_j
        #   Y = sum_j  r0_j (outer) r0_j
        X = r1_i.T @ r0_i
        Y = r0_i.T @ r0_i

        try:
            F = X @ np.linalg.inv(Y)
        except np.linalg.LinAlgError:
            continue  # degenerate (near-coplanar) neighborhood, rare

        residual = r1_i - r0_i @ F.T
        d2min[atom] = np.sum(residual**2)
        F_all[atom] = F

    return d2min, F_all


def d2min_trajectory(trajectory, t0=0, frame_step=1, cutoff=CUTOFF_DICT,
                     reference_positions=None):
    """
    Compute D^2_min at every frame relative to a fixed reference
    frame t0, across a full loaded trajectory.

    Parameters
    ----------
    trajectory : list of ase.Atoms
    t0 : int
        Index of the reference frame.
    frame_step : int
        Stride between compared frames (1 = every frame).
    cutoff : float or dict — see compute_d2min's docstring.
    reference_positions : (N, 3) array, optional
        If given (e.g. the mean structure from
        utils.compute_displacements(utils.unwrap_positions(...))), the
        reference configuration is trajectory[t0] with its positions
        replaced by these, instead of trajectory[t0] itself.

    Returns
    -------
    d2min_series : list of (N,) np.ndarray — one per compared frame
    frame_indices : list of int — which frames were compared to the reference
    """
    atoms_ref = trajectory[t0]
    if reference_positions is not None:
        atoms_ref = atoms_ref.copy()
        atoms_ref.positions = reference_positions
    frame_indices = list(range(0, len(trajectory), frame_step))
    d2min_series = [
        compute_d2min(atoms_ref, trajectory[idx], cutoff=cutoff)[0]
        for idx in frame_indices
    ]
    return d2min_series, frame_indices

def d2min_consecutive_trajectory(trajectory, cutoff=CUTOFF_DICT):
    """
    Compute D^2_min between every pair of CONSECUTIVE frames
    (frame 0→1, 1→2, ..., N-2→N-1) — distinct from d2min_trajectory
    above, which compares every frame against one FIXED reference
    frame. This is the "how much did each atom rearrange during
    this specific step" view, rather than "how far has each atom
    drifted from where it started".

    Also returns the CUMULATIVE sum of D^2_min per atom over the
    trajectory — i.e. cumulative[k] is the running total of
    non-affine squared displacement each atom has accumulated by
    frame k. Since D^2_min >= 0 always, this is monotonically
    non-decreasing per atom — it only ever grows — which is the
    quantity the animations below color atoms by.

    Parameters
    ----------
    trajectory : list of ase.Atoms
    cutoff : float or dict — see compute_d2min's docstring.

    Returns
    -------
    d2min_series : list of (N,) np.ndarray, length n_frames-1 —
        D^2_min for step k -> k+1, at index k.
    cumulative : (n_frames, N) np.ndarray —
        cumulative[0] is all zeros (no motion yet); cumulative[k]
        for k>0 is the running sum of d2min_series[0..k-1].
    """
    n_frames = len(trajectory)
    n_atoms = len(trajectory[0])

    d2min_series = []
    for k in range(n_frames - 1):
        d2min, _ = compute_d2min(trajectory[k], trajectory[k + 1], cutoff=cutoff)
        # NaN atoms (too few neighbors that step) contribute 0 to the
        # running total rather than poisoning every subsequent frame.
        d2min_series.append(np.nan_to_num(d2min, nan=0.0))

    cumulative = np.zeros((n_frames, n_atoms))
    for k in range(1, n_frames):
        cumulative[k] = cumulative[k - 1] + d2min_series[k - 1]

    return d2min_series, cumulative


if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import CUTOFF_ZRZR, CUTOFF_ZRCU, CUTOFF_CUCU
    from visualization.plot import plot_d2min_histogram, plot_d2min_map

    trajectory = load_trajectory()
    print(f"Loaded {len(trajectory)} frames, {len(trajectory[0])} atoms each.")

    print(f"CUTOFF_ZRZR (config.py, g(r)-derived): {CUTOFF_ZRZR:.3f} A")
    print(f"CUTOFF_ZRCU (config.py, g(r)-derived): {CUTOFF_ZRCU:.3f} A")
    print(f"CUTOFF_CUCU (config.py, g(r)-derived): {CUTOFF_CUCU:.3f} A")

    # --- D^2_min between first and last frame ---
    # cutoff defaults to CUTOFF_DICT (species-resolved, minimum-based —
    # see compute_d2min's docstring), so no need to pass it explicitly.
    d2min, F_all = compute_d2min(trajectory[0], trajectory[-1])
    valid = ~np.isnan(d2min)
    print(f"\nD^2_min computed for {valid.sum()}/{len(d2min)} atoms "
          f"({(~valid).sum()} skipped, insufficient neighbors).")
    print(f"D^2_min range:  [{np.nanmin(d2min):.4f}, {np.nanmax(d2min):.4f}] A^2")
    print(f"D^2_min mean:   {np.nanmean(d2min):.4f} A^2")
    print(f"D^2_min median: {np.nanmedian(d2min):.4f} A^2")

    plot_d2min_histogram(d2min)
    plot_d2min_map(trajectory[0], d2min)