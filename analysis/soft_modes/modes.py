# =============================================================
# analysis/soft_modes/modes.py — quasi-harmonic soft-mode / participation-
# ratio analysis ("soft spots")
#
# Implements the method from:
#   Widmer-Cooper, A. & Harrowell, P., PRL 96, 185701 (2006)
#   Widmer-Cooper, A., Perry, H., Harrowell, P. & Reichman, D.R.,
#     Nat. Phys. 4, 711-715 (2008)
#   Manning, M.L. & Liu, A.J., PRL 107, 108302 (2011)
#
# Ports Part 1 (full 3D) and Part 2 (2D, per z-slice) of
# soft_mode_analysis.ipynb into this project's module layout, and
# ties it directly to CAVIAR's own formalism (Gladyshev et al.,
# Nat. Commun. 17, 5173 (2026), their Eqs. 1-2): the covariance
# matrix built here IS the lattice Green's tensor
# G_{l kappa alpha, l' kappa' beta} from that paper — a real CAVIAR
# reconstruction's Green's tensor can be fed straight into
# modes_from_covariance() instead of an MD trajectory.
#
# Sign convention: eigenvalues of the DISPLACEMENT COVARIANCE
# matrix are the inverse of a Hessian's (equipartition:
# <u^2> ~ kT/k), so here LARGE eigenvalue = SOFT (low-frequency)
# mode — opposite of the usual Hessian convention. Modes are always
# returned sorted from softest (largest eigenvalue) to stiffest.
# =============================================================

import numpy as np

from utils import unwrap_positions, compute_displacements

# ------------------------------------------------------------
# Quasi-harmonic mode decomposition
# ------------------------------------------------------------

def modes_from_displacements(u, masses=None, n_modes=None):
    """
    Diagonalize the (N*D, N*D) displacement covariance matrix
    WITHOUT ever forming it explicitly, via the economy SVD of the
    flattened (T, N*D) displacement matrix — the standard "essential
    dynamics" / quasi-harmonic PCA trick (Amadei et al. 1993).
    Correct and efficient whenever T << N*D, the usual case here (a
    few hundred-to-thousand MD frames vs. thousands of atoms x 2-3
    dimensions) — and it's also numerically cleaner than forming and
    diagonalizing a huge, potentially near-singular covariance matrix.

    If u_flat = A @ diag(S) @ Vt (economy SVD), then:
      - columns of V=Vt.T are the eigenvectors of the covariance
        matrix C = u_flat^T @ u_flat / T (equivalently, of CAVIAR's
        Green's tensor)
      - eigenvalues are S^2 / T
      - the per-frame AMPLITUDE along mode n is A[:, n] * S[n],
        returned as `amplitudes` so callers (tls_detection.py) don't
        need to re-project anything for the P(a_n) bimodality test.

    Parameters
    ----------
    u : (T, N, D) array — centered displacements, e.g. from
        compute_displacements().
    masses : (N,) array, optional — atomic masses (amu). If given,
        displacements are mass-weighted (u -> sqrt(m) * u) before the
        SVD — the physically correct quasi-harmonic treatment for a
        system with more than one species (Zr and Cu differ by a
        factor ~1.4 in mass). This mirrors CAVIAR Eq. 6, which
        mass-weights correlations to get a proper dynamical matrix.
        Without this, a genuinely stiff-but-light-atom mode can
        outrank a real soft spot just because light atoms rattle more
        per unit force constant.
    n_modes : int, optional — keep only the top n_modes softest
        modes (default: all min(T, N*D) resolvable by the SVD).

    Returns
    -------
    dict with:
      eigenvalues  : (K,) array, descending (softest first). Units:
                     Å^2 if masses=None; amu·Å^2 (mass-weighted — the
                     physically meaningful generalized-coordinate
                     units for a multi-species system, analogous to
                     CAVIAR's mass-weighted dynamical matrix, Eq. 6)
                     if masses is given. Don't compare eigenvalues
                     across a mass_weight=True run and a
                     mass_weight=False run directly — participation
                     ratios and the softness score (computed from
                     `eigenvectors`, always real-space and unit-norm)
                     stay comparable either way.
      eigenvectors : (N, D, K) array, unit-norm in REAL (not
                     mass-weighted) displacement space —
                     eigenvectors[:, :, n] is mode n.
      amplitudes   : (T, K) array — amplitudes[:, n] = a_n(t), in the
                     same units as eigenvalues' square root (Å if
                     masses=None; sqrt(amu)·Å if mass-weighted) —
                     fine for the bimodality SHAPE test in
                     tls_detection.py, which only cares about
                     relative separation vs. spread, not absolute
                     physical units.
      n_atoms, n_dim : int
    """
    T, N, D = u.shape
    flat = u.reshape(T, N * D).copy()

    sqrt_m = None
    if masses is not None:
        masses = np.asarray(masses, dtype=float)
        sqrt_m = np.repeat(np.sqrt(masses), D)   # (N*D,)
        flat = flat * sqrt_m[None, :]

    A, S, Vt = np.linalg.svd(flat, full_matrices=False)

    if n_modes is not None:
        A, S, Vt = A[:, :n_modes], S[:n_modes], Vt[:n_modes, :]

    eigenvalues = (S ** 2) / T
    amplitudes = A * S[None, :]                    # (T, K)

    V = Vt.T                                        # (N*D, K), mass-weighted basis
    if sqrt_m is not None:
        V = V / sqrt_m[:, None]                      # back to real-space displacement
        norms = np.linalg.norm(V, axis=0)
        norms[norms == 0] = 1.0
        V = V / norms[None, :]                       # re-normalize to unit vectors

    eigenvectors = V.reshape(N, D, -1)

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "amplitudes": amplitudes,
        "n_atoms": N,
        "n_dim": D,
    }


def modes_from_covariance(C, n_atoms, n_dim, masses=None, n_modes=None):
    """
    Same idea as modes_from_displacements(), but starting from an
    already-built (N*D, N*D) covariance / Green's-tensor matrix
    instead of a displacement trajectory — the entry point for real
    CAVIAR/pypty output (the Green's tensor of Gladyshev et al. Eq. 2)
    rather than MD data. Uses a direct symmetric eigh() since a real
    reconstruction's atom count per local analysis is small.

    Parameters
    ----------
    C : (N*D, N*D) array — symmetric covariance/Green's tensor, built
        from RAW (NOT mass-weighted) displacements u = r - r0
        (CAVIAR's own Eq. 2 is exactly this: an unweighted second
        moment), ordered atom-major (row/col index = atom*n_dim + dim),
        matching CAVIAR's (l, kappa, alpha) flattening.
    masses : (N,) array, optional — mass-weighting, applied as the
        similarity transform C -> M^{1/2} C M^{1/2} before
        diagonalizing (M = diag(mass per dof)). This MULTIPLIES by
        sqrt(mass), not divides: if q = M^{1/2}u is the mass-weighted
        displacement, then <qq^T> = M^{1/2}<uu^T>M^{1/2} = M^{1/2} C M^{1/2}
        — matching modes_from_displacements(), which mass-weights
        displacements the same way (u -> sqrt(m)*u) before computing
        their covariance via SVD. (An earlier version of this function
        divided instead of multiplied here — a sign error caught by
        directly comparing this function's output against
        modes_from_displacements() on the same data: eigenvalues were
        off by ~100x and nearly half the eigenvectors didn't match.
        See scripts/test_modes_from_covariance.py.)

    Returns
    -------
    Same shape of dict as modes_from_displacements(), except
    "amplitudes" is None — a bare covariance matrix carries no
    per-frame information, so tls_detection.detect_tls_amplitude()
    (which needs P(a_n)) cannot be run on this path; only
    participation-ratio-based softness works here.
    """
    C = np.asarray(C, dtype=float)
    sqrt_m = None
    if masses is not None:
        masses = np.asarray(masses, dtype=float)
        sqrt_m = np.repeat(np.sqrt(masses), n_dim)
        C = C * sqrt_m[:, None] * sqrt_m[None, :]

    eigenvalues, V = np.linalg.eigh(C)
    order = np.argsort(eigenvalues)[::-1]            # descending = softest first
    eigenvalues, V = eigenvalues[order], V[:, order]

    if n_modes is not None:
        eigenvalues, V = eigenvalues[:n_modes], V[:, :n_modes]

    if sqrt_m is not None:
        V = V / sqrt_m[:, None]
        norms = np.linalg.norm(V, axis=0)
        norms[norms == 0] = 1.0
        V = V / norms[None, :]

    eigenvectors = V.reshape(n_atoms, n_dim, -1)
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "amplitudes": None,
        "n_atoms": n_atoms,
        "n_dim": n_dim,
    }


# ------------------------------------------------------------
# Participation ratio & per-atom softness
# ------------------------------------------------------------

def participation_ratios(eigenvectors):
    """
    Inverse-participation-ratio-derived localization measure for
    every mode:

      PR_n = 1 / (N * sum_i p_i,n^2),   p_i,n = sum_d eigenvectors[i,d,n]^2

      PR -> 1   : mode spread evenly over all N atoms (delocalized,
                  phonon/Bloch-like).
      PR -> 1/N : mode localized on a single atom.

    Parameters
    ----------
    eigenvectors : (N, D, K) array, unit-norm per mode (as returned
        by modes_from_displacements()/modes_from_covariance()).

    Returns
    -------
    pr : (K,) array
    """
    N = eigenvectors.shape[0]
    p = np.sum(eigenvectors ** 2, axis=1)             # (N, K), sums to 1 over atoms per mode
    return 1.0 / (N * np.sum(p ** 2, axis=0))


def atom_softness_scores(eigenvalues, eigenvectors, n_soft_modes=20,
                          pr_max=None, weight_by_eigenvalue=True,
                          pr_weight=False):
    """
    Per-atom "softness": participation, summed over the softest
    `n_soft_modes` modes (optionally restricted to quasi-localized
    ones via a participation-ratio cutoff), optionally weighted by
    each mode's eigenvalue (= mean-squared amplitude, so a mode with
    more actual motion counts for more).

    This is the field plotted directly on the structure in
    Widmer-Cooper & Harrowell, and the ranking score used against
    D^2_min in analysis/soft_modes/comparison.py.

    Parameters
    ----------
    pr_max : float, optional — if given, only modes with
        participation_ratios() < pr_max are included, so extended
        long-wavelength acoustic modes (which, in a small, finite MD
        cell, can still show up among the largest-eigenvalue/softest
        modes) don't get counted as "soft spots". Start around a few
        times 1/N_analyzed and tune by eye against
        plot_participation_spectrum().
    pr_weight : bool — if True, ALSO weight each mode's contribution
        by 1/PR_n (its inverse participation ratio), so spatially
        localized modes count more per unit of an atom's own
        participation, and broad/extended modes are smoothly
        discounted rather than hard-excluded. This is a continuous
        complement to pr_max, not a replacement for it: pr_max drops
        a mode entirely once its PR crosses a line you pick;
        pr_weight never drops a mode, it just keeps downweighting
        less-localized ones the softer that line gets. The two can be
        combined (pr_max to drop clearly-delocalized modes outright,
        pr_weight to grade the rest).

    Returns
    -------
    softness : (N,) array
    mode_mask : (K,) bool array — which modes were actually used
        (handy to pass straight into tls_detection.detect_tls_amplitude()
        so Method B checks exactly the same soft/localized modes
        Method C flagged).
    """
    K = eigenvalues.shape[0]
    n_soft_modes = min(n_soft_modes, K)
    mode_mask = np.zeros(K, dtype=bool)
    mode_mask[:n_soft_modes] = True

    pr = None
    if pr_max is not None or pr_weight:
        pr = participation_ratios(eigenvectors)
    if pr_max is not None:
        mode_mask &= (pr < pr_max)

    p = np.sum(eigenvectors ** 2, axis=1)              # (N, K)
    weights = eigenvalues if weight_by_eigenvalue else np.ones_like(eigenvalues)
    if pr_weight:
        weights = weights / pr
    weights = np.where(mode_mask, weights, 0.0)

    softness = p @ weights
    return softness, mode_mask


# ------------------------------------------------------------
# High-level 3D (full system) wrapper
# ------------------------------------------------------------

def soft_mode_analysis_3d(trajectory, atom_indices=None, mass_weight=True,
                           n_modes=200, n_soft_modes=20, pr_max=None,
                           reference_frame=None):
    """
    Full 3D quasi-harmonic soft-mode analysis over a whole trajectory
    (or a chosen atom subset of it).

    Parameters
    ----------
    trajectory : list of ase.Atoms, from data_io.trajectory.load_trajectory()
    atom_indices : array of int, optional — restrict to a subset of
        atoms (e.g. one species, or a spatial region); defaults to
        all atoms.
    mass_weight : bool — see modes_from_displacements().
    n_modes : int, optional — cap the number of SVD modes computed
        (default 200). Only the softest end of the spectrum is ever
        used downstream, so capping this bounds runtime/memory on a
        large system without losing anything the analysis cares
        about; pass None for the full min(T, N*3) spectrum.
    n_soft_modes, pr_max : passed to atom_softness_scores().
    reference_frame : int, optional — if given, displacements are
        measured from THIS frame's (unwrapped) positions instead of
        the trajectory's time-average, i.e. u(t) = r(t) - r(reference_frame)
        — matching dynamics.flow_units.compute_d2min's convention of
        always comparing against one fixed frame (t0), rather than an
        ensemble mean. Default (None) keeps the standard quasi-
        harmonic choice (mean-centered, see compute_displacements()'s
        docstring for why that's the textbook-correct one for "thermal
        fluctuation around equilibrium"). Use reference_frame=0
        specifically when you want a softness field directly
        comparable, frame-for-frame, to a D^2_min(t0=0, ...) field —
        see softness_trajectory()/animate_softness_3d() in
        visualization/animate.py, which do exactly this.

    Returns
    -------
    dict — modes_from_displacements()'s output, plus:
      "softness"      : (N,) array
      "participation" : (K,) array
      "soft_mode_mask": (K,) bool array
      "atom_indices"  : the indices actually used
      "reference"     : (N, 3) positions used as reference — the mean,
                        unless reference_frame was given
    """
    if atom_indices is not None:
        positions = np.array([a.positions[atom_indices] for a in trajectory])
    else:
        positions = np.array([a.positions for a in trajectory])

    box = np.array([a.cell.array.diagonal() for a in trajectory])
    positions = unwrap_positions(positions, box)

    masses = None
    if mass_weight:
        masses = trajectory[0].get_masses()
        if atom_indices is not None:
            masses = masses[atom_indices]

    reference = None if reference_frame is None else positions[reference_frame]
    u, reference = compute_displacements(positions, reference=reference)
    result = modes_from_displacements(u, masses=masses, n_modes=n_modes)

    softness, mode_mask = atom_softness_scores(
        result["eigenvalues"], result["eigenvectors"],
        n_soft_modes=n_soft_modes, pr_max=pr_max,
    )
    softness_pr_weighted, _ = atom_softness_scores(
        result["eigenvalues"], result["eigenvectors"],
        n_soft_modes=n_soft_modes, pr_max=pr_max, pr_weight=True,
    )
    result["softness"] = softness
    result["softness_pr_weighted"] = softness_pr_weighted
    result["participation"] = participation_ratios(result["eigenvectors"])
    result["soft_mode_mask"] = mode_mask
    result["atom_indices"] = (atom_indices if atom_indices is not None
                               else np.arange(positions.shape[1]))
    result["reference"] = reference
    return result


# ------------------------------------------------------------
# High-level 2D, per-z-slice wrapper (CAVIAR-style projected slices)
# ------------------------------------------------------------

def assign_z_slices(mean_z, n_slices=10):
    """
    Bin atoms into n_slices equally-spaced z-slabs spanning
    [min(mean_z), max(mean_z)], using each atom's TIME-AVERAGED z
    (not its instantaneous z each frame) so slice membership is fixed
    over the whole trajectory. Mirrors how a real CAVIAR
    reconstruction assigns atoms to a fixed potential-slice index —
    see Gladyshev et al.'s "for each unit cell l, ... reconstructed
    object state n" — and avoids atoms flickering between slices
    frame to frame.

    Returns
    -------
    slice_idx : (N,) int array, values in [0, n_slices)
    edges     : (n_slices+1,) array — the z bin edges (Å)
    """
    mean_z = np.asarray(mean_z)
    edges = np.linspace(mean_z.min(), mean_z.max(), n_slices + 1)
    slice_idx = np.clip(np.digitize(mean_z, edges[1:-1]), 0, n_slices - 1)
    return slice_idx, edges


def soft_mode_analysis_2d_slices(trajectory, n_slices=10, mass_weight=True,
                                  n_modes=200, n_soft_modes=20, pr_max=None,
                                  min_atoms_per_slice=8, reference_frame=None):
    """
    2D (in-plane x,y only) soft-mode analysis, run independently in
    each of n_slices equally-spaced z-slabs — the direct analogue of
    what a real CAVIAR/pypty reconstruction with `num_slices` object
    slices sees: each slice only ever carries in-plane displacements
    of the atoms projected into it (CAVIAR's Results section: "we
    limit our analysis to the in-plane components of vibrations" —
    the z-resolution isn't good enough to quantify out-of-plane
    motion).

    Parameters
    ----------
    trajectory : list of ase.Atoms
    n_slices : int — default 10.
    min_atoms_per_slice : int — slices with fewer atoms than this are
        skipped (too few atoms/DOF for a meaningful covariance).
    reference_frame : int, optional — see soft_mode_analysis_3d()'s
        docstring; same meaning here (None = time-mean per atom,
        matching the usual quasi-harmonic convention; an int = that
        frame's position, matching compute_d2min's fixed-t0 convention).
        Slice ASSIGNMENT (which z-slab each atom belongs to) still
        always uses the time-mean z regardless of this setting — that
        part is just bookkeeping, not the displacement reference.

    Returns
    -------
    dict with:
      "slices"    : list of length n_slices; each entry is either
                    None (skipped, too few atoms) or a per-slice
                    result dict shaped like soft_mode_analysis_3d()'s
                    output but with n_dim=2 (x,y only), plus
                    "z_range" = (z_lo, z_hi) and "atom_indices" giving
                    the ORIGINAL (whole-system) atom indices in that
                    slice.
      "slice_idx" : (N,) array — which slice every atom was assigned to
      "z_edges"   : (n_slices+1,) array
      "softness"  : (N,) array — combined softness score; atoms
                    outside any computed slice (skipped/too sparse)
                    are NaN
    """
    positions_3d = np.array([a.positions for a in trajectory])   # (T, N, 3)
    box = np.array([a.cell.array.diagonal() for a in trajectory])
    positions_3d = unwrap_positions(positions_3d, box)
    mean_pos = positions_3d.mean(axis=0)
    slice_idx, z_edges = assign_z_slices(mean_pos[:, 2], n_slices=n_slices)

    masses_all = trajectory[0].get_masses() if mass_weight else None

    N = positions_3d.shape[1]
    softness_full = np.full(N, np.nan)
    softness_pr_weighted_full = np.full(N, np.nan)
    slices = []

    for s in range(n_slices):
        idx = np.where(slice_idx == s)[0]
        if len(idx) < min_atoms_per_slice:
            slices.append(None)
            continue

        pos_xy = positions_3d[:, idx, :2]                        # (T, n_s, 2)
        masses = masses_all[idx] if masses_all is not None else None

        ref_xy = None if reference_frame is None else pos_xy[reference_frame]
        u, reference = compute_displacements(pos_xy, reference=ref_xy)
        result = modes_from_displacements(u, masses=masses, n_modes=n_modes)

        softness, mode_mask = atom_softness_scores(
            result["eigenvalues"], result["eigenvectors"],
            n_soft_modes=n_soft_modes, pr_max=pr_max,
        )
        softness_pr_weighted, _ = atom_softness_scores(
            result["eigenvalues"], result["eigenvectors"],
            n_soft_modes=n_soft_modes, pr_max=pr_max, pr_weight=True,
        )
        result["softness"] = softness
        result["softness_pr_weighted"] = softness_pr_weighted
        result["participation"] = participation_ratios(result["eigenvectors"])
        result["soft_mode_mask"] = mode_mask
        result["atom_indices"] = idx
        result["reference"] = reference
        result["z_range"] = (z_edges[s], z_edges[s + 1])

        softness_full[idx] = softness
        softness_pr_weighted_full[idx] = softness_pr_weighted
        slices.append(result)

    return {
        "slices": slices,
        "slice_idx": slice_idx,
        "z_edges": z_edges,
        "softness": softness_full,
        "softness_pr_weighted": softness_pr_weighted_full,
    }


# ------------------------------------------------------------
# Softness, computed the way dynamics.flow_units.d2min_trajectory
# computes D^2_min -- a growing window relative to a fixed t0
# ------------------------------------------------------------

def softness_trajectory(trajectory, t0=0, frame_step=1, mass_weight=True,
                         n_soft_modes=None, pr_max=None, n_modes=None,
                         atom_indices=None):
    """
    Softness-score analogue of dynamics.flow_units.d2min_trajectory().
    Computes a per-atom softness field at every frame_indices[i] = t from
    the window of frames [t0, t] (inclusive).

    Reference: the MEAN STRUCTURE of the whole trajectory (unwrapped,
    time-average position per atom, see utils.compute_displacements()),
    the same for every window -- u(s) = r(s) - <r>. Because the reference
    does not depend on t, the scale of the softness does not grow with
    the window: a window is just a (less or more noisy) average of the
    per-frame squared displacements from the mean structure. A window of
    a single frame is enough (one displacement vector = one mode).

    Parameters
    ----------
    t0, frame_step : see d2min_trajectory() -- identical meaning.
    n_soft_modes : int or None -- number of softest modes summed per
        window; None (default) uses ALL modes of the window.
    n_modes, pr_max, mass_weight, atom_indices : as in soft_mode_analysis_3d().

    Returns
    -------
    softness_series : list of (N,) arrays, one per frame_indices[i]
    softness_pr_weighted_series : same, participation-weighted softness
    frame_indices : list of int
    """
    if atom_indices is not None:
        positions = np.array([a.positions[atom_indices] for a in trajectory])
    else:
        positions = np.array([a.positions for a in trajectory])
    box = np.array([a.cell.array.diagonal() for a in trajectory])
    u_all, _ = compute_displacements(unwrap_positions(positions, box))   # mean structure

    masses = None
    if mass_weight:
        masses = trajectory[0].get_masses()
        if atom_indices is not None:
            masses = masses[atom_indices]

    frame_indices = list(range(t0, len(trajectory), frame_step))
    softness_series, softness_pr_series = [], []
    for t in frame_indices:
        result = modes_from_displacements(u_all[t0:t + 1], masses=masses, n_modes=n_modes)
        K = len(result["eigenvalues"])
        k_soft = K if n_soft_modes is None else n_soft_modes
        softness, _ = atom_softness_scores(
            result["eigenvalues"], result["eigenvectors"],
            n_soft_modes=k_soft, pr_max=pr_max,
        )
        softness_pr, _ = atom_softness_scores(
            result["eigenvalues"], result["eigenvectors"],
            n_soft_modes=k_soft, pr_max=pr_max, pr_weight=True,
        )
        softness_series.append(softness)
        softness_pr_series.append(softness_pr)

    return softness_series, softness_pr_series, frame_indices
