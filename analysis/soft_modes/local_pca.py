# =============================================================
# analysis/soft_modes/local_pca.py — moving-window ("local") PCA next to the
# full-system PCA, and how the two relate.
#
# QUESTION. The full-system PCA has P = N*D = 20250 coordinates and T = 101 frames, so it
# can return at most T-1 = 100 non-zero modes and each of them is selected on the same
# frames that measure its eigenvalue (see pca_reliability.py). Does decomposing a SMALL
# group of atoms at a time -- K = n*D coordinates, K < T-1 -- identify collective
# fluctuations more reliably?
#
# This module is independent of the TLS detection pipeline (tls.py); it only reads the
# displacement array u that the soft-mode analysis also uses, and does not change it.
#
# ------------------------------------------------------------
# SETUP (the conventions of modes.py / pca_reliability.py, unchanged)
# ------------------------------------------------------------
#   u[t,i,d]   displacement of atom i from its TIME-AVERAGED position over all T frames,
#              after unwrap_positions() (periodic boundaries); utils.compute_displacements().
#   X[t,(i,d)] = sqrt(m_i) u[t,i,d]        mass-weighted (masses=None: no weighting)
#   C = X^T X / T = sum_m lambda_m v_m v_m^T          full system, (P x P), |v_m| = 1
# Large eigenvalue = soft mode. Eigenvectors in this file are unit vectors in the
# MASS-WEIGHTED space, the space in which C is a covariance matrix and its eigenvectors are
# orthonormal (needed for projection matrices and principal angles). modes.py returns
# eigenvectors divided by sqrt(m) and re-normalised ("real space"); the two differ by the
# diagonal rescaling 1/sqrt(m) (Zr/Cu: sqrt(91.2/63.5) = 1.20). Participation ratios are
# computed from the REAL-space vectors exactly as modes.participation_ratios() does, so
# they are directly comparable with the existing ones.
#
# WINDOWS. A window w is a set S_w of n atoms (K = n*D coordinates). Two ways to define them
# (window_mode of local_pca_analysis, LOCAL_PCA_WINDOW_MODE in config.py):
#   "neighborhood"  (spatial) window of centre atom c = the n atoms closest to c, c included
#                   (n-1 nearest neighbours), by minimum-image distance between the TIME-AVERAGED
#                   positions (the structure the displacements are measured from), in the
#                   periodic box (orthorhombic). Every `step`-th atom is a centre (step 1: every atom,
#                   N windows). neighborhood_windows(). The window is a compact cluster, but its
#                   radius varies with the local density (k nearest, not a fixed radius), and the
#                   neighbour list is fixed over the whole trajectory: atoms that leave the cluster
#                   during the 101 frames (the rearrangements one is interested in) stay in it.
#   "index"         atom-index window of the original task: atoms 0..n-1, 1..n, 2..n+1, ... (shift `step`).
#                   Spatial only if the atom numbering follows space; in a LAMMPS dump of a quenched
#                   glass it normally does not (window_spatial_extent() shows this). "Local" then means
#                   "these n atoms", and the motion of unrelated atoms is whatever the global modes
#                   impose on them.
# Any (n_windows, n) integer array can be passed as `windows=` too; nothing else in this module assumes
# how a window was made, with one exception: which windows count as "overlapping neighbours" for the
# consistency analysis (window w + j for index windows; the j-th nearest window CENTRE for neighbourhood
# windows, `partners`, window_partners()). Windows are listed in order of their centre atom index, which is
# NOT a spatial order: the x axis of the heatmaps is the centre atom index; use
# res["windows_info"]["center_positions"] for spatial maps.
#
# WHAT CHANGES WITH SPATIAL WINDOWS (and what does not). Everything in the sections below (principal
# submatrix, interlacing, trace, the exact mixture of global modes, numerical vs statistical accuracy) is
# true for ANY set of atoms and is unchanged. What is different is the physics the window contains:
#   * Neighbouring atoms are strongly correlated through the long-wavelength global modes, so the leading
#     local modes of a compact cluster are expected to be, to a large part, near-uniform motion of the whole
#     cluster (rigid translation, with a high participation ratio), i.e. the restriction of long-wavelength
#     global modes to the cluster; a window has no momentum conservation of its own. Large local-global
#     overlap is then the expected outcome and not a sign of "local" structure; compare with the null and
#     look at modes beyond the first few and at the participation ratio.
#   * Windows of neighbouring centres share a large part of their atoms (the "shared" column of the printed
#     consistency table gives the fraction for each neighbour rank) and all frames: agreement between them
#     is a statement about the window boundary, not independent evidence. At most N/n windows are mutually
#     disjoint.
# The window covariance and its full eigen-decomposition:
#     C_w = X_w^T X_w / T   (K x K),    C_w v^w_n = lambda^w_n v^w_n,   lambda^w_1 >= ... >= lambda^w_K
# with X_w = the K columns of X belonging to S_w. The columns of X already have zero time
# mean, so C_w uses the same mean-subtraction as the full PCA (a warning is issued if u is
# not centred). Every window uses the same T = 101 frames.
#
# ------------------------------------------------------------
# EXACT RELATIONS BETWEEN LOCAL AND GLOBAL PCA
# ------------------------------------------------------------
# Same frames, same mean, same mass weights  =>  C_w is a PRINCIPAL SUBMATRIX of C:
#     C_w = E_w^T C E_w     (E_w selects the K coordinates of the window).
# Consequences (all verified numerically in scripts/test_pca_temporal_and_local.py):
#  (1) Cauchy interlacing:  lambda^w_n <= lambda_n   for every n (n-th largest of each).
#      A local eigenvalue is never larger than the global one of the same rank. This holds
#      for ANY data, so lambda^w_n / lambda_n <= 1 carries no information about the system.
#  (2) Trace:  sum_n lambda^w_n = sum over the window's coordinates of their variance.
#  (3) Decomposition. Let g_m = E_w^T v_m be global mode m restricted to the window (not a
#      unit vector: w_m = |g_m|^2 is the share of the mode's norm sitting on the window), and
#      O_nm = |v^w_n . g_m| / |g_m| the NORMALISED OVERLAP of local mode n with the restricted
#      global mode m. Because C_w = sum_m lambda_m g_m g_m^T,
#          lambda^w_n = sum_m lambda_m w_m O_nm^2 ,        sum_n O_nm^2 = 1  (n = 1..K),
#      the sum running over all T-1 non-zero global modes. So every local eigenvalue is an
#      exactly known mixture of global modes, each weighted by the variance lambda_m w_m it
#      puts on the window. `explained_fraction` is the part of lambda^w_n contributed by the
#      n_global_modes leading global modes (1 minus it comes from the rest).
#  (4) A restricted global mode g_m is an eigenvector of C_w only if C_{w,rest} v_rest = 0,
#      i.e. if the rest of the system does not feed the window. In general local modes are
#      NOT global modes, restricted global modes are NOT local modes, and a low overlap does
#      not show that the local mode is "new": it may be a mixture of many global modes.
#  (5) Local PCA brings NO new data. C_w uses K(K+1)/2 of the entries of C estimated from
#      the same 101 frames; what changes is the question that is asked of it (below).
#
# COMPARISON WITH THE FULL SPECTRUM (what the plots show)
#   lambda^w_n vs lambda_n (interlacing ratio); the fraction of the total variance in the
#   window; for each local mode the best-matching global mode, max_m O_nm (matching by
#   maximum overlap, not by index; `overlap_index` = O_nn for contrast); and eigenSPACES:
#   with U^w_M the K x M matrix of the M leading local eigenvectors and G_M = E_w^T [v_1..v_M]
#   the restricted leading global modes (orthonormalised, Q_M), the overlap
#       S_M = tr(P^w_M P^glob_M) / M = |U^w_M^T Q_M|_F^2 / M = (1/M) sum_i cos^2(theta_i)
#   (principal angles theta_i; projection matrices P = U U^T, never needed explicitly beyond
#   K x K). S_M is invariant to rotations inside the M-dimensional block, so it is the
#   appropriate measure when modes are nearly degenerate; it needs a gap after mode M to be
#   meaningful. Random M-dim subspaces of R^K give M/K. In addition
#       capture_ratio_M = tr(Q_M^T C_w Q_M) / sum_{n<=M} lambda^w_n   in [0, 1]
#   = how much of the window variance the restricted global leading modes explain, relative to
#   the best possible M-dimensional subspace (the local PCA itself). It needs no mode matching
#   at all. M = 1 reduces to the squared single-vector overlap of local mode 1 with global mode 1, O_11^2.
#
# CONSISTENCY IN OVERLAPPING WINDOWS. Window w and window w+j share a set of atoms; the two
# local eigenvectors are compared on those atoms only: A = E_shared^T v^w_n, normalised,
# overlap |A_n . B_m| / (|A_n||B_m|) with the eigenvectors of the other window, matched by
# maximum overlap; `weight` = |A_n|^2 = the share of the mode's norm on the shared atoms (a
# mode that sits on atoms the other window does not contain cannot be compared and is
# excluded below `min_shared_weight`). Eigenspaces: principal angles between the restricted,
# re-orthonormalised leading-M blocks. `track_modes` follows a mode from window to the next
# one (j = 1) while the matched overlap stays >= track_threshold and reports for how many
# consecutive windows it survives (chain length; greedy matching, so two modes may share a
# successor). Shifted windows differ by a few atoms and share 100% of the frames:
# agreement tests robustness against where the window boundary is put, not against
# sampling.
#
# ------------------------------------------------------------
# NUMERICAL RESOLUTION vs STATISTICAL ACCURACY
# ------------------------------------------------------------
# NUMERICAL: K = 45 < T-1 = 100, so C_w is full rank, the eigen-decomposition is exact to
#   rounding, and all K local modes exist as well-defined directions (except exact
#   degeneracies). The global PCA, in contrast, has rank <= 100 out of 20250 and every one
#   of its modes is a fit to 101 points. That is the sense in which local PCA "resolves more":
#   there is no rank-deficiency and no truncation.
# STATISTICAL: C_w is still estimated from T = 101 frames that are autocorrelated
#   (T_eff < T, pca_convergence.py PART 2). Even if the true covariance were the identity,
#   the sample eigenvalues of K coordinates from T_eff independent samples spread over
#   sigma^2 (1 -+ sqrt(K/T_eff))^2 (Marchenko-Pastur; see marchenko_pastur_edges(); a guide
#   only: coordinates here have unequal variances and are correlated in time). For K = 45,
#   T = 101: 0.11 ... 2.8 sigma^2, a factor ~25 of spectral range from pure sampling noise;
#   for T_eff < K the smallest sample eigenvalues collapse towards 0 and eigenvectors of
#   neighbouring eigenvalues are not distinguishable. A "numerically exact" local mode is
#   therefore still a noisy estimate, and selecting the top modes of a noisy matrix inflates
#   their eigenvalues (the same selection bias as in pca_reliability.py, smaller here because
#   K/T << P/T, but present).
# OVERLAPPING WINDOWS ARE NOT INDEPENDENT: windows w and w+1 share K-D coordinates and all
#   frames; every window is a block of the SAME matrix C. Averages over windows (and "N
#   windows agree") do not multiply the evidence by N; the number of statistically distinct
#   windows is at most N/n (disjoint ones), and even disjoint windows are coupled through the
#   global modes that run through them.
# NULL (n_null > 0): the whole analysis (local PCA, global PCA, all comparisons) on data in
#   which every coordinate is shifted cyclically in time by its own random amount
#   (pca_reliability._shift_columns: independent coordinates, same variance and
#   autocorrelation per coordinate). Window-averaged statistics of the real data are shown
#   against their distribution over null runs. Because C_w of the null is diagonal in
#   expectation, it shows what 101 frames of K coordinates produce WITHOUT collective motion
#   (spectral spread, chance overlaps with global modes, chance consistency across windows).
#   It does not model a collective-but-featureless spectrum.
#
# WHAT CAN AND CANNOT BE CONCLUDED
#   Can:  which global modes make up the fluctuations of a given atom group, and how much (3);
#         whether a local mode / leading local eigenspace is reproduced by the restricted
#         global leading modes (overlaps, capture ratio) beyond what the null gives;
#         whether the local spectrum departs from the sampling-noise range; whether a local
#         mode is stable when the window boundary moves by a few atoms; where along the atom
#         index the window spectra change.
#   Cannot: treat a local mode as a global mode (4) or as an independent confirmation of one
#         (5); infer statistical significance from the number of windows; read eigenvalue
#         ratios <= 1 as evidence (1); read participation ratios of windows of n atoms
#         against those of the full system without rescaling (PR here is in [1/n, 1];
#         `n_participating` = PR*n is the effective number of atoms); claim spatial locality
#         for atom-index windows unless the numbering is spatial; claim the modes are the
#         true dynamics of the glass (a single 101-frame trajectory; held-out frames or a
#         second trajectory are needed -- a split-half version of local PCA is the natural
#         next step and is not done here).
# =============================================================

import warnings

import numpy as np

from analysis.soft_modes.modes import participation_ratios
from analysis.soft_modes.pca_reliability import _flat, _pca, _shift_columns


# ------------------------------------------------------------
# Windows
# ------------------------------------------------------------

def index_windows(n_atoms, window_size=15, step=1, cover_all=True):
    """
    Atom-index sliding windows [s, s + window_size) for s = 0, step, 2*step, ...

    cover_all : with step > 1 the last regular window may stop before the last atom; True adds
        one more window at the end (n_atoms - window_size ... n_atoms - 1) so that every atom
        is in at least one window. With step <= window_size every atom is then covered.

    Returns (n_windows, window_size) int array of atom indices. Any other definition of a
    window (e.g. spatial neighbours) only has to produce an array of this shape.
    """
    if not 1 <= window_size <= n_atoms:
        raise ValueError(f"window_size must be in [1, {n_atoms}], got {window_size}")
    if step < 1:
        raise ValueError("step must be >= 1")
    if step > window_size:
        warnings.warn(f"step ({step}) > window_size ({window_size}): atoms between consecutive windows are not covered.")
    starts = list(range(0, n_atoms - window_size + 1, step))
    if cover_all and starts[-1] != n_atoms - window_size:
        starts.append(n_atoms - window_size)
    return np.asarray(starts)[:, None] + np.arange(window_size)[None, :]


def _periodic_tree(points, box):
    """cKDTree with periodic boundaries (orthorhombic box); points are wrapped into [0, box)."""
    from scipy.spatial import cKDTree
    box = np.asarray(box, dtype=float)
    p = np.mod(np.asarray(points, dtype=float), box)
    p = np.where(p >= box, 0.0, p)                  # np.mod can return exactly `box` for tiny negative input
    return cKDTree(p, boxsize=box), p


def _check_box(box, n_dim):
    box = np.asarray(box, dtype=float)
    if box.shape != (n_dim,) or np.any(box <= 0):
        raise ValueError(f"box must be the {n_dim} edge lengths of an orthorhombic cell, got {box!r}")
    return box


def neighborhood_windows(mean_positions, box, window_size=15, step=1, centers=None):
    """
    Spatial windows: for every centre atom c, the `window_size` atoms closest to c (c itself first,
    then its window_size - 1 nearest neighbours in order of distance), by minimum-image distance
    between the time-averaged positions in the orthorhombic periodic box.

    mean_positions : (N, D) time-averaged positions (the reference of the displacements; need not be
        wrapped into the box). box : (D,) cell edge lengths.
    window_size : atoms per window INCLUDING the centre. A fixed number of atoms (not a fixed radius)
        keeps K = window_size * D the same for all windows; the radius then varies with the local density.
    step : every `step`-th atom (0, step, 2 step, ...) is a centre; step 1 = every atom.
    centers : explicit centre atom indices instead (overrides step).

    Returns (windows (n_windows, window_size) int, centers (n_windows,) int).
    """
    pos = np.asarray(mean_positions, dtype=float)
    N, D = pos.shape
    box = _check_box(box, D)
    if not 1 <= window_size <= N:
        raise ValueError(f"window_size must be in [1, {N}], got {window_size}")
    if step < 1:
        raise ValueError("step must be >= 1")
    centers = np.arange(0, N, step) if centers is None else np.asarray(centers, dtype=int)
    tree, wrapped = _periodic_tree(pos, box)
    _, idx = tree.query(wrapped[centers], k=list(range(1, window_size + 1)))     # list -> always 2-D
    idx = np.asarray(idx, dtype=int).reshape(len(centers), window_size)
    for r in np.where(idx[:, 0] != centers)[0]:      # exact ties at distance 0: put the centre first
        row = [centers[r]] + [a for a in idx[r] if a != centers[r]]
        idx[r] = row[:window_size]
    return idx, centers


def window_partners(center_positions, box, max_rank=10):
    """
    For every window, the other windows ordered by the distance between the CENTRES (minimum image,
    periodic box): partners[w, j - 1] = index of the j-th nearest other window. This replaces
    "window w + j" of index windows as the notion of an overlapping neighbour window.

    center_positions : (n_windows, D) time-averaged positions of the centre atoms. max_rank : ranks 1..max_rank.
    Returns (n_windows, min(max_rank, n_windows - 1)) int array.
    """
    c = np.asarray(center_positions, dtype=float)
    n_win, D = c.shape
    box = _check_box(box, D)
    R = min(int(max_rank), n_win - 1)
    if R < 1:
        raise ValueError("need at least two windows and max_rank >= 1")
    tree, wrapped = _periodic_tree(c, box)
    _, idx = tree.query(wrapped, k=list(range(1, R + 2)))
    idx = np.asarray(idx, dtype=int).reshape(n_win, R + 1)
    out = np.empty((n_win, R), dtype=int)
    for w in range(n_win):
        row = [a for a in idx[w] if a != w]          # drop the window itself (normally first)
        out[w] = row[:R]
    return out


def window_columns(windows, n_dim):
    """(n_windows, n*D) column indices of X (atom-major: atom*D + d) for each window."""
    windows = np.asarray(windows, dtype=int)
    return (windows[:, :, None] * n_dim + np.arange(n_dim)[None, None, :]).reshape(len(windows), -1)


def window_spatial_extent(mean_positions, windows, box=None):
    """
    Radius of gyration (Å) of the atoms of every window, from the time-averaged positions
    (N, D); with `box` (D,) the minimum-image convention is applied about the first atom.
    Small and similar to a random group of n atoms' value tells you whether an index
    window is a compact neighbourhood or an arbitrary group.
    """
    pos = np.asarray(mean_positions, dtype=float)[np.asarray(windows, dtype=int)]   # (W, n, D)
    if box is not None:
        d = pos - pos[:, :1, :]
        pos = pos[:, :1, :] + d - np.asarray(box) * np.round(d / np.asarray(box))
    return np.sqrt(((pos - pos.mean(axis=1, keepdims=True)) ** 2).sum(axis=2).mean(axis=1))


def marchenko_pastur_edges(n_coords, n_samples):
    """
    Edges (lower, upper) of the sample-eigenvalue spectrum, in units of the variance sigma^2,
    for n_coords independent equal-variance Gaussian coordinates and n_samples independent
    samples: (1 -+ sqrt(c))^2, c = n_coords / n_samples. For c > 1 the lower edge is 0 (only
    n_samples eigenvalues are non-zero). A guide for "how wide is a spectrum of pure noise".
    """
    c = n_coords / float(n_samples)
    return (1.0 - np.sqrt(c)) ** 2 if c < 1 else 0.0, (1.0 + np.sqrt(c)) ** 2


# ------------------------------------------------------------
# Local PCA of every window
# ------------------------------------------------------------

def _check_centred(X):
    scale = max(float(np.abs(X).max()), 1e-300)
    if float(np.abs(X.mean(axis=0)).max()) > 1e-6 * scale:
        warnings.warn("u is not centred on its time mean: window covariances are not the "
                      "principal submatrices of the full-system covariance (compute_displacements() "
                      "with reference=None gives the convention of the existing PCA).")


def local_pca(u, masses=None, windows=None, window_size=15, step=1,
              store_eigenvectors=True, chunk=256, position=None):
    """
    Covariance matrix and FULL eigen-decomposition of every window (see the file header).

    Parameters
    ----------
    u : (T, N, D) centred displacements (the array given to modes_from_displacements()).
    masses : (N,) or None (mass weighting exactly as modes_from_displacements()).
    windows : (n_windows, n) atom indices, or None for index_windows(N, window_size, step).
    store_eigenvectors : keep the (n_windows, K, K) eigenvector array (K = n*D; 109 MB for
        6736 windows of 15 atoms in float64).
    chunk : windows processed per batch (memory only).
    position : (n_windows,) x-axis value of every window (default: mean atom index of the window; the
        centre atom index for neighbourhood windows).

    Returns dict:
      windows (n_win, n), columns (n_win, K), position (n_win,)   mean atom index of the window (or `position`)
      eigenvalues (n_win, K)           descending, softest first; amu*A^2 (A^2 if masses=None)
      eigenvectors_mw (n_win, K, K)    columns = modes, unit, MASS-WEIGHTED space (optional)
      participation (n_win, K)         PR = 1 / (n sum_i p_i^2), real-space vectors, as modes.py
      n_participating (n_win, K)       PR * n = effective number of atoms
      total_variance (n_win,)          trace C_w
      n_resolved                       min(K, T-1): modes beyond it have zero eigenvalue (rank)
      n_frames, n_atoms_window, n_dim, n_coords
    """
    u = np.asarray(u, dtype=float)
    T, N, D = u.shape
    X = _flat(u, masses)
    _check_centred(X)
    if windows is None:
        windows = index_windows(N, window_size, step)
    windows = np.asarray(windows, dtype=int)
    if windows.ndim != 2:
        raise ValueError("windows must be a (n_windows, n_atoms_per_window) integer array")
    n_win, n = windows.shape
    K = n * D
    if K > T - 1:
        warnings.warn(f"a window has {K} coordinates but the {T} frames give rank <= {T - 1}: "
                      f"modes {T - 1}.. have zero eigenvalue and arbitrary eigenvectors.")
    cols = window_columns(windows, D)
    sqrt_m = np.ones(N * D) if masses is None else np.repeat(np.sqrt(np.asarray(masses, float)), D)

    eig = np.empty((n_win, K))
    pr = np.empty((n_win, K))
    V_all = np.empty((n_win, K, K)) if store_eigenvectors else None
    for s in range(0, n_win, chunk):
        c = cols[s:s + chunk]
        Xw = np.moveaxis(X[:, c], 1, 0)                           # (c, T, K)
        C = np.matmul(Xw.transpose(0, 2, 1), Xw) / T              # (c, K, K)
        lam, V = np.linalg.eigh(0.5 * (C + C.transpose(0, 2, 1)))
        lam, V = lam[:, ::-1], V[:, :, ::-1]                      # descending
        eig[s:s + chunk] = lam
        # participation ratio on the real-space vectors (modes.py convention)
        Vr = V / sqrt_m[c][:, :, None]
        Vr = Vr / np.linalg.norm(Vr, axis=1, keepdims=True)
        p = (Vr.reshape(len(c), n, D, K) ** 2).sum(axis=2)        # (c, n, K), sums to 1 over atoms
        pr[s:s + chunk] = 1.0 / (n * (p ** 2).sum(axis=1))
        if store_eigenvectors:
            V_all[s:s + chunk] = V
    eig = np.maximum(eig, 0.0)                                    # rounding: tiny negatives

    out = {
        "windows": windows, "columns": cols,
        "position": windows.mean(axis=1) if position is None else np.asarray(position, dtype=float),
        "eigenvalues": eig, "participation": pr, "n_participating": pr * n,
        "total_variance": eig.sum(axis=1), "n_resolved": min(K, T - 1),
        "n_frames": T, "n_atoms_window": n, "n_dim": D, "n_coords": K,
    }
    if store_eigenvectors:
        out["eigenvectors_mw"] = V_all
    return out


# ------------------------------------------------------------
# Full-system reference
# ------------------------------------------------------------

def global_reference(u, masses=None, n_modes=20):
    """
    Full-system PCA in the same convention (SVD of X; identical to modes_from_displacements()
    up to the 1/sqrt(m) rescaling of the eigenvectors).

    Returns dict: eigenvalues (T-1,) all non-zero ones; eigenvectors (P, n_modes) unit vectors
    in the mass-weighted space; participation (n_modes,) of the real-space vectors (PR over all N
    atoms, modes.participation_ratios); n_participating = PR * N; n_atoms, n_dim, n_frames.
    """
    u = np.asarray(u, dtype=float)
    T, N, D = u.shape
    X = _flat(u, masses)
    k = max(1, min(n_modes, T - 1))
    lam, V = _pca(X, k)
    sqrt_m = np.ones(N * D) if masses is None else np.repeat(np.sqrt(np.asarray(masses, float)), D)
    Vr = V / sqrt_m[:, None]
    Vr = Vr / np.linalg.norm(Vr, axis=0, keepdims=True)
    pr = participation_ratios(Vr.reshape(N, D, k))
    return {"eigenvalues": np.maximum(lam[:T - 1], 0.0), "eigenvectors": V, "participation": pr,
            "n_participating": pr * N, "n_atoms": N, "n_dim": D, "n_frames": T}


# ------------------------------------------------------------
# Local vs global
# ------------------------------------------------------------

def _orth_batch(G, tol=1e-10):
    """Orthonormal basis of the columns of every (K, M) matrix in the stack G; dependent columns are zeroed."""
    U, s, _ = np.linalg.svd(G, full_matrices=False)
    keep = s > tol * np.maximum(s[:, :1], 1e-300)
    return U * keep[:, None, :], keep.sum(axis=1)


def compare_local_to_global(local, glob, subspace_dims=(1, 3, 5, 10), n_compare_modes=10, chunk=256):
    """
    Local eigenvalues, eigenvectors and eigenspaces against the full-system PCA (header: LOCAL vs GLOBAL).

    local : local_pca(..., store_eigenvectors=True).  glob : global_reference(...).
    subspace_dims : M for the eigenspace comparison (those larger than min(K, number of global
        modes) are dropped).  n_compare_modes : local modes for which the full overlap matrix is stored.

    Returns dict (n_win windows, K coordinates, Kg global modes kept, nc = n_compare_modes):
      lambda_ratio (n_win, Kc)   lambda^w_n / lambda_n, n < Kc = min(K, T-1); <= 1 (interlacing)
      interlacing_violation      max over windows and n of lambda^w_n/lambda_n - 1 (should be <= ~1e-10)
      variance_share (n_win,)    trace C_w / trace C
      global_weight (n_win, Kg)  w_m = |E_w^T v_m|^2 ; uniform modes give n/N
      overlap_matrix (n_win, nc, Kg)   O_nm, normalised overlaps |v^w_n . g_m| / |g_m|
      overlap_best (n_win, K)    max_m O_nm: best-matching global mode of every local mode
      best_global_index (n_win, K)     the m that gives it
      overlap_index (n_win, nc)  O_nn (match by index, for contrast)
      explained_fraction (n_win, nc)   fraction of lambda^w_n supplied by the Kg leading global modes
      subspace_dims (Md,)
      subspace_overlap (n_win, Md)     S_M ; subspace_random (Md,) = M/K
      subspace_cos (list of (n_win, M) arrays)  cosines of the principal angles
      capture_ratio (n_win, Md)
    """
    if "eigenvectors_mw" not in local:
        raise ValueError("compare_local_to_global needs local_pca(..., store_eigenvectors=True)")
    V, lam, cols = local["eigenvectors_mw"], local["eigenvalues"], local["columns"]
    n_win, K, _ = V.shape
    Vg, lam_g = glob["eigenvectors"], glob["eigenvalues"]
    Kg = Vg.shape[1]
    nc = min(n_compare_modes, K)
    dims = np.array([m for m in subspace_dims if 1 <= m <= min(K, Kg)], dtype=int)
    Kc = min(K, len(lam_g))

    ratio = np.full((n_win, Kc), np.nan)
    pos = lam_g[:Kc] > 0
    ratio[:, pos] = lam[:, :Kc][:, pos] / lam_g[:Kc][pos]
    O_mat = np.empty((n_win, nc, Kg))
    O_best = np.empty((n_win, K))
    best_idx = np.empty((n_win, K), dtype=int)
    O_index = np.full((n_win, nc), np.nan)
    explained = np.full((n_win, nc), np.nan)
    w_g = np.empty((n_win, Kg))
    S = np.full((n_win, len(dims)), np.nan)
    cap = np.full((n_win, len(dims)), np.nan)
    cos_all = [np.full((n_win, m), np.nan) for m in dims]

    for s in range(0, n_win, chunk):
        e = min(s + chunk, n_win)
        Vw, lw = V[s:e], lam[s:e]
        G = Vg[cols[s:e]]                                         # (c, K, Kg) restricted global modes
        W = np.matmul(Vw.transpose(0, 2, 1), G)                   # (c, K, Kg): v^w_n . g_m
        wg = (G ** 2).sum(axis=1)                                 # (c, Kg)
        w_g[s:e] = wg
        O = np.abs(W) / np.sqrt(np.maximum(wg, 1e-300))[:, None, :]
        O_best[s:e] = O.max(axis=2)
        best_idx[s:e] = O.argmax(axis=2)
        O_mat[s:e] = O[:, :nc, :]
        mi = min(nc, Kg)
        O_index[s:e, :mi] = O[:, np.arange(mi), np.arange(mi)]
        with np.errstate(divide="ignore", invalid="ignore"):
            explained[s:e] = ((W[:, :nc, :] ** 2) * lam_g[None, None, :Kg]).sum(axis=2) / lw[:, :nc]
        cs = np.cumsum(lw, axis=1)
        for j, M in enumerate(dims):
            Q, m_eff = _orth_batch(G[:, :, :M])
            B = np.matmul(Vw.transpose(0, 2, 1), Q)               # (c, K, M)
            m_eff = np.maximum(m_eff, 1)
            S[s:e, j] = (B[:, :M, :] ** 2).sum(axis=(1, 2)) / m_eff
            cos_all[j][s:e] = np.linalg.svd(B[:, :M, :], compute_uv=False)
            with np.errstate(divide="ignore", invalid="ignore"):
                cap[s:e, j] = (lw[:, :, None] * B ** 2).sum(axis=(1, 2)) / cs[np.arange(e - s), m_eff - 1]
    explained[~np.isfinite(explained)] = np.nan
    cap[~np.isfinite(cap)] = np.nan
    with np.errstate(invalid="ignore"):
        viol = float(np.nanmax(ratio) - 1.0)
    return {
        "lambda_ratio": ratio, "interlacing_violation": viol,
        "variance_share": local["total_variance"] / max(float(lam_g.sum()), 1e-300),
        "global_weight": w_g, "overlap_matrix": O_mat, "overlap_best": O_best,
        "best_global_index": best_idx, "overlap_index": O_index, "explained_fraction": explained,
        "subspace_dims": dims, "subspace_overlap": S, "subspace_random": dims / float(K),
        "subspace_cos": cos_all, "capture_ratio": cap,
    }


# ------------------------------------------------------------
# Consistency between overlapping windows
# ------------------------------------------------------------

def _orth(A, tol=1e-10):
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return U[:, :0]
    return U[:, s > tol * s[0]]


def _shared_rows(idx1, idx2, n_dim):
    """Coordinate rows of the atoms two windows share, in the order of both windows' own vectors."""
    common, i1, i2 = np.intersect1d(idx1, idx2, return_indices=True)
    if common.size == 0:
        return None
    d = np.arange(n_dim)[None, :]
    return (i1[:, None] * n_dim + d).ravel(), (i2[:, None] * n_dim + d).ravel(), common.size


def _restricted_overlaps(V1, V2, r1, r2):
    """Normalised overlaps (K1, K2) of the columns of V1, V2 restricted to the shared rows, and the weights |A1_n|^2."""
    A1, A2 = V1[r1], V2[r2]
    n1, n2 = np.linalg.norm(A1, axis=0), np.linalg.norm(A2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        O = np.abs(A1.T @ A2) / (n1[:, None] * n2[None, :])
    return np.nan_to_num(O), n1 ** 2


def _partner_table(n_win, offsets, partners):
    """
    (offsets kept, (n_off, n_win) index of the partner window or -1). Without `partners` (index windows)
    the partner of window w at offset j is w + j; with `partners` (n_win, R) it is the j-th nearest
    window centre, partners[w, j - 1].
    """
    if partners is None:
        offs = np.array([j for j in offsets if 1 <= j < n_win], dtype=int)
        P = np.full((len(offs), n_win), -1, dtype=int)
        for o, j in enumerate(offs):
            P[o, :n_win - j] = np.arange(j, n_win)
        return offs, P
    partners = np.asarray(partners, dtype=int)
    offs = np.array([j for j in offsets if 1 <= j <= partners.shape[1]], dtype=int)
    return offs, partners[:, offs - 1].T.copy()


def overlap_consistency(local, offsets=(1, 2, 4, 7, 10), n_modes=10, subspace_dims=(1, 3, 5, 10),
                        partners=None):
    """
    Are the local modes of window w found again in an overlapping window? The partner of window w at
    offset j is window w + j (index windows: j atoms for step 1) or, if `partners` is given
    (window_partners(); neighbourhood windows), its j-th nearest window centre. Compared on the atoms
    the two windows share.

    Returns dict (n_off offsets, n_win windows; entry [o, w] belongs to the pair (w, partner of w at
    offsets[o]), NaN where that window does not exist or the windows share no atoms):
      offsets, shared_fraction (n_off,)          mean share of a window's atoms that is common
      matched_overlap (n_off, n_win, n_modes)    max over the modes of window w+j of the restricted overlap
      matched_index (n_off, n_win, n_modes)      the mode of window w+j that matches (an index shift
                                                  means the mode changed rank)
      index_overlap (n_off, n_win, n_modes)      overlap of mode n with mode n of the other window
      weight (n_off, n_win, n_modes)             share of the norm of mode n (window w) on the shared atoms
      subspace_overlap (n_off, n_win, Md)        leading-M eigenspaces, restricted and re-orthonormalised
      subspace_dims (Md,)
    """
    V, windows, D = local["eigenvectors_mw"], local["windows"], local["n_dim"]
    n_win, K, _ = V.shape
    nm = min(n_modes, K)
    dims = np.array([m for m in subspace_dims if 1 <= m <= K], dtype=int)
    offsets, P = _partner_table(n_win, offsets, partners)
    shape = (len(offsets), n_win)
    matched = np.full(shape + (nm,), np.nan)
    midx = np.full(shape + (nm,), -1, dtype=int)
    idx_ov = np.full(shape + (nm,), np.nan)
    weight = np.full(shape + (nm,), np.nan)
    sub = np.full(shape + (len(dims),), np.nan)
    shared = np.zeros(len(offsets))
    for o, j in enumerate(offsets):
        fr = []
        for w in range(n_win):
            p = P[o, w]
            if p < 0:
                continue
            sr = _shared_rows(windows[w], windows[p], D)
            if sr is None:
                continue
            r1, r2, n_common = sr
            fr.append(n_common / windows.shape[1])
            O, wt = _restricted_overlaps(V[w], V[p], r1, r2)
            matched[o, w] = O[:nm].max(axis=1)
            midx[o, w] = O[:nm].argmax(axis=1)
            idx_ov[o, w] = np.diag(O)[:nm]
            weight[o, w] = wt[:nm]
            for i, M in enumerate(dims):
                Q1, Q2 = _orth(V[w][r1][:, :M]), _orth(V[p][r2][:, :M])
                if Q1.shape[1] and Q2.shape[1]:
                    sub[o, w, i] = np.linalg.norm(Q1.T @ Q2) ** 2 / min(Q1.shape[1], Q2.shape[1])
        shared[o] = np.mean(fr) if fr else np.nan
    return {"offsets": offsets, "shared_fraction": shared, "matched_overlap": matched,
            "matched_index": midx, "index_overlap": idx_ov, "weight": weight,
            "subspace_overlap": sub, "subspace_dims": dims}


def track_modes(local, n_modes=10, threshold=0.8, min_weight=0.5, partners=None, max_rank=None):
    """
    Index windows (partners=None): follow every local mode from window to window (j = 1). The mode of
    window w is matched to the mode of window w+1 of maximum restricted overlap; the chain continues
    while that overlap is >= threshold and the mode has >= min_weight of its norm on the shared atoms.
    Chains are truncated at the last window, and for step > 1 a window is a shift of `step` atoms.

    Neighbourhood windows (partners = window_partners()): windows have no linear order, so there is no
    chain. Instead the mode of window w is looked for in each of its max_rank nearest windows (default:
    all of `partners`), with the same overlap and weight criteria, and the count is reported.

    Returns dict: chain_length (n_win, n_modes) = number of consecutive windows (>= 1, the start window
    included) over which mode n of window w is found again (index windows), or 1 + the number of the
    max_rank nearest windows in which it is found again (neighbourhood windows: not consecutive, not a
    chain; "kind" says which); kind ("chain" / "neighbours"), max_rank (None for chains), threshold, min_weight.
    """
    V, windows, D = local["eigenvectors_mw"], local["windows"], local["n_dim"]
    n_win, K, _ = V.shape
    if partners is not None:
        R = partners.shape[1] if max_rank is None else min(int(max_rank), partners.shape[1])
        count = np.zeros((n_win, K), dtype=int)
        for w in range(n_win):
            for j in range(R):
                p = partners[w, j]
                sr = _shared_rows(windows[w], windows[p], D)
                if sr is None:
                    continue
                O, wt = _restricted_overlaps(V[w], V[p], sr[0], sr[1])
                count[w] += (O.max(axis=1) >= threshold) & (wt >= min_weight)
        return {"chain_length": 1 + count[:, :min(n_modes, K)], "kind": "neighbours", "max_rank": R,
                "threshold": threshold, "min_weight": min_weight}
    nxt = np.zeros((n_win, K), dtype=int)
    ok = np.zeros((n_win, K), dtype=bool)
    for w in range(n_win - 1):
        sr = _shared_rows(windows[w], windows[w + 1], D)
        if sr is None:
            continue
        O, wt = _restricted_overlaps(V[w], V[w + 1], sr[0], sr[1])
        nxt[w] = O.argmax(axis=1)
        ok[w] = (O.max(axis=1) >= threshold) & (wt >= min_weight)
    length = np.ones((n_win, K), dtype=int)
    for w in range(n_win - 2, -1, -1):
        length[w] = 1 + np.where(ok[w], length[w + 1][nxt[w]], 0)
    return {"chain_length": length[:, :min(n_modes, K)], "kind": "chain", "max_rank": None,
            "threshold": threshold, "min_weight": min_weight}


# ------------------------------------------------------------
# Window-averaged summary (observed and null runs use the same function)
# ------------------------------------------------------------

def _nanmean(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def summarize(local, comp, cons, trk, min_weight=0.5):
    """Window-averaged curves used to compare the data with the shifted-column null."""
    nc = comp["overlap_matrix"].shape[1]
    w_ok = np.where(cons["weight"] >= min_weight, cons["matched_overlap"], np.nan)    # (off, win, nm)
    return {
        "spectrum_fraction": (local["eigenvalues"] / np.maximum(local["total_variance"], 1e-300)[:, None]).mean(axis=0),
        "participation": local["participation"].mean(axis=0),
        "overlap_best": comp["overlap_best"][:, :nc].mean(axis=0),
        "explained_fraction": _nanmean(comp["explained_fraction"]),
        "subspace_overlap": _nanmean(comp["subspace_overlap"]),
        "capture_ratio": _nanmean(comp["capture_ratio"]),
        "consistency_matched": _nanmean(w_ok, axis=1),
        "consistency_subspace": _nanmean(cons["subspace_overlap"], axis=1),
        "chain_length": trk["chain_length"].mean(axis=0),
    }


def local_pca_band_statistics(res, q=(5, 50, 95)):
    """
    Percentiles over the windows (and the null's median / upper quantile) of the quantities that
    visualization.plot.plot_local_* draw, so that the plot functions only draw.

    Returns dict: q, and for each key in eigenvalues, spectrum_fraction, n_participating, lambda_ratio,
    overlap_best, explained_fraction, subspace_overlap, capture_ratio an array (len(q), ...) of
    the q-th percentiles over windows (NaN ignored); if res["null"] exists also
    null_median[key], null_high[key] for the keys of res["summary"] (median, and the stored quantile, over
    null runs). Percentiles over windows describe the spread along the atom index; they are not
    confidence intervals (windows overlap, see the file header).
    """
    loc, comp = res["local"], res["comparison"]
    lam = loc["eigenvalues"]

    def pct(a):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanpercentile(a, q, axis=0)

    out = {"q": tuple(q),
           "eigenvalues": pct(lam),
           "spectrum_fraction": pct(lam / np.maximum(loc["total_variance"], 1e-300)[:, None]),
           "n_participating": pct(loc["n_participating"]),
           "lambda_ratio": pct(comp["lambda_ratio"]),
           "overlap_best": pct(comp["overlap_best"]),
           "explained_fraction": pct(comp["explained_fraction"]),
           "subspace_overlap": pct(comp["subspace_overlap"]),
           "capture_ratio": pct(comp["capture_ratio"])}
    null = res.get("null")
    if null is not None:
        keys = [k for k in res["summary"]]
        out["null_median"] = {k: np.nanmedian(null[k], axis=0) for k in keys}
        out["null_high"] = {k: np.nanquantile(null[k], null["quantile"], axis=0) for k in keys}
    return out


def save_local_pca(res, path, eigenvectors_dtype=np.float32):
    """
    Write the result of local_pca_analysis() to a compressed .npz (nested keys joined with "/", as
    pca_convergence.save_results). The window eigenvectors (mass-weighted unit vectors) are cast to
    `eigenvectors_dtype` (float32 halves the file; None drops them).
    """
    from analysis.soft_modes.pca_convergence import save_results
    out = dict(res)
    loc = dict(res["local"])
    if "eigenvectors_mw" in loc:
        if eigenvectors_dtype is None:
            del loc["eigenvectors_mw"]
        else:
            loc["eigenvectors_mw"] = loc["eigenvectors_mw"].astype(eigenvectors_dtype)
    out["local"] = loc
    out["comparison"] = {k: v for k, v in res["comparison"].items() if k != "subspace_cos"}
    save_results(out, path)


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------

def _analyze(u, masses, windows, cfg, partners=None, position=None):
    local = local_pca(u, masses, windows=windows, store_eigenvectors=True, position=position)
    glob = global_reference(u, masses, cfg["n_global_modes"])
    comp = compare_local_to_global(local, glob, cfg["subspace_dims"], cfg["n_compare_modes"])
    cons = overlap_consistency(local, cfg["consistency_offsets"], cfg["n_consistency_modes"],
                               cfg["subspace_dims"], partners=partners)
    trk = track_modes(local, cfg["n_consistency_modes"], cfg["track_threshold"], cfg["min_shared_weight"],
                      partners=partners, max_rank=max(cfg["consistency_offsets"]))
    return local, glob, comp, cons, trk


def local_pca_analysis(u, masses=None, window_size=15, step=1, windows=None,
                       window_mode="index", mean_positions=None, box=None, partners=None,
                       n_global_modes=20, subspace_dims=(1, 3, 5, 10), n_compare_modes=10,
                       consistency_offsets=(1, 2, 4, 7, 10), n_consistency_modes=None,
                       track_threshold=0.8, min_shared_weight=0.5, t_eff=None,
                       n_null=0, null_quantile=0.95, seed=0, keep_eigenvectors=True):
    """
    Moving-window local PCA next to the full-system PCA (see the file header for every definition).

    Parameters
    ----------
    u, masses : as in local_pca().
    window_size, step : atoms per window (15) and shift of the window / spacing of the centre atoms (1).
    window_mode : "index" (atoms s..s+n-1; default of this function) or "neighborhood" (centre atom + its
        n - 1 nearest neighbours; needs mean_positions (N, D) = time-averaged positions, the second return
        value of utils.compute_displacements, and box (D,) = orthorhombic cell edges). See the file header.
    windows, partners : alternatively any (n_win, n) index array (window_mode is then ignored), and for it
        the (n_win, R) table of "overlapping neighbour" windows (window_partners(); default: window w + j).
    n_global_modes : leading global modes compared with (default 20; more modes make a larger
        "dictionary" and larger chance overlaps -- see the null).
    subspace_dims : dimensions M of the eigenspaces compared. n_compare_modes: local modes with
        stored overlap matrix / explained fraction. consistency_offsets : window shifts j (index windows)
        or neighbour ranks j = j-th nearest window centre (neighbourhood windows).
    n_consistency_modes : local modes followed across windows (default n_compare_modes).
    track_threshold, min_shared_weight : see track_modes().
    t_eff : effective number of independent frames (pca_convergence.pca_temporal_analysis),
        only used to report the Marchenko-Pastur range for context.
    n_null, null_quantile, seed : number of shifted-column null runs of the whole analysis and the
        quantile stored (0 = none). The null re-runs everything: about as long as the analysis, each.
    keep_eigenvectors : keep local["eigenvectors_mw"] in the result.

    Returns dict: windows, position, windows_info (mode, centers, center_positions, partners), local, global,
    comparison, consistency, tracking, summary,
    null (dict of (n_null, ...) arrays with the keys of `summary` + quantile, or None), params,
    statistics (n_frames, n_coords, coords_per_frame, mp_edges, mp_edges_teff, window geometry).
    """
    u = np.asarray(u, dtype=float)
    T, N, D = u.shape
    cfg = dict(n_global_modes=n_global_modes, subspace_dims=tuple(subspace_dims),
               n_compare_modes=n_compare_modes,
               consistency_offsets=tuple(consistency_offsets),
               n_consistency_modes=n_compare_modes if n_consistency_modes is None else n_consistency_modes,
               track_threshold=track_threshold, min_shared_weight=min_shared_weight)
    centers = None
    mode = "custom"
    if windows is None:
        mode = window_mode
        if window_mode == "index":
            windows = index_windows(N, window_size, step)
        elif window_mode == "neighborhood":
            if mean_positions is None or box is None:
                raise ValueError("window_mode='neighborhood' needs mean_positions (N, D) and box (D,)")
            if len(mean_positions) != N:
                raise ValueError(f"mean_positions has {len(mean_positions)} atoms, u has {N}")
            windows, centers = neighborhood_windows(mean_positions, box, window_size, step)
            partners = window_partners(np.asarray(mean_positions, float)[centers], box,
                                       max(consistency_offsets))
        else:
            raise ValueError(f"window_mode must be 'index' or 'neighborhood', got {window_mode!r}")
    windows = np.asarray(windows, dtype=int)
    position = None if centers is None else centers          # x axis of the heatmaps: centre atom index

    local, glob, comp, cons, trk = _analyze(u, masses, windows, cfg, partners, position)
    summary = summarize(local, comp, cons, trk, min_shared_weight)
    K = local["n_coords"]

    null = None
    if n_null > 0:
        rng = np.random.default_rng(seed)
        flat = u.reshape(T, N * D)
        sums = []
        for _ in range(n_null):
            un = _shift_columns(flat, rng).reshape(T, N, D)
            lo, _, co, cs, tr = _analyze(un, masses, windows, cfg, partners, position)
            sums.append(summarize(lo, co, cs, tr, min_shared_weight))
        null = {k: np.stack([s[k] for s in sums]) for k in summary}
        null["quantile"] = null_quantile
        null["n_null"] = n_null

    if not keep_eigenvectors:
        local = {k: v for k, v in local.items() if k != "eigenvectors_mw"}
    stats = {"n_frames": T, "n_coords": K, "coords_per_frame": K / T,
             "mp_edges": marchenko_pastur_edges(K, T),
             "mp_edges_teff": None if t_eff is None else marchenko_pastur_edges(K, t_eff),
             "t_eff": t_eff, "n_windows": len(windows), "n_atoms": N,
             "n_windows_disjoint": N // windows.shape[1]}
    info = {"mode": mode, "centers": centers, "partners": partners,
            "center_positions": None if centers is None else np.asarray(mean_positions, float)[centers]}
    return {"windows": windows, "position": local["position"], "windows_info": info, "local": local,
            "global": glob, "comparison": comp, "consistency": cons, "tracking": trk, "summary": summary,
            "null": null, "statistics": stats,
            "params": dict(cfg, window_size=windows.shape[1], step=step, seed=seed, window_mode=mode,
                           mass_weighted=masses is not None)}


def print_local_pca(res):
    """Summary table of local_pca_analysis()."""
    st, p, S, comp, cons = res["statistics"], res["params"], res["summary"], res["comparison"], res["consistency"]
    null = res["null"]
    q = int(round(100 * null["quantile"])) if null is not None else None
    lo, hi = st["mp_edges"]
    K = st["n_coords"]
    spatial = p.get("window_mode") == "neighborhood"
    kind = (f"neighbourhood windows (centre atom + {p['window_size'] - 1} nearest neighbours, every {p['step']}. atom a centre)"
            if spatial else f"index windows of {p['window_size']} atoms, step {p['step']}")
    print(f"\n  Local PCA  ({st['n_windows']} {kind}, {K} coordinates, {st['n_frames']} frames)")
    print(f"  {'coordinates / frames':<46}{K} / {st['n_frames']}   (ratio {st['coords_per_frame']:.2f})")
    print(f"  {'white-noise eigenvalue range, T frames':<46}{lo:.2f} ... {hi:.2f} x variance")
    if st["mp_edges_teff"] is not None:
        print(f"  {'white-noise eigenvalue range, T_eff = ' + format(st['t_eff'], '.1f'):<46}"
              f"{st['mp_edges_teff'][0]:.2f} ... {st['mp_edges_teff'][1]:.2f} x variance")
    print(f"  {'statistically distinct (disjoint) windows':<46}{st['n_windows_disjoint']}  (not {st['n_windows']}: windows overlap and share all frames)")
    print(f"  {'interlacing  max(lambda_w/lambda) - 1':<46}{comp['interlacing_violation']:.3g}   (<= 0 up to rounding, any data)")
    print(f"  {'variance share of a window (median)':<46}{np.nanmedian(comp['variance_share']):.2e}   (uniform: {p['window_size'] / st['n_atoms']:.2e})")
    nc = comp["overlap_matrix"].shape[1]
    head = f"\n  local mode vs {res['global']['eigenvectors'].shape[1]} leading global modes (window medians; null = {q}% of the window average)" \
        if null is not None else f"\n  local mode vs {res['global']['eigenvectors'].shape[1]} leading global modes (window medians)"
    print(head + "; 'chain' / 'found' = mean over windows, like its null")
    lab = "found" if spatial else "chain"
    cols = f"  {'n':>3}{'lambda_w':>11}{'n_part':>8}{'best ov.':>10}{'explained':>11}{lab:>8}"
    if null is not None:
        cols += f"{'| null ov.':>12}{'null expl.':>12}{'null ' + lab:>12}"
    print(cols)
    lam = np.median(res["local"]["eigenvalues"], axis=0)
    npart = np.median(res["local"]["n_participating"], axis=0)
    ov = np.median(comp["overlap_best"], axis=0)
    ex = np.nanmedian(comp["explained_fraction"], axis=0)
    ch = np.mean(res["tracking"]["chain_length"], axis=0)          # mean over windows: the null is a mean too
    for n in sorted({0, 1, 2, 4, nc - 1}):
        if n >= nc:
            continue
        row = f"  {n:>3}{lam[n]:>11.3g}{npart[n]:>8.1f}{ov[n]:>10.2f}{ex[n]:>11.2f}{ch[n]:>8.1f}"
        if null is not None:
            row += (f"{np.quantile(null['overlap_best'][:, n], null['quantile']):>12.2f}"
                    f"{np.quantile(null['explained_fraction'][:, n], null['quantile']):>12.2f}"
                    f"{np.median(null['chain_length'][:, n]):>12.1f}")
        print(row)
    print(f"\n  leading M-dimensional eigenspaces, local vs restricted global (window medians)")
    cols = f"  {'M':>3}{'overlap':>10}{'capture':>10}{'random':>9}"
    if null is not None:
        cols += f"{'| null ov.':>12}{'null cap.':>11}"
    print(cols)
    for i, M in enumerate(comp["subspace_dims"]):
        row = (f"  {M:>3}{np.nanmedian(comp['subspace_overlap'][:, i]):>10.2f}{np.nanmedian(comp['capture_ratio'][:, i]):>10.2f}"
               f"{comp['subspace_random'][i]:>9.2f}")
        if null is not None:
            row += (f"{np.quantile(null['subspace_overlap'][:, i], null['quantile']):>12.2f}"
                    f"{np.quantile(null['capture_ratio'][:, i], null['quantile']):>11.2f}")
        print(row)
    if spatial:
        print(f"\n  found = 1 + number of the {res['tracking']['max_rank']} nearest windows (by centre distance) in which the mode is found again"
              f" (overlap >= {res['tracking']['threshold']:g}, >= {res['tracking']['min_weight']:g} of its norm on the shared atoms);"
              f" not a chain of consecutive windows")
    print(f"\n  consistency between overlapping windows: matched overlap of local modes 0 / 1 / 2 (mean over windows with"
          f" >= {res['tracking']['min_weight']:g} of the mode on the shared atoms)")
    cols = f"  {'rank' if spatial else 'shift':>6}{'shared':>9}{'mode 0':>9}{'mode 1':>9}{'mode 2':>9}{'M=3 space':>11}"
    if null is not None:
        cols += f"{'| null mode 0':>15}{'null M=3':>10}"
    print(cols)
    dims = list(cons["subspace_dims"])
    i3 = dims.index(3) if 3 in dims else 0
    for o, j in enumerate(cons["offsets"]):
        m = S["consistency_matched"][o]
        row = (f"  {j:>6}{cons['shared_fraction'][o]:>9.2f}" + "".join(f"{m[k]:>9.2f}" if k < len(m) else f"{'':>9}" for k in range(3))
               + f"{S['consistency_subspace'][o, i3]:>11.2f}")
        if null is not None:
            row += (f"{np.quantile(null['consistency_matched'][:, o, 0], null['quantile']):>15.2f}"
                    f"{np.quantile(null['consistency_subspace'][:, o, i3], null['quantile']):>10.2f}")
        print(row)
    if spatial:
        print("  rank j = the j-th nearest other window centre (the nearest windows share the most atoms).")
    print("  Windows share frames and atoms; agreement between them is not independent evidence."
          "\n  Local modes are modes of the window's own covariance, not global modes (see the header of local_pca.py).")
