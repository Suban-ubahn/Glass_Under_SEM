# =============================================================
# analysis/tls_detection.py — two-level-system (TLS) detection
#
# Two independent methods, matching your professor's email:
#
#   Method A ("direct", detect_tls_direct*): purely structural/
#     real-space. Tracks each atom's position relative to the
#     centroid of its own local coordination shell (its "cage") and
#     tests whether that relative position hops between two discrete,
#     recurring configurations — "structures/neighbourhoods ... that
#     jump between 2 configurations". Independent of soft_modes.py.
#
#   Method B ("amplitude", detect_tls_amplitude): mode-based. Takes
#     the softest/most localized modes already identified by
#     soft_modes.py and tests whether each mode's own amplitude time
#     series a_n(t) is Gaussian (single peak, an ordinary — if soft —
#     vibration) or bimodal (two peaks, a genuine double well) —
#     "a typical [oscillator] distribution would have just one peak
#     ... a TLS one would have two".
#
# Why two different methods, and why Method B needs more than
# soft_modes.py alone: the covariance/participation-ratio analysis in
# soft_modes.py is a purely SECOND-MOMENT method (it only ever sees
# <a_n^2>), so it is structurally blind to well SHAPE — a wide single
# well and a symmetric double well with matching variance give
# IDENTICAL covariance eigenvalues. soft_modes.py can tell you WHERE
# the softest, most localized motion is; only looking at the actual
# distribution P(a_n) (Method B) — or the real-space switching
# pattern directly (Method A) — can tell you whether that motion is
# single- or double-welled, i.e. whether it's a soft spot or a TLS.
#
# Both methods share the same underlying statistical test
# (bimodality_test_batch, via a small dependency-free vectorized EM
# fit — this project's requirements.txt has no scikit-learn), applied
# either to per-atom cage coordinates (Method A) or per-mode
# amplitudes (Method B).
# =============================================================
 
import numpy as np
 
from config import CUTOFF_DICT
 
 
# ------------------------------------------------------------
# Shared statistics: vectorized 1-vs-2-component Gaussian test
# ------------------------------------------------------------
 
def _batched_one_component_bic(X):
    """
    Closed-form single isotropic-Gaussian fit + BIC, for M items each
    with their own (T, D) sample. X : (M, T, D).
    """
    X = np.asarray(X, dtype=float)
    M, T, D = X.shape
    mean = X.mean(axis=1)                                        # (M, D)
    diff = X - mean[:, None, :]
    sigma2 = np.maximum(np.sum(diff ** 2, axis=(1, 2)) / (T * D), 1e-12)
    loglik = (-0.5 * T * D * np.log(2 * np.pi * sigma2)
              - 0.5 * np.sum(diff ** 2, axis=(1, 2)) / sigma2)
    n_params = D + 1                                              # mean (D) + 1 sigma
    bic = n_params * np.log(T) - 2 * loglik
    return {"mean": mean, "sigma": np.sqrt(sigma2), "loglik": loglik, "bic": bic}
 
 
def _batched_two_component_gmm(X, n_iter=100, seed=0, sigma_floor_frac=1e-3):
    """
    Vectorized EM fit of a 2-component isotropic Gaussian mixture,
    fit independently and SIMULTANEOUSLY for M items (atoms or modes)
    that each carry their own (T, D) time series — no sklearn
    dependency (not in this project's requirements.txt), and no
    per-item Python loop, so this stays fast even for M ~ thousands
    of atoms.
 
    Parameters
    ----------
    X : (M, T, D) array
 
    Returns
    -------
    dict of per-item arrays: means (M,2,D), sigmas (M,2),
    weights (M,2), labels (M,T) int (hard assignment via max
    responsibility), loglik (M,), bic (M,)
    """
    X = np.asarray(X, dtype=float)
    M, T, D = X.shape
 
    # --- init: split along the (per-item) top principal axis ---
    Xc = X - X.mean(axis=1, keepdims=True)
    if D > 1:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)         # Vt: (M, K, D)
        top = Vt[:, 0, :]                                          # (M, D)
        proj = np.einsum("mtd,md->mt", Xc, top)                    # (M, T)
    else:
        proj = Xc[..., 0]                                          # (M, T)
 
    med = np.median(proj, axis=1, keepdims=True)
    lo_mask = proj < med                                           # (M, T)
 
    def masked_mean(mask):
        cnt = mask.sum(axis=1, keepdims=True).clip(min=1)
        return (mask[..., None] * X).sum(axis=1) / cnt              # (M, D)
 
    means = np.stack([masked_mean(lo_mask), masked_mean(~lo_mask)], axis=1)   # (M,2,D)
 
    overall_std = X.std(axis=(1, 2)) + 1e-12                        # (M,)
    sigma_floor = sigma_floor_frac * overall_std
    sigmas = np.stack([overall_std, overall_std], axis=1)           # (M,2)
    weights = np.full((M, 2), 0.5)
 
    resp = np.zeros((M, T, 2))
    loglik = np.full(M, -np.inf)
 
    for _ in range(n_iter):
        log_resp = np.zeros((M, T, 2))
        for k in range(2):
            diff = X - means[:, k:k + 1, :]                         # (M,T,D)
            sq = np.sum(diff ** 2, axis=2)                          # (M,T)
            s2 = sigmas[:, k] ** 2 + 1e-300                          # (M,)
            log_resp[:, :, k] = (
                np.log(weights[:, k] + 1e-300)[:, None]
                - 0.5 * D * np.log(2 * np.pi * s2)[:, None]
                - 0.5 * sq / s2[:, None]
            )
        m = log_resp.max(axis=2, keepdims=True)
        r = np.exp(log_resp - m)
        r_sum = r.sum(axis=2, keepdims=True)
        loglik = np.sum(m[..., 0] + np.log(r_sum[..., 0] + 1e-300), axis=1)   # (M,)
        resp = r / r_sum                                            # (M,T,2)
 
        Nk = resp.sum(axis=1) + 1e-12                                # (M,2)
        for k in range(2):
            means[:, k, :] = np.einsum("mt,mtd->md", resp[:, :, k], X) / Nk[:, k:k + 1]
            diff = X - means[:, k:k + 1, :]
            sq = np.sum(diff ** 2, axis=2)
            var = (resp[:, :, k] * sq).sum(axis=1) / (Nk[:, k] * D)
            sigmas[:, k] = np.sqrt(np.maximum(var, sigma_floor ** 2))
        weights = Nk / T
 
    labels = np.argmax(resp, axis=2)                                 # (M, T)
    n_params = 2 * (D + 1) + 1                                       # 2 means(D) + 2 sigmas + 1 free weight
    bic = n_params * np.log(T) - 2 * loglik
 
    return {
        "means": means, "sigmas": sigmas, "weights": weights,
        "labels": labels, "loglik": loglik, "bic": bic,
    }
 
 
def _effective_sample_size(X, floor=5.0):
    """
    Estimate each item's EFFECTIVE sample size from the lag-1
    autocorrelation of its dominant fluctuation direction.
 
    This matters because consecutive MD frames are correlated (smooth
    motion, not independent draws) — using the raw frame count T as
    the BIC's sample size systematically OVERSTATES the independent
    evidence for the more complex (2-component) model, since BIC's
    derivation assumes i.i.d. samples. On this project's own
    trj.zr50cu50_6750_vib (101 highly-autocorrelated frames), skipping
    this correction flagged the large majority of atoms as "TLS" —
    almost certainly mostly false positives from overfitting a short,
    smooth time series, not genuine bistability.
 
    Standard AR(1)-style correction: T_eff = T * (1-rho)/(1+rho),
    rho = lag-1 autocorrelation of the top-principal-axis projection.
    Clipped so the penalty never gets WEAKER than using the true T
    (rho<0 -> rho=0) and never collapses to near-zero for a very
    smooth/slow series (floored at `floor` samples).
 
    X : (M, T, D) array
    Returns T_eff : (M,) array
    """
    X = np.asarray(X, dtype=float)
    M, T, D = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    if D > 1:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        top = Vt[:, 0, :]
        proj = np.einsum("mtd,md->mt", Xc, top)
    else:
        proj = Xc[..., 0]
 
    x0, x1 = proj[:, :-1], proj[:, 1:]
    num = np.sum(x0 * x1, axis=1)
    den = np.sqrt(np.sum(x0 ** 2, axis=1) * np.sum(x1 ** 2, axis=1)) + 1e-300
    rho = np.clip(num / den, 0.0, 0.98)
 
    T_eff = T * (1 - rho) / (1 + rho)
    return np.maximum(T_eff, floor)
 
 
def bimodality_test_batch(X, delta_bic_threshold=6.0, separation_threshold=1.5,
                           min_weight_fraction=0.15, min_switches=2,
                           n_iter=100, seed=0, autocorrelation_correction=True):
    """
    Vectorized bimodality test over M items (atoms or modes) at once.
    Fits 1- vs 2-component isotropic Gaussians to each item's (T, D)
    sample and combines THREE independent criteria into `is_tls`, so
    a single coincidence (e.g. one outlier frame) doesn't trip a
    false positive:
 
      1. delta_bic = BIC(1-component) - BIC(2-component) > threshold.
         Kass & Raftery's rule of thumb: >6 is "strong evidence",
         >10 is "very strong" (default here: 6). By default (see
         `autocorrelation_correction`), the BIC's sample-size penalty
         uses an AUTOCORRELATION-CORRECTED effective T (see
         _effective_sample_size()), not the raw frame count — MD
         frames are not independent samples, and using the raw count
         badly inflates apparent evidence for bimodality on a short,
         smooth trajectory.
      2. separation = |mean1-mean2| / mean(sigma1,sigma2) > threshold.
         The physical distance between the two wells must exceed
         their own thermal in-well spread — this is precisely
         "bimodal rather than Gaussian".
      3. n_switches >= min_switches: at least one full A->B->A round
         trip is required, so a one-off PERMANENT local
         rearrangement (i.e. a D^2_min event) is not mistaken for a
         two-level system that keeps hopping.
 
    Parameters
    ----------
    X : (M, T, D) array
    autocorrelation_correction : bool — set False to use the raw
        frame count for the BIC penalty instead (matches a naive i.i.d.
        BIC; NOT recommended for a short/smooth trajectory, but useful
        to compare against).
 
    Returns
    -------
    dict of per-item (M,) / (M,2) / (M,T) arrays: delta_bic,
    separation, n_switches, min_weight, is_tls (bool), t_eff (the
    effective sample size actually used for the BIC penalty), plus
    the raw 2-component fit (means, sigmas, weights, labels).
    """
    M, T, D = np.asarray(X).shape
    fit1 = _batched_one_component_bic(X)
    fit2 = _batched_two_component_gmm(X, n_iter=n_iter, seed=seed)
 
    t_for_bic = _effective_sample_size(X) if autocorrelation_correction else np.full(M, float(T))
    # Rescale the log-likelihoods themselves by T_eff/T, not just the
    # BIC penalty term. Both the apparent penalty AND the apparent
    # likelihood improvement from adding a second component are
    # inflated by treating T autocorrelated frames as T independent
    # samples -- correcting only the ln(T) penalty term (leaving the
    # inflated log-likelihood untouched) perversely makes a short,
    # smooth trajectory look like STRONGER evidence for bimodality,
    # not weaker. Scaling both consistently is equivalent to asking
    # "if we'd only seen T_eff truly independent samples with this
    # same average per-frame fit quality, how much evidence would
    # that be?"
    scale = t_for_bic / T
    loglik1 = fit1["loglik"] * scale
    loglik2 = fit2["loglik"] * scale
 
    n_params_1 = D + 1
    n_params_2 = 2 * (D + 1) + 1
    bic1 = n_params_1 * np.log(t_for_bic) - 2 * loglik1
    bic2 = n_params_2 * np.log(t_for_bic) - 2 * loglik2
    delta_bic = bic1 - bic2
 
    sep = (np.linalg.norm(fit2["means"][:, 0] - fit2["means"][:, 1], axis=1)
           / (0.5 * (fit2["sigmas"][:, 0] + fit2["sigmas"][:, 1]) + 1e-12))
    labels = fit2["labels"]
    n_switches = np.sum(labels[:, 1:] != labels[:, :-1], axis=1)
    min_weight = fit2["weights"].min(axis=1)
 
    is_tls = (
        (delta_bic > delta_bic_threshold)
        & (sep > separation_threshold)
        & (n_switches >= min_switches)
        & (min_weight > min_weight_fraction)
    )
 
    return {
        "delta_bic": delta_bic, "separation": sep,
        "n_switches": n_switches, "min_weight": min_weight, "t_eff": t_for_bic,
        "is_tls": is_tls, "labels": labels,
        "means": fit2["means"], "sigmas": fit2["sigmas"], "weights": fit2["weights"],
    }
 
 
def bimodality_coefficient(x):
    """
    Cheap secondary/diagnostic bimodality check (the standard
    SAS-style sample bimodality coefficient):
 
        BC = (skew^2 + 1) / (excess_kurtosis + 3*(n-1)^2/((n-2)*(n-3)))
 
    BC > 5/9 (~0.555, the value for a uniform distribution) is the
    conventional rule-of-thumb evidence for bimodality. This is a
    pure distribution-shape statistic — no mixture model fit needed —
    reported alongside bimodality_test_batch()'s result as a
    complementary diagnostic; it does NOT gate `is_tls` above.
    """
    from scipy.stats import skew, kurtosis
    x = np.asarray(x).ravel()
    n = len(x)
    if n <= 3:
        return np.nan
    g = skew(x, bias=False)
    k = kurtosis(x, fisher=True, bias=False)   # excess kurtosis (normal = 0)
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g ** 2 + 1) / (k + correction)
 
 
# ------------------------------------------------------------
# Method A: direct / structural (local-cage bistability)
# ------------------------------------------------------------
 
def local_cage_series(trajectory, atom_indices=None, cutoff=CUTOFF_DICT,
                       reference_frame=0, n_dim=3, min_neighbors=4):
    """
    For each requested atom, build the time series of its position
    relative to the centroid of its own (fixed-topology) coordination
    shell — "where does this atom sit relative to its cage", the
    coordinate a genuine local double-well TLS should flip back and
    forth in.
 
    Uses the same fixed-neighbor-list + fresh-PBC-per-frame
    convention as dynamics.flow_units.compute_d2min (topology from
    one reference frame, but a minimum-image reduction recomputed
    every frame), for the same reason: this project's LAMMPS dumps
    re-wrap coordinates into the primary cell every frame, so a naive
    fixed periodic shift would register spurious jumps.
 
    Because the fixed neighbor set for atom i is {j}, and its
    PBC-correct current bond vectors are r_ij(t), the atom's position
    relative to its cage centroid works out to simply
    -mean_j(r_ij(t)) — zero for a perfectly symmetric cage, non-zero
    when the atom sits off-center in its shell.
 
    Parameters
    ----------
    n_dim : 2 or 3 — 2 keeps only x, y (for analysis restricted to a
        z-slice, matching soft_modes.py's 2D analysis); 3 keeps the
        full 3D relative vector.
 
    Returns
    -------
    series : (M, T, n_dim) array
    used_atom_indices : (M,) array — the subset of atom_indices that
        had >= min_neighbors at the reference frame (same order as
        series' first axis)
    """
    from ase.neighborlist import neighbor_list
 
    n_atoms = len(trajectory[0])
    if atom_indices is None:
        atom_indices = np.arange(n_atoms)
    atom_indices = np.asarray(atom_indices)
 
    atoms_ref = trajectory[reference_frame]
    i_idx, j_idx, S = neighbor_list("ijS", atoms_ref, cutoff)
 
    want = np.zeros(n_atoms, dtype=bool)
    want[atom_indices] = True
    keep = want[i_idx]
    i_idx, j_idx = i_idx[keep], j_idx[keep]
 
    degree = np.bincount(i_idx, minlength=n_atoms)
    used_atom_indices = atom_indices[degree[atom_indices] >= min_neighbors]
 
    row_of = np.full(n_atoms, -1, dtype=int)
    row_of[used_atom_indices] = np.arange(len(used_atom_indices))
    edge_keep = row_of[i_idx] >= 0
    i_idx, j_idx = i_idx[edge_keep], j_idx[edge_keep]
    rows = row_of[i_idx]
 
    T = len(trajectory)
    M = len(used_atom_indices)
    pos_traj = np.array([a.positions for a in trajectory])              # (T, N, 3)
    box_traj = np.array([np.diag(a.cell.array) for a in trajectory])     # (T, 3)
 
    raw = pos_traj[:, j_idx, :] - pos_traj[:, i_idx, :]                  # (T, E, 3)
    r = raw - box_traj[:, None, :] * np.round(raw / box_traj[:, None, :])
 
    counts = np.bincount(rows, minlength=M).astype(float)
    sum_r = np.zeros((T, M, 3))
    for t in range(T):
        np.add.at(sum_r[t], rows, r[t])                                  # scatter-add over edges only
    mean_r = sum_r / counts[None, :, None]
 
    series = -mean_r[..., :n_dim]                                       # (T, M, n_dim)
    series = np.transpose(series, (1, 0, 2))                            # (M, T, n_dim)
 
    return series, used_atom_indices
 
 
def detect_tls_direct(trajectory, atom_indices=None, cutoff=CUTOFF_DICT,
                       reference_frame=0, n_dim=3, min_neighbors=4,
                       delta_bic_threshold=6.0, separation_threshold=1.5,
                       min_weight_fraction=0.15, min_switches=2,
                       n_iter=100, seed=0):
    """
    Method A ("direct"): flag atoms whose position relative to their
    own local coordination cage hops between two discrete, recurring
    configurations — your professor's "jumping between two
    configurations, statistics more bivariate than Gaussian".
 
    Purely structural/real-space; independent of soft_modes.py (see
    detect_tls_amplitude() for the mode-based alternative) — that
    independence is exactly what lets you cross-check one against the
    other, rather than getting the same answer twice under two names.
 
    Tip: running this on all N atoms works, but is wasteful — most
    atoms are just doing ordinary single-well vibration. Cheaper and
    arguably more meaningful to pass atom_indices=<top candidates by
    soft_modes.atom_softness_scores> and only test the ones already
    flagged as soft/localized.
 
    Returns
    -------
    dict with:
      "atom_indices" : (M,) the subset actually analyzed
      "is_tls"       : (N,) bool, False everywhere not analyzed
      "delta_bic"    : (N,) float, NaN everywhere not analyzed
      "separation", "n_switches", "min_weight" : (M,) arrays, in the
          same order as "atom_indices" (not full-length — see below)
      "series"       : (M, T, n_dim) the underlying cage-position
          time series (for plotting with
          visualization.plot.plot_mode_amplitude_distribution)
    """
    series, used_idx = local_cage_series(
        trajectory, atom_indices=atom_indices, cutoff=cutoff,
        reference_frame=reference_frame, n_dim=n_dim, min_neighbors=min_neighbors,
    )
    stats = bimodality_test_batch(
        series, delta_bic_threshold=delta_bic_threshold,
        separation_threshold=separation_threshold,
        min_weight_fraction=min_weight_fraction, min_switches=min_switches,
        n_iter=n_iter, seed=seed,
    )
 
    n_atoms = len(trajectory[0])
    is_tls_full = np.zeros(n_atoms, dtype=bool)
    delta_bic_full = np.full(n_atoms, np.nan)
    is_tls_full[used_idx] = stats["is_tls"]
    delta_bic_full[used_idx] = stats["delta_bic"]
 
    return {
        "atom_indices": used_idx,
        "is_tls": is_tls_full,
        "delta_bic": delta_bic_full,
        "separation": stats["separation"],
        "n_switches": stats["n_switches"],
        "min_weight": stats["min_weight"],
        "t_eff": stats["t_eff"],
        "means": stats["means"], "sigmas": stats["sigmas"], "weights": stats["weights"],
        "labels": stats["labels"],
        "series": series,
    }
 
 
def detect_tls_direct_2d_slices(trajectory, n_slices=10, cutoff=CUTOFF_DICT,
                                 reference_frame=0, min_neighbors=4,
                                 min_atoms_per_slice=8, **kwargs):
    """
    Runs detect_tls_direct() independently within each of n_slices
    equally-spaced z-slabs — the SAME slicing as
    soft_modes.soft_mode_analysis_2d_slices() (both call
    soft_modes.assign_z_slices() with the same n_slices), so slice
    s's atoms line up exactly between the two analyses — using only
    the in-plane (x, y) cage-relative displacement (n_dim=2).
 
    Extra **kwargs are forwarded to detect_tls_direct() (e.g.
    delta_bic_threshold, min_switches, ...).
    """
    from analysis.soft_modes import assign_z_slices
 
    positions_3d = np.array([a.positions for a in trajectory])
    mean_pos = positions_3d.mean(axis=0)
    slice_idx, z_edges = assign_z_slices(mean_pos[:, 2], n_slices=n_slices)
 
    slices = []
    for s in range(n_slices):
        idx = np.where(slice_idx == s)[0]
        if len(idx) < min_atoms_per_slice:
            slices.append(None)
            continue
        result = detect_tls_direct(
            trajectory, atom_indices=idx, cutoff=cutoff,
            reference_frame=reference_frame, n_dim=2,
            min_neighbors=min_neighbors, **kwargs,
        )
        result["z_range"] = (z_edges[s], z_edges[s + 1])
        slices.append(result)
 
    return {"slices": slices, "slice_idx": slice_idx, "z_edges": z_edges}
 
 
# ------------------------------------------------------------
# Method B: mode-amplitude bimodality (P(a_n))
# ------------------------------------------------------------
 
def detect_tls_amplitude(mode_result, delta_bic_threshold=6.0,
                          separation_threshold=1.5, min_weight_fraction=0.15,
                          min_switches=2, n_check_modes=None, n_iter=100, seed=0):
    """
    Method B: for each of the soft/localized modes already identified
    by soft_modes.soft_mode_analysis_3d()/_2d_slices() (`mode_result`
    — one of those functions' return dict), test whether that mode's
    amplitude time series a_n(t) — returned for free by the SVD in
    modes_from_displacements() — is Gaussian (an ordinary, if soft,
    single-well vibration) or bimodal (a double well: a genuine TLS
    "hopping" along that direction).
 
    By default checks exactly the modes soft_modes.py already flagged
    via `mode_result["soft_mode_mask"]` (i.e. the softest N modes,
    optionally participation-ratio-filtered) — so Method B directly
    answers "of the soft spots we found, which ones are actually
    bistable rather than just floppy?" (Falls back to the first 20
    modes if soft_mode_mask isn't present, e.g. if you built
    mode_result by hand.)
 
    Requires mode_result["amplitudes"] — i.e. must come from an MD
    trajectory via soft_modes.py's high-level wrappers, not from
    modes_from_covariance() on a bare CAVIAR Green's tensor alone,
    which has no per-frame amplitudes to test.
 
    Returns
    -------
    dict with:
      "checked_modes" : (M,) the mode indices actually tested
      "is_tls_mode"   : (M,) bool, same order as checked_modes
      "tls_mode_idx"  : indices (into the FULL mode list) of modes
                        flagged bimodal
      "delta_bic", "separation", "n_switches" : (M,) arrays
      "tls_score"     : (N,) per-atom score — sum of |eigenvector|^2
                        over flagged bimodal modes (participation-
                        weighted, same convention as
                        soft_modes.atom_softness_scores)
      "is_tls"        : (N,) bool, tls_score > 0
    """
    if mode_result.get("amplitudes") is None:
        raise ValueError(
            "mode_result has no per-frame amplitudes -- detect_tls_amplitude() "
            "needs output from soft_modes.soft_mode_analysis_3d()/_2d_slices() "
            "(built from an MD trajectory), not modes_from_covariance() on a "
            "bare Green's tensor, which has no per-frame amplitudes to test."
        )
 
    amplitudes = mode_result["amplitudes"]          # (T, K)
    eigenvectors = mode_result["eigenvectors"]       # (N, D, K)
    mode_mask = mode_result.get("soft_mode_mask")
    if mode_mask is None:
        K = amplitudes.shape[1]
        mode_mask = np.zeros(K, dtype=bool)
        mode_mask[:min(20, K)] = True
    else:
        mode_mask = np.asarray(mode_mask).copy()
 
    if n_check_modes is not None:
        idx_true = np.where(mode_mask)[0][:n_check_modes]
        mode_mask = np.zeros_like(mode_mask)
        mode_mask[idx_true] = True
 
    check_idx = np.where(mode_mask)[0]
    X = amplitudes[:, check_idx].T[:, :, None]       # (M, T, 1)
 
    stats = bimodality_test_batch(
        X, delta_bic_threshold=delta_bic_threshold,
        separation_threshold=separation_threshold,
        min_weight_fraction=min_weight_fraction, min_switches=min_switches,
        n_iter=n_iter, seed=seed,
    )
 
    n_atoms = eigenvectors.shape[0]
    tls_mode_idx = check_idx[stats["is_tls"]]
    tls_score = np.zeros(n_atoms)
    for n in tls_mode_idx:
        tls_score += np.sum(eigenvectors[:, :, n] ** 2, axis=1)
 
    return {
        "checked_modes": check_idx,
        "is_tls_mode": stats["is_tls"],
        "tls_mode_idx": tls_mode_idx,
        "delta_bic": stats["delta_bic"],
        "separation": stats["separation"],
        "n_switches": stats["n_switches"],
        "tls_score": tls_score,
        "is_tls": tls_score > 0,
    }
 
 
if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import (
        TRAJECTORY_FILE, N_Z_SLICES, N_SOFT_MODES, SOFT_MODE_PR_MAX,
        TLS_DELTA_BIC_THRESHOLD, TLS_SEPARATION_THRESHOLD,
        TLS_MIN_WEIGHT_FRACTION, TLS_MIN_SWITCHES,
    )
    from analysis.soft_modes import soft_mode_analysis_3d
 
    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames, {len(trajectory[0])} atoms.")
 
    print("\n--- Method A (direct/structural), 3D, all atoms ---")
    tls_direct = detect_tls_direct(
        trajectory,
        delta_bic_threshold=TLS_DELTA_BIC_THRESHOLD,
        separation_threshold=TLS_SEPARATION_THRESHOLD,
        min_weight_fraction=TLS_MIN_WEIGHT_FRACTION,
        min_switches=TLS_MIN_SWITCHES,
    )
    print(f"{tls_direct['is_tls'].sum()} / {len(tls_direct['atom_indices'])} "
          f"analyzed atoms flagged as TLS.")
 
    print("\n--- Method B (mode-amplitude bimodality), 3D ---")
    modes_3d = soft_mode_analysis_3d(
        trajectory, n_soft_modes=N_SOFT_MODES, pr_max=SOFT_MODE_PR_MAX,
    )
    tls_amplitude = detect_tls_amplitude(
        modes_3d,
        delta_bic_threshold=TLS_DELTA_BIC_THRESHOLD,
        separation_threshold=TLS_SEPARATION_THRESHOLD,
        min_weight_fraction=TLS_MIN_WEIGHT_FRACTION,
        min_switches=TLS_MIN_SWITCHES,
    )
    print(f"{tls_amplitude['is_tls_mode'].sum()} / {len(tls_amplitude['checked_modes'])} "
          f"checked soft modes flagged as bimodal (TLS-like).")