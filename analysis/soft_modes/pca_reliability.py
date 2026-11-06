# =============================================================
# analysis/soft_modes/pca_reliability.py — how far can the PCA of this
# trajectory be trusted?
#
# SETUP. X is the (T, P) matrix with one row per frame and P = N*D columns
# (one per coordinate of one atom): X[t, (i,d)] = sqrt(m_i) * u[t,i,d], u = the
# displacement from the time-averaged position, so every column has mean 0.
# Its covariance matrix is C = X^T X / T (P x P). PCA = eigen-decomposition
#     C v_n = lambda_n v_n,   |v_n| = 1,   lambda_1 >= lambda_2 >= ...
# (at most T-1 eigenvalues are non-zero). Project frame t (row x_t) on mode n:
#     a_n(t) = x_t . v_n                    (the amplitude of mode n in frame t)
# Then
#     lambda_n = (1/T) sum_t a_n(t)^2       (mean squared amplitude over the frames)
# and v_n is the unit direction that makes this mean squared amplitude as large
# as possible (given orthogonality to the earlier modes).
#
# PROBLEM. v_n is chosen BECAUSE the frames used moved a lot along it, and
# lambda_n is then measured on those same frames. The sum of all eigenvalues is
# the total variance, tr C = sum over columns of their variance, which does not
# depend on the modes. If no direction were special (every column independent,
# variance sigma^2) that total P*sigma^2 would still be spread over at most T-1
# eigenvalues, each about P*sigma^2/T -- P/T times too large (~200 for
# 20250 coordinates and 101 frames). The PCA returns T-1 modes for ANY data.
#
# TEST 1 (pca_reliability_test): measure lambda_n on frames that did not take part
# in choosing v_n. Frames are cut into a first half A (T_A frames) and a second half
# B (T_B frames); each half is centred on its own mean (X_A, X_B), and
#     C_A = X_A^T X_A / T_A,   C_B = X_B^T X_B / T_B.
# PCA of A gives (lambda^A_n, v^A_n). Then
#     lambda^A_n   = v^A_n^T C_A v^A_n = (1/T_A) sum_{t in A} (x_t . v^A_n)^2   in-sample
#     h^{A->B}_n   = v^A_n^T C_B v^A_n = (1/T_B) sum_{t in B} (x_t . v^A_n)^2   held out
# h is the mean squared amplitude along the SAME direction, taken over frames
# that played no part in choosing it: project every frame of B on v^A_n, square, average.
# If frames of A and B are independent samples of a distribution with true
# covariance C0, then given v^A_n,  E[h] = v^T C0 v  (the true variance along that
# direction), whereas lambda^A_n = v^T C0 v + v^T (C_A - C0) v, and the last term is
# positive on average because v was picked where C_A is large.
# RETAINED FRACTION   R_n = h_n / lambda_n   (R^{A->B} and R^{B->A}, averaged):
#     R ~ 1    the variance of the mode is real: it is as large in new frames
#     R << 1   the eigenvalue was mostly selection gain (fitted noise)
# For pure noise of equal variance sigma^2 in all P coordinates: lambda^A ~ P sigma^2/T_A,
# h ~ sigma^2 (a direction unrelated to B's frames), so R ~ T_A / P.
#
# CHANCE LEVEL. Null hypothesis H0: coordinates are independent of each other, each with
# its own variance and its own autocorrelation in time. Simulated by shifting every
# column c cyclically by its own random offset s_c:  x'[t,c] = x[(t + s_c) mod T, c].
# Each column keeps its values, variance and autocorrelation (up to the splice point);
# the correlation between two columns becomes the original cross-correlation at lag
# s_c - s_c', about 0, so the collective modes are gone. The identical procedure on
# n_null such data sets gives the distribution of R_n under H0; "chance level" at rank n
# is its 95th percentile. A mode is "above chance" if R_n exceeds it; modes are counted as
# an unbroken run from the softest. (Comparing the full-data eigenvalues directly with
# such a baseline is not reliable when the frames are correlated in time -- it flags
# pure noise -- which is why the test is built on held-out frames.)
#
# TEST 2 (softness_reproducibility_test): the same split applied to the per-atom softness
#     s_i = sum_{n soft} lambda_n p_{i,n},     p_{i,n} = sum_d e_{i,d,n}^2     (atom_softness_scores)
# p_{i,n} is the share of mode n that sits on atom i (it sums to 1 over the atoms), so s_i is the
# part of atom i's mean-square motion that lies in the softest modes. The whole soft-mode analysis
# is run on half A and on half B (each re-centred on its own mean); the two maps s^A, s^B are compared by
#   r_s  Spearman rank correlation over the atoms (Pearson correlation of the ranks:
#        1 = same ordering of the atoms, 0 = unrelated)
#   O_f  |top-f of s^A  intersect  top-f of s^B| / |top-f|,  f = 5 % (the atoms that are sent to
#        the TLS detector); two unrelated maps give O_f = f
# The same shifted-column null as above gives the distribution of r_s and O_f for data without
# collective modes; p = (1 + number of null runs with a value >= the observed one) / (n_null + 1).
# That null keeps every coordinate's variance, so it already contains "atoms that simply vibrate more
# look soft in both halves". To see how much of the map is just that, two references without PCA are
# added: the per-atom mean-square displacement map m_i <|u_i|^2> compared half against half in the same
# way, and the Spearman correlation between each half's softness map and its own MSD map.
# =============================================================

import numpy as np


def _flat(u, masses=None):
    """The (T, N*D) matrix modes_from_displacements() decomposes."""
    T, N, D = u.shape
    flat = np.asarray(u, dtype=float).reshape(T, N * D)
    if masses is not None:
        flat = flat * np.repeat(np.sqrt(np.asarray(masses, dtype=float)), D)[None, :]
    return flat


def _pca(X, k):
    """All eigenvalues (S^2 / T) and the top-k eigenvectors (P, k) of X (T, P)."""
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    return S ** 2 / len(X), Vt[:k].T


def _shift_columns(X, rng):
    """Null data: every column shifted circularly in time by its own random amount."""
    T, P = X.shape
    rows = (np.arange(T)[:, None] + rng.integers(1, T, size=P)[None, :]) % T
    return X[rows, np.arange(P)[None, :]]


def _split_half(X, k):
    """
    PCA on the first half of the frames and on the second half; each half's modes
    are applied to the other half. Returns three (k,) arrays, averaged over the two directions:
    in-sample eigenvalue, held-out variance along the same modes, and their ratio.
    """
    h = len(X) // 2
    A = X[:h] - X[:h].mean(axis=0)
    B = X[h:] - X[h:].mean(axis=0)
    lam_a, Va = _pca(A, k)
    lam_b, Vb = _pca(B, k)
    held_ab = ((B @ Va) ** 2).mean(axis=0)          # variance of B along A's modes
    held_ba = ((A @ Vb) ** 2).mean(axis=0)
    retained = 0.5 * (held_ab / lam_a[:k] + held_ba / lam_b[:k])
    return 0.5 * (lam_a[:k] + lam_b[:k]), 0.5 * (held_ab + held_ba), retained


def _effective_frames(X):
    """
    Number of independent frames T_eff = T / (1 + 2 sum_k (1 - k/T) r_k), with r_k the
    autocorrelation in time pooled over all coordinates (summed up to its first
    negative value). T_eff = T for frames that are independent.
    """
    T, P = X.shape
    Z = X - X.mean(axis=0)
    F = np.fft.rfft(Z, n=2 * T, axis=0)
    acf = np.fft.irfft((F.real ** 2 + F.imag ** 2).sum(axis=1), n=2 * T)[:T]
    r = acf / acf[0] * T / (T - np.arange(T))        # unbiased autocorrelation
    neg = np.where(r[1:] < 0)[0]
    kmax = T - 1 if len(neg) == 0 else neg[0] + 1
    k = np.arange(1, kmax)
    return float(T / (1 + 2 * np.sum((1 - k / T) * r[1:kmax])))


def pca_reliability_test(u, masses=None, n_null=100, quantile=0.95,
                         n_modes_tested=20, seed=0):
    """
    Parameters
    ----------
    u : (T, N, D) centred displacements -- the same array that is passed to
        modes_from_displacements() (see utils.compute_displacements()).
    masses : (N,) array or None -- the same masses (None = no mass weighting).
    n_null : number of null data sets.
    quantile : a mode counts as above chance when its retained fraction beats this
        quantile of the null runs (0.95 = beats 95 % of them).
    n_modes_tested : how many of the softest modes to test (limited by the half length).

    Returns
    -------
    dict:
      n_frames, n_coords          T, N*D
      t_eff                       effective number of independent frames (see _effective_frames)
      eigenvalues        (K,)     of the whole trajectory (as modes_from_displacements()), K = min(T-1, N*D)
      noise_floor                 total variance / (N*D): the variance of a random direction
      in_sample          (Ks,)    eigenvalues of the half-trajectory PCAs (mean of the two halves)
      held_out           (Ks,)    variance along the same modes in the other half
      retained           (Ks,)    held_out / in_sample
      retained_null_median / retained_null_high (Ks,)    chance level
      n_reproducible     int      leading modes (from the softest) whose retained fraction beats the null
      n_mostly_real      int      leading modes with retained fraction >= 0.5
      efficiency         float    sum(held_out) / sum(in_sample) over the first n_reproducible modes
      quantile, n_null
    """
    X = _flat(u, masses)
    T, P = X.shape
    K = min(T - 1, P)
    Ks = max(1, min(n_modes_tested, T // 2 - 1, P))

    eig = _pca(X, 1)[0][:K]
    in_s, held, ret = _split_half(X, Ks)

    rng = np.random.default_rng(seed)
    null_ret = np.empty((n_null, Ks))
    for r in range(n_null):
        null_ret[r] = _split_half(_shift_columns(X, rng), Ks)[2]
    null_high = np.quantile(null_ret, quantile, axis=0)

    def n_leading(mask):                               # length of the leading run of True
        return len(mask) if mask.all() else int(np.argmin(mask))

    n_rep = n_leading(ret > null_high)
    return {
        "n_frames": T, "n_coords": P, "t_eff": _effective_frames(X),
        "eigenvalues": eig, "noise_floor": float(eig.sum() / P),
        "in_sample": in_s, "held_out": held, "retained": ret,
        "retained_null_median": np.median(null_ret, axis=0),
        "retained_null_high": null_high,
        "n_reproducible": n_rep,
        "n_mostly_real": n_leading(ret >= 0.5),
        "efficiency": float(held[:n_rep].sum() / in_s[:n_rep].sum()) if n_rep else 0.0,
        "quantile": quantile, "n_null": n_null,
    }


def print_pca_reliability(res):
    """Summary table of pca_reliability_test()."""
    q = int(round(100 * res["quantile"]))
    T, P = res["n_frames"], res["n_coords"]
    print(f"\n  PCA reliability  (split-half test, {res['n_null']} null runs, {q}% level)")
    print(f"  {'coordinates / frames':<46}{P} / {T}   (ratio {P / T:.2f})")
    print(f"  {'effective number of independent frames':<46}{res['t_eff']:.2f}")
    print(f"  {'modes the data can resolve (T-1)':<46}{len(res['eigenvalues'])}")
    print(f"  {'softest modes above chance':<46}{res['n_reproducible']} (of {len(res['retained'])} tested)")
    print(f"  {'softest modes with retained fraction >= 0.5':<46}{res['n_mostly_real']}")
    print(f"  {'variance reproduced in the modes above chance':<46}{100 * res['efficiency']:.2f} %")
    print(f"\n  retained fraction = variance in the held-out half / eigenvalue (1 = fully real)")
    print(f"  {'mode':>5}{'retained':>10}{'chance ' + str(q) + '%':>12}")
    n = len(res["retained"])
    for i in sorted({0, 1, 4, 9, n - 1}):
        if i < n:
            print(f"  {i:>5}{res['retained'][i]:>10.2f}{res['retained_null_high'][i]:>12.2f}")

# ------------------------------------------------------------
# TEST 2: does the soft-spot map survive a split of the frames?
# ------------------------------------------------------------

def _rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def _spearman(a, b):
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _top_overlap(a, b, fraction):
    q = max(1, int(round(fraction * len(a))))
    return len(np.intersect1d(np.argsort(a)[::-1][:q], np.argsort(b)[::-1][:q])) / q


def _half_softness(u, masses, n_soft_modes, pr_max):
    """Softness maps (N,) of the first and of the second half of the frames (same recipe as the pipeline)."""
    from analysis.soft_modes.modes import modes_from_displacements, atom_softness_scores
    h = len(u) // 2
    maps = []
    for part in (u[:h], u[h:]):
        m = modes_from_displacements(part - part.mean(axis=0), masses=masses)
        s, _ = atom_softness_scores(m["eigenvalues"], m["eigenvectors"],
                                    n_soft_modes=n_soft_modes, pr_max=pr_max)
        maps.append(s)
    return maps


def _half_msd(u, masses):
    """Mean-square displacement per atom (mass-weighted, no PCA) of the first and of the second half."""
    h = len(u) // 2
    w = np.ones(u.shape[1]) if masses is None else np.asarray(masses, dtype=float)
    out = []
    for part in (u[:h], u[h:]):
        part = part - part.mean(axis=0)
        out.append((w[None, :, None] * part ** 2).sum(axis=2).mean(axis=0))
    return out


def softness_reproducibility_test(u, masses=None, n_soft_modes=20, pr_max=None,
                                  top_fraction=0.05, n_null=100, quantile=0.95, seed=0):
    """
    TEST 2 (see the header of this file for the definitions).

    Parameters
    ----------
    u, masses : as in pca_reliability_test().
    n_soft_modes, pr_max : as in soft_mode_analysis_3d() (use the same values as the pipeline).
    top_fraction : the share of softest atoms used for the overlap (0.05 = the TLS candidate selection).
    n_null, quantile : number of null data sets, and the quantile of them to compare against.

    Returns
    -------
    dict:
      softness_a, softness_b       (N,)  softness map of the first / second half
      spearman                     Spearman rank correlation of the two maps
      top_overlap                  share of the top_fraction softest atoms of A that are also in B's
      msd_a, msd_b                 (N,)  per-atom mean-square displacement (mass-weighted, no PCA) of each half
      spearman_msd, top_overlap_msd     the same two numbers for the MSD maps (the no-PCA reference)
      spearman_softness_msd        mean over the two halves of the Spearman correlation between that half's
                                   softness map and its MSD map (1 = the PCA map is just the MSD map)
      spearman_null, top_overlap_null   (n_null,) the same two numbers on the null data sets
      spearman_null_high, top_overlap_null_high   their `quantile`
      p_spearman, p_top_overlap    (1 + number of null runs >= observed) / (n_null + 1)
      top_fraction, quantile, n_null, n_soft_modes
    """
    u = np.asarray(u, dtype=float)
    T, N, D = u.shape
    sa, sb = _half_softness(u, masses, n_soft_modes, pr_max)
    rho, top = _spearman(sa, sb), _top_overlap(sa, sb, top_fraction)
    ma, mb = _half_msd(u, masses)

    rng = np.random.default_rng(seed)
    rho_null = np.empty(n_null)
    top_null = np.empty(n_null)
    flat = u.reshape(T, N * D)
    for r in range(n_null):
        a, b = _half_softness(_shift_columns(flat, rng).reshape(T, N, D), masses,
                              n_soft_modes, pr_max)
        rho_null[r], top_null[r] = _spearman(a, b), _top_overlap(a, b, top_fraction)

    return {
        "softness_a": sa, "softness_b": sb,
        "spearman": rho, "top_overlap": top,
        "msd_a": ma, "msd_b": mb,
        "spearman_msd": _spearman(ma, mb), "top_overlap_msd": _top_overlap(ma, mb, top_fraction),
        "spearman_softness_msd": 0.5 * (_spearman(sa, ma) + _spearman(sb, mb)),
        "spearman_null": rho_null, "top_overlap_null": top_null,
        "spearman_null_high": float(np.quantile(rho_null, quantile)),
        "top_overlap_null_high": float(np.quantile(top_null, quantile)),
        "p_spearman": float((1 + np.sum(rho_null >= rho)) / (n_null + 1)),
        "p_top_overlap": float((1 + np.sum(top_null >= top)) / (n_null + 1)),
        "top_fraction": top_fraction, "quantile": quantile, "n_null": n_null,
        "n_soft_modes": n_soft_modes,
    }


def print_softness_reproducibility(res):
    """Summary table of softness_reproducibility_test()."""
    q = int(round(100 * res["quantile"]))
    f = res["top_fraction"]
    n_atoms = len(res["softness_a"])
    print(f"\n  Softness map, first half of the frames vs second half  ({n_atoms} atoms, {res['n_null']} null runs)")
    print(f"  {'':<28}{'observed':>10}{'unrelated':>11}{'null median':>13}{'null ' + str(q) + '%':>10}{'p':>7}")
    print(f"  {'Spearman rank correlation':<28}{res['spearman']:>10.2f}{0.0:>11.2f}"
          f"{np.median(res['spearman_null']):>13.2f}{res['spearman_null_high']:>10.2f}{res['p_spearman']:>7.2f}")
    print(f"  {'top ' + str(int(round(100 * f))) + '% overlap (%)':<28}{100 * res['top_overlap']:>10.2f}{100 * f:>11.2f}"
          f"{100 * np.median(res['top_overlap_null']):>13.2f}{100 * res['top_overlap_null_high']:>10.2f}"
          f"{res['p_top_overlap']:>7.2f}")
    print(f"\n  reference without PCA: per-atom mean-square displacement map, first vs second half")
    print(f"  {'Spearman rank correlation':<28}{res['spearman_msd']:>10.2f}")
    print(f"  {'top ' + str(int(round(100 * f))) + '% overlap (%)':<28}{100 * res['top_overlap_msd']:>10.2f}")
    print(f"  {'Spearman, softness map vs MSD map (same half)':<46}{res['spearman_softness_msd']:>6.2f}")