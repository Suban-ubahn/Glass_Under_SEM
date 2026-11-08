# =============================================================
# analysis/soft_modes/pca_convergence.py — temporal correlations of the PCA mode
# amplitudes, effective sample size, and convergence of the PCA with trajectory length.
#
# Complements pca_reliability.py (split-half test, same conventions): there the question
# was "is the eigenvalue still there in frames that did not choose the mode?"; here it is
# "how many independent frames do the 101 frames amount to, and do the modes stop
# changing when more frames are added?"
#
# SETUP (identical to pca_reliability.py). X is the (T, P) matrix, one row per frame,
# X[t, (i,d)] = sqrt(m_i) u[t,i,d], u = displacement from the time-averaged position.
# C = X^T X / T = sum_n lambda_n v_n v_n^T,  a_n(t) = x_t . v_n,  lambda_n = (1/T) sum_t a_n(t)^2.
#
# ------------------------------------------------------------
# PART 1 — AUTOCORRELATION OF THE MODE AMPLITUDES
# ------------------------------------------------------------
# For a series y_t (t = 0..T-1) with sample mean ybar the autocovariance and autocorrelation are
#     c(k) = (1/T) sum_{t=0}^{T-k-1} (y_t - ybar)(y_{t+k} - ybar),     rho(k) = c(k) / c(0).
# The 1/T normalisation ("biased" / Bartlett estimator) is used on purpose: it has a smaller
# variance than the 1/(T-k) one and gives a positive-semidefinite sequence. It is computed
# with an FFT (zero padded to 2T, so no circular wrap-around).
# Two series are analysed for each of the leading modes n:
#     y = a_n(t)      the amplitude itself
#     y = a_n(t)^2    the instantaneous squared amplitude
# The eigenvalue is lambda_n = mean_t a_n(t)^2, so the statistical error of lambda_n, FOR A FIXED
# DIRECTION v_n, is set by the autocorrelation of a_n^2, not of a_n:
#     Var(lambda_hat_n) ~ Var(a_n^2) * tau_int[a_n^2] / T          (see PART 2 for tau_int)
# If a_n is a stationary zero-mean Gaussian process then Cov(a_t^2, a_{t+k}^2) = 2 c(k)^2, hence
#     rho_{a^2}(k) = rho_a(k)^2,   Var(a^2) = 2 lambda^2.
# The ACF of a^2 is therefore never longer-lived than that of a (for positive rho), and it is
# insensitive to the sign of a: an oscillation a ~ cos(wt) has rho_a negative at half a period,
# but rho_{a^2} stays >= 0. The Gaussian prediction rho_a^2 is drawn next to the measured
# rho_{a^2} so that departures (non-Gaussian, e.g. two-level-like amplitudes) are visible.
# The expected relative error of lambda_n, given v_n, is then
#     sqrt( Var(a^2)/lambda^2 * tau_int[a^2] / T )   (eig_rel_std; = sqrt(2 tau / T) for Gaussian a).
#
# ------------------------------------------------------------
# PART 2 — INTEGRATED AUTOCORRELATION TIME AND EFFECTIVE SAMPLE SIZE
# ------------------------------------------------------------
#     tau_int = 1 + 2 sum_{k=1}^{W} rho(k)          (tau_int = 1 for independent samples)
#     T_eff   = T / tau_int                          (variance of a mean over T frames =
#                                                     variance over T_eff independent frames)
# The sum cannot run to k = T-1: rho(k) for large k is pure noise of size ~1/sqrt(T) and the sum
# of that noise has a variance that grows with W. The truncation lag W is chosen by one of
#     "sokal"           (default) Sokal's automatic window (Sokal 1997): the smallest W with
#                       W >= c * tau_int(W), c = window_c = 5. A positive sum that is still
#                       growing keeps the window open; if no W <= max_lag satisfies it the
#                       estimate is flagged window_not_converged.
#     "geyer"           Geyer's initial monotone sequence: sum the pairs Gamma_m = rho(2m) + rho(2m+1)
#                       while they stay positive (and force them non-increasing);
#                       tau_int = -1 + 2 sum_m Gamma_m. Meant for reversible, non-oscillating
#                       series; the pairing absorbs a single noisy negative lag.
#     "first_negative"  sum up to the first negative rho(k): the rule of
#                       pca_reliability._effective_frames (which pools all coordinates).
# The statistical error of tau_int itself (Madras & Sokal 1988) is
#     relative error = sqrt( 2 (2W + 1) / T )
# so with T = 101 and W = 5 it is ~ 47 %: T_eff is an order-of-magnitude estimate here, never a precise number.
#
# HANDLING OSCILLATORY / NEGATIVE ACFs. tau_int is the zero-frequency value of the spectrum of y; a
# series that oscillates (or alternates) has negative lobes in rho that cancel part of the positive
# sum and can drive tau_int below 1, i.e. "T_eff > T". That is a legitimate property of the
# true spectrum only if the ACF is known precisely; from ~100 frames it is mostly noise. Therefore
#   * T_eff is capped at T (tau_used = max(tau_int, 1)); the raw values are kept (tau_raw, t_eff_raw);
#   * an estimate is FLAGGED (and res[...]["reliable"] is False) when any of these holds:
#       window_not_converged   Sokal / Geyer / first-negative rule found no window before max_lag
#       tau_below_1            tau_int < 1: negative lags dominate the sum
#       oscillatory            rho has >= 2 sign changes among the lags with |rho| > band
#       negative_lobe          rho < -band at >= 2 consecutive lags (anticorrelation)
#       short_series           T < min_T_over_tau * tau_used (too few correlation times)
#       large_error            relative error of tau_int > max_rel_err
#       constant_series        zero variance
#     band = band_z / sqrt(T) is the white-noise 95 % band of rho(k) (heuristic: the real band
#     is wider for correlated series and 5 % of lags exceed it by chance).
#   Do not use a flagged T_eff to rescale error bars; use it only as an indication.
#
# AR(1) COMPARISON. Fit y_t = phi y_{t-1} + noise with phi = rho(1). Then rho(k) = phi^k,
#     tau_AR1 = (1 + phi) / (1 - phi),     T_eff,AR1 = T (1 - phi) / (1 + phi).
# This uses only the first lag, so it is stable but wrong when rho is not geometric. It is flagged
#     phi_nonpositive   phi <= 0 (the formula then gives T_eff >= T; capped at T)
#     phi_near_1        phi >= 0.99 (formula diverges)
#     t_eff_below_1     T (1-phi)/(1+phi) < 1: fewer than one independent frame (phi > (T-1)/(T+1));
#                       the series is a slow drift / one smooth pattern, not a stationary AR(1) process
#     not_geometric     max_{k <= ar1_lags} |rho(k) - phi^k| > band
# phi itself is biased low by ~ (1 + 3 phi)/T for short series (Kendall).
#
# IN-SAMPLE CAVEAT -- READ BEFORE USING THE PER-MODE T_eff. The modes v_n were selected on the same T
# frames whose amplitudes are analysed, and with P >> T the leading amplitudes are partly fitted
# noise (see pca_reliability.py). For TEMPORAL correlation this has a specific, large effect: if the
# coordinates have temporal correlation, the leading principal components of a (T, P) matrix with
# P >> T are the leading eigenvectors of the T x T temporal covariance, i.e. the SMOOTHEST,
# slowest patterns in time (cosine-like, with a half period or so over the whole run). PCA picks
# them because they carry the most variance, whether or not any collective motion exists. Check
# (pure noise, no collective modes, P = 6000, T = 101, every coordinate an AR(1) series):
#       true lag-one correlation of each coordinate     0.0      0.3      0.6
#       lag-one correlation of the leading a_n(t)       -0.1..0.25  ~0.95    ~0.99
#       AR(1) T_eff of the leading a_n(t)               60..100     ~2.4     ~0.5   (true: 101, 54, 25)
# So the ACF / tau_int / T_eff of a leading amplitude measures "how smooth is the pattern PCA
# fitted", which is NOT the number of independent frames of the data, and is biased low by
# up to a factor ~20 as soon as the frames are correlated at all. Two consequences:
#   * Compare with the shifted-column NULL (n_null > 0 in pca_temporal_analysis): same
#     coordinates, same per-coordinate autocorrelation, no collective modes, same selection
#     effect. An observed ACF / T_eff that lies inside the null range carries no information
#     about the dynamics of the mode (flag-like boolean: res["null"]["within"][...]).
#   * The pooled T_eff (ACF pooled over all coordinates, pca_reliability._effective_frames, returned as
#     "pooled_t_eff") is not affected by the selection and is the better estimate of the number of
#     independent frames of the data (check above: 58 vs true 54 for phi = 0.3); the per-mode
#     values are the ones to use only for "how long-lived is the fitted amplitude" or for
#     the error of lambda_n of a direction fixed in advance.
# The amplitudes of different modes are exactly uncorrelated at lag 0 (sum_t a_n a_m = 0), not at other lags.
#
# ------------------------------------------------------------
# PART 3 — CONVERGENCE OF THE PCA WITH THE NUMBER OF FRAMES
# ------------------------------------------------------------
# The PCA is repeated on the first L frames, L in `lengths` (default 20, 40, 60, 80, 101). Every
# segment is re-centred on its own mean (recenter=True, as the halves of pca_reliability.py)
# and normalised by L, so a segment with L frames has at most L-1 non-zero eigenvalues. The
# longest segment is the REFERENCE. Compared are
#   (a) EIGENVALUES. lambda_n(L) vs L. Mode n of a short segment need not be mode n of the
#       reference (near-degenerate modes swap, modes rotate), so eigenvalues are also read off
#       at the MATCHED mode, below.
#   (b) EIGENVECTORS by overlap. For reference mode i and segment mode j
#           O_ij = | v_i^ref . v_j^(L) |      (both unit vectors, sign is arbitrary)
#       and mode i is matched to j*(i) = argmax_j O_ij over ALL non-zero modes of the segment
#       (matching="max"; several reference modes may map to one segment mode, counted in
#       n_duplicate_matches) or one-to-one by maximising the summed overlap
#       (matching="hungarian"). Reported: matched overlap, the second-best overlap (a small gap
#       to the best marks a degenerate / ambiguous match), the overlap by INDEX (O_ii, for
#       contrast), the matched eigenvalue, and the span capture ||P_L v_i||^2 = sum_j O_ij^2,
#       the squared length of the part of v_i that lies in the span of the segment's modes at all.
#       sqrt(span capture) is an upper bound for the overlap of v_i with ANY vector the segment can
#       produce, so it separates "the segment cannot represent the mode" (few frames) from "the
#       segment represents it but ranks it differently".
#   (c) EIGENSPACES. U_ref, U_L: P x M matrices of the M = subspace_dim leading eigenvectors.
#       The projection matrices are P_ref = U_ref U_ref^T, P_L = U_L U_L^T (never formed: P x P).
#       Their overlap is
#           S(L) = tr(P_ref P_L) / M = ||U_ref^T U_L||_F^2 / M = (1/M) sum_i cos^2(theta_i),
#       the cosines of the principal angles theta_i being the singular values of U_ref^T U_L.
#       S = 1: identical subspaces; two random M-dimensional subspaces of R^P give M/P. Unlike
#       single vectors, S is unaffected by rotations inside the subspace, so it is the right measure
#       when modes within the leading block are nearly degenerate (a gap between mode M and M+1
#       is still needed). The chordal distance sqrt(M - sum cos^2) is stored too.
#       The same S is computed between each segment and the NEXT longer one (vs_next), which does not
#       use the reference and therefore does not end at a trivial 1.
#   NULL (optional, n_null > 0). The same procedure on data in which every column is shifted
#       cyclically by its own random amount (pca_reliability._shift_columns: independent
#       coordinates, same variance and autocorrelation), including the nesting of the segments.
#       It shows how large the overlaps get from nested sampling alone.
#
# LIMITATIONS (read before interpreting).
#   * NESTED SEGMENTS ARE NOT INDEPENDENT. The L = 60 segment contains the L = 40 one. Agreement
#     between nested segments is partly shared data, and the last point (L = T) compared with the
#     reference is 1 by construction. Convergence shown here means "adding frames no longer changes
#     the result", not "the result is correct". A mode could be stable across all lengths and still
#     be a feature of this one trajectory (e.g. a slow drift, or an aging transient).
#     Independent evidence needs held-out frames (pca_reliability.py) or a second trajectory.
#   * The frames are autocorrelated (PART 2), so the number of independent frames in a segment is ~ L / tau,
#     smaller than L.
#   * Prefix segments are all taken from the START of the run; a slowly drifting system makes
#     short segments differ from the long one for physical, not statistical, reasons.
#   * With P = 20250 >> L every segment can only represent <= L-1 directions; overlaps of
#     the reference with a segment are bounded by the span capture.
# =============================================================

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis.soft_modes.modes import modes_from_displacements
from analysis.soft_modes.pca_reliability import _flat, _pca, _shift_columns, _effective_frames


# ------------------------------------------------------------
# PART 1/2: autocorrelation, integrated autocorrelation time, T_eff
# ------------------------------------------------------------

def autocorrelation(y, max_lag=None):
    """
    Normalised autocorrelation rho(k), k = 0..max_lag, of every column of y (biased 1/T estimator).

    y : (T,) or (T, M).  Returns (max_lag + 1,) or (max_lag + 1, M); rho(0) = 1.
    A constant series has no defined autocorrelation: its column is NaN.
    """
    y = np.asarray(y, dtype=float)
    squeeze = y.ndim == 1
    if squeeze:
        y = y[:, None]
    T = y.shape[0]
    L = T - 1 if max_lag is None else min(int(max_lag), T - 1)
    d = y - y.mean(axis=0)
    F = np.fft.rfft(d, n=2 * T, axis=0)
    acov = np.fft.irfft(F.real ** 2 + F.imag ** 2, n=2 * T, axis=0)[: L + 1] / T
    c0 = acov[0]
    scale = np.max(np.abs(y), axis=0) ** 2 + np.finfo(float).tiny
    ok = c0 > 1e-24 * scale
    rho = np.full_like(acov, np.nan)
    rho[:, ok] = acov[:, ok] / c0[ok]
    return rho[:, 0] if squeeze else rho


def integrated_autocorr_time(rho, method="sokal", window_c=5.0):
    """
    tau_int = 1 + 2 sum_{k=1}^W rho(k) with the truncation lag W chosen by `method`
    ("sokal", "geyer" or "first_negative"; see the file header).

    rho : (L+1,) autocorrelation, rho[0] = 1; lags up to L are available to the window search.
    Returns dict: tau (raw, may be < 1), window (last lag summed), converged (bool: the rule
    closed the window before the lags ran out).
    """
    L = len(rho) - 1
    r = np.asarray(rho[1:], dtype=float)
    if L < 1 or not np.all(np.isfinite(r)):
        return {"tau": np.nan, "window": 0, "converged": False}

    if method == "sokal":
        tau_w = 1.0 + 2.0 * np.cumsum(r)                    # tau_w[W-1] = tau_int(W)
        ok = np.arange(1, L + 1) >= window_c * tau_w
        converged = bool(ok.any())
        idx = int(np.argmax(ok)) if converged else L - 1
        return {"tau": float(tau_w[idx]), "window": idx + 1, "converged": converged}

    if method == "first_negative":
        neg = np.where(r < 0)[0]
        converged = len(neg) > 0
        W = int(neg[0]) if converged else L                 # lags 1..W are positive
        return {"tau": float(1.0 + 2.0 * r[:W].sum()), "window": W, "converged": converged}

    if method == "geyer":
        full = np.concatenate([[1.0], r])                   # rho(0..L)
        n_pairs = (L + 1) // 2
        gam = full[0:2 * n_pairs:2] + full[1:2 * n_pairs:2]  # rho(2m) + rho(2m+1)
        nonpos = np.where(gam <= 0)[0]
        converged = len(nonpos) > 0
        m_stop = int(nonpos[0]) if converged else n_pairs
        if m_stop == 0:
            return {"tau": 1.0, "window": 0, "converged": converged}
        g = np.minimum.accumulate(gam[:m_stop])             # initial MONOTONE sequence
        return {"tau": float(-1.0 + 2.0 * g.sum()), "window": 2 * m_stop - 1,
                "converged": converged}

    raise ValueError(f"unknown window method {method!r} (use 'sokal', 'geyer' or 'first_negative')")


def _series_flags(rho, T, tau_raw, window, converged, rel_err, band,
                  min_T_over_tau, max_rel_err):
    if not np.isfinite(tau_raw):
        return ["constant_series"]
    flags = []
    if not converged:
        flags.append("window_not_converged")
    if tau_raw < 1.0:
        flags.append("tau_below_1")
    sig = rho[1:]
    big = np.abs(sig) > band
    n_flips = int(np.count_nonzero(np.diff(np.sign(sig[big])))) if big.sum() > 1 else 0
    if n_flips >= 2:
        flags.append("oscillatory")
    elif np.any((sig[:-1] < -band) & (sig[1:] < -band)):
        flags.append("negative_lobe")
    if T < min_T_over_tau * max(tau_raw, 1.0):
        flags.append("short_series")
    if rel_err > max_rel_err:
        flags.append("large_error")
    return flags


def _ar1(rho, T, band, ar1_lags):
    phi = float(rho[1])
    if not np.isfinite(phi):
        return {"phi": np.nan, "tau": np.nan, "t_eff_raw": np.nan, "t_eff": np.nan,
                "misfit": np.nan, "flags": ["constant_series"]}
    flags = []
    if phi <= 0:
        flags.append("phi_nonpositive")
    tau = (1 + phi) / (1 - phi) if phi < 1 else np.inf
    if phi >= 0.99:
        flags.append("phi_near_1")
    if 0 < phi < 1 and T / tau < 1.0:
        flags.append("t_eff_below_1")                       # fewer than one independent frame: not a usable number
    K = min(ar1_lags, len(rho) - 1)
    k = np.arange(1, K + 1)
    misfit = float(np.max(np.abs(rho[1:K + 1] - np.sign(phi) ** k * np.abs(phi) ** k)))
    if misfit > band:
        flags.append("not_geometric")
    return {"phi": phi, "tau": tau, "t_eff_raw": T / tau if tau > 0 else np.inf,
            "t_eff": T / max(tau, 1.0), "misfit": misfit, "flags": flags}


def _series_report(y, max_lag, method, window_c, band, min_T_over_tau, max_rel_err, ar1_lags):
    """ACF, tau_int, T_eff (window-based and AR(1)) of every column of y (T, M)."""
    T, M = y.shape
    rho = autocorrelation(y, max_lag)                      # (L+1, M)
    out = {k: [] for k in ("tau_raw", "tau", "window", "window_converged", "t_eff_raw", "t_eff",
                           "tau_rel_err", "flags")}
    ar = [_ar1(rho[:, m], T, band, ar1_lags) for m in range(M)]
    for m in range(M):
        r = integrated_autocorr_time(rho[:, m], method, window_c)
        tau_raw = r["tau"]
        tau_used = max(tau_raw, 1.0) if np.isfinite(tau_raw) else np.nan
        rel_err = np.sqrt(2.0 * (2 * r["window"] + 1) / T) if np.isfinite(tau_raw) else np.nan
        out["tau_raw"].append(tau_raw)
        out["tau"].append(tau_used)
        out["window"].append(r["window"])
        out["window_converged"].append(r["converged"])
        out["t_eff_raw"].append(T / tau_raw if np.isfinite(tau_raw) and tau_raw > 0 else np.inf)
        out["t_eff"].append(T / tau_used if np.isfinite(tau_used) else np.nan)
        out["tau_rel_err"].append(rel_err)
        out["flags"].append(_series_flags(rho[:, m], T, tau_raw, r["window"], r["converged"],
                                          rel_err, band, min_T_over_tau, max_rel_err))
    res = {k: (v if k == "flags" else np.array(v)) for k, v in out.items()}
    res["acf"] = rho.T                                      # (M, L+1)
    res["reliable"] = np.array([len(f) == 0 for f in res["flags"]])
    for key in ("phi", "tau", "t_eff_raw", "t_eff", "misfit"):
        res["ar1_" + key] = np.array([a[key] for a in ar])
    res["ar1_flags"] = [a["flags"] for a in ar]
    return res


def amplitude_acf_analysis(amplitudes, eigenvalues=None, n_modes=10, max_lag=None,
                           window_method="sokal", window_c=5.0, min_T_over_tau=20.0,
                           max_rel_err=0.3, band_z=1.96, ar1_lags=10):
    """
    ACF, integrated autocorrelation time and effective sample size of the amplitudes a_n(t) and of
    their squares a_n(t)^2 (see the file header, PARTS 1 and 2), from given amplitudes.

    Parameters
    ----------
    amplitudes : (T, K) mode amplitudes, K >= n_modes, columns ordered softest first
        (modes_from_displacements()["amplitudes"]).
    eigenvalues : (K,) optional, only copied to the result.
    n_modes : number of leading modes analysed.
    max_lag : largest lag computed and offered to the window rule (default T // 2).
    window_method, window_c : truncation rule of the sum ("sokal", "geyer", "first_negative") and the
        constant c of Sokal's rule W >= c tau.
    min_T_over_tau : flag short_series when T < min_T_over_tau * tau.
    max_rel_err : flag large_error when sqrt(2(2W+1)/T) exceeds it.
    band_z : half width of the white-noise band of rho, in units of 1/sqrt(T).
    ar1_lags : number of lags over which the AR(1) shape rho(k) = phi^k is checked.

    Returns dict (M = n_modes, L = max_lag, T = frames):
      lags (L+1,), band, n_frames, n_modes, eigenvalues (M,)
      amplitude, squared : dicts for a_n and a_n^2, each with
          acf (M, L+1), tau_raw, tau, window, window_converged, t_eff_raw, t_eff,
          tau_rel_err, flags (list of lists), reliable (M,) bool,
          ar1_phi, ar1_tau, ar1_t_eff_raw, ar1_t_eff, ar1_misfit, ar1_flags
      gaussian_acf_squared (M, L+1)   acf of a_n squared, i.e. the Gaussian prediction for rho of a^2
      eig_rel_std (M,)               sqrt(Var(a^2)/lambda^2 * tau[a^2] / T): error of lambda_n given v_n
      t_eff_eigenvalue (M,)          = squared["t_eff"]
      params
    """
    A = np.asarray(amplitudes, dtype=float)
    T = A.shape[0]
    M = min(n_modes, A.shape[1])
    A = A[:, :M]
    L = T // 2 if max_lag is None else min(int(max_lag), T - 1)
    band = band_z / np.sqrt(T)
    kw = dict(max_lag=L, method=window_method, window_c=window_c, band=band,
              min_T_over_tau=min_T_over_tau, max_rel_err=max_rel_err, ar1_lags=ar1_lags)

    amp = _series_report(A, **kw)
    sq = _series_report(A ** 2, **kw)
    var_ratio = np.var(A ** 2, axis=0) / np.mean(A ** 2, axis=0) ** 2
    return {
        "n_frames": T, "n_modes": M, "lags": np.arange(L + 1), "max_lag": L, "band": band,
        "eigenvalues": None if eigenvalues is None else np.asarray(eigenvalues)[:M],
        "amplitude": amp, "squared": sq,
        "gaussian_acf_squared": amp["acf"] ** 2,
        "eig_rel_std": np.sqrt(var_ratio * sq["tau"] / T),
        "t_eff_eigenvalue": sq["t_eff"],
        "params": dict(window_method=window_method, window_c=window_c,
                       min_T_over_tau=min_T_over_tau, max_rel_err=max_rel_err,
                       band_z=band_z, ar1_lags=ar1_lags),
    }


_NULL_FIELDS = ("acf", "ar1_phi", "ar1_t_eff", "t_eff")


def _temporal_null(u, masses, n_modes, observed, n_null, quantile, seed, acf_kwargs):
    """
    The same PCA + amplitude_acf_analysis on n_null data sets in which every coordinate is shifted
    cyclically in time by its own random amount (pca_reliability._shift_columns): independent
    coordinates, each with its own variance and autocorrelation, hence NO collective modes, but the
    same selection of the leading modes from T frames (see IN-SAMPLE CAVEAT in the file header).

    Returns dict: n_null, quantile, and for series in ("amplitude", "squared") and every field in
    _NULL_FIELDS the arrays <field>_median / _low / _high (median and the (1-q, q) quantiles over the
    runs; shape of the field in `observed`), and "within": bool (M,) arrays "<series>_<field>" saying
    that the observed value lies inside [low, high] (the observed estimate is then not distinguishable
    from what selection alone produces; NaN-safe: False if the observed value is NaN).
    """
    T, N, D = u.shape
    rng = np.random.default_rng(seed)
    flat = np.asarray(u, dtype=float).reshape(T, N * D)
    runs = {s: {f: [] for f in _NULL_FIELDS} for s in ("amplitude", "squared")}
    for _ in range(n_null):
        un = _shift_columns(flat, rng).reshape(T, N, D)
        m = modes_from_displacements(un, masses=masses, n_modes=n_modes)
        r = amplitude_acf_analysis(m["amplitudes"], n_modes=n_modes, **acf_kwargs)
        for s in runs:
            for f in _NULL_FIELDS:
                runs[s][f].append(r[s][f])
    out = {"n_null": n_null, "quantile": quantile, "within": {}}
    for s, d in runs.items():
        out[s] = {}
        for f, lst in d.items():
            a = np.asarray(lst, dtype=float)
            out[s][f + "_median"] = np.nanmedian(a, axis=0)
            out[s][f + "_low"] = np.nanquantile(a, 1.0 - quantile, axis=0)
            out[s][f + "_high"] = np.nanquantile(a, quantile, axis=0)
            if f != "acf":
                obs = np.asarray(observed[s][f], dtype=float)
                out["within"][f"{s}_{f}"] = (obs >= out[s][f + "_low"]) & (obs <= out[s][f + "_high"])
    return out


def pca_temporal_analysis(u, masses=None, n_modes=10, n_null=0, null_quantile=0.95, seed=0,
                          **acf_kwargs):
    """
    Full-trajectory PCA (as the pipeline: modes_from_displacements) followed by
    amplitude_acf_analysis() on its leading amplitudes.

    u, masses : as in pca_reliability_test(). acf_kwargs : see amplitude_acf_analysis().
    n_null : number of shifted-column null data sets (0 = none), see _temporal_null() and the
        IN-SAMPLE CAVEAT in the file header: the per-mode ACF / T_eff of a P >> T PCA is dominated by
        the selection of smooth temporal patterns and must be read against this null.
        null_quantile : upper quantile q; the null range is its (1-q, q) interval. seed : its random seed.
    Adds "pooled_t_eff": the T_eff of pca_reliability._effective_frames (ACF pooled over ALL
    coordinates, first-negative truncation) for comparison with the per-mode values, and
    res["null"] (see _temporal_null) when n_null > 0.
    """
    modes = modes_from_displacements(u, masses=masses, n_modes=n_modes)
    res = amplitude_acf_analysis(modes["amplitudes"], modes["eigenvalues"],
                                 n_modes=n_modes, **acf_kwargs)
    res["pooled_t_eff"] = _effective_frames(_flat(u, masses))
    if n_null > 0:
        res["null"] = _temporal_null(u, masses, n_modes, res, n_null, null_quantile, seed, acf_kwargs)
    return res


def print_amplitude_acf(res):
    """Summary table of amplitude_acf_analysis() / pca_temporal_analysis()."""
    T, M = res["n_frames"], res["n_modes"]
    p = res["params"]
    print(f"\n  Autocorrelation of the mode amplitudes  ({T} frames, window rule '{p['window_method']}', "
          f"c = {p['window_c']:g}, lags <= {res['max_lag']}, white-noise band +-{res['band']:.2f})")
    if "pooled_t_eff" in res:
        print(f"  {'T_eff pooled over all coordinates (pca_reliability)':<56}{res['pooled_t_eff']:.1f}")
    print(f"  T_eff is capped at T = {T}; '*' marks an unreliable estimate (flags below).")
    head = f"  {'mode':>4}"
    for name in ("a", "a^2"):
        head += f"{'tau':>7}{'W':>4}{'T_eff':>8}{'AR1 T_eff':>11}" + ("   |" if name == "a" else "")
    print(head + f"{'  lambda rel.err':>17}")
    for n in range(M):
        row = f"  {n:>4}"
        for key in ("amplitude", "squared"):
            s = res[key]
            star = "" if s["reliable"][n] else "*"
            row += (f"{s['tau_raw'][n]:>7.2f}{s['window'][n]:>4d}{s['t_eff'][n]:>7.1f}{star:<1}"
                    f"{s['ar1_t_eff'][n]:>11.1f}")
            if key == "amplitude":
                row += "   |"
        print(row + f"{100 * res['eig_rel_std'][n]:>15.0f} %")
    print("\n  flags (a^2 series, the one that matters for lambda_n; a series in brackets):")
    for n in range(M):
        fs, fa = res["squared"]["flags"][n], res["amplitude"]["flags"][n]
        a1s, a1a = res["squared"]["ar1_flags"][n], res["amplitude"]["ar1_flags"][n]
        print(f"  {n:>4}  {', '.join(fs) or '-':<46} [{', '.join(fa) or '-'}]"
              f"   AR1: {', '.join(a1s) or '-'} [{', '.join(a1a) or '-'}]")
    print("  lambda rel.err: sqrt(Var(a^2)/lambda^2 * tau[a^2] / T), the error of lambda_n for a FIXED direction;"
          "\n  it does not contain the selection bias of the mode (see pca_reliability).")
    null = res.get("null")
    if null is None:
        print("  NOTE: no null run (n_null = 0). For P >> T the per-mode ACF / T_eff mostly reflects which smooth temporal"
              "\n  pattern PCA selected, not the data's correlation time: compare with the shifted-column null"
              "\n  (pca_temporal_analysis(..., n_null > 0)) before reading any of the numbers above.")
        return
    q = int(round(100 * null["quantile"]))
    lo_q = 100 - q
    print(f"\n  Shifted-column null: {null['n_null']} runs, coordinates independent, NO collective modes, same per-coordinate"
          f"\n  autocorrelation and the same PCA selection: median [{lo_q}%, {q}%] of the null; 'in null' = observed value inside"
          f" that range")
    print(f"  {'mode':>4}{'rho_a(1) obs':>14}{'null':>20}{'in null':>9}"
          f"   |{'T_eff(a^2) obs':>16}{'null':>22}{'in null':>9}")
    for n in range(M):
        a, sq = null["amplitude"], null["squared"]
        w_a, w_s = null["within"]["amplitude_ar1_phi"][n], null["within"]["squared_t_eff"][n]
        print(f"  {n:>4}{res['amplitude']['ar1_phi'][n]:>14.2f}"
              f"{a['ar1_phi_median'][n]:>9.2f} [{a['ar1_phi_low'][n]:.2f},{a['ar1_phi_high'][n]:.2f}]"
              f"{'yes' if w_a else 'NO':>9}"
              f"   |{res['squared']['t_eff'][n]:>16.1f}"
              f"{sq['t_eff_median'][n]:>11.1f} [{sq['t_eff_low'][n]:.1f},{sq['t_eff_high'][n]:.1f}]"
              f"{'yes' if w_s else 'NO':>9}")
    print("  'in null' = yes: the observed ACF / T_eff cannot be told from what the selection of smooth temporal patterns"
          "\n  produces without any collective mode; it then says nothing about the dynamics of that mode. The pooled"
          "\n  T_eff above is not affected by the selection.")


# ------------------------------------------------------------
# PART 3: PCA convergence with the number of frames
# ------------------------------------------------------------

def _segment_modes(X, L, recenter):
    """All non-zero eigenvalues (rank,) and eigenvectors (P, rank) of the first L frames."""
    Xs = X[:L]
    if recenter:
        Xs = Xs - Xs.mean(axis=0)
    rank = min(L - 1 if recenter else L, X.shape[1])
    lam, V = _pca(Xs, rank)
    return lam[:rank], V


def _match(O, matching):
    """Segment mode matched to every reference mode. O: (M, rank) overlaps."""
    if matching == "max":
        return np.argmax(O, axis=1)
    if matching == "hungarian":
        rows, cols = linear_sum_assignment(-O)
        j = np.empty(O.shape[0], dtype=int)
        j[rows] = cols
        return j
    raise ValueError(f"unknown matching {matching!r} (use 'max' or 'hungarian')")


def _subspace_cosines(Ua, Ub):
    """Cosines of the principal angles between span(Ua) and span(Ub) (orthonormal columns)."""
    return np.clip(np.linalg.svd(Ua.T @ Ub, compute_uv=False), 0.0, 1.0)


def _convergence_core(X, lengths, n_modes, subspace_dim, matching, recenter, keep_details):
    n_len = len(lengths)
    M, Ms = n_modes, subspace_dim
    lam_all, V_all = zip(*[_segment_modes(X, L, recenter) for L in lengths])
    ref_lam, V_ref = lam_all[-1], V_all[-1]
    U = [V[:, :Ms] for V in V_all]

    out = {
        "matched_index": np.zeros((n_len, M), dtype=int), "matched_overlap": np.zeros((n_len, M)),
        "second_overlap": np.zeros((n_len, M)), "index_overlap": np.zeros((n_len, M)),
        "span_capture": np.zeros((n_len, M)), "matched_eigenvalue": np.zeros((n_len, M)),
        "leading_eigenvalue": np.zeros((n_len, M)), "n_duplicate_matches": np.zeros(n_len, dtype=int),
        "cos_angles": np.zeros((n_len, Ms)), "cos_angles_next": np.zeros((n_len - 1, Ms)),
    }
    overlaps = []
    for s in range(n_len):
        O = np.abs(V_ref[:, :M].T @ V_all[s])                       # (M, rank_s)
        j = _match(O, matching)
        out["matched_index"][s] = j
        out["matched_overlap"][s] = O[np.arange(M), j]
        # second-best OTHER candidate than the matched one
        masked = O.copy()
        masked[np.arange(M), j] = -1.0
        out["second_overlap"][s] = masked.max(axis=1) if O.shape[1] > 1 else 0.0
        out["index_overlap"][s] = np.diag(O[:, :M]) if O.shape[1] >= M else np.nan
        out["span_capture"][s] = (O ** 2).sum(axis=1)
        out["matched_eigenvalue"][s] = lam_all[s][j]
        out["leading_eigenvalue"][s] = lam_all[s][:M]
        out["n_duplicate_matches"][s] = M - len(np.unique(j))
        out["cos_angles"][s] = _subspace_cosines(U[-1], U[s])
        if s < n_len - 1:
            out["cos_angles_next"][s] = _subspace_cosines(U[s], U[s + 1])
        if keep_details:
            overlaps.append(O)
    out["subspace_overlap"] = (out["cos_angles"] ** 2).mean(axis=1)
    out["chordal_distance"] = np.sqrt(np.maximum(Ms - (out["cos_angles"] ** 2).sum(axis=1), 0.0))
    out["subspace_overlap_next"] = (out["cos_angles_next"] ** 2).mean(axis=1)
    out["total_variance"] = np.array([lam.sum() for lam in lam_all])
    out["reference_eigenvalues"] = ref_lam[:M]
    if keep_details:
        out["eigenvalues"] = list(lam_all)
        out["overlap_matrices"] = overlaps
    return out


def pca_convergence(u, masses=None, lengths=(20, 40, 60, 80, 101), n_modes=10, subspace_dim=5,
                    matching="max", recenter=True, n_null=20, quantile=0.95, seed=0):
    """
    Repeat the PCA on the first L frames, L in `lengths`, and compare with the longest segment
    (see the file header, PART 3, for the definitions and limitations).

    Parameters
    ----------
    u, masses : as in pca_reliability_test().
    lengths : segment lengths in frames (clipped to T, sorted, duplicates removed). The longest one
        is the reference.
    n_modes : number of leading reference modes followed (eigenvalues, matching).
    subspace_dim : M, dimension of the leading eigenspace compared (<= n_modes).
    matching : "max" (each reference mode -> segment mode of maximum overlap, may repeat) or
        "hungarian" (one-to-one, maximal total overlap).
    recenter : subtract each segment's own mean (default, like the halves of pca_reliability).
    n_null : number of null data sets (0 = none); quantile : upper quantile of the null stored.

    Returns dict (n_len = number of lengths; M = n_modes; Msub = subspace_dim):
      lengths (n_len,), n_frames, n_coords, n_modes, subspace_dim, matching, recenter
      eigenvalues            list of (rank_L,) arrays, all non-zero eigenvalues of each segment
      total_variance (n_len,)         sum of the eigenvalues of each segment
      reference_eigenvalues (M,)
      leading_eigenvalue (n_len, M)   eigenvalue at INDEX n
      matched_index / matched_overlap / second_overlap / index_overlap / span_capture /
      matched_eigenvalue (n_len, M)   see the file header; indices refer to the segment's own order
      n_duplicate_matches (n_len,)    reference modes that share their segment mode with another
      overlap_matrices       list of (M, rank_L) arrays |v_i^ref . v_j^(L)|
      cos_angles (n_len, Msub), subspace_overlap (n_len,), chordal_distance (n_len,)   vs the reference
      cos_angles_next (n_len-1, Msub), subspace_overlap_next (n_len-1,)   segment k vs segment k+1
      subspace_random_baseline        Msub / P, the overlap of two random subspaces
      null_subspace_overlap (n_null, n_len), null_subspace_overlap_next (n_null, n_len-1),
      null_matched_overlap (n_null, n_len)   (mean over modes), null_quantile
    """
    X = _flat(u, masses)
    T, P = X.shape
    lengths = np.unique(np.minimum(np.asarray(lengths, dtype=int), T))
    min_rank = int(lengths[0]) - (1 if recenter else 0)
    if len(lengths) < 2:
        raise ValueError("pca_convergence needs at least two different segment lengths")
    if not 1 <= subspace_dim <= n_modes:
        raise ValueError("need 1 <= subspace_dim <= n_modes")
    if n_modes > min_rank:
        raise ValueError(f"n_modes = {n_modes} exceeds the {min_rank} non-zero modes of the "
                         f"shortest segment ({int(lengths[0])} frames)")

    core = _convergence_core(X, lengths, n_modes, subspace_dim, matching, recenter, True)
    res = {"lengths": lengths, "n_frames": T, "n_coords": P, "n_modes": n_modes,
           "subspace_dim": subspace_dim, "matching": matching, "recenter": recenter,
           "subspace_random_baseline": subspace_dim / P, "quantile": quantile, "n_null": n_null}
    res.update(core)

    if n_null > 0:
        rng = np.random.default_rng(seed)
        n_len = len(lengths)
        sub, nxt, mat = (np.empty((n_null, n_len)), np.empty((n_null, n_len - 1)),
                         np.empty((n_null, n_len)))
        for r in range(n_null):
            c = _convergence_core(_shift_columns(X, rng), lengths, n_modes, subspace_dim,
                                  matching, recenter, False)
            sub[r], nxt[r], mat[r] = (c["subspace_overlap"], c["subspace_overlap_next"],
                                      c["matched_overlap"].mean(axis=1))
        res.update(null_subspace_overlap=sub, null_subspace_overlap_next=nxt,
                   null_matched_overlap=mat)
    return res


def print_pca_convergence(res):
    """Summary table of pca_convergence()."""
    L, M, Ms = res["lengths"], res["n_modes"], res["subspace_dim"]
    q = int(round(100 * res["quantile"]))
    has_null = res["n_null"] > 0
    print(f"\n  PCA convergence with the number of frames  (reference: first {L[-1]} frames, "
          f"matching '{res['matching']}', {res['n_coords']} coordinates)")
    print("  Nested segments share frames: values near the reference length are 1 by construction"
          " and are not independent evidence.")
    print(f"\n  eigenvalue of the matched mode / reference eigenvalue  (modes 0..{M - 1})")
    print(f"  {'frames':>7}" + "".join(f"{n:>7d}" for n in range(M)) + f"{'sum(lam)/ref':>14}")
    for s, l in enumerate(L):
        ratio = res["matched_eigenvalue"][s] / res["reference_eigenvalues"]
        print(f"  {l:>7d}" + "".join(f"{x:>7.2f}" for x in ratio)
              + f"{res['total_variance'][s] / res['total_variance'][-1]:>14.2f}")
    print(f"\n  overlap |v_ref . v_segment| of the matched mode  (index-matched overlap in brackets would be"
          f" the naive reading)")
    print(f"  {'frames':>7}" + "".join(f"{n:>7d}" for n in range(M)) + f"{'dups':>6}")
    for s, l in enumerate(L):
        print(f"  {l:>7d}" + "".join(f"{x:>7.2f}" for x in res["matched_overlap"][s])
              + f"{res['n_duplicate_matches'][s]:>6d}")
    print(f"  {'by index':>7}   (frames = {L[0]}): "
          + " ".join(f"{x:.2f}" for x in res["index_overlap"][0]))
    print(f"\n  leading {Ms}-dimensional eigenspace: overlap tr(P_ref P_L)/{Ms}"
          f"  (random subspaces: {res['subspace_random_baseline']:.1e})")
    head = f"  {'frames':>7}{'vs ref':>9}{'min cos':>9}{'vs next':>9}"
    if has_null:
        head += f"{'null ' + str(q) + '% (ref)':>16}{'null ' + str(q) + '% (next)':>17}"
    print(head)
    for s, l in enumerate(L):
        nxt = f"{res['subspace_overlap_next'][s]:>9.2f}" if s < len(L) - 1 else f"{'':>9}"
        row = f"  {l:>7d}{res['subspace_overlap'][s]:>9.2f}{res['cos_angles'][s].min():>9.2f}{nxt}"
        if has_null:
            row += f"{np.quantile(res['null_subspace_overlap'][:, s], res['quantile']):>16.2f}"
            row += (f"{np.quantile(res['null_subspace_overlap_next'][:, s], res['quantile']):>17.2f}"
                    if s < len(L) - 1 else "")
        print(row)


def save_results(res, path):
    """Write the numeric / string arrays of a (nested) result dict to a compressed .npz file."""
    flat = {}

    def add(prefix, d):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                add(key + "/", v)
            elif v is None:
                continue
            elif isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                flat[key] = np.array([", ".join(x) for x in v])
            elif isinstance(v, list):
                if len({np.shape(x) for x in v}) == 1:
                    flat[key] = np.array(v)
                else:                                   # ragged list (one array per segment): key/0, key/1, ...
                    for i, x in enumerate(v):
                        flat[f"{key}/{i}"] = np.asarray(x)
            else:
                flat[key] = np.asarray(v)

    add("", res)
    flat = {k: v for k, v in flat.items() if v.dtype != object}
    np.savez_compressed(path, **flat)
    print(f"Saved: {path}")
