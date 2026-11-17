# =============================================================
# analysis/soft_modes/local_softness.py -- per-atom, per-frame softness from the
# local (moving-window) PCA
#
# The local PCA (local_pca.py) decomposes the covariance of every window (a centre
# atom and its neighbours) but stores no per-atom softness. This module defines one,
# with exactly the conventions of the full-system softness_trajectory() (modes.py),
# so that the two (T, N) series can be put through the same comparisons.
#
# DEFINITION. For a window w (n atoms, K = n*D coordinates) and a frame t:
#     X_w(t) = the frames 0..t of the window's columns of the mass-weighted displacement
#              matrix (displacements from the TIME-AVERAGED structure of the whole
#              trajectory, as in softness_trajectory() and local_pca())
#     C_w(t) = X_w(t)^T X_w(t) / (t + 1)      K x K
#     C_w(t) v_n = lambda_n v_n               ALL K modes (no n_soft_modes cap, as in
#                                             softness_trajectory() with n_soft_modes=None)
#     p^w_{i,n}(t) = sum_d (v_n[i,d])^2       real-space share of mode n on atom i of the
#                                             window: the mass weighting is undone and the
#                                             vector re-normalised (modes.atom_softness_scores)
#     s^w_i(t) = sum_n lambda_n p^w_{i,n}(t)  softness of atom i inside window w
# and the softness of atom i at frame t is the mean of s^w_i(t) over EVERY window that
# contains it (its own window and the windows of its neighbours):
#     S_i(t) = (1 / n_i) sum_{w : i in w} s^w_i(t),     n_i = number of windows containing i
# Atoms that are in no window (windows of a spacing > 1 between the centre atoms may not
# cover every atom) get NaN.
#
# CHECK. For a single window holding all atoms, S_i(t) is the softness of
# softness_trajectory(n_soft_modes=None) (verified by the test of this module).
# =============================================================

import numpy as np

from analysis.soft_modes.local_pca import window_columns
from analysis.soft_modes.pca_reliability import _flat


def local_softness_trajectory(u, masses, windows, chunk=256):
    """
    Per-atom, per-frame softness of the local PCA, averaged over all windows that contain the atom
    (see the file header for the definition).

    Parameters
    ----------
    u : (T, N, D) centred displacements (utils.compute_displacements(); the array given to local_pca()).
    masses : (N,) or None -- mass weighting exactly as modes_from_displacements() / local_pca().
    windows : (n_windows, n) atom indices of every window (local_pca_analysis(...)["windows"]).
    chunk : windows diagonalised per batch (memory only).

    Returns
    -------
    softness_series : (T, N) array, row t = softness at frame t from the frames 0..t (NaN: atom in no window)
    n_windows : (N,) number of windows that contain each atom
    """
    u = np.asarray(u, dtype=float)
    T, N, D = u.shape
    windows = np.asarray(windows, dtype=int)
    if windows.ndim != 2:
        raise ValueError("windows must be a (n_windows, n_atoms_per_window) integer array")
    n_win, n = windows.shape
    K = n * D

    X = _flat(u, masses)                                            # (T, N*D)
    cols = window_columns(windows, D)                               # (n_win, K)
    sqrt_m = np.ones(N * D) if masses is None else np.repeat(np.sqrt(np.asarray(masses, dtype=float)), D)

    n_windows = np.bincount(windows.ravel(), minlength=N)
    sums = np.zeros((T, N))
    for s in range(0, n_win, chunk):
        c = cols[s:s + chunk]
        w = windows[s:s + chunk]
        Xw = np.moveaxis(X[:, c], 1, 0)                             # (c, T, K)
        sm = sqrt_m[c][:, :, None]                                  # (c, K, 1)
        for t in range(T):
            Xt = Xw[:, :t + 1, :]
            C = np.matmul(Xt.transpose(0, 2, 1), Xt) / (t + 1)      # (c, K, K)
            lam, V = np.linalg.eigh(0.5 * (C + C.transpose(0, 2, 1)))
            lam = np.maximum(lam, 0.0)
            Vr = V / sm                                             # real-space vectors (mass weighting undone)
            Vr = Vr / np.linalg.norm(Vr, axis=1, keepdims=True)
            p = (Vr.reshape(len(c), n, D, K) ** 2).sum(axis=2)      # (c, n, K), sums to 1 over atoms
            s_w = (p * lam[:, None, :]).sum(axis=2)                 # (c, n) softness of every window atom
            sums[t] += np.bincount(w.ravel(), weights=s_w.ravel(), minlength=N)

    with np.errstate(divide="ignore", invalid="ignore"):
        softness_series = np.where(n_windows > 0, sums / np.maximum(n_windows, 1), np.nan)
    return softness_series, n_windows
