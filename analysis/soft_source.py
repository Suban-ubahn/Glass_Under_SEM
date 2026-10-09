# =============================================================
# analysis/soft_source.py
#
# Are soft atoms "sources" of plastic activity? I.e. is high D^2_min
# found around soft atoms (WHERE), and does it appear there first (WHEN)?
#
# Both questions are asked of the same D^2_min field, with the same
# thermal correction and the same null model:
#
#   Thermal baseline.  Soft atoms vibrate more, which by itself inflates
#   their own and their neighbours' D^2_min. With delta_ij(s) = u_j(s) -
#   u_i(s) (relative displacement of bond (i,j) about the mean structure)
#   and C_ij(t) = <delta_ij(s) . delta_ij(s+t)>_s, stationary vibration
#   gives E|delta_ij(s+t) - delta_ij(s)|^2 = 2 C_ij(0) [1 - rho(t)], with
#   rho(t) the normalised autocorrelation. rho(t) is averaged over ALL
#   bonds of the same element pair (thousands of bonds), NOT taken from
#   the bond itself: at large t a single bond's own C_ij(t) is one
#   realisation and would absorb the very rearrangements we want to
#   detect. The expected thermal D^2_min of atom i, n_i neighbours, is
#       baseline_i(t) = (n_i - 3) * mean_j 2 C_ij(0) [1 - rho_pair(j)(t)].
#   excess_i(t) = D^2_min,i(0 -> t) - baseline_i(t).
#   Every test is run on the raw field AND on the excess.
#
#   Null model.  Soft labels are permuted among atoms with the same
#   (element, coordination number), so element / coordination effects
#   cancel out.
#
#   WHERE (spatial): non-soft atoms with >= 1 soft neighbour vs. none,
#       compared in the final frame.
#   WHEN (temporal): the onset of an atom is the first frame from which
#       its field stays >= a threshold for n_hold frames. For every bond
#       (soft atom, non-soft neighbour) with both onsets,
#       lag = onset_neighbour - onset_soft  (> 0: the soft atom was first).
# =============================================================

import numpy as np
from ase.neighborlist import neighbor_list

from utils import unwrap_positions, compute_displacements


def thermal_baseline_series(trajectory, i_idx, j_idx, n_neighbors, elements):
    """(T, N) expected thermal D^2_min(0 -> t) of every atom (see header)."""
    T, N = len(trajectory), len(n_neighbors)
    positions = np.array([a.positions for a in trajectory])
    box = np.array([a.cell.array.diagonal() for a in trajectory])
    u, _ = compute_displacements(unwrap_positions(positions, box))
    delta = u[:, j_idx] - u[:, i_idx]                                   # (T, B, 3)

    C = np.empty((T, len(i_idx)))                                       # C_ij(tau)
    for tau in range(T):
        C[tau] = (delta[: T - tau] * delta[tau:]).sum(-1).mean(0)
    # rho(t) per element pair: ratio of bond-averaged C(t) to bond-averaged C(0)
    _, codes = np.unique(elements, return_inverse=True)
    lo, hi = np.minimum(codes[i_idx], codes[j_idx]), np.maximum(codes[i_idx], codes[j_idx])
    pair_type = lo * (codes.max() + 1) + hi
    rho = np.empty_like(C)
    for pt in np.unique(pair_type):
        m = pair_type == pt
        rho[:, m] = (C[:, m].mean(1) / C[0, m].mean())[:, None]
    bond_change = 2.0 * C[0][None, :] * (1.0 - rho)                     # (T, B)
    per_atom = np.stack([np.bincount(i_idx, weights=bond_change[t], minlength=N)
                         for t in range(T)]) / n_neighbors[None, :]
    return (n_neighbors - 3)[None, :] * per_atom


def onset_frames(series, thr_quantile=0.9, n_hold=3):
    """First frame from which each atom stays >= threshold for n_hold
    frames (NaN if never). Threshold = thr_quantile of the final frame."""
    T = series.shape[0]
    thr = np.nanquantile(series[-1], thr_quantile)
    above = series >= thr
    held = np.ones_like(above)
    for k in range(n_hold):
        held[: T - k] &= above[k:]
        held[T - k:] = False
    return np.where(held.any(0), held.argmax(0), np.nan), thr


def _spatial_stat(series_final, mask, i_idx, j_idx, n_atoms):
    """mean field of non-soft atoms with a soft neighbour minus those without."""
    n_soft_nb = np.bincount(i_idx, weights=mask[j_idx], minlength=n_atoms)
    near = ~mask & (n_soft_nb > 0)
    far = ~mask & (n_soft_nb == 0)
    return np.nanmean(series_final[near]) - np.nanmean(series_final[far])


def _temporal_stat(onset, mask, i_idx, j_idx):
    """(mean lag, fraction soft-first with ties dropped) over (soft, non-soft) bonds."""
    sel = mask[i_idx] & ~mask[j_idx] & np.isfinite(onset[i_idx]) & np.isfinite(onset[j_idx])
    lag = onset[j_idx[sel]] - onset[i_idx[sel]]
    return lag.mean(), (lag > 0).sum() / max((lag != 0).sum(), 1), int(sel.sum())


def soft_source_analysis(trajectory, softness, d2min_series, cutoff,
                         soft_frac=0.1, thr_quantile=0.9, n_hold=3,
                         n_perm=1000, seed=0):
    """
    Parameters
    ----------
    softness : (N,) per-atom softness score (e.g. soft_mode_analysis_3d()["softness"])
    d2min_series : (T, N) D^2_min(0 -> t), e.g. from dynamics.flow_units.d2min_trajectory()

    Returns
    -------
    dict with
      soft_mask, n_soft_neighbours, baseline (T, N), excess (T, N)
      fields : {"raw": ..., "excess": ...}, each a dict with
          spatial  : observed, null_mean, null_std, z, p, n_near,
                     by_n_soft_neighbours {k: (n_atoms, mean field)}
          temporal : threshold, n_onsets, n_bonds, mean_lag, mean_lag_null_*,
                     mean_lag_z, frac_soft_first, frac_null_*, frac_z
    """
    rng = np.random.default_rng(seed)
    d2min_series = np.asarray(d2min_series, dtype=float)
    atoms0 = trajectory[0]
    N = len(atoms0)
    elements = np.array(atoms0.get_chemical_symbols())
    i_idx, j_idx = neighbor_list("ij", atoms0, cutoff)         # each bond in both directions
    n_nb = np.bincount(i_idx, minlength=N)

    soft = softness >= np.quantile(softness, 1 - soft_frac)
    n_soft_nb = np.bincount(i_idx, weights=soft[j_idx], minlength=N).astype(int)

    baseline = thermal_baseline_series(trajectory, i_idx, j_idx, n_nb, elements)
    excess = d2min_series - baseline

    # permuted soft labels, shared by every test below
    groups = {}
    for a in range(N):
        groups.setdefault((elements[a], n_nb[a]), []).append(a)
    groups = [np.array(g) for g in groups.values()]
    perm_masks = []
    for _ in range(n_perm):
        m = np.zeros(N, dtype=bool)
        for g in groups:
            m[g] = rng.permutation(soft[g])
        perm_masks.append(m)

    def z_and_null(obs, null):
        null = np.asarray(null)
        return null.mean(), null.std(), (obs - null.mean()) / null.std()

    fields = {}
    for name, series in (("raw", d2min_series), ("excess", excess)):
        final = series[-1]

        obs = _spatial_stat(final, soft, i_idx, j_idx, N)
        null = [_spatial_stat(final, m, i_idx, j_idx, N) for m in perm_masks]
        nm, ns, z = z_and_null(obs, null)
        by_k = {}
        for k in range(5):
            sel = ~soft & ((n_soft_nb == k) if k < 4 else (n_soft_nb >= k))
            if sel.any():
                by_k[k] = (int(sel.sum()), float(np.nanmean(final[sel])))
        spatial = dict(observed=obs, null_mean=nm, null_std=ns, z=z,
                       p=(np.sum(np.array(null) >= obs) + 1) / (n_perm + 1),
                       n_near=int((~soft & (n_soft_nb > 0)).sum()),
                       by_n_soft_neighbours=by_k)

        onset, thr = onset_frames(series, thr_quantile, n_hold)
        lag, frac, n_bonds = _temporal_stat(onset, soft, i_idx, j_idx)
        null = np.array([_temporal_stat(onset, m, i_idx, j_idx)[:2] for m in perm_masks])
        lm, ls, lz = z_and_null(lag, null[:, 0])
        fm, fs, fz = z_and_null(frac, null[:, 1])
        temporal = dict(threshold=thr, n_onsets=int(np.isfinite(onset).sum()),
                        n_bonds=n_bonds, mean_lag=lag, mean_lag_null_mean=lm,
                        mean_lag_null_std=ls, mean_lag_z=lz,
                        frac_soft_first=frac, frac_null_mean=fm, frac_null_std=fs,
                        frac_z=fz)
        fields[name] = dict(spatial=spatial, temporal=temporal)

    return dict(soft_mask=soft, n_soft_neighbours=n_soft_nb,
                baseline=baseline, excess=excess, fields=fields)