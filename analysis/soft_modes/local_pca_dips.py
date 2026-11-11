# =============================================================
# analysis/soft_modes/local_pca_dips.py -- what the local modes at the dips of the participation curve are.
#
# local_pca.detect_participation_dips() finds the local mode indices n at which the median effective number of
# atoms n_part(n) of the windows has a dip (a mode that is more localised than its neighbours in the spectrum).
# This module looks at those modes window by window. It reuses local_pca(), the PCA conventions of the
# soft-mode analysis and the plotting functions that already exist; nothing here changes the PCA.
#
# For every detected mode n and for three representative windows (the windows whose n_part of mode n is at the
# LOW / MEDIAN / HIGH quantile of all windows, LOCAL_PCA_DIP_QUANTILES):
#   * the amplitude a_n(t) = X_w v^w_n (X_w: mass-weighted displacements of the window, v^w_n: unit eigenvector in
#     the mass-weighted space). X is centred, so a_n has zero mean and, with the 1/T covariance convention of the
#     project, standard deviation sqrt(lambda^w_n) EXACTLY; the normalised amplitude a_n / sqrt(lambda^w_n) has
#     zero mean and unit variance, and a single harmonic well would give P(a_n/sigma_n) = N(0, 1).
#   * the REAL-space eigenvector (mass weighting undone: v_i / sqrt(m_i), re-normalised, as modes.py and the
#     participation ratio do), its norm fraction on every atom p_i = sum_d v_{i,d}^2 (sum_i p_i = 1,
#     PR = 1 / (n sum_i p_i^2)), the atoms with the largest p_i, their element and their structural
#     environment: coordination number (neighbours within the species-pair cutoff of the time-averaged structure),
#     the share of the neighbours that is of each species, and the nearest-neighbour distance.
# Over ALL windows, for every detected mode: how often the dominant atom is of each element against the
# composition, the share of the mode's norm that sits on each element against the share of atoms, and the
# environment of the dominant atoms against that of all atoms.
#
# What this does and does not show. The windows overlap and use the same frames; a window is the principal
# submatrix of the global covariance, so a localised local mode need not be localised in the full system. The
# representative windows are chosen by n_part of the mode itself: they illustrate the spread, they are not a
# random sample. Environment descriptors are a first look (a coordination number and a distance from the
# time-averaged structure), not a structural classification; with a few hundred dominant atoms against N the
# comparison with all atoms is descriptive, not a significance test.
# =============================================================

import warnings

import numpy as np

from analysis.soft_modes.local_pca import local_pca, _periodic_tree
from analysis.soft_modes.pca_reliability import _flat

CLASS_NAMES = ("low", "median", "high")


# ------------------------------------------------------------
# Structural environment
# ------------------------------------------------------------

def _pair_cutoff(cutoff, a, b):
    if isinstance(cutoff, dict):
        return cutoff.get((a, b), cutoff.get((b, a)))
    return float(cutoff)


def atom_environment(mean_positions, box, symbols, cutoff):
    """
    Coordination of every atom in the time-averaged structure (periodic, orthorhombic box).

    cutoff : float (Å) or dict {(symbol_a, symbol_b): Å} (either order), e.g. the first minima of the partial g(r)
        (cutoffs.CUTOFF_DICT with the atomic numbers replaced by symbols).
    Returns dict: species (S,) sorted symbols, cn (N,), cn_species (N, S) neighbours of each species,
    other_fraction (N,) share of the neighbours of a different species than the atom's own (nan without
    neighbours), d_nn (N,) distance to the nearest atom.
    """
    pos = np.asarray(mean_positions, float)
    box = np.asarray(box, float)
    symbols = np.asarray(symbols)
    species = np.array(sorted(set(symbols.tolist())))
    sid = np.searchsorted(species, symbols)
    N = len(pos)
    if isinstance(cutoff, dict):
        rmax = max(cutoff.values())
    else:
        rmax = float(cutoff)
    tree, pos_w = _periodic_tree(pos, box)
    pairs = tree.query_pairs(rmax, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = pos[i] - pos[j]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    rc = np.array([_pair_cutoff(cutoff, species[a], species[b]) if _pair_cutoff(cutoff, species[a], species[b]) is not None
                   else 0.0 for a, b in zip(sid[i], sid[j])]) if len(i) else np.zeros(0)
    ok = r <= rc
    i, j = i[ok], j[ok]
    cn_s = np.zeros((N, len(species)), dtype=int)
    np.add.at(cn_s, (i, sid[j]), 1)
    np.add.at(cn_s, (j, sid[i]), 1)
    cn = cn_s.sum(axis=1)
    own = cn_s[np.arange(N), sid]
    with np.errstate(invalid="ignore", divide="ignore"):
        other = np.where(cn > 0, (cn - own) / np.maximum(cn, 1), np.nan)
    d_nn = tree.query(pos_w, k=2)[0][:, 1]
    return {"species": species, "cn": cn, "cn_species": cn_s, "other_fraction": other, "d_nn": d_nn}


# ------------------------------------------------------------
# Windows, eigenvectors, amplitudes
# ------------------------------------------------------------

def representative_windows(values, quantiles=(0.05, 0.5, 0.95)):
    """Index of the window whose value is closest to each quantile of `values` (ties: the first)."""
    values = np.asarray(values, float)
    return np.array([int(np.argmin(np.abs(values - np.quantile(values, q)))) for q in quantiles])


def select_modes(dips, max_modes):
    """
    Detected modes to investigate (at most max_modes, ascending). Minima of the dips come first (largest
    prominence first), then the other modes of the dips (adjacent to a minimum). Returns (modes, truncated).
    """
    if dips is None or len(dips["modes"]) == 0:
        return np.zeros(0, dtype=int), False
    order = np.argsort(-dips["prominence"], kind="stable")
    ranked = [int(dips["minimum"][i]) for i in order]
    ranked += [int(m) for i in order for m in range(dips["first"][i], dips["last"][i] + 1) if m != dips["minimum"][i]]
    chosen = ranked[:max_modes]
    return np.array(sorted(chosen), dtype=int), len(ranked) > max_modes


def real_space_vector(v_mw, masses_window, n_dim):
    """
    Undo the mass weighting of a window eigenvector (K,) = (n atoms * D): v_i / sqrt(m_i), re-normalised to unit
    length -- the vector modes.py returns and whose participation ratio local_pca() stores. Returns (n, D).
    masses_window (n,) or None.
    """
    v = np.asarray(v_mw, float).reshape(-1, n_dim)
    if masses_window is not None:
        v = v / np.sqrt(np.asarray(masses_window, float))[:, None]
    return v / np.linalg.norm(v)


def norm_fractions(v_real):
    """p_i = sum_d v_{i,d}^2 of a real-space unit vector (n, D): sums to 1; n_part = 1 / sum_i p_i^2."""
    return (np.asarray(v_real) ** 2).sum(axis=1)


def normalized_amplitude(X_window, v_mw):
    """
    a_n(t) = X_w v and the same in units of its standard deviation. X_window (T, K) mass-weighted, centred
    displacements of the window, v (K,) unit eigenvector in the mass-weighted space. For an eigenvector of
    C_w = X_w^T X_w / T the standard deviation of a_n is sqrt(lambda^w_n) exactly.
    Returns (a, a_std, sigma).
    """
    a = np.asarray(X_window, float) @ np.asarray(v_mw, float)
    sigma = float(a.std())
    return a, (a - a.mean()) / (sigma if sigma > 0 else 1.0), sigma


def _projection_axes(rel):
    """Two orthonormal axes (3, 2) of the plane of largest extent of the relative positions (n, 3)."""
    c = rel - rel.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[:2].T


# ------------------------------------------------------------
# Investigation
# ------------------------------------------------------------

def investigate_dip_modes(res, u, masses, symbols, mean_positions, box, cutoff, max_modes=6,
                          quantiles=(0.05, 0.5, 0.95), top_atoms=5, seed=0):
    """
    Look at the modes at the detected dips of res["dips"] (local_pca.local_pca_analysis(..., dip_prominence=...)).

    Parameters
    ----------
    res : result of local_pca_analysis(); u, masses : the arrays it was computed from; symbols (N,) element of each
    atom; mean_positions (N, D), box (D,) the time-averaged positions and orthorhombic cell edges; cutoff : see
    atom_environment(). max_modes, quantiles, top_atoms : see config.LOCAL_PCA_DIP_*.

    Returns dict:
      modes (M,)                 investigated detected modes (ascending); truncated (bool) if more were detected
      classes                    ("low", "median", "high") + quantiles
      window (M, 3)              window index of each representative window; centre (M, 3) its centre atom
      n_part, eigenvalue, sigma (M, 3)   of the mode in those windows; excess_kurtosis, skewness (M, 3) of the
                                 normalised amplitude (0 for a Gaussian)
      amplitude_diag             dict in the format of tls.mode_amplitude_diagnostics (M * 3 panels, row-major: mode,
                                 class) for visualization.plot.plot_mode_amplitude_grid
      panels                     list of dicts (one per mode and class) for visualization.plot.plot_dip_eigenvectors
      table                      list of dict rows (compact CSV: write_dip_csv)
      windows_all                per mode: dict of statistics over all windows (see below)
      environment                atom_environment() of the structure
      all windows (windows_all[k]): n_part_median, dominant_share_median (largest p_i), dominant_element_fraction
        (S,) share of the windows whose dominant atom is of each species, composition (S,), norm_share_element (S,)
        median share of the mode's norm on atoms of each species, atom_share_element (S,) median share of the
        window's atoms of that species, dominant_cn_median / all_cn_median, dominant_d_nn_median / all_d_nn_median,
        dominant_cn_rank (mean percentile of the dominant atoms' coordination among all atoms),
        excess_kurtosis_median, excess_kurtosis_q (5, 95 %) of the normalised amplitude over all windows.
    """
    from scipy.stats import kurtosis, skew
    modes, truncated = select_modes(res.get("dips"), max_modes)
    windows = res["windows"]
    n_win, n = windows.shape
    D = res["local"]["n_dim"]
    T = res["local"]["n_frames"]
    symbols = np.asarray(symbols)
    env = atom_environment(mean_positions, box, symbols, cutoff)
    species = env["species"]
    sid_all = np.searchsorted(species, symbols)
    X = _flat(np.asarray(u, float), masses)
    sqrt_m = None if masses is None else np.sqrt(np.asarray(masses, float))
    centers = res["windows_info"].get("centers")
    pos = np.asarray(mean_positions, float)
    box = np.asarray(box, float)
    out = {"modes": modes, "truncated": truncated, "classes": CLASS_NAMES, "quantiles": np.asarray(quantiles, float),
           "environment": env, "windows_all": {}, "panels": [], "table": []}
    M = len(modes)
    if M == 0:
        return out

    # eigenvectors: stored ones, or recomputed for all windows (same function, same numbers)
    V_all = res["local"].get("eigenvectors_mw")
    if V_all is None:
        V_all = local_pca(u, masses, windows=windows, store_eigenvectors=True)["eigenvectors_mw"]
    cols = res["local"]["columns"]

    # ---- statistics over all windows, for every detected mode
    Xw = np.moveaxis(X[:, cols], 1, 0)                                       # (n_win, T, K)
    for k, m in enumerate(modes):
        Vm = np.asarray(V_all[:, :, m], float)                               # (n_win, K)
        a = np.einsum("wtk,wk->wt", Xw, Vm)
        a_std = (a - a.mean(axis=1, keepdims=True)) / np.maximum(a.std(axis=1, keepdims=True), 1e-300)
        Vr = Vm.reshape(n_win, n, D)
        if sqrt_m is not None:
            Vr = Vr / sqrt_m[windows][:, :, None]
        Vr = Vr / np.linalg.norm(Vr.reshape(n_win, -1), axis=1)[:, None, None]
        p = (Vr ** 2).sum(axis=2)                                            # (n_win, n)
        top = np.argmax(p, axis=1)
        dom = windows[np.arange(n_win), top]                                 # dominant atom of every window
        sid_w = sid_all[windows]                                             # (n_win, n)
        norm_share = np.stack([(p * (sid_w == s)).sum(axis=1) for s in range(len(species))], axis=1)
        atom_share = np.stack([(sid_w == s).mean(axis=1) for s in range(len(species))], axis=1)
        kur = kurtosis(a_std, axis=1)
        all_cn = env["cn"]
        rank = np.searchsorted(np.sort(all_cn), env["cn"][dom], side="right") / len(all_cn)
        out["windows_all"][int(m)] = {
            "n_part_median": float(np.median(1.0 / (p ** 2).sum(axis=1))),
            "dominant_share_median": float(np.median(p.max(axis=1))),
            "dominant_element_fraction": np.array([(sid_all[dom] == s).mean() for s in range(len(species))]),
            "composition": np.array([(sid_all == s).mean() for s in range(len(species))]),
            "norm_share_element": np.median(norm_share, axis=0),
            "atom_share_element": np.median(atom_share, axis=0),
            "dominant_cn_median": float(np.median(env["cn"][dom])), "all_cn_median": float(np.median(all_cn)),
            "dominant_d_nn_median": float(np.median(env["d_nn"][dom])), "all_d_nn_median": float(np.median(env["d_nn"])),
            "dominant_cn_rank": float(rank.mean()),
            "excess_kurtosis_median": float(np.median(kur)), "excess_kurtosis_q": np.percentile(kur, [5, 95])}

    del Xw
    # ---- representative windows
    window = np.zeros((M, 3), dtype=int)
    centre = np.zeros((M, 3), dtype=int)
    n_part = np.zeros((M, 3)); lam = np.zeros((M, 3)); sigma = np.zeros((M, 3))
    kur = np.zeros((M, 3)); skw = np.zeros((M, 3))
    A = np.zeros((T, M * 3))
    lam_flat = np.zeros(M * 3); pr_flat = np.zeros(M * 3)
    labels, units = [], []
    for k, m in enumerate(modes):
        window[k] = representative_windows(res["local"]["n_participating"][:, m], quantiles)
        for c, w in enumerate(window[k]):
            atoms = windows[w]
            centre[k, c] = atoms[0] if centers is None else int(centers[w])
            v_mw = np.asarray(V_all[w][:, m], float)
            a, a_std, s = normalized_amplitude(X[:, cols[w]], v_mw)
            v_real = real_space_vector(v_mw, None if masses is None else np.asarray(masses, float)[atoms], D)
            p = norm_fractions(v_real)
            n_part[k, c] = 1.0 / (p ** 2).sum()
            lam[k, c] = res["local"]["eigenvalues"][w, m]
            sigma[k, c] = s
            kur[k, c], skw[k, c] = float(kurtosis(a_std)), float(skew(a_std))
            A[:, k * 3 + c] = a
            lam_flat[k * 3 + c] = lam[k, c]
            pr_flat[k * 3 + c] = n_part[k, c] / n
            labels.append(f"mode {m}, {CLASS_NAMES[c]} n_part: window {w} (centre atom {centre[k, c]}), n_part = {n_part[k, c]:.1f}")
            units.append("√amu·Å" if masses is not None else "Å")
            # ---- atoms with the largest norm fraction
            order = np.argsort(-p, kind="stable")
            cum = np.cumsum(p[order])
            ref = pos[centre[k, c]]
            rel = pos[atoms] - ref
            rel -= box * np.round(rel / box)
            for r in range(min(top_atoms, n)):
                i = order[r]
                g = int(atoms[i])
                out["table"].append({
                    "mode": int(m), "dip": int(res["dips"]["mode_dip"][np.where(res["dips"]["modes"] == m)[0][0]]),
                    "is_dip_minimum": bool(m in res["dips"]["minimum"]),
                    "window_class": CLASS_NAMES[c], "window": int(w), "window_centre_atom": int(centre[k, c]),
                    "n_part": float(n_part[k, c]), "lambda": float(lam[k, c]), "rank": r + 1, "atom": g,
                    "element": str(symbols[g]), "norm_fraction": float(p[i]), "cumulative_fraction": float(cum[r]),
                    "vx": float(v_real[i, 0]), "vy": float(v_real[i, 1]), "vz": float(v_real[i, 2]) if D > 2 else float("nan"),
                    "distance_to_centre": float(np.linalg.norm(rel[i])),
                    "cn": int(env["cn"][g]),
                    **{f"cn_{sp}": int(env["cn_species"][g, s]) for s, sp in enumerate(species)},
                    "other_species_fraction": float(env["other_fraction"][g]), "d_nn": float(env["d_nn"][g])})
            ax2 = _projection_axes(rel) if D == 3 else np.eye(D)[:, :2]
            out["panels"].append({
                "mode": int(m), "class": CLASS_NAMES[c], "window": int(w), "centre": int(centre[k, c]),
                "n_part": float(n_part[k, c]), "xy": rel @ ax2, "vxy": v_real @ ax2, "p": p,
                "symbols": symbols[atoms], "top": order[:top_atoms], "atoms": atoms})

    from analysis.soft_modes.tls import mode_amplitude_diagnostics
    diag = mode_amplitude_diagnostics({"amplitudes": A, "eigenvalues": lam_flat, "participation": pr_flat},
                                      mode_idx=np.arange(M * 3), seed=seed)
    diag["labels"] = labels
    diag["units"] = units
    diag["title"] = ("P(a_n) of the dip modes in representative windows (rows: detected mode; columns: window with low / "
                     "median / high n_part of that mode)")
    out.update({"window": window, "centre": centre, "n_part": n_part, "eigenvalue": lam, "sigma": sigma,
                "excess_kurtosis": kur, "skewness": skw, "amplitude_diag": diag})
    return out


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

def write_dip_csv(inv, path):
    """Compact CSV of the dominant atoms (one row per mode, representative window and atom rank)."""
    import csv
    rows = inv["table"]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Saved: {path}")


def dip_investigation_arrays(inv):
    """The numerical part of investigate_dip_modes() as a nested dict of arrays for local_pca.save_local_pca."""
    if len(inv["modes"]) == 0:
        return {"modes": inv["modes"]}
    out = {k: inv[k] for k in ("modes", "quantiles", "window", "centre", "n_part", "eigenvalue", "sigma",
                               "excess_kurtosis", "skewness")}
    out["species"] = inv["environment"]["species"]
    out["windows_all"] = {str(m): {k: np.asarray(v) for k, v in d.items()} for m, d in inv["windows_all"].items()}
    return out


def print_dip_investigation(inv):
    """Short summary of investigate_dip_modes()."""
    if len(inv["modes"]) == 0:
        print("\n  no dips: nothing to investigate")
        return
    sp = inv["environment"]["species"]
    print(f"\n  dip modes {inv['modes'].tolist()}" + ("  (more were detected: LOCAL_PCA_DIP_MAX_MODES)" if inv["truncated"] else ""))
    for k, m in enumerate(inv["modes"]):
        s = inv["windows_all"][int(m)]
        print(f"  mode {m}: n_part low/median/high windows {', '.join(f'{x:.1f}' for x in inv['n_part'][k])}; "
              f"excess kurtosis of P(a/sigma) {', '.join(f'{x:+.2f}' for x in inv['excess_kurtosis'][k])} "
              f"(all windows: median {s['excess_kurtosis_median']:+.2f}, 5-95% {s['excess_kurtosis_q'][0]:+.2f} .. {s['excess_kurtosis_q'][1]:+.2f})")
        print("      dominant atom: " + ", ".join(f"{e} {100 * f:.0f}% (composition {100 * c:.0f}%)"
              for e, f, c in zip(sp, s["dominant_element_fraction"], s["composition"]))
              + f"; share of the norm per element " + ", ".join(f"{e} {100 * f:.0f}% (atoms {100 * a:.0f}%)"
              for e, f, a in zip(sp, s["norm_share_element"], s["atom_share_element"])))
        print(f"      dominant atoms: coordination median {s['dominant_cn_median']:.0f} (all atoms {s['all_cn_median']:.0f}, mean percentile "
              f"{100 * s['dominant_cn_rank']:.0f}%), nearest-neighbour distance {s['dominant_d_nn_median']:.2f} Å (all {s['all_d_nn_median']:.2f} Å), "
              f"largest norm fraction in a window {s['dominant_share_median']:.2f}")
