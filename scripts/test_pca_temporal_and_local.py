# =============================================================
# scripts/test_pca_temporal_and_local.py
#
# Checks of analysis/soft_modes/pca_convergence.py (ACF, T_eff, convergence) and
# analysis/soft_modes/local_pca.py (moving-window PCA) on synthetic data with known answers.
# No trajectory file is needed.   Run:  python scripts/test_pca_temporal_and_local.py
# (or:  pytest scripts/test_pca_temporal_and_local.py)
# =============================================================

import numpy as np

from analysis.soft_modes.modes import (
    modes_from_displacements, modes_from_covariance, participation_ratios,
)
from analysis.soft_modes.pca_convergence import (
    autocorrelation, integrated_autocorr_time, amplitude_acf_analysis, pca_convergence,
    pca_temporal_analysis,
)
from analysis.soft_modes.local_pca import (
    index_windows, local_pca, global_reference, compare_local_to_global, overlap_consistency,
    track_modes, local_pca_analysis, marchenko_pastur_edges,
    neighborhood_windows, window_partners, _partner_table,
)
from analysis.soft_modes.pca_reliability import _flat
from analysis.soft_modes.local_pca import (
    detect_participation_dips, summarize_median, null_band_diagnostics, local_pca_band_statistics,
    hilbert_index, spatial_window_order, order_locality, global_reference,
)
from analysis.soft_modes.local_pca_dips import (
    atom_environment, investigate_dip_modes, normalized_amplitude, norm_fractions, real_space_vector,
    representative_windows, select_modes, dip_investigation_arrays, write_dip_csv,
)


def ar1(phi, T, rng):
    y, e = np.zeros(T), rng.standard_normal(T)
    for t in range(1, T):
        y[t] = phi * y[t - 1] + e[t]
    return y


def synthetic_displacements(T=101, N=120, D=3, seed=0, masses=True):
    """u (T, N, D) centred: 3 collective modes spread over ALL atoms (AR(1) amplitudes) + a mode
    localised on atoms 40..49 + independent noise; two species with different masses."""
    rng = np.random.default_rng(seed)
    P = N * D
    X = 0.4 * rng.standard_normal((T, P))
    for s in (6.0, 4.0, 3.0):
        v = rng.standard_normal(P)
        X += np.outer(ar1(0.7, T, rng) * s / np.sqrt(P) * 6, v / np.linalg.norm(v))
    loc = np.zeros(P)
    loc[40 * D:50 * D] = rng.standard_normal(10 * D)
    X += np.outer(ar1(0.6, T, rng) * 5.0, loc / np.linalg.norm(loc))
    u = (X - X.mean(axis=0)).reshape(T, N, D)
    m = np.where(np.arange(N) % 2 == 0, 91.224, 63.546) if masses else None
    return u, m


# ---------------- ACF / T_eff ----------------

def test_acf_ar1_matches_theory():
    rng = np.random.default_rng(1)
    for phi in (0.0, 0.5, 0.8):
        y = ar1(phi, 6000, rng)
        rho = autocorrelation(y, 200)
        assert abs(rho[1] - phi) < 0.05
        for method in ("sokal", "geyer", "first_negative"):
            tau = integrated_autocorr_time(rho, method)["tau"]
            assert abs(tau - (1 + phi) / (1 - phi)) < 0.2 * (1 + phi) / (1 - phi) + 0.3, (phi, method, tau)


def test_squared_series_acf_is_gaussian_prediction():
    rng = np.random.default_rng(2)
    y = ar1(0.8, 20000, rng)
    r_a = autocorrelation(y, 5)
    r_sq = autocorrelation(y ** 2, 5)
    assert np.allclose(r_sq[1:4], r_a[1:4] ** 2, atol=0.06)


def test_oscillatory_and_short_series_are_flagged():
    rng = np.random.default_rng(3)
    t = np.arange(101)
    y = np.cos(2 * np.pi * t / 8) + 0.3 * rng.standard_normal(101)
    rep = amplitude_acf_analysis(np.c_[y, y], n_modes=2)
    assert not rep["amplitude"]["reliable"][0]
    flags = rep["amplitude"]["flags"][0]
    assert "oscillatory" in flags or "tau_below_1" in flags
    assert rep["amplitude"]["t_eff"][0] <= 101          # capped at T
    white = amplitude_acf_analysis(rng.standard_normal((101, 3)), n_modes=3)
    assert np.all(white["squared"]["t_eff"] <= 101)
    assert np.all(white["amplitude"]["ar1_t_eff"] <= 101)


def test_leading_amplitude_acf_is_a_selection_artifact_and_the_null_reproduces_it():
    """Pure AR(1) noise (per-coordinate phi = 0.3, no collective mode), P >> T: the leading PCA amplitudes are
    smooth temporal patterns (lag-one correlation ~0.95, T_eff of a few frames) although every coordinate has
    T_eff ~ 54; the pooled T_eff recovers the truth and the shifted-column null reproduces the artifact."""
    rng = np.random.default_rng(12)
    T, N, phi = 101, 400, 0.3
    X, e = np.zeros((T, N * 3)), rng.standard_normal((T, N * 3))
    for t in range(T):
        X[t] = (phi * X[t - 1] if t else 0.0) + e[t]
    u = (X - X.mean(axis=0)).reshape(T, N, 3)
    res = pca_temporal_analysis(u, None, n_modes=3, n_null=6, seed=1)
    assert np.all(res["amplitude"]["ar1_phi"] > 0.85)                       # artifact: >> the true 0.3
    assert np.all(res["amplitude"]["ar1_t_eff"] < 10)                       # true T_eff per coordinate: 54
    assert abs(res["pooled_t_eff"] - T * (1 - phi) / (1 + phi)) < 0.25 * T * (1 - phi) / (1 + phi)
    assert res["null"]["within"]["amplitude_ar1_phi"].all()                 # null cannot be told from observed
    assert np.all(res["null"]["amplitude"]["ar1_phi_median"] > 0.85)
    assert res["null"]["amplitude"]["acf_low"].shape == res["amplitude"]["acf"].shape


def test_ar1_flags_fewer_than_one_independent_frame():
    from analysis.soft_modes.pca_convergence import _ar1
    rho = 0.985 ** np.arange(20)
    out = _ar1(rho, 101, 0.2, 10)
    assert out["t_eff"] < 1 and "t_eff_below_1" in out["flags"]
    ok = _ar1(0.5 ** np.arange(20), 101, 0.2, 10)
    assert "t_eff_below_1" not in ok["flags"] and abs(ok["t_eff"] - 101 / 3) < 1e-9


# ---------------- convergence ----------------

def test_convergence_matches_modes_by_overlap_not_index():
    u, m = synthetic_displacements(N=100, seed=5)
    res = pca_convergence(u, m, lengths=(20, 40, 60, 80, 101), n_modes=6, subspace_dim=3, n_null=3)
    assert res["subspace_overlap"][-1] > 0.999 and res["matched_overlap"][-1].min() > 0.999
    assert np.all(res["matched_overlap"] >= res["index_overlap"] - 1e-12)
    assert np.all(res["subspace_overlap"] >= 0) and np.all(res["subspace_overlap"] <= 1 + 1e-9)
    assert res["subspace_overlap"][0] > res["null_subspace_overlap"][:, 0].mean()


# ---------------- local PCA ----------------

def test_windows():
    w = index_windows(100, 15, 1)
    assert w.shape == (86, 15) and w[0, 0] == 0 and w[-1, -1] == 99 and set(w.ravel()) == set(range(100))
    w = index_windows(100, 15, 10)                      # step 10 -> extra tail window keeps coverage
    assert set(w.ravel()) == set(range(100)) and w[-1, 0] == 85


def test_local_equals_existing_conventions():
    """Window eigenvalues / eigenvectors / PR equal modes.py applied to the same atoms."""
    u, m = synthetic_displacements(N=60, seed=7)
    loc = local_pca(u, m, window_size=12, step=5)
    for w in (0, 3, 9):
        idx = loc["windows"][w]
        ref = modes_from_displacements(u[:, idx], masses=m[idx])
        assert np.allclose(loc["eigenvalues"][w], ref["eigenvalues"][:36], rtol=1e-9, atol=1e-12)
        assert np.allclose(loc["participation"][w], participation_ratios(ref["eigenvectors"])[:36], atol=1e-9)
        # eigenvector comparison up to sign (mass-weighted basis <-> real-space basis)
        sm = np.repeat(np.sqrt(m[idx]), 3)
        Vr = ref["eigenvectors"].reshape(36, -1)
        mine = loc["eigenvectors_mw"][w][:, :36] / sm[:, None]
        mine /= np.linalg.norm(mine, axis=0)
        assert np.allclose(np.abs(np.sum(mine * Vr, axis=0)), 1.0, atol=1e-7)
        # and the covariance route of modes.py gives the same eigenvalues
        C = (u[:, idx].reshape(len(u), -1).T @ u[:, idx].reshape(len(u), -1)) / len(u)
        cov = modes_from_covariance(C, 12, 3, masses=m[idx])
        assert np.allclose(cov["eigenvalues"], loc["eigenvalues"][w], rtol=1e-9, atol=1e-12)


def test_exact_relations_between_local_and_global():
    u, m = synthetic_displacements(N=60, seed=8)
    T = len(u)
    loc = local_pca(u, m, window_size=10, step=3)
    glob = global_reference(u, m, n_modes=T - 1)               # ALL non-zero modes
    comp = compare_local_to_global(loc, glob, subspace_dims=(1, 3, 5), n_compare_modes=10)
    assert comp["interlacing_violation"] < 1e-9                  # (1) Cauchy interlacing
    X = _flat(u, m)                                              # (2) trace = coordinate variances
    for w in (0, 4):
        assert np.isclose(loc["total_variance"][w], (X[:, loc["columns"][w]] ** 2).sum() / T)
    assert np.allclose(comp["explained_fraction"], 1.0, atol=1e-8)   # (3) all T-1 modes: exact identity
    # sum_n O_nm^2 = 1 when all K local modes are used (K = 30 < T-1)
    G = glob["eigenvectors"][loc["columns"][2]]
    W = loc["eigenvectors_mw"][2].T @ G
    assert np.allclose((W ** 2).sum(axis=0), (G ** 2).sum(axis=0))
    assert np.all((comp["overlap_best"] >= 0) & (comp["overlap_best"] <= 1 + 1e-9))
    assert np.all((comp["subspace_overlap"] >= 0) & (comp["subspace_overlap"] <= 1 + 1e-9))
    assert np.all((comp["capture_ratio"] >= 0) & (comp["capture_ratio"] <= 1 + 1e-9))
    # one-dimensional eigenspace overlap S_1 = cos^2 = (single-vector overlap with global mode 0)^2
    assert np.allclose(comp["subspace_overlap"][:, 0], comp["overlap_matrix"][:, 0, 0] ** 2, atol=1e-9)


def test_planted_global_mode_is_found_in_every_window():
    """One dominant collective mode over all atoms: every window's softest mode is its restriction."""
    rng = np.random.default_rng(9)
    T, N, D = 101, 90, 3
    v = rng.standard_normal(N * D)
    X = 0.3 * rng.standard_normal((T, N * D)) + np.outer(ar1(0.5, T, rng) * 40, v / np.linalg.norm(v))
    u = (X - X.mean(axis=0)).reshape(T, N, D)
    res = local_pca_analysis(u, None, window_size=15, step=1, n_global_modes=5, subspace_dims=(1, 2),
                             n_compare_modes=3, consistency_offsets=(1, 5), n_null=2)
    assert res["comparison"]["overlap_best"][:, 0].min() > 0.9
    assert res["comparison"]["capture_ratio"][:, 0].min() > 0.9
    assert res["consistency"]["matched_overlap"][0, :-1, 0].min() > 0.9
    assert res["tracking"]["chain_length"][:, 0].mean() > 30        # mode 1 survives along the chain
    null = res["null"]
    assert res["summary"]["overlap_best"][0] > np.quantile(null["overlap_best"][:, 0], 0.95)


def test_localised_mode_and_overlapping_windows():
    u, m = synthetic_displacements(N=120, seed=10)               # localised mode on atoms 40..49
    loc = local_pca(u, m, window_size=15, step=1)
    inside = loc["windows"][:, 0] >= 40 - 5                      # windows containing most of 40..49
    inside &= loc["windows"][:, 0] <= 40
    outside = loc["windows"][:, 0] >= 70
    assert loc["eigenvalues"][inside, 0].mean() > 2 * loc["eigenvalues"][outside, 0].mean()
    cons = overlap_consistency(loc, offsets=(1, 3), n_modes=3, subspace_dims=(1, 2))
    assert np.nanmean(cons["matched_overlap"][0, inside, 0]) > 0.9
    # chains: tracking must not fail and lengths are >= 1
    trk = track_modes(loc, n_modes=3, threshold=0.8)
    assert trk["chain_length"].min() >= 1


# ---------------- neighbourhood windows ----------------
def _min_image_dist(a, b, box):
    d = a - b
    d -= box * np.round(d / box)
    return np.sqrt((d ** 2).sum(axis=-1))


def test_neighborhood_windows_match_brute_force_with_periodic_boundaries():
    rng = np.random.default_rng(3)
    box = np.array([10.0, 11.0, 12.0])
    pos = rng.uniform(0, 1, (200, 3)) * box
    pos[0], pos[1] = [0.1, 5.0, 5.0], [9.9, 5.0, 5.0]                 # neighbours only through the boundary
    pos_unwrapped = pos + box * rng.integers(-2, 3, pos.shape)         # the reference positions need not be wrapped
    w, c = neighborhood_windows(pos_unwrapped, box, window_size=15, step=1)
    assert w.shape == (200, 15) and np.array_equal(c, np.arange(200)) and np.array_equal(w[:, 0], c)
    assert all(len(set(r)) == 15 for r in w)
    for i in (0, 1, 17, 150):                                          # brute force: 15 smallest minimum-image distances
        d = _min_image_dist(pos[i], pos, box)
        assert np.array_equal(w[i], np.argsort(d)[:15]) and np.all(np.diff(d[w[i]]) >= 0)
    assert 1 in w[0] and 0 in w[1]                                     # across the periodic boundary
    w5, c5 = neighborhood_windows(pos, box, window_size=15, step=5)
    assert np.array_equal(c5, np.arange(0, 200, 5)) and np.array_equal(w5, w[::5])
    assert neighborhood_windows(pos, box, window_size=1)[0].shape == (200, 1)


def test_window_partners_are_nearest_other_window_centres():
    rng = np.random.default_rng(4)
    box = np.array([9.0, 9.0, 9.0])
    cp = rng.uniform(0, 9, (120, 3))
    P = window_partners(cp, box, max_rank=6)
    assert P.shape == (120, 6)
    for w in (0, 30, 119):
        d = _min_image_dist(cp[w], cp, box)
        d[w] = np.inf
        assert np.array_equal(P[w], np.argsort(d)[:6])
    assert not np.any(P == np.arange(120)[:, None])
    # index windows: the partner at offset j is window w + j (-1 where it does not exist)
    offs, T = _partner_table(10, (1, 3, 12), None)
    assert list(offs) == [1, 3] and np.array_equal(T[1], [3, 4, 5, 6, 7, 8, 9, -1, -1, -1])
    offs, T = _partner_table(120, (1, 2, 9), P)
    assert list(offs) == [1, 2] and np.array_equal(T[0], P[:, 0]) and np.array_equal(T[1], P[:, 1])


def spatial_displacements(T=101, N=300, seed=7):
    """Atoms at random positions in a periodic box; a cluster of atoms around the box corner (so it straddles
    the periodic boundary) moves collectively (AR(1) amplitude, random direction per atom); independent noise."""
    rng = np.random.default_rng(seed)
    box = np.array([12.0, 12.0, 12.0])
    pos = rng.uniform(0, 1, (N, 3)) * box
    in_cluster = _min_image_dist(pos, np.zeros(3), box) < 3.0
    P = N * 3
    X = 0.4 * rng.standard_normal((T, P))
    v = np.zeros((N, 3))
    v[in_cluster] = rng.standard_normal((in_cluster.sum(), 3))
    X += np.outer(ar1(0.6, T, rng) * 4.0, (v / np.linalg.norm(v)).ravel())
    return (X - X.mean(axis=0)).reshape(T, N, 3), pos, box, in_cluster


def test_neighborhood_pca_finds_a_spatially_localised_mode_across_the_periodic_boundary():
    u, pos, box, in_cluster = spatial_displacements()
    res = local_pca_analysis(u, None, window_size=10, window_mode="neighborhood", mean_positions=pos, box=box,
                             n_global_modes=10, subspace_dims=(1, 3), n_compare_modes=4,
                             consistency_offsets=(1, 2, 4), n_null=0)
    info, loc = res["windows_info"], res["local"]
    assert info["mode"] == "neighborhood" and res["params"]["window_mode"] == "neighborhood"
    assert np.array_equal(info["centers"], np.arange(300)) and np.array_equal(res["position"], np.arange(300))
    assert info["partners"].shape == (300, 4) and info["center_positions"].shape == (300, 3)
    assert res["comparison"]["interlacing_violation"] < 1e-9                         # holds for any window set
    frac = loc["eigenvalues"][:, 0] / loc["total_variance"]
    share = np.array([in_cluster[w].mean() for w in res["windows"]])               # share of the window inside the cluster
    core, mixed, none = share >= 0.6, (share > 0) & (share < 0.6), share == 0
    assert core.sum() >= 3 and none.sum() > 50
    assert frac[core].mean() > 3 * frac[none].mean()                                # the cluster mode dominates its windows
    assert frac[none].mean() < frac[mixed].mean() < frac[core].mean()               # and grows with the share
    # the cluster is found again in the nearest windows: tracking counts neighbours, not a chain
    trk = res["tracking"]
    assert trk["kind"] == "neighbours" and trk["max_rank"] == 4
    assert trk["chain_length"].min() >= 1 and trk["chain_length"].max() <= 1 + 4
    assert trk["chain_length"][share >= 0.3, 0].mean() > trk["chain_length"][none, 0].mean()
    cons = res["consistency"]
    assert list(cons["offsets"]) == [1, 2, 4] and np.all(cons["shared_fraction"] > 0)
    assert np.all(np.diff(cons["shared_fraction"]) <= 1e-12)                         # farther neighbours share less


def test_neighborhood_mode_needs_positions_and_index_mode_is_unchanged():
    u, m = synthetic_displacements(N=60, seed=2)
    try:
        local_pca_analysis(u, m, window_size=10, window_mode="neighborhood", n_null=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    r = local_pca_analysis(u, m, window_size=10, n_global_modes=5, subspace_dims=(1,), n_compare_modes=3,
                           consistency_offsets=(1, 2), n_null=0)               # default window_mode = "index"
    assert r["params"]["window_mode"] == "index" and r["tracking"]["kind"] == "chain"
    assert r["windows_info"]["centers"] is None and np.array_equal(r["windows"], index_windows(60, 10, 1))
    assert np.allclose(r["position"], r["windows"].mean(axis=1))


def test_marchenko_pastur_edges():
    lo, hi = marchenko_pastur_edges(45, 101)
    assert np.isclose(lo, (1 - np.sqrt(45 / 101)) ** 2) and np.isclose(hi, (1 + np.sqrt(45 / 101)) ** 2)
    assert marchenko_pastur_edges(45, 30)[0] == 0.0
    rng = np.random.default_rng(11)                              # white noise obeys it (approximately)
    ev = np.linalg.eigvalsh(np.cov(rng.standard_normal((4000, 45)), rowvar=False)[:45, :45])
    lo2, hi2 = marchenko_pastur_edges(45, 4000)
    assert lo2 * 0.9 < ev.min() and ev.max() < hi2 * 1.1


# ---------------------------------------------------------------
# Dips of the median participation curve
# ---------------------------------------------------------------

def _dips(y, **kw):
    return detect_participation_dips(np.asarray(y, float), **kw)


def test_dip_detection_isolated_flat_adjacent_and_none():
    d = _dips([9, 9, 5, 9, 9])                                           # isolated: exact index, depth, prominence
    assert list(d["minimum"]) == [2] and list(d["modes"]) == [2]
    assert np.isclose(d["prominence"][0], 4) and np.isclose(d["depth"][0], 4) and np.isclose(d["value"][0], 5)
    d = _dips([9, 9, 5, 5, 5, 9, 9])                                     # flat bottom: one dip, all three modes
    assert list(d["minimum"]) == [2] and list(d["first"]) == [2] and list(d["last"]) == [4]
    assert list(d["modes"]) == [2, 3, 4] and np.isclose(d["prominence"][0], 4)
    d = _dips([9, 5, 5.1, 9, 9])                                         # adjacent: the nearly as low neighbour belongs to it
    assert list(d["minimum"]) == [1] and list(d["modes"]) == [1, 2]
    d = _dips([9, 5, 6, 5.2, 9])                                         # two minima, a maximum between: two dips, it belongs to neither
    assert list(d["minimum"]) == [1, 3] and list(d["modes"]) == [1, 3] and list(d["mode_dip"]) == [0, 1]
    assert np.isclose(d["prominence"][0], 4) and np.isclose(d["prominence"][1], 0.8)
    assert len(_dips([9, 5, 6, 5.2, 9], prominence=1.0)["minimum"]) == 1  # threshold removes the shallow one
    for y in ([9, 8, 7, 6, 5, 4], [3, 3, 3, 3], [9, 9, 5, 9, 9]):         # monotone, flat, below threshold
        assert len(_dips(y, prominence=5.0 if y[0] == 9 and 5 in y else 0.5)["minimum"]) == 0
    assert len(_dips([5, 9, 9, 9, 5])["minimum"]) == 0                    # ends cannot be a dip


def test_dip_detection_has_no_hard_coded_index_and_obeys_scale_search_and_noise():
    base = np.full(40, 9.0)
    for k in (3, 17, 31):
        y = base.copy(); y[k] = 4.0
        assert list(_dips(y)["minimum"]) == [k]                          # the curve decides
    y = base.copy(); y[[5, 6]] = 2.0; y[20] = 6.0; y[33:36] = 7.0
    d = _dips(y)
    assert list(d["minimum"]) == [5, 20, 33] and list(d["first"]) == [5, 20, 33] and list(d["last"]) == [6, 20, 35]
    assert np.allclose(d["prominence"], [7, 3, 2]) and np.array_equal(d["modes"], [5, 6, 20, 33, 34, 35])
    assert list(_dips(y, search=(10, 39))["minimum"]) == [20, 33]       # search range limits the minima, not the prominence
    d2 = _dips(3.0 * y, prominence=1.5)                                   # same curve in other units: same dips
    assert list(d2["minimum"]) == [5, 20, 33] and np.allclose(d2["prominence"], 3 * d["prominence"])
    rng = np.random.default_rng(0)
    noisy = 9 + 0.02 * rng.standard_normal(200)
    assert len(_dips(noisy, prominence=0.5)["minimum"]) == 0              # noise below the prominence is not a dip
    assert len(_dips(noisy, prominence=0.005)["minimum"]) > 5             # ... but is at a small threshold: the threshold is a choice
    assert len(_dips([1.0, np.nan, 0.0, 1.0])["minimum"]) == 0            # no crash on non-finite input


def rattler_displacements(N=360, T=101, L=12.0, seed=1):
    """Atoms at random positions; 3 collective modes with the largest variance, ~10 % rattlers (localised, isotropic,
    4x the amplitude) on an AR(1) noise floor: in windows of 15 atoms the first 3 modes are collective and the next
    ones sit on the rattlers, so n_part(n) has a dip right after the collective modes."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, L, (N, 3))
    disp = 0.05 * rng.standard_normal((T, N, 3))
    for t in range(1, T):
        disp[t] = 0.5 * disp[t - 1] + 0.87 * disp[t]
    ratt = rng.random(N) < 0.1
    disp[:, ratt] *= 4.0
    for sc, phi in ((1.0, .8), (0.8, .7), (0.6, .5)):
        v = rng.standard_normal((N, 3)); v /= np.linalg.norm(v)
        disp += sc * 12 * ar1(phi, T, rng)[:, None, None] * v[None]
    m = np.where(np.arange(N) % 2 == 0, 91.224, 63.546)
    return disp - disp.mean(0), m, pos, np.full(3, L), np.where(np.arange(N) % 2 == 0, "Zr", "Cu")


_RATTLER = {}


def _rattler_result():
    if "res" not in _RATTLER:
        u, m, pos, box, sym = rattler_displacements()
        res = local_pca_analysis(u, m, window_size=15, window_mode="neighborhood", mean_positions=pos, box=box,
                                 n_global_modes=10, subspace_dims=(1, 3), n_compare_modes=2,
                                 consistency_offsets=(1, 2, 4), n_null=2, dip_prominence=0.5)
        _RATTLER.update(res=res, u=u, m=m, pos=pos, box=box, sym=sym)
    return _RATTLER


def test_detected_modes_are_computed_beyond_n_compare_modes_in_data_and_null():
    R = _rattler_result()
    res, d = R["res"], R["res"]["dips"]
    assert len(d["modes"]) > 0 and d["modes"].max() >= 3                  # a dip beyond LOCAL_PCA_N_COMPARE_MODES = 2 here
    need = int(d["modes"].max()) + 1
    assert res["params"]["n_compare_modes"] >= need and res["params"]["n_consistency_modes"] >= need
    assert res["comparison"]["overlap_matrix"].shape[1] >= need
    assert res["consistency"]["matched_overlap"].shape[2] >= need and res["tracking"]["chain_length"].shape[1] >= need
    assert res["summary"]["overlap_best"].shape[0] >= need and res["null"]["overlap_best"].shape[1] >= need
    assert res["null"]["consistency_matched"].shape[2] >= need            # the null is evaluated at the same modes
    assert np.isclose(d["value"].min(), np.median(res["local"]["n_participating"], axis=0)[d["minimum"]].min())
    assert np.array_equal(d["curve"], np.median(res["local"]["n_participating"], axis=0))
    assert np.all(d["prominence"] >= 0.5) and np.all(d["depth"] > 0)


def test_dips_are_saved_in_the_npz():
    import os, tempfile
    from analysis.soft_modes.local_pca import save_local_pca
    R = _rattler_result()
    res = dict(R["res"])
    res["dip_investigation"] = {"modes": np.array([1, 2])}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "r.npz")
        save_local_pca(res, path, eigenvectors_dtype=None)
        z = np.load(path)
        for k in ("modes", "minimum", "first", "last", "depth", "prominence", "value", "curve", "prominence_threshold"):
            assert f"dips/{k}" in z.files
        assert np.array_equal(z["dips/modes"], res["dips"]["modes"]) and "summary_median/n_participating" in z.files
        assert "null/window_median/n_participating" in z.files and "dip_investigation/modes" in z.files


def test_null_is_aggregated_like_the_observed_curve_and_band_widths_are_reported():
    R = _rattler_result()
    res = R["res"]
    sm = res["summary_median"]
    assert np.allclose(sm["n_participating"], np.median(res["local"]["n_participating"], axis=0))
    assert np.allclose(sm["overlap_best"], np.median(res["comparison"]["overlap_best"][:, :sm["overlap_best"].shape[0]], axis=0))
    assert not np.allclose(res["summary"]["participation"] * 15, sm["n_participating"])    # mean PR * n is the mean n_part
    b = local_pca_band_statistics(res)
    assert np.allclose(b["n_participating"][1], sm["n_participating"])                    # blue line = observed median
    nullm = res["null"]["window_median"]["n_participating"]
    assert np.allclose(b["null_median"]["n_participating"], np.median(nullm, axis=0))
    assert np.allclose(b["null_high"]["n_participating"], np.quantile(nullm, res["null"]["quantile"], axis=0))
    rows = {r["key"]: r for r in null_band_diagnostics(res)}
    assert {"n_participating", "overlap_best", "spectrum_fraction"} <= set(rows)
    assert all(r["null_width"] >= 0 and r["window_width"] > 0 for r in rows.values())


def test_amplitude_normalisation_and_eigenvector_participation():
    R = _rattler_result()
    res, u, m = R["res"], R["u"], R["m"]
    X = _flat(u, m)
    loc = res["local"]
    for w, n in ((0, 0), (17, 3), (200, 5), (359, 20)):
        cols = loc["columns"][w]
        v = loc["eigenvectors_mw"][w][:, n]
        a, a_std, sigma = normalized_amplitude(X[:, cols], v)
        assert abs(a.mean()) < 1e-9 * np.abs(a).max()                      # centred data -> zero mean
        assert np.isclose(sigma, np.sqrt(loc["eigenvalues"][w, n]), rtol=1e-8)   # std(a_n) = sqrt(lambda_n), 1/T convention
        assert np.isclose(a_std.mean(), 0, atol=1e-9) and np.isclose(a_std.std(), 1, rtol=1e-12)
        sq = np.sqrt(m[res["windows"][w]])
        vr = real_space_vector(v, m[res["windows"][w]], 3)
        p = norm_fractions(vr)
        assert np.isclose(np.linalg.norm(vr), 1) and np.isclose(p.sum(), 1)
        assert np.isclose(1.0 / (p ** 2).sum(), loc["n_participating"][w, n])   # = PR * n of local_pca()
        assert np.isclose(1.0 / (15 * (p ** 2).sum()), loc["participation"][w, n])
        manual = (v.reshape(15, 3) / sq[:, None])
        assert np.allclose(vr, manual / np.linalg.norm(manual))               # mass weighting undone, re-normalised
    rng = np.random.default_rng(3)                                           # a Gaussian amplitude is N(0, 1) after normalising
    from scipy.stats import kurtosis, skew
    _, z, _ = normalized_amplitude(rng.standard_normal((20000, 4)) * [1, 2, 3, 4], np.array([0.5, 0.5, 0.5, 0.5]))
    assert abs(skew(z)) < 0.05 and abs(kurtosis(z)) < 0.1


def _stub_cutoffs():
    import sys, types
    try:
        import cutoffs                                                       # noqa: F401  (needs the trajectory file)
    except Exception:
        stub = types.ModuleType("cutoffs")
        stub.CUTOFF_DICT = stub.GRAPH_CUTOFF_DICT = {}
        for k in ("CUTOFF_ZRZR", "CUTOFF_ZRCU", "CUTOFF_CUCU", "PEAK_ZRZR", "PEAK_ZRCU", "PEAK_CUCU"):
            setattr(stub, k, 3.5)
        sys.modules["cutoffs"] = stub


def test_investigation_dominant_atoms_elements_environment_and_representative_windows():
    _stub_cutoffs()
    R = _rattler_result()
    res = R["res"]
    cut = {("Zr", "Zr"): 3.0, ("Zr", "Cu"): 2.6, ("Cu", "Cu"): 2.4}
    inv = investigate_dip_modes(res, R["u"], R["m"], R["sym"], R["pos"], R["box"], cut, max_modes=4, top_atoms=3)
    assert 0 < len(inv["modes"]) <= 4 and set(inv["modes"]) <= set(res["dips"]["modes"].tolist())
    assert len(inv["table"]) == len(inv["modes"]) * 3 * 3 and len(inv["panels"]) == len(inv["modes"]) * 3
    npart = res["local"]["n_participating"]
    for k, mo in enumerate(inv["modes"]):
        lo, me, hi = inv["window"][k]
        assert npart[lo, mo] <= npart[me, mo] <= npart[hi, mo]            # low / median / high participation windows
        assert np.allclose(inv["n_part"][k], npart[inv["window"][k], mo])  # recomputed from the real-space vector
        assert np.allclose(inv["eigenvalue"][k], res["local"]["eigenvalues"][inv["window"][k], mo])
    for pn in inv["panels"]:
        assert np.isclose(pn["p"].sum(), 1) and pn["top"][0] == np.argmax(pn["p"])
        assert set(pn["atoms"]) <= set(range(len(R["sym"])))
    last = {}
    for r in inv["table"]:
        key = (r["mode"], r["window_class"])
        assert r["element"] == R["sym"][r["atom"]]                          # element of the atom
        assert r["rank"] == 1 or r["norm_fraction"] <= last[key][0] + 1e-12  # sorted by norm fraction
        assert r["cumulative_fraction"] >= r["norm_fraction"] - 1e-12
        assert np.isclose(r["vx"] ** 2 + r["vy"] ** 2 + r["vz"] ** 2, r["norm_fraction"])
        last[key] = (r["norm_fraction"],)
    env = inv["environment"]                                                # coordination against a brute-force count
    pos, box = R["pos"], R["box"]
    for i in (0, 5, 77, 301):
        d = pos - pos[i]; d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=1)
        cn = sum(1 for j in range(len(pos)) if j != i and r[j] <= {("Zr", "Zr"): 3.0, ("Zr", "Cu"): 2.6, ("Cu", "Zr"): 2.6,
                 ("Cu", "Cu"): 2.4}[(R["sym"][i], R["sym"][j])])
        assert env["cn"][i] == cn
        assert np.isclose(env["d_nn"][i], np.sort(r)[1])
    assert np.array_equal(env["species"], ["Cu", "Zr"])
    for mo, s in inv["windows_all"].items():
        assert np.isclose(s["dominant_element_fraction"].sum(), 1) and np.isclose(s["norm_share_element"].sum(), 1, atol=1e-9)
    arr = dip_investigation_arrays(inv)
    assert np.array_equal(arr["modes"], inv["modes"])
    import os, tempfile, csv
    with tempfile.TemporaryDirectory() as tmp:
        write_dip_csv(inv, os.path.join(tmp, "a.csv"))
        rows = list(csv.DictReader(open(os.path.join(tmp, "a.csv"))))
        assert len(rows) == len(inv["table"]) and {"mode", "atom", "element", "norm_fraction", "cn", "d_nn"} <= set(rows[0])
    modes, trunc = select_modes(res["dips"], 1)
    assert len(modes) == 1 and trunc == (len(res["dips"]["modes"]) > 1)
    assert list(representative_windows([5, 1, 3, 2, 4], (0, 0.5, 1))) == [1, 2, 0]



# ---------------------------------------------------------------
# Spatial ordering of the windows, full-system participation spectrum
# ---------------------------------------------------------------

def test_hilbert_curve_is_continuous_and_a_permutation():
    for D, bits in ((2, 3), (3, 2), (3, 3)):
        g = np.array(np.meshgrid(*[np.arange(1 << bits)] * D, indexing="ij")).reshape(D, -1).T
        h = hilbert_index(g, bits)
        assert np.array_equal(np.sort(h), np.arange(len(g)))               # every cell once
        path = g[np.argsort(h)]
        assert np.all(np.abs(np.diff(path, axis=0)).sum(axis=1) == 1)       # consecutive cells are face neighbours


def test_spatial_window_order_keeps_neighbours_together():
    rng = np.random.default_rng(0)
    box = np.array([30.0, 28.0, 26.0])
    c = rng.uniform(0, 1, (3000, 3)) * box
    order = spatial_window_order(c, box)
    assert np.array_equal(np.sort(order), np.arange(len(c))) and np.array_equal(order, spatial_window_order(c, box))
    near = order_locality(c, box, order)
    assert near < 0.25 * order_locality(c, box)                              # atom-index (here random) order is far worse
    assert near < 0.5 * order_locality(c, box, np.argsort(c[:, 0], kind="stable"))   # and so is sorting along x
    assert np.array_equal(order, spatial_window_order(c + box * np.array([1, -2, 3]), box))   # periodic images, same order


def test_windows_info_has_the_spatial_order_and_the_heat_map_columns_follow_it():
    from visualization.plot import _window_axis
    R = _rattler_result()
    res = R["res"]
    wi = res["windows_info"]
    assert np.array_equal(wi["spatial_order"], spatial_window_order(wi["center_positions"], R["box"]))
    data = res["local"]["n_participating"]
    img, x, label = _window_axis(res, data)
    assert np.array_equal(img, data[wi["spatial_order"]]) and np.array_equal(x, np.arange(len(data))) and "Hilbert" in label
    res_idx = {"windows_info": {}, "params": {"window_mode": "index"}, "position": np.arange(len(data))}
    img, x, label = _window_axis(res_idx, data)
    assert np.array_equal(img, data) and "atom index" in label


def test_full_system_participation_spectrum():
    from analysis.soft_modes.modes import participation_spectrum as global_participation_spectrum
    u, m, pos, box, sym = rattler_displacements()
    T, N, _ = u.shape
    gp = global_participation_spectrum(u, m, n_null=3)
    ref = global_reference(u, m, n_modes=T - 1)
    assert gp["n_participating"].shape == (T - 1,) and np.allclose(gp["n_participating"], ref["n_participating"])
    assert np.all(gp["n_participating"] >= 1 - 1e-9) and np.all(gp["n_participating"] <= N + 1e-9)
    assert gp["null"].shape == (3, T - 1) and np.all(gp["null_high"] >= gp["null_median"])
    assert np.median(gp["null_median"]) > 0.4 * N and gp["n_participating"][3] < 0.5 * gp["null_median"][3]   # rattler modes are below noise
    rng = np.random.default_rng(0)                                           # homogeneous noise: random unit vectors, n_part = 0.6 N for D = 3
    gn = global_participation_spectrum(rng.standard_normal((101, N, 3)), np.ones(N), n_null=2)
    assert abs(np.median(gn["n_participating"]) / N - 0.6) < 0.03 and abs(np.median(gn["null_median"]) / N - 0.6) < 0.03
    u2 = u.copy(); u2[:, 7, :] *= 400.0                                      # one atom vibrating far more than the rest
    g2 = global_participation_spectrum(u2, m)
    assert g2["n_participating"][0] < 1.5 and g2["null"] is None
    # same convention as the local windows: a window of ALL atoms gives the local participation of the global mode
    all_atoms = local_pca(u[:, :20], m[:20], windows=np.arange(20)[None, :], store_eigenvectors=False)
    g20 = global_reference(u[:, :20], m[:20], n_modes=5)
    assert np.allclose(all_atoms["n_participating"][0, :5], g20["n_participating"])



def test_leading_subspace_overlaps_equal_the_svd_definition_for_every_M():
    from analysis.soft_modes.local_pca import leading_subspace_overlaps, _orth
    rng = np.random.default_rng(3)
    K = 30
    for rows in (3, 12, 30, 45):                                   # fewer rows than columns, equal, more
        V1 = np.linalg.qr(rng.standard_normal((90, K)))[0]
        V2 = np.linalg.qr(rng.standard_normal((90, K)))[0]
        A1, A2 = V1[rng.choice(90, rows, replace=False)], V2[rng.choice(90, rows, replace=False)]
        ref = []
        for M in range(1, K + 1):
            Q1, Q2 = _orth(A1[:, :M]), _orth(A2[:, :M])
            ref.append(np.linalg.norm(Q1.T @ Q2) ** 2 / min(Q1.shape[1], Q2.shape[1]))
        assert np.allclose(leading_subspace_overlaps(A1, A2), ref, atol=1e-10)
    A = rng.standard_normal((40, K))                                # identical blocks: overlap 1 for every M
    assert np.allclose(leading_subspace_overlaps(A, A), 1.0)
    B = A.copy(); B[:, 5:] = rng.standard_normal((40, K - 5))      # the first five columns equal: M <= 5 gives 1
    o = leading_subspace_overlaps(A, B)
    assert np.allclose(o[:5], 1.0) and o[9] < 0.999


def test_share_profile_and_offsets_reach_windows_that_barely_overlap():
    from analysis.soft_modes.local_pca import share_profile, offsets_for_shares
    W = index_windows(60, 10, 1)                                    # window w+j shares 10-j atoms
    offs, prof = share_profile(W)
    assert list(offs) == list(range(1, 11)) and np.allclose(prof, (10 - offs) / 10)      # j = 10: nothing shared
    assert offsets_for_shares(offs, prof, (0.5, 0.1), base_offsets=(1,)) == (1, 5, 9)
    assert offsets_for_shares(offs, prof, (0.0,)) == ()             # an offset at which nothing is shared compares nothing
    # neighbourhood windows: the partner table is wide enough for the profile to fall to ~0
    u, pos, box, _ = spatial_displacements()
    kw = dict(window_size=10, window_mode="neighborhood", mean_positions=pos, box=box, n_global_modes=5,
              subspace_dims=(1, 3), n_compare_modes=4, consistency_offsets=(1, 2, 4), n_null=0)
    base = local_pca_analysis(u, None, **kw)
    res = local_pca_analysis(u, None, consistency_shares=(0.5, 0.3, 0.15, 0.05), **kw)
    cons = res["consistency"]
    offs, prof = res["windows_info"]["share_profile"]
    assert np.all(np.diff(prof) <= 0.05) and prof[0] > 0.5 and prof[-1] < 0.01
    assert {1, 2, 4} <= set(cons["offsets"]) and len(cons["offsets"]) > 5
    assert np.all(np.diff(cons["shared_fraction"]) <= 1e-12)        # still ordered: farther windows share less
    assert cons["shared_fraction"].min() < 0.2 < base["consistency"]["shared_fraction"].min()
    assert res["tracking"]["max_rank"] == 4 and np.array_equal(res["tracking"]["chain_length"], base["tracking"]["chain_length"])
    assert res["windows_info"]["partners"].shape[1] >= 100          # rank 10 x window size
    for j in (1, 2, 4):                                              # the old offsets give the old numbers
        o, ob = list(cons["offsets"]).index(j), list(base["consistency"]["offsets"]).index(j)
        assert np.allclose(res["summary"]["consistency_matched"][o], base["summary"]["consistency_matched"][ob], equal_nan=True)
        assert np.allclose(res["summary"]["consistency_subspace"][o], base["summary"]["consistency_subspace"][ob], equal_nan=True)


def test_consistency_covers_all_modes_and_all_subspace_sizes():
    u, pos, box, _ = spatial_displacements(N=120)
    res = local_pca_analysis(u, None, window_size=8, window_mode="neighborhood", mean_positions=pos, box=box,
                             n_global_modes=5, subspace_dims=(1, 3, 10), n_compare_modes=3,
                             consistency_offsets=(1, 3), consistency_shares=(0.3,), n_null=2)
    cons, S, K = res["consistency"], res["summary"], res["local"]["n_coords"]
    n_off, n_win = len(cons["offsets"]), len(res["windows"])
    assert cons["matched_overlap"].shape == (n_off, n_win, K) and cons["weight"].shape == (n_off, n_win, K)
    assert cons["subspace_overlap_all"].shape == (n_off, n_win, K)
    assert cons["matched_index"].shape == (n_off, n_win, 3) and cons["n_modes"] == 3
    dims = cons["subspace_dims"]
    assert np.array_equal(cons["subspace_overlap"], cons["subspace_overlap_all"][:, :, dims - 1], equal_nan=True)
    assert np.allclose(S["consistency_matched"], S["consistency_matched_all"][:, :3], equal_nan=True)
    assert S["consistency_matched_all"].shape == (n_off, K) and S["consistency_subspace_all"].shape == (n_off, K)
    assert S["consistency_matched_count"].shape == (n_off, K) and S["consistency_matched_count"].max() <= n_win
    assert np.array_equal(S["consistency_matched_count"], np.isfinite(np.where(cons["weight"] >= 0.5, cons["matched_overlap"], np.nan)).sum(axis=1))
    sa = cons["subspace_overlap_all"]
    assert np.nanmin(sa) >= 0 and np.nanmax(sa) <= 1 + 1e-9 and np.nanmin(cons["matched_overlap"]) >= 0
    nl = res["null"]
    assert nl["consistency_matched_all"].shape == (2, n_off, K) and nl["consistency_subspace_all"].shape == (2, n_off, K)
    assert nl["consistency_matched_count"].shape == (2, n_off, K)


def test_consistency_plot_names_every_line_has_a_linear_count_axis_and_rising_share_axes():
    import matplotlib
    matplotlib.use("Agg")
    import visualization.plot as vp
    u, pos, box, _ = spatial_displacements(N=200)
    res = local_pca_analysis(u, None, window_size=8, window_mode="neighborhood", mean_positions=pos, box=box,
                             n_global_modes=5, subspace_dims=(1, 3, 5), n_compare_modes=4, consistency_offsets=(1, 2, 4),
                             consistency_shares=(0.5, 0.2, 0.05), track_max_rank=6, n_null=2, dip_prominence=0.2)
    got = {}
    orig = vp._save
    vp._save = lambda fig, path, name: got.setdefault("fig", fig)
    try:
        vp.plot_local_consistency(res, min_windows=5)
    finally:
        vp._save = orig
    fig = got["fig"]
    axs = fig.axes
    a, b, c = axs[0], axs[1], axs[2]
    assert len(axs) == 3                                                   # the 3D landscapes are not in this figure
    for ax in (a, b):
        assert ax.get_xlim() == (0.0, 1.0)                             # rising from 0 to 1 whatever the data cover
        assert all(np.all(np.diff(ln.get_xdata()) > 0) for ln in ax.lines if len(ln.get_xdata()) > 2)
        lab = [ln.get_label() for ln in ax.lines]
        assert all(not x.startswith("_") for x in lab), lab            # every drawn line has a legend entry
        shown = [t.get_text() for t in ax.get_legend().get_texts()]
        assert len(shown) >= len(lab)
    shown_b = [t.get_text() for t in b.get_legend().get_texts()]
    for M in (1, 3, 5):
        assert f"M = {M}" in shown_b and any(t.startswith(f"M = {M}: null") for t in shown_b)   # each tick set is named
    assert c.get_yscale() == "linear" and c.get_ylim()[0] == 0.5          # counts 1..1+max_rank are shown linearly
    assert list(c.get_yticks()[(c.get_yticks() >= 0.5) & (c.get_yticks() <= 7.5)]) == [1, 2, 3, 4, 5, 6, 7]
    fig.tight_layout()
    vp._prune_thin_bands(fig)                                              # what _save does before saving
    for ax in (a, b, c):                                                   # the legend is exactly what is drawn
        h, l = ax.get_legend_handles_labels()
        assert sorted(t.get_text() for t in ax.get_legend().get_texts()) == sorted(l)


def test_consistency_is_pooled_by_exact_number_of_common_atoms():
    u, pos, box, _ = spatial_displacements(N=200)
    n = 8
    res = local_pca_analysis(u, None, window_size=n, window_mode="neighborhood", mean_positions=pos, box=box,
                             n_global_modes=5, subspace_dims=(1, 3), n_compare_modes=3, consistency_offsets=(1, 2),
                             consistency_shares=(0.5, 0.3, 0.1), n_null=2)
    cons, S = res["consistency"], res["summary"]
    assert list(cons["common_atoms"]) == list(range(1, n)) and cons["window_size"] == n   # c = 1..n-1, x = c / n
    ncom = cons["n_common"]
    assert ncom.min() >= 0 and ncom.max() <= n
    pairs = S["consistency_pairs_by_common"]
    assert list(pairs) == [int((ncom == c).sum()) for c in cons["common_atoms"]]
    assert pairs[0] > 0 and pairs[-1] > 0                  # the data reach 1 / n common atoms and n - 1 common atoms
    for i, c in enumerate(cons["common_atoms"]):
        m = ncom == c
        assert np.allclose(S["consistency_subspace_by_common"][i], np.nanmean(cons["subspace_overlap_all"][m], axis=0))
    assert res["null"]["consistency_subspace_by_common"].shape[1:] == S["consistency_subspace_by_common"].shape


def test_subspace_dims_offsets_and_shares_follow_the_rules_of_the_config():
    from analysis.soft_modes.local_pca import auto_subspace_dims, auto_offsets, all_shares
    assert auto_subspace_dims(90, 0.10, 15) == tuple(range(1, 10))            # 10 % of 90
    assert auto_subspace_dims(300, 0.10, 15) == tuple(range(1, 16))            # 30 -> capped at 15
    assert auto_subspace_dims(45, 0.10, 15) == tuple(range(1, 6))              # 4.5 -> 5 (nearest, half up)
    assert auto_offsets(30, 0.33, 15) == tuple(range(1, 11))                   # 9.9 -> 10
    assert auto_offsets(100, 0.33, 15) == tuple(range(1, 16))                  # 33 -> capped at 15
    assert auto_offsets(3, 0.33, 15) == (1,)                                   # never empty
    assert all_shares(4) == (0.25, 0.5, 0.75) and len(all_shares(30)) == 29
    u, pos, box, _ = spatial_displacements(N=200)
    n = 8
    res = local_pca_analysis(u, None, window_size=n, window_mode="neighborhood", mean_positions=pos, box=box,
                             n_global_modes=5, subspace_dims=None, n_compare_modes=3, consistency_offsets=None,
                             consistency_shares="all", n_null=0, subspace_dims_fraction=0.2, subspace_dims_max=3,
                             offsets_fraction=0.5, offsets_max=15)
    p = res["params"]
    assert p["subspace_dims"] == (1, 2, 3)                                     # 20 % of 24 = 4.8 -> 5, capped at 3
    assert p["consistency_shares"] == tuple(c / n for c in range(1, n))
    assert set(range(1, 5)) <= set(p["consistency_offsets"])                   # 1..4 (50 % of 8), plus offsets for the shares
    assert list(res["consistency"]["subspace_dims"]) == [1, 2, 3]
    try:
        local_pca_analysis(u, None, window_size=n, window_mode="neighborhood", mean_positions=pos, box=box,
                           consistency_shares="some")
        raise AssertionError("a wrong string must be refused")
    except ValueError:
        pass


def test_pooled_matched_overlap_is_a_weighted_mean_over_every_pair_and_the_curves_reach_the_data_limits():
    import matplotlib
    matplotlib.use("Agg")
    import visualization.plot as vp
    u, pos, box, _ = spatial_displacements(N=200)
    n = 8
    res = local_pca_analysis(u, None, window_size=n, window_mode="neighborhood", mean_positions=pos, box=box,
                             n_global_modes=5, subspace_dims=None, n_compare_modes=4, consistency_offsets=None,
                             consistency_shares="all", n_null=2)
    cons, S = res["consistency"], res["summary"]
    ncom, ov, wt = cons["n_common"], cons["matched_overlap"], cons["weight"]
    for i, c in enumerate(cons["common_atoms"]):
        m = ncom == c
        if not m.any():
            continue
        exp = (ov[m] * wt[m]).sum(axis=0) / wt[m].sum(axis=0)
        assert np.allclose(S["consistency_matched_by_common"][i], exp, equal_nan=True)
    have = np.where(S["consistency_pairs_by_common"] > 0)[0]
    got = {}
    orig = vp._save
    vp._save = lambda fig, path, name: got.setdefault("fig", fig)
    try:
        vp.plot_local_consistency(res)
    finally:
        vp._save = orig
    for ax in got["fig"].axes[:2]:                                             # every line spans all shares with pairs
        for ln in ax.lines:
            x = np.asarray(ln.get_xdata(), float)
            y = np.asarray(ln.get_ydata(), float)
            if len(x) == len(cons["common_atoms"]):
                assert np.isfinite(y[have]).all(), ln.get_label()
                assert x[have][0] == cons["common_atoms"][have][0] / n and x[have][-1] == cons["common_atoms"][have][-1] / n


def test_interactive_landscapes_are_rotatable_ascending_and_have_one_legend_entry_per_kind():
    import visualization.plot as vp
    try:
        import plotly  # noqa: F401
    except ImportError:
        print("    (plotly is not installed: landscape test skipped)")
        return
    u, pos, box, _ = spatial_displacements(N=200)
    res = local_pca_analysis(u, None, window_size=8, window_mode="neighborhood", mean_positions=pos, box=box,
                             n_global_modes=5, subspace_dims=None, n_compare_modes=4, consistency_offsets=None,
                             consistency_shares="all", n_null=2, dip_prominence=0.2)
    fig = vp._consistency_landscape_figure(res)
    surfaces = [t for t in fig.data if t.type == "surface"]
    assert len(surfaces) == 4                                                  # observed + null, for (a) and (b)
    for t in surfaces:
        assert np.all(np.diff(t.y) > 0) and t.y[0] > 0 and t.y[-1] < 1        # share rises, between 1/n and (n-1)/n
        assert np.all(np.diff(t.x) > 0)
    assert fig.layout.scene.dragmode == "orbit" and fig.layout.scene2.dragmode == "orbit"   # free rotation by dragging
    names = [t.name for t in fig.data if t.showlegend]
    assert len(names) == len(set(names))                                       # one entry per kind, not per landscape
    assert "observed (window mean)" in names and any(x.startswith("shifted-column null") for x in names)
    assert ("detected dip modes" in names) == any(t.type == "scatter3d" for t in fig.data)
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = vp.plot_local_consistency_landscapes(res, save_path=os.path.join(d, "landscapes.html"))
        assert os.path.getsize(path) > 1e6 and "plotly" in open(path, encoding="utf-8").read(2000000).lower()


def test_a_band_thinner_than_a_line_is_not_drawn_and_not_in_the_legend():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import visualization.plot as vp
    x = np.linspace(0, 1, 20)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, x, label="curve")
    vp._band_fill(ax, x, x - 0.2, x + 0.2, color="C0", alpha=0.3, label="thick band")
    vp._band_fill(ax, x, x + 1e-4, x + 2e-4, color="C1", alpha=0.3, label="thin band")
    vp._null_band(ax, x, x + 0.5, x + 0.5 + 1e-4, "thin null", q=95)
    vp._band_span(ax, "v", 0.5, 0.5 + 1e-4, color="C2", label="thin span")
    thick_null = vp._null_band(ax, x, x * 0 - 0.5, x * 0 - 0.2, "thick null", q=95)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 2)
    ax.legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    lim = (ax.get_xlim(), ax.get_ylim())
    assert vp._prune_thin_bands(fig) == 3                       # thin band, thin null band, thin span
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert not any("thin band" in t or "thin span" in t or "thin null: band" in t for t in labels), labels
    assert "thick band" in labels and "curve" in labels and any("thick null: band" in t for t in labels)
    assert "null median" in labels and sum(t.startswith("null 95% (dashed)") for t in labels) == 1   # only the thick one
    assert (ax.get_xlim(), ax.get_ylim()) == lim                # limits are kept
    h, l = ax.get_legend_handles_labels()
    assert sorted(labels) == sorted(l)                          # everything in the legend is drawn, and v.v.
    assert vp._prune_thin_bands(fig) == 0                       # nothing left to remove
    # the rule is "narrower than matplotlib's default line width"
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    fig2.tight_layout()
    h_pt = ax2.get_position().height * fig2.get_figheight() * 72.0         # axis height in points for the range 0..1
    w_line = plt.rcParams["lines.linewidth"]
    vp._band_fill(ax2, [0, 1], [0.5, 0.5], [0.5 + 0.9 * w_line / h_pt] * 2, color="C0", label="just under a line")
    vp._band_fill(ax2, [0, 1], [0.2, 0.2], [0.2 + 1.1 * w_line / h_pt] * 2, color="C1", label="just over a line")
    ax2.legend()
    assert vp._prune_thin_bands(fig2) == 1
    assert [t.get_text() for t in ax2.get_legend().get_texts()] == ["just over a line"]
    plt.close("all")


def test_legend_above_the_data_is_rebuilt_when_a_thin_band_is_dropped():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import visualization.plot as vp
    x = np.linspace(0, 1, 20)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, x, label="curve")
    vp._null_band(ax, x, x, x + 1e-5, "null", q=95)
    ax.set_ylim(0, 1)
    vp._legend_on_top(ax, ncol=1)
    top_with = ax.get_ylim()[1]
    fig.tight_layout()
    assert vp._prune_thin_bands(fig) == 1
    assert sorted(t.get_text() for t in ax.get_legend().get_texts()) == ["curve", "null median"]
    assert ax.get_ylim()[1] < top_with                          # less head room is needed for the shorter legend
    plt.close("all")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} tests passed")
