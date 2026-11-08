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


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} tests passed")
