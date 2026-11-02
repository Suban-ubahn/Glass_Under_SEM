# =============================================================
# scripts/run_soft_tls_dmin_comparison.py -- entry point (was the __main__ block of analysis/soft_modes/comparison.py)
# =============================================================

from analysis.soft_modes.comparison import run_full_comparison


if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import (
        TRAJECTORY_FILE,
        N_SOFT_MODES,
        SOFT_MODE_PR_MAX,
        TLS_DELTA_BIC_THRESHOLD,
        TLS_SEPARATION_THRESHOLD,
        TLS_MIN_WEIGHT_FRACTION,
        TLS_MIN_SWITCHES,
        D2MIN_QUANTILE_THRESHOLD,
        D2MIN_TOP_FRACTION,
    )
    from cutoffs import CUTOFF_DICT
    from analysis.soft_modes.modes import soft_mode_analysis_3d
    from analysis.soft_modes.tls import detect_tls_direct, detect_tls_amplitude

    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames, {len(trajectory[0])} atoms.")

    modes_3d = soft_mode_analysis_3d(trajectory, n_soft_modes=N_SOFT_MODES, pr_max=SOFT_MODE_PR_MAX)
    tls_direct = detect_tls_direct(
        trajectory, delta_bic_threshold=TLS_DELTA_BIC_THRESHOLD,
        separation_threshold=TLS_SEPARATION_THRESHOLD,
        min_weight_fraction=TLS_MIN_WEIGHT_FRACTION, min_switches=TLS_MIN_SWITCHES,
    )
    tls_amp = detect_tls_amplitude(
        modes_3d, delta_bic_threshold=TLS_DELTA_BIC_THRESHOLD,
        separation_threshold=TLS_SEPARATION_THRESHOLD,
        min_weight_fraction=TLS_MIN_WEIGHT_FRACTION, min_switches=TLS_MIN_SWITCHES,
    )
    tls_score = tls_amp["tls_score"].copy()
    tls_score[tls_direct["is_tls"]] += 1.0   # simple union-style combined score

    comparison = run_full_comparison(
        trajectory, modes_3d["softness"], tls_score,
        cutoff=CUTOFF_DICT, quantile_threshold=D2MIN_QUANTILE_THRESHOLD,
        top_frac=D2MIN_TOP_FRACTION,
    )

    print("\n-- Soft spots vs. TLS --")
    st = comparison["soft_vs_tls"]
    print(f"Spearman r = {st['spearman_r']:.3f} (p={st['spearman_p']:.2e})")
    if "jaccard_top_softness_vs_tls" in st:
        print(f"Jaccard(top-softness set, TLS set) = {st['jaccard_top_softness_vs_tls']:.3f}")

    print("\n-- Soft spots -> D^2_min --")
    sd = comparison["soft_vs_d2min"]
    print(f"Spearman r = {sd['spearman_r']:.3f} (p={sd['spearman_p']:.2e}), AUC = {sd['auc']:.3f}")
    print(f"Enrichment (top {D2MIN_TOP_FRACTION:.0%} softest): "
          f"{sd['enrichment']['enrichment']:.2f}x mean D^2_min")

    print("\n-- TLS -> D^2_min --")
    td = comparison["tls_vs_d2min"]
    print(f"Spearman r = {td['spearman_r']:.3f} (p={td['spearman_p']:.2e}), AUC = {td['auc']:.3f}")
    print(f"Enrichment (top {D2MIN_TOP_FRACTION:.0%} by TLS score): "
          f"{td['enrichment']['enrichment']:.2f}x mean D^2_min")
