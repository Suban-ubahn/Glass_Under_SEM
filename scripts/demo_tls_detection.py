# =============================================================
# scripts/demo_tls_detection.py -- entry point (was the __main__ block of analysis/soft_modes/tls.py)
# =============================================================

from analysis.soft_modes.tls import detect_tls_amplitude, detect_tls_direct


if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import (
        TRAJECTORY_FILE, N_Z_SLICES, N_SOFT_MODES, SOFT_MODE_PR_MAX,
        TLS_DELTA_BIC_THRESHOLD, TLS_SEPARATION_THRESHOLD,
        TLS_MIN_WEIGHT_FRACTION, TLS_MIN_SWITCHES,
    )
    from analysis.soft_modes.modes import soft_mode_analysis_3d
 
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
