# =============================================================
# scripts/demo_soft_modes.py -- entry point (was the __main__ block of analysis/soft_modes/modes.py)
# =============================================================

from analysis.soft_modes.modes import (
    soft_mode_analysis_2d_slices,
    soft_mode_analysis_3d,
)


if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import TRAJECTORY_FILE, N_Z_SLICES, N_SOFT_MODES, SOFT_MODE_PR_MAX

    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames, {len(trajectory[0])} atoms.")

    print("\n--- 3D full-system soft-mode analysis ---")
    result_3d = soft_mode_analysis_3d(
        trajectory, n_soft_modes=N_SOFT_MODES, pr_max=SOFT_MODE_PR_MAX,
    )
    print(f"Resolved {len(result_3d['eigenvalues'])} modes.")
    print(f"Softest eigenvalue: {result_3d['eigenvalues'][0]:.4f} A^2")
    print(f"Softest mode participation ratio: {result_3d['participation'][0]:.4f}")
    print(f"Softness score range: [{result_3d['softness'].min():.4f}, "
          f"{result_3d['softness'].max():.4f}]")

    print(f"\n--- 2D per-slice soft-mode analysis ({N_Z_SLICES} slices) ---")
    result_2d = soft_mode_analysis_2d_slices(
        trajectory, n_slices=N_Z_SLICES,
        n_soft_modes=N_SOFT_MODES, pr_max=SOFT_MODE_PR_MAX,
    )
    for s, slc in enumerate(result_2d["slices"]):
        if slc is None:
            print(f"  slice {s}: skipped (too few atoms)")
            continue
        z0, z1 = slc["z_range"]
        print(f"  slice {s} (z={z0:.1f}-{z1:.1f} A, {len(slc['atom_indices'])} atoms): "
              f"softest eigenvalue={slc['eigenvalues'][0]:.4f} A^2, "
              f"PR={slc['participation'][0]:.4f}")
