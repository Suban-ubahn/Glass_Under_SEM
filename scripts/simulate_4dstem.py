# =============================================================
# scripts/simulate_4dstem.py -- entry point (was the __main__ block of diffraction/simulate.py)
# =============================================================

import abtem
import os
from config import BLOCKED_VAL, SCAN_POSITION
from diffraction.simulate import OUTPUT_DIR, compute_average_structure, simulate_4d_stem
from visualization.plot import (
    plot_detector_mask,
    plot_diffraction_single_position,
    plot_virtual_detector_heatmap,
)


if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import TRAJECTORY_FILE
    from analysis.projection_maps import compare_abtem_to_projection_maps
    import time


    # Starting clock
    t0 = time.time()

    print("Loading trajectory...")
    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames")

    avg_atoms = compute_average_structure(trajectory)
    # correct_atom_types() removed — data_io.trajectory.load_trajectory()
    # now sets real atomic numbers on every loaded frame directly, so
    # trajectory[0].numbers (used above via `ref.numbers`) is already
    # correct. Calling the old correction again here would now raise
    # a KeyError (TYPE_TO_Z has no entry for 40 or 29, only 1 and 2).

    # resolution_test(avg_atoms)
    diffraction_patterns = simulate_4d_stem(avg_atoms, output_file="outputs/diffraction/4d_stem.zarr")
    diffraction_patterns = abtem.from_zarr("outputs/diffraction/4d_stem.zarr")
    
    # plot_average_diffraction(
    #     diffraction_patterns,
    #     blocked=BLOCKED_VAL,
    #     save_path="outputs/diffraction/mean_diffraction.png",
    # )
    
    plot_diffraction_single_position(
        diffraction_patterns,
        blocked=BLOCKED_VAL,
        scan_position=SCAN_POSITION,
        save_path=os.path.join(OUTPUT_DIR, "single_position.png"),
    )

    bf_image, bf_mask = plot_virtual_detector_heatmap(
        diffraction_patterns,
        inner_radius_px=0,
        outer_radius_px=None,
        center=None,
        log_scale=False,
        normalize=True,
        title="Virtual Bright-Field Image",
        save_path=os.path.join(OUTPUT_DIR, "virtual_bright_field.png"),
    )
    plot_detector_mask(bf_mask)

    df_image, df_mask = plot_virtual_detector_heatmap(
        diffraction_patterns,
        inner_radius_px=70,
        outer_radius_px=180,
        center=None,
        log_scale=True,
        normalize=True,
        title="Virtual Dark-Field Image",
        save_path=os.path.join(OUTPUT_DIR, "virtual_dark_field.png"),
    )
    plot_detector_mask(df_mask)

    compare_abtem_to_projection_maps(bf_image)

    # simulate_plane_wave(avg_atoms) # also generates a plot

    print(f"Computation time: {(time.time() - t0) / 60:.1f} minutes")
