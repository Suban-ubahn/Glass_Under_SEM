# =============================================================
# scripts/pypty_comparison.py -- entry point (was the __main__ block of analysis/pypty_comparison.py)
# =============================================================

import numpy as np
import os
from analysis.pypty_comparison import (
    OUTPUT_DIR,
    align_and_correlate,
    build_ground_truth_potential,
    load_reconstruction,
    match_shape,
)
from visualization.plot import (
    plot_cross_correlation,
    plot_pypty_reconstruction,
    plot_pypty_vs_ground_truth,
)


if __name__ == "__main__":
    print("Loading pypty reconstruction...")
    co, cp, params = load_reconstruction()
    print("Object shape:", co.shape)   # confirm axis order before trusting the slice below

    # num_slices=1, so squeeze the singleton z axis; adjust indices once you confirm co.shape
    obj_phase = np.angle(co[..., 0, 0])
    obj_amp = np.abs(co[..., 0, 0])
    probe_amp = np.abs(cp[..., 0])

    print("Plotting reconstruction...")
    plot_pypty_reconstruction(
        obj_phase, obj_amp, probe_amp,
        output_path=os.path.join(OUTPUT_DIR, "reconstruction_preview.png"),
    )

    print("Building ground-truth projected potential from MD structure...")
    px, py = params["pixel_size_x_A"], params["pixel_size_y_A"]
    ground_truth = build_ground_truth_potential(px, py)
    ground_truth = match_shape(ground_truth, obj_phase.shape)

    print("Aligning and computing quantitative comparison...")
    recon_norm, gt_aligned, r, p_value, xcorr, shift = align_and_correlate(obj_phase, ground_truth)
    print(f"Pearson r = {r:.3f}, p = {p_value:.2e}")
    print(f"Best-fit shift: dy={shift[0]}, dx={shift[1]}")

    plot_pypty_vs_ground_truth(
        recon_norm, gt_aligned, r,
        output_path=os.path.join(OUTPUT_DIR, "pypty_vs_ground_truth.png"),
    )

    plot_cross_correlation(xcorr)

    print("Done.")
