# =============================================================
# analysis/pypty_comparison.py
#
# Loads the pypty reconstruction, rebuilds the ground-truth
# projected potential from the original MD structure, aligns
# and quantitatively compares the two.
#
# Run with glass_sem_env (needs abtem):
#   conda activate glass_sem_env
#   python analysis/pypty_comparison.py
#
# Requires: outputs/diffraction/ptycho_output/{co.npy,params.pkl}
#           outputs/diffraction/scan_metadata.json
# =============================================================

import os
import pickle
import json
import numpy as np
import abtem
from scipy.signal import correlate2d
from scipy.stats import pearsonr

from config import TRAJECTORY_FILE
from data_io.trajectory import load_trajectory
from diffraction.simulate import compute_average_structure, correct_atom_types
from visualization.plot import plot_pypty_reconstruction, plot_pypty_vs_ground_truth, plot_cross_correlation
from utils import get_output_dir
OUTPUT_DIR = get_output_dir()

CHECKPOINT_DIR = "outputs/diffraction/ptycho_output"


def load_reconstruction():
    co = np.load(os.path.join(CHECKPOINT_DIR, "co.npy"))
    cp = np.load(os.path.join(CHECKPOINT_DIR, "cp.npy"))
    with open(os.path.join(CHECKPOINT_DIR, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    return co, cp, params


def build_ground_truth_potential(pixel_size_x_A, pixel_size_y_A):
    trajectory = load_trajectory(TRAJECTORY_FILE)
    avg_atoms = compute_average_structure(trajectory)
    avg_atoms = correct_atom_types(avg_atoms)

    potential = abtem.Potential(avg_atoms, sampling=(pixel_size_x_A, pixel_size_y_A))
    projected = potential.build().compute().array.sum(axis=0)
    return projected


def match_shape(ground_truth, target_shape):
    """
    Center-crop or center-pad `ground_truth` to exactly match `target_shape`.
    Used because the reconstructed object is padded beyond the raw scan
    extent (probe footprint margin at the scan edges), so its shape can't
    be derived from scan geometry alone — match it directly instead.
    """
    result = ground_truth
    for axis in range(2):
        diff = target_shape[axis] - result.shape[axis]
        if diff > 0:
            # pad
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width = [(0, 0), (0, 0)]
            pad_width[axis] = (pad_before, pad_after)
            result = np.pad(result, pad_width, mode="constant")
        elif diff < 0:
            # crop
            crop_before = (-diff) // 2
            crop_after = result.shape[axis] - ((-diff) - crop_before)
            slicer = [slice(None), slice(None)]
            slicer[axis] = slice(crop_before, crop_after)
            result = result[tuple(slicer)]
    return result


def normalize(a):
    return (a - a.mean()) / a.std()


def align_and_correlate(reconstructed, ground_truth):
    a = normalize(reconstructed)
    b = normalize(ground_truth)

    xcorr = correlate2d(a, b, mode="same")
    dy, dx = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    shift = (dy - a.shape[0] // 2, dx - a.shape[1] // 2)
    b_aligned = np.roll(b, shift, axis=(0, 1))

    r, p_value = pearsonr(a.flatten(), b_aligned.flatten())
    return a, b_aligned, r, p_value, xcorr, shift


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