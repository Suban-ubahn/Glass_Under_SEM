# =============================================================
# visualization/animate/holes.py
# (split out of the old 1300-line visualization/animate.py;
#  `from visualization.animate import <name>` still works)
# =============================================================

from utils import get_output_dir
OUTPUT_DIR = get_output_dir()
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from holes.kdtree import (
    compute_void_grid,
    find_holes,
    get_grid_coords,
    get_largest_holes,
)
from holes.fft import compute_Vcell, compute_Vloch, find_hole_peaks, voxel_to_position
from config import VOXEL_EDGE, R_CUT_VOID


# ------------------------------------------------------------
# 1. KDTree hole animation  (was find_holes_animation.py)
# ------------------------------------------------------------

def animate_holes_KDTree(trajectory, spacing=VOXEL_EDGE, r_cut=R_CUT_VOID,
                  save_path=None):
    """
    Animate the 3 largest KDTree-detected holes over the trajectory.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "holes_KDTree.mp4")
    print("Creating KDTree-hole animation")

    X, Y, Z = get_grid_coords(trajectory[0], spacing)
    hole_colors = ["red", "blue", "green"]

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=5, label="Atoms", alpha=0.4),
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=7, label=f"Hole {i+1}")
        for i, c in enumerate(hole_colors)
    ]

    def update(frame):
        # print(f"Updating animation frame {frame}")
        ax.clear()
        atoms = trajectory[frame]

        void_grid, _    = compute_void_grid(atoms, r_cut, spacing)
        labeled, n_feat = find_holes(void_grid)

        pos = atoms.positions
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   s=1, alpha=0.05, color="gray")

        for idx, (xs, ys, zs) in enumerate(get_largest_holes(labeled, n_feat, X, Y, Z)):
            ax.scatter(xs, ys, zs, s=5, color=hole_colors[idx % len(hole_colors)])

        cell = atoms.cell.lengths()
        ax.set_xlim(0, cell[0]); ax.set_ylim(0, cell[1]); ax.set_zlim(0, cell[2])
        ax.set_xlabel("X (Å)"); ax.set_ylabel("Y (Å)"); ax.set_zlabel("Z (Å)")
        ax.view_init(elev=30, azim=45)
        ax.set_title(f"Frame {frame}")
        ax.legend(handles=legend_elements, loc="upper left", fontsize=7)
        return []

    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=200)
    ani.save(save_path, fps=12, dpi=200)
    print(f"Saved: {save_path}")
    plt.show()


# ------------------------------------------------------------
# 2. FFT hole animation  (was animate_holes in fft_convolve.py)
# ------------------------------------------------------------

def animate_holes_fft(trajectory, voxel_edge=VOXEL_EDGE,
                       save_path=None):
    """
    Animate FFT-detected hole peaks over the trajectory.
    Hole colour and size reflect hole strength (Vloch value at peak).
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "hole_dynamics_fft.mp4")
    print("Creating fft-convolve hole animation")

    all_data = []
    all_strengths = []
    for t, atoms in enumerate(trajectory):
        # print(f"Processing frame {t}")
        Vcell            = compute_Vcell(atoms, voxel_edge)
        Vloch            = compute_Vloch(Vcell, voxel_edge)
        peak_idx, peak_v = find_hole_peaks(Vloch)
        hole_pos         = voxel_to_position(peak_idx, voxel_edge)
        all_data.append((atoms.positions.copy(), hole_pos, peak_v))
        if len(peak_v) > 0:
            all_strengths.extend(peak_v)

    vmin = min(all_strengths) if all_strengths else 0
    vmax = max(all_strengths) if all_strengths else 1
    cmap = cm.inferno
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    sm  = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Hole strength (Vloch)", fontsize=8)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=5, label="Atoms", alpha=0.4),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(0.8),
               markersize=7, label="Hole peaks"),
    ]

    cell = trajectory[0].cell.lengths()

    def update(frame):
        # print(f"Updating animation frame {frame}")
        ax.clear()
        atom_pos, hole_pos, hole_strength = all_data[frame]

        ax.scatter(atom_pos[:, 0], atom_pos[:, 1], atom_pos[:, 2],
                   s=1, color="gray", alpha=0.05)

        if len(hole_pos) > 0:
            sizes = 20 + 80 * (hole_strength - vmin) / (vmax - vmin + 1e-8)
            ax.scatter(hole_pos[:, 0], hole_pos[:, 1], hole_pos[:, 2],
                       c=hole_strength, s=sizes, cmap=cmap,
                       vmin=vmin, vmax=vmax, alpha=0.9)

        ax.set_xlim(0, cell[0]); ax.set_ylim(0, cell[1]); ax.set_zlim(0, cell[2])
        ax.set_xlabel("X (Å)"); ax.set_ylabel("Y (Å)"); ax.set_zlabel("Z (Å)")
        ax.set_title(f"Hole Dynamics — Frame {frame}")
        ax.legend(handles=legend_elements, loc="upper left", fontsize=7)

    ani = FuncAnimation(fig, update, frames=len(all_data), interval=100)
    ani.save(save_path, fps=12, dpi=200)
    print(f"Saved: {save_path}")
