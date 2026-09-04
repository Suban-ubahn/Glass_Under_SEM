# =============================================================
# visualization/animate.py — 3-D animations
#
# Collects animation helpers that were previously scattered
# across three files:
#   Holes_by_Grid/find_holes_animation.py   → animate_holes()
#   Holes_by_Grid/find_holes_fft_convolve.py → animate_holes_fft()
#   PyG_Graph/visualize_defects_animation.py → animate_defects()
# =============================================================


import numpy as np
from utils import get_output_dir
OUTPUT_DIR = get_output_dir()
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import torch

from holes.core import (
    compute_void_grid,
    find_holes,
    get_grid_coords,
    get_largest_holes,
)
from holes.fft import compute_Vcell, compute_Vloch, find_hole_peaks, voxel_to_position
from graph.build import atoms_to_pyg
from graph.defects import (
    compute_degrees,
    compute_degree_deficit,
    classify_defects_from_deficit,
    build_defect_subgraph,
    get_connected_components,
)
from config import VOXEL_EDGE, R_CUT_VOID, GRAPH_CUTOFF_DICT, CUTOFF_DICT, DEFECT_THRESHOLD
from dynamics.flow_units import d2min_consecutive_trajectory

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
        print(f"Updating animation frame {frame}")
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
        print(f"Processing frame {t}")
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
        print(f"Updating animation frame {frame}")
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


# ------------------------------------------------------------
# 3. Defect cluster animation  (was visualize_defects_animation.py)
# ------------------------------------------------------------

def animate_defects(trajectory, cutoff=GRAPH_CUTOFF_DICT,
                    threshold=DEFECT_THRESHOLD,
                    save_path=None):
    """
    Animate connected defect clusters. Cluster colour and edge colour
    reflect cluster size — larger clusters are brighter/warmer.
    A colorbar serves as the legend for cluster size.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "defect_clusters_dynamics.mp4")
    print("Creating graph-structure-defects animation")

    graphs = [atoms_to_pyg(atoms, cutoff) for atoms in trajectory]

    # precompute cluster data and collect all sizes for normalisation
    all_cluster_data = []
    all_sizes = []

    for g in graphs:
        degrees = compute_degrees(g)
        Z       = g.x.view(-1)
        deficit = compute_degree_deficit(degrees, Z)
        defects = classify_defects_from_deficit(
            deficit, degrees, Z, threshold=threshold, print_stats=False
        )
        subgraph, _ = build_defect_subgraph(g, defects, cutoff=cutoff)

        if subgraph is None:
            all_cluster_data.append([])
            continue

        pos        = subgraph.pos.cpu().numpy()
        edge_index = subgraph.edge_index.cpu().numpy()
        components = get_connected_components(edge_index, pos.shape[0])

        frame_clusters = []
        for comp in components:
            comp = np.array(comp)
            size = len(comp)
            all_sizes.append(size)
            mask = np.isin(edge_index[0], comp) & np.isin(edge_index[1], comp)
            frame_clusters.append((pos[comp], edge_index[:, mask], comp, size))

        all_cluster_data.append(frame_clusters)

    # colormap normalised across all frames so colours are comparable over time
    vmin = min(all_sizes) if all_sizes else 1
    vmax = max(all_sizes) if all_sizes else 2
    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Cluster size (atoms)", fontsize=8, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    pos_all = graphs[0].pos

    def update(frame):
        print(f"Updating animation frame {frame}")
        ax.clear()

        for cluster_pos, edges, comp, size in all_cluster_data[frame]:
            color = cmap(norm(size))
            ax.scatter(cluster_pos[:, 0], cluster_pos[:, 1], cluster_pos[:, 2],
                       s=12, color=color, alpha=0.9)
            for i, j in edges.T:
                p1 = cluster_pos[np.where(comp == i)[0][0]]
                p2 = cluster_pos[np.where(comp == j)[0][0]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=color, linewidth=1, alpha=0.7)

        ax.set_title(f"Defect Dynamics (Frame {frame})", color="white")
        ax.set_xlabel("X (Å)"); ax.set_ylabel("Y (Å)"); ax.set_zlabel("Z (Å)")
        ax.set_xlim(0, pos_all[:, 0].max())
        ax.set_ylim(0, pos_all[:, 1].max())
        ax.set_zlim(0, pos_all[:, 2].max())

    ani = FuncAnimation(fig, update, frames=len(graphs), interval=100)
    ani.save(save_path, fps=12, dpi=200)
    print(f"Saved: {save_path}")

def animate_d2min_2d(trajectory, cutoff=CUTOFF_DICT, save_path=None):
    """
    Animate atoms at their actual (x, y) positions each frame,
    colored by the CUMULATIVE D^2_min each atom has accumulated up
    to that frame. Since D^2_min >= 0 always, this cumulative value
    only ever grows — atoms brighten over the animation as they
    accumulate non-affine (flow-unit-like) rearrangement, rather
    than flickering between D^2_min's step-to-step value.

    The color scale (0 to the 99th percentile of the FINAL frame's
    cumulative value) is fixed for the whole animation, so growth
    is directly visible as atoms brightening — not an artifact of
    the colorbar rescaling itself frame to frame.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "d2min_dynamics_2d.mp4")
    print("Creating D^2_min 2D animation")

    d2min_series, cumulative = d2min_consecutive_trajectory(trajectory, cutoff)

    vmax = np.nanpercentile(cumulative[-1], 99)
    cmap = cm.inferno
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Cumulative D^2_min (A^2)", fontsize=9)

    cell = trajectory[0].cell.lengths()

    def update(frame):
        print(f"Updating animation frame {frame}")
        ax.clear()
        pos = trajectory[frame].positions
        ax.scatter(pos[:, 0], pos[:, 1], c=cumulative[frame],
                   cmap=cmap, norm=norm, s=14)
        ax.set_xlim(0, cell[0]); ax.set_ylim(0, cell[1])
        ax.set_xlabel("x (A)"); ax.set_ylabel("y (A)")
        ax.set_title(f"Cumulative D^2_min (Frame {frame})")
        ax.set_aspect("equal")

    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=100)
    ani.save(save_path, fps=12, dpi=150)
    print(f"Saved: {save_path}")


def animate_d2min_3d(trajectory, cutoff=CUTOFF_DICT, save_path=None,
                      elev=20, azim=45):
    """
    Same idea as animate_d2min_2d, but atoms are shown at their real
    (x, y, z) positions each frame instead of a z-integrated 2D
    projection — see plot_d2min_map_3d's docstring for why that
    matters (the 2D version can hide real 3D localization behind
    overlapping points along the viewing axis).
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "d2min_dynamics_3d.mp4")
    print("Creating D^2_min 3D animation")

    d2min_series, cumulative = d2min_consecutive_trajectory(trajectory, cutoff)

    vmax = np.nanpercentile(cumulative[-1], 99)
    cmap = cm.inferno
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label("Cumulative D^2_min (A^2)", fontsize=9)

    cell = trajectory[0].cell.lengths()

    def update(frame):
        print(f"Updating animation frame {frame}")
        ax.clear()
        pos = trajectory[frame].positions
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=cumulative[frame],
                   cmap=cmap, norm=norm, s=10, alpha=0.8)
        ax.set_xlim(0, cell[0]); ax.set_ylim(0, cell[1]); ax.set_zlim(0, cell[2])
        ax.set_xlabel("x (A)"); ax.set_ylabel("y (A)"); ax.set_zlabel("z (A)")
        ax.set_title(f"Cumulative D^2_min (Frame {frame})")
        ax.view_init(elev=elev, azim=azim)

    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=100)
    ani.save(save_path, fps=12, dpi=150)
    print(f"Saved: {save_path}")

# ------------------------------------------------------------
# Main — run all three animations
# ------------------------------------------------------------

if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import TRAJECTORY_FILE

    trajectory = load_trajectory(TRAJECTORY_FILE)

    # animate_holes_KDTree(trajectory, save_path="holes_3d.mp4")
    # animate_holes_fft(trajectory, save_path=None)
    # animate_defects(trajectory, save_path=None)

    print("Done.")