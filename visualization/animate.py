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

from holes.kdtree import compute_void_grid,find_holes,get_grid_coords,get_largest_holes
from holes.fft import compute_Vcell, compute_Vloch, find_hole_peaks, voxel_to_position
from graph.build import atoms_to_pyg
from graph.defects import (
    compute_degrees,
    compute_degree_deficit,
    classify_defects_from_deficit,
    classify_defects_statistical,
    build_defect_subgraph,
    get_connected_components,
)

from config import (
    VOXEL_EDGE,
    R_CUT_VOID,
    DEFECT_THRESHOLD,
    DEFECT_K,
    N_SOFT_MODES,
    SOFT_MODE_PR_MAX,
)
from cutoffs import CUTOFF_DICT
from analysis.d2min.compute import d2min_consecutive_trajectory, d2min_trajectory
from analysis.soft_modes.modes import softness_trajectory

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


# ------------------------------------------------------------
# 3. Defect cluster animation  (was visualize_defects_animation.py)
# ------------------------------------------------------------

def animate_defects(trajectory, cutoff=CUTOFF_DICT,
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
        defects = classify_defects_statistical(degrees, Z, k=DEFECT_K)
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
        # print(f"Updating animation frame {frame}")
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
    colored by D^2_min relative to frame 0.

    For every frame t, D^2_min is calculated between frame 0
    and frame t.

    The color scale is fixed using the 99th percentile of the
    final-frame D^2_min values, so the final animation frame
    corresponds directly to the static d2min_map.png.
    """
    if save_path is None:
        save_path = os.path.join(
            OUTPUT_DIR,
            "d2min_dynamics_2d.mp4"
        )

    print("Creating D^2_min 2D animation")

    # Calculate D^2_min relative to frame 0 for every frame.
    d2min_series, frame_indices = d2min_trajectory(
        trajectory,
        t0=0,
        frame_step=1,
        cutoff=cutoff,
    )

    # Fixed colour scale based on the final frame.
    vmax = np.nanpercentile(d2min_series[-1], 99)

    cmap = cm.inferno
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 6))

    sm = cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax
    )
    cbar.set_label(
        "D²_min (A²)",
        fontsize=9
    )

    cell = trajectory[0].cell.lengths()

    def update(i):
        frame = frame_indices[i]
        # print(f"Updating animation frame {frame}")
        
        ax.clear()
        pos = trajectory[frame].positions
        d2min = d2min_series[i]

        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=d2min,
            cmap=cmap,
            norm=norm,
            s=14
        )
        ax.set_xlim(0, cell[0])
        ax.set_ylim(0, cell[1])
        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")

        ax.set_title(
            f"D²_min relative to frame 0 (Frame {frame})"
        )
        ax.set_aspect("equal")

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=100
    )
    ani.save(
        save_path,
        fps=12,
        dpi=150
    )
    print(f"Saved: {save_path}")

    plt.close(fig)

def animate_d2min_2d_cumulative(trajectory, cutoff=CUTOFF_DICT, save_path=None):
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
        # print(f"Updating animation frame {frame}")
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

def animate_d2min_3d(
    trajectory,
    cutoff=CUTOFF_DICT,
    save_path=None,
    elev=20,
    azim=45
):
    """
    Animate atoms in 3D at their real (x, y, z) positions,
    colored by D^2_min relative to frame 0.

    For every frame t, D^2_min is calculated between frame 0
    and frame t.

    The color scale is fixed using the 99th percentile of the
    final-frame D^2_min values.
    """
    if save_path is None:
        save_path = os.path.join(
            OUTPUT_DIR,
            "d2min_dynamics_3d.mp4"
        )

    print("Creating D^2_min 3D animation")

    # Calculate D^2_min relative to frame 0 for every frame.
    d2min_series, frame_indices = d2min_trajectory(
        trajectory,
        t0=0,
        frame_step=1,
        cutoff=cutoff,
    )

    # Fixed colour scale based on the final frame.
    vmax = np.nanpercentile(d2min_series[-1], 99)

    cmap = cm.inferno
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(
        111,
        projection="3d"
    )
    sm = cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        shrink=0.7,
        pad=0.1
    )
    cbar.set_label(
        "D²_min (A²)",
        fontsize=9
    )
    cell = trajectory[0].cell.lengths()

    def update(i):
        frame = frame_indices[i]

        # print(f"Updating animation frame {frame}")

        ax.clear()

        pos = trajectory[frame].positions
        d2min = d2min_series[i]

        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=d2min,
            cmap=cmap,
            norm=norm,
            s=10,
            alpha=0.8
        )
        ax.set_xlim(0, cell[0])
        ax.set_ylim(0, cell[1])
        ax.set_zlim(0, cell[2])

        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")
        ax.set_zlabel("z (A)")

        ax.set_title(
            f"D²_min relative to frame 0 (Frame {frame})"
        )
        ax.view_init(
            elev=elev,
            azim=azim
        )
    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=100
    )
    ani.save(
        save_path,
        fps=12,
        dpi=150
    )
    print(f"Saved: {save_path}")

    plt.close(fig)

def animate_softness_3d(
    trajectory,
    t0=0,
    frame_step=1,
    mass_weight=True,
    n_soft_modes=N_SOFT_MODES,
    pr_max=SOFT_MODE_PR_MAX,
    min_frames=5,
    save_path=None,
    elev=20,
    azim=45
):
    """
    Softness-score analogue of animate_d2min_3d() -- same animation
    structure, same frame_indices convention, same fixed-vmax-from-
    the-final-frame colorbar, so the two can be watched side by side
    (or dropped into the same figure later) as a direct visual
    comparison.

    Animate atoms in 3D at their real (x, y, z) positions, colored by
    a softness score computed from frames [0, t] relative to frame 0
    -- for every frame t, exactly the same "relative to frame 0,
    growing window" structure animate_d2min_3d() uses for D^2_min,
    via analysis.soft_modes.softness_trajectory() (see that function's
    docstring for the important caveat this reference-frame choice
    brings: this is NOT the same "vibrational softness around
    equilibrium" quantity soft_mode_analysis_3d() computes by default
    -- it's the frame-0-referenced version, built specifically to be
    comparable to D^2_min frame-for-frame).

    The first few frames (fewer than min_frames available) don't carry
    enough data for a meaningful covariance yet and show up blank/gray
    (NaN) -- the softness equivalent of D^2_min being identically zero
    at t0 itself.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "softness_dynamics_3d.mp4")

    print("Creating softness 3D animation")

    softness_series, _, frame_indices = softness_trajectory(
        trajectory, t0=t0, frame_step=frame_step, mass_weight=mass_weight,
        n_soft_modes=n_soft_modes, pr_max=pr_max, min_frames=min_frames,
    )

    # Fixed colour scale based on the final frame (same convention as
    # animate_d2min_3d) -- ignoring the early, still-NaN frames.
    vmax = np.nanpercentile(softness_series[-1], 99)

    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label("Softness score (a.u.)", fontsize=9)
    cell = trajectory[0].cell.lengths()

    def update(i):
        frame = frame_indices[i]
        ax.clear()

        pos = trajectory[frame].positions
        softness = softness_series[i]

        ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=softness, cmap=cmap, norm=norm, s=10, alpha=0.8,
        )
        ax.set_xlim(0, cell[0]); ax.set_ylim(0, cell[1]); ax.set_zlim(0, cell[2])
        ax.set_xlabel("x (A)"); ax.set_ylabel("y (A)"); ax.set_zlabel("z (A)")
        ax.set_title(f"Softness relative to frame 0 (Frame {frame})")
        ax.view_init(elev=elev, azim=azim)

    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=100)
    ani.save(save_path, fps=12, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)

def _draw_distribution_panel(ax, frame_vals, elements, unique_elements, colors,
                             xmin, xmax, bins, xlabel, title, average_pooled=None):
    """
    One distribution panel:
      - filled histogram per element (this frame)
      - black step histogram of ALL atoms combined (this frame)
      - dashed step outline per element = whole-trajectory average
        distribution (average_pooled, identical in every frame)
    """
    edges = np.linspace(xmin, xmax, bins + 1)
    frame_vals = np.asarray(frame_vals, dtype=float)
    finite = np.isfinite(frame_vals)

    if finite.any():
        for color, el in zip(colors, unique_elements):
            v = frame_vals[(elements == el) & finite]
            ax.hist(v, bins=edges, density=True, alpha=0.45, color=color, label=el)
        ax.hist(frame_vals[finite], bins=edges, density=True, histtype="step",
                color="black", linewidth=2, label="Combined")
    else:
        ax.text(0.5, 0.5, "not enough data yet", transform=ax.transAxes,
                ha="center", va="center")

    if average_pooled is not None:
        for color, el in zip(colors, unique_elements):
            ax.hist(average_pooled[el], bins=edges, density=True, histtype="step",
                    color=color, linestyle="--", linewidth=1.8,
                    label=f"{el} (trajectory avg)")

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    if finite.any():
        ax.legend(fontsize=8)


def _distribution_limits(series, pooled_combined):
    """x-range of the distribution panels: identical to plot_distribution_by_element
    (minimum to the 99.5th percentile of the pooled whole-trajectory values)."""
    return float(np.min(pooled_combined)), float(np.percentile(pooled_combined, 99.5))


def animate_distribution_by_element(series, frame_indices, elements, xlabel,
                                    title_prefix, bins=40, save_path=None):
    """
    Distribution of a per-atom field vs. frame: per-element (filled) and
    combined (black) for the current frame, plus the stationary dashed
    whole-trajectory average per element.

    series : (T, N); frame_indices : (T,); elements : (N,) symbols.
    """
    from analysis.soft_modes.element_stats import pool_series_by_element
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "distribution_by_element.mp4")

    elements = np.asarray(elements)
    unique_elements = [str(e) for e in np.unique(elements)]
    colors = plt.cm.tab10(np.arange(len(unique_elements)))
    pooled, combined = pool_series_by_element(series, elements)
    xmin, xmax = _distribution_limits(series, combined)

    fig, ax = plt.subplots(figsize=(8, 5))

    def update(i):
        ax.clear()
        _draw_distribution_panel(
            ax, series[i], elements, unique_elements, colors, xmin, xmax, bins,
            xlabel, f"{title_prefix} -- frame {frame_indices[i]}", average_pooled=pooled)
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    anim.save(save_path, writer="ffmpeg", fps=12, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


def animate_d2min_softness_side_by_side(
    trajectory, cutoff=CUTOFF_DICT, t0=0, frame_step=1, mass_weight=True,
    n_soft_modes=None, pr_max=None, elements=None,
    d2min_series=None, softness_series=None, bins=40, save_path=None,
    elev=20, azim=45, show_distributions=True, fps=12,
):
    """
    Side-by-side animation. Left: D^2_min, right: softness (both relative
    to the MEAN STRUCTURE of the trajectory), 3D atoms coloured by the field.

    show_distributions=True : 2x2 layout, distribution panels (per element,
        combined, stationary whole-trajectory averages) above the 3D maps.
    show_distributions=False: 1x2 layout, only the two 3D maps.

    Pass precomputed d2min_series / softness_series (T, N) to avoid
    recomputing them.
    """
    from analysis.soft_modes.element_stats import pool_series_by_element
    from analysis.soft_modes.modes import softness_trajectory
    if save_path is None:
        save_path = os.path.join(
            OUTPUT_DIR, "d2min_softness_side_by_side_3d.mp4" if show_distributions
            else "d2min_softness_maps_side_by_side_3d.mp4")

    if d2min_series is None:
        from utils import unwrap_positions, compute_displacements
        _pos = np.array([a.positions for a in trajectory])
        _box = np.array([a.cell.array.diagonal() for a in trajectory])
        _, mean_positions = compute_displacements(unwrap_positions(_pos, _box))
        d2min_series, frame_indices = d2min_trajectory(
            trajectory, t0=t0, frame_step=frame_step, cutoff=cutoff,
            reference_positions=mean_positions)
    else:
        frame_indices = np.arange(t0, t0 + len(d2min_series) * frame_step, frame_step)
    if softness_series is None:
        softness_series, _, _ = softness_trajectory(
            trajectory, t0=t0, frame_step=frame_step, mass_weight=mass_weight,
            n_soft_modes=n_soft_modes, pr_max=pr_max)

    n_atoms = len(trajectory[0])
    if elements is None:
        elements = np.array(["all"] * n_atoms)
    elements = np.asarray(elements)
    unique_elements = [str(e) for e in np.unique(elements)]
    colors = plt.cm.tab10(np.arange(len(unique_elements)))

    fields = [("D\u00b2_min (\u00c5\u00b2)", d2min_series, cm.inferno),
              ("Softness score (a.u.)", softness_series, cm.viridis)]
    cell = trajectory[t0].cell.lengths()

    # The two 3D maps are built IDENTICALLY in both modes (same figure size,
    # layout, colour scale, marker size, titles, colourbar). With distributions
    # the figure is two sub-figures: distributions on top, and below them
    # exactly the 16 x 7 in two-map figure of the maps-only animation.
    if show_distributions:
        fig = plt.figure(figsize=(16, 12))
        sub_top, sub_bottom = fig.subfigures(2, 1, height_ratios=[5, 7])
        ax_dist = list(sub_top.subplots(1, 2))
    else:
        fig = plt.figure(figsize=(16, 7))
        sub_bottom = fig
        ax_dist = [None, None]
    ax_3d = [sub_bottom.add_subplot(1, 2, 1, projection="3d"),
             sub_bottom.add_subplot(1, 2, 2, projection="3d")]

    info = []
    for (label, series, cmap), a3 in zip(fields, ax_3d):
        pooled, combined = pool_series_by_element(series, elements)
        xmin, xmax = _distribution_limits(series, combined)
        norm = mcolors.Normalize(vmin=0, vmax=np.nanpercentile(series[-1], 99))
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = sub_bottom.colorbar(sm, ax=a3, shrink=0.7, pad=0.1)
        cb.set_label(label, fontsize=9)
        info.append((pooled, xmin, xmax, norm, cmap))

    def update(i):
        for k, ((label, series, cmap), ad, a3) in enumerate(zip(fields, ax_dist, ax_3d)):
            pooled, xmin, xmax, norm, cm_ = info[k]
            name = label.split(" (")[0]
            if ad is not None:
                ad.clear()
                _draw_distribution_panel(
                    ad, series[i], elements, unique_elements, colors, xmin, xmax, bins,
                    label, f"{name} distribution (Frame {frame_indices[i]})",
                    average_pooled=pooled)
            a3.clear()
            vals = np.asarray(series[i], dtype=float)
            ok = np.isfinite(vals)
            cur = trajectory[frame_indices[i]].positions     # atoms at their CURRENT positions
            a3.scatter(cur[ok, 0], cur[ok, 1], cur[ok, 2], c=vals[ok], cmap=cm_,
                       norm=norm, s=10, alpha=0.8)
            a3.set_title(f"{name} relative to mean structure (Frame {frame_indices[i]})")
            a3.set_xlabel("x (A)"); a3.set_ylabel("y (A)"); a3.set_zlabel("z (A)")
            a3.set_xlim(0, cell[0]); a3.set_ylim(0, cell[1]); a3.set_zlim(0, cell[2])
            a3.view_init(elev=elev, azim=azim)
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=100, blit=False)
    anim.save(save_path, writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def animate_d2min_3d_cumulative(trajectory, cutoff=CUTOFF_DICT, save_path=None,
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
        # print(f"Updating animation frame {frame}")
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

def animate_d2min_defects_overlay_3d(
    trajectory,
    cutoff=CUTOFF_DICT,
    graph_cutoff=CUTOFF_DICT,
    k=DEFECT_K,
    save_path=None,
    elev=20,
    azim=45,
):
    """
    3D animation overlaying D^2_min relative to frame 0 with defect
    clusters and their bonds.

    Non-defect atoms are shown very faintly in the background so that
    defect clusters remain visually prominent.

    D^2_min:
        - fixed reference frame = frame 0
        - color scale fixed using the 99th percentile of the final frame
        - shown with the Blues colormap

    Defect clusters:
        - re-detected independently in every frame
        - cluster atoms and bonds shown on top
        - color represents cluster size
        - fixed color scale across the whole trajectory
        - shown with the Reds colormap

    `cutoff` is used for the D^2_min affine fit.
    `graph_cutoff` is used for building the defect graph.
    """

    if save_path is None:
        save_path = os.path.join(
            OUTPUT_DIR,
            "d2min_defects_overlay_3d.mp4"
        )

    print("Creating D^2_min / defects overlay animation")

    # ------------------------------------------------------------
    # D^2_min relative to frame 0
    # ------------------------------------------------------------

    d2min_series, frame_indices = d2min_trajectory(trajectory, t0=0, frame_step=1,  cutoff=cutoff)

    # ------------------------------------------------------------
    # Precompute defect clusters for every frame
    # ------------------------------------------------------------

    all_cluster_data = []
    all_sizes = []

    for frame in frame_indices:
        atoms = trajectory[frame]

        # Build graph with the graph-specific cutoff
        g = atoms_to_pyg(atoms, graph_cutoff)
        degrees = compute_degrees(g)
        Z = g.x.view(-1)

        # Statistical defect classification
        defects = classify_defects_statistical(degrees, Z, k=k)

        # Build graph containing only defect atoms
        subgraph, _ = build_defect_subgraph(g, defects, cutoff=graph_cutoff)

        if subgraph is None:
            all_cluster_data.append([])
            continue

        pos = subgraph.pos.cpu().numpy()
        edge_index = subgraph.edge_index.cpu().numpy()
        components = get_connected_components(edge_index, pos.shape[0])

        frame_clusters = []

        for comp in components:
            comp = np.array(comp)

            size = len(comp)
            all_sizes.append(size)
            mask_edges = (np.isin(edge_index[0], comp) & np.isin(edge_index[1], comp))

            frame_clusters.append(
                (
                    pos[comp],
                    edge_index[:, mask_edges],
                    comp,
                    size,
                )
            )

        all_cluster_data.append(frame_clusters)

    # ------------------------------------------------------------
    # Colour scales
    # ------------------------------------------------------------

    # D^2_min scale: based on final frame
    vmax_d2min = np.nanpercentile(d2min_series[-1], 99)
    cmap_d2min = cm.Blues
    norm_d2min = mcolors.Normalize(vmin=0, vmax=vmax_d2min)

    # Cluster-size scale: fixed over complete trajectory
    size_vmin = min(all_sizes) if all_sizes else 1
    size_vmax = max(all_sizes) if all_sizes else 2

    cmap_cluster = cm.Reds
    norm_cluster = mcolors.Normalize(vmin=size_vmin, vmax=size_vmax)

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot( 111, projection="3d")

    # D^2_min colorbar
    sm_d2min = cm.ScalarMappable( cmap=cmap_d2min, norm=norm_d2min)
    sm_d2min.set_array([])

    cbar1 = fig.colorbar( sm_d2min, ax=ax, shrink=0.6, pad=0.05)

    cbar1.set_label("D²_min (A²)", fontsize=9)

    # Defect cluster size colorbar
    sm_cluster = cm.ScalarMappable(cmap=cmap_cluster, norm=norm_cluster)
    sm_cluster.set_array([])

    cbar2 = fig.colorbar(sm_cluster, ax=ax, shrink=0.6, pad=0.12)
    cbar2.set_label("Defect cluster size (atoms)", fontsize=9)

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="none",
            markersize=5,
            alpha=0.25,
            label="Non-defect atoms",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap_cluster(0.8),
            markeredgecolor="black",
            markersize=8,
            label="Defect atoms",
        ),

        Line2D(
            [0],
            [0],
            color=cmap_cluster(0.8),
            linewidth=1.5,
            label="Defect cluster bonds",
        ),
    ]

    # ------------------------------------------------------------
    # Fixed simulation-cell limits
    # ------------------------------------------------------------

    cell = trajectory[0].cell.lengths()

    # ------------------------------------------------------------
    # Animation update
    # ------------------------------------------------------------

    def update(i):
        frame = frame_indices[i]

        print(f"Updating animation frame {frame}")

        ax.clear()

        atoms = trajectory[frame]
        pos = atoms.positions
        d2min = d2min_series[i]

        # --------------------------------------------------------
        # Background: all atoms, extremely faint
        # --------------------------------------------------------

        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=d2min,
            cmap=cmap_d2min,
            norm=norm_d2min,
            s=20,
            alpha=0.1,
            edgecolors="none",
        )

        # --------------------------------------------------------
        # Defect clusters: draw prominently on top
        # --------------------------------------------------------

        for cluster_pos, edges, comp, size in all_cluster_data[i]:

            color = cmap_cluster(norm_cluster(size))

            # Cluster atoms
            ax.scatter(
                cluster_pos[:, 0],
                cluster_pos[:, 1],
                cluster_pos[:, 2],
                s=35,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.95,
            )

            # Cluster bonds
            for a, b in edges.T:
                p1 = cluster_pos[np.where(comp == a)[0][0]]
                p2 = cluster_pos[np.where(comp == b)[0][0]]

                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color,
                    linewidth=2,
                    alpha=1,
                )

        # --------------------------------------------------------
        # Axes
        # --------------------------------------------------------

        ax.set_xlim(0, cell[0])
        ax.set_ylim(0, cell[1])
        ax.set_zlim(0, cell[2])

        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")
        ax.set_zlabel("z (A)")

        ax.set_title(f"D²_min & defect clusters relative to frame 0 " f"(Frame {frame})")

        ax.view_init(elev=elev, azim=azim)

        ax.legend(handles=legend_elements, loc="upper left", fontsize=7)

    # ------------------------------------------------------------
    # Create and save animation
    # ------------------------------------------------------------

    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=100)
    ani.save(save_path, fps=12, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)

def animate_d2min_holes_overlay_3d(
    trajectory,
    cutoff=CUTOFF_DICT,
    voxel_edge=VOXEL_EDGE,
    save_path=None,
    elev=20,
    azim=45,
):
    """
    3D animation overlaying D^2_min (relative to frame 0) with FFT
    hole peaks re-detected independently on every frame.

    All atoms are shown very faintly and colored by D^2_min.

    Hole peaks are shown as spheres whose size represents hole
    strength (Vloch). Hole color is kept fixed so that size, rather
    than color, carries the hole-size information.
    """
    if save_path is None:
        save_path = os.path.join(
            OUTPUT_DIR,
            "d2min_holes_overlay_3d.mp4"
        )

    print("Creating D^2_min / holes overlay animation")

    # ------------------------------------------------------------
    # D^2_min relative to frame 0
    # ------------------------------------------------------------

    d2min_series, frame_indices = d2min_trajectory(trajectory, t0=0, frame_step=1, cutoff=cutoff,)

    # ------------------------------------------------------------
    # Re-detect holes independently for every frame
    # ------------------------------------------------------------

    all_hole_data = []
    all_strengths = []

    for frame in frame_indices:
        atoms = trajectory[frame]

        Vcell = compute_Vcell(atoms, voxel_edge)
        Vloch = compute_Vloch(Vcell, voxel_edge)

        peak_idx, peak_v = find_hole_peaks(Vloch)
        hole_pos = voxel_to_position(peak_idx, voxel_edge)

        all_hole_data.append((hole_pos, peak_v))

        if len(peak_v) > 0:
            all_strengths.extend(peak_v)

    # ------------------------------------------------------------
    # Colour scales
    # ------------------------------------------------------------

    vmax_d2min = np.nanpercentile(
        d2min_series[-1],
        99
    )

    cmap_d2min = cm.inferno
    norm_d2min = mcolors.Normalize(vmin=0, vmax=vmax_d2min)

    hole_vmin = min(all_strengths) if all_strengths else 0
    hole_vmax = max(all_strengths) if all_strengths else 1

    # Hole colour is now fixed; strength is represented by size.
    hole_color = cm.viridis(0.8)

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # D^2_min colourbar
    sm_d2min = cm.ScalarMappable(cmap=cmap_d2min, norm=norm_d2min)
    sm_d2min.set_array([])

    cbar = fig.colorbar(
        sm_d2min,
        ax=ax,
        shrink=0.7,
        pad=0.1
    )
    cbar.set_label("D²_min (A²)", fontsize=9)

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="none",
            markersize=5,
            alpha=0.6,
            label="Non-hole atoms (by D²_min)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=hole_color,
            markeredgecolor="black",
            markersize=9,
            label="Hole peaks (size = hole strength)",
        ),
    ]
    cell = trajectory[0].cell.lengths()

    # ------------------------------------------------------------
    # Animation update
    # ------------------------------------------------------------

    def update(i):

        frame = frame_indices[i]
        ax.clear()

        pos = trajectory[frame].positions
        d2min = d2min_series[i]
        hole_pos, peak_v = all_hole_data[i]

        # --------------------------------------------------------
        # Background atoms: very faint
        # --------------------------------------------------------

        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=d2min,
            cmap=cmap_d2min,
            norm=norm_d2min,
            s=8,
            alpha=0.7,
            edgecolors="none",
        )

        # --------------------------------------------------------
        # Hole peaks: spheres with size proportional to strength
        # --------------------------------------------------------

        if len(hole_pos) > 0:

            if hole_vmax > hole_vmin:
                normalized_strength = (
                    (peak_v - hole_vmin)
                    / (hole_vmax - hole_vmin)
                )
            else:
                normalized_strength = np.ones_like(
                    peak_v,
                    dtype=float
                )

            # Marker area scales with hole strength.
            sizes = 25 + 175 * normalized_strength

            ax.scatter(
                hole_pos[:, 0],
                hole_pos[:, 1],
                hole_pos[:, 2],
                s=sizes,
                color=hole_color,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.9,
                marker="o",
            )

        # --------------------------------------------------------
        # Axes
        # --------------------------------------------------------

        ax.set_xlim(0, cell[0])
        ax.set_ylim(0, cell[1])
        ax.set_zlim(0, cell[2])

        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")
        ax.set_zlabel("z (A)")

        ax.set_title(f"D²_min & holes relative to frame 0 " f"(Frame {frame})")

        ax.view_init(elev=elev, azim=azim)
        ax.legend(handles=legend_elements, loc="upper left", fontsize=7)

    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=100)
    ani.save(save_path, fps=12, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)
