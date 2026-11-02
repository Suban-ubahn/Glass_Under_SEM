# =============================================================
# visualization/animate/defects.py
# (split out of the old 1300-line visualization/animate.py;
#  `from visualization.animate import <name>` still works)
# =============================================================

import numpy as np
from utils import get_output_dir
OUTPUT_DIR = get_output_dir()
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from graph.build import atoms_to_pyg
from graph.defects import (
    compute_degrees,
    classify_defects_statistical,
    build_defect_subgraph,
    get_connected_components,
)
from config import DEFECT_THRESHOLD, DEFECT_K
from cutoffs import CUTOFF_DICT


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
