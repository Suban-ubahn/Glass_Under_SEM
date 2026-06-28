import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import torch

from convert_to_PyG_graph import load_trajectory, atoms_to_pyg
from PyG_defects_subgraphs import (
    compute_degree_deficit,
    classify_defects_from_deficit,
    build_defect_subgraph
)
from PyG_graph_edge_dist import compute_degrees


# -------------------------------
# Parameters
# -------------------------------
file_path = "trj.zr50cu50_6750_vib"
cutoff = 3.5
threshold = 1.0
stride = 1   # use ALL frames


# -------------------------------
# Load trajectory
# -------------------------------
trajectory = load_trajectory(file_path)[::stride]
graphs = [atoms_to_pyg(atoms, cutoff) for atoms in trajectory]

print(f"Loaded {len(graphs)} frames")


# -------------------------------
# Identify connected components
# -------------------------------
def get_connected_components(edge_index, num_nodes):
    """
    Returns list of components (each is list of node indices)
    """
    adjacency = [[] for _ in range(num_nodes)]

    if edge_index is not None and edge_index.shape[1] > 0:
        for i, j in edge_index.T:
            adjacency[i].append(j)
            adjacency[j].append(i)

    visited = set()
    components = []

    for node in range(num_nodes):
        if node in visited:
            continue

        stack = [node]
        comp = []

        while stack:
            n = stack.pop()
            if n in visited:
                continue

            visited.add(n)
            comp.append(n)

            for neigh in adjacency[n]:
                if neigh not in visited:
                    stack.append(neigh)

        components.append(comp)

    return components


# -------------------------------
# Precompute cluster data
# -------------------------------
all_cluster_data = []

for g in graphs:
    degrees = compute_degrees(g)
    Z = g.x.view(-1)

    deficit = compute_degree_deficit(degrees, Z)
    defects = classify_defects_from_deficit(
        deficit, degrees, Z, threshold=threshold, print_stats=False
    )

    subgraph, _ = build_defect_subgraph(g, defects, cutoff=cutoff)

    if subgraph is None:
        all_cluster_data.append([])
        continue

    pos = subgraph.pos.cpu().numpy()
    edge_index = subgraph.edge_index.cpu().numpy()
    num_nodes = pos.shape[0]

    components = get_connected_components(edge_index, num_nodes)

    frame_clusters = []

    for comp in components:
        comp = np.array(comp)
        cluster_pos = pos[comp]

        # edges inside this component
        mask = np.isin(edge_index[0], comp) & np.isin(edge_index[1], comp)
        edges = edge_index[:, mask]

        frame_clusters.append((cluster_pos, edges, comp))

    all_cluster_data.append(frame_clusters)


# -------------------------------
# Color mapping by cluster size
# -------------------------------
def size_to_color(size):
    """
    Map cluster size → color (same size = same color)
    """
    # normalize size (simple heuristic)
    return plt.cm.viridis(min(size / 10.0, 1.0))


# -------------------------------
# Setup figure
# -------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor('black')

# -------------------------------
# Animation update
# -------------------------------
def update(frame):
    ax.clear()

    # keep background after clearing
    ax.set_facecolor('black')

    clusters = all_cluster_data[frame]

    for cluster_pos, edges, comp in clusters:
        size = len(comp)
        # color = size_to_color(size)
        color = 'white'

        # --- plot nodes ---
        ax.scatter(
            cluster_pos[:, 0],
            cluster_pos[:, 1],
            cluster_pos[:, 2],
            s=12,
            color=color,
            alpha=0.9
        )

        # --- plot edges ---
        for i, j in edges.T:
            p1 = cluster_pos[np.where(comp == i)[0][0]]
            p2 = cluster_pos[np.where(comp == j)[0][0]]

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=color,
                linewidth=1,
                alpha=0.7
            )

    ax.set_title(f"Defect Dynamics (Frame {frame})")

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    pos_all = graphs[0].pos
    ax.set_xlim(0, pos_all[:, 0].max())
    ax.set_ylim(0, pos_all[:, 1].max())
    ax.set_zlim(0, pos_all[:, 2].max())


# -------------------------------
# Create animation
# -------------------------------
ani = FuncAnimation(
    fig,
    update,
    frames=len(graphs),
    interval=100  # faster playback
)


# -------------------------------
# Save / show
# -------------------------------
save = True

if save:
    ani.save("PyG Graph/defect_clusters_dynamics.mp4", fps=12, dpi=200)
    print("Saved animation as defect_clusters_dynamics.mp4")
else:
    plt.show()