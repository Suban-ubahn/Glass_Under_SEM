import numpy as np
import matplotlib.pyplot as plt
import torch

# ==========================================
# HOLES MODULE
# ==========================================

from Holes_by_Grid.find_holes_fft_convolve import (
    compute_Vcell,
    compute_Vloch
)

# ==========================================
# GRAPH MODULES
# ==========================================

from PyG_Graph.convert_to_PyG_graph import (
    load_trajectory,
    atoms_to_pyg
)

from PyG_Graph.PyG_graph_edge_dist import (
    compute_degrees
)

from PyG_Graph.PyG_defects_subgraphs import (
    compute_degree_deficit,
    classify_defects_from_deficit,
    build_defect_subgraph
)

# ============================================================
# PARAMETERS
# ============================================================

file_path = "trj.zr50cu50_6750_vib"

voxel_edge = 0.3

cutoff = 3.5
threshold = 1.0

cluster_projection_sigma = 1.0


# ============================================================
# PROJECT HOLE FIELD
# ============================================================

def project_holes(Vloch):

    return np.sum(Vloch, axis=2)


# ============================================================
# PROJECT DEFECT CLUSTERS
# ============================================================

def project_defect_clusters(
    graph,
    defects,
    voxel_edge_xy,
    shape_xy,
    cutoff=3.5
):

    cluster_map = np.zeros(shape_xy, dtype=np.float32)

    subgraph, _ = build_defect_subgraph(
        graph,
        defects,
        cutoff=cutoff
    )

    if subgraph is None:
        return cluster_map

    pos = subgraph.pos.cpu().numpy()

    edge_index = subgraph.edge_index.cpu().numpy()

    num_nodes = len(pos)

    adjacency = [[] for _ in range(num_nodes)]

    for i, j in edge_index.T:
        adjacency[i].append(j)
        adjacency[j].append(i)

    visited = set()

    for start in range(num_nodes):

        if start in visited:
            continue

        stack = [start]
        component = []

        while stack:

            n = stack.pop()

            if n in visited:
                continue

            visited.add(n)

            component.append(n)

            for neigh in adjacency[n]:
                if neigh not in visited:
                    stack.append(neigh)

        cluster_size = len(component)

        for idx in component:

            x, y, _ = pos[idx]

            ix = int(x / voxel_edge_xy)
            iy = int(y / voxel_edge_xy)

            if (
                0 <= ix < shape_xy[0]
                and
                0 <= iy < shape_xy[1]
            ):
                cluster_map[ix, iy] += cluster_size

    return cluster_map


# ============================================================
# BUILD CUMULATIVE MAPS
# ============================================================

def build_cumulative_maps(trajectory):

    print("Preparing first frame...")

    first_atoms = trajectory[0]

    Vcell0 = compute_Vcell(
        first_atoms,
        voxel_edge
    )

    hole_projection_shape = Vcell0.shape[:2]

    cumulative_holes = np.zeros(
        hole_projection_shape,
        dtype=np.float64
    )

    cumulative_clusters = np.zeros(
        hole_projection_shape,
        dtype=np.float64
    )

    for frame_idx, atoms in enumerate(trajectory):

        print(
            f"Processing frame "
            f"{frame_idx+1}/{len(trajectory)}"
        )

        # ----------------------------------
        # HOLES
        # ----------------------------------

        Vcell = compute_Vcell(
            atoms,
            voxel_edge
        )

        Vloch = compute_Vloch(
            Vcell,
            voxel_edge
        )

        hole_projection = project_holes(Vloch)

        cumulative_holes += hole_projection

        # ----------------------------------
        # DEFECT CLUSTERS
        # ----------------------------------

        graph = atoms_to_pyg(
            atoms,
            cutoff
        )

        degrees = compute_degrees(graph)

        Z = graph.x.view(-1)

        deficit = compute_degree_deficit(
            degrees,
            Z
        )

        defects = classify_defects_from_deficit(
            deficit,
            degrees,
            Z,
            threshold=threshold,
            print_stats=False
        )

        cluster_projection = project_defect_clusters(
            graph,
            defects,
            voxel_edge_xy=voxel_edge,
            shape_xy=hole_projection_shape,
            cutoff=cutoff
        )

        cumulative_clusters += cluster_projection

    return (
        cumulative_holes,
        cumulative_clusters
    )


# ============================================================
# PLOT MAP
# ============================================================

def plot_map(
    image,
    title,
    cmap="inferno"
):

    plt.figure(figsize=(8, 6))

    plt.imshow(
        image.T,
        origin="lower",
        cmap=cmap
    )

    plt.colorbar()
    plt.title(title)
    plt.xlabel("X voxel")
    plt.ylabel("Y voxel")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=300)

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading trajectory...")

    trajectory = load_trajectory(file_path)

    print(f"Loaded {len(trajectory)} frames")

    (cumulative_holes, cumulative_clusters) = build_cumulative_maps(trajectory)

    np.save("cumulative_hole_map.npy", cumulative_holes)

    np.save("cumulative_cluster_map.npy", cumulative_clusters)

    plot_map(cumulative_holes, "Time-Integrated Hole Activity")

    plot_map(cumulative_clusters, "Time-Integrated Defect Cluster Activity")