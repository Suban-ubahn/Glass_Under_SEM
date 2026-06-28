import ase.io
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from ase.neighborlist import neighbor_list

from .convert_to_PyG_graph import (
    load_trajectory,
    atoms_to_pyg,
    rbf,
    type_to_Z
)


# -------------------------------
# Degree computation
# -------------------------------
def compute_degrees(graph):
    num_nodes = graph.x.shape[0]
    degrees = torch.zeros(num_nodes, dtype=torch.long)

    for idx in graph.edge_index[0]:
        degrees[idx] += 1

    return degrees


# -------------------------------
# Degree distribution by element
# -------------------------------
def plot_degree_by_element(graphs, max_frames=10):
    deg_Zr = []
    deg_Cu = []

    for g in graphs[:max_frames]:
        deg = compute_degrees(g)
        Z = g.x.view(-1)

        deg_Zr.append(deg[Z == 40].numpy())
        deg_Cu.append(deg[Z == 29].numpy())

    deg_Zr = np.concatenate(deg_Zr)
    deg_Cu = np.concatenate(deg_Cu)

    plt.figure()
    plt.hist(deg_Zr, bins=40, alpha=0.6, label="Zr")
    plt.hist(deg_Cu, bins=40, alpha=0.6, label="Cu")
    plt.xlabel("Coordination number (degree)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Degree Distribution by Element")
    plt.savefig('PyG Graph/degree_distribution_by_element.png', dpi=300)
    plt.show()

    return deg_Zr, deg_Cu


# -------------------------------
# Statistical defect definition
# -------------------------------
def classify_defects(degrees, Z, k=1.0):
    defects = torch.zeros_like(degrees, dtype=torch.bool)

    for element in [40, 29]:
        mask = (Z == element)

        deg_elem = degrees[mask].float()
        mean = deg_elem.mean()
        std = deg_elem.std()

        threshold = mean - k * std

        defects[mask] = degrees[mask] < threshold

        print(f"Element {element}: mean={mean:.2f}, std={std:.2f}, threshold={threshold:.2f} (= mean - {k} * std)")

    return defects


# -------------------------------
# Local geometry metrics
# -------------------------------
def compute_local_geometry(graph):
    num_nodes = graph.x.shape[0]

    sum_dist = torch.zeros(num_nodes)
    sum_sq_dist = torch.zeros(num_nodes)
    counts = torch.zeros(num_nodes)

    src = graph.edge_index[0]
    d = graph.distances

    for idx in range(len(src)):
        i = src[idx]
        dist = d[idx]

        sum_dist[i] += dist
        sum_sq_dist[i] += dist ** 2
        counts[i] += 1

    mean_dist = sum_dist / counts
    var_dist = sum_sq_dist / counts - mean_dist ** 2

    return mean_dist, var_dist


# -------------------------------
# Plot geometry vs degree
# -------------------------------
def plot_geometry_vs_degree(graph):
    degrees = compute_degrees(graph)
    mean_dist, var_dist = compute_local_geometry(graph)

    plt.figure()
    plt.scatter(degrees.numpy(), mean_dist.numpy(), s=2)
    plt.xlabel("Degree")
    plt.ylabel("Mean neighbor distance")
    plt.title("Degree vs Mean Distance")
    plt.savefig('PyG Graph/degree_vs_mean_distance.png', dpi=300)
    plt.show()

    plt.figure()
    plt.scatter(degrees.numpy(), var_dist.numpy(), s=2)
    plt.xlabel("Degree")
    plt.ylabel("Distance variance")
    plt.title("Degree vs Distance Variance")
    plt.savefig('PyG Graph/degree_vs_distance_variance.png', dpi=300)
    plt.show()


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    file_path = "trj.zr50cu50_6750_vib"
    cutoff = 3.5

    trajectory = load_trajectory(file_path)
    graphs = [atoms_to_pyg(atoms, cutoff) for atoms in trajectory]

    # 1) Degree distributions by element
    deg_Zr, deg_Cu = plot_degree_by_element(graphs)

    # 2) Defect classification on one frame
    g = graphs[0]
    degrees = compute_degrees(g)
    Z = g.x.view(-1)

    defects = classify_defects(degrees, Z, k=1.0)

    print("Number of defects in first frame:", defects.sum().item())

    # 3) Geometry correlations
    plot_geometry_vs_degree(g)