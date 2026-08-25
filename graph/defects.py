# =============================================================
# graph/defects.py — coordination-based defect detection
#
# Previously split across:
#   PyG_Graph/PyG_graph_edge_dist.py
#   PyG_Graph/PyG_defects_subgraphs.py
#
# Those two files both defined defect-classification functions
# with overlapping purposes. Unified here:
#
#   compute_degrees()               — coordination number per atom
#   compute_degree_deficit()        — continuous deficit from element mean
#   classify_defects_from_deficit() — PRIMARY classifier (absolute threshold)
#   classify_defects_statistical()  — ALTERNATIVE classifier (mean - k*std)
#   build_defect_subgraph()         — connect defect atoms into a subgraph
#   get_connected_components()      — generic DFS  (was duplicated in
#                                     visualize_defects_animation.py and
#                                     inlined in hole_defect_projection_maps.py)
#   extract_defect_neighbor_subgraphs()
#   track_defects_over_time()
# =============================================================


import numpy as np
from utils import get_output_dir
import os
import torch
from torch_geometric.data import Data

from config import CUTOFF, DEFECT_THRESHOLD


# ------------------------------------------------------------
# Coordination numbers
# ------------------------------------------------------------

def compute_degrees(graph):
    """
    Return the out-degree of every node (= number of neighbours
    within the cutoff, i.e. coordination number).

    Parameters
    ----------
    graph : PyG Data with edge_index (2, E)

    Returns
    -------
    degrees : (N,) long Tensor
    """
    num_nodes = graph.x.shape[0]
    degrees   = torch.zeros(num_nodes, dtype=torch.long)
    for idx in graph.edge_index[0]:
        degrees[idx] += 1
    return degrees


# ------------------------------------------------------------
# Defect classification — PRIMARY  (absolute deficit threshold)
# ------------------------------------------------------------

def compute_degree_deficit(degrees, Z):
    """
    For each atom, compute how many bonds it is missing relative
    to the mean coordination of its element type.

    deficit[i] = mean_Z[i] - degree[i]

    Positive → under-coordinated (potential defect).
    Negative → over-coordinated.
    """
    deficit = torch.zeros_like(degrees, dtype=torch.float)
    for element in [40, 29]:   # Zr, Cu
        mask    = (Z == element)
        mean    = degrees[mask].float().mean()
        deficit[mask] = mean - degrees[mask].float()
    return deficit


def classify_defects_from_deficit(deficit, degrees, Z,
                                  threshold=DEFECT_THRESHOLD,
                                  print_stats=False):
    """
    Mark an atom as defective if its degree deficit exceeds
    `threshold` (i.e. it is missing more than `threshold` bonds
    compared to the element mean).

    This is the PRIMARY defect classifier used throughout the
    pipeline (hole_defect_projection_maps.py, animation, etc.).

    Parameters
    ----------
    deficit   : (N,) float Tensor from compute_degree_deficit()
    threshold : float — bonds missing below which atom is a defect

    Returns
    -------
    defects : (N,) bool Tensor
    """
    defects = deficit > threshold

    if print_stats:
        for element in [40, 29]:
            mask = (Z == element)
            mean = degrees[mask].float().mean()
            std  = degrees[mask].float().std()
            print(f"Element {element}: mean={mean:.2f}, std={std:.2f}, "
                  f"degree_threshold={mean - threshold:.2f} (= mean - {threshold})")

    return defects


# ------------------------------------------------------------
# Defect classification — ALTERNATIVE  (statistical, std-based)
# ------------------------------------------------------------

def classify_defects_statistical(degrees, Z, k=1.0):
    """
    Alternative classifier: mark atoms whose coordination falls
    more than k standard deviations below the element mean.

    defect ↔ degree < mean(Z) - k * std(Z)

    Use this if you want a threshold that adapts to the spread
    of the coordination distribution rather than a fixed bond count.
    `classify_defects_from_deficit` with threshold=1.0 is usually
    preferred for this system.
    """
    defects = torch.zeros_like(degrees, dtype=torch.bool)

    for element in [40, 29]:
        mask      = (Z == element)
        deg_elem  = degrees[mask].float()
        mean, std = deg_elem.mean(), deg_elem.std()
        defects[mask] = degrees[mask] < (mean - k * std)
        print(f"Element {element}: mean={mean:.2f}, std={std:.2f}, "
              f"threshold={mean - k * std:.2f} (= mean - {k}*std)")

    return defects


# ------------------------------------------------------------
# Connected-component finder  (generic DFS)
#
# Previously duplicated:
#   - as a standalone function in visualize_defects_animation.py
#   - inlined inside project_defect_clusters() in
#     hole_defect_projection_maps.py
# Now lives here and is imported by both.
# ------------------------------------------------------------

def get_connected_components(edge_index, num_nodes):
    """
    Find connected components in an undirected graph via DFS.

    Parameters
    ----------
    edge_index : (2, E) int array or Tensor
    num_nodes  : int

    Returns
    -------
    list of lists — each inner list is the node indices of one component
    """
    adjacency = [[] for _ in range(num_nodes)]

    if edge_index is not None and edge_index.shape[1] > 0:
        for i, j in edge_index.T:
            adjacency[i].append(j)
            adjacency[j].append(i)

    visited    = set()
    components = []

    for node in range(num_nodes):
        if node in visited:
            continue
        stack, comp = [node], []
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.append(n)
            stack.extend(nb for nb in adjacency[n] if nb not in visited)
        components.append(comp)

    return components


# ------------------------------------------------------------
# Defect subgraph
# ------------------------------------------------------------

def build_defect_subgraph(graph, defects, cutoff=CUTOFF):
    """
    Build a new PyG graph containing only defect atoms, with
    edges between defects that are within `cutoff` Å of each other.

    Returns
    -------
    subgraph   : PyG Data  (or None if no defects)
    defect_idx : (M,) long Tensor — original atom indices
    """
    pos        = graph.pos
    defect_idx = torch.where(defects)[0]

    if len(defect_idx) == 0:
        return None, None

    defect_pos = pos[defect_idx]

    diff = defect_pos.unsqueeze(1) - defect_pos.unsqueeze(0)
    dist = torch.norm(diff, dim=-1)

    mask       = (dist < cutoff) & (dist > 0)
    edge_index = mask.nonzero(as_tuple=False).t()

    subgraph = Data(
        x          = graph.x[defect_idx],
        pos        = defect_pos,
        edge_index = edge_index,
    )
    return subgraph, defect_idx


# ------------------------------------------------------------
# Neighbour-environment subgraphs
# ------------------------------------------------------------

def extract_defect_neighbor_subgraphs(graph, defects):
    """
    For each defect atom, extract its local neighbourhood
    (the defect itself + all its bonded neighbours) as a small
    subgraph.

    Returns
    -------
    list of PyG Data objects — one per defect atom
    """
    edge_index     = graph.edge_index
    neighbor_graphs = []

    for i in torch.where(defects)[0]:
        neighbors = edge_index[1][edge_index[0] == i]
        nodes     = torch.cat([torch.tensor([i]), neighbors]).unique()

        sub_edge_mask  = (
            torch.isin(edge_index[0], nodes) &
            torch.isin(edge_index[1], nodes)
        )
        sub_edge_index = edge_index[:, sub_edge_mask]

        neighbor_graphs.append(Data(
            x          = graph.x[nodes],
            pos        = graph.pos[nodes],
            edge_index = sub_edge_index,
        ))

    return neighbor_graphs


# ------------------------------------------------------------
# Defect tracking over time
# ------------------------------------------------------------

def track_defects_over_time(graphs, threshold=DEFECT_THRESHOLD,
                             max_match_dist=1.5):
    """
    Greedily match defect atoms across consecutive frames by
    nearest-neighbour position matching (within max_match_dist Å).

    Returns
    -------
    tracks : dict  track_id → list of position arrays (one per frame)
    """
    tracks         = {}
    next_id        = 0
    prev_positions = None
    prev_ids       = None

    for t, g in enumerate(graphs):
        degrees = compute_degrees(g)
        Z       = g.x.view(-1)
        deficit = compute_degree_deficit(degrees, Z)
        defects = classify_defects_from_deficit(
            deficit, degrees, Z,
            threshold=threshold, print_stats=False
        )

        defect_idx = torch.where(defects)[0]
        positions  = g.pos[defect_idx]
        current_ids = []

        if t == 0:
            for p in positions:
                tracks[next_id] = [p.numpy()]
                current_ids.append(next_id)
                next_id += 1
        else:
            used = set()
            for p in positions:
                if prev_positions is None or len(prev_positions) == 0:
                    break
                dists = np.linalg.norm(prev_positions - p.numpy(), axis=1)
                idx   = np.argmin(dists)
                if dists[idx] < max_match_dist and idx not in used:
                    track_id = prev_ids[idx]
                    tracks[track_id].append(p.numpy())
                    current_ids.append(track_id)
                    used.add(idx)
                else:
                    tracks[next_id] = [p.numpy()]
                    current_ids.append(next_id)
                    next_id += 1

        prev_positions = positions.numpy()
        prev_ids       = current_ids

    return tracks


# ------------------------------------------------------------
# Degree-distribution plots
# ------------------------------------------------------------

def plot_degree_by_element(graphs, max_frames=10,
                            save_path=None):
    import matplotlib.pyplot as plt
    if save_path is None:
        save_path = os.path.join(get_output_dir(), "degree_distribution_by_element.png")
    deg_Zr, deg_Cu = [], []
    for g in graphs[:max_frames]:
        deg = compute_degrees(g)
        Z   = g.x.view(-1)
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
    plt.savefig(save_path, dpi=300)
    plt.show()

    return deg_Zr, deg_Cu


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from graph.build import atoms_to_pyg
    from config import TRAJECTORY_FILE

    trajectory = load_trajectory(TRAJECTORY_FILE)
    graphs     = [atoms_to_pyg(atoms) for atoms in trajectory[:20]]

    g       = graphs[0]
    degrees = compute_degrees(g)
    Z       = g.x.view(-1)
    deficit = compute_degree_deficit(degrees, Z)
    defects = classify_defects_from_deficit(deficit, degrees, Z, print_stats=True)

    print(f"Defects in first frame: {defects.sum().item()}")

    subgraph, _ = build_defect_subgraph(g, defects)
    print("Defect subgraph:", subgraph)

    tracks = track_defects_over_time(graphs)
    print(f"Tracked defects: {len(tracks)}")
    avg_life = sum(len(t) for t in tracks.values()) / len(tracks) if tracks else 0
    print(f"Average defect lifetime: {avg_life:.1f} frames")
