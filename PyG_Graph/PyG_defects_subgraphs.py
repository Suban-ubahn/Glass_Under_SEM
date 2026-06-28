import torch
import numpy as np

from torch_geometric.data import Data
from .convert_to_PyG_graph import (
    load_trajectory,
    atoms_to_pyg,
)
from .PyG_graph_edge_dist import (
    compute_degrees,
    classify_defects,
)


# -------------------------------
# A more 'continuous' deficit classification
# -------------------------------
def compute_degree_deficit(degrees, Z):

    deficit = torch.zeros_like(degrees, dtype=torch.float)

    for element in [40, 29]:
        mask = (Z == element)
        mean = degrees[mask].float().mean()
        deficit[mask] = mean - degrees[mask].float()

    return deficit


# -------------------------------
# Defect classification based on degree deficit
# -------------------------------
def classify_defects_from_deficit(deficit, degrees, Z, threshold=1.0, print_stats=True):
    """
    threshold = number of missing bonds (default: 1)
    """

    defects = deficit > threshold

    for element in [40, 29]:
        mask = (Z == element)

        deg_elem = degrees[mask].float()

        mean = deg_elem.mean()
        std = deg_elem.std()

        # translate "missing 1 bond" into degree space
        degree_threshold = mean - threshold

        if print_stats:
            print(
                f"Element {element}: "
                f"mean={mean:.2f}, std={std:.2f}, "
                f"degree_threshold={degree_threshold:.2f} "
                f"(= mean - {threshold})"
            )

    return defects


# -------------------------------
# Defect subgraph construction
# -------------------------------
def build_defect_subgraph(graph, defects, cutoff=3.5):
    """
    graph: PyG Data
    defects: boolean tensor (N,)
    cutoff: distance threshold to connect defects (Å)
    """

    pos = graph.pos
    defect_idx = torch.where(defects)[0]

    if len(defect_idx) == 0:
        return None, None

    defect_pos = pos[defect_idx]

    # pairwise distances between defect nodes
    diff = defect_pos.unsqueeze(1) - defect_pos.unsqueeze(0)
    dist = torch.norm(diff, dim=-1)

    # build edges (exclude self-loops)
    mask = (dist < cutoff) & (dist > 0)
    edge_index = mask.nonzero(as_tuple=False).t()

    subgraph = Data(
        x=graph.x[defect_idx],  # keep atom types
        pos=defect_pos,
        edge_index=edge_index
    )

    return subgraph, defect_idx


# -------------------------------
# Defect environment-subgraphs construction
# -------------------------------
def extract_defect_neighbor_subgraphs(graph, defects):

    edge_index = graph.edge_index
    neighbor_graphs = []

    for i in torch.where(defects)[0]:
        # neighbors of node i
        neighbors = edge_index[1][edge_index[0] == i]

        nodes = torch.cat([torch.tensor([i]), neighbors]).unique()

        sub_edge_mask = (
            torch.isin(edge_index[0], nodes) &
            torch.isin(edge_index[1], nodes)
        )

        sub_edge_index = edge_index[:, sub_edge_mask]

        neighbor_graphs.append(Data(
            x=graph.x[nodes],
            pos=graph.pos[nodes],
            edge_index=sub_edge_index
        ))

    return neighbor_graphs


# -------------------------------
# Defect tracking over time
# -------------------------------
def track_defects_over_time(graphs, threshold=1.0):
    tracks = {}     # track_id -> list of positions
    next_id = 0

    prev_positions = None
    prev_ids = None

    max_match_dist = 1.5  # Å - maximum distance to consider the same defect

    for t, g in enumerate(graphs):
        degrees = compute_degrees(g)
        Z = g.x.view(-1)

        deficit = compute_degree_deficit(degrees, Z)
        defects = classify_defects_from_deficit(deficit, degrees, Z, threshold=1.0, print_stats=False)

        defect_idx = torch.where(defects)[0]
        positions = g.pos[defect_idx]

        current_ids = []

        if t == 0:
            for p in positions:
                tracks[next_id] = [p.numpy()]
                current_ids.append(next_id)
                next_id += 1

        else:
            used = set() # to track which previous defects have been matched

            for p in positions:
                if prev_positions is None or len(prev_positions) == 0:
                    break

                dists = np.linalg.norm(prev_positions - p.numpy(), axis=1)
                idx = np.argmin(dists)

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
        prev_ids = current_ids

    return tracks


# -------------------------------
#  Main
# -------------------------------
if __name__ == "__main__":
    file_path = "trj.zr50cu50_6750_vib"
    cutoff = 3.5

    trajectory = load_trajectory(file_path)
    graphs = [atoms_to_pyg(atoms, cutoff) for atoms in trajectory[:20]]

    # --- Analyze one frame ---
    g = graphs[0]

    degrees = compute_degrees(g)
    Z = g.x.view(-1)

    deficit = compute_degree_deficit(degrees, Z)
    defects = classify_defects_from_deficit(deficit, degrees, Z, threshold=1.0)
    # defects = classify_defects(degrees, Z, k=1.0)

    print("Number of defects in first frame:", defects.sum().item())

    # --- Build defect graph ---
    defect_graph, defect_idx = build_defect_subgraph(g, defects)

    print("First frame defect graph:", defect_graph)

    # --- Neighbor graphs ---
    neighbor_graphs = extract_defect_neighbor_subgraphs(g, defects)
    print("Number of neighbor graphs in first frame:", len(neighbor_graphs))

    # --- Track defects ---
    print("Looking at the first 20 frames...")
    print("Tracking defects...")
    tracks = track_defects_over_time(graphs, threshold=1.0)

    print("Number of tracked defects:", len(tracks))
    print("Average defect lifetime (in frames):", sum(len(t) for t in tracks.values()) / len(tracks) if tracks else 0)