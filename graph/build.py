# =============================================================
# graph/build.py — ASE Atoms → PyTorch Geometric graph
#
# Source: PyG_Graph/convert_to_PyG_graph.py
# =============================================================


import numpy as np
import torch
from torch_geometric.data import Data
from ase.neighborlist import neighbor_list

from config import TYPE_TO_Z, GRAPH_CUTOFF_DICT


# ------------------------------------------------------------
# Radial basis function  (encodes bond distance as a vector)
# ------------------------------------------------------------

def rbf(dist, D_min=0.0, D_max=5.0, num_kernels=32):
    """
    Expand scalar distances into a Gaussian RBF feature vector.

    Each kernel is centred at a different distance between D_min
    and D_max, so the output encodes *where* along the distance
    axis a bond sits, not just the raw number.

    Parameters
    ----------
    dist : (E,) Tensor — bond distances in Å
    Returns
    -------
    (E, num_kernels) Tensor
    """
    D_mu    = torch.linspace(D_min, D_max, num_kernels)
    D_sigma = (D_max - D_min) / num_kernels
    dist    = dist.view(-1, 1)
    return torch.exp(-((dist - D_mu) ** 2) / (2 * D_sigma ** 2))


# ------------------------------------------------------------
# Main conversion
# ------------------------------------------------------------

def atoms_to_pyg(atoms, cutoff=GRAPH_CUTOFF_DICT):
    """
    Convert an ASE Atoms object to a PyTorch Geometric Data graph.

    Node features
    -------------
    x   : (N, 1) long tensor — atomic number (Z)
    pos : (N, 3) float tensor — Cartesian positions (Å)

    Edge features  (one entry per directed bond i→j)
    -------------
    edge_index : (2, E) long tensor
    edge_attr  : (E, 32) float tensor — RBF-encoded distance
    edge_vec   : (E, 3) float tensor — PBC-corrected bond vector rij
    distances  : (E,) float tensor — scalar bond lengths (Å)

    Periodic boundary conditions are handled via ASE's
    neighbor_list, which returns lattice shift vectors S.
    The true bond vector is:  rij = pos[j] - pos[i] + S @ cell
    """
    # --- node features ---
    types = atoms.arrays["type"]
    Z     = np.array([TYPE_TO_Z[t] for t in types])
    x     = torch.tensor(Z, dtype=torch.long).view(-1, 1)
    pos   = torch.tensor(atoms.positions, dtype=torch.float)

    # --- edges with PBC ---
    i, j, d, S = neighbor_list("ijdS", atoms, cutoff)
    # i, j : atom index pairs
    # d    : distances (without PBC correction, used only to build list)
    # S    : lattice shift vectors (integer multiples of cell vectors)

    edge_index = torch.tensor(np.vstack([i, j]), dtype=torch.long)

    cell = torch.tensor(atoms.cell.array, dtype=torch.float)   # (3, 3)
    S    = torch.tensor(S, dtype=torch.float)
    rij  = pos[j] - pos[i] + torch.matmul(S, cell)             # PBC-correct

    distances  = torch.norm(rij, dim=1)
    edge_attr  = rbf(distances)

    return Data(
        x          = x,
        pos        = pos,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        edge_vec   = rij,
        distances  = distances,
    )


def trajectory_to_graphs(trajectory, cutoff=GRAPH_CUTOFF_DICT):
    """Convert a full trajectory (list of Atoms) to a list of PyG Data objects."""
    return [atoms_to_pyg(atoms, cutoff) for atoms in trajectory]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import TRAJECTORY_FILE

    trajectory = load_trajectory(TRAJECTORY_FILE)
    graphs     = trajectory_to_graphs(trajectory)
    print("Example graph:", graphs[0])