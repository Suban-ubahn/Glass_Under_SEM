import ase.io
import numpy as np
import torch
from torch_geometric.data import Data
from ase.neighborlist import neighbor_list

# -------------------------------
# Mapping: LAMMPS type -> element
# -------------------------------
type_to_Z = {1: 40, 2: 29}  # Zr=40, Cu=29


# -------------------------------
# Load LAMMPS trajectory
# -------------------------------
def load_trajectory(file_path):
    return ase.io.read(
        file_path,
        format='lammps-dump-text',
        index='::1'  # all timesteps
    )


# -------------------------------
# RBF edge features
# -------------------------------
def rbf(dist, D_min=0.0, D_max=5.0, num_kernels=32):
    D_mu = torch.linspace(D_min, D_max, num_kernels)
    D_sigma = (D_max - D_min) / num_kernels
    dist = dist.view(-1, 1)
    return torch.exp(-((dist - D_mu) ** 2) / (2 * D_sigma ** 2))


# -------------------------------
# ASE Atoms -> PyG graph
# -------------------------------
def atoms_to_pyg(atoms, cutoff):
    # Node features
    types = atoms.arrays['type']
    Z = np.array([type_to_Z[t] for t in types])
    x = torch.tensor(Z, dtype=torch.long).view(-1, 1)

    pos = torch.tensor(atoms.positions, dtype=torch.float)

    # Neighbor list with periodic shifts
    i, j, d, S = neighbor_list('ijdS', atoms, cutoff)       
    # i, j: connected atom indices; 
    # d: distances; 
    # S: PBC shift in lattice coordinates (e.g. S[k] = (1, 0, 0) “To reach atom j from atom i, you must shift by +1 box length in x”)

    edge_index = torch.tensor(np.vstack([i, j]), dtype=torch.long)

    # PBC-corrected relative vectors
    cell = torch.tensor(atoms.cell.array, dtype=torch.float)    # (3, 3) simulation box vectors
    S = torch.tensor(S, dtype=torch.float)
    rij = pos[j] - pos[i] + torch.matmul(S, cell)   # S @ cell gives the PBC shift in Cartesian coordinates

    distances = torch.norm(rij, dim=1) # Euclidean norm
    edge_attr = rbf(distances)

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_vec=rij,
        distances=distances
    )


# -------------------------------
# Convert full trajectory -> PyG graphs
# -------------------------------
def trajectory_to_graphs(trajectory, cutoff):
    graphs = []
    for atoms in trajectory:
        graphs.append(atoms_to_pyg(atoms, cutoff))
    return graphs


# -------------------------------
# Usage
# -------------------------------
if __name__ == "__main__":
    file_path = "trj.zr50cu50_6750_vib"
    cutoff = 3.5

    # Load trajectory (list of ASE Atoms)
    trajectory = load_trajectory(file_path)

    # Convert to PyG graphs
    graphs = trajectory_to_graphs(trajectory, cutoff)

    # Example
    print(graphs[0])