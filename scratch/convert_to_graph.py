import numpy as np
import ase.io
from ase.neighborlist import neighbor_list

type_to_Z = {
    1: 40,  # Zr
    2: 29   # Cu
}

trajectory = ase.io.read(
    'trj.zr50cu50_6750_vib',
    format='lammps-dump-text',
    index='::10'  # every 10th timestep
)

cutoff = 3.5  # Å

graphs = []

def atoms_to_graph(atoms, cutoff):
    """
    Converts ASE Atoms → graph representation
    """
    
    # --- Node features ---
    types = atoms.arrays['type']  # LAMMPS types
    Z = np.array([type_to_Z[t] for t in types])  # atomic numbers
    
    positions = atoms.positions  # (N, 3)
    
    # --- Edges with PBC ---
    i, j, d, D = neighbor_list('ijdD', atoms, cutoff)
    
    # i, j → indices of connected atoms
    # d → distances
    
    edge_index = np.vstack([i, j])   # shape (2, E)
    edge_length = d.reshape(-1, 1)   # distance is edge attribute, shape (E, 1) 
    edge_vector = D                  # (E, 3)
    
    return {
        "atomic_number": Z,          
        "positions": positions,
        "edge_index": edge_index,
        "edge_length": edge_length,
        "edge_vector": edge_vector
    }

for atoms in trajectory:
    graph = atoms_to_graph(atoms, cutoff)
    graphs.append(graph)

print(graphs[0]["atomic_number"])  # print the first graph to check