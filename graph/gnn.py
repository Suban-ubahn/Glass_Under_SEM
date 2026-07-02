
# =============================================================
# graph/gnn.py — GNN for atomic displacement prediction
#
# Source: PyG_Graph/train_gnn_for_dr.py
# =============================================================

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

from graph.build import atoms_to_pyg
from config import GNN_HIDDEN_DIM, GNN_EPOCHS, GNN_LR, GNN_BATCH_SIZE, CUTOFF


# ------------------------------------------------------------
# PBC-correct displacement between consecutive frames
# ------------------------------------------------------------

def compute_displacement(atoms_t, atoms_tp1):
    """
    Compute the per-atom displacement vector from frame t to t+1
    using the minimum-image convention (handles atoms that jump
    across a periodic boundary).

    Returns
    -------
    disp : (N, 3) float Tensor — displacement in Å
    """
    pos_t   = torch.tensor(atoms_t.positions,   dtype=torch.float)
    pos_tp1 = torch.tensor(atoms_tp1.positions, dtype=torch.float)
    cell    = torch.tensor(atoms_t.cell.array,  dtype=torch.float)
    pbc     = atoms_t.pbc

    inv_cell = torch.inverse(cell.T)
    frac_t   = pos_t   @ inv_cell
    frac_tp1 = pos_tp1 @ inv_cell
    dfrac    = frac_tp1 - frac_t

    # minimum image — fold each fractional component into (-0.5, +0.5]
    for dim in range(3):
        if pbc[dim]:
            dfrac[:, dim] -= torch.round(dfrac[:, dim])

    return dfrac @ cell   # (N, 3)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

def build_dataset(trajectory, cutoff=CUTOFF):
    """
    Build a list of PyG Data objects where each graph represents
    the atomic structure at frame t and the displacement to frame
    t+1 is stored as graph.y.
    """
    dataset = []
    for t in range(len(trajectory) - 1):
        data   = atoms_to_pyg(trajectory[t], cutoff)
        data.y = compute_displacement(trajectory[t], trajectory[t + 1])
        dataset.append(data)
    return dataset


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class SimpleGNN(MessagePassing):
    """
    Single-layer message-passing GNN that predicts the 3-D
    displacement vector for each atom.

    Node embedding  : atomic number Z → hidden_dim
    Edge message    : concat(RBF edge features, bond vector) → hidden_dim
    Node update     : aggregated messages → 3-D output
    """
    def __init__(self, hidden_dim=GNN_HIDDEN_DIM):
        super().__init__(aggr="add")

        self.node_emb = nn.Embedding(100, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(32 + 3, hidden_dim),   # 32 RBF dims + 3 bond vector dims
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, data):
        x = self.node_emb(data.x.view(-1))
        return self.propagate(
            data.edge_index,
            x        = x,
            edge_attr = data.edge_attr,
            edge_vec  = data.edge_vec,
        )

    def message(self, x_j, edge_attr, edge_vec):
        m = torch.cat([edge_attr, edge_vec], dim=1)
        return x_j + self.edge_mlp(m)

    def update(self, aggr_out):
        return self.node_mlp(aggr_out)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train(model, dataset, epochs=GNN_EPOCHS, lr=GNN_LR,
          batch_size=GNN_BATCH_SIZE):
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            pred  = model(batch)
            loss  = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:3d}  loss={total_loss:.4f}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import TRAJECTORY_FILE
    import matplotlib.pyplot as plt

    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames")

    dataset = build_dataset(trajectory)
    print(f"Dataset size: {len(dataset)}")

    model = SimpleGNN()
    train(model, dataset)

    # --- evaluate on first graph ---
    model.eval()
    data = dataset[0]
    with torch.no_grad():
        pred = model(data)

    err = (pred - data.y).norm(dim=1)
    print(f"Mean displacement error: {err.mean():.4f} Å")

    plt.hist(err.numpy(), bins=50)
    plt.xlabel("Displacement error (Å)")
    plt.ylabel("Count")
    plt.title("Prediction Error Distribution")
    plt.show()
