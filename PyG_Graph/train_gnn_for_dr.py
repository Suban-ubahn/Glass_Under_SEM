# Training a GNN for Glass Dynamics (PyTorch Geometric)

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

from convert_to_PyG_graph import atoms_to_pyg


def compute_displacement(atoms_t, atoms_tp1):
    pos_t = torch.tensor(atoms_t.positions, dtype=torch.float)
    pos_tp1 = torch.tensor(atoms_tp1.positions, dtype=torch.float)

    cell = torch.tensor(atoms_t.cell.array, dtype=torch.float)
    pbc = atoms_t.pbc

    # fractional coordinates
    inv_cell = torch.inverse(cell.T)

    frac_t = pos_t @ inv_cell
    frac_tp1 = pos_tp1 @ inv_cell

    dfrac = frac_tp1 - frac_t

    # apply minimum image convention
    for dim in range(3):
        if pbc[dim]:
            dfrac[:, dim] -= torch.round(dfrac[:, dim])

    # back to cartesian
    disp = dfrac @ cell

    return disp  # (N, 3)

# 3. Dataset Construction


def build_dataset(trajectory, cutoff):
    dataset = []

    for t in range(len(trajectory) - 1):
        atoms_t = trajectory[t]
        atoms_tp1 = trajectory[t + 1]

        data = atoms_to_pyg(atoms_t, cutoff)

        # PBC-correct displacement target
        y = compute_displacement(atoms_t, atoms_tp1)

        data.y = y  # (N, 3)

        dataset.append(data)

    return dataset

# 4. GNN Model (uses geometry)

class SimpleGNN(MessagePassing):
    def __init__(self, hidden_dim=64):
        super().__init__(aggr='add')

        self.node_emb = nn.Embedding(100, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(32 + 3, hidden_dim),  # RBF + vector
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, data):
        x = self.node_emb(data.x.view(-1))

        return self.propagate(
            data.edge_index,
            x=x,
            edge_attr=data.edge_attr,
            edge_vec=data.edge_vec
        )

    def message(self, x_j, edge_attr, edge_vec):
        m = torch.cat([edge_attr, edge_vec], dim=1)
        m = self.edge_mlp(m)
        return x_j + m

    def update(self, aggr_out):
        return self.node_mlp(aggr_out)


# 5. Training Loop

def train(model, dataset, epochs=10):
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for batch in loader:
            optimizer.zero_grad()

            pred = model(batch)
            loss = loss_fn(pred, batch.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: {total_loss:.4f}")

#   Implementation

if __name__ == "__main__":
    import ase.io

    file_path = "trj.zr50cu50_6750_vib"
    cutoff = 3.5

    # Load trajectory
    trajectory = ase.io.read(
        file_path,
        format='lammps-dump-text',
        index='::1'  # start small!
    )

    print("Loaded trajectory:", len(trajectory), "frames")

    # Build dataset
    dataset = build_dataset(trajectory, cutoff)

    print("Dataset size:", len(dataset))
    print("One graph:", dataset[0])

    # Initialize model
    model = SimpleGNN(hidden_dim=64)

    # Train
    train(model, dataset, epochs=6)


# Visualize predicitons for a single graph

model.eval()

data = dataset[0]

with torch.no_grad():
    pred = model(data)

print("Pred shape:", pred.shape)   # (N, 3)
print("True shape:", data.y.shape)

error = pred - data.y
print("Mean error:", error.abs().mean())

import matplotlib.pyplot as plt

err = (pred - data.y).norm(dim=1)

plt.hist(err.numpy(), bins=50)
plt.xlabel("Displacement error (Å)")
plt.ylabel("Count")
plt.title("Prediction Error Distribution")
plt.show()

true_mag = data.y.norm(dim=1)
pred_mag = pred.norm(dim=1)

plt.scatter(true_mag.numpy(), pred_mag.numpy(), s=1)
plt.xlabel("True |Δr| (Å)")
plt.ylabel("Predicted |Δr| (Å)")
plt.plot([0, true_mag.max()], [0, true_mag.max()], 'r--')
plt.show()