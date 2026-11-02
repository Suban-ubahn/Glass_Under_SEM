# =============================================================
# scripts/train_gnn.py -- entry point (was the __main__ block of graph/gnn.py)
# =============================================================

import torch
from graph.gnn import SimpleGNN, build_dataset, train


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
