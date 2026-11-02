# =============================================================
# scripts/demo_holes_kdtree.py -- entry point (was the __main__ block of holes/kdtree.py)
# =============================================================

from holes.kdtree import analyze_frame
from visualization.plot import visualize_holes_3d


if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import TRAJECTORY_FILE

    trajectory = load_trajectory(TRAJECTORY_FILE, index="50")
    atoms      = trajectory if hasattr(trajectory, "positions") else trajectory[0]

    volumes, void_grid, labeled, num_features = analyze_frame(atoms)

    print(f"Holes detected: {len(volumes)}")
    visualize_holes_3d(atoms, void_grid, labeled, num_features)
