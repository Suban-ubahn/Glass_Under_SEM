# =============================================================
# scripts/demo_graph_build.py -- entry point (was the __main__ block of graph/build.py)
# =============================================================

from graph.build import trajectory_to_graphs


if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import TRAJECTORY_FILE

    trajectory = load_trajectory(TRAJECTORY_FILE)
    graphs     = trajectory_to_graphs(trajectory)
    print("Example graph:", graphs[0])
