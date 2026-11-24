# =============================================================
# scripts/view_projection_maps.py -- entry point (was the __main__ block of analysis/projection_maps.py)
# =============================================================

import numpy as np
from analysis.projection_maps import compute_cross_correlation, HOLE_MAP_PATH, CLUSTER_MAP_PATH
from visualization.plot import plot_cross_correlation, plot_map


if __name__ == "__main__":
    # from data_io.trajectory import load_trajectory
    # from config import TRAJECTORY_FILE

    # print("Loading trajectory...")
    # trajectory = load_trajectory(TRAJECTORY_FILE)
    # print(f"Loaded {len(trajectory)} frames")

    # cumulative_holes, cumulative_clusters = build_cumulative_maps(trajectory)

    # np.save(HOLE_MAP_PATH,    cumulative_holes)
    # np.save(CLUSTER_MAP_PATH, cumulative_clusters)

    cumulative_holes    = np.load(HOLE_MAP_PATH)
    cumulative_clusters = np.load(CLUSTER_MAP_PATH)

    plot_map(cumulative_holes,    "Time-Integrated Hole Activity",           colorbar_label="Cumulative free-volume density (a.u.)")
    plot_map(cumulative_clusters, "Time-Integrated Defect Cluster Activity", colorbar_label="Cumulative defect cluster weight (atom·frames)")

    xcorr = compute_cross_correlation()
    plot_cross_correlation(xcorr)
