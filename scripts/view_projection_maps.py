# =============================================================
# scripts/view_projection_maps.py -- entry point (was the __main__ block of analysis/projection_maps.py)
# =============================================================

import numpy as np
from analysis.projection_maps import compute_cross_correlation
from visualization.plot import plot_cross_correlation, plot_map


if __name__ == "__main__":
    # from data_io.trajectory import load_trajectory
    # from config import TRAJECTORY_FILE

    # print("Loading trajectory...")
    # trajectory = load_trajectory(TRAJECTORY_FILE)
    # print(f"Loaded {len(trajectory)} frames")

    # cumulative_holes, cumulative_clusters = build_cumulative_maps(trajectory)

    # np.save("outputs/projection_maps/cumulative_hole_map.npy",    cumulative_holes)
    # np.save("outputs/projection_maps/cumulative_cluster_map.npy", cumulative_clusters)

    cumulative_holes    = np.load("outputs/projection_maps/cumulative_hole_map.npy")
    cumulative_clusters = np.load("outputs/projection_maps/cumulative_cluster_map.npy")

    plot_map(cumulative_holes,    "Time-Integrated Hole Activity",           colorbar_label="Cumulative free-volume density (a.u.)")
    plot_map(cumulative_clusters, "Time-Integrated Defect Cluster Activity", colorbar_label="Cumulative defect cluster weight (atom·frames)")

    xcorr = compute_cross_correlation()
    plot_cross_correlation(xcorr)
