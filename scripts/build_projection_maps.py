# =============================================================
# scripts/build_projection_maps.py — run the hole/defect
# projection-map pipeline stage over the full trajectory.
#
# This performs the computation that analysis/projection_maps.py's
# own __main__ block currently has commented out (it only reloads
# saved .npy files by default). Kept as a separate driver script so
# it can be re-run on its own or as part of run_pipeline.sh.
# =============================================================

from data_io.trajectory import load_trajectory
from config import TRAJECTORY_FILE
from analysis.projection_maps import (
    build_cumulative_maps,
    compute_cross_correlation,
    plot_cross_correlation,
)
from visualization.plot import plot_map
import os
import numpy as np

if __name__ == "__main__":
    print("Loading trajectory...")
    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames")

    cumulative_holes, cumulative_clusters = build_cumulative_maps(trajectory)

    os.makedirs("outputs/projection_maps", exist_ok=True)
    np.save("outputs/projection_maps/cumulative_hole_map.npy", cumulative_holes)
    np.save("outputs/projection_maps/cumulative_cluster_map.npy", cumulative_clusters)

    plot_map(
        cumulative_holes, "Time-Integrated Hole Activity",
        colorbar_label="Cumulative free-volume density (a.u.)",
    )
    plot_map(
        cumulative_clusters, "Time-Integrated Defect Cluster Activity",
        colorbar_label="Cumulative defect density (a.u.)",
    )

    xcorr = compute_cross_correlation(cumulative_holes, cumulative_clusters)
    plot_cross_correlation(xcorr)

    print("Done.")