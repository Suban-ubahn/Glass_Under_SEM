from data_io.trajectory import load_trajectory
from config import TRAJECTORY_FILE

from visualization.animate import animate_d2min_holes_overlay_3d
from analysis.d2min.comparison import compute_d2min_hole_distance_stats
from visualization.plot import plot_d2min_hole_distance

if __name__ == "__main__":
    trajectory = load_trajectory(TRAJECTORY_FILE)

    animate_d2min_holes_overlay_3d(trajectory)

    stats = compute_d2min_hole_distance_stats(trajectory)
    plot_d2min_hole_distance(stats)

    print("Done.")