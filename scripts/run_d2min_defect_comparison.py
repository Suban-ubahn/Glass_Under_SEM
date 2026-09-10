from data_io.trajectory import load_trajectory
from config import TRAJECTORY_FILE

from visualization.animate import animate_d2min_defects_overlay_3d
from analysis.d2min_comparison import compute_d2min_defect_stats
from visualization.plot import plot_d2min_defect_comparison

if __name__ == "__main__":
    trajectory = load_trajectory(TRAJECTORY_FILE)

    stats = compute_d2min_defect_stats(trajectory)
    plot_d2min_defect_comparison(stats)

    animate_d2min_defects_overlay_3d(trajectory)

    print("Done.")