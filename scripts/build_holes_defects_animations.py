# =============================================================
# scripts/build_animations.py — generate all three trajectory
# animations (KDTree holes, FFT holes, defect clusters).
#
# visualization/animate.py's own __main__ only runs one of these
# at a time (the other two calls are commented out there, for
# quick interactive use). This driver runs all three, for use in
# run_pipeline.sh or on its own.
# =============================================================

from data_io.trajectory import load_trajectory
from config import TRAJECTORY_FILE
from visualization.animate import (
    animate_holes_KDTree,
    animate_holes_fft,
    animate_defects,
)

if __name__ == "__main__":
    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames")

    animate_holes_KDTree(trajectory)
    animate_holes_fft(trajectory)
    animate_defects(trajectory)

    print("Done.")