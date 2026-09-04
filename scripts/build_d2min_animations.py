#!/usr/bin/env python3
"""Build 2D and 3D D2min animations from the MD trajectory.

Run from the repository root:

    python scripts/build_d2min_animations.py

The actual animation implementation lives in visualization/animate.py.
This script only loads the trajectory and calls the visualization functions.
"""

from pathlib import Path
import sys

# Make imports work when the script is called as:
#     python scripts/build_d2min_animations.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_io.trajectory import load_trajectory
from visualization.animate import animate_d2min_2d, animate_d2min_3d


def main() -> None:
    print("Loading trajectory...")
    trajectory = load_trajectory()
    print(f"Loaded {len(trajectory)} frames with {len(trajectory[0])} atoms.")

    print("\nBuilding 2D D²min animation...")
    animate_d2min_2d(trajectory)

    print("\nBuilding 3D D²min animation...")
    animate_d2min_3d(trajectory)

    print("\nD²min animations completed.")


if __name__ == "__main__":
    main()