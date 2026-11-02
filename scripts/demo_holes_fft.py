# =============================================================
# scripts/demo_holes_fft.py -- entry point (was the __main__ block of holes/fft.py)
# =============================================================

from config import VOXEL_EDGE
from holes.fft import analyze_frame


if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import TRAJECTORY_FILE

    atoms = load_trajectory(TRAJECTORY_FILE, index="0")
    if not hasattr(atoms, "positions"):
        atoms = atoms[0]

    Vcell, Vloch, hole_positions, hole_strengths = analyze_frame(atoms)

    print(f"Voxel edge : {VOXEL_EDGE} Å")
    print(f"Holes found: {len(hole_positions)}")
    print(f"Vcell  min={Vcell.min():.3f}  max={Vcell.max():.3f}  "
          f"mean={Vcell.mean():.3f}  std={Vcell.std():.3f}")
