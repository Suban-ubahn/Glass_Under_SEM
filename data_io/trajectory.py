# =============================================================
# io/trajectory.py — LAMMPS trajectory loading
#
# Single place for reading the dump file. All other modules
# import load_trajectory() from here instead of calling
# ase.io.read() directly.
# =============================================================

import ase.io

from config import TRAJECTORY_FILE


def load_trajectory(file_path=TRAJECTORY_FILE, index="::1"):
    """
    Load a LAMMPS dump trajectory via ASE.

    Parameters
    ----------
    file_path : str
        Path to the LAMMPS dump file.
    index : str
        ASE index string.
        "::1"  → every frame  (default)
        "::10" → every 10th frame
        "0"    → first frame only
        ":"    → all frames (same as ::1)

    Returns
    -------
    list of ase.Atoms
        One Atoms object per timestep.
    """
    return ase.io.read(
        file_path,
        format="lammps-dump-text",
        index=index
    )
