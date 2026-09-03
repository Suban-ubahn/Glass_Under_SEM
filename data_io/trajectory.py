# =============================================================
# io/trajectory.py — LAMMPS trajectory loading
#
# Single place for reading the dump file. All other modules
# import load_trajectory() from here instead of calling
# ase.io.read() directly.
# =============================================================

import ase.io

from config import TRAJECTORY_FILE, TYPE_TO_Z


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
        One Atoms object per timestep, with REAL atomic numbers set
        (Zr=40, Cu=29 — see config.TYPE_TO_Z), not the raw LAMMPS
        dump reader's default.

    Note
    ----
    ASE's LAMMPS dump reader sets atoms.numbers to the raw LAMMPS
    type column (1, 2 here) — NOT real atomic numbers. Left
    uncorrected, this silently makes every atom look like hydrogen
    or helium to anything relying on ASE's species-aware machinery
    (e.g. neighbor_list's per-species-pair cutoff dict, or
    get_chemical_symbols()) — no error, just wrong results. Fixed
    here, once, so every caller gets correct atomic numbers
    automatically instead of needing its own ad-hoc correction.
    """
    trajectory = ase.io.read(file_path, format="lammps-dump-text", index=index)

    # ase.io.read returns a single Atoms object for a plain integer index
    # (e.g. index="0"), but a list for slice-style indices (e.g. "::1").
    # Handle both so this fix applies no matter how it's called.
    frames = trajectory if isinstance(trajectory, list) else [trajectory]
    for atoms in frames:
        atoms.set_atomic_numbers([TYPE_TO_Z[t] for t in atoms.arrays["type"]])

    return trajectory