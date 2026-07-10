# =============================================================
# diffraction/simulate.py — 4D-STEM simulation for ptychography
#
# Workflow:
#   1. Load the full MD trajectory
#   2. Compute the time-averaged structure (average atom positions
#      across all frames) — this is what we simulate, since the
#      ptychographic reconstruction also integrates over time
#   3. Run a full GridScan with a PixelatedDetector to produce
#      the 4D dataset (scan_x, scan_y, kx, ky) that the
#      ptychography reconstruction code expects
#   4. Save to .zarr (abtem's preferred format for large datasets)
#
# The frozen-phonon averaging handles within-frame thermal motion;
# averaging over MD frames handles between-frame structural evolution.
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import abtem
from ase import Atoms
import time

from config import (
    TYPE_TO_Z,
    TYPE_TO_RADIUS,
    BEAM_ENERGY_KV,
    MAX_ANGLE_MRAD,
    GPTS,
    SLICE_THICKNESS,
    NUM_PHONON_CONFIGS,
    PHONON_SIGMA,
    CONVERGENCE_SEMIANGLE_MRAD,
)

from visualization.plot import plot_mean_diffraction


# ------------------------------------------------------------
# Average structure
# ------------------------------------------------------------

def compute_average_structure(trajectory):
    """
    Average the atomic positions across all MD frames to produce
    a single representative structure.

    The cell, element types, and periodic boundary conditions are
    taken from the first frame (they don't change during the MD run).

    Returns
    -------
    avg_atoms : ase.Atoms
    """
    print(f"Averaging positions over {len(trajectory)} frames...")

    positions = np.mean(
        [atoms.positions for atoms in trajectory], axis=0
    )

    ref       = trajectory[0]
    avg_atoms = Atoms(
        numbers   = ref.numbers,
        positions = positions,
        cell      = ref.cell,
        pbc       = ref.pbc,
    )
    return avg_atoms

# ------------------------------------------------------------
# Atom type correction  (LAMMPS int → atomic number)
# ------------------------------------------------------------

def correct_atom_types(atoms):
    """Replace LAMMPS type integers with proper atomic numbers."""
    for atom in atoms:
        atom.number = TYPE_TO_Z[atom.number]
    return atoms


# ------------------------------------------------------------
# 4D-STEM simulation
# ------------------------------------------------------------

def simulate_4d_stem(atoms, output_file="diffraction/4d_stem.zarr"):
    """
    Run a full GridScan with a PixelatedDetector over the entire
    sample area and save the resulting 4D dataset.

    The scan sampling is set to the probe's Nyquist sampling rate,
    which is the standard choice for ptychography.

    Parameters
    ----------
    atoms       : ase.Atoms — the structure to simulate (already
                  type-corrected, i.e. real atomic numbers)
    output_file : str — path for the .zarr output

    Returns
    -------
    diffraction_patterns : abtem DiffractionPatterns measurement
    """
    
    print("Building potential...")
    frozen_phonons = abtem.FrozenPhonons(
        atoms,
        num_configs = NUM_PHONON_CONFIGS,
        sigmas      = PHONON_SIGMA,
    )
    potential = abtem.Potential(
        frozen_phonons,
        gpts            = GPTS,
        slice_thickness = SLICE_THICKNESS,
        projection      = "finite",
    )

    print("Building probe...")
    probe = abtem.Probe(
        energy           = BEAM_ENERGY_KV * 1e3,
        semiangle_cutoff = CONVERGENCE_SEMIANGLE_MRAD,
    )

    print("Building scan grid (Nyquist sampling)...")
    scan = abtem.GridScan(
        start      = (0, 0),
        end        = potential.extent,
        sampling   = 3*probe.aperture.nyquist_sampling,
    )
    print(f"  Scan grid: {scan.shape} positions")
    print(f"  Probe Nyquist sampling: {probe.aperture.nyquist_sampling:.3f} Å")

    detector = abtem.PixelatedDetector(
        max_angle = MAX_ANGLE_MRAD,
        url       = output_file,   # stream directly to disk
    )

    # Starting clock
    t0 = time.time()

    print("Running 4D-STEM scan (this will take a while)...")
    diffraction_patterns = probe.scan(
        potential,
        scan      = scan,
        detectors = detector,
    )

    diffraction_patterns.compute()
    diffraction_patterns.to_zarr(output_file)

    print(f"Computation time: {(time.time() - t0) / 60:.1f} minutes")
    print(f"Saved 4D dataset to: {output_file}")
    print(f"  Shape: {diffraction_patterns.array.shape}  "
          f"(scan_x, scan_y, kx, ky)")

    return diffraction_patterns

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    import zarr
    # from data_io.trajectory import load_trajectory
    # from config import TRAJECTORY_FILE

    # print("Loading trajectory...")
    # trajectory = load_trajectory(TRAJECTORY_FILE)
    # print(f"Loaded {len(trajectory)} frames")

    # avg_atoms = compute_average_structure(trajectory)
    # avg_atoms = correct_atom_types(avg_atoms)
    # avg_atoms = avg_atoms.repeat((1, 1, 4))  # tile 4x in Z only

    # diffraction_patterns = simulate_4d_stem(avg_atoms)
    diffraction_patterns = abtem.from_zarr("diffraction/4d_stem.zarr")
    plot_mean_diffraction(diffraction_patterns)