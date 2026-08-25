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
from utils import get_output_dir
import os
from ase import Atoms

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
    BLOCKED_VAL,
    SAMPLING_FACTOR,
    TO_ZARR,
    DEFOCUS,
    SCAN_POSITION
)

from visualization.plot import (
    plot_average_diffraction, 
    plot_diffraction_single_position,
    plot_virtual_detector_heatmap,
    plot_detector_mask
    )


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

    # #  Potential visualization
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) 
    # abtem.show_atoms(atoms, ax=ax1, legend=True) 
    # visualization = (potential.build() * 0.1).show(ax=ax2) 

    print("Building probe...")
    probe = abtem.Probe(
        energy           = BEAM_ENERGY_KV * 1e3,
        semiangle_cutoff = CONVERGENCE_SEMIANGLE_MRAD,
        defocus          = DEFOCUS,   # Å 
    )
    probe.grid.match(potential)

    print("Building scan grid (Nyquist sampling)...")

    scan_sampling = (
        SAMPLING_FACTOR * probe.aperture.nyquist_sampling
    )
    scan = abtem.GridScan(
        start      = (0, 0),
        end        = (1, 1),       # fractional: 1 = full cell
        sampling   = scan_sampling,
        fractional = True,
        potential  = potential,
    )
    print(f"  Scan grid: {scan.shape} positions")
    print(f"  Probe Nyquist sampling: {probe.aperture.nyquist_sampling:.3f} Å")
    print(f"  Actual scan sampling: {scan_sampling:.3f} Å")

    detector = abtem.PixelatedDetector(
        max_angle = MAX_ANGLE_MRAD,
    )

    print("Running 4D-STEM scan (this will take a while)...")

    diffraction_patterns = probe.scan(
        potential,
        scan      = scan,
        detectors = detector,
    )

    print("Computing 4D-STEM scan...")
    diffraction_patterns.compute()

    print("4D-STEM computation finished.")
    print(
    f"  Shape: {diffraction_patterns.array.shape} "
    f"(scan_x, scan_y, kx, ky)"
    )

    # diffraction_patterns = abtem.stack(diffraction_patterns, ("BF", "MAADF", "HAADF"))

    if TO_ZARR:
      diffraction_patterns.to_zarr(output_file)
      print(f"Saved 4D dataset to: {output_file}")
    else:
        print("Zarr output disabled; dataset was not saved.")

    return diffraction_patterns
    
def simulate_plane_wave(atoms, save_path=None):
    """
    Simulate a single diffraction pattern using a plane wave
    (equivalent to selected area electron diffraction).
    This illuminates the entire sample at once and produces
    a single 2D diffraction pattern.
    """
    if save_path is None:
        save_path = os.path.join(get_output_dir(), "plane_wave_diffraction.png")
    print("Building potential for plane wave simulation...")
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

    print("Running plane wave multislice...")
    wave        = abtem.PlaneWave(energy=BEAM_ENERGY_KV * 1e3)
    exit_waves  = wave.multislice(potential)
    measurement = exit_waves.diffraction_patterns(max_angle=MAX_ANGLE_MRAD)
    measurement = measurement.mean(0)
    measurement.compute()

    measurement.to_zarr("diffraction/plane_wave_diffraction.zarr")
    plot_average_diffraction(measurement, blocked=BLOCKED_VAL, save_path=save_path)

# ========================
# RESOLUTION TEST
# ========================

def resolution_test(atoms):
    """
    Run plane wave simulations at different resolutions and save
    one diffraction plot per resolution level to compare quality
    without running the full 4D-STEM scan.
    """
    from visualization.plot import plot_diffraction

    configs = [
        {"gpts": 128, "slice_thickness": 4, "label": "low"},
        {"gpts": 256, "slice_thickness": 2, "label": "medium"},
        {"gpts": 512, "slice_thickness": 1, "label": "high"},
    ]

    for cfg in configs:
        print(f"Running resolution test: {cfg['label']} "
              f"({cfg['gpts']} gpts, {cfg['slice_thickness']}Å slice)...")

        frozen_phonons = abtem.FrozenPhonons(
            atoms, num_configs=NUM_PHONON_CONFIGS, sigmas=PHONON_SIGMA
        )
        potential = abtem.Potential(
            frozen_phonons,
            gpts            = cfg["gpts"],
            slice_thickness = cfg["slice_thickness"],
            projection      = "finite",
        )
        wave        = abtem.PlaneWave(energy=BEAM_ENERGY_KV * 1e3)
        exit_waves  = wave.multislice(potential)
        measurement = exit_waves.diffraction_patterns(max_angle=MAX_ANGLE_MRAD)
        measurement = measurement.mean(0)
        measurement.compute()

        plot_average_diffraction(
            measurement,
            blocked   = BLOCKED_VAL,
            save_path = os.path.join(get_output_dir(), f"resolution_test_{cfg['label']}.png"),
        )

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import TRAJECTORY_FILE
    from analysis.projection_maps import compare_abtem_to_projection_maps
    import time


    # Starting clock
    t0 = time.time()

    print("Loading trajectory...")
    trajectory = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(trajectory)} frames")

    avg_atoms = compute_average_structure(trajectory)
    avg_atoms = correct_atom_types(avg_atoms)

    # resolution_test(avg_atoms)
    diffraction_patterns = simulate_4d_stem(avg_atoms, output_file="diffraction/4d_stem.zarr")
    diffraction_patterns = abtem.from_zarr("diffraction/4d_stem.zarr")
    
    # plot_average_diffraction(
    #     diffraction_patterns,
    #     blocked=BLOCKED_VAL,
    #     save_path="diffraction/mean_diffraction.png",
    # )
    
    plot_diffraction_single_position(
        diffraction_patterns,
        blocked=BLOCKED_VAL,
        scan_position=SCAN_POSITION,
        save_path=os.path.join(get_output_dir(), "single_position.png"),
    )

    bf_image, bf_mask = plot_virtual_detector_heatmap(
        diffraction_patterns,
        inner_radius_px=0,
        outer_radius_px=None,
        center=None,
        log_scale=False,
        normalize=True,
        title="Virtual Bright-Field Image",
        save_path=os.path.join(get_output_dir(), "virtual_bright_field.png"),
    )
    plot_detector_mask(bf_mask)

    df_image, df_mask = plot_virtual_detector_heatmap(
        diffraction_patterns,
        inner_radius_px=70,
        outer_radius_px=180,
        center=None,
        log_scale=True,
        normalize=True,
        title="Virtual Dark-Field Image",
        save_path=os.path.join(get_output_dir(), "virtual_dark_field.png"),
    )
    plot_detector_mask(df_mask)

    compare_abtem_to_projection_maps(bf_image)

    # simulate_plane_wave(avg_atoms) # also generates a plot

    print(f"Computation time: {(time.time() - t0) / 60:.1f} minutes")