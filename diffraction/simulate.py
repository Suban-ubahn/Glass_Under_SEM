# =============================================================
# diffraction/simulate.py — abtem electron diffraction pipeline
#
# Source: Diffractions_Ptychography/simulate_diffraction_patterns.py
# (Already well-organised; only import paths updated.)
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio.v2 as imageio
import abtem

from config import (
    TYPE_TO_Z,
    TYPE_TO_RADIUS,
    BEAM_ENERGY_KV,
    MAX_ANGLE_MRAD,
    GPTS,
    SLICE_THICKNESS,
    NUM_PHONON_CONFIGS,
    PHONON_SIGMA,
)

plt.rcParams["figure.dpi"]       = 100
plt.rcParams["savefig.dpi"]      = 200
plt.rcParams["image.interpolation"] = "bilinear"
plt.rcParams["image.cmap"]       = "inferno"


# ------------------------------------------------------------
# Frame preprocessing
# ------------------------------------------------------------

def correct_atom_types(atoms):
    """
    Replace LAMMPS integer type numbers with proper atomic numbers
    so abtem knows which elements are present.
    """
    for atom in atoms:
        atom.number = TYPE_TO_Z[atom.number]
    return atoms


def add_radii(atoms):
    """Attach per-atom covalent/metallic radii needed by abtem."""
    atoms.radii = np.array([TYPE_TO_RADIUS[a.number] for a in atoms])
    return atoms


def process_frame(frame):
    """Apply type correction and radius assignment to one frame."""
    frame = correct_atom_types(frame)
    frame = add_radii(frame)
    return frame


# ------------------------------------------------------------
# abtem simulation objects
# ------------------------------------------------------------

def build_frozen_phonons(frame, sigma=PHONON_SIGMA,
                          num_configs=NUM_PHONON_CONFIGS):
    return abtem.FrozenPhonons(frame, num_configs=num_configs, sigmas=sigma)


def build_potential(frozen_phonons):
    return abtem.Potential(
        frozen_phonons,
        gpts            = GPTS,
        projection      = "infinite",
        slice_thickness = SLICE_THICKNESS,
    )


def build_wave():
    return abtem.PlaneWave(energy=BEAM_ENERGY_KV * 1e3)


# ------------------------------------------------------------
# Diffraction simulation
# ------------------------------------------------------------

def simulate_diffraction_single_frame(frame):
    """
    Run a frozen-phonon multislice simulation for one preprocessed
    frame and return the averaged diffraction pattern.
    """
    print("  Building frozen phonons...")
    fp        = build_frozen_phonons(frame)
    potential = build_potential(fp)
    wave      = build_wave()

    print("  Running multislice...")
    exit_waves = wave.multislice(potential)

    print("  Computing diffraction patterns...")
    measurement = exit_waves.diffraction_patterns(max_angle=MAX_ANGLE_MRAD)
    measurement = measurement.mean(0)
    measurement.compute()

    return measurement


def simulate_diffraction_trajectory(frames):
    """Run simulate_diffraction_single_frame for every frame."""
    results = []
    for i, frame in enumerate(frames):
        print(f"\nFrame {i + 1}/{len(frames)}")
        results.append(simulate_diffraction_single_frame(frame))
    return results


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

def show_diffraction_pattern(measurement, frame_index=0):
    measurement.block_direct().show(
        explode          = True,
        figsize          = (8, 8),
        cbar             = True,
        common_color_scale = True,
        title            = f"Diffraction pattern — Frame {frame_index}",
    )
    plt.show()


def save_diffraction_frames(measurements,
                             output_folder="diffraction_frames"):
    os.makedirs(output_folder, exist_ok=True)
    for i, m in enumerate(measurements):
        print(f"Saving frame {i + 1}/{len(measurements)}")
        m.block_direct().show(
            explode          = True,
            figsize          = (8, 8),
            cbar             = True,
            common_color_scale = True,
            title            = f"Diffraction pattern — Frame {i}",
        )
        plt.savefig(
            os.path.join(output_folder, f"frame_{i:04d}.png"),
            dpi=200, bbox_inches="tight",
        )
        plt.close("all")


def create_diffraction_animation(frame_folder="diffraction_frames",
                                  output_file="diffraction_animation.mp4",
                                  fps=10):
    frame_files = sorted(
        os.path.join(frame_folder, f)
        for f in os.listdir(frame_folder)
        if f.endswith(".png")
    )
    print(f"Creating animation from {len(frame_files)} frames...")
    writer = imageio.get_writer(output_file, fps=fps)
    for filename in frame_files:
        writer.append_data(imageio.imread(filename))
    writer.close()
    print(f"Saved: {output_file}")


def animate_atomic_structure(frames, interval=100, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame_idx):
        ax.clear()
        abtem.show_atoms(frames[frame_idx], plane="xy", ax=ax, legend=True)
        ax.set_title(f"Atomic Structure — Frame {frame_idx}")
        return (ax,)

    ani = FuncAnimation(fig, update, frames=len(frames),
                         interval=interval, blit=False)
    if save_path:
        ani.save(save_path, fps=1000 // interval, dpi=200)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    return ani


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from config import TRAJECTORY_FILE

    frames = load_trajectory(TRAJECTORY_FILE)
    print(f"Loaded {len(frames)} frames")

    frames = [process_frame(f) for f in frames]

    measurements = simulate_diffraction_trajectory(frames)

    save_diffraction_frames(measurements, output_folder="diffraction_frames")

    create_diffraction_animation(
        frame_folder = "diffraction_frames",
        output_file  = "diffraction_animation.mp4",
        fps          = 8,
    )
