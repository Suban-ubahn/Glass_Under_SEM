import ase.io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import abtem
import os
import imageio.v2 as imageio


# ============================================================
# PARAMETERS
# ============================================================

beam_energy_kV = 200
max_angle_mrad = 80

gpts = 512
slice_thickness = 2

# frozen phonon settings
num_phonon_configs = 8
phonon_sigma = 0.08  # Å

# atom sphere radii (Å)
r_Cu = 1.3
r_Zr = 1.6

# atom types
type_to_radius = {
    1: r_Zr,
    2: r_Cu
}

# Zr=40, Cu=29
type_to_Z = {
    1: 40,
    2: 29
}


# ============================================================
# PLOTTING
# ============================================================

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['image.interpolation'] = 'bilinear'
plt.rcParams['image.cmap'] = 'inferno'


# ============================================================
# LOAD TRAJECTORY
# ============================================================

def load_atoms(frame_index="::1",
               file_path="trj.zr50cu50_6750_vib"):

    atoms = ase.io.read(
        file_path,
        format="lammps-dump-text",
        index=frame_index
    )

    return atoms


# ============================================================
# CONVERT TYPES
# ============================================================

def correct_atom_types(atoms):

    for atom in atoms:
        atom.number = type_to_Z[atom.number]

    return atoms


# ============================================================
# ADD RADII
# ============================================================

def add_radii(atoms):

    atoms.radii = np.array(
        [type_to_radius[a.number] for a in atoms]
    )

    return atoms


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(frame):

    frame = add_radii(frame)

    frame = correct_atom_types(frame)

    return frame


# ============================================================
# BUILD FROZEN PHONONS
# ============================================================

def build_frozen_phonons(frame,
                         sigma=phonon_sigma,
                         num_configs=num_phonon_configs):

    frozen_phonons = abtem.FrozenPhonons(
        frame,
        num_configs=num_configs,
        sigmas=sigma
    )

    return frozen_phonons


# ============================================================
# BUILD POTENTIAL
# ============================================================

def build_potential(frozen_phonons):

    potential = abtem.Potential(
        frozen_phonons,
        gpts=gpts,
        projection="infinite",
        slice_thickness=slice_thickness
    )

    return potential


# ============================================================
# BUILD ELECTRON WAVE
# ============================================================

def build_wave():

    wave = abtem.PlaneWave(
        energy=beam_energy_kV * 1e3
    )

    return wave


# ============================================================
# SIMULATE DIFFRACTION FOR ONE FRAME
# ============================================================

def simulate_diffraction_single_frame(frame):

    print("Building frozen phonons...")

    frozen_phonons = build_frozen_phonons(frame)

    print("Building potential...")

    potential = build_potential(frozen_phonons)

    print("Building electron wave...")

    wave = build_wave()

    print("Running multislice...")

    exit_waves = wave.multislice(potential)

    print("Computing diffraction ensemble...")

    measurement_ensemble = exit_waves.diffraction_patterns(
        max_angle=max_angle_mrad
    )

    print("Ensemble shape:")
    print(measurement_ensemble.shape)

    print("Averaging phonon configurations...")

    measurement = measurement_ensemble.mean(0)

    print("Computing diffraction image...")

    measurement.compute()

    return measurement


# ============================================================
# SIMULATE ENTIRE TRAJECTORY
# ============================================================

def simulate_diffraction_trajectory(frames):

    results = []

    for i, frame in enumerate(frames):

        print(f"Simulating diffraction \nFRAME {i+1}/{len(frames)}...")

        measurement = simulate_diffraction_single_frame(frame)

        results.append(measurement)

    return results


# ============================================================
# SHOW DIFFRACTION PATTERN
# ============================================================

def show_diffraction_pattern(measurement,
                             frame_index=0):

    measurement.block_direct().show(
        explode=True,
        figsize=(8, 8),
        cbar=True,
        common_color_scale=True,
        title=f"Diffraction pattern — Frame {frame_index}"
    )

    plt.show()


# ============================================================
# ANIMATE ATOMIC STRUCTURE USING ABTEM
# ============================================================

def animate_atomic_structure(frames,
                             interval=100,
                             save_path=None):

    fig, ax = plt.subplots(figsize=(7, 7))

    def update(frame_idx):

        ax.clear()

        abtem.show_atoms(
            frames[frame_idx],
            plane="xy",
            ax=ax,
            legend=True
        )

        ax.set_title(
            f"Atomic Structure — Frame {frame_idx}"
        )

        return ax,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        blit=False
    )

    if save_path is not None:

        ani.save(
            save_path,
            fps=1000 // interval,
            dpi=200
        )

        print(f"Saved atomic animation to: {save_path}")

    else:
        plt.show()

    return ani

# ============================================================
# SAVE DIFFRACTION FRAMES USING ABTEM'S OWN SHOW()
# ============================================================

def save_diffraction_frames(measurements,
                            output_folder="diffraction_frames"):

    os.makedirs(output_folder, exist_ok=True)

    for i, measurement in enumerate(measurements):

        print(f"Saving diffraction frame {i+1}/{len(measurements)}")

        fig = plt.figure(figsize=(8, 8))

        measurement.block_direct().show(
            explode=True,
            figsize=(8, 8),
            cbar=True,
            common_color_scale=True,
            title=f"Diffraction pattern — Frame {i}"
        )

        filename = os.path.join(
            output_folder,
            f"frame_{i:04d}.png"
        )

        plt.savefig(
            filename,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)
        plt.close()


# ============================================================
# CREATE MP4 FROM SAVED PNG FRAMES
# ============================================================

def create_diffraction_animation(frame_folder="Diffraction_Ptychography/diffraction_frames",
                                 output_file="Diffraction_Ptychography/diffraction_animation.mp4",
                                 fps=10):

    frame_files = sorted([
        os.path.join(frame_folder, f)
        for f in os.listdir(frame_folder)
        if f.endswith(".png")
    ])

    print(f"Creating animation from {len(frame_files)} frames...")

    writer = imageio.get_writer(
        output_file,
        fps=fps
    )

    for filename in frame_files:

        image = imageio.imread(filename)

        writer.append_data(image)

    writer.close()

    print(f"Saved animation to: {output_file}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    file_path = "trj.zr50cu50_6750_vib"

    # load ALL frames
    frames = load_atoms(
        frame_index="::1",
        file_path=file_path
    )

    print(f"Loaded {len(frames)} frames")

    # preprocess frames
    frames = [process_frame(f) for f in frames]

    print(f"Processed {len(frames)} frames for atom types and radii")

    # # simulate only ONE frame first
    # example_frame_index = 30

    # measurement = simulate_diffraction_single_frame(
    #     frames[example_frame_index]
    # )

    # # Show the atomic structure for example frame 
    # abtem.show_atoms(frames[example_frame_index], plane="xy", legend=True)

    # # visualize
    # show_diffraction_pattern(
    #     measurement,
    #     frame_index=example_frame_index
    # )

    # --------------------------------------------------------
    # ATOMIC STRUCTURE ANIMATION
    # --------------------------------------------------------

    # animate_atomic_structure(
    #     frames,
    #     interval=80,
    #     save_path="atomic_structure_animation.mp4"
    # )

    # --------------------------------------------------------
    # DIFFRACTION SIMULATIONS
    # --------------------------------------------------------

    measurements = simulate_diffraction_trajectory(frames)

    # --------------------------------------------------------
    # SAVE INDIVIDUAL DIFFRACTION PNGs
    # --------------------------------------------------------

    save_diffraction_frames(
    measurements,
    output_folder="diffraction_frames"
    )

    # --------------------------------------------------------
    # CREATE MP4 ANIMATION FROM SAVED PNGs
    # --------------------------------------------------------

    create_diffraction_animation(
    frame_folder="diffraction_frames",
    output_file="diffraction_animation.mp4",
    fps=8
    )