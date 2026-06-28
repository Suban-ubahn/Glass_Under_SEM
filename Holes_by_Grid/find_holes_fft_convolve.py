import numpy as np
import ase.io

from scipy.fft import fftn, ifftn
from scipy.ndimage import (
    gaussian_filter,
    maximum_filter
    )

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# PARAMETERS
# ============================================================

file_path = "trj.zr50cu50_6750_vib"

# voxel edge size (Å)
voxel_edge = 0.3

# atom sphere radii (Å)
r_Cu = 1.3
r_Zr = 1.6

# Gaussian smoothing for Vloch
gaussian_sigma_A = 1.0

# peak detection thresholds
peak_threshold = 0.1
laplace_threshold = 0.01

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
# CREATE EMPTY 3D GRID
# ============================================================

def create_grid(atoms, voxel_edge):

    cell_lengths = atoms.cell.lengths()

    nx = int(cell_lengths[0] / voxel_edge)
    ny = int(cell_lengths[1] / voxel_edge)
    nz = int(cell_lengths[2] / voxel_edge)

    grid = np.zeros((nx, ny, nz), dtype=np.float32)

    return grid


# ============================================================
# PUT ATOMS INTO GRID (delta peaks)
# ============================================================

def atoms_to_grid(atoms, voxel_edge, atom_type):

    grid = create_grid(atoms, voxel_edge)

    positions = atoms.positions
    types = atoms.arrays['type']

    nx, ny, nz = grid.shape

    for pos, t in zip(positions, types):

        if t != atom_type:
            continue

        ix = int(pos[0] / voxel_edge) % nx
        iy = int(pos[1] / voxel_edge) % ny
        iz = int(pos[2] / voxel_edge) % nz

        grid[ix, iy, iz] += 1.0

    return grid


# ============================================================
# CREATE SPHERE KERNEL
# ============================================================

def create_sphere_kernel(radius_A, voxel_edge):

    r_vox = int(np.ceil(radius_A / voxel_edge)) # number of voxels to cover radius_A

    size = 2 * r_vox + 1 # kernel size to cover full sphere

    kernel = np.zeros((size, size, size), dtype=np.float32) # create empty kernel

    cx = cy = cz = r_vox # center of kernel in voxel coordinates

    for x in range(size):
        for y in range(size):
            for z in range(size): 

                dx = (x - cx) * voxel_edge 
                dy = (y - cy) * voxel_edge
                dz = (z - cz) * voxel_edge

                r = np.sqrt(dx**2 + dy**2 + dz**2)

                if r <= radius_A:
                    kernel[x, y, z] = 1.0

    return kernel


# ============================================================
# PAD KERNEL TO GRID SIZE
# ============================================================

# Pads the kernel to the same shape as the grid, centering it correctly for convolution. 
# This is necessary because FFT convolution assumes the kernel is centered at the origin.

def pad_kernel(kernel, shape):

    padded = np.zeros(shape, dtype=np.float32)

    kx, ky, kz = kernel.shape

    padded[:kx, :ky, :kz] = kernel

    # FFT shift -> center kernel correctly
    padded = np.roll(padded, -kx // 2, axis=0)
    padded = np.roll(padded, -ky // 2, axis=1)
    padded = np.roll(padded, -kz // 2, axis=2)

    return padded


# ============================================================
# FFT CONVOLUTION
# ============================================================

def fft_convolve(volume, kernel):

    F_volume = fftn(volume)
    F_kernel = fftn(kernel)

    result = np.real(ifftn(F_volume * F_kernel))

    return result


# ============================================================
# BUILD OCCUPANCY FIELD Vcell
# ============================================================

# Uses all of the functions above to compute the occupancy field - 
# by convolving the atom grid with the sphere kernel for each atom type, then summing the contributions.

def compute_Vcell(atoms, voxel_edge):

    shape = create_grid(atoms, voxel_edge).shape

    total_field = np.zeros(shape, dtype=np.float32)

    for atom_type, radius in type_to_radius.items():

        atom_grid = atoms_to_grid(
            atoms,
            voxel_edge,
            atom_type
        )

        sphere_kernel = create_sphere_kernel(
            radius,
            voxel_edge
        )
        sphere_kernel = pad_kernel(
            sphere_kernel,
            shape
        )

        field = fft_convolve(
            atom_grid,
            sphere_kernel
        )

        total_field += field

    return total_field


# ============================================================
# BUILD HOLE FIELD Vloch
# ============================================================

def compute_Vloch(Vcell, voxel_edge):

    # invert occupancy
    Vloch = 0.5 - Vcell

    # smooth with Gaussian
    sigma_vox = gaussian_sigma_A / voxel_edge

    Vloch = gaussian_filter(
        Vloch,
        sigma=sigma_vox,
        mode='wrap'
    )

    return Vloch


# ============================================================
# COMPUTE LAPLACIAN
# ============================================================

def compute_laplacian(field):

    lap = (
        -6 * field
        + np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        + np.roll(field, 1, axis=2)
        + np.roll(field, -1, axis=2)
    )

    return lap


# ============================================================
# FIND HOLE PEAKS
# ============================================================


def find_hole_peaks(Vloch):

    # local maxima
    local_max = maximum_filter(Vloch, size=3)

    maxima_mask = (Vloch == local_max)

    # gradient magnitude
    gx, gy, gz = np.gradient(Vloch)

    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # Laplacian
    lap = compute_laplacian(Vloch)

    # peak criteria
    peak_mask = (
        maxima_mask
        & (grad_mag < peak_threshold)
        & (lap < -laplace_threshold)
        & (Vloch > 0)
    )

    peak_indices = np.argwhere(peak_mask)

    peak_values = Vloch[peak_mask]

    return peak_indices, peak_values


# ============================================================
# VOXEL -> REAL POSITION
# ============================================================

def voxel_to_position(indices, voxel_edge):

    return indices * voxel_edge


# ============================================================
# VISUALIZE HOLES
# ============================================================

def visualize_holes(atoms, hole_positions, hole_strengths, frame):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # atoms
    pos = atoms.positions

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        s=1,
        alpha=0.05,
        color='gray'
    )

    # holes
    if len(hole_positions) > 0:

        sizes = 50 * hole_strengths / np.max(hole_strengths)

        sc = ax.scatter(
            hole_positions[:, 0],
            hole_positions[:, 1],
            hole_positions[:, 2],
            c=hole_strengths,
            s=sizes,
            cmap='inferno'
        )

        plt.colorbar(sc, label='Hole strength')

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    plt.title(f"Detected Hole Peaks in Frame {frame}")
    plt.savefig(f"Holes by Grid/hole_peaks_frame{frame}.png", dpi=300)

    plt.show()

# ============================================================
# ANIMATE HOLES OVER TRAJECTORY
# ============================================================

def animate_holes(
    file_path,
    voxel_size=voxel_edge,
    stride=1,
    save_path="hole_animation.mp4"
):

    # --------------------------------
    # Load trajectory
    # --------------------------------
    trajectory = ase.io.read(
        file_path,
        format='lammps-dump-text',
        index=f'::{stride}'
    )

    print(f"Loaded {len(trajectory)} frames")

    # --------------------------------
    # Precompute hole data
    # --------------------------------
    all_data = []

    for t, atoms in enumerate(trajectory):

        print(f"Processing frame {t}")

        Vcell = compute_Vcell(atoms, voxel_size)

        Vloch = compute_Vloch(Vcell, voxel_size)

        peak_indices, peak_values = find_hole_peaks(Vloch)

        hole_positions = voxel_to_position(
            peak_indices,
            voxel_size
        )

        all_data.append((
            atoms.positions.copy(),
            hole_positions,
            peak_values
        ))

    # --------------------------------
    # Figure
    # --------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # fixed box size
    cell = trajectory[0].cell.lengths()

    # --------------------------------
    # Update function
    # --------------------------------
    def update(frame):

        ax.clear()

        atom_pos, hole_pos, hole_strength = all_data[frame]

        # atoms
        ax.scatter(
            atom_pos[:, 0],
            atom_pos[:, 1],
            atom_pos[:, 2],
            s=1,
            color='gray',
            alpha=0.05
        )

        # holes
        if len(hole_pos) > 0:

            sizes = 50 * hole_strength / np.max(hole_strength)

            sc = ax.scatter(
                hole_pos[:, 0],
                hole_pos[:, 1],
                hole_pos[:, 2],
                c=hole_strength,
                s=sizes,
                cmap='inferno',
                alpha=0.9
            )

        ax.set_xlim(0, cell[0])
        ax.set_ylim(0, cell[1])
        ax.set_zlim(0, cell[2])

        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")

        ax.set_title(f"Hole Dynamics — Frame {frame}")

    # --------------------------------
    # Animation
    # --------------------------------
    ani = FuncAnimation(
        fig,
        update,
        frames=len(all_data),
        interval=100
    )

    # --------------------------------
    # Save
    # --------------------------------
    ani.save(
        save_path,
        fps=12,
        dpi=200
    )

    print(f"Saved animation: {save_path}")

# ============================================================
# FULL ANALYSIS
# ============================================================

def analyze_frame(atoms):

    print("Computing Vcell...")
    Vcell = compute_Vcell(atoms, voxel_edge)

    print("Computing Vloch...")
    Vloch = compute_Vloch(Vcell, voxel_edge)

    print("Finding hole peaks...")
    peak_indices, peak_values = find_hole_peaks(Vloch)

    hole_positions = voxel_to_position(
        peak_indices,
        voxel_edge
    )

    return (
        Vcell,
        Vloch,
        hole_positions,
        peak_values
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    frame = 0

    trajectory = ase.io.read(
        file_path,
        format='lammps-dump-text',
        index=str(frame)
    )

    atoms = trajectory

    Vcell, Vloch, hole_positions, hole_strengths = analyze_frame(atoms)

    print(f"Voxel edge: {voxel_edge} Å")
    print("Number of detected holes:", len(hole_positions))
    print("Vcell statistics:")
    print("min =", Vcell.min())
    print("max =", Vcell.max())
    print("mean =", Vcell.mean())
    print("std =", Vcell.std())

    visualize_holes(
        atoms,
        hole_positions,
        hole_strengths,
        frame
    )

    # animate_holes(file_path=file_path, stride=1, save_path="Holes by Grid/hole_dynamics.mp4")