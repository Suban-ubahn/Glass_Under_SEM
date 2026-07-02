# =============================================================
# holes/fft.py — FFT-convolution occupancy & hole field
#
# Builds a 3-D occupancy field (Vcell) by convolving atom
# positions with element-specific sphere kernels via FFT, then
# inverts it to get a hole-density field (Vloch).
#
# Source: Holes_by_Grid/find_holes_fft_convolve.py
# =============================================================


import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import gaussian_filter, maximum_filter

from config import (
    TYPE_TO_RADIUS,
    VOXEL_EDGE,
    GAUSSIAN_SIGMA_A,
    PEAK_THRESHOLD,
    LAPLACE_THRESHOLD,
)


# ------------------------------------------------------------
# Grid initialisation
# ------------------------------------------------------------

def create_grid(atoms, voxel_edge=VOXEL_EDGE):
    """Empty 3-D float32 grid matching the simulation cell."""
    cell_lengths = atoms.cell.lengths()
    nx = int(cell_lengths[0] / voxel_edge)
    ny = int(cell_lengths[1] / voxel_edge)
    nz = int(cell_lengths[2] / voxel_edge)
    return np.zeros((nx, ny, nz), dtype=np.float32)


# ------------------------------------------------------------
# Atom → grid (delta peaks)
# ------------------------------------------------------------

def atoms_to_grid(atoms, voxel_edge=VOXEL_EDGE, atom_type=None):
    """
    Place each atom of the given LAMMPS type as a unit delta
    peak on the voxel grid (with periodic wrapping).
    """
    grid   = create_grid(atoms, voxel_edge)
    nx, ny, nz = grid.shape

    positions = atoms.positions
    types     = atoms.arrays["type"]

    for pos, t in zip(positions, types):
        if atom_type is not None and t != atom_type:
            continue
        ix = int(pos[0] / voxel_edge) % nx
        iy = int(pos[1] / voxel_edge) % ny
        iz = int(pos[2] / voxel_edge) % nz
        grid[ix, iy, iz] += 1.0

    return grid


# ------------------------------------------------------------
# Sphere kernel
# ------------------------------------------------------------

def create_sphere_kernel(radius_A, voxel_edge=VOXEL_EDGE):
    """
    Binary sphere kernel: voxels within radius_A of the centre
    are 1, outside are 0.
    """
    r_vox = int(np.ceil(radius_A / voxel_edge))
    size  = 2 * r_vox + 1
    kernel = np.zeros((size, size, size), dtype=np.float32)
    cx = cy = cz = r_vox

    for x in range(size):
        for y in range(size):
            for z in range(size):
                dx = (x - cx) * voxel_edge
                dy = (y - cy) * voxel_edge
                dz = (z - cz) * voxel_edge
                if np.sqrt(dx**2 + dy**2 + dz**2) <= radius_A:
                    kernel[x, y, z] = 1.0

    return kernel


def pad_kernel(kernel, shape):
    """
    Pad kernel to the full grid shape and FFT-shift it so the
    kernel centre sits at the array origin — required for
    circular convolution via FFT to behave as a real-space
    convolution.
    """
    padded = np.zeros(shape, dtype=np.float32)
    kx, ky, kz = kernel.shape
    padded[:kx, :ky, :kz] = kernel
    padded = np.roll(padded, -kx // 2, axis=0)
    padded = np.roll(padded, -ky // 2, axis=1)
    padded = np.roll(padded, -kz // 2, axis=2)
    return padded


# ------------------------------------------------------------
# FFT convolution
# ------------------------------------------------------------

def fft_convolve(volume, kernel):
    return np.real(ifftn(fftn(volume) * fftn(kernel)))


# ------------------------------------------------------------
# Occupancy field  Vcell
# ------------------------------------------------------------

def compute_Vcell(atoms, voxel_edge=VOXEL_EDGE,
                  type_to_radius=TYPE_TO_RADIUS):
    """
    Build the occupancy field by convolving each element's
    atom grid with its sphere kernel and summing.

    Vcell[i,j,k] ≈ number of atom spheres overlapping voxel (i,j,k).
    Values > 0 → occupied; 0 → free space.
    """
    shape       = create_grid(atoms, voxel_edge).shape
    total_field = np.zeros(shape, dtype=np.float32)

    for atom_type, radius in type_to_radius.items():
        atom_grid = atoms_to_grid(atoms, voxel_edge, atom_type)
        kernel    = pad_kernel(create_sphere_kernel(radius, voxel_edge), shape)
        total_field += fft_convolve(atom_grid, kernel)

    return total_field


# ------------------------------------------------------------
# Hole field  Vloch
# ------------------------------------------------------------

def compute_Vloch(Vcell, voxel_edge=VOXEL_EDGE,
                  sigma_A=GAUSSIAN_SIGMA_A):
    """
    Derive the hole-density field from the occupancy field.

    Steps:
      1. Invert: Vloch = 0.5 - Vcell  (positive → free volume)
      2. Smooth with a Gaussian to remove single-voxel noise.
    """
    Vloch      = 0.5 - Vcell
    sigma_vox  = sigma_A / voxel_edge
    Vloch      = gaussian_filter(Vloch, sigma=sigma_vox, mode="wrap")
    return Vloch


# ------------------------------------------------------------
# Laplacian  (finite differences, periodic)
# ------------------------------------------------------------

def compute_laplacian(field):
    """
    Second-order finite-difference Laplacian with periodic BCs.
    Negative Laplacian at a local maximum confirms it is a true
    peak (concave down in all directions).
    """
    return (
        -6 * field
        + np.roll(field,  1, axis=0) + np.roll(field, -1, axis=0)
        + np.roll(field,  1, axis=1) + np.roll(field, -1, axis=1)
        + np.roll(field,  1, axis=2) + np.roll(field, -1, axis=2)
    )


# ------------------------------------------------------------
# Peak detection
# ------------------------------------------------------------

def find_hole_peaks(Vloch,
                    peak_threshold=PEAK_THRESHOLD,
                    laplace_threshold=LAPLACE_THRESHOLD):
    """
    Locate hole centres as local maxima of Vloch that satisfy:
      - local maximum within a 3x3x3 neighbourhood
      - gradient magnitude below peak_threshold
      - Laplacian below -laplace_threshold  (concave peak)
      - Vloch > 0  (actually free volume, not occupied)

    Returns
    -------
    peak_indices : (N, 3) int ndarray — voxel indices
    peak_values  : (N,) float ndarray — Vloch value at each peak
    """
    local_max   = maximum_filter(Vloch, size=3)
    maxima_mask = (Vloch == local_max)

    gx, gy, gz = np.gradient(Vloch)
    grad_mag    = np.sqrt(gx**2 + gy**2 + gz**2)
    lap         = compute_laplacian(Vloch)

    peak_mask = (
        maxima_mask
        & (grad_mag < peak_threshold)
        & (lap < -laplace_threshold)
        & (Vloch > 0)
    )

    return np.argwhere(peak_mask), Vloch[peak_mask]


# ------------------------------------------------------------
# Voxel → Cartesian position
# ------------------------------------------------------------

def voxel_to_position(indices, voxel_edge=VOXEL_EDGE):
    return indices * voxel_edge


# ------------------------------------------------------------
# Single-frame pipeline
# ------------------------------------------------------------

def analyze_frame(atoms, voxel_edge=VOXEL_EDGE):
    """
    Full FFT hole-detection pipeline for one ASE Atoms object.

    Returns
    -------
    Vcell          : 3-D float32 ndarray — occupancy field
    Vloch          : 3-D float32 ndarray — hole-density field
    hole_positions : (N, 3) float ndarray — Cartesian (Å)
    peak_values    : (N,) float ndarray
    """
    Vcell              = compute_Vcell(atoms, voxel_edge)
    Vloch              = compute_Vloch(Vcell, voxel_edge)
    peak_indices, peak_values = find_hole_peaks(Vloch)
    hole_positions     = voxel_to_position(peak_indices, voxel_edge)
    return Vcell, Vloch, hole_positions, peak_values


# ------------------------------------------------------------
# Main — single-frame demo
# ------------------------------------------------------------

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
