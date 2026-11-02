# =============================================================
# holes/kdtree.py — KDTree / voxel hole detection
#
# Detects free-volume holes by building a 3-D voxel grid,
# marking voxels that are farther than r_cut from every atom
# as "void", then labelling connected void regions.
#
# Previously split across:
#   Holes_by_Grid/find_holes_voxel.py
#   Holes_by_Grid/find_holes_animation.py
# Those two files shared five identical functions — they are
# now unified here.
# =============================================================


import numpy as np
from utils import get_output_dir
OUTPUT_DIR = get_output_dir()
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.spatial import cKDTree
from scipy.ndimage import label

from config import R_CUT_VOID, VOXEL_EDGE


# ------------------------------------------------------------
# Periodic boundary conditions — KDTree over all image cells
# ------------------------------------------------------------

def build_pbc_kdtree(atoms):
    """
    Build a KDTree that includes 26 periodic image cells so
    that nearest-neighbour queries near box boundaries work
    correctly.
    """
    pos  = atoms.positions
    cell = atoms.cell.array

    images = []
    for sx in [-1, 0, 1]:
        for sy in [-1, 0, 1]:
            for sz in [-1, 0, 1]:
                shift = sx * cell[0] + sy * cell[1] + sz * cell[2]
                images.append(pos + shift)

    return cKDTree(np.vstack(images))


# ------------------------------------------------------------
# 3-D voxel grid
# ------------------------------------------------------------

def generate_grid(atoms, spacing=VOXEL_EDGE):
    """
    Generate a uniform grid of sample points that fills the
    simulation cell.

    Returns
    -------
    points : (N, 3) ndarray
        Cartesian coordinates of every grid point.
    shape : (nx, ny, nz) tuple
        Grid dimensions.
    """
    cell = atoms.cell.lengths()

    nx = int(cell[0] / spacing)
    ny = int(cell[1] / spacing)
    nz = int(cell[2] / spacing)

    x = np.linspace(0, cell[0], nx, endpoint=False)
    y = np.linspace(0, cell[1], ny, endpoint=False)
    z = np.linspace(0, cell[2], nz, endpoint=False)

    grid   = np.array(np.meshgrid(x, y, z, indexing="ij"))
    points = grid.reshape(3, -1).T

    return points, (nx, ny, nz)


# ------------------------------------------------------------
# Void / occupied classification
# ------------------------------------------------------------

def compute_void_grid(atoms, r_cut=R_CUT_VOID, spacing=VOXEL_EDGE):
    """
    Mark each voxel as void (True) if its nearest atom centre
    is farther than r_cut.

    Returns
    -------
    void_grid : boolean ndarray, shape (nx, ny, nz)
    shape     : (nx, ny, nz) tuple
    """
    tree   = build_pbc_kdtree(atoms)
    points, shape = generate_grid(atoms, spacing)

    distances, _ = tree.query(points, k=1)
    void_mask    = distances > r_cut

    return void_mask.reshape(shape), shape


# ------------------------------------------------------------
# Connected-component labelling
# ------------------------------------------------------------

def find_holes(void_grid):
    """
    Label connected void regions using full 26-connectivity.

    Returns
    -------
    labeled      : int ndarray, same shape as void_grid
    num_features : int — number of distinct holes found
    """
    structure = np.ones((3, 3, 3))   # 26-connectivity
    labeled, num_features = label(void_grid, structure=structure)
    return labeled, num_features


# ------------------------------------------------------------
# Hole volumes
# ------------------------------------------------------------

def compute_hole_volumes(labeled, num_features, spacing=VOXEL_EDGE):
    """
    Compute the volume (Å³) of each labelled hole.
    Holes smaller than 10 voxels are discarded as noise.

    Returns
    -------
    volumes : 1-D ndarray of float
    """
    voxel_volume = spacing ** 3
    volumes = []

    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if size > 10:
            volumes.append(size * voxel_volume)

    return np.array(volumes)


# ------------------------------------------------------------
# Single-frame pipeline
# ------------------------------------------------------------

def analyze_frame(atoms, r_cut=R_CUT_VOID, spacing=VOXEL_EDGE):
    """
    Run the full KDTree hole-detection pipeline on one frame.

    Returns
    -------
    volumes      : 1-D ndarray — hole volumes (Å³)
    void_grid    : boolean ndarray
    labeled      : int ndarray
    num_features : int
    """
    void_grid, shape   = compute_void_grid(atoms, r_cut, spacing)
    labeled, num_features = find_holes(void_grid)
    volumes            = compute_hole_volumes(labeled, num_features, spacing)
    return volumes, void_grid, labeled, num_features


# ------------------------------------------------------------
# Grid coordinate helper  (used by animation)
# ------------------------------------------------------------

def get_grid_coords(atoms, spacing=VOXEL_EDGE):
    """
    Return meshgrid coordinate arrays (X, Y, Z) matching the
    voxel grid for the given atoms object.
    """
    cell = atoms.cell.lengths()

    nx = int(cell[0] / spacing)
    ny = int(cell[1] / spacing)
    nz = int(cell[2] / spacing)

    x = np.linspace(0, cell[0], nx, endpoint=False)
    y = np.linspace(0, cell[1], ny, endpoint=False)
    z = np.linspace(0, cell[2], nz, endpoint=False)

    return np.meshgrid(x, y, z, indexing="ij")


def get_largest_holes(labeled, num_features, X, Y, Z,
                      max_holes=3, min_size=10):
    """
    Return voxel coordinates for the N largest holes.

    Returns
    -------
    list of (xs, ys, zs) tuples
    """
    hole_sizes = [
        (i, np.sum(labeled == i))
        for i in range(1, num_features + 1)
    ]
    hole_sizes.sort(key=lambda x: x[1], reverse=True)

    hole_points = []
    for hole_id, size in hole_sizes:
        if size < min_size:
            continue
        mask = (labeled == hole_id)
        hole_points.append((X[mask], Y[mask], Z[mask]))
        if len(hole_points) >= max_holes:
            break

    return hole_points
