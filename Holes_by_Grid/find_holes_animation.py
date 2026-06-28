import numpy as np
import ase.io
from scipy.spatial import cKDTree
from scipy.ndimage import label
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# -------------------------------
# Parameters (Zr-Cu metallic glass)
# -------------------------------
r_Zr = 1.60
r_Cu = 1.28

r_cut = 2.0   # Å
grid_spacing = 0.3  # Å


# -------------------------------
# Helper: build periodic images
# -------------------------------
def build_pbc_kdtree(atoms):
    pos = atoms.positions
    cell = atoms.cell.array

    shifts = [-1, 0, 1]
    images = []

    for sx in shifts:
        for sy in shifts:
            for sz in shifts:
                shift_vec = sx * cell[0] + sy * cell[1] + sz * cell[2]
                images.append(pos + shift_vec)

    all_pos = np.vstack(images)
    return cKDTree(all_pos)


# -------------------------------
# Generate grid points
# -------------------------------
def generate_grid(atoms, spacing):
    cell = atoms.cell.lengths()

    nx = int(cell[0] / spacing)
    ny = int(cell[1] / spacing)
    nz = int(cell[2] / spacing)

    x = np.linspace(0, cell[0], nx, endpoint=False)
    y = np.linspace(0, cell[1], ny, endpoint=False)
    z = np.linspace(0, cell[2], nz, endpoint=False)

    grid = np.array(np.meshgrid(x, y, z, indexing='ij'))
    points = grid.reshape(3, -1).T

    return points, (nx, ny, nz)


# -------------------------------
# Identify void voxels
# -------------------------------
def compute_void_grid(atoms, r_cut, spacing):
    tree = build_pbc_kdtree(atoms)
    points, shape = generate_grid(atoms, spacing)

    distances, _ = tree.query(points, k=1)
    void_mask = distances > r_cut

    return void_mask.reshape(shape), shape


# -------------------------------
# Find connected holes
# -------------------------------
def find_holes(void_grid):
    structure = np.ones((3, 3, 3))
    labeled, num_features = label(void_grid, structure=structure)
    return labeled, num_features


# -------------------------------
# Compute hole volumes
# -------------------------------
def compute_hole_volumes(labeled, num_features, spacing):
    voxel_volume = spacing ** 3
    volumes = []

    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if size > 10:
            volumes.append(size * voxel_volume)

    return np.array(volumes)


# -------------------------------
# Prepare grid coordinates once
# -------------------------------
def prepare_grid_coords(atoms, spacing):
    cell = atoms.cell.lengths()

    nx = int(cell[0] / spacing)
    ny = int(cell[1] / spacing)
    nz = int(cell[2] / spacing)

    x = np.linspace(0, cell[0], nx, endpoint=False)
    y = np.linspace(0, cell[1], ny, endpoint=False)
    z = np.linspace(0, cell[2], nz, endpoint=False)

    return np.meshgrid(x, y, z, indexing='ij')


# -------------------------------
# Extract largest holes
# -------------------------------
def get_largest_holes(labeled, num_features, X, Y, Z, max_holes=3, min_size=10):
    hole_sizes = [(i, np.sum(labeled == i)) for i in range(1, num_features + 1)]
    hole_sizes.sort(key=lambda x: x[1], reverse=True)

    hole_points = []

    for hole_id, size in hole_sizes:
        if size < min_size:
            continue

        mask = (labeled == hole_id)
        xs = X[mask]
        ys = Y[mask]
        zs = Z[mask]

        hole_points.append((xs, ys, zs))

        if len(hole_points) >= max_holes:
            break

    return hole_points


# -------------------------------
# Full pipeline for one frame
# -------------------------------
def analyze_frame(atoms):
    void_grid, shape = compute_void_grid(atoms, r_cut, grid_spacing)
    labeled, num_features = find_holes(void_grid)
    volumes = compute_hole_volumes(labeled, num_features, grid_spacing)

    return volumes, void_grid, labeled, num_features


# -------------------------------
# Create 3D animation
# -------------------------------
def create_animation(file_path):
    trajectory = ase.io.read(
        file_path,
        format='lammps-dump-text',
        index='::1'
    )

    # Precompute grid coordinates
    X, Y, Z = prepare_grid_coords(trajectory[0], grid_spacing)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")

        atoms = trajectory[frame]

        void_grid, _ = compute_void_grid(atoms, r_cut, grid_spacing)
        labeled, num_features = find_holes(void_grid)

        # Plot atoms
        pos = atoms.positions
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   s=1, alpha=0.05, color='gray', label="Atom centers")

        # Plot holes
        hole_points = get_largest_holes(labeled, num_features, X, Y, Z)

        for xs, ys, zs in hole_points:
            ax.scatter(xs, ys, zs, s=5)

        ax.set_xlim(0, atoms.cell.lengths()[0])
        ax.set_ylim(0, atoms.cell.lengths()[1])
        ax.set_zlim(0, atoms.cell.lengths()[2])

        ax.view_init(elev=30, azim=45)  # fixed camera
        ax.set_title(f"Frame {frame}")

        return []

    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=200)

    ani.save("holes_3d.mp4", dpi=200, fps=12)

    plt.show()


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    file_path = "trj.zr50cu50_6750_vib"

    create_animation(file_path)