import numpy as np
import ase.io
from scipy.spatial import cKDTree
from scipy.ndimage import label
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# -------------------------------
# Parameters (Zr-Cu metallic glass)
# -------------------------------
# approximate metallic radii (Å)
r_Zr = 1.60
r_Cu = 1.28

# conservative "occupied" cutoff (slightly larger than atoms)
r_cut = 2  # Å (tunable, but reasonable for Zr-Cu)

# grid resolution
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
    tree = cKDTree(all_pos)

    return tree


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

    # print("Min distance:", distances.min())
    # print("Max distance:", distances.max())

    # # Check how many points are near threshold
    # near = np.abs(distances - r_cut) < 0.1
    # print("Points near threshold:", np.sum(near))

    void_mask = distances > r_cut

    void_grid = void_mask.reshape(shape)

    return void_grid, shape


# -------------------------------
# Find connected holes
# -------------------------------
def find_holes(void_grid):
    structure = np.ones((3, 3, 3))  # connectivity
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
        if size > 10:  # filter out tiny holes (noise)  
            volume = size * voxel_volume
            volumes.append(volume)

    return np.array(volumes)

# -------------------------------
# Compute hole locations
# -------------------------------
def visualize_holes_3d(atoms, void_grid, labeled, num_features, spacing, max_holes=5):

    cell = atoms.cell.lengths()

    nx, ny, nz = void_grid.shape

    # Recreate grid coordinates
    x = np.linspace(0, cell[0], nx, endpoint=False)
    y = np.linspace(0, cell[1], ny, endpoint=False)
    z = np.linspace(0, cell[2], nz, endpoint=False)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot atoms (light gray for context)
    pos = atoms.positions
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=1, alpha=0.1, label="Atoms", color='gray')

    # Sort holes by size (largest first)
    hole_sizes = [(i, np.sum(labeled == i)) for i in range(1, num_features + 1)]
    hole_sizes.sort(key=lambda x: x[1], reverse=True)

    # Plot largest holes
    for idx, (hole_id, size) in enumerate(hole_sizes[:max_holes]):
        mask = (labeled == hole_id)

        xs = X[mask]
        ys = Y[mask]
        zs = Z[mask]

        ax.scatter(xs, ys, zs, s=5, label=f"Hole {idx} (size={size})")

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.legend()
    plt.title("Largest Holes (voxel representation)")
    plt.savefig("largest_holes_3d.png", dpi=300)
    plt.show()


# -------------------------------
# Full pipeline for one frame
# -------------------------------
def analyze_frame(atoms):
    void_grid, shape = compute_void_grid(atoms, r_cut, grid_spacing)

    labeled, num_features = find_holes(void_grid)

    volumes = compute_hole_volumes(labeled, num_features, grid_spacing)

    return volumes, void_grid, labeled, num_features


# -------------------------------
# Analyze trajectory
# -------------------------------
def analyze_trajectory(file_path):
    trajectory = ase.io.read(
        file_path,
        format='lammps-dump-text',
        index='::10'
    )

    all_volumes = []

    for t, atoms in enumerate(trajectory):
        volumes, void_grid, labeled, num_features = analyze_frame(atoms)
        all_volumes.append(volumes)

        print(f"Frame {t}: {len(volumes)} holes")

    return all_volumes


# -------------------------------
# Plot distribution
# -------------------------------
def plot_distribution(volumes):

    all_v = np.concatenate(volumes)

    plt.hist(all_v, bins=50)
    plt.xlabel("Hole Volume (Å³)")
    plt.ylabel("Count")
    plt.title("Hole Size Distribution")
    plt.savefig(f"hole_distribution_cutoff{r_cut}Å_spacing{grid_spacing}Å.png", dpi=300)
    plt.show()


# -------------------------------
# Run
# -------------------------------

# For plotting hole size distribution across entire trajectory (all frames:
# if __name__ == "__main__":
#     file_path = "trj.zr50cu50_6750_vib"

#     volumes = analyze_trajectory(file_path)

#     plot_distribution(volumes)

# For visualizing holes in a single frame (e.g. first frame):
if __name__ == "__main__":
    file_path = "trj.zr50cu50_6750_vib"

    trajectory = ase.io.read(
        file_path,
        format='lammps-dump-text',
        index='50'   # just one frame for visualization
    )

    atoms = trajectory

    volumes, void_grid, labeled, num_features = analyze_frame(atoms)

    print(f"Number of holes: {len(volumes)}")

    visualize_holes_3d(atoms, void_grid, labeled, num_features, grid_spacing)