# =============================================================
# analysis/projection_maps.py — time-integrated 2-D maps
#
# Combines the two analysis branches (holes + graph defects)
# by projecting both onto the XY plane and accumulating over
# the full trajectory to produce heatmaps of where holes and
# defect clusters are most active.
#
# Source: hole_defect_projection_maps.py (top-level)
# Only change: now imports from the unified modules instead of
# duplicating logic.
# =============================================================


import numpy as np
import matplotlib.pyplot as plt

from holes.fft import compute_Vcell, compute_Vloch
from graph.build import atoms_to_pyg
from graph.defects import (
    compute_degrees,
    compute_degree_deficit,
    classify_defects_from_deficit,
    build_defect_subgraph,
    get_connected_components,   
)
from visualization.plot import plot_map, plot_cross_correlation
from config import VOXEL_EDGE, DEFECT_THRESHOLD
from cutoffs import GRAPH_CUTOFF_DICT


# ------------------------------------------------------------
# Project hole field onto XY plane
# ------------------------------------------------------------

def project_holes(Vloch):
    """
    Sum the 3-D hole-density field along Z → 2-D XY projection.
    """
    return np.sum(Vloch, axis=2)


# ------------------------------------------------------------
# Project defect clusters onto XY plane
# ------------------------------------------------------------

def project_defect_clusters(graph, defects,
                              voxel_edge_xy=VOXEL_EDGE,
                              shape_xy=None,
                              cutoff=GRAPH_CUTOFF_DICT):
    """
    Build the defect subgraph, find connected components, and
    paint each component onto a 2-D grid weighted by cluster size.

    Parameters
    ----------
    shape_xy : (nx, ny) tuple — must match the hole projection grid
    """
    cluster_map = np.zeros(shape_xy, dtype=np.float32)

    subgraph, _ = build_defect_subgraph(graph, defects, cutoff=cutoff)
    if subgraph is None:
        return cluster_map

    pos        = subgraph.pos.cpu().numpy()
    edge_index = subgraph.edge_index.cpu().numpy()
    num_nodes  = len(pos)

    # get_connected_components is now imported from graph.defects
    components = get_connected_components(edge_index, num_nodes)

    for comp in components:
        cluster_size = len(comp)
        for idx in comp:
            x, y, _ = pos[idx]
            ix = int(x / voxel_edge_xy)
            iy = int(y / voxel_edge_xy)
            if 0 <= ix < shape_xy[0] and 0 <= iy < shape_xy[1]:
                cluster_map[ix, iy] += cluster_size

    return cluster_map


# ------------------------------------------------------------
# Main loop — accumulate over trajectory
# ------------------------------------------------------------

def build_cumulative_maps(trajectory,
                           voxel_edge=VOXEL_EDGE,
                           cutoff=GRAPH_CUTOFF_DICT,
                           threshold=DEFECT_THRESHOLD):
    """
    Loop over every frame, compute hole and defect-cluster
    projections, and accumulate into two 2-D heatmaps.

    Returns
    -------
    cumulative_holes    : (nx, ny) float64 ndarray
    cumulative_clusters : (nx, ny) float64 ndarray
    """
    print("Preparing first frame...")
    Vcell0              = compute_Vcell(trajectory[0], voxel_edge)
    hole_projection_shape = Vcell0.shape[:2]

    cumulative_holes    = np.zeros(hole_projection_shape, dtype=np.float64)
    cumulative_clusters = np.zeros(hole_projection_shape, dtype=np.float64)

    for frame_idx, atoms in enumerate(trajectory):
        print(f"Processing frame {frame_idx + 1}/{len(trajectory)}")

        # --- holes ---
        Vcell            = compute_Vcell(atoms, voxel_edge)
        Vloch            = compute_Vloch(Vcell, voxel_edge)
        cumulative_holes += project_holes(Vloch)

        # --- defect clusters ---
        graph   = atoms_to_pyg(atoms, cutoff)
        degrees = compute_degrees(graph)
        Z       = graph.x.view(-1)
        deficit = compute_degree_deficit(degrees, Z)
        defects = classify_defects_from_deficit(
            deficit, degrees, Z,
            threshold=threshold, print_stats=False,
        )
        cumulative_clusters += project_defect_clusters(
            graph, defects,
            voxel_edge_xy = voxel_edge,
            shape_xy      = hole_projection_shape,
            cutoff        = cutoff,
        )

    return cumulative_holes, cumulative_clusters

# ------------------------------------------------------------
# Cross-Correlation Analysis
# ------------------------------------------------------------

def compute_cross_correlation(holes=None, clusters=None):
    from scipy.signal import fftconvolve

    if holes is None:
        holes    = np.load("outputs/projection_maps/cumulative_hole_map.npy")
    if clusters is None:
        clusters = np.load("outputs/projection_maps/cumulative_cluster_map.npy")

    def normalise(m):
        m = m - m.mean()
        return m / m.std()

    xcorr = fftconvolve(normalise(holes), normalise(clusters)[::-1, ::-1], mode="full")

    cx, cy = np.array(xcorr.shape) // 2
    peak   = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    print(f"Cross-correlation peak at shift: dx={peak[0]-cx}, dy={peak[1]-cy} voxels")
    print(f"  = {(peak[0]-cx)*VOXEL_EDGE:.2f} Å, {(peak[1]-cy)*VOXEL_EDGE:.2f} Å")

    return xcorr

def resample_to_match(source, target):
    """
    Resample `source` array to match the shape of `target`
    using bilinear interpolation.

    Used to match the abtem scan grid resolution to the
    voxel grid resolution of the hole/defect maps.
    """
    from scipy.ndimage import zoom

    zoom_x = target.shape[0] / source.shape[0]
    zoom_y = target.shape[1] / source.shape[1]
    return zoom(source, (zoom_x, zoom_y), order=1)


def compare_abtem_to_projection_maps(
    abtem_image,
    holes_path    = "outputs/projection_maps/cumulative_hole_map.npy",
    clusters_path = "outputs/projection_maps/cumulative_cluster_map.npy",
    save_dir      = "analysis",
):
    """
    Cross-correlate the abtem virtual detector heatmap with the
    cumulative hole and defect cluster projection maps.

    Parameters
    ----------
    abtem_image : 2D ndarray — virtual BF/DF image from
                  plot_virtual_detector_heatmap (first return value)
    holes_path, clusters_path : paths to the saved .npy maps
    save_dir    : folder for output plots
    """
    from scipy.signal import fftconvolve

    holes    = np.load(holes_path)
    clusters = np.load(clusters_path)

    # --- resample abtem image to match voxel grid ---
    abtem_resampled = resample_to_match(abtem_image, holes)
    print(f"abtem image shape:    {abtem_image.shape}")
    print(f"Resampled to:         {abtem_resampled.shape}")
    print(f"Hole map shape:       {holes.shape}")
    print(f"Cluster map shape:    {clusters.shape}")

    # --- save resampled abtem image for visual inspection ---
    plot_map(
        abtem_resampled,
        title           = "abtem Virtual Detector (resampled)",
        cmap            = "inferno",
        colorbar_label  = "Intensity (a.u.)",
    )

    # --- normalise all three maps ---
    def normalise(m):
        m = m - m.mean()
        return m / (m.std() + 1e-10)

    a = normalise(abtem_resampled)
    h = normalise(holes)
    c = normalise(clusters)

    # --- cross-correlate abtem vs holes ---
    xcorr_holes = fftconvolve(a, h[::-1, ::-1], mode="full")
    cx, cy = np.array(xcorr_holes.shape) // 2
    peak   = np.unravel_index(np.argmax(xcorr_holes), xcorr_holes.shape)
    print(f"\nabtem × holes peak shift: "
          f"dx={peak[0]-cx}, dy={peak[1]-cy} voxels"
          f" = {(peak[0]-cx)*VOXEL_EDGE:.2f} Å, "
          f"{(peak[1]-cy)*VOXEL_EDGE:.2f} Å")

    plot_map(
        xcorr_holes,
        title          = "Cross-Correlation abtem vs Holes",
        cmap           = "RdBu_r",
        colorbar_label = "Cross-correlation (a.u.)",
    )

    # --- cross-correlate abtem vs defect clusters ---
    xcorr_clusters = fftconvolve(a, c[::-1, ::-1], mode="full")
    peak = np.unravel_index(np.argmax(xcorr_clusters), xcorr_clusters.shape)
    print(f"abtem × clusters peak shift: "
          f"dx={peak[0]-cx}, dy={peak[1]-cy} voxels"
          f" = {(peak[0]-cx)*VOXEL_EDGE:.2f} Å, "
          f"{(peak[1]-cy)*VOXEL_EDGE:.2f} Å")

    plot_map(
        xcorr_clusters,
        title          = "Cross-Correlation abtem vs Defect Clusters",
        cmap           = "RdBu_r",
        colorbar_label = "Cross-correlation (a.u.)",
    )

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    # from data_io.trajectory import load_trajectory
    # from config import TRAJECTORY_FILE

    # print("Loading trajectory...")
    # trajectory = load_trajectory(TRAJECTORY_FILE)
    # print(f"Loaded {len(trajectory)} frames")

    # cumulative_holes, cumulative_clusters = build_cumulative_maps(trajectory)

    # np.save("outputs/projection_maps/cumulative_hole_map.npy",    cumulative_holes)
    # np.save("outputs/projection_maps/cumulative_cluster_map.npy", cumulative_clusters)

    cumulative_holes    = np.load("outputs/projection_maps/cumulative_hole_map.npy")
    cumulative_clusters = np.load("outputs/projection_maps/cumulative_cluster_map.npy")

    plot_map(cumulative_holes,    "Time-Integrated Hole Activity",           colorbar_label="Cumulative free-volume density (a.u.)")
    plot_map(cumulative_clusters, "Time-Integrated Defect Cluster Activity", colorbar_label="Cumulative defect cluster weight (atom·frames)")

    xcorr = compute_cross_correlation()
    plot_cross_correlation(xcorr)