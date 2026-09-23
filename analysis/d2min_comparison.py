import numpy as np

from graph.build import atoms_to_pyg
from graph.defects import compute_degrees, classify_defects_statistical
from dynamics.flow_units import d2min_trajectory

from scipy.spatial import cKDTree
from holes.fft import compute_Vcell, compute_Vloch, find_hole_peaks, voxel_to_position
from config import CUTOFF_DICT, DEFECT_K, VOXEL_EDGE

def compute_d2min_defect_stats(trajectory, cutoff=CUTOFF_DICT, k=DEFECT_K):
    """
    Split D^2_min (relative to frame 0) into defect-atom and non-defect-
    atom populations, using a FIXED defect mask computed once from
    frame 0 — so any change in the two populations' d2min over time
    reflects dynamics, not a shifting defect definition.
    """
    d2min_series, frame_indices = d2min_trajectory(
        trajectory, t0=0, frame_step=1, cutoff=cutoff,
    )

    atoms0 = trajectory[0]
    g0 = atoms_to_pyg(atoms0, cutoff)
    degrees0 = compute_degrees(g0)
    Z0 = g0.x.view(-1)
    mask = np.asarray(classify_defects_statistical(degrees0, Z0, k=k))

    mean_defect, mean_nondefect = [], []
    median_defect, median_nondefect = [], []

    for i in range(len(frame_indices)):
        d2min = d2min_series[i]
        mean_defect.append(np.nanmean(d2min[mask]) if mask.any() else np.nan)
        mean_nondefect.append(np.nanmean(d2min[~mask]) if (~mask).any() else np.nan)
        median_defect.append(np.nanmedian(d2min[mask]) if mask.any() else np.nan)
        median_nondefect.append(np.nanmedian(d2min[~mask]) if (~mask).any() else np.nan)

    return {
        "frame_indices": np.array(frame_indices),
        "mean_defect": np.array(mean_defect),
        "mean_nondefect": np.array(mean_nondefect),
        "median_defect": np.array(median_defect),
        "median_nondefect": np.array(median_nondefect),
        "defect_fraction": mask.mean(),  # now a single fixed number, not a series
        "mask": mask,
    }

def compute_d2min_hole_distance_stats(trajectory, cutoff=CUTOFF_DICT,
                                       voxel_edge=VOXEL_EDGE, n_bins=3):
    """
    For every frame, redetect FFT hole peaks (holes/fft.py) on that
    frame's own density field, compute each atom's distance to the
    nearest current hole peak, and bin atoms into n_bins groups by
    that distance (tertiles by default: near/mid/far). Tracks mean
    D^2_min per distance group over time.
 
    Bin membership is recomputed every frame since hole positions move
    — this tests whether d2min tracks proximity to wherever holes
    currently are, not proximity to a fixed initial layout. Since the
    quantile edges are recomputed per frame too, each bin's distance
    range drifts over the trajectory; bin_edges is returned so callers
    can report e.g. an averaged range per bin.
    """
    d2min_series, frame_indices = d2min_trajectory(
        trajectory, t0=0, frame_step=1, cutoff=cutoff,
    )
 
    n_frames = len(frame_indices)
    bin_means = np.full((n_frames, n_bins), np.nan)
    bin_edges = np.full((n_frames, n_bins + 1), np.nan)
 
    for i, frame in enumerate(frame_indices):
        atoms = trajectory[frame]
        d2min = d2min_series[i]
 
        Vcell = compute_Vcell(atoms, voxel_edge)
        Vloch = compute_Vloch(Vcell, voxel_edge)
        peak_idx, peak_v = find_hole_peaks(Vloch)
        hole_pos = voxel_to_position(peak_idx, voxel_edge)
 
        if len(hole_pos) == 0:
            continue
 
        tree = cKDTree(hole_pos)
        dist, _ = tree.query(atoms.positions)
 
        edges = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
        bin_edges[i] = edges
        bin_idx = np.clip(np.digitize(dist, edges[1:-1]), 0, n_bins - 1)
 
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                bin_means[i, b] = np.nanmean(d2min[mask])
 
    return {
        "frame_indices": np.array(frame_indices),
        "bin_means": bin_means,  # shape (n_frames, n_bins); column 0 = nearest to holes
        "bin_edges": bin_edges,  # shape (n_frames, n_bins+1), Å; quantile edges, recomputed per frame
    }