# =============================================================
# analysis/soft_modes/softness_holes_defects.py -- softness against free-volume holes (FFT)
# and structural defects
#
# The softness counterpart of analysis/d2min/comparison.py. It takes ANY per-atom, per-frame
# softness series (T, N): the full-system one (modes.softness_trajectory) or the local-PCA one
# (local_softness.local_softness_trajectory); nothing here depends on how the softness was made.
#
# HOLES (FFT approach only, holes/fft.py). For every frame the hole peaks are detected on that
# frame's own atoms; every atom's distance to the nearest peak is measured (cKDTree on the peak
# positions, without periodic images -- exactly as analysis/d2min/comparison.py) and the atoms of
# the frame are split into SOFTNESS_HOLE_N_BINS groups by the quantiles of that frame's distances
# (3: near / mid / far). Groups are recomputed every frame, since the holes move.
# A frame without a hole peak has group -1 for every atom and is left out of the statistics.
#
# DEFECTS. Counted anew on every frame, like the holes: the graph of graph/build.py is built from that
# frame's own atoms, and the only thing fixed in advance is the edge cutoff (cutoffs.CUTOFF_DICT, the first
# minimum of the g(r) of each species pair, defined once). Coordination numbers -> defect = graph/defects.py
# classify_defects_statistical rule, applied to that frame (coordination more than DEFECT_K standard
# deviations below the mean coordination of the element in that frame). Degree of defect: z = (mean -
# coordination) / std of the atom's element in that frame (std with one degree of freedom less, as torch's
# .std() there); the atoms are grouped by the number of SOFTNESS_DEFECT_K_LEVELS that z exceeds, so z > DEFECT_K
# is the defect classification whenever DEFECT_K is one of the levels. Defect membership and group of an atom
# therefore change from frame to frame.
# =============================================================

import numpy as np
from scipy.spatial import cKDTree

from config import (
    VOXEL_EDGE, DEFECT_K,
    SOFTNESS_HOLE_N_BINS, SOFTNESS_DEFECT_K_LEVELS,
)
from cutoffs import CUTOFF_DICT
from holes.fft import compute_Vcell, compute_Vloch, find_hole_peaks, voxel_to_position


# ------------------------------------------------------------
# Groups
# ------------------------------------------------------------

def hole_proximity_groups(trajectory, voxel_edge=VOXEL_EDGE, n_bins=SOFTNESS_HOLE_N_BINS):
    """
    Group every atom of every frame by its distance to the nearest FFT hole peak (see the file header).

    Returns dict:
      frame_indices : (T,) 0 .. T-1
      group         : (T, N) int, 0 = nearest to a hole ... n_bins-1 = farthest; -1 = no hole peak in that frame
      edges         : (T, n_bins + 1) distance quantile edges (Å) of every frame; NaN without hole peaks
    """
    n_frames, n_atoms = len(trajectory), len(trajectory[0])
    group = np.full((n_frames, n_atoms), -1, dtype=int)
    edges = np.full((n_frames, n_bins + 1), np.nan)

    for t, atoms in enumerate(trajectory):
        Vcell = compute_Vcell(atoms, voxel_edge)
        Vloch = compute_Vloch(Vcell, voxel_edge)
        peak_idx, _ = find_hole_peaks(Vloch)
        hole_pos = voxel_to_position(peak_idx, voxel_edge)
        if len(hole_pos) == 0:
            continue
        dist, _ = cKDTree(hole_pos).query(atoms.positions)
        e = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
        edges[t] = e
        group[t] = np.clip(np.digitize(dist, e[1:-1]), 0, n_bins - 1)

    return {"frame_indices": np.arange(n_frames), "group": group, "edges": edges}


def defect_groups_per_frame(trajectory, cutoff=CUTOFF_DICT, k=DEFECT_K, k_levels=SOFTNESS_DEFECT_K_LEVELS):
    """
    Defect membership and degree of defect of every atom on every frame (see the file header): the graph is
    rebuilt on each frame with the fixed g(r) cutoff, the coordination statistics are those of that frame.

    Returns dict:
      mask     : (T, N) bool, defect (coordination < mean - k * std of the element, per frame)
      z        : (T, N) float, standard deviations below the mean coordination of the atom's element (that frame)
      group    : (T, N) int, number of k_levels that z exceeds (0 ... len(k_levels))
      k_levels : tuple, the levels used
      degrees  : (T, N) int, coordination numbers
    """
    from graph.build import atoms_to_pyg
    from graph.defects import compute_degrees

    n_frames, n_atoms = len(trajectory), len(trajectory[0])
    degrees = np.zeros((n_frames, n_atoms), dtype=int)
    z = np.zeros((n_frames, n_atoms))
    mask = np.zeros((n_frames, n_atoms), dtype=bool)

    for t, atoms in enumerate(trajectory):
        g = atoms_to_pyg(atoms, cutoff)
        deg = compute_degrees(g).numpy()
        Zn = g.x.view(-1).numpy()
        degrees[t] = deg
        for element in np.unique(Zn):
            m = Zn == element
            d = deg[m].astype(float)
            mean = d.mean()
            std = d.std(ddof=1) if m.sum() > 1 else 0.0
            z[t, m] = (mean - d) / std if std > 0 else 0.0
            mask[t, m] = d < mean - k * std

    levels = np.asarray(k_levels, dtype=float)
    group = (z[:, :, None] > levels[None, None, :]).sum(axis=2)
    return {"mask": mask, "z": z, "group": group, "k_levels": tuple(float(x) for x in levels),
            "degrees": degrees}


# ------------------------------------------------------------
# Softness per group
# ------------------------------------------------------------

def _check_series(softness_series, n_frames, n_atoms):
    S = np.asarray(softness_series, dtype=float)
    if S.shape != (n_frames, n_atoms):
        raise ValueError(f"softness_series has shape {S.shape}, expected (frames, atoms) = ({n_frames}, {n_atoms})")
    return S


def _group_mean_per_frame(S, member):
    """Mean over the atoms of each frame that are members and have a finite value; NaN where there are none."""
    ok = member & np.isfinite(S)
    count = ok.sum(axis=1)
    total = np.where(ok, S, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def softness_by_hole_proximity(softness_series, hole_groups):
    """
    Softness per distance-to-nearest-hole group, per frame and pooled over the trajectory.

    softness_series : (T, N) per-atom softness of every frame (NaN ignored)
    hole_groups : hole_proximity_groups()

    Returns dict (same keys as analysis/d2min/comparison.compute_d2min_hole_distance_stats, plus the pooled values):
      frame_indices : (T,)
      bin_means     : (T, n_bins) mean softness per group, column 0 = nearest to the holes
      bin_edges     : (T, n_bins + 1) distance quantile edges (Å), recomputed per frame
      pooled        : list of n_bins 1D arrays -- softness of every atom of the group at every frame
    """
    group = hole_groups["group"]
    S = _check_series(softness_series, *group.shape)
    n_bins = hole_groups["edges"].shape[1] - 1
    bin_means = np.stack([_group_mean_per_frame(S, group == b) for b in range(n_bins)], axis=1)
    pooled = [S[(group == b) & np.isfinite(S)] for b in range(n_bins)]
    return {"frame_indices": hole_groups["frame_indices"], "bin_means": bin_means,
            "bin_edges": hole_groups["edges"], "pooled": pooled}


def softness_by_defect(softness_series, defect_groups, frame_indices=None):
    """
    Softness of defect and non-defect atoms per frame, and pooled over the trajectory by degree of defect.
    Membership of every frame is that frame's own (defect_groups_per_frame()).

    softness_series : (T, N) per-atom softness of every frame (NaN ignored)
    defect_groups : defect_groups_per_frame()
    frame_indices : (T,) frame number of every row (default 0 .. T-1)

    Returns dict:
      frame_indices   : (T,)
      mean_defect, mean_nondefect : (T,) mean softness of the defect / non-defect atoms of each frame
      defect_fraction : (T,) fraction of the atoms that are defects in each frame
      mask            : (T, N) bool defect membership
      k_levels        : the levels of the degree of defect
      group           : (T, N) int degree-of-defect group of every atom in every frame
      group_counts    : (T, n_groups) atoms per group in each frame
      pooled          : list of n_groups 1D arrays -- softness of every atom of the group, every frame it is in it
    """
    mask, group = np.asarray(defect_groups["mask"], dtype=bool), np.asarray(defect_groups["group"])
    S = _check_series(softness_series, *mask.shape)
    n_groups = len(defect_groups["k_levels"]) + 1
    return {
        "frame_indices": np.arange(len(S)) if frame_indices is None else np.asarray(frame_indices),
        "mean_defect": _group_mean_per_frame(S, mask),
        "mean_nondefect": _group_mean_per_frame(S, ~mask),
        "defect_fraction": mask.mean(axis=1),
        "mask": mask,
        "k_levels": tuple(defect_groups["k_levels"]),
        "group": group,
        "group_counts": np.stack([(group == g).sum(axis=1) for g in range(n_groups)], axis=1),
        "pooled": [S[(group == g) & np.isfinite(S)] for g in range(n_groups)],
    }
