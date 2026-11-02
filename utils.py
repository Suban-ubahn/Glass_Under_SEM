# =============================================================
# utils.py
#
# Shared infrastructure helpers (not physical constants — see config.py).
#
# Deliberately has NO dependency on config.py or data_io/trajectory.py
# (both of which import config) — that keeps this module a "leaf" with
# nothing importing it back, so config.py can safely import from here.
# =============================================================

import os
import sys
import json

import numpy as np
from ase.neighborlist import neighbor_list

OUTPUTS_ROOT = "outputs"


def unwrap_positions(positions, box):
    """
    Remove periodic-boundary "teleports" from a position trajectory by
    accumulating minimum-image-corrected frame-to-frame displacements
    — the standard MD unwrap procedure.

    This project's LAMMPS dumps re-wrap coordinates into the primary
    cell every frame (same issue flagged in
    dynamics.flow_units.compute_d2min's docstring). Left uncorrected,
    a SINGLE atom crossing a periodic boundary at any point in the
    trajectory contributes a spurious ~(box length)^2 term to its
    displacement covariance — many orders of magnitude larger than a
    real ~0.01-0.1 Å^2 thermal vibration — and dominates the entire
    quasi-harmonic mode decomposition (confirmed on this project's own
    trj.zr50cu50_6750_vib: a handful of such wraps out of >2e6
    atom-component-frames inflated the softest eigenvalue by roughly
    7 orders of magnitude before this fix).

    Parameters
    ----------
    positions : (T, N, D) array — raw (PBC-wrapped) positions.
    box : (D,) or (T, D) array — orthogonal box lengths (same
        assumption as dynamics.flow_units.compute_d2min).

    Returns
    -------
    unwrapped : (T, N, D) array — continuous positions.
    """
    positions = np.asarray(positions, dtype=float)
    box = np.asarray(box, dtype=float)
    if box.ndim == 1:
        box = np.tile(box, (positions.shape[0], 1))

    unwrapped = np.empty_like(positions)
    unwrapped[0] = positions[0]
    for t in range(1, positions.shape[0]):
        raw = positions[t] - positions[t - 1]
        b = box[t]
        correction = b * np.round(raw / b)
        unwrapped[t] = unwrapped[t - 1] + (raw - correction)
    return unwrapped


def compute_displacements(positions, reference=None):
    """
    positions : (T, N, D) array — ALREADY-UNWRAPPED atomic positions
                over T frames, N atoms, D=2 or D=3 spatial dimensions
                (see unwrap_positions() — call it yourself first if
                you're passing in raw trajectory positions). Frames
                must be topologically aligned (same atom order every
                frame).
    reference : (N, D) array, optional — subtracted from every
                frame. Defaults to the time-average position per
                atom, matching CAVIAR's own normalization
                (u = R - mean(R), their Eq. 1) — see
                analysis/soft_modes/modes.py's module docstring.

    Returns
    -------
    u : (T, N, D) array — centered displacements.
    reference : (N, D) array — the reference actually used.
    """
    positions = np.asarray(positions, dtype=float)
    if reference is None:
        reference = positions.mean(axis=0)
    return positions - reference[None, :, :], reference


def get_output_dir(name=None):
    """
    Return outputs/<name>/, creating it if it doesn't exist.

    If `name` is omitted, it's inferred from the currently running
    script's filename — e.g. `python analysis/pypty_comparison.py`
    resolves to outputs/pypty_comparison/. Pass `name` explicitly to
    override (useful when called from a notebook or REPL, where
    sys.argv[0] isn't meaningful).
    """
    if name is None:
        name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    path = os.path.join(OUTPUTS_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path


# =============================================================
# Pair correlation g(r) — used to pick a physically motivated
# neighbor cutoff radius, instead of a hand-picked value.
# =============================================================

def compute_pair_correlation(atoms, r_max=8.0, dr=0.05):
    """
    Compute g(r) for a single ASE Atoms configuration with PBC.

    Parameters
    ----------
    atoms : ase.Atoms
    r_max : float — max radius to histogram out to (Å)
    dr : float — bin width (Å)

    Returns
    -------
    r   : (n_bins,) np.ndarray — bin centers (Å)
    g_r : (n_bins,) np.ndarray — pair correlation function
    """
    i_idx, j_idx, d = neighbor_list("ijd", atoms, r_max)

    n_atoms = len(atoms)
    volume = atoms.get_volume()
    rho = n_atoms / volume  # number density

    bins = np.arange(0, r_max + dr, dr)
    counts, edges = np.histogram(d, bins=bins)
    r = 0.5 * (edges[1:] + edges[:-1])

    shell_volume = 4.0 * np.pi * r**2 * dr
    ideal_counts = shell_volume * rho * n_atoms

    with np.errstate(divide="ignore", invalid="ignore"):
        g_r = np.where(ideal_counts > 0, counts / ideal_counts, 0.0)

    return r, g_r


def average_pair_correlation(trajectory, r_max=8.0, dr=0.05, n_frames=10):
    """
    Average g(r) over several frames of a trajectory to reduce
    thermal noise in the estimate.

    Parameters
    ----------
    trajectory : list of ase.Atoms
    n_frames : int — frames to average over, evenly spaced

    Returns
    -------
    r, g_r_avg : as in compute_pair_correlation
    """
    idx = np.linspace(0, len(trajectory) - 1, min(n_frames, len(trajectory)))
    idx = np.unique(idx.astype(int))

    g_r_all = [compute_pair_correlation(trajectory[i], r_max, dr)[1] for i in idx]
    r, _ = compute_pair_correlation(trajectory[idx[0]], r_max, dr)

    return r, np.mean(g_r_all, axis=0)


def find_cutoff_from_gr(r, g_r, peak_search_window=(2.0, 4.0)):
    """
    Estimate a neighbor cutoff from g(r): locate the first
    coordination-shell peak within `peak_search_window`, then walk
    forward to the first following minimum — the standard choice
    for a neighbor cutoff (separates 1st and 2nd shells).

    Returns
    -------
    cutoff_r : float — r at the first minimum after the first peak
    peak_r   : float — r at the first peak (nearest-neighbor distance)
    """
    mask = (r >= peak_search_window[0]) & (r <= peak_search_window[1])
    if not np.any(mask):
        raise ValueError(
            f"No bins found in peak_search_window={peak_search_window}. "
            "Check r_max/dr or widen the window."
        )

    peak_idx_local = np.argmax(g_r[mask])
    peak_idx = np.where(mask)[0][peak_idx_local]

    min_idx = peak_idx
    for k in range(peak_idx, len(g_r) - 1):
        if g_r[k + 1] > g_r[k]:
            min_idx = k
            break
        min_idx = k + 1

    return r[min_idx], r[peak_idx]


# =============================================================
# Partial pair correlation g_ab(r) — species-resolved version of
# the g(r) above. Used to derive a separate neighbor cutoff (and
# separate first-peak bond length) for each of the three pair
# types in a binary system: A-A, A-B, B-B.
# =============================================================

def compute_partial_pair_correlation(atoms, z_i, z_j, r_max=8.0, dr=0.05):
    """
    Compute the partial pair correlation g_ij(r): for each atom of
    atomic number z_i, how many atoms of atomic number z_j sit in
    each radial shell, relative to random chance.

    Requires atoms.numbers to hold REAL atomic numbers — true for
    anything loaded via data_io.trajectory.load_trajectory(), NOT
    true for a raw ase.io.read() of this project's LAMMPS dumps
    (see load_trajectory's docstring).

    Returns
    -------
    r, g_ij : as in compute_pair_correlation
    """
    Z = atoms.get_atomic_numbers()
    i_idx, j_idx, d = neighbor_list("ijd", atoms, r_max)

    mask = (Z[i_idx] == z_i) & (Z[j_idx] == z_j)
    d_ij = d[mask]

    n_i = int(np.sum(Z == z_i))
    n_j = int(np.sum(Z == z_j))
    volume = atoms.get_volume()
    rho_j = n_j / volume

    bins = np.arange(0, r_max + dr, dr)
    counts, edges = np.histogram(d_ij, bins=bins)
    r = 0.5 * (edges[1:] + edges[:-1])

    shell_volume = 4.0 * np.pi * r**2 * dr
    ideal_counts = shell_volume * rho_j * n_i

    with np.errstate(divide="ignore", invalid="ignore"):
        g_ij = np.where(ideal_counts > 0, counts / ideal_counts, 0.0)

    return r, g_ij


def average_partial_pair_correlation(trajectory, z_i, z_j, r_max=8.0, dr=0.05, n_frames=10):
    """Average g_ij(r) over several frames — same idea as average_pair_correlation."""
    idx = np.linspace(0, len(trajectory) - 1, min(n_frames, len(trajectory)))
    idx = np.unique(idx.astype(int))

    g_all = [compute_partial_pair_correlation(trajectory[i], z_i, z_j, r_max, dr)[1] for i in idx]
    r, _ = compute_partial_pair_correlation(trajectory[idx[0]], z_i, z_j, r_max, dr)

    return r, np.mean(g_all, axis=0)


# =============================================================
# Cached, species-resolved cutoffs — one entry point for all
# three pair types (A-A, A-B, B-B) of a binary system.
# =============================================================

def _trajectory_fingerprint(path):
    """
    Cheap identity check for a trajectory file: absolute path +
    size + modification time. Distinguishes "different simulation"
    or "file was regenerated" from "same file as before", without
    hashing the full (multi-MB) trajectory on every call.
    """
    abspath = os.path.abspath(path)
    stat = os.stat(abspath)
    return f"{abspath}:{stat.st_size}:{stat.st_mtime}"


def _default_cache_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cutoff_cache.json")


def _load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_cache(cache_path, cache):
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def get_cached_partial_cutoffs(trajectory_file, z_a, z_b, cache_path=None):
    """
    Return g(r)-derived cutoffs AND first-peak distances for all
    three pair types of a binary A/B system: (A,A), (A,B), (B,B).
    Computed once per distinct trajectory file (and z_a/z_b choice)
    and cached to disk:
      - same trajectory again (new process, later run) -> instant.
      - a DIFFERENT trajectory file -> computed fresh, cached under
        its own entry; other simulations' cached values untouched.
      - same path but file regenerated in place (size/mtime changed)
        -> detected as new, recomputed rather than reusing a stale value.

    Parameters
    ----------
    trajectory_file : str
    z_a, z_b : int — the two real atomic numbers present (e.g. 40, 29)
    cache_path : str, optional

    Returns
    -------
    dict with keys "AA_cutoff", "AA_peak", "AB_cutoff", "AB_peak",
    "BB_cutoff", "BB_peak" (all floats, Å). "AA"/"AB"/"BB" refer to
    (z_a,z_a), (z_a,z_b), (z_b,z_b) respectively — the caller (config.py)
    maps these onto ZrZr/ZrCu/CuCu.
    """
    if cache_path is None:
        cache_path = _default_cache_path()

    fingerprint = _trajectory_fingerprint(trajectory_file) + f":{z_a}:{z_b}"
    cache = _load_cache(cache_path)

    if fingerprint in cache:
        return cache[fingerprint]

    import ase.io
    from config import TYPE_TO_Z  # local import: avoids a load-time cycle

    trajectory = ase.io.read(trajectory_file, format="lammps-dump-text", index="::1")
    for atoms in trajectory:
        atoms.set_atomic_numbers([TYPE_TO_Z[t] for t in atoms.arrays["type"]])

    result = {}
    for label, (zi, zj) in [("AA", (z_a, z_a)), ("AB", (z_a, z_b)), ("BB", (z_b, z_b))]:
        r, g_r = average_partial_pair_correlation(trajectory, zi, zj, r_max=5.0, n_frames=10)
        cutoff, peak = find_cutoff_from_gr(r, g_r)
        result[f"{label}_cutoff"] = cutoff
        result[f"{label}_peak"] = peak

    cache[fingerprint] = result
    _save_cache(cache_path, cache)

    return result