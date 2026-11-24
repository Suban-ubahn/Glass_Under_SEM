# =============================================================
# utils.py
#
# Shared infrastructure helpers (not physical constants — see config.py).
#
# Deliberately has NO import-time dependency on config.py or data_io/trajectory.py
# (both of which import config; a few functions import config lazily, inside the
# function) — that keeps this module a "leaf" with nothing importing it back, so
# config.py can safely import from here.
# =============================================================

import atexit
import os
import sys
import json

import numpy as np
from ase.neighborlist import neighbor_list

OUTPUTS_ROOT = "outputs"
MISC_NAME = "miscellaneous"          # home of output from non-script runs (python -c, REPL, notebooks)
_made_output_dirs = set()


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


def _default_output_name():
    """
    Name of the program that is running, from which the folder under outputs/ is
    made (get_output_dir): the stem of the script (python scripts/run_d2min.py ->
    "run_d2min").

    Anything that is not a .py script -- `python -c ...`, `python -` (heredoc),
    a REPL, a Jupyter kernel -- has no script name, so its files go to
    outputs/miscellaneous/ instead of a meaningless "outputs/-c/".
    Shell scripts that feed python through a heredoc can name their folder
    explicitly:  export GLASS_OUTPUT_NAME=<name>
    """
    forced = os.environ.get("GLASS_OUTPUT_NAME")
    if forced:
        return forced
    argv0 = sys.argv[0] if sys.argv else ""
    stem, ext = os.path.splitext(os.path.basename(argv0))
    if ext == ".py" and stem and not stem.startswith("ipykernel"):
        return stem
    return MISC_NAME


def _remove_if_empty(path):
    try:
        os.rmdir(path)          # only succeeds when nothing was written into it
    except OSError:
        pass


def get_output_dir(name=None, dataset=None):
    """
    Return outputs/<name>_<dataset>/, creating it if it doesn't exist.
    Convention: every script writes only into outputs/<its own file name>_<dataset>/,
    where <dataset> is the name (key in config.DATASETS) of the dataset it works on (config.py,
    config.output_folder) -- see _default_output_name(). Folders that stay empty
    are removed at exit, so merely importing a module never leaves an empty
    folder behind.

    name    : program name (default: the running script)
    dataset : key of config.DATASETS to use (default: the active dataset);
              scripts.generate_md.py passes MD_DATASET, because it works on another
              dataset than the analysis scripts
    """
    from config import output_folder   # local import: see the header of this module
    if name is None:
        name = _default_output_name()
    path = os.path.join(OUTPUTS_ROOT, output_folder(name, dataset))
    os.makedirs(path, exist_ok=True)
    if path not in _made_output_dirs:
        _made_output_dirs.add(path)
        atexit.register(_remove_if_empty, path)
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
# separate first-peak bond length) for every pair of species in the
# dataset (A-A, A-B, B-B, ... for as many species as are present).
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
# species pairs of the dataset.
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


def get_cached_partial_cutoffs(trajectory_file, type_to_z=None, peak_search_window=None, cache_path=None):
    """
    Return g(r)-derived cutoffs AND first-peak distances for every
    pair of species present in the trajectory (A-A, A-B, B-B, ...).
    Computed once per distinct trajectory file (and type mapping /
    peak search window) and cached to disk:
      - same trajectory again (new process, later run) -> instant.
      - a DIFFERENT trajectory file -> computed fresh, cached under
        its own entry; other simulations' cached values untouched.
      - same path but file regenerated in place (size/mtime changed)
        -> detected as new, recomputed rather than reusing a stale value.

    Parameters
    ----------
    trajectory_file : str
    type_to_z : dict, optional — LAMMPS type -> atomic number
        (default: config.TYPE_TO_Z)
    peak_search_window : (float, float), optional — r-range (Å) in which the
        first peak of each partial g(r) is searched
        (default: config.CUTOFF_PEAK_SEARCH_WINDOW)
    cache_path : str, optional

    Returns
    -------
    dict with keys "cutoff" and "peak", each a dict {(z_i, z_j): float (Å)}
    over all pairs of the species present with z_i <= z_j. "cutoff" is the
    first minimum after the first peak, "peak" the first peak itself.
    """
    if type_to_z is None or peak_search_window is None:
        from config import TYPE_TO_Z, CUTOFF_PEAK_SEARCH_WINDOW  # local import: avoids a load-time cycle
        type_to_z = TYPE_TO_Z if type_to_z is None else type_to_z
        peak_search_window = CUTOFF_PEAK_SEARCH_WINDOW if peak_search_window is None else peak_search_window

    if cache_path is None:
        cache_path = _default_cache_path()

    types_key = ",".join(f"{t}={z}" for t, z in sorted(type_to_z.items()))
    window_key = f"{peak_search_window[0]:g}-{peak_search_window[1]:g}"
    fingerprint = _trajectory_fingerprint(trajectory_file) + f":{types_key}:{window_key}"
    cache = _load_cache(cache_path)

    if fingerprint not in cache:
        import ase.io

        trajectory = ase.io.read(trajectory_file, format="lammps-dump-text", index="::1")
        for atoms in trajectory:
            atoms.set_atomic_numbers([type_to_z[t] for t in atoms.arrays["type"]])

        zs = sorted(set(type_to_z.values()))
        entry = {}
        for i, z_i in enumerate(zs):
            for z_j in zs[i:]:
                r, g_r = average_partial_pair_correlation(trajectory, z_i, z_j, r_max=5.0, n_frames=10)
                cutoff, peak = find_cutoff_from_gr(r, g_r, peak_search_window)
                entry[f"{z_i}-{z_j}"] = {"cutoff": cutoff, "peak": peak}

        cache[fingerprint] = entry
        _save_cache(cache_path, cache)

    result = {"cutoff": {}, "peak": {}}
    for key, vals in cache[fingerprint].items():
        pair = tuple(int(z) for z in key.split("-"))
        result["cutoff"][pair] = vals["cutoff"]
        result["peak"][pair] = vals["peak"]
    return result


# =============================================================
# Per-species radii from the first g(r) peaks
# =============================================================

def fit_species_radii(peaks):
    """
    Per-species radius from the first-peak distances of the partial g(r)s.

    For every species pair (i, j) the first peak d_ij of g_ij(r) is the
    typical contact distance, modelled as r_i + r_j (like pairs: 2 r_i = d_ii).
    All pairs together give an overdetermined linear system in the radii,
    solved in the least-squares sense.

    Parameters
    ----------
    peaks : dict {(z_i, z_j): float (Å)} — first-peak distance of every species
        pair, e.g. the "peak" entry of get_cached_partial_cutoffs()

    Returns
    -------
    dict {z: radius (Å)}
    """
    zs = sorted({z for pair in peaks for z in pair})
    col = {z: k for k, z in enumerate(zs)}
    A = np.zeros((len(peaks), len(zs)))
    d = np.zeros(len(peaks))
    for row, ((z_i, z_j), dist) in enumerate(peaks.items()):
        A[row, col[z_i]] += 1.0
        A[row, col[z_j]] += 1.0
        d[row] = dist
    r, *_ = np.linalg.lstsq(A, d, rcond=None)
    if np.any(r <= 0):
        raise ValueError(
            f"Fitted species radii are not all positive ({ {z: round(float(x), 3) for z, x in zip(zs, r)} }); "
            f"check the first-peak distances {peaks} and the peak_search_window."
        )
    return {z: float(r[col[z]]) for z in zs}


# =============================================================
# Composition label
# =============================================================

def atomic_percent_formula(atoms):
    """
    Formula in atomic percent, highest atomic number first, e.g. 'Zr50Cu50' or 'Al40O60'.
    Requires REAL atomic numbers in atoms (anything loaded via data_io.trajectory.load_trajectory()).
    """
    from ase.data import chemical_symbols

    z, counts = np.unique(atoms.get_atomic_numbers(), return_counts=True)
    order = np.argsort(-z)
    return "".join(f"{chemical_symbols[z[i]]}{100.0 * counts[i] / len(atoms):g}" for i in order)
