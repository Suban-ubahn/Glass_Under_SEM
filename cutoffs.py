# =============================================================
# cutoffs.py -- g(r)-derived neighbour cutoffs
#
# Split out of config.py: importing this module reads the trajectory
# (cached in .cutoff_cache.json), so config.py stays a cheap pure-constants
# module that any environment (e.g. the pypty env) can import.
# =============================================================

from config import TRAJECTORY_FILE, TYPE_TO_Z, CUTOFF_PEAK_SEARCH_WINDOW
from utils import get_cached_partial_cutoffs, fit_species_radii

# ------------------------------------------------------------
#
# Species-resolved cutoffs, derived from g(r) for every pair of species
# of the active dataset (config.TYPE_TO_Z; e.g. Zr-Zr, Zr-Cu, Cu-Cu or
# Al-Al, Al-O, O-O) rather than one blended cutoff:
#
#   PARTIAL_CUTOFFS — first MINIMUM after the first peak. The boundary of
#               the full first coordination shell — used wherever
#               code needs "every atom in the local neighborhood"
#               (e.g. analysis/d2min/compute.py's D^2_min: the affine
#               fit needs the complete local shell, not just the
#               single most-common bond distance).
#
#   PARTIAL_PEAKS   — the first peak itself. The single most probable
#               pair distance for that species combination — used
#               specifically for graph EDGES (graph/build.py and
#               anything downstream of it: graph/gnn.py,
#               graph/defects.py, defect-cluster code), since those
#               edges are meant to represent literal bonds.
#
#               NOTE: using the peak rather than the minimum means
#               roughly half of the true first-shell neighbors (the
#               half sitting farther than the peak, but still inside
#               the first shell) will NOT get an edge — the peak is
#               the mode of the bond-length distribution, not its
#               outer boundary. This will show up as noticeably
#               lower coordination numbers than PARTIAL_CUTOFFS would give.
#               Worth checking degree/coordination stats after
#               switching graph code to PARTIAL_PEAKS for the first time.
#
#   SPECIES_RADII / TYPE_TO_RADIUS — one radius per species, fitted (least
#               squares, utils.fit_species_radii) to the PARTIAL_PEAKS of
#               all species pairs: r_i + r_j ~ first-peak distance d_ij.
#               Used for the sphere kernels of the FFT hole finder
#               (holes/fft.py); TYPE_TO_RADIUS is keyed by LAMMPS type.
# ------------------------------------------------------------

_partial = get_cached_partial_cutoffs(TRAJECTORY_FILE, TYPE_TO_Z, CUTOFF_PEAK_SEARCH_WINDOW)

PARTIAL_CUTOFFS = _partial["cutoff"]   # {(z_i, z_j): Å} first minimum, one entry per species pair (z_i <= z_j)
PARTIAL_PEAKS   = _partial["peak"]     # {(z_i, z_j): Å} first peak,    one entry per species pair (z_i <= z_j)

# Ready-to-use dicts for ase.neighborlist.neighbor_list's per-species-pair
# cutoff argument (keyed by REAL atomic numbers — see load_trajectory(),
# both orderings of every pair).
CUTOFF_DICT = {}
GRAPH_CUTOFF_DICT = {}
for (_z_i, _z_j), _r in PARTIAL_CUTOFFS.items():
    CUTOFF_DICT[(_z_i, _z_j)] = CUTOFF_DICT[(_z_j, _z_i)] = _r
for (_z_i, _z_j), _r in PARTIAL_PEAKS.items():
    GRAPH_CUTOFF_DICT[(_z_i, _z_j)] = GRAPH_CUTOFF_DICT[(_z_j, _z_i)] = _r

SPECIES_RADII  = fit_species_radii(PARTIAL_PEAKS)                     # {z: Å}
TYPE_TO_RADIUS = {t: SPECIES_RADII[z] for t, z in TYPE_TO_Z.items()}  # {LAMMPS type: Å}
