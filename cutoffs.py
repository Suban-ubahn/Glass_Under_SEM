# =============================================================
# cutoffs.py -- g(r)-derived neighbour cutoffs
#
# Split out of config.py: importing this module reads the trajectory
# (cached in .cutoff_cache.json), so config.py stays a cheap pure-constants
# module that any environment (e.g. the pypty env) can import.
# =============================================================

from config import TRAJECTORY_FILE, Z_ZR, Z_CU
from utils import get_cached_partial_cutoffs

# ------------------------------------------------------------
#
# Species-resolved cutoffs, derived from g(r) for each of the three
# pair types (Zr-Zr, Zr-Cu, Cu-Cu) rather than one blended cutoff:
#
#   CUTOFF_*  — first MINIMUM after the first peak. The boundary of
#               the full first coordination shell — used wherever
#               code needs "every atom in the local neighborhood"
#               (e.g. analysis/d2min/compute.py's D^2_min: the affine
#               fit needs the complete local shell, not just the
#               single most-common bond distance).
#
#   PEAK_*    — the first peak itself. The single most probable
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
#               lower coordination numbers than CUTOFF_* would give.
#               Worth checking degree/coordination stats after
#               switching graph code to PEAK_* for the first time.
# ------------------------------------------------------------

_zrzr_zrcu_cucu = get_cached_partial_cutoffs(TRAJECTORY_FILE, Z_ZR, Z_CU)

CUTOFF_ZRZR = _zrzr_zrcu_cucu["AA_cutoff"]   # Å — first minimum, Zr-Zr
CUTOFF_ZRCU = _zrzr_zrcu_cucu["AB_cutoff"]   # Å — first minimum, Zr-Cu
CUTOFF_CUCU = _zrzr_zrcu_cucu["BB_cutoff"]   # Å — first minimum, Cu-Cu

PEAK_ZRZR = _zrzr_zrcu_cucu["AA_peak"]       # Å — first peak, Zr-Zr
PEAK_ZRCU = _zrzr_zrcu_cucu["AB_peak"]       # Å — first peak, Zr-Cu
PEAK_CUCU = _zrzr_zrcu_cucu["BB_peak"]       # Å — first peak, Cu-Cu

# Ready-to-use dicts for ase.neighborlist.neighbor_list's per-species-pair
# cutoff argument (keyed by REAL atomic numbers — see load_trajectory()).
CUTOFF_DICT = {
    (Z_ZR, Z_ZR): CUTOFF_ZRZR,
    (Z_ZR, Z_CU): CUTOFF_ZRCU, (Z_CU, Z_ZR): CUTOFF_ZRCU,
    (Z_CU, Z_CU): CUTOFF_CUCU,
}
GRAPH_CUTOFF_DICT = {
    (Z_ZR, Z_ZR): PEAK_ZRZR,
    (Z_ZR, Z_CU): PEAK_ZRCU, (Z_CU, Z_ZR): PEAK_ZRCU,
    (Z_CU, Z_CU): PEAK_CUCU,
}
