# =============================================================
# config.py — project-wide constants
#
# All physical parameters, file paths, and tunable settings
# live here. Import from this module everywhere instead of
# re-defining values in each script.
# =============================================================

from utils import get_cached_partial_cutoffs

# ------------------------------------------------------------
# Trajectory
# ------------------------------------------------------------

TRAJECTORY_FILE = "trj.zr50cu50_6750_vib"

# ------------------------------------------------------------
# Atom types  (LAMMPS type index → atomic number)
# ------------------------------------------------------------

TYPE_TO_Z = {1: 29, 2: 40}   # Zr = 40, Cu = 29
Z_ZR = 40
Z_CU = 29

# ------------------------------------------------------------
# Atomic radii (Å)
#
# Using physical metallic radii:  Zr = 1.60, Cu = 1.28
#
# Note: the FFT hole-finder and diffraction scripts previously
# used r_Cu = 1.3. Unified to 1.28 (more accurate).
# If you want to revert for a specific module, override locally.
# ------------------------------------------------------------

R_ZR = 1.60
R_CU = 1.28

TYPE_TO_RADIUS = {1: R_ZR, 2: R_CU}   # LAMMPS type → radius

# ------------------------------------------------------------
# Voxel grid
# ------------------------------------------------------------

VOXEL_EDGE = 0.3   # Å  — grid resolution for hole/occupancy fields

# ------------------------------------------------------------
# Hole detection — KDTree / voxel approach  (holes/core.py)
# ------------------------------------------------------------

R_CUT_VOID = 2.0   # Å  — voxels closer than this to any atom are "occupied"

# ------------------------------------------------------------
# Hole detection — FFT convolution approach  (holes/fft.py)
# ------------------------------------------------------------

GAUSSIAN_SIGMA_A  = 1.0    # Å  — smoothing applied to Vloch
PEAK_THRESHOLD    = 0.1    # gradient magnitude threshold for peak detection
LAPLACE_THRESHOLD = 0.01   # Laplacian threshold for peak detection

# ------------------------------------------------------------
# Graph / defect analysis  (graph/)
#
# Species-resolved cutoffs, derived from g(r) for each of the three
# pair types (Zr-Zr, Zr-Cu, Cu-Cu) rather than one blended cutoff:
#
#   CUTOFF_*  — first MINIMUM after the first peak. The boundary of
#               the full first coordination shell — used wherever
#               code needs "every atom in the local neighborhood"
#               (e.g. dynamics/flow_units.py's D^2_min: the affine
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

DEFECT_THRESHOLD = 1.0   # missing bonds below element mean → defect
DEFECT_K = 1.0   # defects are >1 standard deviation below element mean

# ------------------------------------------------------------
# GNN training  (graph/gnn.py)
# ------------------------------------------------------------

GNN_HIDDEN_DIM = 64
GNN_EPOCHS     = 10
GNN_LR         = 1e-3
GNN_BATCH_SIZE = 2

# ------------------------------------------------------------
# Diffraction simulation  (diffraction/simulate.py)
# ------------------------------------------------------------

BEAM_ENERGY_KV             = 200   
MAX_ANGLE_MRAD             = 80    
GPTS                       = 256   
SLICE_THICKNESS            = 1     
NUM_PHONON_CONFIGS         = 4     
PHONON_SIGMA               = 0.08  # Å  — frozen phonon displacement amplitude
CONVERGENCE_SEMIANGLE_MRAD = 25    
DEFOCUS                    = 100   
SAMPLING_FACTOR            = 3
TO_ZARR                    = True

# ------------------------------------------------------------
# Simulation Plot
# ------------------------------------------------------------

BLOCKED_VAL   = False
SCAN_POSITION = (31,7)

INNER_RADIUS  = 0     # px
OUTER_RADIUS  = None  # px
CENTER_VAL    = None
LOG_SCALE_VAL = True
NORMALIZE_VAL = False



# ------------------------------------------------------------
# Soft-mode / TLS analysis
#   (analysis/soft_modes.py, analysis/tls_detection.py,
#    analysis/soft_spot_tls_dmin_comparison.py)
# ------------------------------------------------------------

N_Z_SLICES       = 10     # equally spaced z-slabs for the 2D (CAVIAR-style) analysis
N_SOFT_MODES     = 20     # how many of the softest modes feed the per-atom softness score
SOFT_MODE_PR_MAX = None   # participation-ratio cutoff for "quasi-localized" -- None means
                           # no filtering (all n_soft_modes count toward softness). With only
                           # ~100 frames available (trj.zr50cu50_6750_vib), the top whole-
                           # system modes run PR ~0.15-0.6 (fairly delocalized -- with this
                           # few frames vs. 3*6750 DOF, the SVD can only resolve large-scale,
                           # multi-atom motion reliably, not single-atom localization; see
                           # module docstring in analysis/soft_modes.py). Set a cutoff (e.g.
                           # 0.2-0.3) once you've looked at plot_participation_spectrum() for
                           # your own run, or restrict atom_indices to a small local
                           # neighborhood instead (matching the notebook's Part 2 approach)
                           # for cleaner localization.
MAX_SOFT_NB = 5        # last group = "MAX_SOFT_NB or more soft neighbours"

TLS_DELTA_BIC_THRESHOLD  = 6.0    # Kass & Raftery "strong evidence" rule of thumb
TLS_SEPARATION_THRESHOLD = 1.5    # well separation, in units of mean in-well sigma
TLS_MIN_WEIGHT_FRACTION  = 0.15   # smaller cluster must hold >= this fraction of frames
TLS_MIN_SWITCHES         = 2      # >=1 full A->B->A round trip (not a 1-way D^2_min-like event)

D2MIN_QUANTILE_THRESHOLD = 0.9    # "rearranged" = D^2_min above this quantile of its distribution
D2MIN_TOP_FRACTION       = 0.1    # top X% by score, for enrichment comparisons