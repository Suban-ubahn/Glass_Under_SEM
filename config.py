# =============================================================
# config.py — project-wide constants
#
# All physical parameters, file paths, and tunable settings
# live here. Import from this module everywhere instead of
# re-defining values in each script.
# =============================================================


# ------------------------------------------------------------
# Trajectory
# ------------------------------------------------------------

TRAJECTORY_FILE = "data/trj.zr50cu50_6750_vib"

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
# Hole detection — KDTree / voxel approach  (holes/kdtree.py)
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
# The g(r)-derived neighbour cutoffs (CUTOFF_*, PEAK_*, CUTOFF_DICT,
# GRAPH_CUTOFF_DICT) live in cutoffs.py: deriving them reads the trajectory,
# which a pure-constants module shouldn't do on import.
# ------------------------------------------------------------

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
#   (analysis/soft_modes/modes.py, analysis/soft_modes/tls.py,
#    analysis/soft_modes/comparison.py)
# ------------------------------------------------------------

N_Z_SLICES       = 10     # equally spaced z-slabs for the 2D (CAVIAR-style) analysis
N_SOFT_MODES     = 20     # how many of the softest modes feed the per-atom softness score
SOFT_MODE_PR_MAX = None   # participation-ratio cutoff for "quasi-localized" -- None means
                           # no filtering (all n_soft_modes count toward softness). With only
                           # ~100 frames available (trj.zr50cu50_6750_vib), the top whole-
                           # system modes run PR ~0.15-0.6 (fairly delocalized -- with this
                           # few frames vs. 3*6750 DOF, the SVD can only resolve large-scale,
                           # multi-atom motion reliably, not single-atom localization; see
                           # module docstring in analysis/soft_modes/modes.py). Set a cutoff (e.g.
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


# ------------------------------------------------------------
# PCA temporal analysis, convergence and local (moving-window) PCA
#   (analysis/soft_modes/pca_convergence.py, analysis/soft_modes/local_pca.py;
#    scripts/run_pca_convergence.py, scripts/run_local_pca.py)
# None of these enters the soft-mode or TLS detection criteria above.
# ------------------------------------------------------------

# Autocorrelation of the mode amplitudes a_n(t), integrated autocorrelation time, T_eff
PCA_ACF_N_MODES        = 10        # leading modes analysed
PCA_ACF_MAX_LAG        = None      # largest lag (None = T // 2)
PCA_ACF_WINDOW_METHOD  = "sokal"   # truncation of the sum: "sokal", "geyer" or "first_negative"
PCA_ACF_WINDOW_C       = 5.0       # Sokal's window constant: W >= c * tau
PCA_ACF_MIN_T_OVER_TAU = 20.0      # flag "short_series" when T < this * tau
PCA_ACF_MAX_REL_ERR    = 0.3       # flag "large_error" when the relative error of tau exceeds it
PCA_ACF_BAND_Z         = 1.96      # white-noise band of the ACF, in units of 1/sqrt(T)
PCA_ACF_AR1_LAGS       = 10        # lags over which the AR(1) shape phi^k is checked
PCA_ACF_N_NULL         = 20        # shifted-column null runs for the per-mode ACF / T_eff (0 = none); ~1 s each at 6750 atoms

# PCA on progressively longer prefixes of the trajectory
PCA_CONV_LENGTHS       = (20, 40, 60, 80, 101)   # frames; the longest one is the reference
PCA_CONV_N_MODES       = 10        # reference modes followed (eigenvalues, overlaps)
PCA_CONV_SUBSPACE_DIM  = 5         # M: leading M-dimensional eigenspace compared
PCA_CONV_MATCHING      = "max"     # "max" (max overlap, may repeat) or "hungarian" (one-to-one)
PCA_CONV_N_NULL        = 20        # shifted-column null runs (0 = none)

# Moving-window local PCA
LOCAL_PCA_WINDOW_MODE        = "neighborhood"  # "neighborhood": centre atom + nearest neighbours (spatial); "index": atoms s..s+n-1
LOCAL_PCA_WINDOW_SIZE        = 15              # atoms per window (neighborhood: including the centre atom)
LOCAL_PCA_STEP               = 1               # shift between windows (index) / spacing of the centre atom indices (neighborhood), in atoms
LOCAL_PCA_N_GLOBAL_MODES     = 20              # leading global modes compared with
LOCAL_PCA_SUBSPACE_DIMS      = (1, 3, 5, 10)   # eigenspace dimensions M compared
LOCAL_PCA_N_COMPARE_MODES    = 10              # local modes with stored overlaps / followed across windows
LOCAL_PCA_CONSISTENCY_OFFSETS = (1, 2, 4, 7, 10)   # windows compared: window w+j (index) / j-th nearest window centre (neighborhood)
LOCAL_PCA_TRACK_THRESHOLD    = 0.8             # a mode is "found again" when the restricted overlap >= this
LOCAL_PCA_MIN_SHARED_WEIGHT  = 0.5             # ... and >= this share of its norm lies on the shared atoms
LOCAL_PCA_N_NULL             = 10              # shifted-column null runs (0 = none); ~10 s each at 6750 atoms
LOCAL_PCA_SAVE_EIGENVECTORS  = True            # write all window eigenvectors (float32, ~55 MB) to the .npz
