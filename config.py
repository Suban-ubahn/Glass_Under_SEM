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

TRAJECTORY_FILE = "trj.zr50cu50_6750_vib"

# ------------------------------------------------------------
# Atom types  (LAMMPS type index → atomic number)
# ------------------------------------------------------------

TYPE_TO_Z = {1: 40, 2: 29}   # Zr = 40, Cu = 29

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
# ------------------------------------------------------------

CUTOFF           = 3.5   # Å  — neighbor cutoff for bond edges
DEFECT_THRESHOLD = 1.0   # missing bonds below element mean → defect

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

BEAM_ENERGY_KV      = 200
MAX_ANGLE_MRAD      = 80
GPTS                = 512
SLICE_THICKNESS     = 2
NUM_PHONON_CONFIGS  = 4
PHONON_SIGMA        = 0.08   # Å  — frozen phonon displacement amplitude
CONVERGENCE_SEMIANGLE_MRAD = 25   # mrad