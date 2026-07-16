# Glass Under SEM

Computational pipeline for studying Zr₅₀Cu₅₀ metallic glass under simulated scanning/transmission electron microscopy conditions. Built around a LAMMPS molecular dynamics trajectory (`trj.zr50cu50_6750_vib`).

The pipeline has three analysis branches — free-volume hole detection, graph-based structural defect detection, and 4D-STEM diffraction simulation — which are combined into time-integrated 2D projection maps and cross-correlated to study the spatial relationship between holes and defects.

---

## Installation

Requires conda. After cloning the repository:

```bash
conda activate glass_sem_env
pip install -e .
```

Key dependencies: `ase`, `numpy`, `scipy`, `matplotlib`, `torch`, `torch_geometric`, `abtem`.
A full list is in `requirements.txt`.

> **WSL2 note:** the 4D-STEM simulation requires significant RAM. Add `memory=12GB` to `C:\Users\<name>\.wslconfig` on Windows and run `wsl --shutdown` before starting.

---

## Project Structure

Glass_Under_SEM/
│
├── config.py                 # All shared constants — edit this before running anything
│
├── data_io/
│   └── trajectory.py         # Single LAMMPS dump loader used by all modules
│
├── holes/
│   ├── core.py               # KDTree/voxel hole detection
│   └── fft.py                # FFT-convolution hole density field
│
├── graph/
│   ├── build.py              # ASE Atoms → PyTorch Geometric graph
│   ├── defects.py            # Coordination defect detection, clustering, tracking
│   └── gnn.py                # GNN for atomic displacement prediction
│
├── diffraction/
│   └── simulate.py           # 4D-STEM and plane wave simulations (abtem)
│
├── analysis/
│   └── projection_maps.py    # Time-integrated hole/defect heatmaps + cross-correlation
│
├── visualization/
│   ├── animate.py            # 3D animations of holes and defect clusters
│   └── plot.py               # All static plot functions
│
└── scratch/                  # Exploratory scripts, not part of the main pipeline

---

## Configuration

All physical parameters live in `config.py`. The most important ones to check before running:

| Parameter | Description |
|---|---|
| `TRAJECTORY_FILE` | Path to the LAMMPS dump file |
| `VOXEL_EDGE` | Grid resolution for hole detection (Å) |
| `R_CUT_VOID` | Void threshold radius for KDTree detection (Å) |
| `CUTOFF` | Bond cutoff for graph construction (Å) |
| `DEFECT_THRESHOLD` | Missing bonds below element mean → defect |
| `CONVERGENCE_SEMIANGLE_MRAD` | Electron probe aperture (confirm with collaborator) |
| `BEAM_ENERGY_KV` | Electron beam energy (keV) |

---

## Usage

All scripts are run from the project root:

```bash
python diffraction/simulate.py
python analysis/projection_maps.py
python visualization/animate.py
```

### Hole detection

Two approaches in `holes/`:

- **KDTree/voxel** (`core.py`): marks voxels farther than `R_CUT_VOID` from any atom as void, labels connected void regions. Fast and physically intuitive.
- **FFT convolution** (`fft.py`): convolves atom grids with element-specific sphere kernels to build an occupancy field, finds peaks in the inverted field.

### Defect detection

In `graph/defects.py`. Atoms form a graph with edges between neighbours within `CUTOFF` Å. Defects are atoms whose coordination number falls more than `DEFECT_THRESHOLD` bonds below the element mean. Connected defect atoms are grouped into clusters.

Two classifiers are available:
- `classify_defects_from_deficit()` — primary, absolute threshold
- `classify_defects_statistical()` — alternative, mean − k·std threshold

### Projection maps & cross-correlation

`analysis/projection_maps.py` loops over the full trajectory, projects both hole density and defect cluster maps onto XY, and accumulates them into time-integrated heatmaps saved as:

analysis/cumulative_hole_map.npy
analysis/cumulative_cluster_map.npy

To re-run only the cross-correlation without recomputing the maps:

```python
from analysis.projection_maps import compute_cross_correlation, plot_cross_correlation
plot_cross_correlation(compute_cross_correlation())
```

### 4D-STEM diffraction simulation

`diffraction/simulate.py` averages atomic positions across all MD frames and runs a full `GridScan` with a `PixelatedDetector` (abtem). Output is saved to `diffraction/4d_stem.zarr`.

To load a previously computed dataset and plot without re-running the simulation:

```python
import abtem
from visualization.plot import plot_diffraction
dp = abtem.from_zarr("diffraction/4d_stem.zarr")
plot_diffraction(dp, blocked=True)
```

A `resolution_test()` function runs plane wave simulations at low/medium/high resolution to check whether simulation quality is the limiting factor before committing to a full 4D-STEM run on stronger hardware.

### Animations

Three animations in `visualization/animate.py`:

- `animate_holes()` — KDTree-detected hole dynamics
- `animate_holes_fft()` — FFT hole peaks, coloured by strength
- `animate_defects()` — defect cluster dynamics, coloured by cluster size

### Plotting

All plot functions are in `visualization/plot.py`:

- `plot_map(image, title, cmap, colorbar_label)` — 2D heatmap
- `plot_diffraction(diffraction_patterns, blocked, save_path)` — diffraction pattern; `blocked=True` masks the direct beam
- `plot_cross_correlation(xcorr)` — 2D map + radial profile

---

## Physical Notes

- **Sample dimensions**: the MD cell is ~49 × 49 × 49 Å (~5 nm per side). This is thinner than a typical TEM lamella (20–50 nm), resulting in weak diffuse scattering in 4D-STEM. Tiling in Z (`atoms.repeat((1, 1, 4))`) increases scattering but assumes periodic repetition.
- **Atomic radii**: unified to Zr = 1.60 Å, Cu = 1.28 Å across all modules. Previous scripts used Cu = 1.30 Å — change only in `config.py` if needed.
- **Defect–hole relationship**: holes (free volume) and coordination defects are physically related — defect atoms tend to sit at hole boundaries. The cross-correlation analysis in `analysis/projection_maps.py` quantifies this spatial relationship.