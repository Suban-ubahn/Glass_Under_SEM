# Glass Under SEM

Computational pipeline for studying amorpheous materials under simulated scanning/transmission electron microscopy conditions. Built around a LAMMPS molecular dynamics trajectory (`trj.zr50cu50_6750_vib`).

The pipeline has three analysis branches — free-volume hole detection, graph-based structural defect detection, and 4D-STEM diffraction simulation — which are combined into time-integrated 2D projection maps and cross-correlated to study the spatial relationship between holes and defects.

---

> **Large files:** outputs or checkpoints over ~50MB should be tracked
> with `git lfs track` before committing — GitHub rejects files over
> 100MB outright.

## Installation

Requires conda. After cloning the repository, either:

    setup_env.sh
    
or:

    conda env create -f environment.yml -n glass_sem_env    # or: conda create -n glass_sem_env python=3.11
    conda activate glass_sem_env
    pip install -r requirements.txt

This creates a `glass_sem_env` conda environment, installs all
dependencies from `requirements.txt`— it does **not** install the
project itself as a package.

Only run the next step if you want `import config`, `from holes...`,
etc. to work from any directory:

    pip install -e .

After setup, activate the environment before running anything:

```bash
conda activate glass_sem_env
```

Key dependencies: `ase`, `numpy`, `scipy`, `matplotlib`, `torch`, `torch_geometric`, `abtem`.
A full list is in `requirements.txt`.

> **WSL2 note:** the 4D-STEM simulation requires significant RAM. Add `memory=12GB` to `C:\Users\<name>\.wslconfig` on Windows and run `wsl --shutdown` before starting.

---

## Project Structure

Glass_Under_SEM/
│
├── config.py                 # All shared constants — edit this before running anything
├── utils.py                  # Shared infrastructure (get_output_dir, etc.) — not physical constants
│
├── data_io/
│   └── trajectory.py         # Single LAMMPS dump loader used by all modules
│
├── holes/
│   ├── core.py               # KDTree/voxel hole detection
│   └── fft.py                # FFT-convolution hole density field
│
├── graph/
│   ├── build.py               # ASE Atoms → PyTorch Geometric graph
│   ├── defects.py             # Coordination defect detection, clustering, tracking
│   └── gnn.py                 # GNN for atomic displacement prediction
│
├── diffraction/
│   ├── simulate.py                  # 4D-STEM and plane wave simulations (abtem)
│   ├── reconstruct_pypty_setup.py   # Converts 4d_stem.zarr → .h5 for pypty
│   ├── reconstruct_pypty.py         # Iterative ptychographic reconstruction (pypty)
│   └── ptycho_output/               # pypty's own checkpoint/resume state (co.npy, cp.npy, cg.npy, loss.csv)
│
├── analysis/
│   ├── projection_maps.py     # Time-integrated hole/defect heatmaps + cross-correlation
│   └── pypty_comparison.py    # Compares pypty reconstruction against ground-truth MD structure
│
├── visualization/
│   ├── animate.py             # 3D animations of holes and defect clusters
│   └── plot.py                # All static plot functions
│
├── scripts/                   # SLURM submission scripts
│   ├── run_simulate.sh
│   ├── run_pypty.sh
│   └── param_sweep.sh
│
├── outputs/                   # Auto-generated plots, one subfolder per producing script
│                               #   (gitignored — see utils.get_output_dir)
│
└── scratch/                   # Exploratory scripts, not part of the main pipeline (gitignored)

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

**Run heavy, long-running computations via SLURM (`sbatch`); run everything else normally with `python`.**

The two SLURM-based pipelines — the 4D-STEM simulation and the pypty
reconstruction — are both multi-hour-to-multi-day jobs meant to run
unattended on the cluster, independent of your own SSH session staying
connected:

```bash
sbatch scripts/run_simulate.sh
sbatch scripts/run_pypty.sh
```

Everything else — hole detection, defect detection, projection maps,
animations, plotting, and the ground-truth comparison — is fast enough
to just run directly from the project root:

```bash
python holes/core.py
python graph/defects.py
python analysis/projection_maps.py
python analysis/pypty_comparison.py
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

`diffraction/simulate.py` averages atomic positions across all MD frames and runs a full `GridScan` with a `PixelatedDetector` (abtem). Output is saved to `diffraction/4d_stem.zarr`. Run via SLURM, since a full simulation can run for hours:

```bash
sbatch scripts/run_simulate.sh
```

To load a previously computed dataset and plot without re-running the simulation:

```python
import abtem
from visualization.plot import plot_diffraction
dp = abtem.from_zarr("diffraction/4d_stem.zarr")
plot_diffraction(dp, blocked=True)
```

A `resolution_test()` function runs plane wave simulations at low/medium/high resolution to check whether simulation quality is the limiting factor before committing to a full 4D-STEM run on stronger hardware.

### Ptychographic reconstruction (pypty)

Two steps, run in order via SLURM — a full reconstruction can take days:

```bash
sbatch scripts/run_pypty.sh
```

This runs `diffraction/reconstruct_pypty_setup.py` (converts the 4D-STEM zarr dataset to `.h5`, in the `glass_sem_env` environment) followed by `diffraction/reconstruct_pypty.py` (the iterative reconstruction itself, in the `pypty` environment). Checkpoints (`co.npy`, `cp.npy`, `cg.npy`, `loss.csv`) are written to `diffraction/ptycho_output/` every epoch — if the job is killed or times out, resubmitting `scripts/run_pypty.sh` automatically resumes from the last saved epoch rather than starting over.

To compare a completed reconstruction against the original ground-truth MD structure it was simulated from:

```bash
python analysis/pypty_comparison.py
```

### Animations

Three animations in `visualization/animate.py`:

- `animate_holes_KDTree()` — KDTree-detected hole dynamics
- `animate_holes_fft()` — FFT hole peaks, coloured by strength
- `animate_defects()` — defect cluster dynamics, coloured by cluster size

### Plotting

All plot functions are in `visualization/plot.py`:

- `plot_map(image, title, cmap, colorbar_label)` — 2D heatmap
- `plot_diffraction(diffraction_patterns, blocked, save_path)` — diffraction pattern; `blocked=True` masks the direct beam
- `plot_cross_correlation(xcorr)` — 2D map + radial profile
- `plot_pypty_reconstruction(obj_phase, obj_amp, probe_amp, output_path)` — reconstructed object phase/amplitude and probe amplitude
- `plot_pypty_vs_ground_truth(reconstructed, ground_truth, pearson_r, output_path)` — side-by-side comparison against the MD ground truth, with difference map

### Output convention

Any script that produces a plot, heatmap, or animation saves it automatically to `outputs/<script_name>/` — e.g. running `python analysis/pypty_comparison.py` writes into `outputs/pypty_comparison/`. This folder is gitignored; nothing in it is meant to be committed. See `utils.get_output_dir()`.

---

## Physical Notes

- **Sample dimensions**: the MD cell is ~49 × 49 × 49 Å (~5 nm per side). This is thinner than a typical TEM lamella (20–50 nm), resulting in weak diffuse scattering in 4D-STEM. Tiling in Z (`atoms.repeat((1, 1, 4))`) increases scattering but assumes periodic repetition.
- **Atomic radii**: unified to Zr = 1.60 Å, Cu = 1.28 Å across all modules. Previous scripts used Cu = 1.30 Å — change only in `config.py` if needed.
- **Defect–hole relationship**: holes (free volume) and coordination defects are physically related — defect atoms tend to sit at hole boundaries. The cross-correlation analysis in `analysis/projection_maps.py` quantifies this spatial relationship.