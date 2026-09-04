# Glass Under SEM

Computational pipeline for studying the structure and dynamics of amorphous materials under simulated electron-microscopy conditions.

The project combines **LAMMPS molecular-dynamics simulations**, **free-volume analysis**, **graph-based structural analysis**, **4D-STEM diffraction simulations**, and **ptychographic reconstruction** to investigate the relationship between structural defects and experimentally observable electron-scattering signals.

---

## Overview

The main input is a LAMMPS molecular-dynamics trajectory of a Zr–Cu metallic glass.

The analysis is organized into three main branches:

```text
                         LAMMPS trajectory
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
        Free-volume          Structural        4D-STEM
          analysis            defects          simulation
              │                 │                 │
              │                 │                 ▼
              │                 │          Ptychographic
              │                 │          reconstruction
              │                 │                 │
              └─────────────────┴─────────────────┘
                                │
                                ▼
                     Projection & correlation
                                │
                                ▼
                     Comparison with structure
```

The central scientific question is how **free volume and local structural defects** are related, and whether these structural features can be identified from simulated electron-microscopy measurements.

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Suban-ubahn/Glass_Under_SEM.git
cd Glass_Under_SEM
```

### 2. Create the Conda environment

```bash
conda create -n glass_sem_env python=3.11
conda activate glass_sem_env
pip install -r requirements.txt
pip install -e .
```

The editable installation (`pip install -e .`) is recommended because it allows modules such as `config`, `holes`, and `analysis` to be imported from anywhere.

### 3. Configure the simulation

Before running the pipeline, check `config.py`.

The most important parameters are:

| Parameter                    | Description                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| `TRAJECTORY_FILE`            | Path to the LAMMPS trajectory                                  |
| `VOXEL_EDGE`                 | Grid resolution used for free-volume detection (Å)             |
| `R_CUT_VOID`                 | Void threshold radius for KDTree detection (Å)                 |
| `CUTOFF`                     | Neighbour/bond cutoff used for graph construction (Å)          |
| `DEFECT_THRESHOLD`           | Coordination deficit required to classify an atom as defective |
| `CONVERGENCE_SEMIANGLE_MRAD` | Electron probe convergence semi-angle                          |
| `BEAM_ENERGY_KV`             | Electron beam energy                                           |

All physical parameters shared between modules should be changed in `config.py` rather than inside individual scripts.

---

# Project structure

```text
Glass_Under_SEM/
│
├── config.py
├── utils.py
│
├── data_io/
│   └── trajectory.py
│
├── holes/
│   ├── core.py
│   └── fft.py
│
├── graph/
│   ├── build.py
│   ├── defects.py
│   └── gnn.py
│
├── diffraction/
│   ├── simulate.py
│   ├── reconstruct_pypty_setup.py
│   ├── reconstruct_pypty.py
│   └── ptycho_output/
│
├── analysis/
│   ├── projection_maps.py
│   └── pypty_comparison.py
│
├── visualization/
│   ├── animate.py
│   └── plot.py
│
├── scripts/
│   ├── run_simulate.sh
│   ├── run_pypty.sh
│   └── param_sweep.sh
│
├── outputs/
│
└── scratch/
```

### Core configuration

* **`config.py`** — shared physical and numerical parameters.
* **`utils.py`** — shared infrastructure, including output-directory handling.

### Input data

* **`data_io/trajectory.py`** — loads the LAMMPS trajectory and provides a common interface to the rest of the pipeline.

### Free-volume analysis

* **`holes/core.py`** — KDTree/voxel-based free-volume detection.
* **`holes/fft.py`** — FFT-convolution approach for constructing an occupancy/free-volume field.

### Structural defect analysis

* **`graph/build.py`** — converts atomic coordinates into PyTorch Geometric graphs.
* **`graph/defects.py`** — detects, clusters, and tracks coordination defects.
* **`graph/gnn.py`** — graph neural network for predicting atomic displacements.

### Electron-microscopy simulation

* **`diffraction/simulate.py`** — 4D-STEM and plane-wave diffraction simulations using `abTEM`.
* **`diffraction/reconstruct_pypty_setup.py`** — converts the 4D-STEM dataset to the format required by `pypty`.
* **`diffraction/reconstruct_pypty.py`** — performs the iterative ptychographic reconstruction.
* **`diffraction/ptycho_output/`** — checkpoints and reconstruction state generated by `pypty`.

### Analysis

* **`analysis/projection_maps.py`** — creates time-integrated projection maps and calculates spatial cross-correlations.
* **`analysis/pypty_comparison.py`** — compares the ptychographic reconstruction with the original MD structure.

### Visualization

* **`visualization/plot.py`** — reusable static plotting functions.
* **`visualization/animate.py`** — 3D animations of holes and defect clusters.

### HPC scripts

The scripts in `scripts/` are intended for SLURM-based cluster execution.

### Generated files

`outputs/` contains automatically generated plots, heatmaps, and animations. It is gitignored and should not be committed.

`scratch/` contains exploratory code that is not part of the main pipeline. It is also gitignored.

---

# Analysis pipeline

## 1. Free-volume / hole detection

Two complementary approaches are implemented in `holes/`.

### KDTree / voxel method

```bash
python holes/core.py
```

`core.py` constructs a voxel grid and determines which voxels are sufficiently far from every atom to be considered part of a void. Connected void regions are then identified.

This method is fast and provides a direct geometric interpretation of free volume.

### FFT-convolution method

`holes/fft.py` uses element-specific spherical kernels and FFT convolution to construct an occupancy field. Peaks in the inverted field are used to identify candidate holes.

The two approaches provide complementary descriptions of the same underlying free-volume structure.

---

# 2. Structural defect detection

Run:

```bash
python graph/defects.py
```

Atoms are represented as nodes in a graph, with edges connecting neighbouring atoms within `CUTOFF`.

For each atom, its coordination number is compared with the mean coordination of atoms of the same element. Atoms whose coordination is sufficiently below the corresponding mean are classified as defects.

Connected defective atoms are grouped into clusters, allowing their spatial distribution and temporal evolution to be studied.

The graph representation also provides the basis for the GNN in `graph/gnn.py`.

---

# 3. Projection maps and correlation

Once hole and defect data have been generated, run:

```bash
python analysis/projection_maps.py
```

The analysis integrates the three-dimensional structures over time and projects them onto two-dimensional maps.

The resulting maps can be cross-correlated to quantify the spatial relationship between free-volume regions and structural defects.

For example:

```python
from analysis.projection_maps import (
    compute_cross_correlation,
    plot_cross_correlation,
)

xcorr = compute_cross_correlation()
plot_cross_correlation(xcorr)
```

---

# 4. 4D-STEM simulation

The 4D-STEM simulation is implemented in:

```text
diffraction/simulate.py
```

and should normally be run on the cluster:

```bash
sbatch scripts/run_simulate.sh
```

The simulation uses `abTEM` to perform a `GridScan` with a `PixelatedDetector`.

The resulting 4D-STEM dataset is stored as:

```text
diffraction/4d_stem.zarr
```

A complete simulation can take several hours, so it is intended to run as an unattended SLURM job.

To inspect an existing dataset without rerunning the simulation:

```python
import abtem

from visualization.plot import plot_diffraction

dp = abtem.from_zarr("diffraction/4d_stem.zarr")
plot_diffraction(dp, blocked=True)
```

### Resolution test

`simulate.py` also contains a `resolution_test()` function for comparing low-, medium-, and high-resolution plane-wave simulations.

This can be used to determine whether numerical resolution is limiting the simulation before committing to a full 4D-STEM calculation.

---

# 5. Ptychographic reconstruction

Ptychographic reconstruction uses `pypty` and is substantially more computationally expensive than the other analysis steps.

Run:

```bash
sbatch scripts/run_pypty.sh
```

The workflow consists of two stages:

```text
4d_stem.zarr
      │
      ▼
reconstruct_pypty_setup.py
      │
      ▼
      .h5
      │
      ▼
reconstruct_pypty.py
      │
      ▼
ptychographic reconstruction
```

The conversion step runs in `glass_sem_env`, while the reconstruction itself uses the separate `pypty` environment.

### Checkpointing

The reconstruction periodically writes checkpoints to:

```text
diffraction/ptycho_output/
```

including:

```text
co.npy
cp.npy
cg.npy
loss.csv
```

If the SLURM job is terminated or reaches its time limit, resubmitting:

```bash
sbatch scripts/run_pypty.sh
```

will resume from the latest saved epoch rather than restarting the reconstruction.

---

# 6. Comparison with ground truth

Once a reconstruction is complete:

```bash
python analysis/pypty_comparison.py
```

This compares the reconstructed object with the original MD structure used to generate the simulated diffraction data.

The comparison includes reconstructed phase/amplitude, the ground-truth structure, and their difference.

---

# Visualization

All reusable plotting functions are located in:

```text
visualization/plot.py
```

Important functions include:

```text
plot_map()
plot_diffraction()
plot_cross_correlation()
plot_pypty_reconstruction()
plot_pypty_vs_ground_truth()
```

3D animations are implemented in:

```text
visualization/animate.py
```

including:

```text
animate_holes_KDTree()
animate_holes_fft()
animate_defects()
```

Generated figures and animations are automatically written to:

```text
outputs/<script_name>/
```

For example:

```text
outputs/
└── pypty_comparison/
```

The `outputs/` directory is gitignored.

---

# Running on the cluster

Long-running calculations should be submitted through SLURM rather than executed directly in an SSH session.

### 4D-STEM

```bash
sbatch scripts/run_simulate.sh
```

### Ptychography

```bash
sbatch scripts/run_pypty.sh
```

Check running jobs with:

```bash
squeue -u $USER
```

Inspect a submitted job with:

```bash
scontrol show job <JOBID>
```

The SLURM scripts are designed so that the calculations continue running after disconnecting from the cluster.

---

# WSL2 memory requirement

The 4D-STEM simulation can require substantial RAM.

If running the project under WSL2, increase the WSL memory allocation if necessary by editing:

```text
C:\Users\<username>\.wslconfig
```

For example:

```ini
[wsl2]
memory=12GB
```

Then restart WSL:

```powershell
wsl --shutdown
```

The required memory depends on the simulation parameters and dataset size.

---

# Physical assumptions

Several choices in the current implementation are important for interpreting the results.

### Simulation cell thickness

The MD cell is approximately:

```text
49 × 49 × 49 Å
```

or roughly 5 nm per side.

This is substantially thinner than a typical TEM lamella. Consequently, the simulated 4D-STEM signal can contain relatively weak diffuse scattering.

The current diffraction workflow can repeat the structure along `z`:

```python
atoms.repeat((1, 1, 4))
```

to increase the effective scattering thickness. This assumes periodic repetition of the MD structure.

### Atomic radii

The current unified values are:

```text
Zr = 1.60 Å
Cu = 1.28 Å
```

These values are used consistently across the relevant modules.

If these values are changed, they should be changed in `config.py` rather than individual analysis scripts.

### Defects and free volume

The project treats free-volume regions and coordination defects as related but distinct structural descriptors.

A useful physical picture is that coordination-deficient atoms tend to occur near the boundaries of regions of increased free volume.

The cross-correlation analysis in `analysis/projection_maps.py` is intended to quantify this relationship rather than assume it a priori.

---

# Dependencies

The main scientific Python dependencies include:

* [ASE](https://wiki.fysik.dtu.dk/ase/)
* NumPy
* SciPy
* Matplotlib
* PyTorch
* PyTorch Geometric
* [abTEM](https://abtem.readthedocs.io/)

The complete dependency list is provided in:

```text
requirements.txt
```

The ptychographic reconstruction additionally requires the `pypty` environment.

---

# Git LFS

Large binary files should not be stored directly in normal Git history.

This repository uses **Git LFS** for large files such as:

```text
diffraction/ptycho_output/co.npy
```

If you clone the repository and need to work with LFS-tracked files, install Git LFS and initialize it:

```bash
git lfs install
```

Check which files are tracked by LFS with:

```bash
git lfs ls-files
```

Do not commit generated output from `outputs/` unless it is intentionally being versioned.

---

# Reproducibility

For reproducible results:

1. Use the specified Conda environment.
2. Keep the input trajectory unchanged.
3. Record the values in `config.py`.
4. Keep simulation parameters under version control.
5. Avoid modifying generated files manually.
6. Record the Git commit used for an analysis.

For computationally expensive simulations, the SLURM scripts should be treated as part of the reproducibility configuration because they define the requested computational resources.

---

# Project status

This repository is research code under active development. Some modules are experimental and may change as the analysis develops.

In particular:

* the GNN workflow is under development;
* simulation parameters may require further validation;
* ptychographic reconstruction is computationally expensive;
* some exploratory functionality remains in `scratch/`.

The `scratch/` directory is not required for the main analysis pipeline.

---

# License and citation

This project is currently research code associated with the study of metallic glasses and electron microscopy.

A formal license and citation information should be added before distributing the repository as a public research software release.
