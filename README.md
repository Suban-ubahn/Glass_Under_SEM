# Glass Under SEM

Computational pipeline for studying the structure and dynamics of a Zr–Cu metallic glass under simulated electron-microscopy conditions.

The project combines **LAMMPS molecular-dynamics trajectories**, **free-volume (hole) analysis**, **graph-based structural defects**, **non-affine dynamics (D²min)**, **soft-mode / two-level-system (TLS) analysis**, **4D-STEM diffraction simulation** and **ptychographic reconstruction**, to ask how structural defects and dynamics relate to what an electron microscope can actually observe.

---

## Overview

```text
                         LAMMPS trajectory  (data/)
                                  │
      ┌───────────────┬───────────┼───────────────┬────────────────┐
      ▼               ▼           ▼               ▼                ▼
 Free volume     Structural   Non-affine      Soft modes /      4D-STEM
 (holes)         defects      dynamics        TLS               simulation
      │               │       (D²min)              │                │
      └───────┬───────┘           │                │                ▼
              ▼                   │                │          Ptychographic
     Projection maps &            │                │          reconstruction
     cross-correlation            │                │                │
              │                   │                │                │
              └───────────────────┴───────┬────────┴────────────────┘
                                          ▼
                       Comparisons (D²min vs defects / holes / soft spots,
                                    reconstruction vs ground truth)
```

---

## Quick start

```bash
git clone https://github.com/Suban-ubahn/Glass_Under_SEM.git
cd Glass_Under_SEM

conda create -n glass_sem_env python=3.11
conda activate glass_sem_env
pip install -r requirements.txt
pip install -e .          # (or: bash setup_env.sh, which does all of the above)
```

The editable install only puts the project root on `sys.path`, so that every script can `import config`, `import utils` and the packages (`holes`, `graph`, `analysis`, …) no matter where it is started from. Re-run it after pulling if a new top-level module appeared.

### Input data

The trajectory is **not tracked by git**. Put the LAMMPS dump here (the path is set by `TRAJECTORY_FILE` in `config.py`):

```text
data/trj.zr50cu50_6750_vib
```

### Smoke test

```bash
python scripts/run_d2min.py        # quick; writes outputs/run_d2min/
```

### Whole pipeline

```bash
bash scripts/run_pipeline.sh [--skip-diffraction] [--skip-animations]
```

---

## Conventions (read this first)

1. **Libraries vs. entry points.** The packages (`holes/`, `graph/`, `analysis/`, `diffraction/`, `visualization/`, `data_io/`) only contain functions. They have no `__main__` blocks. Everything you *run* lives in `scripts/` (and `slurm/` for cluster jobs).
2. **Computation is separate from plotting.** Computation lives in `analysis/`, `holes/`, `graph/`, `diffraction/`. Every plotting function is in `visualization/plot.py`, animations in `visualization/animate/`.
3. **Every script writes only to `outputs/<script_name>/`.** `python scripts/run_d2min.py` writes into `outputs/run_d2min/`, nothing else. See [Outputs](#outputs).
4. **Inputs go in `data/`, generated files in `outputs/`.** Both are gitignored.
5. **Constants live in `config.py`.** It contains only constants and is cheap to import in any environment (including the `pypty` env). Neighbour cutoffs derived from g(r) are computed in `cutoffs.py`, because importing that module reads the trajectory (cached in `.cutoff_cache.json`).

---

## Project structure

```text
Glass_Under_SEM/
├── config.py                 constants only (paths, radii, thresholds, simulation parameters)
├── cutoffs.py                g(r)-derived neighbour cutoffs (CUTOFF_ZRZR, PEAK_ZRCU, …)
├── utils.py                  shared helpers: output folders, g(r), unwrapping, cutoff cache
│
├── data_io/trajectory.py     load_trajectory() – the one place the dump file is read
│
├── holes/
│   ├── kdtree.py             voxel / KDTree free-volume detection
│   └── fft.py                FFT-convolution occupancy and hole field
│
├── graph/
│   ├── build.py              ASE Atoms → PyTorch-Geometric graph
│   ├── defects.py            coordination defects, clustering, tracking
│   └── gnn.py                GNN predicting atomic displacements
│
├── analysis/
│   ├── d2min/                compute.py (Falk–Langer D²min), comparison.py (vs defects / holes)
│   ├── soft_modes/           modes.py (quasi-harmonic soft modes), tls.py (TLS detection),
│   │                         source.py (soft atoms as sources of D²min), comparison.py,
│   │                         element_stats.py, pca_reliability.py (how far the PCA can be trusted)
│   ├── projection_maps.py    time-integrated hole/defect maps, cross-correlation
│   └── pypty_comparison.py   reconstruction vs ground-truth projected potential
│
├── diffraction/simulate.py   abTEM 4D-STEM / plane-wave simulation
│
├── visualization/
│   ├── plot.py               ALL static plotting functions
│   └── animate/              holes.py, defects.py, d2min.py, softness.py
│
├── scripts/                  every entry point (see the table below)
├── slurm/                    SLURM batch scripts: run_simulate.sh, run_pypty.sh, _gpu_guard.sh
│
├── data/                     input trajectory            (gitignored)
└── outputs/                  everything generated        (gitignored)
```

---

## Scripts and where their results go

| Script | What it does | Output folder |
| --- | --- | --- |
| `run_d2min.py` | g(r) cutoffs, D²min histogram and map | `outputs/run_d2min/` |
| `run_defect_stats.py` | graph defect statistics | `outputs/run_defect_stats/` |
| `build_projection_maps.py` | cumulative hole / defect maps (`.npy`) + plots + cross-correlation | `outputs/build_projection_maps/` |
| `view_projection_maps.py` | re-plot the saved maps without recomputing | `outputs/view_projection_maps/` |
| `run_d2min_defect_comparison.py` | D²min vs defect clusters (plot + overlay animation) | `outputs/run_d2min_defect_comparison/` |
| `run_d2min_hole_comparison.py` | D²min vs distance to holes (plot + overlay animation) | `outputs/run_d2min_hole_comparison/` |
| `run_soft_mode_tls_pipeline.py` | soft modes (3D and per z-slice), TLS detection (two methods), soft vs TLS vs D²min, per-element statistics, soft atoms as sources of D²min, plots and animations | `outputs/run_soft_mode_tls_pipeline/` |
| `run_pca_reliability.py` | split-half reliability of the PCA behind the soft modes: eigenvalues and softness map (see [5.1](#51-how-far-can-the-pca-be-trusted)) | `outputs/run_pca_reliability/` |
| `run_soft_tls_dmin_comparison.py` | soft spots vs TLS vs D²min | `outputs/run_soft_tls_dmin_comparison/` |
| `build_holes_defects_animations.py` | hole (KDTree, FFT) and defect-cluster animations | `outputs/build_holes_defects_animations/` |
| `build_d2min_animations.py` | 2D / 3D D²min animations | `outputs/build_d2min_animations/` |
| `simulate_4dstem.py` | 4D-STEM simulation + virtual BF/DF images | `outputs/simulate_4dstem/` (`4d_stem.zarr`) |
| `pypty_setup.py` | zarr → `.h5` for pypty (`glass_sem_env`) | `outputs/pypty_setup/` |
| `pypty_reconstruct.py` | iterative reconstruction (`pypty` env), resumable | `outputs/pypty_reconstruct/` |
| `pypty_comparison.py` | reconstruction vs ground truth | `outputs/pypty_comparison/` |
| `train_gnn.py` | GNN displacement prediction | `outputs/train_gnn/` |
| `analyze_mass_dependence.py`, `explore_modes_from_covariance.py` | numerical checks of soft-mode theory | `outputs/<script_name>/` |
| `demo_holes_kdtree.py`, `demo_holes_fft.py`, `demo_graph_build.py`, `demo_soft_modes.py`, `demo_tls_detection.py` | quick demos of one library module | `outputs/<script_name>/` |
| `param_sweep.sh` | sweep of 4D-STEM simulation parameters | `outputs/param_sweep/` |
| `run_pipeline.sh` | runs the main stages in order | one folder per stage |

All scripts are run from anywhere inside the project, e.g. `python scripts/run_defect_stats.py`.

---

## Analysis stages

### 1. Free volume (holes)

`holes/kdtree.py` builds a voxel grid, marks voxels farther than `R_CUT_VOID` from every atom as void and labels connected void regions (fast, direct geometric meaning). `holes/fft.py` convolves the atoms with element-specific spherical kernels via FFT and finds peaks in the inverted occupancy field. The two give complementary views of the same free volume. Quick looks: `scripts/demo_holes_kdtree.py`, `scripts/demo_holes_fft.py`.

### 2. Structural defects

`graph/defects.py` turns atoms into a graph (edges within the species-resolved cutoffs from `cutoffs.py`), compares each atom's coordination with the mean of its element and flags atoms that fall sufficiently far below it (`DEFECT_THRESHOLD`, `DEFECT_K` in `config.py`). Connected defective atoms form clusters that can be tracked over time. `python scripts/run_defect_stats.py`.

### 3. Non-affine dynamics (D²min)

`analysis/d2min/compute.py` implements the Falk–Langer local non-affine squared displacement between two frames of a purely thermal trajectory. `python scripts/run_d2min.py`, then the comparisons `run_d2min_defect_comparison.py` / `run_d2min_hole_comparison.py`.

### 4. Projection maps and cross-correlation

`python scripts/build_projection_maps.py` integrates holes and defect clusters over the trajectory, projects them on the xy plane, saves them (`outputs/build_projection_maps/cumulative_*_map.npy`) and cross-correlates them. `scripts/view_projection_maps.py` reloads the saved maps. The abTEM virtual-detector image from step 5 is compared with the same maps by `analysis.projection_maps.compare_abtem_to_projection_maps`.

### 5. Soft modes and TLS

`analysis/soft_modes/` covers the quasi-harmonic soft-mode ("soft spot") analysis, two-level-system (TLS) detection by two independent methods, and their comparison with each other and with D²min.

* **Soft modes** (`modes.py`). PCA of the mass-weighted displacement matrix (frames × 3N coordinates, displacements from the time-averaged structure). The eigenvalue of a mode is the mean squared amplitude of the trajectory along it, so a **large eigenvalue = a soft mode**. The per-atom softness is `s_i = Σ_n λ_n p_{i,n}` over the `N_SOFT_MODES` softest modes (`p_{i,n}` = the share of mode n that sits on atom i; modes can be restricted to quasi-localized ones with `SOFT_MODE_PR_MAX`).
* **TLS, Method A** (`detect_tls_direct`, real space). An atom's position relative to the centroid of its own coordination cage is tested for hopping between two recurring positions (one vs two Gaussians by BIC, with an autocorrelation-corrected number of independent frames; thresholds `TLS_*` in `config.py`).
* **TLS, Method B** (`detect_tls_amplitude`, mode space). The amplitude distribution P(a_n) of each soft mode is tested for two peaks.
* **Comparisons** (`comparison.py`, `source.py`). Soft spots vs TLS; whether softness or TLS-ness predicts where D²min is high (ROC / AUC); whether soft atoms are *sources* of plastic activity (where D²min is elevated, and whether it rises there first), always against a thermal baseline and a null model that permutes the soft labels within (element, coordination number) groups.

`python scripts/run_soft_mode_tls_pipeline.py` runs everything below; the thresholds are in `config.py`.

| Stage | Main files in `outputs/run_soft_mode_tls_pipeline/` |
| --- | --- |
| 1. 3D soft modes | `participation_spectrum_3d.png`, `participation_per_mode_<element>.png`, `softness_map_3d_flat.png`, `softness_map_3d_pr_weighted.png` |
| 2. 2D per z-slice | `softness_map_2d_overview.png`, `softness_map_2d_overview_pr_weighted.png` |
| 3. TLS detection | `mode_amplitude_distributions_3d.png` (P(a_n) and a_n(t) of the softest modes), `example_tls_mode_amplitude.png` (only if Method B flags a mode); for every atom found by Method A, `tls_candidate_atom_<i>_softest_modes.png` and `tls_candidate_atom_<i>_strongest_modes.png` |
| 4. Comparisons | `soft_vs_tls.png`, `roc_soft_vs_d2min.png`, `roc_tls_vs_d2min.png` |
| 5. Per-element statistics | `distribution_<field>_by_element_average.png` and `distribution_<field>_by_element.mp4` for `<field>` = `softness`, `softness_pr_weighted`, `d2min`; `d2min_softness_side_by_side_3d.mp4`, `d2min_softness_maps_side_by_side_3d.mp4` |
| 6. Soft atoms as sources | `d2min_by_soft_neighbours.png` |

The candidate grids of stage 3 have the same layout as the P(a_n) grid. The first panel is the atom's cage coordinate along the axis connecting the two Gaussians Method A found (what Method A actually tested), the second panel its direction of largest variance, and the remaining panels are covariance modes: the 10 softest ones, or the 10 that carry most of that atom's own motion (`mode_selection="softest"` / `"largest_swing"` in `candidate_diagnostics`; further selection rules are added in `MODE_SELECTORS` in `tls.py`). The candidate plots are produced for whatever Method A flags, so they work unchanged on another dataset.

#### 5.1 How far can the PCA be trusted?

The covariance matrix has 3N = 20250 rows but is estimated from about 100 frames, so it has at most T−1 non-zero eigenvalues, and the SVD returns those modes for *any* data, noise included. Each eigenvalue is measured on the same frames that were used to choose its mode, which inflates it (for pure noise by roughly the ratio 3N/T). `python scripts/run_pca_reliability.py` quantifies this with two split-half tests (first half of the frames against the second half):

1. **Eigenvalues.** Find the modes on one half, project the frames of the other half on them, and compare the mean squared amplitude there (the *held-out variance*) with the eigenvalue found on the first half. Their ratio, the *retained fraction*, is about 1 for a real mode and about T_half/3N for fitted noise. Chance level per mode: the same procedure on data in which every coordinate is shifted in time by its own random amount (independent coordinates, same variance and autocorrelation each).
2. **Softness map.** The whole soft-mode analysis is run on each half; the two per-atom softness maps are compared by Spearman rank correlation and by the overlap of their softest 5 % of atoms, against the same null and against the plain per-atom mean-square-displacement map (no PCA).

The definitions, the null model and the interpretation are written out in the header of `analysis/soft_modes/pca_reliability.py`. Both tests print a summary table and write `pca_reliability.png` and `softness_reproducibility.png` to `outputs/run_pca_reliability/`; they take about a minute for 6750 atoms and 101 frames.

### 6. 4D-STEM simulation

```bash
python scripts/simulate_4dstem.py          # locally (memory heavy, see the WSL2 note)
sbatch slurm/run_simulate.sh               # on the cluster
```

abTEM `GridScan` with a `PixelatedDetector`; the dataset is saved as `outputs/simulate_4dstem/4d_stem.zarr`. To look at an existing dataset without rerunning:

```python
import abtem
from visualization.plot import plot_average_diffraction

dp = abtem.from_zarr("outputs/simulate_4dstem/4d_stem.zarr")
plot_average_diffraction(dp, blocked=True)      # saved to outputs/<your script>/ by default
```

### 7. Ptychographic reconstruction

```text
outputs/simulate_4dstem/4d_stem.zarr
        │  scripts/pypty_setup.py        (glass_sem_env)
        ▼
outputs/pypty_setup/4d_stem.h5 + scan_metadata.json
        │  scripts/pypty_reconstruct.py  (pypty env)
        ▼
outputs/pypty_reconstruct/   (co.npy object, cp.npy probe, cg.npy positions, loss.csv, …)
        │  scripts/pypty_comparison.py   (glass_sem_env)
        ▼
outputs/pypty_comparison/    (comparison with the MD ground truth)
```

Both pypty scripts import only `config`, so they run in the `pypty` env without the editable install. The reconstruction writes a checkpoint every epoch; if the job is killed or hits its time limit, run it again and it resumes from the last saved epoch.

---

## Outputs

* Every script writes into `outputs/<script_name>/` (the name of the script file without extension) and nowhere else. The folder is created on demand and removed again when empty.
* Code that is not a script (`python -c …`, `python -`, a REPL, a notebook) writes to `outputs/miscellaneous/`.
* A script reads the results of earlier stages from *their* folders (for example `pypty_setup.py` reads `outputs/simulate_4dstem/4d_stem.zarr`).
* Shell scripts that feed Python through a heredoc (such as `param_sweep.sh`) pick their folder with `export GLASS_OUTPUT_NAME=<name>`.
* Plot functions default to `save_path=None`, which means "`<OUTPUT_DIR>/<default file name>`". Pass `save_path=` to write somewhere else.
* `outputs/miscellaneous/` is also the place for old results worth keeping that belong to no script. Any other folder under `outputs/` that does not match a file in `scripts/` is stale and can be deleted.

To add a new analysis: put the computation in the appropriate package, the plotting in `visualization/plot.py`, and write a small entry point `scripts/<name>.py`. Its results will appear in `outputs/<name>/` automatically.

---

## Running on the cluster

Long calculations go through SLURM, not a foreground SSH session. **Submit from the project root** (the scripts source `slurm/_gpu_guard.sh` relative to it).

```bash
sbatch slurm/run_simulate.sh                               # 4D-STEM simulation
sbatch --dependency=afterok:<SIMULATE_JOBID> slurm/run_pypty.sh   # pypty setup + reconstruction
squeue -u $USER
scontrol show job <JOBID>
```

`run_pypty.sh` needs the simulation's output, so chain it with `--dependency=afterok`. Both scripts check GPU memory at start-up (`slurm/_gpu_guard.sh`); if the node's GPU is too full they exclude that node and requeue themselves. Logs are written to `slurm_*_<jobid>.out/.err` in the submit directory (gitignored). The reconstruction is expected to run for days, so `run_pypty.sh` uses the `defq` partition (7-day limit).

Do not `git pull` on the cluster while a job is running: the job reads and writes files under `outputs/`.

---

## WSL2 memory requirement

The 4D-STEM simulation can need a lot of RAM. Under WSL2, raise the allocation in `C:\Users\<username>\.wslconfig`:

```ini
[wsl2]
memory=12GB
```

then run `wsl --shutdown` from PowerShell.

---

## Physical assumptions

* **Cell size.** The MD cell is about 49 × 49 × 49 Å (≈ 5 nm), much thinner than a TEM lamella, so the simulated diffuse scattering is weak. The diffraction workflow can repeat the cell along `z` (`atoms.repeat((1, 1, 4))`), which assumes periodicity.
* **Atomic radii.** Zr = 1.60 Å, Cu = 1.28 Å (`R_ZR`, `R_CU` in `config.py`) – change them there only.
* **Free volume vs defects.** They are treated as related but distinct descriptors; the cross-correlation analysis is meant to measure their relation, not assume it.
* **Few frames for the PCA.** The soft-mode analysis decomposes a 3N × 3N covariance matrix estimated from about 100 frames, so only the strongest modes can be resolved and the eigenvalues are inflated. Run `scripts/run_pca_reliability.py` before interpreting individual modes or the softness map.
* **Unwrapped positions.** The LAMMPS dumps re-wrap coordinates into the cell every frame; displacement-based analyses (D²min, soft modes) use `utils.unwrap_positions` to remove periodic jumps.

---

## Dependencies

ASE, NumPy, SciPy, Matplotlib, PyTorch, PyTorch Geometric, [abTEM](https://abtem.readthedocs.io/), h5py, zarr (full pinned list: `requirements.txt`). Ptychographic reconstruction additionally needs `pypty` (Anton Gladyshev's ptychography library) in its own `pypty` conda environment (installed under `external/pypty`, gitignored).

---

## Reproducibility

1. Use the pinned environment (`requirements.txt`).
2. Keep the input trajectory unchanged.
3. Change parameters only in `config.py`, and record its values with the run.
4. Record the git commit used for an analysis.
5. Treat the SLURM scripts as part of the configuration; they define the requested resources.
6. Don't edit generated files by hand; regenerate them.

---

## Project status

Research code under active development. Some modules are experimental (the GNN workflow, soft-mode/TLS thresholds), the simulation parameters need further validation, and exploratory code lives in `scratch/` (gitignored, not part of the pipeline). No licence has been chosen yet; add one and citation information before releasing the code publicly.