#!/bin/bash
# Melt-quench MD of config.MD_DATASET (settings: MD_* in config.py); the final dump goes to data/.
# Submit from the project root:   sbatch slurm/run_md.sh
# Other number of MPI ranks:      sbatch --ntasks=16 slurm/run_md.sh
# LAMMPS runs on the CPU only (no GPU is used, so no GPU guard like in the pypty/simulate scripts).
# Needs a LAMMPS build with the KSPACE package and MPI in glass_sem_env
# (e.g. conda install -c conda-forge lammps).
#SBATCH --job-name=glass_md
#SBATCH --output=slurm_md_%j.out
#SBATCH --error=slurm_md_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -e

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate glass_sem_env

cd "$SLURM_SUBMIT_DIR"   # submit from the project root

python -c "
import shutil
import config
exe = shutil.which(config.MD_LAMMPS_COMMAND)
assert exe, f'LAMMPS executable not found: {config.MD_LAMMPS_COMMAND} (set config.MD_LAMMPS_COMMAND)'
print('LAMMPS executable OK:', exe)
"

echo "== Running melt-quench MD (config.MD_*) on $SLURMD_NODENAME with $SLURM_NTASKS MPI ranks =="
python scripts/generate_md.py --ranks "$SLURM_NTASKS"

echo "== Verifying output =="
python -c "
import os
import config
path = config.DATASETS[config.MD_DATASET]['trajectory_file']
assert os.path.exists(path), f'Expected output {path} not found'
print('Trajectory created:', path)
"
