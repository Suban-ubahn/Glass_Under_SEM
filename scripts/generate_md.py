# =============================================================
# scripts/generate_md.py
#
# Melt-quench MD of the dataset config.MD_DATASET with LAMMPS (all parameters:
# config.py, MD_* block; method: md/). Steps:
#   1. write the LAMMPS input to outputs/generate_md_<tag>/in.lammps
#   2. run LAMMPS (log: outputs/generate_md_<tag>/log.lammps; production dump:
#      outputs/generate_md_<tag>/trj.raw)
#      (<tag> = name of the dataset MD_DATASET, config.py)
#   3. check the dump (frames, atoms, types, constant cell), print per-stage
#      temperature / energy statistics, move the dump to the dataset's
#      trajectory_file in data/
#
# Usage:
#   python scripts/generate_md.py                  all three steps
#   python scripts/generate_md.py --write-only     step 1 only (inspect the input)
#   python scripts/generate_md.py --finalize-only  step 3 only (LAMMPS was run by hand)
#   python scripts/generate_md.py --ranks N        MPI ranks (default config.MD_MPI_RANKS)
#
# Cluster: sbatch slurm/run_md.sh
# =============================================================

import argparse
import os

from md.lammps_input import build_input, describe_plan, make_plan, settings_from_config
from md.run import check_dump, deliver_dump, inspect_dump, launch_command, run_lammps, summarize_log
import config
from utils import get_output_dir

OUTPUT_DIR = get_output_dir(dataset=config.MD_DATASET)
INPUT_PATH = os.path.join(OUTPUT_DIR, "in.lammps")
LOG_PATH   = os.path.join(OUTPUT_DIR, "log.lammps")
RAW_DUMP   = os.path.join(OUTPUT_DIR, "trj.raw")


def finalize(plan):
    print("Checking dump...")
    info = inspect_dump(RAW_DUMP)
    check_dump(info, plan)
    print(f"  {info['n_frames']} frames x {plan['n_atoms']} atoms, types {info['types']}, "
          f"cell {info['box'][0][0]:.3f} x {info['box'][0][1]:.3f} x {info['box'][0][2]:.3f} A")

    print("Per-stage statistics from the LAMMPS log:")
    stats = summarize_log(LOG_PATH, plan["n_atoms"])
    for name, st in stats.items():
        print(f"  {name:<13s} T {st['T_mean']:8.1f} +- {st['T_std']:6.1f} K   "
              f"pe/atom {st['pe_per_atom_start']:9.4f} -> {st['pe_per_atom_end']:9.4f} eV   "
              f"drift {st['pe_drift_eV_per_atom_per_ns']:+9.4f} eV/atom/ns   "
              f"density {st['density_mean']:6.3f} g/cm^3   p {st['pressure_mean_bar']:+10.1f} bar")

    final_path = plan["settings"]["trajectory_file"]
    deliver_dump(RAW_DUMP, final_path)
    print(f"Dump written to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Melt-quench MD of config.MD_DATASET with LAMMPS (parameters: MD_* in config.py).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write-only", action="store_true", help="only write the LAMMPS input")
    mode.add_argument("--finalize-only", action="store_true", help="only check the existing dump and move it to data/")
    parser.add_argument("--ranks", type=int, default=None, help="MPI ranks (default: config.MD_MPI_RANKS)")
    args = parser.parse_args()

    settings = settings_from_config()
    plan = make_plan(settings)
    print(describe_plan(plan))

    if args.finalize_only:
        finalize(plan)
    else:
        with open(INPUT_PATH, "w") as f:
            f.write(build_input(plan, RAW_DUMP))
        print(f"LAMMPS input written to {INPUT_PATH}")
        if not args.write_only:
            ranks = args.ranks if args.ranks is not None else settings["mpi_ranks"]
            run_lammps(launch_command(settings["lammps_command"], INPUT_PATH, LOG_PATH, ranks, settings["mpi_launcher"]))
            finalize(plan)
