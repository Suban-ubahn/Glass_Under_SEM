# =============================================================
# md/run.py -- running LAMMPS, checking the result, delivering the dump
#
#   launch_command(...)    the command line (serial, or MPI launcher + ranks)
#   run_lammps(...)        runs it; raises on a non-zero exit
#   inspect_dump(path)     frames / atoms / types / box of a LAMMPS dump (plain text parse)
#   check_dump(info, plan) raises ValueError if the dump does not match the plan
#   summarize_log(path)    per-stage temperature / energy statistics from the LAMMPS log
#   deliver_dump(src, dst) moves the checked dump to its final place (data/)
# =============================================================

import os
import shutil
import subprocess

import numpy as np


def launch_command(lammps_command, input_path, log_path, ranks=1, mpi_launcher="mpirun"):
    cmd = [lammps_command, "-in", input_path, "-log", log_path]
    if ranks > 1:
        cmd = [mpi_launcher, "-np", str(ranks)] + cmd
    return cmd


def run_lammps(cmd, env=None):
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def inspect_dump(path):
    """
    Parse a LAMMPS custom dump (columns id type x y z) without ASE.

    Returns dict: n_frames, n_atoms (list, one per frame), types (sorted list of the type
    values that occur), box (list of (3,) edge lengths per frame), timesteps.
    """
    n_atoms, timesteps, boxes, types = [], [], [], set()
    with open(path) as f:
        line = f.readline()
        while line:
            if line.startswith("ITEM: TIMESTEP"):
                timesteps.append(int(f.readline()))
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms.append(int(f.readline()))
            elif line.startswith("ITEM: BOX BOUNDS"):
                edges = []
                for _ in range(3):
                    lo, hi = map(float, f.readline().split()[:2])
                    edges.append(hi - lo)
                boxes.append(edges)
            elif line.startswith("ITEM: ATOMS"):
                cols = line.split()[2:]
                it = cols.index("type")
                for _ in range(n_atoms[-1]):
                    types.add(int(f.readline().split()[it]))
            line = f.readline()
    return {"n_frames": len(timesteps), "n_atoms": n_atoms, "types": sorted(types),
            "box": boxes, "timesteps": timesteps}


def check_dump(info, plan):
    """Raise ValueError unless the dump has the planned frames, atoms and types in a constant cell."""
    expected_frames = int(plan["settings"]["dump_n_frames"])
    if info["n_frames"] != expected_frames:
        raise ValueError(f"dump has {info['n_frames']} frames, expected {expected_frames}")
    if set(info["n_atoms"]) != {plan["n_atoms"]}:
        raise ValueError(f"dump atom counts {sorted(set(info['n_atoms']))}, expected {plan['n_atoms']}")
    if info["types"] != plan["types"]:
        raise ValueError(f"dump atom types {info['types']}, expected {plan['types']}")
    box = np.array(info["box"])
    if not np.allclose(box, box[0], atol=1e-9):
        raise ValueError("the cell changes between frames (production stage is meant to be fixed-cell)")


def summarize_log(log_path, n_atoms):
    """
    Per-stage statistics of the thermodynamic output of a LAMMPS log written by build_input().

    Returns {stage: dict(T_mean, T_std, pe_per_atom_start, pe_per_atom_end, pe_drift_eV_per_atom_per_ns,
    density_mean, pressure_mean_bar)}; the energy drift is a linear fit of pe/atom against time
    (needs at least 3 thermo rows), so e.g. the 'relax' and 'production' entries show whether the
    structure is still relaxing.
    """
    stages, current, header = {}, None, None
    with open(log_path) as f:
        for line in f:
            if line.startswith("STAGE "):
                current, header = line.split()[1], None
                stages[current] = []
            elif current and line.lstrip().startswith("Step "):
                header = line.split()
            elif current and header:
                parts = line.split()
                if len(parts) == len(header):
                    try:
                        stages[current].append([float(x) for x in parts])
                    except ValueError:
                        header = None
                else:
                    header = None
    out = {}
    for name, rows in stages.items():
        if not rows:
            continue
        a = np.array(rows)
        col = {"Time": 1, "Temp": 2, "PotEng": 3, "Press": 6, "Density": 8}
        t_ps, temp, pe = a[:, col["Time"]], a[:, col["Temp"]], a[:, col["PotEng"]]
        drift = np.nan
        if len(a) >= 3:
            drift = np.polyfit(t_ps, pe / n_atoms, 1)[0] * 1000.0     # eV/atom/ps -> eV/atom/ns
        out[name] = {"T_mean": temp.mean(), "T_std": temp.std(),
                     "pe_per_atom_start": pe[0] / n_atoms, "pe_per_atom_end": pe[-1] / n_atoms,
                     "pe_drift_eV_per_atom_per_ns": drift,
                     "density_mean": a[:, col["Density"]].mean(), "pressure_mean_bar": a[:, col["Press"]].mean()}
    return out


def deliver_dump(src, dst):
    """Move the checked dump to `dst` (creating its folder, replacing an existing file)."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.exists(dst):
        print(f"Replacing existing {dst}")
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
