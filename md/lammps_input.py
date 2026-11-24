# =============================================================
# md/lammps_input.py -- config.MD_* settings -> plan -> LAMMPS input script
#
#   settings_from_config()   the MD_* constants of config.py as one dict
#   make_plan(settings)      validates the settings and derives everything the
#                            input needs (atom counts per type, box edge, steps of
#                            every stage); raises ValueError on inconsistent input
#   build_input(plan, ...)   the LAMMPS input script as a string
#   describe_plan(plan)      human-readable summary
#
# Nothing here is specific to a material: species and type numbering come from
# settings["type_to_z"], composition from settings["composition"], the force
# field from md/potentials.py.
#
# Initial structure: a simple-cubic lattice with as many sites as needed, surplus
# sites deleted at random, species assigned at random with the exact composition.
# Starting from a lattice (not random placement) guarantees that no two atoms
# start closer than the lattice spacing, which Buckingham potentials need.
# =============================================================

import math

from ase.data import atomic_masses, chemical_symbols

from md.potentials import get_potential, pair_parameters

AMU_TO_G = 1.66053906660e-24
CELL_AVERAGE_SAMPLE_EVERY = 10   # steps between box samples during the cell average


def settings_from_config():
    """The MD_* constants of config.py (and the dataset entry they point to) as one dict."""
    import config

    dataset = config.DATASETS[config.MD_DATASET]
    return {
        "dataset":              config.MD_DATASET,
        "trajectory_file":      dataset["trajectory_file"],
        "type_to_z":            dict(dataset["type_to_z"]),
        "potential":            config.MD_POTENTIAL,
        "composition":          dict(config.MD_COMPOSITION),
        "n_formula_units":      config.MD_N_FORMULA_UNITS,
        "density_g_cm3":        config.MD_DENSITY_G_CM3,
        "seed":                 config.MD_SEED,
        "timestep_fs":          config.MD_TIMESTEP_FS,
        "cutoff_A":             config.MD_CUTOFF_A,
        "kspace_accuracy":      config.MD_KSPACE_ACCURACY,
        "soft_start_ps":        config.MD_SOFT_START_PS,
        "soft_start_max_disp_A": config.MD_SOFT_START_MAX_DISP_A,
        "t_melt_K":             config.MD_T_MELT_K,
        "melt_ps":              config.MD_MELT_PS,
        "cooling_rate_K_per_ps": config.MD_COOLING_RATE_K_PER_PS,
        "t_final_K":            config.MD_T_FINAL_K,
        "pressure_bar":         config.MD_PRESSURE_BAR,
        "anneal_ps":            config.MD_ANNEAL_PS,
        "cell_average_ps":      config.MD_CELL_AVERAGE_PS,
        "relax_ps":             config.MD_RELAX_PS,
        "production_ensemble":  config.MD_PRODUCTION_ENSEMBLE,
        "dump_n_frames":        config.MD_DUMP_N_FRAMES,
        "dump_every_ps":        config.MD_DUMP_EVERY_PS,
        "t_damp_ps":            config.MD_T_DAMP_PS,
        "p_damp_ps":            config.MD_P_DAMP_PS,
        "thermo_every_ps":      config.MD_THERMO_EVERY_PS,
        "lammps_command":       config.MD_LAMMPS_COMMAND,
        "mpi_launcher":         config.MD_MPI_LAUNCHER,
        "mpi_ranks":            config.MD_MPI_RANKS,
    }


def make_plan(s):
    """Validate the settings dict and derive the quantities the LAMMPS input needs."""
    type_to_z = {int(t): int(z) for t, z in s["type_to_z"].items()}
    types = sorted(type_to_z)
    if types != list(range(1, len(types) + 1)):
        raise ValueError(f"type_to_z must number the LAMMPS types 1..n, got {types}")
    if len(set(type_to_z.values())) != len(type_to_z):
        raise ValueError(f"type_to_z maps two types to the same element: {type_to_z}")

    composition = {int(z): int(n) for z, n in s["composition"].items()}
    if set(composition) != set(type_to_z.values()):
        raise ValueError(f"composition elements {sorted(composition)} differ from the elements of the dataset "
                         f"{sorted(type_to_z.values())}")
    if any(n <= 0 for n in composition.values()):
        raise ValueError("composition counts must be positive")

    potential = get_potential(s["potential"])
    for z in composition:
        if z not in potential["charges"]:
            raise ValueError(f"potential '{s['potential']}' has no parameters for element {z}")
    for ia, a in enumerate(types):
        for b in types[ia:]:
            pair_parameters(potential, type_to_z[a], type_to_z[b])   # raises if missing

    n_fu = int(s["n_formula_units"])
    counts = {t: composition[type_to_z[t]] * n_fu for t in types}
    n_atoms = sum(counts.values())

    charge_per_fu = sum(potential["charges"][z] * n for z, n in composition.items())
    if abs(charge_per_fu) > 1e-6:
        raise ValueError(f"composition {composition} is not charge neutral with potential "
                         f"'{s['potential']}' (charge per formula unit {charge_per_fu:+.6f} e)")

    masses = {t: float(atomic_masses[type_to_z[t]]) for t in types}
    mass_amu = sum(masses[t] * counts[t] for t in types)
    volume_A3 = mass_amu * AMU_TO_G / s["density_g_cm3"] * 1e24
    box_A = volume_A3 ** (1.0 / 3.0)

    sites_per_edge = math.ceil(round(n_atoms ** (1.0 / 3.0), 9))
    n_sites = sites_per_edge ** 3

    if s["production_ensemble"] not in ("nve", "nvt"):
        raise ValueError("production_ensemble must be 'nve' or 'nvt'")
    if s["t_melt_K"] <= s["t_final_K"]:
        raise ValueError("t_melt_K must be larger than t_final_K")
    if s["cooling_rate_K_per_ps"] <= 0:
        raise ValueError("cooling_rate_K_per_ps must be positive")
    if s["cell_average_ps"] > s["anneal_ps"]:
        raise ValueError("cell_average_ps must not exceed anneal_ps")
    if s["dump_n_frames"] < 2:
        raise ValueError("dump_n_frames must be at least 2")

    dt_ps = s["timestep_fs"] / 1000.0

    def steps(ps):
        return max(1, int(round(ps / dt_ps)))

    cell_average_steps = max(CELL_AVERAGE_SAMPLE_EVERY,
                             CELL_AVERAGE_SAMPLE_EVERY * int(round(steps(s["cell_average_ps"]) / CELL_AVERAGE_SAMPLE_EVERY)))
    anneal_steps = max(steps(s["anneal_ps"]), cell_average_steps)
    dump_every_steps = steps(s["dump_every_ps"])
    quench_ps = (s["t_melt_K"] - s["t_final_K"]) / s["cooling_rate_K_per_ps"]

    stage_steps = [
        ("soft_start", steps(s["soft_start_ps"])),
        ("melt", steps(s["melt_ps"])),
        ("quench", steps(quench_ps)),
        ("anneal", anneal_steps - cell_average_steps),
        ("cell_average", cell_average_steps),
        ("relax", steps(s["relax_ps"])),
        ("production", (int(s["dump_n_frames"]) - 1) * dump_every_steps),
    ]

    return {
        "settings": dict(s),
        "type_to_z": type_to_z,
        "types": types,
        "counts": counts,
        "n_atoms": n_atoms,
        "masses": masses,
        "charges": {t: potential["charges"][type_to_z[t]] for t in types},
        "potential": potential,
        "box_A": box_A,
        "sites_per_edge": sites_per_edge,
        "n_sites": n_sites,
        "n_delete": n_sites - n_atoms,
        "dt_ps": dt_ps,
        "stage_steps": stage_steps,
        "dump_every_steps": dump_every_steps,
        "thermo_every_steps": steps(s["thermo_every_ps"]),
        "quench_ps": quench_ps,
    }


def describe_plan(plan):
    s = plan["settings"]
    z = plan["type_to_z"]
    lines = [
        f"dataset            : {s['dataset']}  ->  {s['trajectory_file']}",
        f"potential          : {s['potential']}",
        "atoms              : " + ", ".join(f"{plan['counts'][t]} x {chemical_symbols[z[t]]} (type {t})" for t in plan["types"])
                                + f"  = {plan['n_atoms']}",
        f"initial cell       : cubic, {plan['box_A']:.3f} A  ({s['density_g_cm3']} g/cm^3)",
        f"timestep           : {s['timestep_fs']} fs",
        "stages (steps, ps) :",
    ]
    total = 0
    for name, n in plan["stage_steps"]:
        total += n
        lines.append(f"    {name:<13s}{n:>11d}  {n * plan['dt_ps']:>10.1f} ps")
    lines.append(f"    {'total':<13s}{total:>11d}  {total * plan['dt_ps']:>10.1f} ps")
    lines.append(f"cooling            : {s['t_melt_K']:g} -> {s['t_final_K']:g} K at {s['cooling_rate_K_per_ps']:g} K/ps")
    lines.append(f"dump               : {s['dump_n_frames']} frames, every {plan['dump_every_steps']} steps")
    return "\n".join(lines)


def build_input(plan, dump_path):
    """LAMMPS input script for `plan`; the production dump is written to `dump_path`."""
    s = plan["settings"]
    types = plan["types"]
    stage = dict(plan["stage_steps"])
    L = plan["box_A"]
    M = plan["sites_per_edge"]
    t_damp = s["t_damp_ps"]
    p_damp = s["p_damp_ps"]
    seed = int(s["seed"])
    P = s["pressure_bar"]
    Tm, Tf = s["t_melt_K"], s["t_final_K"]

    o = []
    w = o.append

    w("# Generated by scripts/generate_md.py from config.MD_* -- edit config.py, not this file.")
    w("")
    w("units           metal")
    w("dimension       3")
    w("boundary        p p p")
    w("atom_style      charge")
    w("")
    w("# --- initial structure: cubic lattice, surplus sites deleted, species placed at random ---")
    w(f"lattice         sc {L / M:.10g}")
    w(f"region          box block 0 {M} 0 {M} 0 {M} units lattice")
    w(f"create_box      {len(types)} box")
    for t in types:
        w(f"mass            {t} {plan['masses'][t]:.8g}")
    w("create_atoms    1 box")
    if plan["n_delete"] > 0:
        w(f"delete_atoms    random count {plan['n_delete']} yes all NULL {seed + 1} compress yes")
    for k, t in enumerate(types[1:], start=2):
        w(f"set             type 1 type/subset {t} {plan['counts'][t]} {seed + 10 + k}")
    for t in types:
        w(f"set             type {t} charge {plan['charges'][t]:.8g}")
    w("")
    w("# --- interactions ---")
    w(f"pair_style      {plan['potential']['style']} {s['cutoff_A']:g}")
    for ia, a in enumerate(types):
        for b in types[ia:]:
            A, rho, C = pair_parameters(plan["potential"], plan["type_to_z"][a], plan["type_to_z"][b])
            w(f"pair_coeff      {a} {b} {A!r} {rho!r} {C!r}")
    w(f"kspace_style    pppm {s['kspace_accuracy']:g}")
    w("neighbor        2.0 bin")
    w("neigh_modify    every 1 delay 0 check yes")
    w("")
    w(f"timestep        {plan['dt_ps']:.10g}")
    w(f"thermo          {plan['thermo_every_steps']}")
    w("thermo_style    custom step time temp pe ke etotal press vol density lx")
    w("thermo_modify   flush yes")
    w(f"velocity        all create {Tm:g} {seed + 2} mom yes rot yes dist gaussian")
    w("")

    def run(name, body_open, body_close=()):
        w(f'print           "STAGE {name}"')
        for line in body_open:
            w(line)
        w(f"run             {stage[name]}")
        for line in body_close:
            w(line)
        w("")

    run("soft_start",
        [f"fix             softnve all nve/limit {s['soft_start_max_disp_A']:g}",
         f"fix             softT all temp/berendsen {Tm:g} {Tm:g} {t_damp:g}"],
        ["unfix           softnve", "unfix           softT"])
    run("melt",
        [f"fix             melt all nvt temp {Tm:g} {Tm:g} {t_damp:g}"],
        ["unfix           melt"])
    run("quench",
        [f"fix             quench all npt temp {Tm:g} {Tf:g} {t_damp:g} iso {P:g} {P:g} {p_damp:g}"],
        ["unfix           quench"])

    # anneal (NPT) in two runs under one fix; the box is averaged over the second one
    w(f"fix             anneal all npt temp {Tf:g} {Tf:g} {t_damp:g} iso {P:g} {P:g} {p_damp:g}")
    if stage["anneal"] > 0:
        w('print           "STAGE anneal"')
        w(f"run             {stage['anneal']}")
    n_samples = stage["cell_average"] // CELL_AVERAGE_SAMPLE_EVERY
    # fix ave/time only delivers its average on timesteps that are multiples of its window length
    # (Nfreq); restarting the step counter makes the last step of the run such a step.
    w("reset_timestep  0")
    w("variable        boxx equal lx")
    w("variable        boxy equal ly")
    w("variable        boxz equal lz")
    w(f"fix             cellavg all ave/time {CELL_AVERAGE_SAMPLE_EVERY} {n_samples} {stage['cell_average']} "
      f"v_boxx v_boxy v_boxz")
    w('print           "STAGE cell_average"')
    w(f"run             {stage['cell_average']}")
    w("variable        avgx equal $(f_cellavg[1])")
    w("variable        avgy equal $(f_cellavg[2])")
    w("variable        avgz equal $(f_cellavg[3])")
    w("unfix           cellavg")
    w("unfix           anneal")
    w('print           "Cell averaged over the last stage: ${avgx} ${avgy} ${avgz} A"')
    w("change_box      all x final 0 ${avgx} y final 0 ${avgy} z final 0 ${avgz} remap units box")
    w("")

    run("relax",
        [f"fix             relax all nvt temp {Tf:g} {Tf:g} {t_damp:g}"],
        ["unfix           relax"])

    w("reset_timestep  0")
    if s["production_ensemble"] == "nvt":
        prod_fix = f"fix             prod all nvt temp {Tf:g} {Tf:g} {t_damp:g}"
    else:
        prod_fix = "fix             prod all nve"
    w(f"dump            traj all custom {plan['dump_every_steps']} {dump_path} id type x y z")
    w("dump_modify     traj sort id first yes")
    run("production", [prod_fix])

    return "\n".join(o) + "\n"
