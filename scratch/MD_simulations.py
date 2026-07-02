from lammps import lammps

lmp = lammps()

lmp.commands_string("""

# ---------- Initialization ----------
units           lj
dimension       3
boundary        p p p
atom_style      atomic

# ---------- Create atoms ----------
lattice         fcc 0.8442
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box

mass            1 1.0

# ---------- Interatomic potential ----------
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5

# ---------- Neighbor settings ----------
neighbor        0.3 bin
neigh_modify    delay 0 every 1

# ---------- Output ----------
thermo          100
thermo_style    custom step temp pe etotal press

dump            1 all atom 100 dump.lammpstrj

# ---------- Initial velocities ----------
velocity        all create 3.0 12345

# ---------- Melt ----------
fix             1 all nvt temp 3.0 3.0 1.0
timestep        0.005
run             5000

# ---------- Quench ----------
fix             1 all nvt temp 3.0 0.1 1.0
run             10000

# ---------- Relax ----------
unfix           1
fix             2 all nvt temp 0.1 0.1 1.0
run             5000

# ---------- Output final structure ----------
write_data      glass.data

""")