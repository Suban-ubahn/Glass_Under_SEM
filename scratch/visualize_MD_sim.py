from ase.io import read
from ase.visualize import view

# Load trajectory (all frames)
atoms = read("dump.lammpstrj", index=":")

# Open interactive viewer
view(atoms)