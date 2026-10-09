# =============================================================
# scripts/analyze_mass_dependence.py
#
# Numerical checks of the analytic mass-dependence argument:
#   1. frame-to-frame step size  <|du|^2>   ~ 1/m        (dynamics)
#   2. mean-squared displacement <|u|^2>    ~ k_B T/k    (statics, no m)
#   3. MSD ratio vs. window length (ballistic -> equilibrium crossover)
#   4. softness == MSD (top soft modes carry ~all the variance)
#   5. D^2_min = (thermal baseline) + (excess), per element
# =============================================================

import numpy as np
from ase.neighborlist import neighbor_list

from analysis.soft_modes import modes_from_displacements
from analysis.element_statistics import print_element_table
from config import CUTOFF_DICT
from data_io.trajectory import load_trajectory
from dynamics.flow_units import compute_d2min
from utils import unwrap_positions, compute_displacements

trajectory = load_trajectory()
atoms0 = trajectory[0]
elements = np.array(atoms0.get_chemical_symbols())
masses = atoms0.get_masses()
species = list(np.unique(elements))

positions = np.array([a.positions for a in trajectory])
box = np.array([a.cell.array.diagonal() for a in trajectory])
unwrapped = unwrap_positions(positions, box)
u, mean_positions = compute_displacements(unwrapped)        # (T, N, 3)


def by_el(x):
    return {el: float(np.nanmean(x[elements == el])) for el in species}


msd = (u ** 2).sum(-1).mean(0)                                # <|u_i|^2>
step = (np.diff(unwrapped, axis=0) ** 2).sum(-1).mean(0)      # <|du_i|^2> per frame

i_idx, j_idx = neighbor_list("ij", atoms0, CUTOFF_DICT)
cn = np.bincount(i_idx, minlength=len(atoms0))

print("\n-- 1/2. mass vs. statics/dynamics (last row = Cu/Zr ratio) --")
quantities = {
    "mass": masses,
    "du2 [1e-3]": step * 1e3,
    "u2 [1e-2]": msd * 1e2,
    "CN": cn.astype(float),
    "u2*CN [1e-2]": msd * cn * 1e2,
}
table = {el: {name: by_el(x)[el] for name, x in quantities.items()} for el in species}
if species == ["Cu", "Zr"]:
    table["Cu/Zr"] = {name: table["Cu"][name] / table["Zr"][name] for name in quantities}
print_element_table(table)
print("  (du2 = <|du|^2> per frame, u2 = <|u|^2> in A^2, CN = coordination number)")

print("\n-- 3. MSD ratio Cu/Zr vs. window length --")
for T in (10, 20, 40, 70, len(trajectory)):
    uT, _ = compute_displacements(unwrapped[:T])
    m = (uT ** 2).sum(-1).mean(0)
    print(f"  first {T:3d} frames: Cu/Zr = {m[elements=='Cu'].mean() / m[elements=='Zr'].mean():.2f}")
print(f"  (m_Zr/m_Cu = {masses[elements=='Zr'][0] / masses[elements=='Cu'][0]:.2f})")

print("\n-- 4. variance carried by the softest modes --")
lam = modes_from_displacements(u, masses=None, n_modes=100)["eigenvalues"]
print(f"  top 5 modes: {lam[:5].sum() / lam.sum():.2f}, top 20 modes: {lam[:20].sum() / lam.sum():.2f} of total variance")

print("\n-- 5. D^2_min = thermal baseline (n-3)<|d|^2> + excess --")
atoms_ref = atoms0.copy()
atoms_ref.positions = mean_positions
d2min, _ = compute_d2min(atoms_ref, trajectory[-1], cutoff=CUTOFF_DICT)
rel = ((u[:, j_idx] - u[:, i_idx]) ** 2).sum(-1).mean(0)       # <|u_j - u_i|^2> per pair
baseline = (cn - 3) * np.bincount(i_idx, weights=rel, minlength=len(atoms0)) / cn
print_element_table({
    el: {"D2min": by_el(d2min)[el], "baseline": by_el(baseline)[el],
         "excess": by_el(d2min)[el] - by_el(baseline)[el],
         "D2min/CN [1e-2]": by_el(d2min / cn)[el] * 1e2}
    for el in species
})