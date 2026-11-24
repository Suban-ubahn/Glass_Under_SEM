# =============================================================
# md/potentials.py -- interatomic potentials for scripts/generate_md.py
#
# One entry per potential, keyed by the name used in config.MD_POTENTIAL.
# Everything is keyed by REAL atomic numbers, so the LAMMPS type numbering of a
# dataset (config.DATASETS[...]["type_to_z"]) can be chosen freely.
#
#   style      LAMMPS pair style the parameters are written for
#   charges    {Z: partial charge in units of e}
#   buckingham {(Z_i, Z_j): (A [eV], rho [Å], C [eV Å^6])}, Z_i <= Z_j:
#                  V(r) = A exp(-r/rho) - C / r^6   (+ Coulomb, handled by kspace)
# =============================================================

POTENTIALS = {
    # Matsui, Mineral. Mag. 58A, 571 (1994) -- fixed partial charges, Buckingham short
    # range. The pair parameters are those tabulated in Table I of arXiv:1411.0529
    # ("force field I"); they agree with the A/B/C ion parameters of Matsui's original
    # combination-rule table (rho_ij = B_i + B_j, C_ij = C_i C_j, A_ij = f rho_ij exp((A_i + A_j)/rho_ij),
    # f = 4.184 kJ/mol/A). Charge neutral for Al2O3 (2 x 1.4175 = 3 x 0.945).
    "matsui_al2o3": {
        "style": "buck/coul/long",
        "charges": {13: 1.4175, 8: -0.9450},
        "buckingham": {
            (13, 13): (31570911.694, 0.068, 14.051),
            (8, 13):  (28476.897,    0.172, 34.578),
            (8, 8):   (6462.668,     0.276, 85.092),
        },
    },
}


def get_potential(name):
    if name not in POTENTIALS:
        raise ValueError(f"unknown potential '{name}'; available: {sorted(POTENTIALS)}")
    return POTENTIALS[name]


def pair_parameters(potential, z_i, z_j):
    """(A, rho, C) of the Buckingham term between the elements z_i and z_j (either order)."""
    key = (min(z_i, z_j), max(z_i, z_j))
    if key not in potential["buckingham"]:
        raise ValueError(f"the potential has no parameters for the element pair {key}")
    return potential["buckingham"][key]
