# =============================================================
# scripts/run_d2min.py -- entry point (was the __main__ block of analysis/d2min/compute.py)
# =============================================================

import numpy as np
from analysis.d2min.compute import compute_d2min, mean_structure_positions


if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from ase.data import chemical_symbols
    from cutoffs import PARTIAL_CUTOFFS
    from visualization.plot import plot_d2min_histogram, plot_d2min_map

    trajectory = load_trajectory()
    print(f"Loaded {len(trajectory)} frames, {len(trajectory[0])} atoms each.")

    for (z_i, z_j), r in PARTIAL_CUTOFFS.items():
        print(f"{chemical_symbols[z_i]}-{chemical_symbols[z_j]} cutoff (cutoffs.py, g(r)-derived): {r:.3f} A")

    # --- D^2_min between the mean structure and the last frame ---
    # cutoff defaults to CUTOFF_DICT (species-resolved, minimum-based —
    # see compute_d2min's docstring), so no need to pass it explicitly.
    atoms_ref = trajectory[0].copy()
    atoms_ref.positions = mean_structure_positions(trajectory)
    d2min, F_all = compute_d2min(atoms_ref, trajectory[-1])
    valid = ~np.isnan(d2min)
    print(f"\nD^2_min computed for {valid.sum()}/{len(d2min)} atoms "
          f"({(~valid).sum()} skipped, insufficient neighbors).")
    print(f"D^2_min range:  [{np.nanmin(d2min):.4f}, {np.nanmax(d2min):.4f}] A^2")
    print(f"D^2_min mean:   {np.nanmean(d2min):.4f} A^2")
    print(f"D^2_min median: {np.nanmedian(d2min):.4f} A^2")

    plot_d2min_histogram(d2min)
    plot_d2min_map(trajectory[0], d2min)
