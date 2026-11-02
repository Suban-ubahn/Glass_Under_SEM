# =============================================================
# scripts/run_d2min.py -- entry point (was the __main__ block of analysis/d2min/compute.py)
# =============================================================

import numpy as np
from analysis.d2min.compute import compute_d2min


if __name__ == "__main__":
    from data_io.trajectory import load_trajectory
    from cutoffs import CUTOFF_ZRZR, CUTOFF_ZRCU, CUTOFF_CUCU
    from visualization.plot import plot_d2min_histogram, plot_d2min_map

    trajectory = load_trajectory()
    print(f"Loaded {len(trajectory)} frames, {len(trajectory[0])} atoms each.")

    print(f"CUTOFF_ZRZR (config.py, g(r)-derived): {CUTOFF_ZRZR:.3f} A")
    print(f"CUTOFF_ZRCU (config.py, g(r)-derived): {CUTOFF_ZRCU:.3f} A")
    print(f"CUTOFF_CUCU (config.py, g(r)-derived): {CUTOFF_CUCU:.3f} A")

    # --- D^2_min between first and last frame ---
    # cutoff defaults to CUTOFF_DICT (species-resolved, minimum-based —
    # see compute_d2min's docstring), so no need to pass it explicitly.
    d2min, F_all = compute_d2min(trajectory[0], trajectory[-1])
    valid = ~np.isnan(d2min)
    print(f"\nD^2_min computed for {valid.sum()}/{len(d2min)} atoms "
          f"({(~valid).sum()} skipped, insufficient neighbors).")
    print(f"D^2_min range:  [{np.nanmin(d2min):.4f}, {np.nanmax(d2min):.4f}] A^2")
    print(f"D^2_min mean:   {np.nanmean(d2min):.4f} A^2")
    print(f"D^2_min median: {np.nanmedian(d2min):.4f} A^2")

    plot_d2min_histogram(d2min)
    plot_d2min_map(trajectory[0], d2min)
