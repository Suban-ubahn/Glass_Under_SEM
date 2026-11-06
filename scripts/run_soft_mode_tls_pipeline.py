# =============================================================
# scripts/run_soft_mode_tls_pipeline.py
#
# End-to-end pipeline for the soft-spot / TLS analysis:
#   1. 3D full-system soft-mode analysis
#   2. 2D per-z-slice soft-mode analysis (N_Z_SLICES slabs,
#      CAVIAR-style projected slices)
#   3. TLS detection: Method A (direct/structural) and Method B
#      (mode-amplitude bimodality), in 3D and per-slice
#   4. Compare: soft spots vs TLS, soft spots vs D^2_min,
#      TLS vs D^2_min
#   5. Element statistics, distributions, animations
#   6. Are soft atoms sources of plastic activity? (where + when)
#
# Outputs go to outputs/run_soft_mode_tls_pipeline/ (see
# utils.get_output_dir(), same convention as the rest of the project,
# e.g. scripts/build_d2min_animations.py).
# =============================================================

import os

import numpy as np

from analysis.soft_modes.modes import (
    soft_mode_analysis_3d, soft_mode_analysis_2d_slices, softness_trajectory,
)
from analysis.soft_modes.tls import (
    detect_tls_direct, detect_tls_direct_2d_slices, detect_tls_amplitude,
    bimodality_test_batch, mode_amplitude_diagnostics, candidate_diagnostics,
)
from analysis.soft_modes.comparison import run_full_comparison
from analysis.soft_modes.source import soft_source_analysis
from analysis.soft_modes.element_stats import (
    get_element_symbols, score_by_element, mode_dominant_elements,
    persistence_above_threshold, persistence_by_element, print_element_table,
)
from config import (
    TRAJECTORY_FILE,
    N_Z_SLICES,
    N_SOFT_MODES,
    SOFT_MODE_PR_MAX,
    TLS_DELTA_BIC_THRESHOLD,
    TLS_SEPARATION_THRESHOLD,
    TLS_MIN_WEIGHT_FRACTION,
    TLS_MIN_SWITCHES,
    D2MIN_QUANTILE_THRESHOLD,
    D2MIN_TOP_FRACTION,
    MAX_SOFT_NB,
)
from cutoffs import CUTOFF_DICT
from data_io.trajectory import load_trajectory
from analysis.d2min.compute import d2min_trajectory
from utils import get_output_dir, unwrap_positions, compute_displacements
from visualization.plot import (
    plot_participation_spectrum, plot_mode_atom_distribution_by_element, plot_softness_map,
    plot_mode_amplitude_distribution, plot_roc_curve, plot_soft_tls_scatter,
    plot_distribution_by_element, plot_d2min_by_soft_neighbours, plot_mode_amplitude_grid,
)
from visualization.animate import (
    animate_distribution_by_element, animate_d2min_softness_side_by_side,
)

OUTPUT_DIR = get_output_dir()


if __name__ == "__main__":
    trajectory = load_trajectory(TRAJECTORY_FILE)
    n_atoms = len(trajectory[0])
    elements = get_element_symbols(trajectory[0])
    print(f"Loaded {len(trajectory)} frames, {n_atoms} atoms "
          f"({dict(zip(*np.unique(elements, return_counts=True)))}).")

    # --- 1. 3D soft-mode analysis ---
    print("\n=== 3D soft-mode analysis ===")
    modes_3d = soft_mode_analysis_3d(
        trajectory, n_soft_modes=N_SOFT_MODES, pr_max=SOFT_MODE_PR_MAX,
    )
    dominant_elements_3d, dominant_tally_3d = mode_dominant_elements(
        modes_3d["eigenvectors"], elements, mode_mask=modes_3d["soft_mode_mask"],
    )
    print(f"  dominant element of the soft/localized modes: {dominant_tally_3d}")
    plot_participation_spectrum(
        modes_3d["eigenvalues"], modes_3d["participation"],
        soft_mode_mask=modes_3d["soft_mode_mask"], elements=dominant_elements_3d,
        save_path=os.path.join(OUTPUT_DIR, "participation_spectrum_3d.png"),
    )
    # with which atoms is each mode populated? one figure per element
    plot_mode_atom_distribution_by_element(
        modes_3d["eigenvectors"], elements, soft_mode_mask=modes_3d["soft_mode_mask"],
        save_dir=OUTPUT_DIR,
    )
    plot_softness_map(trajectory[0], modes_3d["softness"],
                       save_path=os.path.join(OUTPUT_DIR, "softness_map_3d_flat.png"),
                       title="Soft-Spot Map (3D analysis, x-y projection)")
    plot_softness_map(trajectory[0], modes_3d["softness_pr_weighted"],
                       save_path=os.path.join(OUTPUT_DIR, "softness_map_3d_pr_weighted.png"),
                       title="Soft-Spot Map (3D, participation-weighted)")

    # --- 2. 2D per-slice soft-mode analysis ---
    print(f"\n=== 2D soft-mode analysis ({N_Z_SLICES} slices) ===")
    modes_2d = soft_mode_analysis_2d_slices(
        trajectory, n_slices=N_Z_SLICES,
        n_soft_modes=N_SOFT_MODES, pr_max=SOFT_MODE_PR_MAX,
    )
    # One combined overview (all atoms, colored by whichever slice
    # they belong to) ...
    plot_softness_map(trajectory[0], modes_2d["softness"],
                       save_path=os.path.join(OUTPUT_DIR, "softness_map_2d_overview.png"),
                       title="Soft-Spot Map (2D per-slice analysis, all slices combined)")
    plot_softness_map(trajectory[0], modes_2d["softness_pr_weighted"],
                       save_path=os.path.join(OUTPUT_DIR, "softness_map_2d_overview_pr_weighted.png"),
                       title="Soft-Spot Map (2D per-slice, participation-weighted, all slices combined)")
    
    # --- 3a. TLS detection: Method A (direct/structural), 3D ---
    # Only test the top 5% softest atoms from step 1 -- TLS are
    # expected to be rare, and running the bistability fit on all
    # n_atoms would mostly be spent confirming ordinary atoms aren't
    # bistable. Drop atom_indices (or pass None) to scan everyone.
    print("\n=== TLS detection: Method A (direct/structural), 3D ===")
    n_candidates = max(50, n_atoms // 20)
    candidates = modes_3d["softness"].argsort()[::-1][:n_candidates]
    tls_A_3d = detect_tls_direct(
        trajectory, atom_indices=candidates, cutoff=CUTOFF_DICT,
        delta_bic_threshold=TLS_DELTA_BIC_THRESHOLD,
        separation_threshold=TLS_SEPARATION_THRESHOLD,
        min_weight_fraction=TLS_MIN_WEIGHT_FRACTION,
        min_switches=TLS_MIN_SWITCHES,
    )
    print(f"  {tls_A_3d['is_tls'].sum()} / {len(tls_A_3d['atom_indices'])} "
          f"candidate atoms flagged as TLS (direct method).")

    # one P(a_n) / a_n(t) grid per structure-based TLS candidate (none -> nothing is drawn)
    for atom in np.where(tls_A_3d["is_tls"])[0]:
        for sel, tag in [("softest", "softest_modes"), ("largest_swing", "strongest_modes")]:
            cand_diag = candidate_diagnostics(tls_A_3d, modes_3d, atom, n_modes=10,
                                              mode_selection=sel)
            plot_mode_amplitude_grid(
                cand_diag,
                save_path=os.path.join(OUTPUT_DIR, f"tls_candidate_atom_{atom}_{tag}.png"))

    # --- 3b. TLS detection: Method B (mode-amplitude bimodality), 3D ---
    print("\n=== TLS detection: Method B (mode-amplitude bimodality), 3D ===")
    tls_B_3d = detect_tls_amplitude(
        modes_3d,
        delta_bic_threshold=TLS_DELTA_BIC_THRESHOLD,
        separation_threshold=TLS_SEPARATION_THRESHOLD,
        min_weight_fraction=TLS_MIN_WEIGHT_FRACTION,
        min_switches=TLS_MIN_SWITCHES,
    )
    print(f"  {tls_B_3d['is_tls_mode'].sum()} / {len(tls_B_3d['checked_modes'])} "
          f"checked soft modes flagged as bimodal (TLS-like).")
    
    amp_diag = mode_amplitude_diagnostics(modes_3d, n_modes=20)
    plot_mode_amplitude_grid(
        amp_diag, save_path=os.path.join(OUTPUT_DIR, "mode_amplitude_distributions_3d.png"))

    if len(tls_B_3d["tls_mode_idx"]) > 0:
        example_mode = int(tls_B_3d["tls_mode_idx"][0])
        a_n = modes_3d["amplitudes"][:, example_mode][None, :, None]
        fit = bimodality_test_batch(a_n)
        gmm_fit = {"means": fit["means"][0], "sigmas": fit["sigmas"][0], "weights": fit["weights"][0]}
        plot_mode_amplitude_distribution(
            modes_3d["amplitudes"][:, example_mode], gmm_fit=gmm_fit,
            label=f"mode {example_mode}",
            save_path=os.path.join(OUTPUT_DIR, "example_tls_mode_amplitude.png"),
        )
    else:
        print("  (no bimodal modes found -- skipping example amplitude plot)")

    # --- 3c. TLS detection, 2D per-slice, Method A ---
    print("\n=== TLS detection: 2D per-slice, Method A ===")
    tls_A_2d = detect_tls_direct_2d_slices(
        trajectory, n_slices=N_Z_SLICES, cutoff=CUTOFF_DICT,
        delta_bic_threshold=TLS_DELTA_BIC_THRESHOLD,
        separation_threshold=TLS_SEPARATION_THRESHOLD,
        min_weight_fraction=TLS_MIN_WEIGHT_FRACTION,
        min_switches=TLS_MIN_SWITCHES,
    )
    for s, slc in enumerate(tls_A_2d["slices"]):
        if slc is None:
            continue
        print(f"  slice {s}: {slc['is_tls'].sum()} TLS candidates "
              f"/ {len(slc['atom_indices'])} atoms analyzed.")

    # --- 4. Comparisons ---
    print("\n=== Comparing soft spots, TLS, and D^2_min ===")
    # simple union-style combined TLS score: amplitude-method score,
    # bumped for atoms the direct method also flagged
    tls_score_combined = tls_B_3d["tls_score"].copy()
    tls_score_combined[tls_A_3d["is_tls"]] += 1.0

    comparison = run_full_comparison(
        trajectory, modes_3d["softness"], tls_score_combined, cutoff=CUTOFF_DICT,
        quantile_threshold=D2MIN_QUANTILE_THRESHOLD, top_frac=D2MIN_TOP_FRACTION,
    )

    print("\n-- Soft spots vs. TLS --")
    st = comparison["soft_vs_tls"]
    print(f"  Spearman r = {st['spearman_r']:.3f} (p={st['spearman_p']:.2e})")
    if "jaccard_top_softness_vs_tls" in st:
        print(f"  Jaccard(top-softness set, TLS set) = {st['jaccard_top_softness_vs_tls']:.3f}")
    plot_soft_tls_scatter(modes_3d["softness"], tls_score_combined,
                           save_path=os.path.join(OUTPUT_DIR, "soft_vs_tls.png"))

    print("\n-- Soft spots -> D^2_min --")
    sd = comparison["soft_vs_d2min"]
    print(f"  Spearman r = {sd['spearman_r']:.3f} (p={sd['spearman_p']:.2e}), AUC = {sd['auc']:.3f}")
    print(f"  Enrichment (top {D2MIN_TOP_FRACTION:.0%} softest): "
          f"{sd['enrichment']['enrichment']:.2f}x mean D^2_min")
    plot_roc_curve(sd["fpr"], sd["tpr"], sd["auc"],
                    title="Soft spots predicting D\u00b2_min",
                    save_path=os.path.join(OUTPUT_DIR, "roc_soft_vs_d2min.png"))

    print("\n-- TLS -> D^2_min --")
    td = comparison["tls_vs_d2min"]
    print(f"  Spearman r = {td['spearman_r']:.3f} (p={td['spearman_p']:.2e}), AUC = {td['auc']:.3f}")
    print(f"  Enrichment (top {D2MIN_TOP_FRACTION:.0%} by TLS score): "
          f"{td['enrichment']['enrichment']:.2f}x mean D^2_min")
    plot_roc_curve(td["fpr"], td["tpr"], td["auc"],
                    title="TLS candidates predicting D\u00b2_min",
                    save_path=os.path.join(OUTPUT_DIR, "roc_tls_vs_d2min.png"))

    # --- 5. Element statistics ---
    print("\n=== Element statistics ===")

    # softness_series / d2min_series are each atom's field value at every
    # frame, measured from the mean structure (see
    # analysis.soft_modes.softness_trajectory() /
    # dynamics.flow_units.d2min_trajectory()) -- computed once here and
    # reused for the persistence numbers AND the distribution plots/
    # animations below, rather than recomputed per use.
    print("\nComputing per-frame softness and D^2_min trajectories "
          "(used for persistence + distribution plots below)...")
    # Reference for both: the MEAN STRUCTURE of the whole trajectory (unwrapped,
    # time-average position per atom), fixed for every frame -- so neither
    # field grows with the frame number. Softness uses ALL modes of each
    # window (no n_soft_modes cap, no minimum number of frames).
    _positions = np.array([a.positions for a in trajectory])
    _box = np.array([a.cell.array.diagonal() for a in trajectory])
    _, mean_positions = compute_displacements(unwrap_positions(_positions, _box))
    d2min_series, frame_indices_traj = d2min_trajectory(
        trajectory, t0=0, frame_step=1, cutoff=CUTOFF_DICT,
        reference_positions=mean_positions,
    )
    softness_series, softness_pr_series, _ = softness_trajectory(
        trajectory, t0=0, frame_step=1, pr_max=SOFT_MODE_PR_MAX,
    )

    # -- softness by element (plain + participation-weighted) --
    # "mean"/"median" here are the mean/median, ACROSS ATOMS of that
    # element, of each atom's single whole-trajectory softness score
    # (modes_3d["softness"] / modes_3d["softness_pr_weighted"], one
    # number per atom, built from the full covariance over all frames
    # -- see soft_mode_analysis_3d()). Not a per-frame quantity. See
    # the distribution plots below for the full shape this is
    # summarizing, including how it evolves over frames.
    #
    # "enrichment" = (element's share among the top 10% highest-scoring
    # atoms) / (element's share of the whole population); >1 means the
    # element is over-represented among the top-scoring atoms.
    print("\n-- softness by element --")
    softness_el = score_by_element(modes_3d["softness"], elements)
    softness_pr_el = score_by_element(modes_3d["softness_pr_weighted"], elements)
    print_element_table({
        el: {
            "mean": softness_el["by_element"][el]["mean"],
            "median": softness_el["by_element"][el]["median"],
            "enrichment": softness_el["by_element"][el]["enrichment"],
            "pr_weighted_mean": softness_pr_el["by_element"][el]["mean"],
            "pr_weighted_median": softness_pr_el["by_element"][el]["median"],
            "pr_weighted_enrichment": softness_pr_el["by_element"][el]["enrichment"],
        }
        for el in softness_el["by_element"]
    })
    for pair, stats in softness_el["pairwise_tests"].items():
        print(f"  {pair}: p = {stats['p_value']:.2e}")

    # -- D^2_min by element --
    # reuse the D^2_min already computed for the soft-spot vs. D^2_min
    # comparison above -- no need to recompute it.
    print("\n-- D^2_min by element --")
    d2min_el = score_by_element(comparison["soft_vs_d2min"]["d2min"], elements)
    print_element_table({
        el: {"mean": s["mean"], "median": s["median"], "enrichment": s["enrichment"]}
        for el, s in d2min_el["by_element"].items()
    })
    for pair, stats in d2min_el["pairwise_tests"].items():
        print(f"  {pair}: p = {stats['p_value']:.2e}")

    # -- persistence by element --
    print("\n-- persistence by element (fraction of the trajectory each atom "
          "spends in the top 10% of the population) --")
    persistence_d2min = persistence_by_element(
        persistence_above_threshold(d2min_series, quantile=0.9), elements)
    persistence_softness = persistence_by_element(
        persistence_above_threshold(softness_series, quantile=0.9), elements)
    print_element_table({
        el: {
            "d2min_mean": persistence_d2min[el]["mean_persistence"],
            "d2min_median": persistence_d2min[el]["median_persistence"],
            "softness_mean": persistence_softness[el]["mean_persistence"],
            "softness_median": persistence_softness[el]["median_persistence"],
        }
        for el in persistence_d2min
    })

    # -- distributions by element: how the softness/D^2_min distributions
    # actually look and evolve, rather than collapsing each to one number.
    # Each animation frame shows: per-element distribution (this frame),
    # combined distribution (this frame), and the per-element
    # whole-trajectory AVERAGE distribution as a stationary dashed
    # outline that does not change between frames. --
    print("\n-- distributions by element --")
    plot_distribution_by_element(
        softness_series, elements, xlabel="Softness score (a.u.)",
        title="Average Softness Distribution by Element",
        bins=100,
        save_path=os.path.join(OUTPUT_DIR, "distribution_softness_by_element_average.png"),
    )
    plot_distribution_by_element(
        softness_pr_series, elements, xlabel="PR-weighted softness score (a.u.)",
        title="Average PR-Weighted Softness Distribution by Element",
        bins=100,
        save_path=os.path.join(OUTPUT_DIR, "distribution_softness_pr_weighted_by_element_average.png"),
    )
    plot_distribution_by_element(
        d2min_series, elements, xlabel="D\u00b2_min (\u00c5\u00b2)",
        title="Average D\u00b2_min Distribution by Element",
        bins=100,   
        save_path=os.path.join(OUTPUT_DIR, "distribution_d2min_by_element_average.png"),
    )
    animate_distribution_by_element(
        softness_series, frame_indices_traj, elements, xlabel="Softness score (a.u.)",
        title_prefix="Softness distribution",
        bins=100,
        save_path=os.path.join(OUTPUT_DIR, "distribution_softness_by_element.mp4"),
    )
    animate_distribution_by_element(
        softness_pr_series, frame_indices_traj, elements, xlabel="PR-weighted softness score (a.u.)",
        title_prefix="PR-weighted softness distribution",
        bins=100,
        save_path=os.path.join(OUTPUT_DIR, "distribution_softness_pr_weighted_by_element.mp4"),
    )
    animate_distribution_by_element(
        d2min_series, frame_indices_traj, elements, xlabel="D\u00b2_min (\u00c5\u00b2)",
        title_prefix="D\u00b2_min distribution",
        bins=100,
        save_path=os.path.join(OUTPUT_DIR, "distribution_d2min_by_element.mp4"),
    )

    # -- combined 3D + distribution animation: D^2_min and softness side
    # by side, each with its own evolving distribution (same per-element
    # + combined + stationary-average treatment as above) above it.
    # Reuses the d2min_series/softness_series already computed above
    # instead of recomputing them a third time. --
    print("\n-- combined D^2_min / softness 3D + distribution animation --")
    animate_d2min_softness_side_by_side(
        trajectory, cutoff=CUTOFF_DICT, elements=elements,
        d2min_series=d2min_series, softness_series=softness_series,
        save_path=os.path.join(OUTPUT_DIR, "d2min_softness_side_by_side_3d.mp4"),
    )
    # ... and the same two 3D maps on their own, without the distributions
    animate_d2min_softness_side_by_side(
        trajectory, cutoff=CUTOFF_DICT, elements=elements,
        d2min_series=d2min_series, softness_series=softness_series,
        show_distributions=False,
        save_path=os.path.join(OUTPUT_DIR, "d2min_softness_maps_side_by_side_3d.mp4"),
    )


    # --- 6. Are soft atoms "sources" of plastic activity? ---
    # WHERE: is D^2_min elevated next to soft atoms? (final frame, D^2_min
    #   measured from the mean structure -- the series computed above)
    # WHEN: do soft atoms cross a D^2_min threshold before their neighbours?
    #   This needs a field that GROWS from a fixed starting frame, so it uses
    #   D^2_min(0 -> t); with the stationary mean-structure field an "onset"
    #   would only be the waiting time to the first fluctuation.
    # Both are asked of the raw D^2_min AND of D^2_min minus its thermal
    # baseline (soft atoms vibrate more, which alone inflates their
    # surroundings), against a null with soft labels permuted within
    # (element, CN) groups.
    print("\n=== Are soft atoms sources of plastic activity? ===")
    print("  (raw = D^2_min; excess = D^2_min minus its thermal baseline)")
    labels = {"raw": "raw", "excess": "excess"}

    where = soft_source_analysis(
        trajectory, modes_3d["softness"], d2min_series, cutoff=CUTOFF_DICT,
        soft_frac=D2MIN_TOP_FRACTION, reference="mean", temporal=False,
        max_group=MAX_SOFT_NB,
    )
    print("\n-- WHERE: non-soft atoms next to a soft atom minus those that are not "
          "(final frame, A^2) --")
    print_element_table({
        labels[k]: {"observed": f["spatial"]["observed"], "null": f["spatial"]["null_mean"],
                    "null_std": f["spatial"]["null_std"], "z": f["spatial"]["z"]}
        for k, f in where["fields"].items()
    }, row_label="field", col_width=12)
    print("\n-- D^2_min of non-soft atoms vs. number of soft neighbours --")
    by_k_raw = where["fields"]["raw"]["spatial"]["by_n_soft_neighbours"]
    by_k_exc = where["fields"]["excess"]["spatial"]["by_n_soft_neighbours"]
    print_element_table({
        f"{k}{'+' if k == 4 else ''} soft nb": {
            "n_atoms": by_k_raw[k][0], "raw": by_k_raw[k][1], "excess": by_k_exc[k][1],
        } for k in sorted(by_k_raw)
    }, row_label="group", col_width=14)
    plot_d2min_by_soft_neighbours(
        where["n_soft_neighbours"], where["soft_mask"],
        values={"raw D\u00b2_min": np.asarray(d2min_series)[-1],
                "excess (D\u00b2_min - thermal baseline)": where["excess"][-1]},
        null={"raw D\u00b2_min": where["fields"]["raw"]["spatial"]["null_by_n_soft_neighbours"],
              "excess (D\u00b2_min - thermal baseline)":
                  where["fields"]["excess"]["spatial"]["null_by_n_soft_neighbours"]},
        save_path=os.path.join(OUTPUT_DIR, "d2min_by_soft_neighbours.png"),
    )

    print("\nComputing D^2_min(0 -> t) for the temporal test...")
    d2min_from_first, _ = d2min_trajectory(trajectory, t0=0, frame_step=1, cutoff=CUTOFF_DICT)
    when = soft_source_analysis(
        trajectory, modes_3d["softness"], d2min_from_first, cutoff=CUTOFF_DICT,
        soft_frac=D2MIN_TOP_FRACTION, reference="first", spatial=False,
    )
    print("\n-- WHEN: bond (soft atom, non-soft neighbour), lag = onset_neighbour - "
          "onset_soft (frames; > 0: soft atom first) --")
    print_element_table({
        labels[k]: {"n_bonds": f["temporal"]["n_bonds"], "mean_lag": f["temporal"]["mean_lag"],
                    "null": f["temporal"]["mean_lag_null_mean"], "z": f["temporal"]["mean_lag_z"],
                    "soft_first": f["temporal"]["frac_soft_first"],
                    "null_frac": f["temporal"]["frac_null_mean"], "z_frac": f["temporal"]["frac_z"]}
        for k, f in when["fields"].items()
    }, row_label="field", col_width=12)

    print("\nDone.")