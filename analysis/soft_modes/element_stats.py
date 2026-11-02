# =============================================================
# analysis/soft_modes/element_stats.py
#
# Per-element (Zr vs Cu) breakdowns of the soft-spot / TLS / D^2_min
# fields computed elsewhere in this package:
#   - which element dominates the softest, most localized modes
#   - which element shows higher softness / D^2_min on average, and
#     is over/under-represented among the highest-scoring atoms
#   - which element stays "high" (soft, or high D^2_min) for the
#     largest fraction of a growing-window trajectory
# =============================================================

import numpy as np


def pool_series_by_element(series, elements):
    """
    Pool every atom's value at every frame, per element and combined.

    series : (T, N) array (NaN rows/entries = "not enough data yet", dropped)
    elements : (N,) array of element symbols

    Returns
    -------
    pooled : dict {element: 1D array of all finite values of that element}
    combined : 1D array of all finite values (all elements)
    """
    series = np.asarray(series, dtype=float)
    elements = np.asarray(elements)
    pooled = {}
    for el in np.unique(elements):
        vals = series[:, elements == el].ravel()
        pooled[str(el)] = vals[np.isfinite(vals)]
    allv = series.ravel()
    return pooled, allv[np.isfinite(allv)]


def get_element_symbols(atoms):
    """Per-atom element symbols ('Zr', 'Cu', ...), as a numpy array."""
    return np.array(atoms.get_chemical_symbols())


def score_by_element(score, elements, top_frac=0.1):
    """
    Per-element summary of a continuous per-atom score (softness,
    D^2_min, ...): count, mean/median/std, and -- the more useful
    number -- ENRICHMENT: among the top `top_frac` of atoms by score,
    what fraction belongs to each element, divided by that element's
    baseline fraction of the whole population. Enrichment > 1 means
    that element is over-represented among the highest scorers;
    < 1 means under-represented; 1 means exactly proportional to its
    overall abundance (so e.g. a 50/50 Zr/Cu system with no real
    elemental effect should land close to 1 for both).

    Also runs a pairwise Mann-Whitney U test between every pair of
    elements' score distributions (non-parametric: doesn't assume
    Gaussian scores, appropriate here since softness/D^2_min are
    both long-tailed).

    Parameters
    ----------
    score : (N,) array
    elements : (N,) array of element symbols, same order as score

    Returns
    -------
    dict: "by_element" (per-element stats dict), "pairwise_tests"
    (Mann-Whitney U/p per element pair), "top_frac", "n_top"
    """
    from scipy.stats import mannwhitneyu

    score = np.asarray(score, dtype=float)
    elements = np.asarray(elements)
    mask = ~np.isnan(score)
    score, elements = score[mask], elements[mask]

    unique = np.unique(elements)
    n_total = len(score)

    by_element = {}
    for el in unique:
        vals = score[elements == el]
        by_element[el] = {
            "n": int(len(vals)),
            "fraction_of_population": len(vals) / n_total,
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
        }

    n_top = max(1, int(np.ceil(top_frac * n_total)))
    top_elements = elements[np.argsort(score)[::-1][:n_top]]
    for el in unique:
        frac_in_top = np.sum(top_elements == el) / n_top
        base = by_element[el]["fraction_of_population"]
        by_element[el]["fraction_of_top"] = frac_in_top
        by_element[el]["enrichment"] = frac_in_top / base if base > 0 else np.nan

    pairwise_tests = {}
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            a, b = unique[i], unique[j]
            stat, p = mannwhitneyu(score[elements == a], score[elements == b],
                                    alternative="two-sided")
            pairwise_tests[f"{a}_vs_{b}"] = {"u_stat": float(stat), "p_value": float(p)}

    return {
        "by_element": by_element, "pairwise_tests": pairwise_tests,
        "top_frac": top_frac, "n_top": n_top,
    }


def element_mode_participation(eigenvectors, elements):
    """
    How every mode is populated, atom by atom and element by element.

    Returns
    -------
    p : (N, K) per-atom participation  p[i, n] = sum_d e_n[i, d]^2
        (each column sums to 1)
    share : dict {element: (K,) array} -- the part of each mode's total
        participation carried by that element's atoms (sums to 1 over elements)
    population_fraction : dict {element: float} -- the element's share of the atoms
    """
    elements = np.asarray(elements)
    p = np.sum(eigenvectors ** 2, axis=1)
    share = {str(el): p[elements == el].sum(axis=0) for el in np.unique(elements)}
    population_fraction = {str(el): float(np.mean(elements == el)) for el in np.unique(elements)}
    return p, share, population_fraction


def mode_dominant_elements(eigenvectors, elements, mode_mask=None):
    """
    For every mode, decide which element dominates its participation
    -- normalized by how many atoms of each element exist in the
    system, so a more numerous species doesn't win just by headcount.
    For each element, this is the atom-averaged participation
    (total participation summed over that element's atoms, divided by
    how many atoms of that element there are); the dominant element
    for a mode is whichever has the larger atom-averaged value.

    Parameters
    ----------
    eigenvectors : (N, D, K) array
    elements : (N,) array of element symbols
    mode_mask : (K,) bool array, optional -- restricts the returned
        `tally` to these modes (e.g. soft_mode_analysis_3d()'s
        "soft_mode_mask", for "which element dominates the SOFT
        modes specifically"). `dominant` itself always covers every
        mode, so it can be handed straight to
        visualization.plot.plot_participation_spectrum() regardless.

    Returns
    -------
    dominant : (K,) array of element symbols, one per mode
    tally : dict -- how many of the mode_mask-selected modes (or all
        modes, if mode_mask is None) each element dominates
    """
    elements = np.asarray(elements)
    unique = np.unique(elements)
    p = np.sum(eigenvectors ** 2, axis=1)   # (N, K) per-atom participation

    counts = {el: np.sum(elements == el) for el in unique}
    per_element_participation = np.array([
        p[elements == el].sum(axis=0) / counts[el] for el in unique
    ])   # (n_elements, K)

    dominant = unique[np.argmax(per_element_participation, axis=0)]   # (K,)

    selected = dominant if mode_mask is None else dominant[np.asarray(mode_mask)]
    tally = {el: int(np.sum(selected == el)) for el in unique}

    return dominant, tally


def persistence_above_threshold(series, quantile=0.9):
    """
    For a growing-window trajectory of a per-atom field (e.g.
    d2min_series from dynamics.flow_units.d2min_trajectory(), or
    softness_series from analysis.soft_modes.softness_trajectory()),
    compute, per atom, what FRACTION of the evaluated frames it spent
    above a quantile threshold of that frame's own distribution.

    The threshold is recomputed fresh per frame (the `quantile`
    quantile of that frame's values) rather than fixed in absolute
    units -- a self-relative "was this atom in the top (1-quantile)
    of the population right now" bar, which stays meaningful even as
    the whole population's scale drifts over the trajectory.

    Frames that are entirely NaN (e.g. softness_trajectory()'s first
    few frames, before enough data accumulates) are skipped and don't
    count toward the fraction.

    Returns
    -------
    persistence : (N,) array in [0, 1]
    """
    persistence = None
    n_valid = 0
    for frame_vals in series:
        frame_vals = np.asarray(frame_vals, dtype=float)
        if np.all(np.isnan(frame_vals)):
            continue
        if persistence is None:
            persistence = np.zeros(len(frame_vals))
        thr = np.nanquantile(frame_vals, quantile)
        above = (frame_vals > thr) & ~np.isnan(frame_vals)
        persistence += above.astype(float)
        n_valid += 1
    if persistence is None:
        return np.array([])
    return persistence / n_valid


def persistence_by_element(persistence, elements):
    """Mean/median persistence (see persistence_above_threshold), per element."""
    elements = np.asarray(elements)
    unique = np.unique(elements)
    return {
        el: {
            "mean_persistence": float(np.mean(persistence[elements == el])),
            "median_persistence": float(np.median(persistence[elements == el])),
        }
        for el in unique
    }


def print_element_table(rows, row_label="Element", decimals=2, col_width=16):
    """
    Plain-text table for a {element: {stat_name: value, ...}} dict --
    e.g. straight from score_by_element()["by_element"] or a merge of
    several persistence_by_element() calls. Works for any number of
    elements (rows) and any set of stat names (columns), as long as
    every element has the same stat names. Floats are rounded to
    `decimals` places; anything else is printed as-is.

    This is a formatting helper only -- it doesn't compute anything,
    just renders a dict of dicts legibly instead of a raw Python repr.
    """
    elements = list(rows.keys())
    if not elements:
        print("  (nothing to show)")
        return
    stat_names = list(next(iter(rows.values())).keys())
    col_width = max(col_width, max(len(name) for name in stat_names) + 2)

    header = f"  {row_label:<10}" + "".join(f"{name:>{col_width}}" for name in stat_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for el in elements:
        line = f"  {str(el):<10}"
        for name in stat_names:
            val = rows[el][name]
            if isinstance(val, (int, float, np.floating, np.integer)):
                line += f"{val:>{col_width}.{decimals}f}"
            else:
                line += f"{str(val):>{col_width}}"
        print(line)