# =============================================================
# analysis/soft_modes/comparison.py
#
# Compares the three quantities from analysis/soft_modes/modes.py and
# analysis/soft_modes/tls.py against each other and against
# dynamics.flow_units' D^2_min:
#
#   1. soft spots vs. TLS         -> compare_soft_tls()
#   2. soft spots -> D^2_min      -> predict_d2min_from_score()
#   3. TLS        -> D^2_min      -> predict_d2min_from_score()
#
# The D^2_min prediction test follows the same logic as
# Widmer-Cooper & Harrowell (2006): is a field defined at t0
# (softness, or TLS-ness) predictive of WHERE non-affine
# rearrangement happens later?
#
# No scikit-learn dependency (not in requirements.txt) — AUC/ROC are
# computed from first principles (rank-sum / threshold sweep).
# =============================================================

import numpy as np

from cutoffs import CUTOFF_DICT


# ------------------------------------------------------------
# Generic predictor-vs-binary-outcome tools (no sklearn)
# ------------------------------------------------------------

def compute_auc(score, label):
    """
    AUC of `score` as a predictor of binary `label`, via the
    Mann-Whitney U / rank-sum identity:

        AUC = (sum of ranks of the positive class - n_pos*(n_pos+1)/2)
              / (n_pos * n_neg)

    Equivalent to sklearn.metrics.roc_auc_score, without the
    dependency.
    """
    from scipy.stats import rankdata

    score = np.asarray(score, dtype=float)
    label = np.asarray(label).astype(bool)
    valid = ~np.isnan(score)
    score, label = score[valid], label[valid]

    n_pos, n_neg = label.sum(), (~label).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = rankdata(score)
    rank_sum_pos = ranks[label].sum()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc_curve_manual(score, label, n_thresholds=200):
    """Threshold sweep giving (fpr, tpr, thresholds) — for plotting."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label).astype(bool)
    valid = ~np.isnan(score)
    score, label = score[valid], label[valid]

    thresholds = np.linspace(score.max(), score.min(), n_thresholds)
    n_pos, n_neg = label.sum(), (~label).sum()

    tpr, fpr = np.zeros(n_thresholds), np.zeros(n_thresholds)
    for k, thr in enumerate(thresholds):
        pred = score >= thr
        tpr[k] = np.sum(pred & label) / n_pos if n_pos else np.nan
        fpr[k] = np.sum(pred & ~label) / n_neg if n_neg else np.nan

    return fpr, tpr, thresholds


def enrichment_top_fraction(score, values, top_frac=0.1):
    """
    Mean of `values` among the top `top_frac` of atoms by `score`,
    relative to the mean over all atoms — e.g. "atoms in the softest
    10% end up with 3.2x the mean D^2_min of the full population".
    """
    score = np.asarray(score, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = ~np.isnan(score) & ~np.isnan(values)
    score, values = score[mask], values[mask]

    n_top = max(1, int(np.ceil(top_frac * len(score))))
    top_idx = np.argsort(score)[::-1][:n_top]
    mean_top = values[top_idx].mean()
    mean_all = values.mean()

    return {
        "mean_top": mean_top,
        "mean_all": mean_all,
        "enrichment": mean_top / mean_all if mean_all != 0 else np.nan,
        "top_frac": top_frac,
        "n_top": n_top,
    }


# ------------------------------------------------------------
# 1. Soft spots vs. TLS
# ------------------------------------------------------------

def compare_soft_tls(softness, tls_score):
    """
    Correlate the continuous soft-spot score against the TLS score
    (works whether tls_score is boolean/0-1 or continuous — a
    point-biserial correlation is computed automatically whenever
    tls_score is binary, IN ADDITION to the always-available Spearman
    correlation).

    Returns
    -------
    dict with "spearman_r"/"spearman_p", and, if tls_score is binary:
    "point_biserial_r"/"_p", "n_tls", "overlap_with_top_softness"
    (atoms flagged TLS that also rank among the n_tls softest atoms),
    "jaccard_top_softness_vs_tls".
    """
    from scipy.stats import spearmanr, pointbiserialr

    softness = np.asarray(softness, dtype=float)
    tls_score = np.asarray(tls_score, dtype=float)
    mask = ~np.isnan(softness) & ~np.isnan(tls_score)
    s, t = softness[mask], tls_score[mask]

    r, p = spearmanr(s, t)
    result = {"spearman_r": r, "spearman_p": p}

    is_binary = np.all(np.isin(np.unique(t), [0, 1]))
    if is_binary:
        t_bool = t.astype(bool)
        if 0 < t_bool.sum() < len(t_bool):
            pb_r, pb_p = pointbiserialr(t_bool, s)
            result["point_biserial_r"], result["point_biserial_p"] = pb_r, pb_p

        n_tls = int(t_bool.sum())
        top_n = np.argsort(s)[::-1][:max(1, n_tls)]
        tls_idx = np.where(t_bool)[0]
        overlap = np.intersect1d(tls_idx, top_n)
        union = np.union1d(tls_idx, top_n)
        result.update({
            "n_tls": n_tls,
            "overlap_with_top_softness": int(len(overlap)),
            "jaccard_top_softness_vs_tls": (len(overlap) / len(union)) if len(union) else np.nan,
        })

    return result


# ------------------------------------------------------------
# 2/3. Predicting D^2_min from a score (soft-spot or TLS)
# ------------------------------------------------------------

def predict_d2min_from_score(score, trajectory, t0=0, delta_frames=None,
                              cutoff=CUTOFF_DICT, quantile_threshold=0.9,
                              d2min_threshold=None, top_frac=0.1):
    """
    Tests whether a per-atom predictor `score` (softness or
    TLS-ness, evaluated at/around frame t0) forecasts which atoms go
    on to undergo a large non-affine rearrangement — "how good does a
    soft spot predict a D^2_min event?"

    Parameters
    ----------
    delta_frames : int, optional — how many frames ahead to look;
        defaults to the whole remaining trajectory (t0 -> last frame).
    quantile_threshold : float — an atom "rearranges" if its D^2_min
        exceeds this quantile of the full D^2_min distribution
        (default: top 10%). Ignored if d2min_threshold is given.
    d2min_threshold : float, optional — an absolute D^2_min cutoff
        (Å^2) instead of a quantile.

    Returns
    -------
    dict: "d2min" (N,), "rearranged" (N,) bool, "threshold_A2",
    "spearman_r"/"_p", "auc", "fpr"/"tpr"/"thresholds" (for
    visualization.plot.plot_roc_curve), "enrichment" (dict from
    enrichment_top_fraction).
    """
    from analysis.d2min.compute import compute_d2min
    from scipy.stats import spearmanr
    from utils import unwrap_positions, compute_displacements

    t_end = (len(trajectory) - 1 if delta_frames is None
             else min(t0 + delta_frames, len(trajectory) - 1))

    pos = np.array([a.positions for a in trajectory[t0:t_end + 1]])
    box = np.array([a.cell.array.diagonal() for a in trajectory[t0:t_end + 1]])
    _, mean_positions = compute_displacements(unwrap_positions(pos, box))

    atoms_ref = trajectory[t0].copy()
    atoms_ref.positions = mean_positions

    d2min, _ = compute_d2min(atoms_ref, trajectory[t_end], cutoff=cutoff)
    valid = ~np.isnan(d2min)

    thr = d2min_threshold if d2min_threshold is not None else np.nanquantile(d2min, quantile_threshold)
    rearranged = d2min > thr

    score = np.asarray(score, dtype=float)
    mask = valid & ~np.isnan(score)

    r, p = spearmanr(score[mask], d2min[mask])
    auc = compute_auc(score[mask], rearranged[mask])
    fpr, tpr, thresholds = roc_curve_manual(score[mask], rearranged[mask])
    enrich = enrichment_top_fraction(score[mask], d2min[mask], top_frac=top_frac)

    return {
        "t0": t0, "t_end": t_end,
        "d2min": d2min, "rearranged": rearranged, "threshold_A2": thr,
        "spearman_r": r, "spearman_p": p,
        "auc": auc, "fpr": fpr, "tpr": tpr, "thresholds": thresholds,
        "enrichment": enrich,
    }


def run_full_comparison(trajectory, softness, tls_score, t0=0, delta_frames=None,
                         cutoff=CUTOFF_DICT, quantile_threshold=0.9, top_frac=0.1):
    """
    Runs all three comparisons in one call:
      1. soft spots vs. TLS      -> compare_soft_tls()
      2. soft spots -> D^2_min   -> predict_d2min_from_score(softness, ...)
      3. TLS -> D^2_min          -> predict_d2min_from_score(tls_score, ...)
    """
    return {
        "soft_vs_tls": compare_soft_tls(softness, tls_score),
        "soft_vs_d2min": predict_d2min_from_score(
            softness, trajectory, t0=t0, delta_frames=delta_frames,
            cutoff=cutoff, quantile_threshold=quantile_threshold, top_frac=top_frac,
        ),
        "tls_vs_d2min": predict_d2min_from_score(
            tls_score, trajectory, t0=t0, delta_frames=delta_frames,
            cutoff=cutoff, quantile_threshold=quantile_threshold, top_frac=top_frac,
        ),
    }
