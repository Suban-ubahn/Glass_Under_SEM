# =============================================================
# visualization/animate/softness.py
# (split out of the old 1300-line visualization/animate.py;
#  `from visualization.animate import <name>` still works)
# =============================================================

import numpy as np
from utils import get_output_dir
OUTPUT_DIR = get_output_dir()
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from config import N_SOFT_MODES, SOFT_MODE_PR_MAX
from cutoffs import CUTOFF_DICT
from analysis.d2min.compute import d2min_trajectory
from analysis.soft_modes.modes import softness_trajectory


def animate_softness_3d(
    trajectory,
    t0=0,
    frame_step=1,
    mass_weight=True,
    n_soft_modes=N_SOFT_MODES,
    pr_max=SOFT_MODE_PR_MAX,
    min_frames=5,
    save_path=None,
    elev=20,
    azim=45
):
    """
    Softness-score analogue of animate_d2min_3d() -- same animation
    structure, same frame_indices convention, same fixed-vmax-from-
    the-final-frame colorbar, so the two can be watched side by side
    (or dropped into the same figure later) as a direct visual
    comparison.

    Animate atoms in 3D at their real (x, y, z) positions, colored by
    a softness score computed from frames [0, t] relative to frame 0
    -- for every frame t, exactly the same "relative to frame 0,
    growing window" structure animate_d2min_3d() uses for D^2_min,
    via analysis.soft_modes.softness_trajectory() (see that function's
    docstring for the important caveat this reference-frame choice
    brings: this is NOT the same "vibrational softness around
    equilibrium" quantity soft_mode_analysis_3d() computes by default
    -- it's the frame-0-referenced version, built specifically to be
    comparable to D^2_min frame-for-frame).

    The first few frames (fewer than min_frames available) don't carry
    enough data for a meaningful covariance yet and show up blank/gray
    (NaN) -- the softness equivalent of D^2_min being identically zero
    at t0 itself.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "softness_dynamics_3d.mp4")

    print("Creating softness 3D animation")

    softness_series, _, frame_indices = softness_trajectory(
        trajectory, t0=t0, frame_step=frame_step, mass_weight=mass_weight,
        n_soft_modes=n_soft_modes, pr_max=pr_max, min_frames=min_frames,
    )

    # Fixed colour scale based on the final frame (same convention as
    # animate_d2min_3d) -- ignoring the early, still-NaN frames.
    vmax = np.nanpercentile(softness_series[-1], 99)

    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label("Softness score (a.u.)", fontsize=9)
    cell = trajectory[0].cell.lengths()

    def update(i):
        frame = frame_indices[i]
        ax.clear()

        pos = trajectory[frame].positions
        softness = softness_series[i]

        ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=softness, cmap=cmap, norm=norm, s=10, alpha=0.8,
        )
        ax.set_xlim(0, cell[0]); ax.set_ylim(0, cell[1]); ax.set_zlim(0, cell[2])
        ax.set_xlabel("x (A)"); ax.set_ylabel("y (A)"); ax.set_zlabel("z (A)")
        ax.set_title(f"Softness relative to frame 0 (Frame {frame})")
        ax.view_init(elev=elev, azim=azim)

    ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=100)
    ani.save(save_path, fps=12, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def _draw_distribution_panel(ax, frame_vals, elements, unique_elements, colors,
                             xmin, xmax, bins, xlabel, title, average_pooled=None):
    """
    One distribution panel:
      - filled histogram per element (this frame)
      - black step histogram of ALL atoms combined (this frame)
      - dashed step outline per element = whole-trajectory average
        distribution (average_pooled, identical in every frame)
    """
    edges = np.linspace(xmin, xmax, bins + 1)
    frame_vals = np.asarray(frame_vals, dtype=float)
    finite = np.isfinite(frame_vals)

    if finite.any():
        for color, el in zip(colors, unique_elements):
            v = frame_vals[(elements == el) & finite]
            ax.hist(v, bins=edges, density=True, alpha=0.45, color=color, label=el)
        ax.hist(frame_vals[finite], bins=edges, density=True, histtype="step",
                color="black", linewidth=2, label="Combined")
    else:
        ax.text(0.5, 0.5, "not enough data yet", transform=ax.transAxes,
                ha="center", va="center")

    if average_pooled is not None:
        for color, el in zip(colors, unique_elements):
            ax.hist(average_pooled[el], bins=edges, density=True, histtype="step",
                    color=color, linestyle="--", linewidth=1.8,
                    label=f"{el} (trajectory avg)")

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    if finite.any():
        ax.legend(fontsize=8)


def _distribution_limits(series, pooled_combined):
    """x-range of the distribution panels: identical to plot_distribution_by_element
    (minimum to the 99.5th percentile of the pooled whole-trajectory values)."""
    return float(np.min(pooled_combined)), float(np.percentile(pooled_combined, 99.5))


def animate_distribution_by_element(series, frame_indices, elements, xlabel,
                                    title_prefix, bins=40, save_path=None):
    """
    Distribution of a per-atom field vs. frame: per-element (filled) and
    combined (black) for the current frame, plus the stationary dashed
    whole-trajectory average per element.

    series : (T, N); frame_indices : (T,); elements : (N,) symbols.
    """
    from analysis.soft_modes.element_stats import pool_series_by_element
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "distribution_by_element.mp4")

    elements = np.asarray(elements)
    unique_elements = [str(e) for e in np.unique(elements)]
    colors = plt.cm.tab10(np.arange(len(unique_elements)))
    pooled, combined = pool_series_by_element(series, elements)
    xmin, xmax = _distribution_limits(series, combined)

    fig, ax = plt.subplots(figsize=(8, 5))

    def update(i):
        ax.clear()
        _draw_distribution_panel(
            ax, series[i], elements, unique_elements, colors, xmin, xmax, bins,
            xlabel, f"{title_prefix} -- frame {frame_indices[i]}", average_pooled=pooled)
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    anim.save(save_path, writer="ffmpeg", fps=12, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")


def animate_d2min_softness_side_by_side(
    trajectory, cutoff=CUTOFF_DICT, t0=0, frame_step=1, mass_weight=True,
    n_soft_modes=None, pr_max=None, elements=None,
    d2min_series=None, softness_series=None, bins=40, save_path=None,
    elev=20, azim=45, show_distributions=True, fps=12,
):
    """
    Side-by-side animation. Left: D^2_min, right: softness (both relative
    to the MEAN STRUCTURE of the trajectory), 3D atoms coloured by the field.

    show_distributions=True : 2x2 layout, distribution panels (per element,
        combined, stationary whole-trajectory averages) above the 3D maps.
    show_distributions=False: 1x2 layout, only the two 3D maps.

    Pass precomputed d2min_series / softness_series (T, N) to avoid
    recomputing them.
    """
    from analysis.soft_modes.element_stats import pool_series_by_element
    from analysis.soft_modes.modes import softness_trajectory
    if save_path is None:
        save_path = os.path.join(
            OUTPUT_DIR, "d2min_softness_side_by_side_3d.mp4" if show_distributions
            else "d2min_softness_maps_side_by_side_3d.mp4")

    if d2min_series is None:
        from utils import unwrap_positions, compute_displacements
        _pos = np.array([a.positions for a in trajectory])
        _box = np.array([a.cell.array.diagonal() for a in trajectory])
        _, mean_positions = compute_displacements(unwrap_positions(_pos, _box))
        d2min_series, frame_indices = d2min_trajectory(
            trajectory, t0=t0, frame_step=frame_step, cutoff=cutoff,
            reference_positions=mean_positions)
    else:
        frame_indices = np.arange(t0, t0 + len(d2min_series) * frame_step, frame_step)
    if softness_series is None:
        softness_series, _, _ = softness_trajectory(
            trajectory, t0=t0, frame_step=frame_step, mass_weight=mass_weight,
            n_soft_modes=n_soft_modes, pr_max=pr_max)

    n_atoms = len(trajectory[0])
    if elements is None:
        elements = np.array(["all"] * n_atoms)
    elements = np.asarray(elements)
    unique_elements = [str(e) for e in np.unique(elements)]
    colors = plt.cm.tab10(np.arange(len(unique_elements)))

    fields = [("D\u00b2_min (\u00c5\u00b2)", d2min_series, cm.inferno),
              ("Softness score (a.u.)", softness_series, cm.viridis)]
    cell = trajectory[t0].cell.lengths()

    # The two 3D maps are built IDENTICALLY in both modes (same figure size,
    # layout, colour scale, marker size, titles, colourbar). With distributions
    # the figure is two sub-figures: distributions on top, and below them
    # exactly the 16 x 7 in two-map figure of the maps-only animation.
    if show_distributions:
        fig = plt.figure(figsize=(16, 12))
        sub_top, sub_bottom = fig.subfigures(2, 1, height_ratios=[5, 7])
        ax_dist = list(sub_top.subplots(1, 2))
    else:
        fig = plt.figure(figsize=(16, 7))
        sub_bottom = fig
        ax_dist = [None, None]
    ax_3d = [sub_bottom.add_subplot(1, 2, 1, projection="3d"),
             sub_bottom.add_subplot(1, 2, 2, projection="3d")]

    info = []
    for (label, series, cmap), a3 in zip(fields, ax_3d):
        pooled, combined = pool_series_by_element(series, elements)
        xmin, xmax = _distribution_limits(series, combined)
        norm = mcolors.Normalize(vmin=0, vmax=np.nanpercentile(series[-1], 99))
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = sub_bottom.colorbar(sm, ax=a3, shrink=0.7, pad=0.1)
        cb.set_label(label, fontsize=9)
        info.append((pooled, xmin, xmax, norm, cmap))

    def update(i):
        for k, ((label, series, cmap), ad, a3) in enumerate(zip(fields, ax_dist, ax_3d)):
            pooled, xmin, xmax, norm, cm_ = info[k]
            name = label.split(" (")[0]
            if ad is not None:
                ad.clear()
                _draw_distribution_panel(
                    ad, series[i], elements, unique_elements, colors, xmin, xmax, bins,
                    label, f"{name} distribution (Frame {frame_indices[i]})",
                    average_pooled=pooled)
            a3.clear()
            vals = np.asarray(series[i], dtype=float)
            ok = np.isfinite(vals)
            cur = trajectory[frame_indices[i]].positions     # atoms at their CURRENT positions
            a3.scatter(cur[ok, 0], cur[ok, 1], cur[ok, 2], c=vals[ok], cmap=cm_,
                       norm=norm, s=10, alpha=0.8)
            a3.set_title(f"{name} relative to mean structure (Frame {frame_indices[i]})")
            a3.set_xlabel("x (A)"); a3.set_ylabel("y (A)"); a3.set_zlabel("z (A)")
            a3.set_xlim(0, cell[0]); a3.set_ylim(0, cell[1]); a3.set_zlim(0, cell[2])
            a3.view_init(elev=elev, azim=azim)
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=100, blit=False)
    anim.save(save_path, writer="ffmpeg", fps=fps, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
