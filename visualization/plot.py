# =============================================================
# visualization/plot.py — static plot functions
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import inspect
from scipy.signal import savgol_filter
from utils import get_output_dir
from config import VOXEL_EDGE
from config import R_CUT_VOID
OUTPUT_DIR = get_output_dir()



def plot_map(image, title, cmap="inferno", colorbar_label=""):
    plt.figure(figsize=(8, 6))
    plt.imshow(image.T, origin="lower", cmap=cmap)
    plt.colorbar().set_label(colorbar_label, fontsize=9)
    plt.title(title)
    plt.xlabel("X voxel")
    plt.ylabel("Y voxel")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title.replace(' ', '_').lower()}.png"), dpi=300)
    print(f"Saved: {title}")

def plot_cross_correlation(xcorr):
    cx, cy = np.array(xcorr.shape) // 2

    # 2D map
    plot_map(xcorr, "Cross_Correlation 2D", cmap="RdBu_r",
             colorbar_label="Cross-correlation (a.u.)")

    # radial profile
    x = np.arange(xcorr.shape[0]) - cx
    y = np.arange(xcorr.shape[1]) - cy
    X, Y = np.meshgrid(x, y, indexing="ij")
    R    = np.sqrt(X**2 + Y**2).astype(int)

    max_r       = min(cx, cy)
    radii       = np.arange(0, max_r)
    radial_mean = np.array([xcorr[R == r].mean() if np.any(R == r) else 0 for r in radii])

    plt.figure(figsize=(7, 4))
    plt.plot(radii, radial_mean)
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.xlabel("Shift distance (voxels)")
    plt.ylabel("Mean cross-correlation (a.u.)")
    plt.title("Radial Profile of Cross-Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cross_correlation_radial.png"), dpi=300)
    print("Saved: cross_correlation_radial.png")

def plot_average_diffraction(diffraction_patterns, 
                             blocked=True,
                     save_path=None):
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "diffraction.png")
    if diffraction_patterns.array.ndim == 4:
        mean_dp = diffraction_patterns.mean(axis=(0, 1))
        mean_dp.compute()
    else:
        mean_dp = diffraction_patterns

    if blocked:
        result = mean_dp.block_direct()
        result.compute()
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_blocked{ext}"
    else:
        result = mean_dp

    arr = np.array(result.array)
    plt.figure(figsize=(6, 6))
    plt.imshow(np.log1p(arr / arr.max()), cmap="inferno", origin="lower")
    plt.colorbar().set_label("log(1 + normalised intensity) (a.u.)", fontsize=9)
    title = "Mean Diffraction Pattern (BF blocked)" if blocked else "Mean Diffraction Pattern"
    plt.title(title)
    plt.xlabel("kx (pixels)")
    plt.ylabel("ky (pixels)")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    
    print(f"Saved: {save_path}")

def plot_diffraction_single_position(
    diffraction_patterns,
    blocked=True,
    scan_position=None,
    save_path=None,
):
    """
    Plot either:
      - one diffraction pattern at a specified scan position, or
      - the scan-averaged diffraction pattern.

    Parameters
    ----------
    diffraction_patterns
        4D abTEM DiffractionPatterns measurement.
    blocked
        Whether to block the direct beam.
    scan_position
        Optional tuple (ix, iy). When None, average over the scan.
    save_path
        Output image path.
    """

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "diffraction.png")
    measurement = diffraction_patterns

    if measurement.array.ndim == 4:
        if scan_position is None:
            result = measurement.mean(axis=(0, 1))
            description = "Mean diffraction pattern"
        else:
            ix, iy = scan_position
            result = measurement[ix, iy]
            description = (
                f"Diffraction pattern at scan position "
                f"({ix}, {iy})"
            )
    else:
        result = measurement
        description = "Diffraction pattern"

    if blocked:
        result = result.block_direct()
        description += " — direct beam blocked"

    result.compute()

    arr = np.asarray(result.array, dtype=np.float64)

    print("Diffraction plot statistics:")
    print(f"  shape:       {arr.shape}")
    print(f"  minimum:     {np.nanmin(arr):.6e}")
    print(f"  maximum:     {np.nanmax(arr):.6e}")
    print(f"  mean:        {np.nanmean(arr):.6e}")
    print(f"  median:      {np.nanmedian(arr):.6e}")
    print(f"  99th pct.:   {np.nanpercentile(arr, 99):.6e}")
    print(f"  99.9th pct.: {np.nanpercentile(arr, 99.9):.6e}")

    # Apply logarithm before display normalization.
    display = np.log1p(np.maximum(arr, 0.0))

    finite = np.isfinite(display)

    if not np.any(finite):
        raise ValueError(
            "Diffraction pattern contains no finite values."
        )

    vmin = np.nanpercentile(display[finite], 1.0)
    vmax = np.nanpercentile(display[finite], 99.9)

    if vmax <= vmin:
        vmin = np.nanmin(display[finite])
        vmax = np.nanmax(display[finite])

    plt.figure(figsize=(7, 6))

    plt.imshow(
        display,
        cmap="inferno",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar().set_label(
        "log(1 + intensity) (a.u.)",
        fontsize=9,
    )

    plt.title(description)
    plt.xlabel("kx (pixels)")
    plt.ylabel("ky (pixels)")
    plt.tight_layout()

    if blocked:
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_blocked{ext}"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

def plot_virtual_detector_heatmap(
    diffraction_patterns,
    inner_radius_px=0,
    outer_radius_px=None,
    center=None,
    log_scale=False,
    normalize=True,
    title="Virtual Detector Heatmap",
    save_path=None,
):
    """
    Combine all scan-position diffraction patterns into one 2D heatmap.

    For each scan position, the intensity inside a circular or annular
    diffraction-space region is integrated. The resulting scalar values
    form an image in real-space scan coordinates.

    Parameters
    ----------
    diffraction_patterns
        abTEM DiffractionPatterns measurement or NumPy-like 4D array
        with shape:

            (scan_x, scan_y, kx, ky)

    inner_radius_px : float
        Inner radius of the virtual detector in diffraction pixels.
        Use 0 for a circular detector.

    outer_radius_px : float or None
        Outer radius of the virtual detector in diffraction pixels.
        If None, the detector extends to the edge of the diffraction
        pattern.

    center : tuple[float, float] or None
        Diffraction-pattern center as (kx_center, ky_center).
        If None, the geometric center is used.

    log_scale : bool
        Apply log(1 + intensity) to the final scan-space image.

    normalize : bool
        Normalize the final image to the range [0, 1].

    title : str
        Plot title.

    save_path : str
        Path at which the heatmap is saved.

    Returns
    -------
    heatmap : np.ndarray
        Two-dimensional array with shape (scan_x, scan_y).

    detector_mask : np.ndarray
        Boolean mask applied to each diffraction pattern.
    """

    # Ensure the abTEM calculation has been evaluated.
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "virtual_detector_heatmap.png")
    if hasattr(diffraction_patterns, "compute"):
        diffraction_patterns.compute()

    if hasattr(diffraction_patterns, "array"):
        array = diffraction_patterns.array
    else:
        array = diffraction_patterns

    # Dask/CuPy-backed arrays may require explicit conversion.
    if hasattr(array, "compute"):
        array = array.compute()

    if hasattr(array, "get"):
        array = array.get()

    array = np.asarray(array)

    if array.ndim != 4:
        raise ValueError(
            "Expected a 4D array with shape "
            "(scan_x, scan_y, kx, ky), but received "
            f"shape {array.shape}."
        )

    scan_x, scan_y, nkx, nky = array.shape

    if center is None:
        center_kx = (nkx - 1) / 2
        center_ky = (nky - 1) / 2
    else:
        center_kx, center_ky = center

    if inner_radius_px < 0:
        raise ValueError("inner_radius_px must be non-negative.")

    if outer_radius_px is None:
        outer_radius_px = np.sqrt(nkx**2 + nky**2)

    if outer_radius_px <= inner_radius_px:
        raise ValueError(
            "outer_radius_px must be larger than inner_radius_px."
        )

    # Build the same diffraction-space mask for every scan position.
    kx, ky = np.meshgrid(
        np.arange(nkx),
        np.arange(nky),
        indexing="ij",
    )

    radius = np.sqrt(
        (kx - center_kx) ** 2
        + (ky - center_ky) ** 2
    )

    detector_mask = (
        (radius >= inner_radius_px)
        & (radius < outer_radius_px)
    )

    if not np.any(detector_mask):
        raise ValueError(
            "The selected virtual-detector region contains no pixels."
        )

    # Integrate the chosen detector region for every scan position.
    #
    # array shape:          (scan_x, scan_y, kx, ky)
    # detector_mask shape:  (kx, ky)
    # heatmap shape:        (scan_x, scan_y)
    heatmap = np.sum(
        array * detector_mask[None, None, :, :],
        axis=(-2, -1),
        dtype=np.float64,
    )

    if log_scale:
        heatmap = np.log1p(np.maximum(heatmap, 0))

    if normalize:
        minimum = np.nanmin(heatmap)
        maximum = np.nanmax(heatmap)

        if maximum > minimum:
            heatmap = (
                (heatmap - minimum)
                / (maximum - minimum)
            )

    output_directory = os.path.dirname(save_path)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    plt.figure(figsize=(7, 6))

    # Transpose to follow your existing convention:
    # horizontal axis = scan x, vertical axis = scan y.
    image = plt.imshow(
        heatmap.T,
        origin="lower",
        cmap="inferno",
        aspect="equal",
    )

    plt.colorbar(
        image,
        label=(
            "Normalized integrated intensity"
            if normalize
            else "Integrated intensity (a.u.)"
        ),
    )

    plt.xlabel("Scan x")
    plt.ylabel("Scan y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Virtual-detector mask pixels: {detector_mask.sum()}")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap minimum: {np.nanmin(heatmap):.6e}")
    print(f"Heatmap maximum: {np.nanmax(heatmap):.6e}")
    print(f"Saved: {save_path}")

    return heatmap, detector_mask

def plot_detector_mask(
    detector_mask,
):
    caller_frame = inspect.currentframe().f_back
    mask_name = next(
        (
            variable_name
            for variable_name, value in caller_frame.f_locals.items()
            if value is detector_mask
        ),
        "detector_mask",
    )

    save_path = os.path.join(OUTPUT_DIR, f"virtual_{mask_name}.png")

    plt.figure(figsize=(6, 6))
    plt.imshow(
        detector_mask.T,
        origin="lower",
        cmap="gray",
    )
    plt.xlabel("kx pixel")
    plt.ylabel("ky pixel")
    plt.title("Virtual Detector Region")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

def plot_pypty_reconstruction(obj_phase, obj_amp, probe_amp, output_path=None):
    """
    Plot a pypty single-slice reconstruction: object phase, object amplitude,
    and probe amplitude, side by side.
    """
    if hasattr(obj_phase, "compute"):
        obj_phase = obj_phase.compute()
    if hasattr(obj_phase, "get"):
        obj_phase = obj_phase.get()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(obj_phase.T, cmap="gray")
    axes[0].set_title("Object phase (projected potential)")
    axes[1].imshow(obj_amp.T, cmap="gray")
    axes[1].set_title("Object amplitude")
    axes[2].imshow(probe_amp.T, cmap="viridis")
    axes[2].set_title("Probe amplitude")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    return fig

def plot_pypty_vs_ground_truth(reconstructed, ground_truth, pearson_r, output_path=None):
    """
    Side-by-side comparison of the pypty-reconstructed phase map against the
    ground-truth projected potential from the original MD structure, plus
    their difference map.
    """
    diff = reconstructed - ground_truth

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(reconstructed.T, cmap="gray")
    axes[0].set_title("Reconstructed (pypty)")
    axes[1].imshow(ground_truth.T, cmap="gray")
    axes[1].set_title("Ground truth (avg MD structure)")
    im = axes[2].imshow(diff.T, cmap="RdBu")
    axes[2].set_title(f"Difference (Pearson r = {pearson_r:.3f})")
    for ax in axes:
        ax.axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    return fig

def plot_pair_correlation(r, g_r, cutoff_r=None, peak_r=None,
                           save_path=None):
    """
    Plot g(r) with the detected first peak and neighbor cutoff marked.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "pair_correlation.png")
    plt.figure(figsize=(7, 4))
    plt.plot(r, g_r, color="black", linewidth=1.2)
    if peak_r is not None:
        plt.axvline(peak_r, color="tab:blue", linestyle="--",
                     label=f"1st peak: {peak_r:.2f} A")
    if cutoff_r is not None:
        plt.axvline(cutoff_r, color="tab:red", linestyle="--",
                     label=f"cutoff (1st minimum): {cutoff_r:.2f} A")
    plt.xlabel("r (A)")
    plt.ylabel("g(r)")
    plt.title("Radial Pair Correlation Function")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_d2min_histogram(d2min, save_path=None):
    """
    Histogram of D^2_min values across all atoms (log-y, since the
    distribution is typically heavy-tailed: most atoms sit near the
    bulk value, with a minority showing much larger non-affine motion).
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "d2min_histogram.png")
    valid = d2min[~np.isnan(d2min)]
    plt.figure(figsize=(7, 4))
    plt.hist(valid, bins=80, color="steelblue", edgecolor="none")
    plt.yscale("log")
    plt.axvline(np.median(valid), color="black", linestyle="--",
                 label=f"median = {np.median(valid):.2f} A^2")
    plt.xlabel("D^2_min (A^2)")
    plt.ylabel("Atom count (log scale)")
    plt.title("Distribution of Non-Affine Squared Displacement")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_d2min_map(atoms_ref, d2min, save_path=None):
    """
    2D (x-y) scatter of atoms, colored by D^2_min — a real-space map
    of where non-affine (flow-unit-like) activity is concentrated,
    integrated over the whole z-thickness of the cell.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "d2min_map.png")
    pos = atoms_ref.positions
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(pos[:, 0], pos[:, 1], c=d2min, cmap="inferno",
                      s=14, vmax=np.nanpercentile(d2min, 99))
    plt.colorbar(sc).set_label("D^2_min (A^2)", fontsize=9)
    plt.xlabel("x (A)")
    plt.ylabel("y (A)")
    plt.title("Non-Affine Displacement Map (D^2_min)")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_d2min_map_3d(atoms_ref, d2min, save_path=None,
                       elev=20, azim=45):
    """
    3D (x-y-z) scatter of atoms, colored by D^2_min. Unlike
    plot_d2min_map (which integrates over the full z-thickness and
    can hide real 3D clustering behind overlapping points), this
    shows each atom at its actual depth — rotate/re-render with
    different elev/azim to check for localization the 2D projection
    might be washing out.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "d2min_map_3d.png")
    pos = atoms_ref.positions
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=d2min, cmap="inferno",
                     s=10, vmax=np.nanpercentile(d2min, 99), alpha=0.8)
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1).set_label("D^2_min (A^2)", fontsize=9)
    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_zlabel("z (A)")
    ax.set_title("Non-Affine Displacement Map (D^2_min) — 3D")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_d2min_defect_comparison(stats, save_path=None):
    """
    Mean D^2_min for defect vs non-defect atoms over the trajectory
    (relative to frame 0). If the defect line consistently sits above
    the non-defect line, defects are tracking — and plausibly driving —
    the non-affine dynamics; if the two lines stay close, d2min is
    more diffuse and not concentrated at defect sites.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "d2min_defect_comparison.png")
    fig, ax = plt.subplots(figsize=(7, 5))

    frames = stats["frame_indices"]
    ax.plot(frames, stats["mean_defect"], label="Defect atoms", color="crimson")
    ax.plot(frames, stats["mean_nondefect"], label="Non-defect atoms", color="steelblue")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean D²_min (Å²)")
    ax.set_title("D²_min by defect membership, relative to frame 0")
    ax.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

    return fig, ax


def _smooth_series(y, window=7, polyorder=2):
    """
    Savitzky-Golay smoothing for display only (doesn't touch the
    underlying stats). matplotlib has no built-in smoothing — this is
    what scipy.signal provides for it. NaN frames (e.g. no hole peaks
    detected that frame) are linearly interpolated over just for the
    filter pass, then restored as gaps in the output so missing data
    still reads as missing rather than a fabricated smooth value.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    nan_mask = np.isnan(y)
    if n < 3 or nan_mask.all():
        return y
 
    x = np.arange(n)
    filled = y.copy()
    if nan_mask.any():
        filled[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], y[~nan_mask])
 
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    w = max(w, 3)
    p = min(polyorder, w - 1)
 
    smoothed = savgol_filter(filled, window_length=w, polyorder=p)
    smoothed[nan_mask] = np.nan
    return smoothed
 
 
def plot_d2min_hole_distance(stats, save_path=None, smooth=False, smooth_window=10):
    """
    Mean D^2_min per distance-to-nearest-hole group (near/mid/far,
    recomputed each frame), over the trajectory. If the "near" line
    sits consistently above "far", d2min is concentrated close to
    wherever holes currently are.
 
    smooth=True applies a Savitzky-Golay filter (window=smooth_window,
    cubic-ish local fit) to each bin's line for display; the returned
    stats dict / bin_means are untouched.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "d2min_hole_distance.png")
 
    fig, ax = plt.subplots(figsize=(7, 5))
 
    frames = stats["frame_indices"]
    bin_means = stats["bin_means"]
    n_bins = bin_means.shape[1]
    prefixes = ["Near", "Mid", "Far"] if n_bins == 3 else [f"Bin {b}" for b in range(n_bins)]
    mean_edges = np.nanmean(stats["bin_edges"], axis=0)  # shape (n_bins+1,), Å
    labels = [
        f"{prefixes[b]} ({mean_edges[b]:.1f}\u2013{mean_edges[b + 1]:.1f} \u00c5, avg)"
        for b in range(n_bins)
    ]
    colors = cm.viridis(np.linspace(0, 0.85, n_bins))
 
    for b in range(n_bins):
        y = bin_means[:, b]
        if smooth:
            y = _smooth_series(y, window=smooth_window)
        ax.plot(frames, y, label=labels[b], color=colors[b])
 
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean D²_min (Å²)")
    ax.set_title("D²_min by distance to nearest hole peak")
    ax.legend()
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
 
    return fig, ax



# =============================================================
# Soft-mode / TLS analysis plots
#   (analysis/soft_modes/modes.py, analysis/soft_modes/tls.py,
#    analysis/soft_modes/comparison.py)
# =============================================================

def plot_participation_spectrum(eigenvalues, participation_ratio,
                                soft_mode_mask=None, elements=None, save_path=None,
                                all_elements=None, element_offset_pt=7.0):
    """
    Participation ratio vs. mode eigenvalue.

    colour  : soft_mode_mask -- soft/localized modes (red) vs. other modes (blue)
    marker  : element that dominates the mode (`elements`, a (K,) array of symbols);
              two separate legends, one for colour and one for marker shape.

    Every element stays visible, for any number of elements: modes with
    (almost) the same eigenvalue and participation ratio -- e.g. the many
    noise-floor modes -- sit exactly on top of each other, so one element would
    hide the others. Therefore
      * each element is shifted sideways by `element_offset_pt` points ON THE
        SCREEN (drawing offset only, the data values are unchanged), turning a
        stack into one column per element;
      * the most numerous element is drawn first in the original small style,
        every other (rarer) element larger, opaque, black-edged and on top;
      * the legend shows the number of modes per element.
    all_elements : optional list of every element in the simulation. Gives each
        element the same marker in every plot (e.g. the 3D plot and the
        per-slice plots) and lists elements that dominate no mode (n=0).
    """
    from matplotlib.transforms import ScaledTranslation
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "participation_spectrum.png")
    eigenvalues = np.asarray(eigenvalues)
    participation_ratio = np.asarray(participation_ratio)
    n_modes = len(eigenvalues)
    if len(participation_ratio) != n_modes:
        raise ValueError("eigenvalues and participation_ratio must have the same length")
    if soft_mode_mask is not None:
        soft_mode_mask = np.asarray(soft_mode_mask, bool)
    markers = ["o", "^", "s", "D", "v", "P", "X", "*", "<", ">"]

    if elements is not None:
        elements = np.asarray(elements)
        if len(elements) != n_modes:
            raise ValueError("elements must have one entry per mode")
        names = list(np.unique(elements))
        if all_elements is not None:
            names = sorted(set(map(str, all_elements)) | set(map(str, names)))
        counts = {el: int((elements == el).sum()) for el in names}
        order = sorted(names, key=lambda el: -counts[el])          # most numerous first
        marker_of = {el: markers[i % len(markers)] for i, el in enumerate(names)}
        n_el = len(names)
        shift = {el: (i - (n_el - 1) / 2) * element_offset_pt for i, el in enumerate(names)}
    else:
        names, counts, order = [None], {None: n_modes}, [None]
        marker_of, shift = {None: "o"}, {None: 0.0}

    fig, ax = plt.subplots(figsize=(7, 5))
    for rank, el in enumerate(order):
        main = rank == 0                                            # most numerous element
        el_mask = (elements == el) if elements is not None else np.ones(n_modes, bool)
        if not el_mask.any():
            continue
        # most numerous: original look; rarer elements: bigger, opaque, edged, on top
        style = dict(s_other=12, s_soft=24, a_other=0.6, a_soft=0.85, edge="none", lw=0)
        if not main:
            style = dict(s_other=40, s_soft=60, a_other=1.0, a_soft=1.0, edge="black", lw=0.4)
        groups = ([(el_mask & ~soft_mode_mask, "steelblue", style["s_other"], style["a_other"]),
                   (el_mask & soft_mode_mask, "crimson", style["s_soft"], style["a_soft"])]
                  if soft_mode_mask is not None else
                  [(el_mask, "steelblue", style["s_other"], 0.7 if main else 1.0)])
        for sel, color, size, alpha in groups:
            if not sel.any():
                continue
            sc = ax.scatter(eigenvalues[sel], participation_ratio[sel], s=size, c=color,
                            alpha=alpha, marker=marker_of[el], edgecolors=style["edge"],
                            linewidths=style["lw"], zorder=3 + rank)
            if shift[el]:
                # drawing offset only (data unchanged; also works with log axes)
                sc.set_offset_transform(sc.get_offset_transform() +
                                        ScaledTranslation(shift[el] / 72, 0, fig.dpi_scale_trans))

    if soft_mode_mask is not None:
        color_handles = [plt.Line2D([0], [0], marker="o", color="steelblue", linestyle="", markersize=6, label="other modes"),
                         plt.Line2D([0], [0], marker="o", color="crimson", linestyle="", markersize=7, label="soft/localized modes")]
        color_legend = ax.legend(handles=color_handles, loc="upper right", title="mode type")
        ax.add_artist(color_legend)
    if elements is not None:
        shape_handles = [plt.Line2D([0], [0], marker=marker_of[el], color="gray", linestyle="",
                                    markersize=7, label=f"{el} (n={counts[el]})") for el in names]
        ax.legend(handles=shape_handles, loc="lower right",
                  title="dominant element" + ("\n(shifted sideways)" if len(names) > 1 else ""))
    ax.set_xlabel(r"Mode eigenvalue $\langle a_n^2\rangle$ (Å$^2$) — larger = softer")
    ax.set_ylabel("Participation ratio")
    ax.set_title("Participation Ratio vs. Mode Softness")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_mode_atom_distribution_by_element(eigenvectors, elements, soft_mode_mask=None,
                                           save_dir=None, bins=60):
    """
    One figure per element showing with which atoms every mode is populated.

    top    : share of each mode's total participation carried by this
             element's atoms (dashed line = the element's share of the atoms;
             above it the element is over-represented in that mode)
    bottom : for every mode, the distribution over this element's atoms of
             the per-atom participation p[i, n] (shown as N * p[i, n]:
             1 = an equal share of the mode, 10 = ten times an equal share)
             -- colour = fraction of this element's atoms in that bin.
    Modes are in the order given (softest first); soft modes are shaded.
    """
    from matplotlib.colors import LogNorm
    from analysis.soft_modes.element_stats import element_mode_participation
    if save_dir is None:
        save_dir = OUTPUT_DIR
    elements = np.asarray(elements)
    p, share, population = element_mode_participation(eigenvectors, elements)
    N, K = p.shape
    soft = np.zeros(K, dtype=bool) if soft_mode_mask is None else np.asarray(soft_mode_mask, bool)

    rel = np.log10(np.maximum(N * p, 1e-8))               # log10 of N * p[i, n]
    edges = np.linspace(rel.min(), rel.max(), bins + 1)

    for el in share:
        sel = elements == el
        hist = np.stack([np.histogram(rel[sel, n], bins=edges)[0] / sel.sum()
                         for n in range(K)], axis=1)         # (bins, K)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [1, 2]})
        modes = np.arange(K)
        ax1.plot(modes, share[el], "-", color="tab:blue")
        ax1.axhline(population[el], ls="--", color="k", lw=1,
                    label=f"{el} share of atoms ({population[el]:.2f})")
        ax1.set_ylabel(f"{el} share of mode")
        ax1.set_title(f"How the modes are populated -- {el} atoms (n={int(sel.sum())})")
        ax1.legend()
        im = ax2.pcolormesh(np.arange(K + 1) - 0.5, edges, hist, cmap="viridis",
                            norm=LogNorm(vmin=1.0 / sel.sum(), vmax=hist.max()))
        ax2.set_xlabel("mode index (0 = softest)")
        ax2.set_ylabel(r"log$_{10}$(N $\cdot$ participation of atom)")
        fig.colorbar(im, ax=[ax1, ax2], label=f"fraction of {el} atoms", pad=0.02)
        if soft.any():
            for ax in (ax1, ax2):
                ax.axvspan(modes[soft].min() - 0.5, modes[soft].max() + 0.5,
                           color="tab:red", alpha=0.12, lw=0)
        path = os.path.join(save_dir, f"participation_per_mode_{el}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def plot_softness_map(atoms_ref, softness, atom_indices=None,
                       save_path=None, title="Soft-Spot Map"):
    """
    Flat 2D (x-y) PROJECTION of atoms colored by softness score — same
    layout as plot_d2min_map. This is a projection, not a true 3D
    view: two atoms at very different z can land on top of each other
    here. Fine for a single z-slice (where all plotted atoms already
    share roughly the same z by construction) or a quick 2D overview;
    for the actual 3D analysis (soft_modes.soft_mode_analysis_3d()),
    use plot_softness_map_3d() instead so z isn't thrown away.

    `atom_indices` restricts the plotted subset (e.g. a single
    z-slice's atoms); `softness` may be full-length (indexed by real
    atom index, NaNs elsewhere) or already restricted to atom_indices.
    `title` lets you label which slice/subset this is.
    """

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "softness_map.png")

    softness = np.asarray(softness)
    if atom_indices is not None and len(softness) == len(atoms_ref):
        pos = atoms_ref.positions[atom_indices]
        vals = softness[atom_indices]
    elif atom_indices is not None:
        pos = atoms_ref.positions[atom_indices]
        vals = softness
    else:
        pos = atoms_ref.positions
        vals = softness

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(pos[:, 0], pos[:, 1], c=vals, cmap="viridis", s=14,
                      vmax=np.nanpercentile(vals, 99))
    plt.colorbar(sc).set_label("Softness score (a.u.)", fontsize=9)
    plt.xlabel("x (Å)")
    plt.ylabel("y (Å)")
    plt.title(title)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_mode_amplitude_distribution(a_n, gmm_fit=None, label="mode amplitude",
                                      save_path=None):
    """
    Histogram of P(a_n) for a single mode/cage-coordinate amplitude
    time series, with the fitted single- and two-Gaussian models
    overlaid — the plot behind "a typical [oscillator] distribution
    would have just one peak ... a TLS one would have two." `gmm_fit`
    is the per-item dict from tls_detection.bimodality_test_batch()
    (needs "means", "sigmas", "weights" for the 2-component curve).
    """

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "amplitude_distribution.png")

    a_n = np.asarray(a_n).ravel()
    plt.figure(figsize=(7, 5))
    plt.hist(a_n, bins=60, density=True, color="steelblue", alpha=0.6, label="P(a_n)")
    x = np.linspace(a_n.min(), a_n.max(), 400)

    mu, sigma = a_n.mean(), a_n.std()
    plt.plot(x, np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
              "k--", label="single Gaussian fit")

    if gmm_fit is not None:
        means = np.ravel(gmm_fit["means"])
        sigmas = np.ravel(gmm_fit["sigmas"])
        weights = np.ravel(gmm_fit["weights"])
        mix = sum(w * np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
                   for m, s, w in zip(means, sigmas, weights))
        plt.plot(x, mix, "crimson", linewidth=2, label="2-component (TLS) fit")

    plt.xlabel(f"{label} (Å)")
    plt.ylabel("Probability density")
    plt.title(f"Amplitude Distribution — {label}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curve(fpr, tpr, auc, title="Predicting D²_min rearrangement",
                    save_path=None):
    """
    ROC curve for a score (softness or TLS-ness) predicting a later
    D^2_min rearrangement event — see
    analysis.soft_spot_tls_dmin_comparison.predict_d2min_from_score().
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "roc_curve.png")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="crimson", linewidth=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_soft_tls_scatter(softness, tls_score, save_path=None):
    """
    Two-panel visualization of
    analysis.soft_spot_tls_dmin_comparison.compare_soft_tls().

    tls_score is typically sparse/near-binary in this pipeline (zero
    for the large majority of atoms neither TLS method flagged, a
    small continuous participation-weighted value for the rest) --
    a bare scatter plot of that against softness tends to collapse
    into one dense horizontal line at y=0 with a scattering of points
    above it, which is hard to read at a glance. So:

      Left  : the scatter itself (with a little y-jitter purely for
              visibility — the jitter carries no information, don't
              read anything into its vertical spread at a given
              tls_score value).
      Right : the SAME comparison redrawn as two box plots -- softness
              scores for atoms flagged as TLS (tls_score > 0) vs.
              atoms that weren't. This is usually the more directly
              readable version: if soft spots and TLS genuinely
              overlap, the "TLS flagged" box should sit visibly higher
              than the "not TLS" box.
    """

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "soft_vs_tls.png")

    softness = np.asarray(softness, dtype=float)
    tls_score = np.asarray(tls_score, dtype=float)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    jitter_scale = 0.03 * (np.nanmax(tls_score) - np.nanmin(tls_score) + 1e-9)
    jitter = rng.normal(0, jitter_scale, size=len(tls_score))
    axes[0].scatter(softness, tls_score + jitter, s=10, c="steelblue", alpha=0.4)
    axes[0].set_xlabel("Softness score (a.u.)")
    axes[0].set_ylabel("TLS score (+ small y-jitter, for visibility only)")
    axes[0].set_title("Per-atom scatter")

    flagged = tls_score > 0
    groups = [softness[~flagged], softness[flagged]]
    axes[1].boxplot(groups, showmeans=True)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels([f"not TLS\n(n={int((~flagged).sum())})",
                              f"TLS flagged\n(n={int(flagged.sum())})"])
    axes[1].set_ylabel("Softness score (a.u.)")
    axes[1].set_title("Softness, split by TLS status")

    fig.suptitle("Soft-Spot Score vs. TLS Score")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_distribution_by_element(series, elements, xlabel, title, bins=40, save_path=None):
    """
    Average distribution of a per-atom field, per element, pooled over the
    whole trajectory (every atom at every frame), plus the combined
    (all-elements) distribution.

    series : (T, N) array (NaN = "not enough data yet", ignored)
    elements : (N,) array of element symbols
    """
    from analysis.soft_modes.element_stats import pool_series_by_element
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "distribution_by_element.png")

    pooled, combined = pool_series_by_element(series, elements)
    xmin, xmax = np.min(combined), np.percentile(combined, 99.5)
    edges = np.linspace(xmin, xmax, bins + 1)
    colors = plt.cm.tab10(np.arange(len(pooled)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for color, (el, vals) in zip(colors, pooled.items()):
        ax.hist(vals, bins=edges, density=True, alpha=0.45, color=color,
                label=f"{el} (n={len(vals)})")
    ax.hist(combined, bins=edges, density=True, histtype="step", color="black",
            linewidth=2, label=f"Combined (n={len(combined)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_d2min_by_soft_neighbours(n_soft_neighbours, soft_mask, values, null=None,
                                  value_label="D²_min (Å²)", max_group=5, save_path=None):
    """
    D^2_min of NON-soft atoms grouped by how many soft atoms they have as
    neighbours (0, 1, 2, ... and "k+" for the last group).

    n_soft_neighbours : (N,) int array -- soft neighbours of every atom
    soft_mask : (N,) bool array -- soft atoms (excluded from the groups)
    values : dict {panel title: (N,) array}, e.g. {"raw D2_min": ..., "excess": ...}
        -- one panel per entry (sharing the x axis)
    null : dict {panel title: {k: (mean, std)}}, optional -- the group mean expected
        by chance (soft labels permuted within (element, coordination number)),
        drawn as a grey band, e.g. soft_source_analysis()[...]["spatial"]
        ["null_by_n_soft_neighbours"]

    max_group : the last group collects every atom with max_group or more soft neighbours.

    Per group: box = quartiles (line = median), whiskers = 5th-95th
    percentile, red dot with bar = mean +- standard error, n written on top.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "d2min_by_soft_neighbours.png")
    n_soft_neighbours = np.asarray(n_soft_neighbours)
    soft_mask = np.asarray(soft_mask, bool)
    # use exactly the groups the null was computed for (last group = "top or more")
    top = max_group                      # last group = "max_group or more"
    if null is not None:
        for d in null.values():
            if set(d) != set(range(top + 1)):
                raise ValueError(f"null was computed for groups {sorted(d)} but max_group={max_group}; "
                                 "pass the same max_group to soft_source_analysis")
    groups = list(range(top + 1))
    labels = [str(g) if g < top else f"{g}+" for g in groups]
    in_group = [(~soft_mask) & ((n_soft_neighbours == g) if g < top else (n_soft_neighbours >= g))
                for g in groups]

    fig, axes = plt.subplots(1, len(values), figsize=(5.5 * len(values), 5), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (title, v) in zip(axes, values.items()):
        v = np.asarray(v, dtype=float)
        data = [v[m][np.isfinite(v[m])] for m in in_group]
        ax.boxplot(data, positions=groups, whis=(5, 95), showfliers=False, widths=0.55,
                   medianprops=dict(color="black"), boxprops=dict(color="tab:blue"),
                   whiskerprops=dict(color="tab:blue"), capprops=dict(color="tab:blue"))
        means = np.array([d.mean() for d in data])
        sem = np.array([d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan for d in data])
        if null is not None and title in null:
            for g in groups:
                m0, s0 = null[title][g]
                ax.fill_between([g - 0.4, g + 0.4], [m0 - s0] * 2, [m0 + s0] * 2,
                                color="grey", alpha=0.35, zorder=1,
                                label="expected by chance (mean ± std)" if g == 0 else None)
        ax.errorbar(groups, means, yerr=sem, fmt="o", color="tab:red", capsize=3, zorder=5,
                    label="mean ± s.e.m.")
        ax.set_xticks(groups)
        ax.set_xticklabels([f"{lab}\n(n={len(d)})" for lab, d in zip(labels, data)])
        ax.set_xlabel("number of soft neighbours")
        ax.set_ylabel(value_label)
        ax.set_title(title)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=len(handles), fontsize=9)
    fig.suptitle("D²_min of non-soft atoms vs. number of soft neighbours (final frame)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_mode_amplitude_grid(diag, ncols=4, bins=None, show_time_series=True,
                             amplitude_unit="√amu·Å", save_path=None):
    """
    P(a_n) and a_n(t), one panel per mode (or per quantity).
    `diag` = analysis.soft_modes.tls.mode_amplitude_diagnostics(...)
    (softest modes) or candidate_diagnostics(...) (one TLS candidate).

    a_n(t) is the amplitude of mode n in frame t: ONE number per frame that
    says how far the whole configuration is displaced along that mode.
    It is shown in units of its own standard deviation sigma_n (so the
    axes are dimensionless and all panels can be compared; sigma_n itself,
    in its own unit, is written in each title).

    Top panel (histogram):
      x axis      a_n / sigma_n
      y axis      probability density (per sigma_n); the area of all bars is 1
      grey bars   share of the frames in which a_n fell into that bin
      ticks       every single frame, along the bottom edge
      red line    ONE Gaussian (mean 0, width 1): an ordinary single-well vibration
      green dash  the 2-Gaussian mixture the TLS detector fitted
    Bottom panel (show_time_series=True):
      x axis      frame number,   y axis  a_n / sigma_n at that frame
    Title: mode index, eigenvalue, participation ratio, sigma_n,
    DeltaBIC (> 6: two Gaussians fit clearly better), BC = bimodality
    coefficient (> 0.555 hints at two peaks), T_eff = effective number of
    independent frames. A panel the detector flags as TLS gets a red title.

    If diag has "labels" / "units" / "title" (candidate_diagnostics), they
    replace the default mode title, the unit and the figure title.

    amplitude_unit : unit of sigma_n for the title when diag has no "units":
        "√amu·Å" if the modes were mass-weighted (the default pipeline), "Å" otherwise.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "mode_amplitude_distributions.png")
    a = np.asarray(diag["a_std"])                               # (M, T)
    M, T = a.shape
    if bins is None:
        bins = int(np.clip(np.sqrt(T), 8, 30))
    nrows = int(np.ceil(M / ncols))
    rows_per_mode = 2 if show_time_series else 1
    fig_h = (3.9 if show_time_series else 3.0) * nrows + 1.0
    fig = plt.figure(figsize=(3.9 * ncols, fig_h))
    gs = fig.add_gridspec(nrows * rows_per_mode, ncols,
                          height_ratios=([3, 1.2] if show_time_series else [1]) * nrows,
                          hspace=1.0 if show_time_series else 0.7, wspace=0.4,
                          top=1 - 1.5 / fig_h, bottom=0.6 / fig_h)

    lim = max(3.5, float(np.abs(a).max()) * 1.05)
    x = np.linspace(-lim, lim, 400)
    gauss = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    for m in range(M):
        r, c = divmod(m, ncols)
        ax = fig.add_subplot(gs[r * rows_per_mode, c])
        ax.hist(a[m], bins=bins, range=(-lim, lim), density=True, color="0.7", edgecolor="white")
        ax.plot(x, gauss(0.0, 1.0), color="tab:red", lw=1.5)
        mix = sum(diag["weights"][m, k] * gauss(diag["means"][m, k], diag["sigmas"][m, k])
                  for k in range(2))
        ax.plot(x, mix, color="tab:green", ls="--", lw=1.5)
        ax.plot(a[m], np.full(T, -0.02 * ax.get_ylim()[1]), "|", color="k", alpha=0.5, ms=5,
                clip_on=False)
        ax.set_xlim(-lim, lim)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(r"$a_n/\sigma_n$  (dimensionless)", fontsize=7)
        ax.set_ylabel(r"probability density (per $\sigma_n$)", fontsize=7)
        unit = diag["units"][m] if diag.get("units") is not None else amplitude_unit
        if diag.get("labels") is not None:
            head = f"{diag['labels'][m]}\n"
        else:
            pr = "" if diag["participation"] is None else f"  PR={diag['participation'][m]:.2f}"
            head = f"mode {diag['mode_idx'][m]}  λ={diag['eigenvalues'][m]:.2g}{pr}  "
        ax.set_title(f"{head}σ={diag['sigma'][m]:.2g} {unit}\n"
                     f"ΔBIC={diag['delta_bic'][m]:.1f}  BC={diag['bimodality_coefficient'][m]:.2f}  "
                     f"T_eff={diag['t_eff'][m]:.0f}",
                     fontsize=7.5, color="tab:red" if diag["is_tls"][m] else "black")
        if show_time_series:
            axt = fig.add_subplot(gs[r * rows_per_mode + 1, c])
            axt.plot(a[m], color="k", lw=0.8)
            axt.axhline(0, color="0.6", lw=0.5)
            axt.set_ylim(-lim, lim)
            axt.set_xlim(0, T - 1)
            axt.tick_params(labelsize=7)
            axt.set_xlabel("frame", fontsize=7)
            axt.set_ylabel(r"$a_n/\sigma_n$", fontsize=7)

    fig.suptitle(diag.get("title", "P(a_n) of the softest modes") +
                 "\n(red = single Gaussian, green dashed = detector's 2-Gaussian fit)",
                 fontsize=11, y=1 - 0.05 / fig_h)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

def visualize_holes_3d(atoms, void_grid, labeled, num_features,
                       spacing=VOXEL_EDGE, max_holes=5,
                       save_path=None):
    """
    Scatter plot of atoms (grey) overlaid with the largest
    detected hole regions (coloured).
    """
    from holes.kdtree import get_grid_coords
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "largest_holes_3d.png")
    X, Y, Z = get_grid_coords(atoms, spacing)

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")

    pos = atoms.positions
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               s=1, alpha=0.1, color="gray", label="Atoms")

    hole_sizes = [
        (i, np.sum(labeled == i))
        for i in range(1, num_features + 1)
    ]
    hole_sizes.sort(key=lambda x: x[1], reverse=True)

    for idx, (hole_id, size) in enumerate(hole_sizes[:max_holes]):
        mask = (labeled == hole_id)
        ax.scatter(X[mask], Y[mask], Z[mask],
                   s=5, label=f"Hole {idx} (size={size})")

    ax.set_xlabel("X (Å)"); ax.set_ylabel("Y (Å)"); ax.set_zlabel("Z (Å)")
    ax.legend()
    plt.title("Largest Holes (voxel representation)")
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_distribution(volumes, r_cut=R_CUT_VOID, spacing=VOXEL_EDGE,
                      save_path=None):
    all_v = np.concatenate(volumes)
    plt.hist(all_v, bins=50)
    plt.xlabel("Hole Volume (Å³)")
    plt.ylabel("Count")
    plt.title("Hole Size Distribution")
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f"hole_distribution_cutoff{r_cut}Å_spacing{spacing}Å.png")
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_degree_by_element(graphs, max_frames=10,
                            save_path=None):
    from graph.defects import compute_degrees
    import matplotlib.pyplot as plt
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "degree_distribution_by_element.png")
    deg_Zr, deg_Cu = [], []
    for g in graphs[:max_frames]:
        deg = compute_degrees(g)
        Z   = g.x.view(-1)
        deg_Zr.append(deg[Z == 40].numpy())
        deg_Cu.append(deg[Z == 29].numpy())

    deg_Zr = np.concatenate(deg_Zr)
    deg_Cu = np.concatenate(deg_Cu)

    plt.figure()
    plt.hist(deg_Zr, bins=40, alpha=0.6, label="Zr")
    plt.hist(deg_Cu, bins=40, alpha=0.6, label="Cu")
    plt.xlabel("Coordination number (degree)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Degree Distribution by Element")
    plt.savefig(save_path, dpi=300)
    plt.show()

    return deg_Zr, deg_Cu


def plot_pca_reliability(res, eigenvalue_unit="amu·Å²", save_path=None):
    """
    Left: eigenvalues of the displacement covariance matrix (the PCA spectrum),
    and how much of them is still there in frames that were not used to find the modes.
    Right: that fraction, mode by mode, against what pure chance gives.
    `res` = analysis.soft_modes.pca_reliability.pca_reliability_test(...).

    Left panel (log scale)
      x axis      mode index n (0 = softest = largest eigenvalue)
      y axis      eigenvalue lambda_n = variance of the amplitude a_n(t);
                  unit: amu·Å² if the displacements were mass-weighted, else Å²
      black       all frames (the eigenvalues the pipeline uses)
      grey        half of the frames: the eigenvalues of the PCA done on one half
      red         the variance along those same modes in the OTHER half
      dotted      variance of a random direction (total variance / number of coordinates)
    Right panel (log scale)
      y axis      retained fraction = red / grey:  1 = the mode is as strong in frames it
                  never saw (real);  small = the eigenvalue was mostly fitted noise
      grey band   chance (median to q of the null runs: same procedure on data in which
                  every coordinate is shifted in time by its own random amount)
      red dots    the leading modes above the band; open dots: the rest
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "pca_reliability.png")
    eig = np.asarray(res["eigenvalues"])
    ret = np.asarray(res["retained"])
    Ks = len(ret)
    n_rep = res["n_reproducible"]
    q = int(round(100 * res["quantile"]))
    T, P = res["n_frames"], res["n_coords"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax.plot(np.arange(len(eig)), eig, color="k", lw=1, marker="o", ms=3, label="all frames")
    ax.plot(np.arange(Ks), res["in_sample"], color="0.5", ls="--", lw=1, marker="o", ms=3,
            label="half of the frames (in-sample)")
    ax.plot(np.arange(Ks), res["held_out"], color="tab:red", lw=1, marker="o", ms=3,
            label="same modes, other half (held out)")
    ax.axhline(res["noise_floor"], color="0.3", ls=":", lw=1, label="random direction")
    ax.set_yscale("log")
    ax.set_xlabel("mode index n  (0 = softest)")
    ax.set_ylabel(f"eigenvalue $\\lambda_n$  ({eigenvalue_unit})")
    ax.set_title(f"Covariance eigenvalues:  {T} frames ($T_{{eff}}$ = {res['t_eff']:.1f}), "
                 f"{P} coordinates (P/T = {P / T:.1f})", fontsize=10)
    ax.legend(fontsize=8)

    n = np.arange(Ks)
    ax2.fill_between(n, res["retained_null_median"], res["retained_null_high"], color="0.85",
                     label=f"chance (median to {q}% of {res['n_null']} runs)")
    ax2.plot(n, ret, color="k", lw=1)
    ax2.scatter(n[n_rep:], ret[n_rep:], facecolors="none", edgecolors="k", s=18, zorder=3)
    ax2.scatter(n[:n_rep], ret[:n_rep], color="tab:red", s=24, zorder=4,
                label=f"above chance: {n_rep} modes")
    ax2.axhline(1.0, color="0.3", ls=":", lw=1)
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=min(float(np.min(res["retained_null_median"])), float(ret.min())) / 2, top=2.0)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.set_xlabel("mode index n  (0 = softest)")
    ax2.set_ylabel("retained fraction  (held-out variance / eigenvalue)")
    ax2.set_title("How much of each eigenvalue is real?", fontsize=10)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_softness_reproducibility(res, save_path=None):
    """
    Does the per-atom softness map survive when it is computed from only half of the frames?
    `res` = analysis.soft_modes.pca_reliability.softness_reproducibility_test(...).

    The whole soft-mode analysis is run on the first half of the frames and on the second half;
    each gives one softness value per atom.
    Left      every atom is one dot: its softness RANK among all atoms in the first half (x) against
              its rank in the second half (y), as a percentile (100 = softest). Dots on the diagonal =
              same ordering. The red box holds the softest top_fraction of the atoms in both halves.
    Middle    Spearman rank correlation of the two maps (1 = identical ordering, 0 = unrelated).
    Right     share of the top_fraction softest atoms of one half that are also among the top_fraction
              softest of the other half.
    Middle and right: red line = observed; grey histogram = the same number on `n_null` data sets in
    which every coordinate is shifted in time by its own random amount (independent coordinates, each
    with its own variance and autocorrelation); dashed grey = its `quantile`; dotted = two unrelated maps;
    blue = the same number for the per-atom mean-square displacement map (no PCA).
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "softness_reproducibility.png")
    a, b = np.asarray(res["softness_a"]), np.asarray(res["softness_b"])
    N = len(a)
    f = res["top_fraction"]
    q = int(round(100 * res["quantile"]))
    pa = 100 * np.argsort(np.argsort(a)) / (N - 1)
    pb = 100 * np.argsort(np.argsort(b)) / (N - 1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    ax.scatter(pa, pb, s=3, alpha=0.25, color="k", linewidths=0)
    ax.plot([0, 100], [0, 100], color="0.5", lw=0.8)
    lo = 100 * (1 - f)
    ax.add_patch(plt.Rectangle((lo, lo), 100 - lo, 100 - lo, fill=False, ec="tab:red", lw=1.2))
    ax.set_xlabel("softness percentile, first half of the frames")
    ax.set_ylabel("softness percentile, second half of the frames")
    ax.set_title(f"{N} atoms", fontsize=10)

    panels = [
        (axs[1], "spearman_null", res["spearman"], res["spearman_msd"], 0.0, res["p_spearman"], 1.0,
         "Spearman rank correlation of the two maps"),
        (axs[2], "top_overlap_null", res["top_overlap"], res["top_overlap_msd"], f, res["p_top_overlap"], 100.0,
         f"overlap of the top {100 * f:.0f}% softest atoms (%)")]
    for ax, key, obs, msd, unrelated, p, sc, xlabel in panels:
        null = np.asarray(res[key]) * sc
        ax.hist(null, bins=20, color="0.8", label=f"independent coordinates ({res['n_null']} runs)")
        ax.axvline(np.quantile(null, res["quantile"]), color="0.4", ls="--", lw=1, label=f"null {q}%")
        ax.axvline(unrelated * sc, color="0.3", ls=":", lw=1, label="two unrelated maps")
        ax.axvline(msd * sc, color="tab:blue", lw=1.5, label=f"MSD map, no PCA: {msd * sc:.2f}")
        ax.axvline(obs * sc, color="tab:red", lw=2, label=f"softness map: {obs * sc:.2f}  (p = {p:.2f})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("number of null runs")
        ax.legend(fontsize=8)
    axs[1].set_title("Spearman rank correlation", fontsize=10)
    axs[2].set_title("Overlap of the softest atoms", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

# =============================================================
# PCA temporal analysis, convergence and local (moving-window) PCA
#   analysis/soft_modes/pca_convergence.py, analysis/soft_modes/local_pca.py
# Colours: fixed order, one entity = one colour (blue, orange, aqua, yellow, magenta);
# modes ranked by index are drawn light -> dark in one hue; grey = chance / reference.
# =============================================================

_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
_GRID = dict(color="0.9", lw=0.6)


def _ranked_colors(n):
    return plt.cm.Blues(np.linspace(0.35, 0.95, max(n, 1)))


def _style(ax):
    ax.grid(True, **_GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, save_path, default_name):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, default_name)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_amplitude_acf(res, n_show=4, save_path=None):
    """
    Autocorrelation of the PCA mode amplitudes and the effective number of independent frames.
    `res` = analysis.soft_modes.pca_convergence.pca_temporal_analysis(...) (or amplitude_acf_analysis).

    Left    ACF rho(k) of a_n(t) for the n_show softest modes (k = lag in frames); grey band = white-noise
            band +-band_z/sqrt(T): lags inside it are not distinguishable from 0.
    Middle  ACF of a_n(t)^2, the series whose mean is the eigenvalue lambda_n. Solid = measured, dashed =
            rho_a(k)^2, what a stationary Gaussian amplitude would give.
    Right   T_eff = T / tau_int against mode index: window truncation of the ACF of a (grey) and of a^2
            (blue), and the lag-one AR(1) estimate T (1-phi)/(1+phi) of a^2 (orange). Open marker =
            flagged unreliable (oscillatory / negative ACF / window not closed / tau < 1 /
            error > max_rel_err ...; flags are printed by print_amplitude_acf). Dotted = T_eff pooled over all
            coordinates (pca_reliability), dashed = T. T_eff is capped at T.
    If `res` has a shifted-column null (pca_temporal_analysis(..., n_null > 0)) it is drawn in orange: the envelope
    of the null ACFs in the left and middle panels, the null range of T_eff in the right one. The null has no
    collective modes, so whatever lies inside it is produced by the selection of smooth temporal patterns from T
    frames alone (see the IN-SAMPLE CAVEAT in pca_convergence.py) and says nothing about the mode.
    """
    T, M = res["n_frames"], res["n_modes"]
    lags, band = res["lags"], res["band"]
    n_show = min(n_show, M)
    null = res.get("null")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

    for ax, key, title in ((axs[0], "amplitude", "ACF of the amplitude $a_n(t)$"),
                           (axs[1], "squared", "ACF of the squared amplitude $a_n(t)^2$")):
        ax.axhspan(-band, band, color="0.9", label="white-noise band")
        if null is not None:
            q = int(round(100 * null["quantile"]))
            ax.fill_between(lags, null[key]["acf_low"][:n_show].min(axis=0),
                            null[key]["acf_high"][:n_show].max(axis=0), color="#f2a65a", alpha=0.28, lw=0,
                            label=f"shifted-column null ({100 - q}-{q}%)")
        ax.axhline(0, color="0.5", lw=0.8)
        for n in range(n_show):
            ax.plot(lags, res[key]["acf"][n], color=_SERIES[n], lw=1.4, marker="o", ms=2.5,
                    label=f"mode {n}")
            if key == "squared":
                ax.plot(lags, res["gaussian_acf_squared"][n], color=_SERIES[n], lw=1, ls="--")
        ax.set_xlabel("lag k (frames)")
        ax.set_ylabel(r"$\rho(k)$")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-1.0, 1.05)
        ax.legend(fontsize=8, ncol=2)
        _style(ax)
    axs[1].text(0.98, 0.02, "dashed: $\\rho_a(k)^2$ (Gaussian prediction)", transform=axs[1].transAxes,
                ha="right", va="bottom", fontsize=8, color="0.3")

    ax = axs[2]
    n = np.arange(M)
    series = [("amplitude", "t_eff", "a(t), window rule", "0.45", "s"),
              ("squared", "t_eff", r"$a(t)^2$, window rule", _SERIES[0], "o"),
              ("squared", "ar1_t_eff", r"$a(t)^2$, AR(1)", _SERIES[1], "^")]
    for key, field, label, color, mk in series:
        y = np.asarray(res[key][field], dtype=float)
        ok = np.asarray(res[key]["reliable"] if field == "t_eff" else
                        [len(f) == 0 for f in res[key]["ar1_flags"]])
        ax.plot(n, y, color=color, lw=1, alpha=0.6)
        ax.scatter(n[ok], y[ok], color=color, marker=mk, s=34, zorder=3, label=label)
        ax.scatter(n[~ok], y[~ok], facecolors="none", edgecolors=color, marker=mk, s=34, zorder=3)
    if null is not None:
        q = int(round(100 * null["quantile"]))
        ax.fill_between(n, null["squared"]["t_eff_low"], null["squared"]["t_eff_high"], color="#f2a65a",
                        alpha=0.28, lw=0, label=f"null, $a^2$ window rule ({100 - q}-{q}%)")
        ax.fill_between(n, null["squared"]["ar1_t_eff_low"], null["squared"]["ar1_t_eff_high"], color="#c0392b",
                        alpha=0.18, lw=0, label=f"null, $a^2$ AR(1) ({100 - q}-{q}%)")
    ax.axhline(T, color="0.3", ls="--", lw=1, label=f"T = {T}")
    if "pooled_t_eff" in res:
        ax.axhline(res["pooled_t_eff"], color="0.3", ls=":", lw=1.2,
                   label=f"pooled over coordinates: {res['pooled_t_eff']:.0f}")
    ax.set_ylim(0, T * 1.08)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("mode index n  (0 = softest)")
    ax.set_ylabel(r"$T_{eff}$ (frames)")
    ax.set_title(f"Effective sample size  (open marker = flagged unreliable)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    _style(ax)
    p = res["params"]
    fig.suptitle(f"{T} frames, window rule '{p['window_method']}' (c = {p['window_c']:g}), lags <= {res['max_lag']}."
                 f"  Relative error of $\\tau$ is ~ $\\sqrt{{2(2W+1)/T}}$ (about 50% for W = 5): "
                 f"order of magnitude only.", fontsize=9, y=1.0)
    _save(fig, save_path, "pca_amplitude_acf.png")


def plot_pca_convergence(res, save_path=None):
    """
    Convergence of the PCA with the number of frames: the same PCA on the first L frames, L = res["lengths"],
    compared with the longest segment (the reference). `res` = pca_convergence.pca_convergence(...).
    Nested segments share frames and the last point is 1 by construction: this shows that adding frames
    stops changing the result, not that the result is right.

    (a) eigenvalue of the MATCHED mode / reference eigenvalue, one line per reference mode (light -> dark =
        softer -> ... mode index 0 darkest); black dashed = total variance / reference.
    (b) overlap |v_ref . v_L| of each reference mode with its best-matching mode of the L-frame PCA (matched
        by maximum overlap, not by index; light -> dark as in a). Orange dashed: mean overlap if modes are
        compared by index. Grey: chance (mean matched overlap on shifted-column null data, median to upper quantile).
    (c) leading M-dimensional eigenspace: overlap tr(P_ref P_L)/M = mean cos^2 of the principal angles, against the
        reference (blue) and against the next longer segment (orange); grey bands = null; dotted = random
        subspaces M/P.
    (d) overlap matrix |v_ref,i . v_j| of the SHORTEST segment (columns = its modes, rows = reference modes);
        dots mark the matched mode: off-diagonal dots are modes that changed rank.
    """
    L = np.asarray(res["lengths"])
    M, Ms = res["n_modes"], res["subspace_dim"]
    cols = _ranked_colors(M)[::-1]
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))

    ax = axs[0, 0]
    ratio = res["matched_eigenvalue"] / res["reference_eigenvalues"][None, :]
    for n in range(M):
        ax.plot(L, ratio[:, n], color=cols[n], lw=1.3, marker="o", ms=3, label=f"mode {n}" if n in (0, M - 1) else None)
    ax.plot(L, res["total_variance"] / res["total_variance"][-1], color="k", ls="--", lw=1, label="total variance")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_xlabel("frames L")
    ax.set_ylabel(r"$\lambda$(matched mode) / $\lambda_{ref}$")
    ax.set_title("(a) eigenvalue convergence", fontsize=10)
    ax.legend(fontsize=8)
    _style(ax)

    ax = axs[0, 1]
    for n in range(M):
        ax.plot(L, res["matched_overlap"][:, n], color=cols[n], lw=1.3, marker="o", ms=3)
    ax.plot(L, np.nanmean(res["index_overlap"], axis=1), color=_SERIES[1], ls="--", lw=1.3,
            label="mean, matched by INDEX")
    ax.plot([], [], color=cols[0], label="matched by max overlap (one line per mode)")
    if res["n_null"] > 0:
        nm = res["null_matched_overlap"]
        ax.fill_between(L, np.median(nm, axis=0), np.quantile(nm, res["quantile"], axis=0), color="0.8",
                        label=f"chance (mean over modes, null median to {int(round(100 * res['quantile']))}%)")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("frames L")
    ax.set_ylabel(r"overlap $|v_{ref}\cdot v_L|$")
    ax.set_title("(b) eigenvector overlap with the reference", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    _style(ax)

    ax = axs[1, 0]
    q = int(round(100 * res["quantile"]))
    ax.plot(L, res["subspace_overlap"], color=_SERIES[0], lw=1.6, marker="o", ms=4, label="vs reference")
    ax.plot(L[:-1], res["subspace_overlap_next"], color=_SERIES[1], lw=1.6, marker="s", ms=4, label="vs next longer segment")
    if res["n_null"] > 0:
        ax.fill_between(L, np.median(res["null_subspace_overlap"], axis=0),
                        np.quantile(res["null_subspace_overlap"], res["quantile"], axis=0), color=_SERIES[0],
                        alpha=0.15, label=f"null vs reference (median to {q}%)")
        ax.fill_between(L[:-1], np.median(res["null_subspace_overlap_next"], axis=0),
                        np.quantile(res["null_subspace_overlap_next"], res["quantile"], axis=0), color=_SERIES[1],
                        alpha=0.15, label=f"null vs next (median to {q}%)")
    ax.axhline(res["subspace_random_baseline"], color="0.3", ls=":", lw=1, label=f"random subspaces M/P")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("frames L")
    ax.set_ylabel(rf"subspace overlap  tr($P_{{ref}}P_L$)/{Ms}")
    ax.set_title(f"(c) leading {Ms}-dimensional eigenspace", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    _style(ax)

    ax = axs[1, 1]
    O = res["overlap_matrices"][0]
    im = ax.imshow(O, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="upper", interpolation="nearest")
    ax.scatter(res["matched_index"][0], np.arange(M), color="k", s=14, zorder=3, label="matched mode")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel(f"mode j of the {L[0]}-frame PCA")
    ax.set_ylabel("reference mode i")
    ax.set_title(f"(d) overlap matrix, {L[0]} frames vs reference", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, label=r"$|v_i^{ref}\cdot v_j|$")
    fig.suptitle(f"PCA on the first L frames, reference L = {L[-1]} ({res['n_coords']} coordinates). "
                 f"Nested segments share frames: not independent samples.", fontsize=10, y=1.0)
    _save(fig, save_path, "pca_convergence.png")


def _window_xlabel(res):
    """x axis label of the window list: centre atom index (neighbourhood windows) or mean atom index (index windows)."""
    if res["params"].get("window_mode") == "neighborhood":
        return "window centre (atom index; not a spatial order)"
    return "window position (mean atom index)"


def _heatmap(ax, data, position, cmap="Blues", vmin=None, vmax=None, label="", title="", n_rows=None,
             xlabel="window position (mean atom index)"):
    n_rows = data.shape[1] if n_rows is None else n_rows
    ext = [position[0], position[-1], -0.5, n_rows - 0.5]
    im = ax.imshow(np.asarray(data)[:, :n_rows].T, aspect="auto", origin="lower", cmap=cmap, vmin=vmin,
                   vmax=vmax, extent=ext, interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("local mode index n  (0 = softest)")
    ax.set_title(title, fontsize=10)
    ax.get_figure().colorbar(im, ax=ax, label=label)


def _band(ax, x, pct, color, label, ls="-"):
    """Median line and 5-95 % band of a (3, len(x)) percentile array."""
    ax.fill_between(x, pct[0], pct[2], color=color, alpha=0.18, lw=0)
    ax.plot(x, pct[1], color=color, lw=1.5, ls=ls, label=label)


def plot_local_pca_spectra(res, eigenvalue_unit="amu·Å²", save_path=None):
    """
    Local eigenvalue spectra along the window list (centre atom index for neighbourhood windows; the list is not a
    spatial order). `res` = local_pca.local_pca_analysis(...).

    Top left    log10 of the local eigenvalues lambda^w_n: x = window position (mean atom index), y = local mode
                index (0 = softest), colour = log10(lambda^w_n) in `eigenvalue_unit`.
    Top right   the same spectra divided by the window's total variance (the shape of the spectrum, independent
                of how much the atoms of the window vibrate): lambda^w_n / trace C_w.
    Bottom left Median and 5-95 % band over the windows of lambda^w_n (blue), the full-system eigenvalues lambda_n
                of the same rank (black): lambda^w_n <= lambda_n always (Cauchy interlacing) so this is a bound, not a result.
    Bottom right Normalised spectrum lambda^w_n / trace C_w: windows (blue), shifted-column null (grey, median to upper
                quantile of its window average: independent coordinates, same variance and autocorrelation),
                and the Marchenko-Pastur range of pure noise for K coordinates and T frames (dotted; dashed with
                T_eff if given). Local spectra inside the grey/dotted range are not distinguishable from sampling noise.
    """
    from analysis.soft_modes.local_pca import local_pca_band_statistics
    loc, st = res["local"], res["statistics"]
    pos = res["position"]
    K = loc["n_coords"]
    n = np.arange(K)
    bands = local_pca_band_statistics(res)
    lam = np.maximum(loc["eigenvalues"], 1e-300)
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    _heatmap(axs[0, 0], np.log10(lam), pos, label=rf"$\log_{{10}}\lambda^w_n$ ({eigenvalue_unit})",
             title=f"(a) local eigenvalues, {res['params']['window_size']}-atom "
                   f"{'neighbourhood ' if res['params'].get('window_mode') == 'neighborhood' else ''}windows, step {res['params']['step']}",
             xlabel=_window_xlabel(res))
    frac = loc["eigenvalues"] / np.maximum(loc["total_variance"], 1e-300)[:, None]
    _heatmap(axs[0, 1], frac, pos, label=r"$\lambda^w_n\,/\,\mathrm{tr}\,C_w$", title="(b) spectrum shape",
             xlabel=_window_xlabel(res))

    ax = axs[1, 0]
    _band(ax, n, bands["eigenvalues"], _SERIES[0], "local, median and 5-95% of windows")
    lg = res["global"]["eigenvalues"][:K]
    ax.plot(np.arange(len(lg)), lg, color="k", lw=1.3, label=r"full system $\lambda_n$ (upper bound)")
    ax.set_yscale("log")
    ax.set_xlabel("mode index n  (0 = softest)")
    ax.set_ylabel(rf"eigenvalue ({eigenvalue_unit})")
    ax.set_title("(c) local vs full-system eigenvalues", fontsize=10)
    ax.legend(fontsize=8)
    _style(ax)

    ax = axs[1, 1]
    _band(ax, n, bands["spectrum_fraction"], _SERIES[0], "windows, median and 5-95%")
    if res["null"] is not None:
        q = int(round(100 * res["null"]["quantile"]))
        ax.fill_between(n, bands["null_median"]["spectrum_fraction"], bands["null_high"]["spectrum_fraction"],
                        color="0.6", alpha=0.5, label=f"shifted-column null (median to {q}% of window average)")
    for edges, ls, lab in ((st["mp_edges"], ":", f"white-noise range, T = {st['n_frames']}"),
                           (st["mp_edges_teff"], "--",
                            None if st["mp_edges_teff"] is None else rf"white-noise range, $T_{{eff}}$ = {st['t_eff']:.0f}")):
        if edges is None:
            continue
        ax.axhline(edges[1] / K, color="0.3", ls=ls, lw=1.1, label=lab)       # upper edge
        if edges[0] > 0:                                                       # lower edge (0 when T_eff < K)
            ax.axhline(edges[0] / K, color="0.3", ls=ls, lw=1.1)
    ax.set_yscale("log")
    ax.set_xlabel("local mode index n  (0 = softest)")
    ax.set_ylabel(r"$\lambda^w_n\,/\,\mathrm{tr}\,C_w$")
    ax.set_title("(d) spectrum shape against sampling noise (lines: upper / lower edge)", fontsize=10)
    ax.legend(fontsize=8)
    _style(ax)
    fig.suptitle(f"{st['n_windows']} overlapping windows, {K} coordinates each, {st['n_frames']} frames "
                 f"(coordinates/frames = {st['coords_per_frame']:.2f}). Windows share frames and atoms: "
                 f"numerically exact, statistically not independent.", fontsize=10, y=1.0)
    _save(fig, save_path, "local_pca_spectra.png")


def plot_local_participation(res, save_path=None):
    """
    Participation ratios of the local modes (modes.participation_ratios convention, real-space vectors).
    `res` = local_pca.local_pca_analysis(...). PR = 1 / (n sum_i p_i^2) lies in [1/n, 1]; the effective number of
    atoms n_part = PR * n lies in [1, n] and is what is drawn, so windows of different size are comparable.

    Left   n_part of every local mode along the atom index (x: window position, y: local mode index).
    Right  median and 5-95 % band over windows of n_part against mode index (blue); shifted-column null
           (grey, median to upper quantile of the window average of PR times n); dashed = n (fully delocalised
           over the window). Mean n_part of the leading full-system modes is printed in the legend: n_part there is
           an effective number of atoms out of N, not comparable to a number out of n.
    """
    from analysis.soft_modes.local_pca import local_pca_band_statistics
    loc, glob = res["local"], res["global"]
    n_at = loc["n_atoms_window"]
    K = loc["n_coords"]
    bands = local_pca_band_statistics(res)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    _heatmap(axs[0], loc["n_participating"], res["position"], vmin=1, vmax=n_at,
             label=r"effective number of atoms  $n_{part}$ = PR$\cdot n$",
             title="(a) participation of the local modes", n_rows=K, xlabel=_window_xlabel(res))
    ax = axs[1]
    n = np.arange(K)
    _band(ax, n, bands["n_participating"], _SERIES[0], "windows, median and 5-95%")
    if res["null"] is not None:
        q = int(round(100 * res["null"]["quantile"]))
        ax.fill_between(n, bands["null_median"]["participation"] * n_at, bands["null_high"]["participation"] * n_at,
                        color="0.6", alpha=0.5, label=f"shifted-column null (median to {q}%)")
    ax.axhline(n_at, color="0.3", ls="--", lw=1, label=f"n = {n_at} (delocalised over the window)")
    ax.axhline(1, color="0.3", ls=":", lw=1, label="1 (single atom)")
    ax.set_xlabel("local mode index n  (0 = softest)")
    ax.set_ylabel(r"$n_{part}$ = PR$\cdot n$")
    ax.set_title(f"(b) by mode index  (full-system modes 0-{min(4, len(glob['n_participating']) - 1)}: "
                 f"n_part {glob['n_participating'][0]:.0f}-{glob['n_participating'][:5].max():.0f} of {glob['n_atoms']} atoms)",
                 fontsize=9)
    ax.set_ylim(0, n_at * 1.08)
    ax.legend(fontsize=8, loc="lower center")
    _style(ax)
    _save(fig, save_path, "local_pca_participation.png")


def plot_local_vs_global(res, save_path=None):
    """
    Local PCA against the full-system PCA of the same trajectory. `res` = local_pca.local_pca_analysis(...).
    Blue = windows (median and 5-95 % over windows), grey = shifted-column null (median to upper quantile of the
    window average), so the grey band is what chance gives for the same window size, frames and number of global
    modes compared with.

    (a) lambda^w_n / lambda_n: local over full-system eigenvalue of the same rank. <= 1 for any data (Cauchy
        interlacing); it measures how much of the global mode's variance is inside the window.
    (b) best normalised overlap max_m |v^w_n . g_m| / |g_m| of local mode n with the restrictions g_m of the
        leading global modes (matching by maximum overlap, not by index).
    (c) fraction of lambda^w_n contributed by the leading global modes, sum_{m<=Kg} lambda_m |v^w_n . g_m|^2 / lambda^w_n
        (exact decomposition; 1 would mean the window variance is entirely made of those modes).
    (d) leading M-dimensional eigenspaces: overlap S_M = tr(P^w_M P^glob_M)/M (solid) and variance capture ratio
        (dashed: the restricted global eigenspace explains this share of the best possible M-dimensional share
        of the window variance); dotted = random subspaces M/K.
    """
    from analysis.soft_modes.local_pca import local_pca_band_statistics
    comp, st = res["comparison"], res["statistics"]
    b = local_pca_band_statistics(res)
    nc = comp["overlap_matrix"].shape[1]
    Kg = comp["overlap_matrix"].shape[2]
    dims = comp["subspace_dims"]
    has_null = res["null"] is not None
    q = int(round(100 * res["null"]["quantile"])) if has_null else None
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    ax = axs[0, 0]
    Kc = comp["lambda_ratio"].shape[1]
    _band(ax, np.arange(Kc), b["lambda_ratio"], _SERIES[0], "windows")
    ax.axhline(1, color="0.3", ls="--", lw=1, label="interlacing bound")
    ax.set_yscale("log")
    ax.set_xlabel("mode index n  (0 = softest)")
    ax.set_ylabel(r"$\lambda^w_n\,/\,\lambda_n$")
    ax.set_title("(a) local / full-system eigenvalue of the same rank", fontsize=10)
    ax.legend(fontsize=8)
    _style(ax)

    for ax, key, title, ylabel in (
            (axs[0, 1], "overlap_best", f"(b) best overlap with one of the {Kg} leading global modes",
             "best normalised overlap"),
            (axs[1, 0], "explained_fraction", f"(c) share of $\\lambda^w_n$ made of the {Kg} leading global modes",
             "explained fraction")):
        x = np.arange(nc)
        _band(ax, x, b[key][:, :nc], _SERIES[0], "windows, median and 5-95%")
        if has_null:
            ax.fill_between(x, b["null_median"][key][:nc], b["null_high"][key][:nc], color="0.6", alpha=0.5,
                            label=f"shifted-column null (median to {q}%)")
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("local mode index n  (0 = softest)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=8, loc="lower left")
        _style(ax)

    ax = axs[1, 1]
    for key, color, ls, lab in (("subspace_overlap", _SERIES[0], "-", r"overlap $S_M$"),
                                ("capture_ratio", _SERIES[1], "--", "variance capture ratio")):
        ax.fill_between(dims, b[key][0], b[key][2], color=color, alpha=0.15, lw=0)
        ax.plot(dims, b[key][1], color=color, ls=ls, lw=1.6, marker="o", ms=4, label=lab + " (median, 5-95%)")
        if has_null:
            ax.plot(dims, b["null_high"][key], color=color, ls=ls, lw=1, alpha=0.5, marker="_", ms=8,
                    label=f"null {q}%")
    ax.plot(dims, comp["subspace_random"], color="0.3", ls=":", lw=1, label="random subspaces M/K")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("eigenspace dimension M")
    ax.set_ylabel("overlap / ratio")
    ax.set_title("(d) leading eigenspaces: local vs restricted global", fontsize=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="lower left")
    _style(ax)
    fig.suptitle(f"Local ({res['params']['window_size']} atoms) vs full-system PCA, {st['n_frames']} frames. "
                 f"Local modes are not global modes, and both use the same frames.", fontsize=10, y=1.0)
    _save(fig, save_path, "local_pca_vs_global.png")


def plot_local_consistency(res, save_path=None):
    """
    Are local modes found again in overlapping windows? `res` = local_pca.local_pca_analysis(...).
    Window w is compared with window w + j (index windows) or with its j-th nearest window centre (neighbourhood
    windows) on the atoms they share (x axis: share of the atoms that are common). Overlapping windows use the
    same frames, so this tests robustness against the position of the window boundary, not against sampling.

    (a) matched restricted overlap (maximum over the modes of the other window), mean over windows where the mode
        has >= min_shared_weight of its norm on the shared atoms; one line per local mode (light -> dark = 0, ...);
        grey = shifted-column null, mean over the same modes (median to upper quantile).
    (b) leading-M eigenspace overlap (restricted, re-orthonormalised principal angles), one colour per M; thin
        grey ticks = null upper quantile of the same M.
    (c) index windows: chain length = number of consecutive windows (shift 1) over which mode n is found again with
        overlap >= track_threshold. Neighbourhood windows have no linear order: 1 + the number of the nearest windows
        in which it is found again (not a chain). Median and 5-95 % over start windows (blue band), their mean (blue
        dashed) and the null's mean over windows (grey dashed): compare mean with mean, the counts are skewed.
    """
    cons, trk, st = res["consistency"], res["tracking"], res["statistics"]
    n_at = res["local"]["n_atoms_window"]
    shared = cons["shared_fraction"]
    has_null = res["null"] is not None
    q = int(round(100 * res["null"]["quantile"])) if has_null else None
    S = res["summary"]
    nm = S["consistency_matched"].shape[1]
    show = sorted({0, 1, 2, 4, min(9, nm - 1)} & set(range(nm)))
    cols = dict(zip(show, _ranked_colors(len(show))[::-1]))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axs[0]
    for n in show:
        ax.plot(shared, S["consistency_matched"][:, n], color=cols[n], lw=1.4, marker="o", ms=3, label=f"mode {n}")
    if has_null:
        nl = np.nanmean(res["null"]["consistency_matched"], axis=2)
        ax.fill_between(shared, np.median(nl, axis=0), np.quantile(nl, res["null"]["quantile"], axis=0), color="0.6",
                        alpha=0.5, label=f"null, mean over {nm} modes (median to {q}%)")
    ax.set_ylim(0, 1.03)
    ax.invert_xaxis()
    ax.set_xlabel("share of atoms common to both windows")
    ax.set_ylabel("matched restricted overlap")
    ax.set_title("(a) single modes, matched by maximum overlap", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    _style(ax)

    ax = axs[1]
    dims = cons["subspace_dims"]
    for i, M in enumerate(dims):
        ax.plot(shared, S["consistency_subspace"][:, i], color=_SERIES[i % len(_SERIES)], lw=1.4, marker="o", ms=3,
                label=f"M = {M}")
        if has_null:
            ax.plot(shared, np.quantile(res["null"]["consistency_subspace"][:, :, i], res["null"]["quantile"], axis=0),
                    color="0.45", ls="none", marker="_", ms=9)
    ax.plot([], [], color="0.45", ls="none", marker="_", ms=9, label="null upper quantile")
    ax.set_ylim(0, 1.03)
    ax.invert_xaxis()
    ax.set_xlabel("share of atoms common to both windows")
    ax.set_ylabel("eigenspace overlap")
    ax.set_title("(b) leading eigenspaces", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    _style(ax)

    ax = axs[2]
    L = trk["chain_length"]
    n = np.arange(L.shape[1])
    pct = np.percentile(L, [5, 50, 95], axis=0)
    _band(ax, n, pct, _SERIES[0], "windows, median and 5-95%")
    ax.plot(n, L.mean(axis=0), color=_SERIES[0], lw=1.4, ls="--", marker="o", ms=3, label="windows, mean")
    if has_null:
        ax.plot(n, np.median(res["null"]["chain_length"], axis=0), color="0.45", lw=1.4, ls="--", marker="s", ms=3,
                label="null, mean over windows (median of runs)")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("local mode index n  (0 = softest)")
    if trk.get("kind") == "neighbours":
        ax.set_ylabel(f"1 + nearest windows (of {trk['max_rank']}) in which the mode is found")
        ax.set_title(f"(c) found again in neighbouring windows (overlap >= {trk['threshold']:g})", fontsize=10)
    else:
        ax.set_ylabel("consecutive windows in which the mode is found again")
        ax.set_title(f"(c) chain length (overlap >= {trk['threshold']:g}, shift 1)", fontsize=10)
    ax.legend(fontsize=8)
    _style(ax)
    fig.suptitle(f"Consistency across overlapping windows of {n_at} atoms ({st['n_windows']} windows, same "
                 f"{st['n_frames']} frames: not independent)", fontsize=10, y=1.0)
    _save(fig, save_path, "local_pca_consistency.png")
