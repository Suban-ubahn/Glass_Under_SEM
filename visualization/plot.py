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
    P(a_n) of the softest modes, one panel per mode.
    `diag` = analysis.tls_detection.mode_amplitude_diagnostics(...).

    a_n(t) is the amplitude of mode n in frame t: ONE number per frame that
    says how far the whole configuration is displaced along that mode.
    It is shown in units of its own standard deviation sigma_n (so the
    axes are dimensionless and all modes can be compared; sigma_n itself,
    in `amplitude_unit`, is written in each title).

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
    independent frames. A mode the detector flags as TLS gets a red title.

    amplitude_unit : unit of sigma_n for the title: "√amu·Å" if the modes were
        mass-weighted (the default pipeline), "Å" otherwise.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "mode_amplitude_distributions.png")
    a = np.asarray(diag["a_std"])                               # (M, T)
    M, T = a.shape
    if bins is None:
        bins = int(np.clip(np.sqrt(T), 8, 30))
    nrows = int(np.ceil(M / ncols))
    rows_per_mode = 2 if show_time_series else 1
    fig_h = (3.9 if show_time_series else 3.0) * nrows + 0.6
    fig = plt.figure(figsize=(3.9 * ncols, fig_h))
    gs = fig.add_gridspec(nrows * rows_per_mode, ncols,
                          height_ratios=([3, 1.2] if show_time_series else [1]) * nrows,
                          hspace=1.0 if show_time_series else 0.7, wspace=0.4,
                          top=1 - 1.1 / fig_h, bottom=0.6 / fig_h)

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
        pr = "" if diag["participation"] is None else f"  PR={diag['participation'][m]:.2f}"
        ax.set_title(f"mode {diag['mode_idx'][m]}  λ={diag['eigenvalues'][m]:.2g}{pr}  "
                     f"σ={diag['sigma'][m]:.2g} {amplitude_unit}\n"
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

    fig.suptitle("P(a_n) of the softest modes   (red = single Gaussian, "
                 "green dashed = detector's 2-Gaussian fit)", fontsize=11, y=1 - 0.05 / fig_h)
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
