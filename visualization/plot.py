# =============================================================
# visualization/plot.py — static plot functions
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import inspect
from utils import get_output_dir
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
                     save_path="diffraction/diffraction.png"):
    
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
    save_path="diffraction/diffraction.png",
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
    save_path="diffraction/virtual_detector_heatmap.png",
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

    save_path = f"diffraction/virtual_{mask_name}.png"

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
                           save_path="dynamics/pair_correlation.png"):
    """
    Plot g(r) with the detected first peak and neighbor cutoff marked.
    """
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

def plot_d2min_histogram(d2min, save_path="dynamics/d2min_histogram.png"):
    """
    Histogram of D^2_min values across all atoms (log-y, since the
    distribution is typically heavy-tailed: most atoms sit near the
    bulk value, with a minority showing much larger non-affine motion).
    """
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

def plot_d2min_map(atoms_ref, d2min, save_path="dynamics/d2min_map.png"):
    """
    2D (x-y) scatter of atoms, colored by D^2_min — a real-space map
    of where non-affine (flow-unit-like) activity is concentrated,
    integrated over the whole z-thickness of the cell.
    """
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

def plot_d2min_map_3d(atoms_ref, d2min, save_path="dynamics/d2min_map_3d.png",
                       elev=20, azim=45):
    """
    3D (x-y-z) scatter of atoms, colored by D^2_min. Unlike
    plot_d2min_map (which integrates over the full z-thickness and
    can hide real 3D clustering behind overlapping points), this
    shows each atom at its actual depth — rotate/re-render with
    different elev/azim to check for localization the 2D projection
    might be washing out.
    """
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

def plot_d2min_hole_distance(stats, save_path=None):
    """
    Mean D^2_min per distance-to-nearest-hole group (near/mid/far,
    recomputed each frame), over the trajectory. If the "near" line
    sits consistently above "far", d2min is concentrated close to
    wherever holes currently are.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    frames = stats["frame_indices"]
    bin_means = stats["bin_means"]
    n_bins = bin_means.shape[1]
    labels = ["Near", "Mid", "Far"] if n_bins == 3 else [f"Bin {b}" for b in range(n_bins)]
    colors = cm.viridis(np.linspace(0, 0.85, n_bins))

    for b in range(n_bins):
        ax.plot(frames, bin_means[:, b], label=labels[b], color=colors[b])

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean D²_min (Å²)")
    ax.set_title("D²_min by distance to nearest hole peak")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")

    plt.show()
    return fig, ax