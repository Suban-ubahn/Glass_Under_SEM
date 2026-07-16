# =============================================================
# visualization/plot.py — static plot functions
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_map(image, title, cmap="inferno", colorbar_label=""):
    plt.figure(figsize=(8, 6))
    plt.imshow(image.T, origin="lower", cmap=cmap)
    plt.colorbar().set_label(colorbar_label, fontsize=9)
    plt.title(title)
    plt.xlabel("X voxel")
    plt.ylabel("Y voxel")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=300)
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
    plt.savefig("cross_correlation_radial.png", dpi=300)
    print("Saved: cross_correlation_radial.png")


def plot_diffraction(diffraction_patterns, blocked=True,
                     save_path="diffraction/diffraction.png"):
    
    if diffraction_patterns.array.ndim == 4:
        mean_dp = diffraction_patterns.mean(axis=(0, 1))
        mean_dp.compute()
    else:
        mean_dp = diffraction_patterns

    if blocked:
        result = mean_dp.block_direct()
        result.compute()
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

    if blocked:
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_blocked{ext}"

    plt.savefig(save_path, dpi=300)
    
    print(f"Saved: {save_path}")