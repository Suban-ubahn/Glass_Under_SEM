# import zarr
# import numpy as np
# import matplotlib.pyplot as plt
# import abtem

# dp = abtem.from_zarr("diffraction/4d_stem.zarr")
# mean_dp = np.array(dp.array.mean(axis=(0, 1)))

# print("min:", mean_dp.min())
# print("max:", mean_dp.max())
# print("95th percentile:", np.percentile(mean_dp, 95))
# print("99th percentile:", np.percentile(mean_dp, 99))
# print("shape:", mean_dp.shape)

# import zarr
# import numpy as np
# import matplotlib.pyplot as plt
# import abtem

# dp = abtem.from_zarr("diffraction/4d_stem.zarr")
# mean_dp = np.array(dp.array.mean(axis=(0, 1)))

# # mask the bright field disk by zeroing the central region
# cx, cy = mean_dp.shape[0] // 2, mean_dp.shape[1] // 2
# Y, X = np.ogrid[:mean_dp.shape[0], :mean_dp.shape[1]]
# disk_radius = 80  # adjust this — it's the BF disk radius in pixels
# mask = (X - cx)**2 + (Y - cy)**2 < disk_radius**2
# masked = mean_dp.copy()
# masked[mask] = 0

# print("outside disk max:", masked.max())
# print("outside disk 99th percentile:", np.percentile(masked[~mask], 99))

# plt.figure(figsize=(6, 6))
# plt.imshow(np.log1p(masked / masked.max()), cmap="inferno", origin="lower")
# plt.colorbar().set_label("log(1 + normalised intensity)", fontsize=9)
# plt.title("Diffuse Scattering (BF disk masked)")
# plt.savefig("diffraction/diffuse_scattering.png", dpi=300)

# import zarr
# import numpy as np
# import matplotlib.pyplot as plt
# import abtem

# dp = abtem.from_zarr("diffraction/4d_stem.zarr")
# mean_dp = np.array(dp.array.mean(axis=(0, 1)))

# # mask the bright field disk by zeroing the central region
# cx, cy = mean_dp.shape[0] // 2, mean_dp.shape[1] // 2
# Y, X = np.ogrid[:mean_dp.shape[0], :mean_dp.shape[1]]
# disk_radius = 80  # adjust this — it's the BF disk radius in pixels
# mask = (X - cx)**2 + (Y - cy)**2 < disk_radius**2
# masked = mean_dp.copy()
# masked[mask] = 0

# print("outside disk max:", masked.max())
# print("outside disk 99th percentile:", np.percentile(masked[~mask], 99))

# plt.figure(figsize=(6, 6))
# plt.imshow(np.log1p(masked / masked.max()), cmap="inferno", origin="lower")
# plt.colorbar().set_label("log(1 + normalised intensity)", fontsize=9)
# plt.title("Diffuse Scattering (BF disk masked)")
# plt.savefig("diffraction/diffuse_scattering.png", dpi=300)

# import numpy as np
# import matplotlib.pyplot as plt
# import abtem

# dp = abtem.from_zarr("diffraction/4d_stem.zarr")

# # average over scan positions, then block the direct beam
# mean_dp = dp.mean(axis=(0, 1))
# mean_dp.compute()
# blocked = mean_dp.block_direct()
# blocked.compute()

# arr = np.array(blocked.array)
# print("min:", arr.min())
# print("max:", arr.max())

# plt.figure(figsize=(6, 6))
# plt.imshow(np.log1p(arr / arr.max()), cmap="inferno", origin="lower")
# plt.colorbar().set_label("log(1 + normalised intensity)", fontsize=9)
# plt.title("Mean Diffraction Pattern (BF blocked)")
# plt.savefig("diffraction/mean_diffraction_blocked.png", dpi=300)
# print("Saved")

from data_io.trajectory import load_trajectory
atoms = load_trajectory()[0]
print(atoms.cell.lengths())