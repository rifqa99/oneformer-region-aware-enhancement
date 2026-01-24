import numpy as np
import matplotlib.pyplot as plt

# ADE plots
# base = "/content/drive/MyDrive/oneformer_ade/plots/base"
# ours = "/content/drive/MyDrive/oneformer_ade/plots/ours"

lol_plots = "/content/drive/MyDrive/oneformer/plots"

plt.figure()
plt.plot(np.load(f"{lol_plots}/val_loss_baseline.npy"), label="Baseline")
plt.plot(np.load(f"{lol_plots}/val_loss.npy"), label="Region-aware")
plt.title("Validation Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(np.load(f"{lol_plots}/val_psnr_baseline.npy"), label="Baseline")
plt.plot(np.load(f"{lol_plots}/val_psnr.npy"), label="Region-aware")
plt.title("PSNR")
plt.legend()
plt.show()

plt.figure()
plt.plot(np.load(f"{lol_plots}/val_ssim_baseline.npy"), label="Baseline")
plt.plot(np.load(f"{lol_plots}/val_ssim.npy"), label="Region-aware")
plt.title("SSIM")
plt.legend()
plt.show()
