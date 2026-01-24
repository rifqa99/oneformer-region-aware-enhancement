import numpy as np
import matplotlib.pyplot as plt

base = "/content/drive/MyDrive/oneformer/plots/base"
ours = "/content/drive/MyDrive/oneformer/plots/ours"

plt.figure()
plt.plot(np.load(f"{base}/val_loss_baseline.npy"), label="Baseline")
plt.plot(np.load(f"{ours}/val_loss.npy"), label="Region-aware")
plt.title("Validation Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(np.load(f"{base}/val_psnr_baseline.npy"), label="Baseline")
plt.plot(np.load(f"{ours}/val_psnr.npy"), label="Region-aware")
plt.title("PSNR")
plt.legend()
plt.show()

plt.figure()
plt.plot(np.load(f"{base}/val_ssim_baseline.npy"), label="Baseline")
plt.plot(np.load(f"{ours}/val_ssim.npy"), label="Region-aware")
plt.title("SSIM")
plt.legend()
plt.show()
