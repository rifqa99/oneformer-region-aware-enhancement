import numpy as np
import matplotlib.pyplot as plt

# base = "/content/drive/MyDrive/oneformer/plots"
ours = "/content/drive/MyDrive/oneformer/plots"

plt.figure()
# plt.plot(np.load(f"{base}_val_loss.npy"), label="Baseline")
plt.plot(np.load(f"val_loss.npy"), label="Region-aware")
plt.plot(np.load(f"train_loss.npy"), label="Region-aware")
plt.title("Validation Loss")
plt.legend()
plt.show()

plt.figure()
# plt.plot(np.load(f"{base}_val_psnr.npy"), label="Baseline")
plt.plot(np.load(f"{ours}_val_psnr.npy"), label="Region-aware")
plt.title("PSNR")
plt.legend()
plt.show()

plt.figure()
# plt.plot(np.load(f"{base}_val_ssim.npy"), label="Baseline")
plt.plot(np.load(f"{ours}_val_ssim.npy"), label="Region-aware")
plt.title("SSIM")
plt.legend()
plt.show()
