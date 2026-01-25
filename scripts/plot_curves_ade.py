import matplotlib.pyplot as plt
import numpy as np

ade_plots = "/content/drive/MyDrive/oneformer_ade/plots"

files = [
    "val_loss_baseline.npy",
    "val_loss.npy",
    "val_psnr_baseline.npy",
    "val_psnr.npy",
    "val_ssim_baseline.npy",
    "val_ssim.npy"
]


def safe_plot(path, fname_base, fname_ours, title, ylabel, save_as):
    base = np.load(f"{path}/{fname_base}")
    ours = np.load(f"{path}/{fname_ours}")

    epochs = np.arange(1, len(base) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, base, label="Baseline", marker="o")
    plt.plot(epochs, ours, label="Region-aware", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.show()


safe_plot(ade_plots, "val_loss_baseline.npy", "val_loss.npy",
          "Validation Loss (ADE20K)", "Loss", "ade_val_loss.png")

safe_plot(ade_plots, "val_psnr_baseline.npy", "val_psnr.npy",
          "Validation PSNR (ADE20K)", "PSNR (dB)", "ade_val_psnr.png")

safe_plot(ade_plots, "val_ssim_baseline.npy", "val_ssim.npy",
          "Validation SSIM (ADE20K)", "SSIM", "ade_val_ssim.png")


def plot_train_val(path, train_file, val_file, title, save_as):
    train = np.load(f"{path}/{train_file}")
    val = np.load(f"{path}/{val_file}")

    epochs = np.arange(1, len(train) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train, label="Train", marker="o")
    plt.plot(epochs, val, label="Validation", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.show()


# Baseline
plot_train_val(
    ade_plots,
    "train_loss_baseline.npy",
    "val_loss_baseline.npy",
    "Train vs Validation Loss (Baseline)",
    "ade_train_val_loss_baseline.png"
)
# Ours
plot_train_val(
    ade_plots,
    "train_loss.npy",
    "val_loss.npy",
    "Train vs Validation Loss (Region-aware)",
    "ade_train_val_loss_ours.png"
)
