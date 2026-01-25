import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def plot_baseline_vs_ours(
    path,
    base_file,
    ours_file,
    title,
    ylabel,
    save_name
):
    base = np.load(os.path.join(path, base_file))
    ours = np.load(os.path.join(path, ours_file))

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
    plt.savefig(save_name, dpi=300)
    plt.show()


def plot_train_vs_val(
    path,
    train_file,
    val_file,
    title,
    ylabel,
    save_name
):
    train = np.load(os.path.join(path, train_file))
    val = np.load(os.path.join(path, val_file))

    epochs = np.arange(1, len(train) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train, label="Train", marker="o")
    plt.plot(epochs, val, label="Validation", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()


def main(args):
    plot_dir = args.plot_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Validation metrics
    plot_baseline_vs_ours(
        plot_dir,
        "val_loss_baseline.npy",
        "val_loss.npy",
        f"Validation Loss ({args.dataset})",
        "Loss",
        os.path.join(out_dir, "val_loss.png")
    )

    plot_baseline_vs_ours(
        plot_dir,
        "val_psnr_baseline.npy",
        "val_psnr.npy",
        f"Validation PSNR ({args.dataset})",
        "PSNR (dB)",
        os.path.join(out_dir, "val_psnr.png")
    )

    plot_baseline_vs_ours(
        plot_dir,
        "val_ssim_baseline.npy",
        "val_ssim.npy",
        f"Validation SSIM ({args.dataset})",
        "SSIM",
        os.path.join(out_dir, "val_ssim.png")
    )

    # Train vs Val (optional)
    if args.plot_train_val:
        plot_train_vs_val(
            plot_dir,
            "train_loss_baseline.npy",
            "val_loss_baseline.npy",
            f"Train vs Validation Loss (Baseline – {args.dataset})",
            "Loss",
            os.path.join(out_dir, "train_val_baseline.png")
        )

        plot_train_vs_val(
            plot_dir,
            "train_loss.npy",
            "val_loss.npy",
            f"Train vs Validation Loss (Region-aware – {args.dataset})",
            "Loss",
            os.path.join(out_dir, "train_val_ours.png")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_dir", type=str, required=True,
                        help="Directory containing .npy metric files")
    parser.add_argument("--out_dir", type=str, default="figures",
                        help="Where to save plots")
    parser.add_argument("--dataset", type=str, default="Dataset",
                        help="Dataset name for titles (LOL, ADE20K, etc.)")
    parser.add_argument("--plot_train_val", action="store_true",
                        help="Plot train vs validation curves")

    args = parser.parse_args()
    main(args)
