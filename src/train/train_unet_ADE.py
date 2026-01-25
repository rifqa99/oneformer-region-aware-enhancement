import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.paired_dataset_ade import ADEPairedMaskDataset
from src.models.unet import UNet
from src.utils.metrics import psnr, ssim
from src.utils.perceptual_loss import VGGPerceptualLoss

# ===================== CONFIG =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
LAMBDA_PERC = 0.1

CKPT_DIR = "/content/drive/MyDrive/checkpoints_ade"
PLOT_DIR = "/content/drive/MyDrive/oneformer_ade/plots"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ===================== DATA =====================
train_ds = ADEPairedMaskDataset(
    input_dir="/content/Datasets/train/input",
    target_dir="/content/Datasets/train/target",
    mask_dir="/content/drive/MyDrive/ade20k_oneformer_masks/train",
    size=(512, 512),
)

val_ds = ADEPairedMaskDataset(
    input_dir="/content/Datasets/val/input",
    target_dir="/content/Datasets/val/target",
    mask_dir="/content/drive/MyDrive/ade20k_oneformer_masks/val",
    size=(512, 512),
)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=2)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

# ===================== MODEL =====================
model = UNet(in_channels=6, out_channels=3).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
l1_loss = nn.L1Loss()
perc_loss = VGGPerceptualLoss(device=DEVICE)

# ===================== LOGGING =====================
train_losses, val_losses = [], []
val_psnrs, val_ssims = [], []

# ===================== TRAINING =====================
for epoch in range(1, EPOCHS + 1):

    # -------- TRAIN --------
    model.train()
    train_loss = 0.0

    for x, y in train_dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)

        loss_l1 = l1_loss(out, y)
        loss_perc = perc_loss(out, y)
        loss = loss_l1 + LAMBDA_PERC * loss_perc

        loss.backward()
        optimizer.step()

        train_loss += loss.item()   # ✅ FIX

    train_loss /= len(train_dl)
    train_losses.append(train_loss)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x)

            loss_l1 = l1_loss(out, y)
            loss_perc = perc_loss(out, y)
            loss = loss_l1 + LAMBDA_PERC * loss_perc

            val_loss += loss.item()
            val_psnr += psnr(out, y).item()
            val_ssim += ssim(out, y).item()

    val_loss /= len(val_dl)
    val_psnr /= len(val_dl)
    val_ssim /= len(val_dl)

    val_losses.append(val_loss)
    val_psnrs.append(val_psnr)
    val_ssims.append(val_ssim)

    print(
        f"Epoch {epoch:02d} | "
        f"train={train_loss:.4f} | "
        f"val={val_loss:.4f} | "
        f"PSNR={val_psnr:.2f} | "
        f"SSIM={val_ssim:.4f}"
    )

    torch.save(
        model.state_dict(),
        os.path.join(CKPT_DIR, f"unet_epoch_{epoch}.pt")
    )

# ===================== SAVE CURVES =====================
np.save(os.path.join(PLOT_DIR, "train_loss.npy"), train_losses)
np.save(os.path.join(PLOT_DIR, "val_loss.npy"), val_losses)
np.save(os.path.join(PLOT_DIR, "val_psnr.npy"), val_psnrs)
np.save(os.path.join(
