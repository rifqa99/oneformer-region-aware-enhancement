import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.datasets.ade_enhance_dataset import ADEEnhancementDataset
from src.utils.metrics import psnr, ssim

# ===================== CONFIG =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

START_EPOCH = 1
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-4

CKPT_DIR = "/content/drive/MyDrive/checkpoints_ade_baseline"
PLOT_DIR = "/content/drive/MyDrive/oneformer_ade_baseline/plots"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ROOT = "/content/Datasets"

# ===================== DATA =====================
train_ds = ADEEnhancementDataset(root=ROOT, split="train")
val_ds = ADEEnhancementDataset(root=ROOT, split="val")

train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_dl = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# sanity check
inp, seg, tgt = next(iter(train_dl))
print(inp.shape, tgt.shape)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

# ===================== MODEL =====================
# Baseline: NO segmentation → 3 input channels
model = UNet(in_channels=3, out_channels=3).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
l1_loss = nn.L1Loss()

# ===================== LOGGING =====================
train_losses, val_losses = [], []
val_psnrs, val_ssims = [], []

# ===================== TRAINING =====================
for epoch in range(START_EPOCH, EPOCHS + 1):

    # -------- TRAIN --------
    model.train()
    train_loss = 0.0

    for inp, _, tgt in train_dl:   # ignore seg
        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)

        optimizer.zero_grad()
        out = model(inp)

        loss = l1_loss(out, tgt)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dl)
    train_losses.append(train_loss)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for inp, _, tgt in val_dl:
            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)

            out = model(inp)

            loss = l1_loss(out, tgt)

            val_loss += loss.item()
            val_psnr += psnr(out, tgt).item()
            val_ssim += ssim(out, tgt).item()

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
np.save(os.path.join(PLOT_DIR, "val_ssim.npy"), val_ssims)
