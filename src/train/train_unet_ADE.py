import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.datasets.ade_enhance_dataset import ADEEnhancementDataset
from src.utils.metrics import psnr, ssim
from src.utils.perceptual_loss import VGGPerceptualLoss

# ===================== CONFIG =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4

CKPT_DIR = "/content/drive/MyDrive/checkpoints"
PLOT_DIR = "/content/drive/MyDrive/oneformer_ade/plots"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

LAMBDA_PERC = 0.1

# ===================== DATA =====================
ROOT = "/content/Dataset"

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

    for inp, seg, tgt in train_dl:
        inp = inp.to(DEVICE)
        tgt = tgt.to(DEVICE)

        seg = seg.unsqueeze(1).float() / 150.0          # (B,1,H,W)
        seg = seg.repeat(1, 3, 1, 1).to(DEVICE)         # (B,3,H,W)

        x = torch.cat([inp, seg], dim=1)                # (B,6,H,W)

        optimizer.zero_grad()
        out = model(x)

        loss_l1 = l1_loss(out, tgt)
        loss_perc = perc_loss(out, tgt)
        loss = loss_l1 + LAMBDA_PERC * loss_perc

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dl)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for inp, seg, tgt in val_dl:
            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)

            seg = seg.unsqueeze(1).float() / 150.0
            seg = seg.repeat(1, 3, 1, 1).to(DEVICE)

            x = torch.cat([inp, seg], dim=1)

            out = model(x)

            loss_l1 = l1_loss(out, tgt)
            loss_perc = perc_loss(out, tgt)
            loss = loss_l1 + LAMBDA_PERC * loss_perc

            val_loss += loss.item()
            val_psnr += psnr(out, tgt).item()
            val_ssim += ssim(out, tgt).item()

    val_loss /= len(val_dl)
    val_psnr /= len(val_dl)
    val_ssim /= len(val_dl)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_psnrs.append(val_psnr)
    val_ss_
