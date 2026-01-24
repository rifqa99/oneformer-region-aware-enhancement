import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.datasets.paired_dataset_rgb import PairedImageDatasetRGB
from src.utils.metrics import psnr, ssim

# ---------------- config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", DEVICE)

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
IMG_SIZE = (512, 512)

CKPT_DIR = "/content/drive/MyDrive/checkpoints_baseline"
os.makedirs(CKPT_DIR, exist_ok=True)

os.makedirs("/content/drive/MyDrive/oneformer/plots_LOL_baseline", exist_ok=True)


# ---------------- data ----------------
train_ds = PairedImageDatasetRGB(
    "data/train/input",
    "data/train/target",
    size=IMG_SIZE,
)

val_ds = PairedImageDatasetRGB(
    "data/val/input",
    "data/val/target",
    size=IMG_SIZE,
)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=2)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

# ---------------- model ----------------
model = UNet(in_channels=3, out_channels=3).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
l1_loss = nn.L1Loss()

# ---------------- for the graph -------------
train_losses, val_losses = [], []
val_psnrs, val_ssims = [], []

# ---------------- training ----------------
for epoch in range(1, EPOCHS + 1):

    # ---- train ----
    model.train()
    train_loss = 0.0

    for x, y in train_dl:
        x = x.to(DEVICE)      # (B, 6, H, W)
        y = y.to(DEVICE)      # (B, 3, H, W)

        optimizer.zero_grad()
        out = model(x)

        loss_l1 = l1_loss(out, y)
        loss = loss_l1
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dl)

    # ---- validation ----
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
            loss = loss_l1

            val_loss += loss.item()
            val_psnr += psnr(out, y).item()
            val_ssim += ssim(out, y).item()

    val_loss /= len(val_dl)
    val_psnr /= len(val_dl)
    val_ssim /= len(val_dl)

    train_losses.append(train_loss)
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
        os.path.join(CKPT_DIR, f"unet_baseline_epoch_{epoch}.pt")
    )
np.save("/content/drive/MyDrive/oneformer/plots/trainloss_baseline.npy", train_losses)
np.save("/content/drive/MyDrive/oneformer/plots/val_loss_baseline.npy", val_losses)
np.save("/content/drive/MyDrive/oneformer/plots/val_psnr_baseline.npy", val_psnrs)
np.save("/content/drive/MyDrive/oneformer/plots/val_ssim_baseline.npy", val_ssims)
