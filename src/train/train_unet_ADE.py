import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.datasets.ade_enhance_dataset import ADEEnhancementDataset
from src.utils.metrics import psnr, ssim
from src.utils.perceptual_loss import VGGPerceptualLoss

# ---------------- config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", DEVICE)
EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4
IMG_SIZE = (512, 512)

CKPT_DIR = "/content/drive/MyDrive/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

os.makedirs("/content/drive/MyDrive/oneformer_ade/plots", exist_ok=True)


LAMBDA_PERC = 0.1

# ---------------- data ----------------
train_ds = ADEEnhancementDataset(
    root="/content/Dataset",
    split="train"
)

val_ds = ADEEnhancementDataset(
    root="/content/Dataset",
    split="validation"
)


train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=2)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

# ---------------- model ----------------
model = UNet(in_channels=6, out_channels=3).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
l1_loss = nn.L1Loss()
perc_loss = VGGPerceptualLoss(device=DEVICE)

# ---------------- for the graph -------------
train_losses, val_losses = [], []
val_psnrs, val_ssims = [], []

# ---------------- training ----------------
for epoch in range(1, EPOCHS + 1):

    # ---- train ----
    model.train()
    train_loss = 0.0

    for inp, seg, y in train_dl:
        seg = seg.unsqueeze(1).float() / 150.0   # (B,1,H,W) normalized
        x = torch.cat([inp, seg.repeat(1, 3, 1, 1)], dim=1)  # (B,6,H,W)

        x = x.to(DEVICE)      # (B, 6, H, W)
        y = y.to(DEVICE)      # (B, 3, H, W)

        optimizer.zero_grad()
        out = model(x)

        loss_l1 = l1_loss(out, y)
        loss_perc = perc_loss(out, y)
        loss = loss_l1 + LAMBDA_PERC * loss_perc

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
        for inp, seg, y in val_dl:
            seg = seg.unsqueeze(1).float() / 150.0
            x = torch.cat([inp, seg.repeat(1, 3, 1, 1)], dim=1)

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x)

            loss_l1 = l1_loss(out, y)
            loss_perc = perc_loss(out, y).detach()
            loss = loss_l1 + LAMBDA_PERC * loss_perc

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
        os.path.join(CKPT_DIR, f"unet_epoch_{epoch}.pt")
    )
np.save("/content/drive/MyDrive/oneformer_ade/plots/train_loss.npy", train_losses)
np.save("/content/drive/MyDrive/oneformer_ade/plots/val_loss.npy", val_losses)
np.save("/content/drive/MyDrive/oneformer_ade/plots/val_psnr.npy", val_psnrs)
np.save("/content/drive/MyDrive/oneformer_ade/plots/val_ssim.npy", val_ssims)
