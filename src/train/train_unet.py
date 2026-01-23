import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.datasets.paired_dataset import PairedImageDataset
from src.utils.metrics import psnr, ssim

# ---------------- config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4
IMG_SIZE = (512, 512)

CKPT_DIR = "/content/drive/MyDrive/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)


# ---------------- data ----------------
train_ds = PairedImageDataset(
    input_dir="data/train/input",
    target_dir="data/train/target",
    mask_dir="/content/drive/MyDrive/oneformer_masks/train",
    size=IMG_SIZE,
)

val_ds = PairedImageDataset(
    input_dir="data/val/input",
    target_dir="data/val/target",
    mask_dir="/content/drive/MyDrive/oneformer_masks/val",
    size=IMG_SIZE,
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
loss_fn = nn.L1Loss()

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
        loss = loss_fn(out, y)
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
            loss = loss_fn(out, y)

            val_loss += loss.item()
            val_psnr += psnr(out, y).item()
            val_ssim += ssim(out, y).item()

    val_loss /= len(val_dl)
    val_psnr /= len(val_dl)
    val_ssim /= len(val_dl)

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
