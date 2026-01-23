import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.datasets.paired_dataset import PairedImageDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# paths
CKPT = "outputs/checkpoints/unet_epoch_20.pt"
OUT_DIR = "outputs/visuals"
os.makedirs(OUT_DIR, exist_ok=True)

# dataset
val_ds = PairedImageDataset(
    "data/val/input",
    "data/val/target",
    "/content/drive/MyDrive/oneformer_masks/val",
    size=(512, 512),
)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

# model
model = UNet(in_channels=6, out_channels=3).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# run
with torch.no_grad():
    for i, (x, y) in enumerate(val_dl):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        out = model(x)

        grid = torch.cat([x[:, :3], out, y], dim=0)
        vutils.save_image(
            grid,
            f"{OUT_DIR}/sample_{i}.png",
            nrow=3,
            normalize=True
        )

        if i == 4:  # save 5 samples only
            break

print("Saved visual results to", OUT_DIR)
