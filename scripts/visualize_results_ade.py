import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.datasets.ade_enhance_dataset import ADEEnhancementDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= paths =================
CKPT = "/content/drive/MyDrive/checkpoints_ade/unet_epoch_18.pt"  # BEST epoch
OUT_DIR = "/content/drive/MyDrive/oneformer_ade/outputs/visuals"

os.makedirs(OUT_DIR, exist_ok=True)

# ================= dataset =================
val_ds = ADEEnhancementDataset(
    root="/content/Datasets",
    split="val"
)

val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)

# ================= model =================
model = UNet(in_channels=6, out_channels=3).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# ================= run =================
with torch.no_grad():
    for i, (inp, seg, tgt) in enumerate(val_dl):
        inp = inp.to(DEVICE)          # (1,3,H,W)
        tgt = tgt.to(DEVICE)          # (1,3,H,W)

        # seg → (1,3,H,W)
        seg = seg.unsqueeze(1).float() / 150.0
        seg = seg.repeat(1, 3, 1, 1).to(DEVICE)

        x = torch.cat([inp, seg], dim=1)  # (1,6,H,W)
        out = model(x)

        # Input | Output | GT
        grid = torch.cat([inp, out, tgt], dim=0)

        vutils.save_image(
            grid,
            f"{OUT_DIR}/sample_{i}.png",
            nrow=3,
            normalize=True
        )

        if i == 4:  # save 5 samples only
            break

print("Saved visual results to:", OUT_DIR)
