import matplotlib.pyplot as plt
from src.datasets.ade_enhance_dataset import ADEEnhancementDataset
import random
import torch

from src.models.unet import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", DEVICE)
# paths
CKPT = "/content/drive/MyDrive/checkpoints_ade/unet_epoch_18.pt"

model = UNet(in_channels=6, out_channels=3).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

print("Loaded best model (epoch 18)")


val_ds = ADEEnhancementDataset(
    root="/content/Datasets",
    split="val"
)

indices = random.sample(range(len(val_ds)), 5)
samples = [val_ds[i] for i in indices]


def show_triplet(inp, out, tgt):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(inp.permute(1, 2, 0).cpu())
    axs[0].set_title("Input (Degraded)")
    axs[0].axis("off")

    axs[1].imshow(out.permute(1, 2, 0).cpu())
    axs[1].set_title("Output (Ours)")
    axs[1].axis("off")

    axs[2].imshow(tgt.permute(1, 2, 0).cpu())
    axs[2].set_title("Target (GT)")
    axs[2].axis("off")

    plt.show()


with torch.no_grad():
    for inp, seg, tgt in samples:
        inp = inp.unsqueeze(0).to(DEVICE)
        seg = seg.unsqueeze(0).unsqueeze(1).float() / 150.0
        seg = seg.repeat(1, 3, 1, 1).to(DEVICE)

        x = torch.cat([inp, seg], dim=1)
        out = model(x)[0]

        show_triplet(inp[0], out, tgt)
