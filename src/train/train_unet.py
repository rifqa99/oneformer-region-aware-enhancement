import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms as T

from src.models.unet import UNet
from src.datasets.paired_dataset import PairedImageDataset
from src.utils.concat_inputs import concat_image_and_masks
from src.models.oneformer_wrapper import OneFormerWrapper
from src.utils.region_masks import build_region_masks


from src.utils.metrics import psnr, ssim

# -------- config --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 2
LR = 1e-3
IMG_SIZE = (512, 512)
CKPT_DIR = "outputs/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

SEMANTIC_GROUPS = {
    "human": ["person"],
    "vegetation": ["plant", "tree"],
    "structure": ["wall", "building"],
}

# -------- data --------
train_ds = PairedImageDataset(
    "data/train/input", "data/train/target", size=IMG_SIZE)
val_ds = PairedImageDataset(
    "data/val/input",   "data/val/target",   size=IMG_SIZE)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# -------- models --------
segmenter = OneFormerWrapper(device=DEVICE)   # frozen
model = UNet(in_channels=6, out_channels=3).to(DEVICE)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.L1Loss()
to_tensor = T.ToTensor()

# -------- train --------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for inp_img, tgt_img in train_dl:
        inp_img = inp_img.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)

        # build region-aware input per image
        inputs = []
        for i in range(inp_img.size(0)):
            pil = T.ToPILImage()(inp_img[i].cpu())
            seg = segmenter.predict(pil)
            id2label = segmenter.model.config.id2label
            masks = build_region_masks(
                seg, id2label, SEMANTIC_GROUPS)
            x = concat_image_and_masks(inp_img[i], masks)
            inputs.append(x)

        x = torch.stack(inputs).to(DEVICE)

        opt.zero_grad()
        out = model(x)

        # crop target to output size (safe)
        _, _, h, w = out.shape
        loss = loss_fn(out, tgt_img[:, :, :h, :w])
        loss.backward()
        opt.step()

        train_loss += loss.item()

    train_loss /= max(1, len(train_dl))

    # -------- val --------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inp_img, tgt_img in val_dl:
            inp_img = inp_img.to(DEVICE)
            tgt_img = tgt_img.to(DEVICE)

            inputs = []
            for i in range(inp_img.size(0)):
                pil = T.ToPILImage()(inp_img[i].cpu())
                seg = segmenter.predict(pil)
                id2label = segmenter.model.config.id2label
                masks = build_region_masks(
                    seg, id2label, SEMANTIC_GROUPS)
                x = concat_image_and_masks(inp_img[i], masks)
                inputs.append(x)

            x = torch.stack(inputs).to(DEVICE)
            out = model(x)
            _, _, h, w = out.shape
            val_loss += loss_fn(out, tgt_img[:, :, :h, :w]).item()
            p = psnr(out, tgt_img[:, :, :h, :w]).item()
            s = ssim(out, tgt_img[:, :, :h, :w]).item()

            val_psnr += p
            val_ssim += s

            val_psnr = 0.0
            val_ssim = 0.0
            val_psnr /= max(1, len(val_dl))
            val_ssim /= max(1, len(val_dl))
            print(
                f"Epoch {epoch}: "
                f"train={train_loss:.4f}, "
                f"val={val_loss:.4f}, "
                f"PSNR={val_psnr:.2f}, "
                f"SSIM={val_ssim:.4f}"
            )

    val_loss /= max(1, len(val_dl))

    torch.save(model.state_dict(), f"{CKPT_DIR}/unet_epoch_{epoch}.pt")
    print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
