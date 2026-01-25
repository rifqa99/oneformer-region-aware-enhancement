import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.datasets import PairedMaskDataset, PairedRGBDataset
from src.utils.metrics import psnr, ssim
from src.utils.perceptual_loss import VGGPerceptualLoss


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # ---------------- dataset ----------------
    if args.use_masks:
        train_ds = PairedMaskDataset(
            args.train_input,
            args.train_target,
            args.train_masks,
            size=args.img_size
        )
        val_ds = PairedMaskDataset(
            args.val_input,
            args.val_target,
            args.val_masks,
            size=args.img_size
        )
        in_channels = 3 + args.num_masks
    else:
        train_ds = PairedRGBDataset(
            args.train_input,
            args.train_target,
            size=args.img_size
        )
        val_ds = PairedRGBDataset(
            args.val_input,
            args.val_target,
            size=args.img_size
        )
        in_channels = 3

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    # ---------------- model ----------------
    model = UNet(in_channels=in_channels, out_channels=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    l1_loss = nn.L1Loss()
    perc_loss = VGGPerceptualLoss(
        device=device) if args.use_perceptual else None

    # ---------------- logs ----------------
    train_losses, val_losses = [], []
    val_psnrs, val_ssims = [], []

    # ---------------- training ----------------
    for epoch in range(1, args.epochs + 1):

        # ---- train ----
        model.train()
        train_loss = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)

            loss = l1_loss(out, y)
            if perc_loss:
                loss = loss + args.lambda_perc * perc_loss(out, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)
        train_losses.append(train_loss)

        # ---- validation ----
        model.eval()
        val_loss = val_psnr = val_ssim = 0.0

        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)

                loss = l1_loss(out, y)
                if perc_loss:
                    loss = loss + args.lambda_perc * perc_loss(out, y)

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
            os.path.join(args.ckpt_dir, f"unet_epoch_{epoch}.pt")
        )

    # ---------------- save curves ----------------
    np.save(os.path.join(args.plot_dir, "train_loss.npy"), train_losses)
    np.save(os.path.join(args.plot_dir, "val_loss.npy"), val_losses)
    np.save(os.path.join(args.plot_dir, "val_psnr.npy"), val_psnrs)
    np.save(os.path.join(args.plot_dir, "val_ssim.npy"), val_ssims)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, nargs=2, default=(512, 512))

    parser.add_argument("--use_masks", action="store_true")
    parser.add_argument("--num_masks", type=int, default=3)
    parser.add_argument("--use_perceptual", action="store_true")
    parser.add_argument("--lambda_perc", type=float, default=0.1)

    parser.add_argument("--train_input", required=True)
    parser.add_argument("--train_target", required=True)
    parser.add_argument("--val_input", required=True)
    parser.add_argument("--val_target", required=True)
    parser.add_argument("--train_masks")
    parser.add_argument("--val_masks")

    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--plot_dir", required=True)

    args = parser.parse_args()
    main(args)
