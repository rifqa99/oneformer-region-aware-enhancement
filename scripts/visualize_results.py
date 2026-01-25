import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import argparse

from src.models.unet import UNet
from src.datasets.ade_enhance_dataset import ADEEnhancementDataset
from src.datasets.PairedMaskDataset import PairedImageDataset


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == "ade":
        dataset = ADEEnhancementDataset(
            root=args.data_root,
            split="val"
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        dataset = PairedImageDataset(
            args.input_dir,
            args.target_dir,
            args.mask_dir,
            size=(512, 512),
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ---------------- model ----------------
    model = UNet(in_channels=6, out_channels=3).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # ---------------- run ----------------
    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            if args.dataset == "ade":
                inp, seg, tgt = batch
                inp = inp.to(device)
                tgt = tgt.to(device)

                # seg: (1, H, W) → (1, 3, H, W)
                seg = seg.unsqueeze(1).float() / 150.0
                seg = seg.repeat(1, 3, 1, 1).to(device)

                x = torch.cat([inp, seg], dim=1)
                gt = tgt

            else:
                x, gt = batch
                x = x.to(device)
                gt = gt.to(device)

            out = model(x)

            # Input | Output | GT
            grid = torch.cat([x[:, :3], out, gt], dim=0)

            vutils.save_image(
                grid,
                os.path.join(args.out_dir, f"sample_{i}.png"),
                nrow=3,
                normalize=True
            )

            if i == args.max_samples - 1:
                break

    print(f"✓ Saved {args.max_samples} samples to {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["ade", "paired"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=5)

    # ADE
    parser.add_argument("--data_root", type=str,
                        help="Root dataset dir for ADE")

    # Paired
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--mask_dir", type=str)

    args = parser.parse_args()
    main(args)
