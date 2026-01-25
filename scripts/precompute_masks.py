import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

from src.models.oneformer_wrapper import OneFormerWrapper
from src.utils.region_masks import build_region_masks


SEMANTIC_GROUPS = {
    "human": ["person"],
    "vegetation": ["plant", "tree"],
    "structure": ["wall", "building"],
}


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------- model ----------------
    segmenter = OneFormerWrapper(device=device)
    id2label = segmenter.model.config.id2label

    for split, img_dir in args.splits.items():
        out_dir = os.path.join(args.out_root, split)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nPrecomputing masks for split: {split}")

        fnames = sorted(os.listdir(img_dir))
        for fname in tqdm(fnames):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(img_dir, fname)
            img = Image.open(img_path).convert("RGB")

            # OneFormer inference
            seg = segmenter.predict(img)

            # Build region masks
            masks = build_region_masks(
                seg,
                id2label,
                SEMANTIC_GROUPS
            )

            # Stack as (K, H, W)
            stacked = np.stack(
                [m.cpu().numpy() for m in masks.values()],
                axis=0
            )

            np.save(
                os.path.join(out_dir, os.path.splitext(fname)[0] + ".npy"),
                stacked
            )

    print("\n✓ Mask precomputation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Root directory to save precomputed masks"
    )

    parser.add_argument(
        "--splits",
        type=lambda s: dict(
            item.split(":") for item in s.split(",")
        ),
        required=True,
        help=(
            "Comma-separated split:path pairs. "
            "Example: train:data/train/input,val:data/val/input"
        )
    )

    args = parser.parse_args()
    main(args)
