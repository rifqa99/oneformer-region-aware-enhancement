import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.models.oneformer_wrapper import OneFormerWrapper
from src.utils.region_masks import build_region_masks

# ---------------- config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

SEMANTIC_GROUPS = {
    "human": ["person"],
    "vegetation": ["plant", "tree"],
    "structure": ["wall", "building"],
}

SPLITS = {
    "train": "data/train/input",
    "val":   "data/val/input",
}

BASE_OUT = "/content/drive/MyDrive/oneformer_masks"

# ---------------- model ----------------
segmenter = OneFormerWrapper(device=DEVICE)
id2label = segmenter.model.config.id2label

# ---------------- run ----------------
for split, img_dir in SPLITS.items():
    out_dir = os.path.join(BASE_OUT, split)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nPrecomputing masks for {split} set...")
    for fname in tqdm(os.listdir(img_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        seg = segmenter.predict(img)

        masks = build_region_masks(seg, id2label, SEMANTIC_GROUPS)
        stacked = np.stack([m.cpu().numpy() for m in masks.values()], axis=0)

        np.save(os.path.join(out_dir, os.path.splitext(
            fname)[0] + ".npy"), stacked)


print("\nMask precomputation complete.")
