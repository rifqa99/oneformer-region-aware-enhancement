import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.models.oneformer_wrapper import OneFormerWrapper
from src.utils.region_masks import build_region_masks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEMANTIC_GROUPS = {
    "human": ["person"],
    "vegetation": ["plant", "tree"],
    "structure": ["wall", "building"],
}

IMG_DIR = "data/train/input"
OUT_DIR = "data/train/masks"
os.makedirs(OUT_DIR, exist_ok=True)

segmenter = OneFormerWrapper(device=DEVICE)
id2label = segmenter.model.config.id2label

for fname in tqdm(os.listdir(IMG_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
    seg = segmenter.predict(img)

    masks = build_region_masks(seg, id2label, SEMANTIC_GROUPS)
    stacked = np.stack([m.cpu().numpy() for m in masks.values()], axis=0)

    np.save(os.path.join(OUT_DIR, fname.replace(".jpg", ".npy")), stacked)
